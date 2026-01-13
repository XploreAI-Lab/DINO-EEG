from collections import defaultdict
from typing import Any

import numpy as np


# detected_bbs是所有预测的集合，list长度为QN*切片数量
# detected_bbs的item包括'image_id'，'label'，'box'，'score'四个字段，对应该预测所在的图片，预测的类，预测的绝对坐标，置信度
# detected_bbs 一开始就不包含背景类
# groundtruth_bbs是所有标注的集合，list的长度为事件数量*验证集/测试集包含的切片数量
# groundtruth_bbs的item包括'image_id'，'label'，'box'三个字段，对应标签所在的图片，类，绝对坐标
def get_coco_summary(
    groundtruth_bbs: list[dict[str, Any]], detected_bbs: list[dict[str, Any]]
):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious 计算每个预测
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool_)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append(
                {
                    "class": class_id,
                    **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
                }
            )
        return res

    iou_thresholds = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }

    AP50 = np.mean([x["AP"] for x in full[0.50] if x["AP"] is not None])
    AP75 = np.mean([x["AP"] for x in full[0.75] if x["AP"] is not None])
    AP = np.mean([x["AP"] for k in full for x in full[k] if x["AP"] is not None])

    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [
            x["TP"] / x["total positives"]
            for k in full
            for x in full[k]
            if x["TP"] is not None
        ]
    )

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR1 = np.mean(
        [
            x["TP"] / x["total positives"]
            for k in max_det1
            for x in max_det1[k]
            if x["TP"] is not None
        ]
    )

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR10 = np.mean(
        [
            x["TP"] / x["total positives"]
            for k in max_det10
            for x in max_det10[k]
            if x["TP"] is not None
        ]
    )

    return {
        "AP": AP,
        "AP50": AP50,
        "AP75": AP75,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
    }


# 将模型的预测框先按照每一张图片，每一种类别，按照置信度从大到小得到maxDet个框，
# 然后将测试集中特定类别的所有框按置信度总的排序，继而再对排序后的指定类别的所有框计算tp、fp
def get_coco_metrics(
    groundtruth_bbs,
    detected_bbs,
    iou_threshold=0.5,
    area_range=(0, np.inf),
    max_dets=100,
):
    """Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class bbs中已经没有包含背景类
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    # 对于第img_id个切片的第class_id个类别（分别有gt，dt）
    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )
        # 第class_id个类别的匹配情况 这里也已经不包含背景类的class_id
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations 全部连起来方便计算 _evals中此时已经不包含背景类的class_id
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool_)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class 第i个类别的AP
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
        }
    return res


def _group_detections(dt: list[dict[str, Any]], gt: list[dict[str, Any]]):
    """simply group gts and dts on a imageXclass basis"""
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d["image_id"]
        c_id = d["label"]
        # 名字为filename的图片的label标签上有一个预测
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g["image_id"]
        c_id = g["label"]
        # 名字为filename的图片的label标签上有一个事件
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(box):
    # 绝对坐标，返回事件长度
    return box[1] - box[0]


def _jaccard(box1, box2):
    # 计算IoU
    area1 = box1[1] - box1[0]
    area2 = box2[1] - box2[0]
    lt = max(box1[0], box2[0])
    rb = min(box1[1], box2[1])
    inter = max(0, rb - lt)
    union = area1 + area2 - inter
    iou = inter / union

    return iou


def _compute_ious(dt: list[dict[str, Any]], gt: list[dict[str, Any]]):
    """compute pairwise ious"""

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            # 这里不管是不是同一张图片，先生成所有预测与所有图片事件的iou
            ious[d_idx, g_idx] = _jaccard(d["box"], g["box"])
    
    return ious


def _evaluate_image(
    dt: list[dict[str, Any]],
    gt: list[dict[str, Any]],
    ious,
    iou_threshold,
    max_dets=None,
    area_range=None,
):
    """use COCO's method to associate detections to ground truths"""
    # sort dts by increasing confidence
    # 将该图片上的该类别的所有预测按置信度从小到大（取负了，实际上从大到小）排序（的下标）
    dt_sort = np.argsort([-d["score"] for d in dt], kind="stable")
    # sort list of dts and chop by max dets
    # 只取前max_dets（这里是100）个预测，足够了，一个切片上没有那么多事件，若max_dets>NQ，则dt长度为NQ
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    # ious也是按照dt置信度从大到小的顺序排序的
    ious = ious[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    # 如果事件长度不在area_range之内（太小／太大）就忽略掉，这里area_range为0到inf，所以没有忽略
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb["box"]) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]
    # sort gts by ignore last 需要忽略的事件都放在后面（为True的都在后面）
    gt_sort = np.argsort(gt_ignore, kind="stable")
    # 按gt_sort的下标重新排序
    gt = [gt[idx] for idx in gt_sort]
    # 保存排序后的gt_ignore（True都在后面）
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    # 对于每一个预测，去匹配使得iou最大的gt dt是按置信度从大到小的顺序排序的
    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched) iou至少为iou_threshold
        iou = min(iou_threshold, 1 - 1e-10)
        # 该dt当前匹配到的gt的下标
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            # 如果一个标签已经被匹配到了，就不再参与匹配
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop 开始匹配到忽略的事件，不用再继续匹配了
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made 
            # 最开始iou没有超过阈值，没有匹配上；后面的iou如果没有超过前面匹配上的iou，也算没有匹配上
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately 
            # 存储当前dt匹配到的gt的·最·大·的iou，iou是按dt置信度排序的，不是按照iou大小排序的
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt 全都没有匹配上
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts 如果匹配到了忽略的事件（经过上面的处理理论上不会）或预测本身太长/太短
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d)
        for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts 获取没有忽略的事件的置信度
    scores = [dt[d_idx]["score"] for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    # 匹配到的没有忽略的事件在gt中的下标
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced.
    """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    # sort in descending score order 按照置信度从高到低排序
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }
