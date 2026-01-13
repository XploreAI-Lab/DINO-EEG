from collections import defaultdict
from typing import Any, List, Tuple, Dict, DefaultDict

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def _group_detections(dt: List[Dict[str, Any]], gt: List[Dict[str, Any]]):
    """
    simply group gts and dts on a imageXclass basis
    对于每张图片的每个类有几个gt，又有几个dt
    """
    bb_info: DefaultDict[Tuple[str, str], Dict[str, List[Any]]] = defaultdict(
        lambda: {"dt": [], "gt": []}
    )
    for d in dt:
        i_id = d["image_id"]
        c_id = d["label"]
        # id为image_id的图片的label标签上有一个预测
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g["image_id"]
        c_id = g["label"]
        # id为image_id的图片的label标签上有一个事件
        bb_info[i_id, c_id]["gt"].append(g)

    class_gts = defaultdict(lambda: 0)
    # dt中仍然可能预测不存在的类别，直接删除
    for img_id, class_id in bb_info:
        class_gts[class_id] += len(bb_info[img_id, class_id]["gt"])

    for class_id, n_gt in class_gts.items():
        if n_gt == 0:
            bb_info = {k: v for k, v in bb_info.items() if k[1] != class_id}

    return bb_info


def _get_class_mapper(bb_info: Dict[Tuple[int, int], Dict[str, List[Any]]]):
    """
    由于label不一定连续并且从0开始，返回label到idx的映射
    """
    c = []
    c_mapper = {}
    for _, class_id in bb_info:
        if class_id not in c:
            c.append(class_id)
    for idx, cls in enumerate(c):
        c_mapper[cls] = idx
    return c_mapper


def _get_iou(box1, box2):
    # 计算IoU
    area1 = box1[1] - box1[0]
    area2 = box2[1] - box2[0]
    lt = max(box1[0], box2[0])
    rb = min(box1[1], box2[1])
    inter = max(0, rb - lt)
    union = area1 + area2 - inter
    iou = inter / union

    return iou


def _compute_ious(dt: List[Dict[str, Any]], gt: List[Dict[str, Any]]):
    """compute pairwise ious"""
    # 因为这个是针对一个image的一个class可能会出现，gt长度为0 shape为[len(dt), 0]，dt长度为0，shape为[0, len(gt)]
    ious = np.zeros(shape=(len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            # 这里不管是不是同一张图片，先生成所有预测与所有图片事件的iou
            ious[d_idx, g_idx] = _get_iou(d["box"], g["box"])
    return ious


def _iou_score(ious, iou_threshold: float):
    """
    一张图片的其中一个类class的tp，fp，fn
    """
    # 当ious的shape为[len(dt), 0]或[0, len(gt)]时，linear_sum_assignment返回长度为0的两个空数组
    d_idx, g_idx = linear_sum_assignment(cost_matrix=ious, maximize=True)
    # ious.shape[0]即为len(dt)，代表dt预测的数量，为0代表该图片的该类上没有预测
    hits_dt = np.zeros(shape=(ious.shape[0],), dtype=int)
    # ious.shape[1]即为len(gt)，代表gt真实值的数量，为0代表该图片没有该类
    hits_gt = np.zeros(shape=(ious.shape[1],), dtype=int)

    # 包含0的ious自然不会经过该遍历
    for d_index, g_index in zip(d_idx, g_idx):
        if ious[d_index, g_index] >= iou_threshold:
            hits_dt[d_index] = 1
            hits_gt[g_index] = 1

    # 因此，无论len(dt)为0还是len(gt)为0，tp都应该为0，因为要么有gt无预测，要么有预测却没有dt
    tp: int = np.sum(hits_dt)
    # 当存在没有匹配到的gt时，自然其数量为fn；只要len(gt)不为0
    fn: int = np.sum(hits_gt == 0)
    # 当存在没有匹配到的dt时，自然其数量为fp；只要len(dt)不为0
    fp: int = np.sum(hits_dt == 0)

    return tp, fn, fp


def _get_ovlp_events(start_a, stop_a, events_a):
    starts = []
    stops = []
    for event in events_a:
        # if the event overlaps partially with the interval,
        # it is a match. this means:
        #              start               stop
        #   |------------|<---------------->|-------------|
        #          |---------- event -----|
        #
        if (event[1] > start_a) and (event[0] < stop_a):
            starts.append(event[0])
            stops.append(event[1])

    return starts, stops


def _ovlp_score(dt: List[Dict[str, Any]], gt: List[Dict[str, Any]]):
    tp = int(0)
    fp = int(0)
    fn = int(0)

    for event in gt:
        starts, _ = _get_ovlp_events(
            event["box"][0], event["box"][1], list(map(lambda x: x["box"], dt))
        )
        if len(starts) != 0:
            tp += 1
        else:
            fn += 1
    # loop over the hyp annotation to collect fps
    for event in dt:
        starts, _ = _get_ovlp_events(
            event["box"][0], event["box"][1], list(map(lambda x: x["box"], gt))
        )
        if len(starts) == 0:
            fp += 1

    return tp, fn, fp


def _evaluate_image(
    dt: List[Dict[str, Any]],
    gt: List[Dict[str, Any]],
    ious,
    method: List[str],
):
    """
    对一张图片的一个class进行评估
    """
    if method == "iou50":
        tp, fn, fp = _iou_score(ious, 0.5)
    elif method == "iou75":
        tp, fn, fp = _iou_score(ious, 0.75)
    elif method == "ovlp":
        tp, fn, fp = _ovlp_score(dt, gt)
    else:
        raise NotImplementedError(f"Methed {method} not implemented.")

    return {"TP": tp, "FP": fp, "FN": fn, "PN": len(gt)}


def _get_recall(tp, fn, _):
    return tp / (tp + fn)


def _get_precision(tp, _, fp):
    return tp / (tp + fp)


def _compute_f1_score(tp, fn, fp, pn, average):
    """
    tp/fn/fp/pn : [n_methods, n_thresholds, n_class] 每种方法的每个阈值的每种class的tp等
    """
    # micro-F1 通过所有class的tp，fp，fn的总和计算p，r，进而计算F1分数
    if average == "micro":
        # [n_methods, n_thresholds, 1] tp，fp，fn的总和
        tp_t = np.sum(tp, axis=-1, keepdims=True)
        fp_t = np.sum(fp, axis=-1, keepdims=True)
        fn_t = np.sum(fn, axis=-1, keepdims=True)
        # [n_methods, n_thresholds, 1]
        recall = _get_recall(tp_t, fn_t, fp_t)
        precision = _get_precision(tp_t, fn_t, fp_t)
        # [n_methods, n_thresholds, 1] 每种方法的每个阈值的micro-f1分数
        # f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    # macro-F1 先计算出每一个类别的p，r和f1，然后计算平均值
    elif average == "macro":
        # [n_methods, n_thresholds, n_class] 每种class的recall和precision
        recall = _get_recall(tp, fn, fp)
        precision = _get_precision(tp, fn, fp)
        # [n_methods, n_thresholds, 1] 每种方法的每个阈值的macro-f1分数
        # f1 = np.mean(
        #     2 * (precision * recall) / (precision + recall + 1e-10), axis=-1, keepdims=True
        # )
        # [n_methods, n_thresholds, 1]
        recall = np.mean(recall, axis=-1, keepdims=True)
        precision = np.mean(precision, axis=-1, keepdims=True)
    # weighted-F1 通过真实样本数量的比例作为权重计算加权版本的macro-F1
    elif average == "weighted":
        # [n_methods, n_thresholds, n_class] 每种class的recall和precision
        recall = _get_recall(tp, fn, fp)
        precision = _get_precision(tp, fn, fp)
        # [n_methods, n_thresholds, 1] 每种方法的每个阈值的weighted-f1分数
        # f1 = np.average(
        #     2 * (precision * recall) / (precision + recall + 1e-10),
        #     axis=-1,
        #     weights=pn,
        #     keepdims=True,
        # )
        recall = np.average(recall, axis=-1, weights=pn, keepdims=True)
        precision = np.average(precision, axis=-1, weights=pn, keepdims=True)
    else:
        raise NotImplementedError(f"Average {average} not implemented.")

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    # tuple([n_methods, n_thresholds, 1], [n_methods, n_thresholds, 1], [n_methods, n_thresholds, 1])
    return f1, recall, precision


def _f1_score(tp, fn, fp, pn):
    """
    返回3种F1分数
    Parameters
    ----------
    dict_res_item: Dict[float]
    分别存储不同的F1分数
    """
    # 不同的f1分数计算方式
    averages = ["micro", "macro", "weighted"]

    f1 = []
    recall = []
    precision = []

    for average in averages:
        f1_c, recall_c, precision_c = _compute_f1_score(tp, fn, fp, pn, average)
        f1.append(f1_c)
        recall.append(recall_c)
        precision.append(precision_c)

    f1 = np.concatenate(f1, axis=-1)
    recall = np.concatenate(recall, axis=-1)
    precision = np.concatenate(precision, axis=-1)

    # tuple([n_methods, n_thresholds, 3], [n_methods, n_thresholds, 3], [n_methods, n_thresholds, 3])
    return f1, recall, precision


def get_event_metrics(groundtruth_bbs, detected_bbs, selected_thresholds_index=None):
    """
    获取使得3种F1分数最大的阈值及其F1分数。
    对于背景类的处理：由于detected_bbs中不会包含背景类，因此通过sigmoid后只会出现NC个值的置信度都很低的情况，可以被阈值过滤掉；
    如果一个地方有gt却没有dt（预测为背景类导致所有类都是低置信度被过滤掉），会被算作一个fn；反之如果一个地方有dt却没有gt，会被算作一个fp
    Parameters
    ----------
    groundtruth_bbs : list
        A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
        groundtruth_bbs的item包括'image_id'，'label'，'box'三个字段，对应标签所在的图片，类，绝对坐标
    detected_bbs : list
        A list containing objects of type BoundingBox representing the detected bounding boxes.
        detected_bbs的item包括'image_id'，'label'，'box'，'score'四个字段，对应该预测所在的图片，预测的类，预测的绝对坐标，置信度
    selected_thresholds_index : [n_methods, 3] 3种评估指标的3种F1分数的阈值下标
        验证集上selected_thresholds为空，测试集上selected_thresholds_index为在验证集上选择的阈值的下标
    """
    # selected_thresholds_index = None
    # tusz二分类的
    # selected_thresholds_index = np.array([[26,26,26],[27,27,27],[22,22,22]])
    # tusz多分类的
    # selected_thresholds_index = np.array([[24, 25, 24], [26, 23, 26], [22, 23, 22]])
    # tusz在chbmit验证集上找到的阈值
    # selected_thresholds_index = np.array([[19,19,19],[23,23,23],[19,19,19]])
    # tusz在neonatal验证集上找到的阈值
    # selected_thresholds_index = np.array([[7,7,7],[6,6,6],[7,7,7]])

    # 不同的评估指标
    methods = ["iou50", "iou75", "ovlp"]
    n_methods = len(methods)
    # 置信度阈值
    n_thresholds = 100
    # 为了评估在不同置信度阈值下模型的表现
    thresholds = np.linspace(start=1e-2, stop=1, num=n_thresholds)
    # label到dict_res下标idx的映射
    class_mapper = _get_class_mapper(_group_detections(detected_bbs, groundtruth_bbs))
    # 类别的数量
    n_class = len(class_mapper)
    # 第method个方法的第i_thresh阈值的class_id的tp/fp/fn/pn
    tp = np.zeros(shape=(n_methods, n_thresholds, n_class))
    fp = np.zeros(shape=(n_methods, n_thresholds, n_class))
    fn = np.zeros(shape=(n_methods, n_thresholds, n_class))
    # pn代表total number of ground truth positives
    pn = np.zeros(shape=(n_methods, n_thresholds, n_class))

    duration_all = np.sum([g["orig_size"] for g in groundtruth_bbs])
    duration_all = np.divide(duration_all, 3600)

    # 在不同评估指标下的不同置信度阈值下的表现
    for i_thresh, thresh in enumerate(thresholds):
        # 使用不同的置信度阈值产生不同的dt
        _detected_bbs = list(filter(lambda x: x["score"] >= thresh, detected_bbs))
        # 对于groundtruth和detected分布按照每张图片的每个类组织
        _bbs = _group_detections(_detected_bbs, groundtruth_bbs)
        # 计算对于每个图片image的每个类class上的每个dt和其gt的iou item这里是image_id,class_id -> dt,gt
        _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

        # 对于每个阈值产生的dt，分别使用三种方法进行评估
        for i_method, method in enumerate(methods):
            _evals: DefaultDict[int, Dict[str, int | List[int]]] = defaultdict(
                lambda: {"TP": [], "FP": [], "FN": [], "PN": []}
            )
            for img_id, class_id in _bbs:
                ev = _evaluate_image(
                    _bbs[img_id, class_id]["dt"],
                    _bbs[img_id, class_id]["gt"],
                    _ious[img_id, class_id],
                    method,
                )
                # if ev["PN"] == 0:
                #     continue
                # 第class_id个类别的匹配情况
                acc = _evals[class_id]
                acc["TP"].append(ev["TP"])
                acc["FP"].append(ev["FP"])
                acc["FN"].append(ev["FN"])
                acc["PN"].append(ev["PN"])

            for class_id in _evals:
                acc = _evals[class_id]
                acc["TP"] = np.sum(acc["TP"])
                acc["FP"] = np.sum(acc["FP"])
                acc["FN"] = np.sum(acc["FN"])
                acc["PN"] = np.sum(acc["PN"])

                tp[i_method][i_thresh][class_mapper[class_id]] += acc["TP"]
                fp[i_method][i_thresh][class_mapper[class_id]] += acc["FP"]
                fn[i_method][i_thresh][class_mapper[class_id]] += acc["FN"]
                pn[i_method][i_thresh][class_mapper[class_id]] += acc["PN"]

    # 存储结果 [n_methods, n_thresholds, 3] 每种方法的每个阈值的3种不同的F1分数
    f1_all, recall_all, precision_all = _f1_score(tp, fn, fp, pn)
    f1_all = np.nan_to_num(f1_all)
    recall_all = np.nan_to_num(recall_all)
    precision_all = np.nan_to_num(precision_all)
    # [n_methods, 3] 分别找到每种方法使得3种不同的f1分数最大的阈值的下标，通过下标找到使得f1分数最大的阈值
    select_thresholds_index = np.nanargmax(f1_all, axis=1)
    select_thresholds = thresholds[select_thresholds_index]
    # [n_methods, 3] 获取最大的F1分数
    # max_f1 = np.nanmax(f1_all, axis=1)
    max_f1 = np.take_along_axis(
        f1_all, select_thresholds_index[:, None, :], axis=1
    ).squeeze()
    max_f1_recall = np.take_along_axis(
        recall_all, select_thresholds_index[:, None, :], axis=1
    ).squeeze()
    max_f1_precision = np.take_along_axis(
        precision_all, select_thresholds_index[:, None, :], axis=1
    ).squeeze()

    r = defaultdict(lambda: {})

    cms = np.zeros((n_methods, 3, n_class, n_class))

    for i_method, method in enumerate(methods):
        for i_average, average in enumerate(["micro", "macro", "weighted"]):
            r[method][average] = {
                "f1": max_f1[i_method, i_average],
                "recall": max_f1_recall[i_method, i_average],
                "precision": max_f1_precision[i_method, i_average],
                "threshold": select_thresholds[i_method, i_average],
            }

            # fp[n_methods, n_thresholds, n_class=1]
            # r[method][average]["FPR"]=fp[i_method][]

            # 在验证集上找到的阈值在测试集上的表现
            if selected_thresholds_index is not None:
                selected_threshold_index = selected_thresholds_index[
                    i_method, i_average
                ]
                r[method][average]["f1_s"] = f1_all[
                    i_method, selected_threshold_index, i_average
                ]
                r[method][average]["recall_s"] = recall_all[
                    i_method, selected_threshold_index, i_average
                ]
                r[method][average]["precision_s"] = precision_all[
                    i_method, selected_threshold_index, i_average
                ]
                r[method][average]["select_threshold"] = thresholds[
                    selected_threshold_index
                ]

                fpr_s = np.divide(
                    fp[i_method][selected_threshold_index][0], duration_all
                )
                r[method][average]["FPR_s"] = fpr_s

                # [nc, nc]
                cm = generate_confusion_matrix(
                    list(
                        filter(
                            lambda x: x["score"]
                            >= thresholds[selected_threshold_index],
                            detected_bbs,
                        )
                    ),
                    groundtruth_bbs,
                    iou_threshold=0,
                )
                cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                cms[i_method, i_average] = cm_normalized

    print(cms[2, 2])

    iras = None
    if selected_thresholds_index is not None:
        iras = compute_FEA_FEDA(
            detected_bbs,
            groundtruth_bbs,
            thresholds[selected_thresholds_index][2, 0],  # [2,0]是ovlp，micro的下标
        )

    return r, select_thresholds_index, iras


def _get_iou2(box1, box2):
    area1 = box1[1] - box1[0]
    area2 = box2[1] - box2[0]
    lt = max(box1[0], box2[0])
    rb = min(box1[1], box2[1])
    inter = max(0, rb - lt)
    union = area1 + area2 - inter

    return inter, union


def _compute_one(dt: List[Any], gt: List[Any]):
    hits_dt = np.zeros(shape=(len(dt),), dtype=int)
    hits_gt = np.zeros(shape=(len(gt),), dtype=int)
    agreements = 0
    disagreements = 0
    # FEDA是所有事件的单个FEDA的平均值
    idx = 0
    tuple_list_mapper: Dict[int, Tuple[List, List]] = {}
    inters: Dict[int, float] = {}
    unions: Dict[int, float] = {}
    dt = sorted(dt, key=lambda it: it["box"][0])
    gt = sorted(gt, key=lambda it: it["box"][0])
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            if _get_iou2(d["box"], g["box"])[0] >= 1:
                if hits_dt[d_idx] == 0 and hits_gt[g_idx] == 0:
                    hits_dt[d_idx] = 1
                    hits_gt[g_idx] = 1

                    agreements += 1
                    # 需要生成一个新的FEDA的分子和分母
                    i, u = _get_iou2(g["box"], d["box"])
                    inters[idx] = i
                    unions[idx] = u
                    tuple_list_mapper[idx] = ([g_idx], [d_idx])
                    idx += 1

                elif hits_gt[g_idx] == 0 and hits_dt[d_idx] == 1:
                    hits_gt[g_idx] = 1

                    i, _ = _get_iou2(g["box"], d["box"])
                    for _idx, v in tuple_list_mapper.items():
                        if d_idx in v[1]:
                            v[0].append(g_idx)
                            inters[_idx] += i
                            # FEDA 如果框都是互斥的，下面的代码没有问题，但模型会生成大量重复的框
                            # 如果重复的框被另一个框完全包含，就会导致分子不断变大而分母不变
                            # 最终导致出现FEDA大于1的情况
                            # unions[idx] += (g["box"][1] - g["box"][0]) - i
                            # 为了避免这种情况，当完全包含时，分子分母同时加上长度为事件长度的值
                            # 这样的值，FEDA总体会变大，但小于上面的代码的结果，下面同理
                            _c = (g["box"][1] - g["box"][0]) - i
                            unions[_idx] += _c if _c != 0 else i
                            break

                elif hits_gt[g_idx] == 1 and hits_dt[d_idx] == 0:
                    hits_dt[d_idx] = 1

                    i, _ = _get_iou2(g["box"], d["box"])
                    for _idx, v in tuple_list_mapper.items():
                        if g_idx in v[0]:
                            v[1].append(d_idx)
                            inters[_idx] += i
                            # unions[idx] += (d["box"][1] - d["box"][0]) - i
                            _c = (d["box"][1] - d["box"][0]) - i
                            unions[_idx] += _c if _c != 0 else i
                            break

                else:
                    pass

    disagreements += np.sum(hits_dt == 0) + np.sum(hits_gt == 0)

    fea = np.nan_to_num(np.divide(agreements, agreements + disagreements))
    feda = np.nan_to_num(
        np.mean(
            [inter / union for inter, union in zip(inters.values(), unions.values())]
        )
    )
    # ira = np.divide(fea + feda, 2) if not np.isnan(feda) else fea
    ira = np.divide(fea + feda, 2)

    return {"FEA": fea, "FEDA": feda, "IRA": ira}


def compute_FEA_FEDA(
    detected_bbs: List[Dict[str, Any]],
    groundtruth_bbs: List[Dict[str, Any]],
    selected_thresholds_index,
):
    res = defaultdict(lambda: [])

    _detected_bbs = list(
        filter(lambda x: x["score"] >= selected_thresholds_index, detected_bbs)
    )
    _bbs = _group_detections(_detected_bbs, groundtruth_bbs)
    # _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    for img_id, class_id in _bbs:
        # 首先只考虑一个类，一个EEG的全部事件的均值
        ev = _compute_one(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
        )
        res["FEA"].append(ev["FEA"])
        res["FEDA"].append(ev["FEDA"])
        res["IRA"].append(ev["IRA"])

    # res["FEDA"] = list(filter(lambda x: not np.isnan(x), res["FEDA"]))
    # res["FEDA"] = np.nan_to_num(res["FEDA"])

    fea_mean = np.mean(res["FEA"])
    fea_std = np.std(res["FEA"])
    feda_mean = np.mean(res["FEDA"])
    feda_std = np.std(res["FEDA"])
    ira_mean = np.mean(res["IRA"])
    ira_std = np.std(res["IRA"])

    print("FEA mean:{}".format(fea_mean))
    print("FEA std:{}".format(fea_std))
    print("FEDA mean:{}".format(feda_mean))
    print("FEDA std:{}".format(feda_std))
    print("IRA mean:{}".format(ira_mean))
    print("IRA std:{}".format(ira_std))

    return fea_mean, fea_std, feda_mean, feda_std, ira_mean, ira_std


def _get_n_patients(groundtruth_bbs: List[Dict[str, Any]]):
    patients = defaultdict(lambda: 0.0)
    for g in groundtruth_bbs:
        patients[g["patient"]] += g["orig_size"]

    return len(patients), patients


def _get_fpr(patients: Dict[str, float], fp):
    """
    fp: [n_methods, n_thresholds, n_patients, n_class]
    """
    # print(fp.shape)
    # 先按求和 [n_methods, n_thresholds, n_patients]
    fp_all = np.sum(fp, axis=-1, keepdims=False)
    # 准备计算fpr [n_patients,]
    duration_all = np.array(list(patients.values()))
    # [n_methods, n_thresholds, n_patients]
    fpr_all = np.divide(fp_all, duration_all)
    # [n_methods, n_thresholds, 1] 转换成为小时
    fpr_mean = np.mean(np.multiply(fpr_all, 3600), axis=-1, keepdims=True)

    return fpr_mean


def get_patient_metircs(groundtruth_bbs, detected_bbs, selected_thresholds=None):
    methods = ["iou50", "iou75", "ovlp"]
    n_methods = len(methods)

    # thresholds = np.sort(np.unique(list(map(lambda x: x["score"], detected_bbs))))
    # n_thresholds = len(thresholds)

    # 置信度阈值
    n_thresholds = 100
    # 为了评估在不同置信度阈值下模型的表现
    thresholds = np.linspace(start=1e-2, stop=1, num=n_thresholds)

    # label到dict_res下标idx的映射
    class_mapper = _get_class_mapper(_group_detections(detected_bbs, groundtruth_bbs))
    # 类别的数量
    n_class = len(class_mapper)

    n_patients, patients = _get_n_patients(groundtruth_bbs)

    # 第method个方法的第i_thresh阈值的patient的class_id的tp/fp/fn/pn
    tp = np.zeros(shape=(n_methods, n_thresholds, n_patients, n_class))
    fp = np.zeros(shape=(n_methods, n_thresholds, n_patients, n_class))
    fn = np.zeros(shape=(n_methods, n_thresholds, n_patients, n_class))
    # pn代表total number of ground truth positives
    pn = np.zeros(shape=(n_methods, n_thresholds, n_patients, n_class))

    # 在不同评估指标下的不同置信度阈值下的表现
    for i_thresh, thresh in enumerate(tqdm(thresholds, desc="thresh", position=0)):
        # 使用不同的置信度阈值产生不同的dt
        _detected_bbs = list(filter(lambda x: x["score"] >= thresh, detected_bbs))

        for i_patient, patient in enumerate(patients.keys()):
            _detected_bbs_p = list(
                filter(lambda x: x["patient"] == patient, _detected_bbs)
            )
            _groundtruth_bbs_p = list(
                filter(lambda x: x["patient"] == patient, groundtruth_bbs)
            )

            # 对于groundtruth和detected分布按照每张图片的每个类组织
            _bbs = _group_detections(_detected_bbs_p, _groundtruth_bbs_p)
            # 计算对于每个图片image的每个类class上的每个dt和其gt的iou item这里是image_id,class_id -> dt,gt
            _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

            # 对于每个阈值产生的dt，分别使用三种方法进行评估
            for i_method, method in enumerate(methods):
                _evals: DefaultDict[int, Dict[str, int | List[int]]] = defaultdict(
                    lambda: {"TP": [], "FP": [], "FN": [], "PN": []}
                )

                for img_id, class_id in _bbs:
                    ev = _evaluate_image(
                        _bbs[img_id, class_id]["dt"],
                        _bbs[img_id, class_id]["gt"],
                        _ious[img_id, class_id],
                        method,
                    )
                    # if ev["PN"] == 0:
                    #     continue
                    # 第class_id个类别的匹配情况
                    acc = _evals[class_id]
                    acc["TP"].append(ev["TP"])
                    acc["FP"].append(ev["FP"])
                    acc["FN"].append(ev["FN"])
                    acc["PN"].append(ev["PN"])

                for class_id in _evals:
                    acc = _evals[class_id]
                    acc["TP"] = np.sum(acc["TP"])
                    acc["FP"] = np.sum(acc["FP"])
                    acc["FN"] = np.sum(acc["FN"])
                    acc["PN"] = np.sum(acc["PN"])

                    tp[i_method][i_thresh][i_patient][class_mapper[class_id]] += acc[
                        "TP"
                    ]
                    fp[i_method][i_thresh][i_patient][class_mapper[class_id]] += acc[
                        "FP"
                    ]
                    fn[i_method][i_thresh][i_patient][class_mapper[class_id]] += acc[
                        "FN"
                    ]
                    pn[i_method][i_thresh][i_patient][class_mapper[class_id]] += acc[
                        "PN"
                    ]

    # [n_methods, n_thresholds, n_patients, 3] 每种方法的每个阈值的每个病人的3种不同的tpr
    _, recall_all, _ = _f1_score(tp, fn, fp, pn)

    # [n_methods, n_thresholds, 3]
    tpr_mean = np.mean(recall_all, axis=-2, keepdims=False)
    # [n_methods, n_thresholds, 1]
    fpr_mean = _get_fpr(patients, fp)

    return tpr_mean, fpr_mean, thresholds


def generate_confusion_matrix(
    predictions, ground_truths, num_classes=3, iou_threshold=0.5
):
    """
    为目标检测生成混淆矩阵

    参数:
    - predictions: 每张图片的预测列表，格式为 [image_id, [box, class_id, confidence]]
    - ground_truths: 每张图片的真实标注列表，格式为 [image_id, [box, class_id]]
    - class_names: 类别名称列表
    - iou_threshold: IoU阈值，用于匹配预测框和真实框

    返回:
    - confusion_matrix: 混淆矩阵
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # 按图片ID组织数据
    pred_by_image = defaultdict(list)
    gt_by_image = defaultdict(list)

    _mapper = {
        1: 0,
        3: 1,
        4: 2,
    }

    for pred in predictions:
        image_id, box, class_id, _, _ = pred.values()
        if class_id == 2 or class_id == 5:
            continue
        pred_by_image[image_id].append((box, _mapper[class_id]))

    for gt in ground_truths:
        image_id, box, class_id, _, _ = gt.values()
        gt_by_image[image_id].append((box, _mapper[class_id]))

    # 处理每张图片
    for image_id in tqdm(gt_by_image.keys(), desc="Generating confusion matrix"):
        gt_boxes = gt_by_image[image_id]
        pred_boxes = pred_by_image.get(image_id, [])

        # 如果图片中没有预测框，所有真实框都视为漏检
        if not pred_boxes:
            for gt_box, gt_class in gt_boxes:
                # 将真实类别对应的行的背景类（可选）列加1，表示漏检
                # 这里以-1表示背景类别（如果不想包含背景类，可以省略这一步）
                # confusion_matrix[gt_class, -1] += 1
                pass
            continue

        # 计算IoU矩阵
        ious = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, (gt_box, _) in enumerate(gt_boxes):
            for j, (pred_box, _) in enumerate(pred_boxes):
                ious[i, j] = _get_iou(gt_box, pred_box)

        # 贪婪匹配: 为每个GT找最好的预测
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # 首先匹配IoU大于阈值的框
        for i in range(len(gt_boxes)):
            best_iou = iou_threshold
            best_match = -1

            for j in range(len(pred_boxes)):
                if ious[i, j] > best_iou and not pred_matched[j]:
                    best_iou = ious[i, j]
                    best_match = j

            if best_match != -1:
                gt_class = gt_boxes[i][1]
                pred_class = pred_boxes[best_match][1]
                confusion_matrix[gt_class, pred_class] += 1

                gt_matched[i] = True
                pred_matched[best_match] = True

        # 处理未匹配的GT (漏检)
        for i, matched in enumerate(gt_matched):
            if not matched:
                gt_class = gt_boxes[i][1]
                # 漏检: 未被检测到的真实物体
                # 如果包含背景类，可以记录为该类被分类为背景
                # confusion_matrix[gt_class, -1] += 1
                pass

        # 处理未匹配的预测 (误检)
        for j, matched in enumerate(pred_matched):
            if not matched:
                pred_class = pred_boxes[j][1]
                # 误检: 检测到不存在的物体
                # 如果包含背景类，可以记录为背景被分类为某个类
                # confusion_matrix[-1, pred_class] += 1
                pass

    return confusion_matrix
