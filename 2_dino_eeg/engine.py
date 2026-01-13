# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable

import torch
from torch.cuda.amp import GradScaler
from util.utils import to_device
import util.misc as utils
from datasets import get_summary, get_metrics, get_event_metrics, get_patient_metircs
import json

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
    ema_m=None,
):
    scaler = GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Epoch: [{}]".format(epoch)

    print_freq = 100

    _cnt = 0
    # 初始化梯度为零（梯度累积需要）
    optimizer.zero_grad()
    
    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header, logger=logger
    ):
        # samples: NestedTensor
        # tensors: [bs, 3, W, H]
        # mask: [bs, H, W]
        samples = samples.to(device)
        # targets: list: bs
        # 每张图片dict 7
        # 'boxes'=[num, 4] 'labels'=num prig_size: 原始大小 size: pad后大小 area image_id iscrowd
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        # 前向传播
        # dict:6
        # pred_logits 分类头输出[bs, 900, 92(类别数)]
        # pred_boxes 回归头输出[bs, 900, 4]
        # aux_outputs list: 5 前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)]和5个pred_boxes[bs, 100, 4]
        # interm_outputs encoder输出 {pred_logits:[bs, 900, num_classes],pred_boxes:[bs, 900, 4]}
        # interm_outputs_for_matching_pre encoder输出 {pred_logits:[bs, 900, num_classes],pred_boxes:[bs, 900, 4]} 初始化检测框
        # dn_meta CDN相关信息
        with torch.amp.autocast(device_type="cuda", enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # 计算损失 用于log日志: 'class_error' + 'cardinality_error'
            loss_dict = criterion(outputs, targets)
            # 权重系数 {'loss_ce':1, 'loss_bbox':5, 'loss_giou':2, loss_ce_dn = 1, loss_bbox_dn = 5, loss_giou = 2}
            weight_dict = criterion.weight_dict

            # 总损失 = 回归损失: loss_bbox(L1) + loss_bbox + 分类损失: loss_ce
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 梯度累积支持
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        losses = losses / gradient_accumulation_steps  # 缩放损失
        
        # amp backward function
        if args.amp:
            scaler.scale(losses).backward()  # 反向传播计算梯度 并累加梯度
            
            # 只在累积步数达到时更新参数
            if (_cnt + 1) % gradient_accumulation_steps == 0:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)  # 更新参数
                scaler.update()
                optimizer.zero_grad()  # 梯度清零
        else:
            # original backward function
            losses.backward()
            
            # 只在累积步数达到时更新参数
            if (_cnt + 1) % gradient_accumulation_steps == 0:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

        # torch.cuda.empty_cache()

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    device,
    output_dir,
    wo_class_error=False,
    select_thresholds_index=None,
    args=None,
    logger=None,
):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    print_fn = logger.info if logger else print

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    # groundtruth_bbs的item包括'image_id'，'label'，'box'三个字段，对应标签所在的图片，类，绝对坐标
    groundtruth_bbs: list[dict] = []
    # detected_bbs的item包括'image_id'，'label'，'box'，'score'四个字段，对应该预测所在的图片，预测的类，预测的绝对坐标，置信度
    detected_bbs: list[dict] = []
    j_data_gt=[]
    j_data_dt=[]
    for samples, targets in metric_logger.log_every(
        data_loader, 100, header, logger=logger
    ):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type="cuda", enabled=args.amp):
            if need_tgt_for_training:
                # 前向传播
                # dict: 3
                # 0 pred_logits 分类头输出[bs, 100, 92(类别数)]
                # 1 pred_boxes 回归头输出[bs, 100, 4]
                # 3 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])

        # [B,]  B张图片的原图大小
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 长度为B的list，每个item为dict，包含{'scores' [NS,]，'labels' [NS,]，'boxes' [NS, 2]}
        results = postprocessors["bbox"](outputs, orig_target_sizes)

        # targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, "cpu") for k, v in t.items()} for t in targets]

        for target in targets:
            for box, label in zip(target["boxes_eval"], target["labels"]):
                groundtruth_bbs.append(
                    {
                        "image_id": target["image_id"],
                        "box": box.tolist(),
                        "label": int(label.item()),
                        "patient": target["patient"],
                        "orig_size": target["orig_size"].item(),
                    }
                )
                bbox = box.tolist()
                j_data_gt.append({
                    "image_id": target["image_id"],
                    "bbox": [bbox[0], 0, max(bbox[1] - bbox[0], 0), 63],
                    "category_id": int(label.item()),
                    "width": target["orig_size"].item(),
                })
        for target, result in zip(targets, results):
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                # 过滤掉score小于0.1的样本
                if score.item() >= 0.05:
                    detected_bbs.append(
                        {
                            "image_id": target["image_id"],
                            "box": box.tolist(),
                            "label": int(label.item()),
                            "score": score.item(),
                            "patient": target["patient"],
                        }
                    )

                bbox = box.tolist()
                # 过滤掉score小于0.1的样本
                if score.item() >= 0.05:
                    j_data_dt.append({
                        "image_id": target["image_id"],
                        "bbox": [bbox[0], 0, max(bbox[1] - bbox[0], 0), 63],
                        "score": score.item(),
                        "category_id": int(label.item()),
                        "width": target["orig_size"].item(),
                    })
        # del targets, results
        # torch.cuda.empty_cache()

    # 保存为JSON文件（可选，用于调试）
    with open('ground_truth.bbox.json', 'w') as f:
        json.dump(j_data_gt, f, ensure_ascii=False)
    with open('results.bbox.json', 'w') as f:
        json.dump(j_data_dt, f, ensure_ascii=False)
    
    # 直接从内存数据计算F1分数使用integrated_evaluation
    f1_score = 0.0
    try:
        from integrated_evaluation import IntegratedEvaluator
        
        evaluator = IntegratedEvaluator()
        result = evaluator.run_quick_evaluation_from_data(
            j_data_gt,  # 直接使用内存中的GT数据
            j_data_dt,  # 直接使用内存中的预测数据
            "quick_eval_results"
        )
        f1_score = result.get('best_f1_score', 0.0)
        print_fn(f"Integrated F1 Score: {f1_score:.4f}")
    except Exception as e:
        print_fn(f"Warning: F1 score calculation failed: {e}")
    
    print("Averaged stats:", metric_logger)

    res_dict = get_metrics(groundtruth_bbs, detected_bbs, max_dets=100)
    for v in res_dict.values():
        print_fn(
            "class: {}, AP: {}, total positives: {}, TP: {}, FP: {}.".format(
                v["class"],
                v["AP"],
                v["total positives"],
                v["TP"],
                v["FP"],
            )
        )

    res_dict, select_thresholds_index, iras = get_event_metrics(
        groundtruth_bbs, detected_bbs, select_thresholds_index
    )
    for m, d_aver in res_dict.items():
        for a, d_res in d_aver.items():
            for k, v in d_res.items():
                print_fn(f"method: {m}, average: {a}, {k}: {v}")

    res_dict = get_summary(groundtruth_bbs, detected_bbs)
    for k, v in res_dict.items():
        print_fn(f"{k}: {v}")

    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    stats["eval_bbox"] = res_dict["AP"]
    stats["eval_f1"] = f1_score

    print_fn(select_thresholds_index)
    if iras:
        print_fn("FEA mean:{}".format(iras[0]))
        print_fn("FEA std:{}".format(iras[1]))
        print_fn("FEDA mean:{}".format(iras[2]))
        print_fn("FEDA std:{}".format(iras[3]))
        print_fn("IRA mean:{}".format(iras[4]))
        print_fn("IRA std:{}".format(iras[5]))

    return stats, select_thresholds_index
