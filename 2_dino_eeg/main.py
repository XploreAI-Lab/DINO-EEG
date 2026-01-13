# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import pickle
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
    model.eval()
    criterion.eval()

    amp_enabled = bool(getattr(args, "amp", False))
    output_dir = "." if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)

    gt_json_path = os.path.join(output_dir, "ground_truth.bbox.json")
    raw_predictions_path = os.path.join(output_dir, "raw_predictions.pkl")

    j_data_gt = []
    with open(raw_predictions_path, "wb") as pf:
        for samples, targets in data_loader:
            samples = samples.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(samples)

            pred_logits = outputs["pred_logits"].detach().cpu().to(torch.float16)
            pred_boxes = outputs["pred_boxes"].detach().cpu().to(torch.float16)
            for i, target in enumerate(targets):
                pickle.dump(
                    {
                        "image_id": target["image_id"],
                        "pred_logits": pred_logits[i],
                        "pred_boxes": pred_boxes[i],
                    },
                    pf,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            for target in targets:
                for box, label in zip(target["boxes_eval"], target["labels"]):
                    bbox = box.tolist()
                    j_data_gt.append(
                        {
                            "image_id": target["image_id"],
                            "bbox": [bbox[0], 0, max(bbox[1] - bbox[0], 0), 63],
                            "category_id": int(label.item()),
                            "width": target["orig_size"].item(),
                        }
                    )

            del outputs, pred_logits, pred_boxes

    with open(gt_json_path, "w") as f:
        json.dump(j_data_gt, f, ensure_ascii=False)

    stats = {
        "eval_bbox": 0.0,
        "eval_f1": 0.0,
        "gt_json_path": gt_json_path,
        "raw_predictions_path": raw_predictions_path,
    }
    return stats, select_thresholds_index
