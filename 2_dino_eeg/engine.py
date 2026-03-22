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
    # 鍒濆鍖栨搴︿负闆讹紙姊害绱Н闇€瑕侊級
    optimizer.zero_grad()
    
    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header, logger=logger
    ):
        # samples: NestedTensor
        # tensors: [bs, 3, W, H]
        # mask: [bs, H, W]
        samples = samples.to(device)
        # targets: list: bs
        # 姣忓紶鍥剧墖dict 7
        # 'boxes'=[num, 4] 'labels'=num prig_size: 鍘熷澶у皬 size: pad鍚庡ぇ灏?area image_id iscrowd
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        # 鍓嶅悜浼犳挱
        # dict:6
        # pred_logits 鍒嗙被澶磋緭鍑篬bs, 900, 92(绫诲埆鏁?]
        # pred_boxes 鍥炲綊澶磋緭鍑篬bs, 900, 4]
        # aux_outputs list: 5 鍓?涓猟ecoder灞傝緭鍑?5涓猵red_logits[bs, 100, 92(绫诲埆鏁?]鍜?涓猵red_boxes[bs, 100, 4]
        # interm_outputs encoder杈撳嚭 {pred_logits:[bs, 900, num_classes],pred_boxes:[bs, 900, 4]}
        # interm_outputs_for_matching_pre encoder杈撳嚭 {pred_logits:[bs, 900, num_classes],pred_boxes:[bs, 900, 4]} 鍒濆鍖栨娴嬫
        # dn_meta CDN鐩稿叧淇℃伅
        with torch.amp.autocast(device_type="cuda", enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # 璁＄畻鎹熷け 鐢ㄤ簬log鏃ュ織: 'class_error' + 'cardinality_error'
            loss_dict = criterion(outputs, targets)
            # 鏉冮噸绯绘暟 {'loss_ce':1, 'loss_bbox':5, 'loss_giou':2, loss_ce_dn = 1, loss_bbox_dn = 5, loss_giou = 2}
            weight_dict = criterion.weight_dict

            # 鎬绘崯澶?= 鍥炲綊鎹熷け: loss_bbox(L1) + loss_bbox + 鍒嗙被鎹熷け: loss_ce
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

        # 姊害绱Н鏀寔
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        losses = losses / gradient_accumulation_steps  # 缂╂斁鎹熷け
        
        # amp backward function
        if args.amp:
            scaler.scale(losses).backward()  # 鍙嶅悜浼犳挱璁＄畻姊害 骞剁疮鍔犳搴?
            
            # 鍙湪绱Н姝ユ暟杈惧埌鏃舵洿鏂板弬鏁?
            if (_cnt + 1) % gradient_accumulation_steps == 0:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)  # 鏇存柊鍙傛暟
                scaler.update()
                optimizer.zero_grad()  # 姊害娓呴浂
        else:
            # original backward function
            losses.backward()
            
            # 鍙湪绱Н姝ユ暟杈惧埌鏃舵洿鏂板弬鏁?
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

    # groundtruth_bbs鐨刬tem鍖呮嫭'image_id'锛?label'锛?box'涓変釜瀛楁锛屽搴旀爣绛炬墍鍦ㄧ殑鍥剧墖锛岀被锛岀粷瀵瑰潗鏍?
    groundtruth_bbs: list[dict] = []
    # detected_bbs鐨刬tem鍖呮嫭'image_id'锛?label'锛?box'锛?score'鍥涗釜瀛楁锛屽搴旇棰勬祴鎵€鍦ㄧ殑鍥剧墖锛岄娴嬬殑绫伙紝棰勬祴鐨勭粷瀵瑰潗鏍囷紝缃俊搴?
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
                # 鍓嶅悜浼犳挱
                # dict: 3
                # 0 pred_logits 鍒嗙被澶磋緭鍑篬bs, 100, 92(绫诲埆鏁?]
                # 1 pred_boxes 鍥炲綊澶磋緭鍑篬bs, 100, 4]
                # 3 aux_outputs list: 5  鍓?涓猟ecoder灞傝緭鍑?5涓猵red_logits[bs, 100, 92(绫诲埆鏁?] 鍜?5涓猵red_boxes[bs, 100, 4]
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

        # [B,]  B寮犲浘鐗囩殑鍘熷浘澶у皬
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 闀垮害涓築鐨刲ist锛屾瘡涓猧tem涓篸ict锛屽寘鍚珄'scores' [NS,]锛?labels' [NS,]锛?boxes' [NS, 2]}
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
                # 杩囨护鎺塻core灏忎簬0.1鐨勬牱鏈?
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
                # 杩囨护鎺塻core灏忎簬0.1鐨勬牱鏈?
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

    # 淇濆瓨涓篔SON鏂囦欢锛堝彲閫夛紝鐢ㄤ簬璋冭瘯锛?
    gt_json_path = os.path.join(output_dir, 'ground_truth.bbox.json')
    pred_json_path = os.path.join(output_dir, 'results.bbox.json')
    with open(gt_json_path, 'w') as f:
        json.dump(j_data_gt, f, ensure_ascii=False)
    with open(pred_json_path, 'w') as f:
        json.dump(j_data_dt, f, ensure_ascii=False)
    
    # 鐩存帴浠庡唴瀛樻暟鎹绠桭1鍒嗘暟浣跨敤integrated_evaluation
    f1_score = 0.0
    try:
        from integrated_evaluation import IntegratedEvaluator
        
        evaluator = IntegratedEvaluator()
        result = evaluator.run_quick_evaluation_from_data(
            j_data_gt,  # 鐩存帴浣跨敤鍐呭瓨涓殑GT鏁版嵁
            j_data_dt,  # 鐩存帴浣跨敤鍐呭瓨涓殑棰勬祴鏁版嵁
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

