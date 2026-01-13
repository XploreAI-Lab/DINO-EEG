#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合评估脚本
直接通过三个JSON文件快速得到多通道NMS合并后的最高事件级别F1分数
整合了DataProcessor、Evaluator和QuickEvaluator的功能
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

try:
    from epilepsy2bids.annotations import Annotations, SeizureType
    from timescoring import annotations, scoring
except ImportError:
    print("Warning: epilepsy2bids or timescoring not available. Some functions may not work.")
    Annotations = None
    annotations = None
    scoring = None

class IntegratedEvaluator:
    """整合评估器 - 包含数据处理、评估和快速评估功能"""
    
    def __init__(self, fs: int = 200):
        self.fs = fs  # 采样频率
    
    # ========== 数据处理功能 ==========
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = x1_min + box1[2], y1_min + box1[3]
        
        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = x2_min + box2[2], y2_min + box2[3]
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0
        return inter_area / union_area
    
    def merge_multichannel_predictions(self, predictions_path: str, output_path: str, 
                                     merge_strategy: str = 'nms', 
                                     score_threshold: float = 0.2,
                                     iou_threshold: float = 0.0) -> None:
        """合并多通道预测结果"""
        preds = self.load_json(predictions_path)
        
        # 按段分组（去除通道名）
        merged = defaultdict(list)
        for pred in preds:
            raw_id = pred['image_id']
            base_id = os.path.splitext(raw_id)[0]
            seg_id = '_'.join(base_id.split('_')[:3])  # 去除通道名
            merged[seg_id].append(pred)
        
        final_preds = []
        
        for seg_id, boxes in merged.items():
            if merge_strategy == 'nms':
                # NMS策略：保留非重叠的高置信度框
                kept = []
                for box in sorted(boxes, key=lambda x: -x['score']):
                    if box['score'] < score_threshold:
                        break
                    overlap_found = False
                    for kept_box in kept:
                        if self.calculate_iou(box['bbox'], kept_box['bbox']) > iou_threshold:
                            overlap_found = True
                            break
                    if not overlap_found:
                        new_box = box.copy()
                        new_box['image_id'] = seg_id
                        final_preds.append(new_box)
                        kept.append(box)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_preds, f, indent=2)
        
        print(f"合并完成，共输出 {len(final_preds)} 条预测。结果保存在 {output_path}")
    
    def json_to_tsv(self, gt_json_path: str, pred_json_path: str, meta_json_path: Optional[str], 
                   gt_output_dir: str, hyp_output_dir: str, 
                   score_threshold: float = 0.3, max_predictions: int = 18) -> None:
        """将JSON格式的标注和预测转换为TSV格式"""
        # 加载数据
        gt_data = self.load_json(gt_json_path)
        pred_data = self.load_json(pred_json_path)
        
        # 构建映射
        h5_to_jpg = {}
        filename_to_width = {}
        seg_to_width = {}  # 段级别的宽度映射
        
        # 如果提供了元数据文件，使用元数据
        if meta_json_path and os.path.exists(meta_json_path):
            meta_data = self.load_json(meta_json_path)
            
            # 处理不同格式的元数据
            if 'images' in meta_data:
                images_data = meta_data['images']
            else:
                images_data = meta_data
            
            for item in images_data:
                if 'file_name' in item:
                    jpg_name = item['file_name']
                    width = item['width']
                    h5_name = os.path.splitext(jpg_name)[0] + '.h5'
                else:
                    h5_name = item['h5_name']
                    jpg_name = item['jpg_name']
                    width = item['width']
                
                h5_to_jpg[h5_name] = jpg_name
                filename_to_width[jpg_name] = width
                
                # 提取段级别ID
                base_id = os.path.splitext(h5_name)[0]
                seg_id = '_'.join(base_id.split('_')[:3])
                if seg_id not in seg_to_width:
                    seg_to_width[seg_id] = width
        else:
            print("未提供元数据文件，将从GT和预测数据中获取宽度信息...")
            # 从GT和预测数据中获取宽度信息
            all_data = gt_data + pred_data
            for item in all_data:
                image_id = item['image_id']
                bbox = item['bbox']
                # 优先使用JSON条目中的width字段，如果没有则从bbox推断
                if 'width' in item:
                    inferred_width = item['width']
                else:
                    inferred_width = bbox[0] + bbox[2] + 10  # 添加一些缓冲
                
                # 检查是否为合并格式
                if len(image_id.split('_')) == 3:
                    # 合并格式，直接使用seg_id
                    seg_id = image_id
                    if seg_id not in seg_to_width:
                        seg_to_width[seg_id] = inferred_width
                    else:
                        seg_to_width[seg_id] = max(seg_to_width[seg_id], inferred_width)
                else:
                    # 原始格式
                    h5_name = image_id
                    jpg_name = os.path.splitext(h5_name)[0] + '.jpg'
                    h5_to_jpg[h5_name] = jpg_name
                    
                    if jpg_name not in filename_to_width:
                        filename_to_width[jpg_name] = inferred_width
                    else:
                        filename_to_width[jpg_name] = max(filename_to_width[jpg_name], inferred_width)
                    
                    # 同时更新段级别映射
                    base_id = os.path.splitext(h5_name)[0]
                    seg_id = '_'.join(base_id.split('_')[:3])
                    if seg_id not in seg_to_width:
                        seg_to_width[seg_id] = inferred_width
                    else:
                        seg_to_width[seg_id] = max(seg_to_width[seg_id], inferred_width)
        
        # 检查预测数据是否为合并后的格式
        is_merged_format = False
        if pred_data:
            sample_id = pred_data[0]['image_id']
            if len(sample_id.split('_')) == 3:
                is_merged_format = True
        
        if is_merged_format:
            print("检测到多通道合并格式，按段级别处理...")
            gt_annotations = defaultdict(list)
            for ann in gt_data:
                h5_name = ann['image_id']
                base_id = os.path.splitext(h5_name)[0]
                seg_id = '_'.join(base_id.split('_')[:3])
                gt_annotations[seg_id].append(ann)
            
            hyp_predictions = defaultdict(list)
            for pred in pred_data:
                seg_id = pred['image_id']
                hyp_predictions[seg_id].append(pred)
            
            target_mapping = seg_to_width
            target_keys = seg_to_width.keys()
        else:
            print("检测到原始多通道格式，按文件级别处理...")
            gt_annotations = defaultdict(list)
            for ann in gt_data:
                h5_name = ann['image_id']
                if h5_name in h5_to_jpg:
                    jpg_name = h5_to_jpg[h5_name]
                    gt_annotations[jpg_name].append(ann)
            
            hyp_predictions = defaultdict(list)
            for pred in pred_data:
                h5_name = pred['image_id']
                if h5_name in h5_to_jpg:
                    jpg_name = h5_to_jpg[h5_name]
                    hyp_predictions[jpg_name].append(pred)
            
            target_mapping = filename_to_width
            target_keys = filename_to_width.keys()
        
        # 创建输出目录
        os.makedirs(gt_output_dir, exist_ok=True)
        os.makedirs(hyp_output_dir, exist_ok=True)
        
        # 生成TSV文件
        for key_name in target_keys:
            width = target_mapping[key_name]
            
            if is_merged_format:
                base_name = key_name + "_events.tsv"
            else:
                base_name = os.path.splitext(key_name)[0] + "_events.tsv"
            
            # 写GT文件
            gt_path = os.path.join(gt_output_dir, base_name)
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n")
                if key_name in gt_annotations:
                    for ann in gt_annotations[key_name]:
                        bbox = ann['bbox']
                        onset = bbox[0]
                        duration = bbox[2]
                        line = f"{onset:.2f}\t{duration:.2f}\tsz\t1.00\tn/a\t2014-01-01 00:00:00\t{width:.2f}\n"
                        f.write(line)
                else:
                    line = f"0.01\t1.00\tbckg\t1.00\tn/a\t2015-01-01 00:00:00\t{width:.2f}\n"
                    f.write(line)
            
            # 写预测文件
            hyp_path = os.path.join(hyp_output_dir, base_name)
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n")
                if key_name in hyp_predictions:
                    any_valid = False
                    cnt = 0
                    for pred in hyp_predictions[key_name]:
                        cnt += 1
                        score = pred['score']
                        if cnt > max_predictions or score < score_threshold:
                            continue
                        bbox = pred['bbox']
                        onset = bbox[0]
                        duration = bbox[2]
                        line = f"{onset:.2f}\t{duration:.2f}\tsz\t{score:.4f}\tn/a\t2014-01-01 00:00:00\t{width:.2f}\n"
                        f.write(line)
                        any_valid = True
                    if not any_valid:
                        line = f"0.01\t1.00\tbckg\t1.00\tn/a\t2015-01-01 00:00:00\t{width:.2f}\n"
                        f.write(line)
                else:
                    line = f"0.01\t1.00\tbckg\t1.00\tn/a\t2015-01-01 00:00:00\t{width:.2f}\n"
                    f.write(line)
        
        print(f"TSV转换完成，共生成 {len(target_keys)} 对文件")
    
    # ========== 评估功能 ==========
    
    def to_mask(self, annotations_obj) -> np.ndarray:
        """将标注对象转换为掩码"""
        if not annotations_obj or not hasattr(annotations_obj, 'events') or not annotations_obj.events:
            return np.array([])
        
        mask = np.zeros(int(annotations_obj.events[0]["recordingDuration"] * self.fs))
        for event in annotations_obj.events:
            if event["eventType"].value != "bckg":
                start = round(event["onset"] * self.fs)
                end = round((event["onset"] + event["duration"]) * self.fs)
                mask[start:end] = 1
        return mask
    
    def compute_scores(self, tp: int, fp: int, ref_true: int, duration: float) -> Tuple[float, float, float, float]:
        """计算评估指标"""
        if ref_true > 0:
            sensitivity = tp / ref_true
        else:
            sensitivity = np.nan
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = np.nan
        
        if np.isnan(sensitivity) or np.isnan(precision):
            f1 = np.nan
        elif (sensitivity + precision) == 0:
            f1 = 0
        else:
            f1 = 2 * sensitivity * precision / (sensitivity + precision)
        
        fp_rate = fp / (duration / 3600 / 24)  # FP per 24h
        return sensitivity, precision, f1, fp_rate
    
    def load_and_filter_tsv(self, tsv_path: str, conf_threshold: float) -> Optional[Any]:
        """加载并过滤TSV文件"""
        if not Annotations:
            raise ImportError("epilepsy2bids not available")
        
        try:
            anno = Annotations.loadTsv(tsv_path)
            if hasattr(anno, "events"):
                anno.events = [
                    ev for ev in anno.events
                    if ev["eventType"].value != "bckg" and ev.get("confidence", 1.0) >= conf_threshold
                ]
            if not anno.events:
                return None
            return anno
        except Exception as e:
            print(f"Failed to load {tsv_path}: {e}")
            return None
    
    def evaluate_dataset(self, ref_folder: str, hyp_folder: str, output_file: str, 
                        conf_threshold: float = 0.0, avg_per_subject: bool = True) -> Dict[str, Any]:
        """评估数据集"""
        if not all([Annotations, annotations, scoring]):
            raise ImportError("Required packages not available")
        
        ref_folder = Path(ref_folder)
        hyp_folder = Path(hyp_folder)
        
        sample_scores = {}
        event_scores = {}
        
        # 递归查找所有tsv文件
        for ref_tsv in ref_folder.glob("*.tsv"):
            # 从文件名前缀获取subject id
            base = ref_tsv.stem.split('_')[0]
            subject = f"sub-{base}"
            
            if subject not in sample_scores:
                sample_scores[subject] = {"tp": 0, "fp": 0, "refTrue": 0, "duration": 0}
                event_scores[subject] = {"tp": 0, "fp": 0, "refTrue": 0, "duration": 0}
            
            # 加载参考标注
            ref = Annotations.loadTsv(str(ref_tsv))
            ref_ann = annotations.Annotation(self.to_mask(ref), self.fs)
            
            # 对应的hyp文件
            hyp_tsv = hyp_folder / ref_tsv.name
            if hyp_tsv.exists():
                hyp = self.load_and_filter_tsv(str(hyp_tsv), conf_threshold)
                if hyp is None:
                    hyp_ann = annotations.Annotation(np.zeros_like(ref_ann.mask), self.fs)
                else:
                    hyp_ann = annotations.Annotation(self.to_mask(hyp), self.fs)
            else:
                hyp_ann = annotations.Annotation(np.zeros_like(ref_ann.mask), self.fs)
            
            # 计算分数
            sample_score = scoring.SampleScoring(ref_ann, hyp_ann)
            event_score = scoring.EventScoring(ref_ann, hyp_ann)
            
            # 累加样本级统计
            sample_scores[subject]["tp"] += sample_score.tp
            sample_scores[subject]["fp"] += sample_score.fp
            sample_scores[subject]["refTrue"] += sample_score.refTrue
            sample_scores[subject]["duration"] += len(ref_ann.mask) / ref_ann.fs
            
            # 累加事件级统计
            event_scores[subject]["tp"] += event_score.tp
            event_scores[subject]["fp"] += event_score.fp
            event_scores[subject]["refTrue"] += event_score.refTrue
            event_scores[subject]["duration"] += len(ref_ann.mask) / ref_ann.fs
        
        # 汇总结果
        sample_results = {}
        event_results = {}
        
        if avg_per_subject:
            for score_dict, out_dict in [(sample_scores, sample_results), (event_scores, event_results)]:
                for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                    values = []
                    for stats in score_dict.values():
                        vals = self.compute_scores(stats["tp"], stats["fp"], stats["refTrue"], stats["duration"])
                        idx = ["sensitivity", "precision", "f1", "fpRate"].index(metric)
                        values.append(vals[idx])
                    out_dict[metric] = np.nanmean(values)
                    out_dict[f"{metric}_std"] = np.nanstd(values)
        else:
            # 累加所有subject
            for total_dict, out_dict in [({"tp":0, "fp":0, "refTrue":0, "duration":0}, sample_results),
                                         ({"tp":0, "fp":0, "refTrue":0, "duration":0}, event_results)]:
                source = sample_scores if out_dict is sample_results else event_scores
                for stats in source.values():
                    for k in total_dict:
                        total_dict[k] += stats[k]
                sens, prec, f1, fp_r = self.compute_scores(total_dict["tp"], total_dict["fp"], 
                                                          total_dict["refTrue"], total_dict["duration"])
                out_dict.update({"sensitivity": sens, "precision": prec, "f1": f1, "fpRate": fp_r})
        
        output = {"sample_results": sample_results, "event_results": event_results}
        
        # 保存结果
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        return output
    
    # ========== 快速评估功能 ==========
    
    def run_quick_evaluation_from_data(self,
                                     gt_data: List[Dict[str, Any]],
                                     pred_data: List[Dict[str, Any]],
                                     output_dir: str = "quick_eval_results") -> Dict[str, Any]:
        """直接从内存数据进行快速评估流程"""
        print("=== 快速评估流程启动（内存数据）===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 步骤1: 多通道NMS合并（阈值0.01）
        print("步骤1: 执行多通道NMS合并（阈值0.01）...")
        
        # 按段分组（去除通道名）
        merged = defaultdict(list)
        for pred in pred_data:
            raw_id = pred['image_id']
            base_id = os.path.splitext(raw_id)[0]
            seg_id = '_'.join(base_id.split('_')[:3])  # 去除通道名
            merged[seg_id].append(pred)
        
        merged_predictions = []
        for seg_id, boxes in merged.items():
            # NMS策略：保留非重叠的高置信度框
            kept = []
            for box in sorted(boxes, key=lambda x: -x['score']):
                if box['score'] < 0.01:
                    break
                overlap_found = False
                for kept_box in kept:
                    if self.calculate_iou(box['bbox'], kept_box['bbox']) > 0.0:
                        overlap_found = True
                        break
                if not overlap_found:
                    new_box = box.copy()
                    new_box['image_id'] = seg_id
                    merged_predictions.append(new_box)
                    kept.append(box)
        
        print(f"多通道NMS合并完成，共输出 {len(merged_predictions)} 条预测")
        
        # 步骤2: 对0.1到0.9的9个置信度阈值进行评估
        print("步骤2: 评估不同置信度阈值...")
        thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
        best_f1 = 0.0
        best_threshold = 0.0
        best_result = None
        all_results = {}
        
        for threshold in thresholds:
            print(f"  评估阈值: {threshold:.1f}")
            
            # 为每个阈值创建TSV文件
            threshold_dir = output_dir / f"threshold_{threshold:.1f}"
            gt_dir = threshold_dir / "gt"
            hyp_dir = threshold_dir / "hyp"
            
            # 转换为TSV格式（使用内存数据）
            self._json_data_to_tsv(
                gt_data, merged_predictions, None,
                str(gt_dir), str(hyp_dir),
                score_threshold=threshold,
                max_predictions=18
            )
            
            # 执行评估
            try:
                output_file = threshold_dir / "evaluation_result.json"
                result = self.evaluate_dataset(
                    str(gt_dir), str(hyp_dir), str(output_file),
                    avg_per_subject=True
                )
                
                # 获取事件级别F1分数
                event_f1 = result.get('event_results', {}).get('f1', 0.0)
                all_results[threshold] = {
                    'threshold': threshold,
                    'event_f1': event_f1,
                    'full_result': result
                }
                
                if not np.isnan(event_f1) and event_f1 > best_f1:
                    best_f1 = event_f1
                    best_threshold = threshold
                    best_result = result
                    
                print(f"    事件级别F1: {event_f1:.4f}")
                
            except Exception as e:
                print(f"    评估失败: {e}")
                all_results[threshold] = {
                    'threshold': threshold,
                    'event_f1': 0.0,
                    'error': str(e)
                }
        
        # 步骤3: 汇总结果
        print("\n=== 评估结果汇总 ===")
        print(f"最佳置信度阈值: {best_threshold:.1f}")
        print(f"最高事件级别F1分数: {best_f1:.4f}")
        
        # 保存汇总结果
        summary_result = {
            'best_threshold': best_threshold,
            'best_f1_score': best_f1,
            'best_result': best_result,
            'all_thresholds': all_results,
            'summary': {
                'total_gt_annotations': len(gt_data),
                'total_predictions_before_merge': len(pred_data),
                'total_predictions_after_merge': len(merged_predictions),
                'thresholds_evaluated': len(thresholds)
            }
        }
        
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估完成！结果保存在: {output_dir}")
        print(f"汇总文件: {summary_file}")
        
        return summary_result
    
    def _json_data_to_tsv(self, gt_data: List[Dict[str, Any]], pred_data: List[Dict[str, Any]], 
                         meta_json_path: Optional[str], 
                         gt_output_dir: str, hyp_output_dir: str, 
                         score_threshold: float = 0.3, max_predictions: int = 18) -> None:
        """将内存中的JSON数据转换为TSV格式"""
        # 构建映射
        seg_to_width = {}  # 段级别的宽度映射
        
        # 从GT和预测数据中获取宽度信息
        all_data = gt_data + pred_data
        for item in all_data:
            image_id = item['image_id']
            bbox = item['bbox']
            # 优先使用JSON条目中的width字段，如果没有则从bbox推断
            if 'width' in item:
                inferred_width = item['width']
            else:
                raise ValueError("未在JSON条目中找到width字段,且无法从bbox推断宽度")
                
            
            # 检查是否为合并格式
            if len(image_id.split('_')) == 3:
                # 合并格式，直接使用seg_id
                seg_id = image_id
                if seg_id not in seg_to_width:
                    seg_to_width[seg_id] = inferred_width
                else:
                    seg_to_width[seg_id] = max(seg_to_width[seg_id], inferred_width)
            else:
                # 原始格式
                h5_name = image_id
                base_id = os.path.splitext(h5_name)[0]
                seg_id = '_'.join(base_id.split('_')[:3])
                if seg_id not in seg_to_width:
                    seg_to_width[seg_id] = inferred_width
                else:
                    seg_to_width[seg_id] = max(seg_to_width[seg_id], inferred_width)
        
        # 检查预测数据是否为合并后的格式
        is_merged_format = False
        if pred_data:
            sample_id = pred_data[0]['image_id']
            if len(sample_id.split('_')) == 3:
                is_merged_format = True
        
        if is_merged_format:
            print("检测到多通道合并格式，按段级别处理...")
            gt_annotations = defaultdict(list)
            for ann in gt_data:
                h5_name = ann['image_id']
                base_id = os.path.splitext(h5_name)[0]
                seg_id = '_'.join(base_id.split('_')[:3])
                gt_annotations[seg_id].append(ann)
            
            hyp_predictions = defaultdict(list)
            for pred in pred_data:
                seg_id = pred['image_id']
                hyp_predictions[seg_id].append(pred)
            
            target_keys = seg_to_width.keys()
        else:
            print("检测到原始多通道格式，按文件级别处理...")
            gt_annotations = defaultdict(list)
            for ann in gt_data:
                h5_name = ann['image_id']
                base_id = os.path.splitext(h5_name)[0]
                seg_id = '_'.join(base_id.split('_')[:3])
                gt_annotations[seg_id].append(ann)
            
            hyp_predictions = defaultdict(list)
            for pred in pred_data:
                h5_name = pred['image_id']
                base_id = os.path.splitext(h5_name)[0]
                seg_id = '_'.join(base_id.split('_')[:3])
                hyp_predictions[seg_id].append(pred)
            
            target_keys = seg_to_width.keys()
        
        # 创建输出目录
        os.makedirs(gt_output_dir, exist_ok=True)
        os.makedirs(hyp_output_dir, exist_ok=True)
        
        # 生成TSV文件
        for key_name in target_keys:
            width = seg_to_width[key_name]
            base_name = key_name + "_events.tsv"
            
            # 写GT文件
            gt_path = os.path.join(gt_output_dir, base_name)
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n")
                if key_name in gt_annotations:
                    for ann in gt_annotations[key_name]:
                        bbox = ann['bbox']
                        onset = bbox[0]
                        duration = bbox[2]
                        line = f"{onset:.2f}\t{duration:.2f}\tsz\t1.00\tn/a\t2014-01-01 00:00:00\t{width:.2f}\n"
                        f.write(line)
                else:
                    line = f"0.01\t1.00\tbckg\t1.00\tn/a\t2015-01-01 00:00:00\t{width:.2f}\n"
                    f.write(line)
            
            # 写预测文件
            hyp_path = os.path.join(hyp_output_dir, base_name)
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n")
                if key_name in hyp_predictions:
                    any_valid = False
                    cnt = 0
                    for pred in hyp_predictions[key_name]:
                        cnt += 1
                        score = pred['score']
                        if cnt > max_predictions or score < score_threshold:
                            continue
                        bbox = pred['bbox']
                        onset = bbox[0]
                        duration = bbox[2]
                        line = f"{onset:.2f}\t{duration:.2f}\tsz\t{score:.4f}\tn/a\t2014-01-01 00:00:00\t{width:.2f}\n"
                        f.write(line)
                        any_valid = True
                    if not any_valid:
                        line = f"0.01\t1.00\tbckg\t1.00\tn/a\t2015-01-01 00:00:00\t{width:.2f}\n"
                        f.write(line)
                else:
                    line = f"0.01\t1.00\tbckg\t1.00\tn/a\t2015-01-01 00:00:00\t{width:.2f}\n"
                    f.write(line)
        
        print(f"TSV转换完成，共生成 {len(target_keys)} 对文件")
    
    def run_quick_evaluation(self, 
                           gt_json_path: str,
                           pred_json_path: str,
                           meta_json_path: Optional[str] = None,
                           output_dir: str = "quick_eval_results") -> Dict[str, Any]:
        """快速评估流程"""
        print("=== 快速评估流程启动 ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 步骤1: 多通道NMS合并（阈值0.01）
        print("步骤1: 执行多通道NMS合并（阈值0.01）...")
        merged_path = output_dir / "merged_predictions_nms_0.01.json"
        
        self.merge_multichannel_predictions(
            pred_json_path, str(merged_path),
            merge_strategy='nms',
            score_threshold=0.01,
            iou_threshold=0.0
        )
        
        print(f"多通道NMS合并完成，结果保存至: {merged_path}")
        
        # 步骤2: 对0.1到0.9的9个置信度阈值进行评估
        print("步骤2: 评估不同置信度阈值...")
        thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
        best_f1 = 0.0
        best_threshold = 0.0
        best_result = None
        all_results = {}
        
        for threshold in thresholds:
            print(f"  评估阈值: {threshold:.1f}")
            
            # 为每个阈值创建TSV文件
            threshold_dir = output_dir / f"threshold_{threshold:.1f}"
            gt_dir = threshold_dir / "gt"
            hyp_dir = threshold_dir / "hyp"
            
            # 转换为TSV格式
            self.json_to_tsv(
                gt_json_path, str(merged_path), meta_json_path,
                str(gt_dir), str(hyp_dir),
                score_threshold=threshold,
                max_predictions=18
            )
            
            # 执行评估
            try:
                output_file = threshold_dir / "evaluation_result.json"
                result = self.evaluate_dataset(
                    str(gt_dir), str(hyp_dir), str(output_file),
                    avg_per_subject=True
                )
                
                # 获取事件级别F1分数
                event_f1 = result.get('event_results', {}).get('f1', 0.0)
                if event_f1 is None or np.isnan(event_f1):
                    event_f1 = 0.0
                all_results[f"threshold_{threshold:.1f}"] = {
                    'threshold': threshold,
                    'event_f1': event_f1,
                    'full_result': result
                }
                
                print(f"    事件级别F1分数: {event_f1:.4f}")
                
                # 更新最佳结果
                if event_f1 > best_f1:
                    best_f1 = event_f1
                    best_threshold = threshold
                    best_result = result
                    
            except Exception as e:
                print(f"    评估阈值 {threshold:.1f} 时出错: {e}")
                all_results[f"threshold_{threshold:.1f}"] = {
                    'threshold': threshold,
                    'event_f1': 0.0,
                    'error': str(e)
                }
        
        # 步骤3: 汇总结果
        print("\n=== 评估结果汇总 ===")
        print(f"最高事件级别F1分数: {best_f1:.4f}")
        print(f"对应置信度阈值: {best_threshold:.1f}")
        
        if best_result:
            print("\n最佳阈值详细结果:")
            if 'event_level' in best_result:
                event_metrics = best_result['event_level']
                print(f"  精确率: {event_metrics.get('precision', 0):.4f}")
                print(f"  召回率: {event_metrics.get('recall', 0):.4f}")
                print(f"  F1分数: {event_metrics.get('f1_score', 0):.4f}")
            
            if 'by_subject' in best_result:
                print(f"  受试者数量: {len(best_result['by_subject'])}")
        
        # 保存完整结果
        summary_result = {
            'best_f1_score': best_f1,
            'best_threshold': best_threshold,
            'best_result': best_result,
            'all_thresholds': all_results,
            'evaluation_summary': {
                'total_thresholds_evaluated': len([r for r in all_results.values() if 'error' not in r]),
                'failed_evaluations': len([r for r in all_results.values() if 'error' in r]),
                'f1_scores': {k: v['event_f1'] for k, v in all_results.items() if 'error' not in v}
            }
        }
        
        summary_path = output_dir / "quick_evaluation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n完整结果已保存至: {summary_path}")
        print("=== 快速评估完成 ===")
        
        return summary_result

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="整合评估脚本 - 直接获取多通道NMS合并后的最高事件级别F1分数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python integrated_evaluation.py --gt-json ground_truth.bbox.json --pred-json predict.json --meta-json TUSZ_tcp_test_annotations_full.json
        """
    )
    
    parser.add_argument('--gt-json', required=True, help='真实标注JSON文件路径')
    parser.add_argument('--pred-json', required=True, help='预测结果JSON文件路径')
    parser.add_argument('--meta-json', required=False, help='元数据JSON文件路径（可选）')
    parser.add_argument('--output-dir', default='quick_eval_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for file_path, name in [(args.gt_json, 'GT JSON'), (args.pred_json, 'Prediction JSON')]:
        if not Path(file_path).exists():
            print(f"错误: {name} 文件不存在: {file_path}")
            sys.exit(1)
    
    # 检查可选的meta_json文件
    if args.meta_json and not Path(args.meta_json).exists():
        print(f"错误: Meta JSON 文件不存在: {args.meta_json}")
        sys.exit(1)
    
    try:
        evaluator = IntegratedEvaluator()
        result = evaluator.run_quick_evaluation(
            args.gt_json, args.pred_json, args.meta_json, args.output_dir
        )
        
        print(f"\n✓ 快速评估成功完成！")
        print(f"✓ 最高事件级别F1分数: {result['best_f1_score']:.4f} (阈值: {result['best_threshold']:.1f})")
        
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()