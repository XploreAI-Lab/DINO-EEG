#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO-EEG 测试集数据筛选和评估程序
实现以下功能：
1. Channel级别数据筛选和临时文件生成
2. Global级别数据筛选和临时文件生成
3. 随机通道丢弃实验数据筛选和临时文件生成
4. 调用integrated_evaluation.py进行SampleScoring和EventScoring评估
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import random
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import os
from integrated_evaluation import IntegratedEvaluator

class DinoEEGEvaluator:
    """DINO-EEG数据筛选和评估器"""
    
    def __init__(self, fs: int = 200):
        self.fs = fs
        self.meta_data = None
        self.evaluator = IntegratedEvaluator(fs=fs)
        # 定义22通道TCP montage
        self.tcp_channels = [
            'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'CZ-C4', 'C3-CZ', 'T3-C3', 'C4-T4',
            'A1-T3', 'T4-A2'
        ]
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_meta_json(self, meta_json_path: Optional[str]) -> None:
        """加载meta-json文件，获取文件名到宽度的映射"""
        if meta_json_path and os.path.exists(meta_json_path):
            with open(meta_json_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # 构建文件名到宽度的映射
            self.meta_data = {}
            if 'images' in meta_data:
                for img in meta_data['images']:
                    file_name = img['file_name']
                    width = img['width']
                    self.meta_data[file_name] = width
            print(f"从meta文件加载了 {len(self.meta_data)} 个文件的宽度信息")
        else:
            print("未提供meta文件或文件不存在，将从bbox推断宽度")
            self.meta_data = {}
    
    def get_width_from_data(self, data: List[Dict[str, Any]], image_id: str) -> float:
        """从数据中获取图像宽度，优先使用meta数据"""
        # 首先尝试从meta数据获取宽度
        if self.meta_data:
            # 处理不同的文件名格式
            if image_id.endswith('.h5'):
                jpg_name = image_id.replace('.h5', '.jpg')
            else:
                jpg_name = image_id
            
            if jpg_name in self.meta_data:
                return self.meta_data[jpg_name]
        
        # 如果meta数据中没有，从JSON数据中获取
        for item in data:
            if item['image_id'] == image_id:
                if 'width' in item:
                    return item['width']
                else:
                    # 从bbox推断宽度
                    bbox = item['bbox']
                    return bbox[0] + bbox[2] + 10  # 添加一些缓冲
        return 1000  # 默认宽度
    

    
    def extract_channel_from_image_id(self, image_id: str) -> str:
        """从image_id中提取通道信息"""
        # 移除.h5后缀
        base_name = os.path.splitext(image_id)[0]
        # 提取通道部分（最后一个下划线后的部分）
        parts = base_name.split('_')
        if len(parts) >= 4:
            return parts[-1]  # 通道名
        return ""
    
    def extract_segment_id(self, filename: str) -> str:
        """从文件名中提取段ID（不包含通道）"""
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        if len(parts) >= 4:
            return '_'.join(parts[:3])  # subject_session_trial
        return base_name
    
    def generate_global_gt(self, gt_data: List[Dict]) -> List[Dict]:
        """生成全通道GT - 按段ID取并集"""
        print("生成全通道GT（按段ID取并集）...")
        
        # 按段ID分组GT数据
        gt_by_segment = defaultdict(list)
        for item in gt_data:
            segment_id = self.extract_segment_id(item['image_id'])
            if segment_id:
                gt_by_segment[segment_id].append(item)
        
        global_gt = []
        
        for segment_id, segment_gts in gt_by_segment.items():
            if not segment_gts:
                continue
                
            # 从metadata获取该段的宽度信息
            sample_image_id = segment_gts[0]['image_id']
            width = self.get_width_from_data([], sample_image_id)
            
            # 收集所有bbox的时间范围
            time_ranges = []
            for gt_item in segment_gts:
                bbox = gt_item['bbox']
                start_time = bbox[0]
                duration = bbox[2]
                end_time = start_time + duration
                time_ranges.append((start_time, end_time))
            
            # 合并重叠的时间范围
            if time_ranges:
                # 按开始时间排序
                time_ranges.sort()
                merged_ranges = [time_ranges[0]]
                
                for current_start, current_end in time_ranges[1:]:
                    last_start, last_end = merged_ranges[-1]
                    
                    # 如果当前范围与上一个范围重叠或相邻，则合并
                    if current_start <= last_end:
                        merged_ranges[-1] = (last_start, max(last_end, current_end))
                    else:
                        merged_ranges.append((current_start, current_end))
                
                # 为每个合并后的时间范围创建GT项
                for i, (start_time, end_time) in enumerate(merged_ranges):
                    # 创建统一的image_id格式（不包含通道信息）
                    unified_image_id = f"{segment_id}_global.h5"
                    
                    global_gt.append({
                        'image_id': unified_image_id,
                        'bbox': [start_time, 0, end_time - start_time, 63],
                        'category_id': 1,
                        'width': width
                    })
        
        print(f"原始GT数量: {len(gt_data)}, 全通道GT数量: {len(global_gt)}")
        return global_gt
    
    def align_predictions_with_global_gt(self, pred_data: List[Dict], global_gt: List[Dict]) -> List[Dict]:
        """将预测数据的image_id对齐到全通道GT格式，并从metadata获取width"""
        # 创建段ID到全通道GT image_id的映射
        segment_to_global_id = {}
        for gt_item in global_gt:
            image_id = gt_item['image_id']
            # 从image_id中提取段ID
            segment_id = self.extract_segment_id(image_id)
            if segment_id:
                segment_to_global_id[segment_id] = image_id
        
        aligned_preds = []
        for pred_item in pred_data:
            segment_id = self.extract_segment_id(pred_item['image_id'])
            aligned_pred = pred_item.copy()
            if segment_id in segment_to_global_id:
                aligned_pred['image_id'] = segment_to_global_id[segment_id]
            else:
                unified_image_id = f"{segment_id}_global.h5"
                aligned_pred['image_id'] = unified_image_id
            if not aligned_pred.get('width'):
                width = self.get_width_from_data([], aligned_pred['image_id'])
                if width != 1000:
                    aligned_pred['width'] = width
            aligned_preds.append(aligned_pred)
        
        return aligned_preds
    
    def evaluate_channel_level_with_shared_gt(self, shared_global_gt: List[Dict], pred_data: List[Dict], meta_json_path: Optional[str] = None) -> Dict[str, Any]:
        """Channel级别评估 - 使用共享的全通道GT，从metadata获取width"""
        print("=== Channel级别评估（使用共享GT和metadata width）===")
        
        # 按通道分组预测数据
        pred_by_channel = defaultdict(list)
        
        for item in pred_data:
            channel = self.extract_channel_from_image_id(item['image_id'])
            if channel:
                pred_by_channel[channel].append(item)
        
        channel_results = {}
        
        # 为每个通道创建临时文件并评估
        for channel in pred_by_channel.keys():
            pred_channel = pred_by_channel[channel]
            
            if not pred_channel:
                continue
            
            # 将该通道的预测数据对齐到全通道GT格式
            aligned_pred_channel = self.align_predictions_with_global_gt(pred_channel, shared_global_gt)
                
            print(f"评估通道: {channel} (共享全通道GT: {len(shared_global_gt)}, 该通道预测: {len(aligned_pred_channel)})")
            
            # 使用integrated_evaluation进行评估
            try:
                result = self.evaluator.run_quick_evaluation_from_data(
                    gt_data=shared_global_gt,
                    pred_data=aligned_pred_channel,
                    output_dir=f"temp_channel_{channel.replace('-', '_')}",
                    meta_json_path=meta_json_path
                )
                
                channel_results[channel] = {
                    'sample_scoring': result.get('sample_results', {}),
                    'event_scoring': result.get('event_results', {}),
                    'global_gt_count': len(shared_global_gt),
                    'channel_pred_count': len(aligned_pred_channel)
                }
            except Exception as e:
                print(f"通道 {channel} 评估失败: {e}")
                channel_results[channel] = {
                    'error': str(e),
                    'global_gt_count': len(shared_global_gt),
                    'channel_pred_count': len(aligned_pred_channel)
                }
        
        return {
            'channel_results': channel_results,
            'total_channels': len(channel_results),
            'global_gt_count': len(shared_global_gt)
        }

    
    def evaluate_global_level_with_shared_gt(self, shared_global_gt: List[Dict], pred_data: List[Dict], meta_json_path: Optional[str] = None) -> Dict[str, Any]:
        """Global级别评估 - 使用共享的全通道GT，从metadata获取width"""
        print("=== Global级别评估（使用共享GT和metadata width）===")
        
        # 将预测数据对齐到全通道GT格式
        aligned_pred_data = self.align_predictions_with_global_gt(pred_data, shared_global_gt)
        
        print(f"共享全通道GT数据量: {len(shared_global_gt)}, 对齐后预测数据量: {len(aligned_pred_data)}")
        
        try:
            gt_segments = set(self.extract_segment_id(item['image_id']) for item in shared_global_gt)
            pred_segments = set(self.extract_segment_id(item['image_id']) for item in aligned_pred_data)
            missing_segments = pred_segments - gt_segments
            use_meta = meta_json_path if not missing_segments else None
            result = self.evaluator.run_quick_evaluation_from_data(
                gt_data=shared_global_gt,
                pred_data=aligned_pred_data,
                output_dir="temp_global",
                meta_json_path=use_meta
            )
            
            return {
                'sample_scoring': result.get('sample_results', {}),
                'event_scoring': result.get('event_results', {}),
                'global_gt_count': len(shared_global_gt),
                'aligned_pred_count': len(aligned_pred_data)
            }
        except Exception as e:
            print(f"Global级别评估失败: {e}")
            return {
                'error': str(e),
                'global_gt_count': len(shared_global_gt),
                'aligned_pred_count': len(aligned_pred_data)
            }

    
    def random_channel_dropout_experiment_with_shared_gt(self, shared_global_gt: List[Dict], pred_data: List[Dict],
                                        dropout_counts: List[int] = [2, 4, 8, 16, 21],
                                        num_trials: int = 20, meta_json_path: Optional[str] = None) -> Dict[str, Any]:
        """随机通道丢弃实验 - 使用共享的全通道GT，从metadata获取width"""
        print("=== 随机通道丢弃实验（使用共享GT和metadata width）===")
        
        # 首先获取完整22通道的基线结果
        baseline_result = self.evaluate_global_level_with_shared_gt(shared_global_gt, pred_data)
        baseline_f1 = baseline_result.get('sample_scoring', {}).get('f1', 0.0)
        
        print(f"基线F1分数（完整22通道）: {baseline_f1:.4f}")
        
        dropout_results = {}
        
        for dropout_count in dropout_counts:
            print(f"\n测试丢弃 {dropout_count} 个通道...")
            
            trial_results = []
            
            for trial in range(num_trials):
                # 随机选择要丢弃的通道
                channels_to_drop = random.sample(self.tcp_channels, dropout_count)
                remaining_channels = [ch for ch in self.tcp_channels if ch not in channels_to_drop]
                
                # 过滤预测数据，只保留剩余通道的预测
                filtered_preds = []
                for pred in pred_data:
                    channel = self.extract_channel_from_image_id(pred['image_id'])
                    if channel in remaining_channels:
                        filtered_preds.append(pred)
                
                # 将筛选后的预测数据对齐到全通道GT格式
                aligned_filtered_preds = self.align_predictions_with_global_gt(filtered_preds, shared_global_gt)
                
                print(f"  试验 {trial + 1}: 共享全通道GT数量: {len(shared_global_gt)}, 筛选后预测数量: {len(aligned_filtered_preds)}")
                
                # 使用integrated_evaluation进行评估
                try:
                    result = self.evaluator.run_quick_evaluation_from_data(
                        gt_data=shared_global_gt,
                        pred_data=aligned_filtered_preds,
                        output_dir=f"temp_dropout_{dropout_count}_{trial}",
                        meta_json_path=meta_json_path
                    )
                    
                    trial_f1 = result.get('sample_results', {}).get('f1', 0.0)
                    sample_results = result.get('sample_results', {})
                    
                    trial_results.append({
                        'trial': trial + 1,
                        'dropped_channels': channels_to_drop,
                        'remaining_channels': remaining_channels,
                        'f1': trial_f1,
                        'precision': sample_results.get('precision', 0.0),
                        'recall': sample_results.get('recall', 0.0),
                        'f1_drop': baseline_f1 - trial_f1,
                        'sample_scoring': sample_results,
                        'event_scoring': result.get('event_results', {}),
                        'best_threshold': result.get('best_threshold', 0.0),
                        'global_gt_count': len(shared_global_gt),
                        'filtered_pred_count': len(aligned_filtered_preds)
                    })
                    
                except Exception as e:
                    print(f"  试验 {trial + 1} 评估失败: {e}")
                    trial_results.append({
                        'trial': trial + 1,
                        'dropped_channels': channels_to_drop,
                        'remaining_channels': remaining_channels,
                        'f1': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_drop': baseline_f1,
                        'error': str(e),
                        'global_gt_count': len(shared_global_gt),
                        'filtered_pred_count': len(aligned_filtered_preds) if 'aligned_filtered_preds' in locals() else 0
                    })
            
            # 计算该丢弃数量的统计结果
            f1_scores = [r['f1'] for r in trial_results]
            f1_drops = [r['f1_drop'] for r in trial_results]
            
            dropout_results[dropout_count] = {
                'trials': trial_results,
                'statistics': {
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'min_f1': np.min(f1_scores),
                    'max_f1': np.max(f1_scores),
                    'mean_f1_drop': np.mean(f1_drops),
                    'std_f1_drop': np.std(f1_drops),
                    'remaining_channels_count': 22 - dropout_count
                }
            }
            
            print(f"  平均F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            print(f"  平均F1下降: {np.mean(f1_drops):.4f} ± {np.std(f1_drops):.4f}")
        
        return {
            'baseline_f1': baseline_f1,
            'dropout_results': dropout_results
        }

    def aggregate_single_channel_results(self, base_dir: str = ".", thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        channels = []
        for name in os.listdir(base_dir):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p) and name.startswith("temp_channel_"):
                channels.append(p)
        if not channels:
            return {"error": "no temp_channel folders"}
        if thresholds is None:
            sample_dir = channels[0]
            ths = []
            for name in os.listdir(sample_dir):
                if name.startswith("threshold_"):
                    try:
                        ths.append(float(name.split("_")[1]))
                    except:
                        pass
            thresholds = sorted(ths)
        results = {}
        for th in thresholds:
            total = {"tp": 0, "fp": 0, "refTrue": 0, "duration": 0}
            by_subject = {}
            for ch in channels:
                eval_path = os.path.join(ch, f"threshold_{th:.2f}", "evaluation_result.json")
                if not os.path.exists(eval_path):
                    continue
                try:
                    with open(eval_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except:
                    continue
                subj_counts = data.get("by_subject_counts", {})
                for subj, cnts in subj_counts.items():
                    ev = cnts.get("event", {})
                    if subj not in by_subject:
                        by_subject[subj] = {"tp": 0, "fp": 0, "refTrue": 0, "duration": 0}
                    by_subject[subj]["tp"] += ev.get("tp", 0)
                    by_subject[subj]["fp"] += ev.get("fp", 0)
                    by_subject[subj]["refTrue"] += ev.get("refTrue", 0)
                    by_subject[subj]["duration"] += ev.get("duration", 0)
            for subj, stats in by_subject.items():
                total["tp"] += stats["tp"]
                total["fp"] += stats["fp"]
                total["refTrue"] += stats["refTrue"]
                total["duration"] += stats["duration"]
            sens, prec, f1, fp_r = self.evaluator.compute_scores(total["tp"], total["fp"], total["refTrue"], total["duration"])
            subj_metrics = {"sensitivity": [], "precision": [], "f1": [], "fpRate": []}
            for stats in by_subject.values():
                s, p, f, fr = self.evaluator.compute_scores(stats["tp"], stats["fp"], stats["refTrue"], stats["duration"])
                subj_metrics["sensitivity"].append(s)
                subj_metrics["precision"].append(p)
                subj_metrics["f1"].append(f)
                subj_metrics["fpRate"].append(fr)
            subject_avg = {}
            for k, vals in subj_metrics.items():
                subject_avg[k] = float(np.nanmean(vals)) if vals else np.nan
                subject_avg[f"{k}_std"] = float(np.nanstd(vals)) if vals else np.nan
            results[float(th)] = {
                "event_counts_total": total,
                "event_results_total": {"sensitivity": sens, "precision": prec, "f1": f1, "fpRate": fp_r},
                "subject_event_avg": subject_avg,
                "subjects": list(by_subject.keys())
            }
        return {"thresholds": results}


    def run_complete_evaluation(self, gt_path: str, pred_path: str, 
                              output_dir: str = "dino_eeg_results",
                              meta_json_path: Optional[str] = None) -> Dict[str, Any]:
        """运行完整评估流程 - 数据筛选和调用integrated_evaluation"""
        print("=== DINO-EEG 完整评估流程 ===")
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 加载meta数据
        if meta_json_path:
            print("加载meta数据...")
            self.load_meta_json(meta_json_path)
        
        # 加载数据
        print("加载数据...")
        gt_data = self.load_json(gt_path)
        pred_data = self.load_json(pred_path)
        
        print(f"GT数据: {len(gt_data)} 条")
        print(f"预测数据: {len(pred_data)} 条")
        
        # 生成共享的全通道GT，避免重复计算
        print("生成共享的全通道GT...")
        shared_global_gt = self.generate_global_gt(gt_data)
        print(f"全通道GT数据: {len(shared_global_gt)} 条")
        
        # 1. Channel级别评估 - 按通道筛选数据并调用integrated_evaluation
        channel_results = self.evaluate_channel_level_with_shared_gt(shared_global_gt, pred_data, meta_json_path)
        
        # 2. Global级别评估 - 融合多通道数据并调用integrated_evaluation
        global_results = self.evaluate_global_level_with_shared_gt(shared_global_gt, pred_data, meta_json_path)
        
        # 3. 随机通道丢弃实验 - 随机丢弃通道并调用integrated_evaluation
        dropout_results = self.random_channel_dropout_experiment_with_shared_gt(shared_global_gt, pred_data, meta_json_path=meta_json_path)
        
        # 汇总所有结果
        complete_results = {
            'channel_level': channel_results,
            'global_level': global_results,
            'dropout_experiment': dropout_results,
            'metadata': {
                'gt_count': len(gt_data),
                'pred_count': len(pred_data),
                'tcp_channels': self.tcp_channels
            }
        }
        
        # 保存结果
        output_file = output_dir / "dino_eeg_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_file}")
        
        # 生成简要报告
        self.generate_summary_report(complete_results, output_dir)
        
        return complete_results
    
    def generate_summary_report(self, results: Dict[str, Any], output_dir: Path):
        """生成简要报告"""
        report_file = output_dir / "evaluation_summary.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DINO-EEG 测试集评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # Channel级别结果
            f.write("1. Channel级别评估结果\n")
            f.write("-" * 30 + "\n")
            channel_results = results['channel_level']['channel_results']
            f.write(f"评估通道数: {len(channel_results)}\n")
            
            # 各通道详细结果
            f.write("各通道详细结果:\n")
            for channel, result in channel_results.items():
                if 'error' not in result:
                    sample_scoring = result.get('sample_scoring', {})
                    f.write(f"  {channel}: F1={sample_scoring.get('f1', 0):.4f}, P={sample_scoring.get('precision', 0):.4f}, R={sample_scoring.get('recall', 0):.4f}\n")
                else:
                    f.write(f"  {channel}: 评估失败 - {result['error']}\n")
            f.write("\n")
            
            # Global级别结果
            f.write("2. Global级别评估结果（使用全通道GT）\n")
            f.write("-" * 30 + "\n")
            global_level = results['global_level']
            sample_scoring = global_level.get('sample_scoring', {})
            f.write(f"精确率: {sample_scoring.get('precision', 0):.4f}\n")
            f.write(f"召回率: {sample_scoring.get('recall', 0):.4f}\n")
            f.write(f"F1分数: {sample_scoring.get('f1', 0):.4f}\n")
            f.write(f"最佳阈值: {global_level.get('best_threshold', 0):.4f}\n")
            f.write(f"全通道GT数量: {global_level.get('global_gt_count', 0)}\n")
            f.write(f"对齐后预测数量: {global_level.get('aligned_pred_count', 0)}\n\n")
            
            # 随机通道丢弃实验结果
            f.write("3. 随机通道丢弃实验结果\n")
            f.write("-" * 30 + "\n")
            baseline_f1 = results['dropout_experiment']['baseline_f1']
            f.write(f"基线F1分数（完整22通道）: {baseline_f1:.4f}\n\n")
            
            for dropout_count, dropout_result in results['dropout_experiment']['dropout_results'].items():
                stats = dropout_result['statistics']
                f.write(f"丢弃 {dropout_count} 个通道（剩余 {stats['remaining_channels_count']} 个）:\n")
                f.write(f"  平均F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}\n")
                f.write(f"  F1范围: [{stats['min_f1']:.4f}, {stats['max_f1']:.4f}]\n")
                f.write(f"  平均F1下降: {stats['mean_f1_drop']:.4f} ± {stats['std_f1_drop']:.4f}\n\n")
        
        print(f"简要报告已保存到: {report_file}")


def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 创建评估器
    evaluator = DinoEEGEvaluator()
    
    # 运行完整评估
    results = evaluator.run_complete_evaluation(
        gt_path=r"D:\python\dino_eval\TUSZ_gt.json",
        pred_path=r"D:\python\dino_eval\TUSZ_full.json",
        output_dir=r"dino_eeg_tcp_test_results",
        meta_json_path=r"D:\python\dino_eval\TUSZ_tcp_test_annotations_full.json"  # 使用正确的meta文件路径
    )
    
    print("\n=== 评估完成 ===")
    print(f"Global级别F1: {results['global_level'].get('best_f1', 0):.4f}")
    print(f"基线F1（完整22通道）: {results['dropout_experiment']['baseline_f1']:.4f}")


if __name__ == "__main__":
    main()
