#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估脚本
对多个JSON文件调用evaluate_fine_threshold.py进行评估
将结果保存到统一的result文件夹下的子文件夹中
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# 添加父目录到路径以导入integrated_evaluation
sys.path.append(str(Path(__file__).parent))
from integrated_evaluation import IntegratedEvaluator

def get_model_name_from_file(file_path):
    """
    从文件路径提取模型名称
    
    Args:
        file_path (str): JSON文件路径
    
    Returns:
        str: 模型名称
    """
    file_name = Path(file_path).stem
    
    # 定义文件名到模型名的映射
    name_mapping = {
        'chbmit_results.bbox (4)': 'chbmit'
    }
    
    return name_mapping.get(file_name, file_name)

def evaluate_single_file(json_file_path, result_base_dir):
    """
    评估单个JSON文件
    
    Args:
        json_file_path (str): JSON文件路径
        result_base_dir (Path): 结果基础目录
    
    Returns:
        dict: 评估结果
    """
    current_dir = Path(__file__).parent
    gt_file = current_dir / "chbmit_ground_truth.bbox (7).json"
    pred_file = Path(json_file_path)
    meta_file = current_dir / "anno_chbmit_dedup.json"
    
    # 检查文件是否存在
    if not gt_file.exists():
        print(f"错误: GT文件不存在: {gt_file}")
        return None
        
    if not pred_file.exists():
        print(f"错误: 预测文件不存在: {pred_file}")
        return None
    
    if not meta_file.exists():
        print(f"警告: 元数据文件不存在: {meta_file}")
        print("将使用默认宽度进行评估")
        meta_file = None
    
    # 获取模型名称并设置输出目录
    model_name = get_model_name_from_file(json_file_path)
    output_dir = result_base_dir / model_name
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始评估: {model_name}")
    print(f"{'='*60}")
    print(f"GT文件: {gt_file}")
    print(f"预测文件: {pred_file}")
    print(f"元数据文件: {meta_file if meta_file else '未使用'}")
    print(f"输出目录: {output_dir}")
    
    evaluator = IntegratedEvaluator()
    
    try:
        # 加载数据
        print("\n加载GT数据...")
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        print(f"GT数据加载完成，共 {len(gt_data)} 条")
        
        print("\n加载预测数据...")
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        print(f"预测数据加载完成，共 {len(pred_data)} 条")
        
        # 执行多阈值评估
        print("\n开始多阈值评估...")
        result = evaluator.run_fine_threshold_evaluation_from_data(
            gt_data=gt_data,
            pred_data=pred_data,
            output_dir=str(output_dir),
            threshold_min=0.1,
            threshold_max=0.95,
            threshold_step=0.05,  # 使用0.05步长进行评估
            meta_json_path=str(meta_file) if meta_file else None
            # meta_json_path=None
        )
        
        # 显示结果
        print(f"\n=== {model_name.upper()} 模型评估结果 ===")
        print(f"最佳阈值: {result['best_threshold']:.2f}")
        print(f"最高F1分数: {result['best_f1_score']:.4f}")
        print(f"评估的阈值数量: {result['summary']['thresholds_evaluated']}")
        print(f"GT标注总数: {result['summary']['total_gt_annotations']}")
        print(f"合并前预测数: {result['summary']['total_predictions_before_merge']}")
        print(f"合并后预测数: {result['summary']['total_predictions_after_merge']}")
        
        # 显示前5个最佳阈值
        print("\n=== 前5个最佳阈值 ===")
        sorted_results = sorted(
            [(t, data['event_f1']) for t, data in result['all_thresholds'].items() 
             if not isinstance(data.get('event_f1'), str) and data.get('event_f1', 0) > 0],
            key=lambda x: x[1], reverse=True
        )
        
        for i, (threshold, f1_score) in enumerate(sorted_results[:5]):
            print(f"{i+1}. 阈值 {threshold:.2f}: F1 = {f1_score:.4f}")
        
        print(f"\n✓ {model_name.upper()} 模型评估完成！")
        print(f"✓ 详细结果已保存到: {output_dir}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ {model_name} 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_evaluate(json_files, result_dir="result"):
    """
    批量评估多个JSON文件
    
    Args:
        json_files (list): JSON文件路径列表
        result_dir (str): 结果目录名称
    
    Returns:
        dict: 所有评估结果
    """
    current_dir = Path(__file__).parent
    result_base_dir = current_dir / result_dir
    
    # 创建结果基础目录
    result_base_dir.mkdir(exist_ok=True)
    
    print(f"批量评估开始")
    print(f"结果将保存到: {result_base_dir}")
    print(f"待评估文件数量: {len(json_files)}")
    
    all_results = {}
    successful_evaluations = 0
    failed_evaluations = 0
    
    start_time = datetime.now()
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*80}")
        print(f"进度: {i}/{len(json_files)} - 评估文件: {Path(json_file).name}")
        print(f"{'='*80}")
        
        result = evaluate_single_file(json_file, result_base_dir)
        
        if result:
            model_name = get_model_name_from_file(json_file)
            all_results[model_name] = result
            successful_evaluations += 1
            print(f"✓ {model_name} 评估成功")
        else:
            failed_evaluations += 1
            print(f"❌ {Path(json_file).name} 评估失败")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 生成总结报告
    print(f"\n{'='*80}")
    print(f"批量评估完成")
    print(f"{'='*80}")
    print(f"总耗时: {duration}")
    print(f"成功评估: {successful_evaluations}")
    print(f"失败评估: {failed_evaluations}")
    print(f"总文件数: {len(json_files)}")
    
    # 保存总结报告
    summary_file = result_base_dir / "evaluation_summary.json"
    summary_data = {
        "evaluation_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "successful_evaluations": successful_evaluations,
        "failed_evaluations": failed_evaluations,
        "total_files": len(json_files),
        "evaluated_files": [Path(f).name for f in json_files],
        "results_summary": {}
    }
    
    # 添加每个模型的最佳结果到总结
    for model_name, result in all_results.items():
        summary_data["results_summary"][model_name] = {
            "best_threshold": result["best_threshold"],
            "best_f1_score": result["best_f1_score"],
            "total_gt_annotations": result["summary"]["total_gt_annotations"],
            "total_predictions_after_merge": result["summary"]["total_predictions_after_merge"]
        }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 总结报告已保存到: {summary_file}")
    
    # 显示所有模型的最佳结果对比
    if all_results:
        print(f"\n{'='*80}")
        print(f"所有模型最佳结果对比")
        print(f"{'='*80}")
        print(f"{'模型名称':<20} {'最佳阈值':<10} {'最高F1分数':<12} {'GT总数':<8} {'预测数':<8}")
        print("-" * 80)
        
        # 按F1分数排序
        sorted_models = sorted(
            all_results.items(),
            key=lambda x: x[1]["best_f1_score"],
            reverse=True
        )
        
        for model_name, result in sorted_models:
            print(f"{model_name:<20} {result['best_threshold']:<10.2f} "
                  f"{result['best_f1_score']:<12.4f} "
                  f"{result['summary']['total_gt_annotations']:<8} "
                  f"{result['summary']['total_predictions_after_merge']:<8}")
    
    return all_results

def main():
    """主函数"""
    # 定义要评估的JSON文件列表
    json_files = [
        r"d:\python\dino_0917\chbmit_results.bbox (4).json",
        ]
    
    # 检查文件是否存在
    existing_files = []
    missing_files = []
    
    for file_path in json_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    if missing_files:
        print("警告: 以下文件不存在，将跳过:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print()
    
    if not existing_files:
        print("错误: 没有找到任何可评估的文件")
        return
    
    print(f"将评估以下 {len(existing_files)} 个文件:")
    for file_path in existing_files:
        print(f"  - {Path(file_path).name}")
    
    # 执行批量评估
    results = batch_evaluate(existing_files, "result")
    
    print(f"\n批量评估完成！所有结果已保存到 result 文件夹下的各个子文件夹中。")

if __name__ == "__main__":
    main()