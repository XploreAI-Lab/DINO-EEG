#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为预测结果添加width信息的脚本

从anno_siena.json中提取width信息，
并为converted_results_siena.json中的每条预测结果添加对应的width字段。
"""

import json
import os
from typing import Dict, List, Any


def load_width_mapping(annotations_file: str) -> Dict[str, int]:
    """
    从annotations文件中加载file_name到width的映射关系
    
    Args:
        annotations_file: anno_siena.json文件路径
        
    Returns:
        Dict[str, int]: file_name到width的映射字典
    """
    print(f"正在加载width映射信息从: {annotations_file}")
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    width_mapping = {}
    # Siena格式直接包含images数组
    for image_info in data['images']:
        file_name = image_info['file_name']
        width = image_info['width']
        width_mapping[file_name] = width
    
    print(f"成功加载 {len(width_mapping)} 个图像的width信息")
    return width_mapping


def add_width_to_predictions(predictions_file: str, width_mapping: Dict[str, int], output_file: str) -> None:
    """
    为预测结果添加width信息
    
    Args:
        predictions_file: 原始预测结果文件路径
        width_mapping: file_name到width的映射字典
        output_file: 输出文件路径
    """
    print(f"正在处理预测结果文件: {predictions_file}")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # 统计信息
    total_predictions = len(predictions)
    matched_count = 0
    unmatched_count = 0
    unmatched_images = set()
    
    # 为每条预测结果添加width信息
    for prediction in predictions:
        image_id = prediction['image_id']
        
        # 对于Siena数据集，image_id格式为：sub-17_ses-01_task-szMonitoring_run-01_eeg_T6-O2.h5
        # 需要转换为对应的jpg文件名：sub-17_ses-01_task-szMonitoring_run-01.jpg
        # 提取前四部分（去掉_eeg_和通道信息）
        parts = image_id.split('_')
        if len(parts) >= 4:
            # 构建对应的jpg文件名
            jpg_filename = '_'.join(parts[:4]) + '.jpg'
        else:
            jpg_filename = image_id
        
        if jpg_filename in width_mapping:
            prediction['width'] = width_mapping[jpg_filename]
            matched_count += 1
        else:
            unmatched_count += 1
            unmatched_images.add(image_id)
            print(f"警告: 无法找到图像 {image_id} (对应 {jpg_filename}) 的width信息")
    
    # 保存结果
    print(f"正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n处理完成!")
    print(f"总预测数量: {total_predictions}")
    print(f"成功匹配: {matched_count}")
    print(f"未匹配: {unmatched_count}")
    
    if unmatched_images:
        print(f"\n未匹配的图像列表:")
        for img in sorted(unmatched_images):
            print(f"  - {img}")


def main():
    """主函数"""
    # 文件路径
    base_dir = r"d:\python\dino_0917"
    annotations_file = os.path.join(base_dir, "anno_siena.json")
    predictions_file = os.path.join(base_dir, "converted_Siena3600_n.json")
    output_file = os.path.join(base_dir, "converted_Siena3600_with_width.json")
    
    # 检查文件是否存在
    if not os.path.exists(annotations_file):
        print(f"错误: 找不到annotations文件: {annotations_file}")
        return
    
    if not os.path.exists(predictions_file):
        print(f"错误: 找不到predictions文件: {predictions_file}")
        return
    
    try:
        # 加载width映射
        width_mapping = load_width_mapping(annotations_file)
        
        # 为预测结果添加width信息
        add_width_to_predictions(predictions_file, width_mapping, output_file)
        
        print(f"\n✅ 成功完成! 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()