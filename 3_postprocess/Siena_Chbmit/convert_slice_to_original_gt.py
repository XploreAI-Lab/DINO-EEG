import json
import re
from typing import Dict, List, Any
import numpy as np

def load_tusz_annotations(tusz_file_path: str) -> Dict[str, int]:
    """
    从TUSZ_tcp_test_annotations_full.json中提取文件名和width信息
    返回一个字典，键为文件名（去掉.jpg后缀），值为width
    """
    with open(tusz_file_path, 'r', encoding='utf-8') as f:
        tusz_data = json.load(f)
    
    width_mapping = {}
    for image in tusz_data['images']:
        file_name = image['file_name']
        # 去掉.jpg后缀，转换为.h5格式的文件名
        base_name = file_name.replace('.jpg', '.h5')
        width_mapping[base_name] = image['width']
    
    return width_mapping

def parse_slice_filename(slice_filename: str) -> tuple:
    """
    解析切片文件名，提取原始文件名和切片索引
    例如: "aaaaatdt_s004_t012_P4-O2_slice_000.h5" -> ("aaaaatdt_s004_t012_P4-O2.h5", 0)
    """
    # 使用正则表达式匹配切片文件名格式
    pattern = r'(.+)_slice_(\d+)\.h5$'
    match = re.match(pattern, slice_filename)
    
    if match:
        base_name = match.group(1) + '.h5'
        slice_index = int(match.group(2))
        return base_name, slice_index
    else:
        raise ValueError(f"无法解析切片文件名: {slice_filename}")

def convert_slice_coordinates(bbox: List[float], slice_index: int, slice_width: float, original_width: int) -> List[float]:
    """
    将切片坐标转换为原始文件坐标
    
    参数:
    - bbox: [x, y, width, height] 切片中的边界框坐标
    - slice_index: 切片索引 (0, 1, 2, ...)
    - slice_width: 切片宽度 (通常是200)
    - original_width: 原始文件的总宽度 (未使用，保持接口兼容)
    
    返回:
    - 转换后的边界框坐标 [x, y, width, height]
    """
    x, y, width, height = bbox
    
    # 计算切片的起始偏移：切片索引 × 100s (50%重叠)
    slice_start_offset = slice_index * 1800
    
    # 转换坐标：新坐标 = 切片坐标 + 偏移
    new_x = x + slice_start_offset
    new_y = y  # y坐标不变
    new_width = width  # 宽度不变
    new_height = height  # 高度不变
    
    return [new_x, new_y, new_width, new_height]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的IoU (Intersection over Union)
    
    参数:
    - box1, box2: [x, y, width, height] 格式的边界框
    
    返回:
    - IoU值 (0-1之间)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算交集区域
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 并集面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def apply_nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    对检测结果应用非极大值抑制 (NMS)
    
    参数:
    - detections: 检测结果列表，每个元素包含bbox和score
    - iou_threshold: IoU阈值，超过此值的重叠框会被抑制
    
    返回:
    - 经过NMS处理后的检测结果
    """
    if not detections:
        return []
    
    # 按score降序排序
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep = []
    while sorted_detections:
        # 取出得分最高的检测框
        current = sorted_detections.pop(0)
        keep.append(current)
        
        # 计算与剩余框的IoU，移除重叠度过高的框
        remaining = []
        for detection in sorted_detections:
            iou = calculate_iou(current['bbox'], detection['bbox'])
            if iou <= iou_threshold:
                remaining.append(detection)
        
        sorted_detections = remaining
    
    return keep


def merge_event_intervals(detections: List[Dict], merge_tolerance: float = 0.0) -> List[Dict]:
    """
    直接合并同一 image_id 下的事件区间，不使用score。
    合并规则：在 x 轴上相互重叠或相邻（间隔<=merge_tolerance）的事件合并为一个，
    y 轴方向取并集，最终输出为合并后的包围框。
    """
    if not detections:
        return []

    # 按 x 起点排序，忽略 score
    sorted_dets = sorted(detections, key=lambda d: d['bbox'][0])

    merged = []
    cur = None

    for det in sorted_dets:
        x, y, w, h = det['bbox']
        start = float(x)
        end = float(x) + float(w)
        y_min = float(y)
        y_max = float(y) + float(h)

        if cur is None:
            cur = {
                'image_id': det['image_id'],
                'x_min': start,
                'x_max': end,
                'y_min': y_min,
                'y_max': y_max,
                'score': det.get('score')
            }
            continue

        # 与当前区间重叠或相邻则合并（不使用score）
        if start <= cur['x_max'] + merge_tolerance:
            cur['x_max'] = max(cur['x_max'], end)
            cur['y_min'] = min(cur['y_min'], y_min)
            cur['y_max'] = max(cur['y_max'], y_max)
            s = det.get('score')
            if s is not None:
                if cur['score'] is None:
                    cur['score'] = s
                else:
                    try:
                        cur['score'] = max(cur['score'], s)
                    except Exception:
                        cur['score'] = s
        else:
            # 关闭当前区间并输出
            merged.append({
                'image_id': cur['image_id'],
                'bbox': [cur['x_min'], cur['y_min'], cur['x_max'] - cur['x_min'], cur['y_max'] - cur['y_min']],
                **({'score': cur['score']} if cur['score'] is not None else {})
            })
            # 开启新区间
            cur = {
                'image_id': det['image_id'],
                'x_min': start,
                'x_max': end,
                'y_min': y_min,
                'y_max': y_max,
                'score': det.get('score')
            }

    # 输出最后一个
    if cur is not None:
        merged.append({
            'image_id': cur['image_id'],
            'bbox': [cur['x_min'], cur['y_min'], cur['x_max'] - cur['x_min'], cur['y_max'] - cur['y_min']],
            **({'score': cur['score']} if cur['score'] is not None else {})
        })

    return merged


def find_width_from_same_eeg(target_filename: str, width_mapping: Dict[str, int]) -> int:
    """
    从同一段脑电的其他切片文件中查找width信息
    
    参数:
    - target_filename: 目标文件名 (例如: "aaaaaaaq_s006_t000_A1-T3.h5")
    - width_mapping: 文件名到width的映射字典
    
    返回:
    - 找到的width值，如果找不到返回None
    """
    # 提取脑电段标识符 (例如: "aaaaaaaq_s006_t000")
    if '_' in target_filename:
        parts = target_filename.replace('.h5', '').split('_')
        if len(parts) >= 4:
            eeg_prefix = '_'.join(parts[:4])  # 取前3部分作为脑电段标识
            
            # 在width_mapping中查找具有相同前缀的文件
            for filename, width in width_mapping.items():
                if filename.startswith(eeg_prefix) and filename != target_filename:
                    print(f"为 {target_filename} 找到同段脑电文件 {filename} 的width: {width}")
                    return width
    
    return None

def convert_slice_data_to_original(input_file: str, tusz_file: str, output_file: str):
    """
    将包含切片信息的数据转换为原始文件坐标的数据（不获取width，精简输出）
    """
    # 加载切片数据
    print("正在加载切片数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        slice_data = json.load(f)
    print(f"已加载 {len(slice_data)} 条切片数据")
    
    # 转换数据
    converted_data = []
    processed_count = 0
    skipped_count = 0
    
    for item in slice_data:
        try:
            slice_filename = item['image_id']
            original_filename, slice_index = parse_slice_filename(slice_filename)
            
            slice_width = item.get('width', 3600.0) # 默认切片宽度为200
            
            original_bbox = convert_slice_coordinates(
                item['bbox'], 
                slice_index, 
                slice_width, 
                0  # 不使用原始width，传入占位值
            )
            
            converted_item = {
                'image_id': original_filename,
                'bbox': original_bbox,
                'score': item.get('score')  # 兼容GT：不依赖score，仅保留（如存在）
            }
            
            converted_data.append(converted_item)
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"已处理 {processed_count} 条数据...")
                
        except Exception as e:
            print(f"处理数据时出错: {e}")
            print(f"问题数据: {item}")
            skipped_count += 1
            continue
    
    # 对每个image_id的检测结果进行事件区间合并（按时间轴合并）
    print("\n正在对每个image_id进行事件区间合并...")
    image_groups = {}
    for item in converted_data:
        image_id = item['image_id']
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(item)
    
    final_data = []
    merged_removed_count = 0
    
    for image_id, detections in image_groups.items():
        merged_detections = merge_event_intervals(detections, merge_tolerance=0.0)
        final_data.extend(merged_detections)
        
        removed_count = len(detections) - len(merged_detections)
        merged_removed_count += removed_count
        
        if removed_count > 0:
            print(f"  {image_id}: 合并了 {removed_count} 个重叠/相邻事件")
    
    print(f"区间合并完成，总共合并减少 {merged_removed_count} 个事件")
    print(f"最终保留 {len(final_data)} 个事件")
    
    # 保存转换后的数据
    print("正在保存转换后的数据...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成!")
    print(f"成功处理: {processed_count} 条数据")
    print(f"跳过: {skipped_count} 条数据")
    print(f"区间合并减少: {merged_removed_count} 条")
    print(f"最终输出: {len(final_data)} 条数据")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 文件路径
    input_file = r"d:\python\dino_0917\Siena3600_ground_truth.bbox.json"
    tusz_file = r"d:\python\dino_0917\anno_siena.json"
    output_file = r"d:\python\dino_0917\gt_converted_Siena3600.json"
    
    # 执行转换
    convert_slice_data_to_original(input_file, tusz_file, output_file)