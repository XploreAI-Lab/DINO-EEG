import h5py
import json
import os
from pathlib import Path

def process_h5_files():
    """
    读取dev文件夹中所有h5文件，生成TUSZ_dev_annotations_full.json
    """
    dev_folder = Path("Siena_stft_3label")
    annotations = []
    
    # 获取所有h5文件
    h5_files = list(dev_folder.glob("*.h5"))
    h5_files.sort()  # 按文件名排序
    
    print(f"找到 {len(h5_files)} 个h5文件")
    
    for h5_file in h5_files:
        try:
            # 读取h5文件
            with h5py.File(h5_file, 'r') as f:
                # 假设label数据在某个键下，需要找到正确的键
                # 先查看文件结构
                print(f"处理文件: {h5_file.name}")
                
                # 查看h5文件的键
                keys = list(f.keys())
                print(f"  文件键: {keys}")
                
                # 尝试找到label数据
                label_length = 0
                if 'label' in f:
                    label_data = f['label']
                    label_length = len(label_data)
                elif 'labels' in f:
                    label_data = f['labels']
                    label_length = len(label_data)
                else:
                    # 如果没有找到label键，查看所有数据集的形状
                    for key in keys:
                        try:
                            data = f[key]
                            if hasattr(data, 'shape'):
                                print(f"    {key}: shape = {data.shape}")
                                # 假设最长的数据集是我们需要的
                                if len(data.shape) > 0 and data.shape[0] > label_length:
                                    label_length = data.shape[0]
                        except Exception as e:
                            print(f"    无法读取 {key}: {e}")
                
                # 计算width (label长度除以200)
                width = label_length // 200
                
                # 生成jpg文件名 (将.h5替换为.jpg)
                jpg_name = h5_file.name.replace('.h5', '.jpg')
                
                # 添加到注释列表
                annotation = {
                    "h5_name": h5_file.name,
                    "jpg_name": jpg_name,
                    "width": width
                }
                annotations.append(annotation)
                
                print(f"  Label长度: {label_length}, Width: {width}")
                
        except Exception as e:
            print(f"处理文件 {h5_file} 时出错: {e}")
            continue
    
    # 保存为JSON文件
    output_file = "TUSZ_dev_annotations_full.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n已生成 {output_file}，包含 {len(annotations)} 个条目")
    return annotations

if __name__ == "__main__":
    annotations = process_h5_files()
    
    # 显示前几个条目作为示例
    print("\n前5个条目:")
    for i, annotation in enumerate(annotations[:5]):
        print(f"{i+1}. {annotation}")