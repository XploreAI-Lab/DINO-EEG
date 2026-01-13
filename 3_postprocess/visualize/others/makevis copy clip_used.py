import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.signal import resample  # 新增
# ============================ CONFIG ============================ #
h5_dir = r"D:\TUSZ\lbh\eval"  # 你的.h5文件路径
output_dir = "vis_output_with_signal_clip_0804636"
os.makedirs(output_dir, exist_ok=True)
# 你想绘制的特定 signal_id 列表（支持多个）
target_signal_ids = ['aaaaaqek_s010_t003']  # 举例，替换为你要分析的信号名

# 通道顺序
channel_order = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
    ("C4", "T4"), ("T4", "A2"), ("FP1", "F3"), ("F3", "C3"),
    ("C3", "P3"), ("P3", "O1"), ("FP2", "F4"), ("F4", "C4"),
    ("C4", "P4"), ("P4", "O2")
]
channel_names = [f"{a}-{b}" for a, b in channel_order]

# ====================== CSV 读取与筛选 ====================== #
df = pd.read_csv('results_conf_0.50.csv')
df = df[df['refTrue_event'] > 0]
df['score'] = df['tp_event'] / df['refTrue_event']
top_df = df.sort_values(['tp_event', 'score'], ascending=[False, False]).head(177)
selected_ids = top_df['file'].apply(lambda x: '_'.join(x.split('_')[:3]))

# ====================== 长度读取 ====================== #
with open('TUSZ_merged_annotations.json') as f:
    tusz_data = json.load(f)

signal_lengths = {}
for img in tusz_data['images']:
    fname = img['file_name']
    signal_id = '_'.join(fname.split('_')[:3])
    signal_lengths[signal_id] = max(signal_lengths.get(signal_id, 0), img['width'])

# ====================== BBOX 加载 ====================== #
# ====================== BBOX 加载（带 score 筛选） ====================== #
def load_bbox_by_signal(bbox_file):
    with open(bbox_file) as f:
        data = json.load(f)

    result = defaultdict(list)
    is_prediction = os.path.basename(bbox_file) == "TUSZ_yolov5.json"

    for item in data:
        score = item.get('score', 1.0)
        # 对预测框进行 score 筛选
        if is_prediction and score < 0:
            continue

        img_id = item['image_id'].replace('.h5', '')
        signal_id = '_'.join(img_id.split('_')[:3])
        channel = img_id.split('_')[-1]
        bbox = item['bbox']
        result[signal_id].append((channel, bbox, score))
    return result
def load_bbox_by_signal_yolo(bbox_file):
    with open(bbox_file) as f:
        data = json.load(f)

    result = defaultdict(list)
    is_prediction = os.path.basename(bbox_file) == "TUSZ_yolov5.json"

    for item in data:
        score = item.get('score', 1.0)
        # 对预测框进行 score 筛选
        if is_prediction and score < 0:
            continue

        img_id = item['image_name'].replace('.jpg', '')
        signal_id = '_'.join(img_id.split('_')[:3])
        channel = img_id.split('_')[-1]
        bbox = item['bbox']
        result[signal_id].append((channel, bbox, score))
    return result
# 加载 bbox 数据
pred_bboxes = load_bbox_by_signal_yolo('TUSZ_yolov5.json')
gt_bboxes = load_bbox_by_signal('ground_truth.bbox.json')


# ====================== SIGNAL 可视化函数 ====================== #
# ====================== SIGNAL 处理函数 ====================== #
def process_signals(signal_map, target_len):
    """处理信号：去趋势、标准化和重采样"""
    processed = {}
    for ch, signal in signal_map.items():
        # 1. 去趋势 (移除线性趋势)
        detrended = signal - np.linspace(signal[0], signal[-1], len(signal))
        
        # 2. 带限幅的标准化 (避免极端值)
        median = np.median(detrended)
        mad = np.median(np.abs(detrended - median))
        upper_bound = median + 4 * mad
        lower_bound = median - 4 * mad
        clipped = np.clip(detrended, lower_bound, upper_bound)
        
        # 3. 标准化到[-1, 1]范围
        min_val, max_val = np.min(clipped), np.max(clipped)
        if max_val - min_val > 1e-6:
            normalized = 2 * (clipped - min_val) / (max_val - min_val) - 1
        else:
            normalized = np.zeros_like(clipped)
        
        # 4. 重采样到目标长度
        resampled = resample(normalized, target_len)
        processed[ch] = resampled
    
    return processed
def load_signals(signal_id):
    """读取并处理某个信号下的所有通道"""
    signal_map = {}
    target_len = signal_lengths.get(signal_id, None)
    if target_len is None:
        print(f"[跳过] 无图像宽度信息：{signal_id}")
        return None

    # 首先收集所有原始信号
    for ch in channel_names:
        fname = f"{signal_id}_{ch}.h5"
        fpath = os.path.join(h5_dir, fname)
        if not os.path.exists(fpath):
            continue
        with h5py.File(fpath, 'r') as f:
            signal_map[ch] = np.array(f['signal'])[0]
    
    if not signal_map:
        return None
    
    # 统一处理所有信号
    return process_signals(signal_map, target_len)


# ====================== 绘图主函数 ====================== #
# ====================== 修改后的绘图主函数 ====================== #
def plot_signal_windows(signal_id):
    signal_len = signal_lengths.get(signal_id, 3338)
    signal_data = load_signals(signal_id)

    if not signal_data:
        print(f"[跳过] 无可用信号数据：{signal_id}")
        return

    window_sec = 400
    stride_sec = 200
    sample_rate = len(next(iter(signal_data.values()))) / signal_len
    window_size = int(window_sec * sample_rate)
    stride_size = int(stride_sec * sample_rate)
    total_len = len(next(iter(signal_data.values())))

    windows = []
    for start in range(0, total_len, stride_size):
        end = start + window_size
        if end > total_len:
            break
        windows.append((start, end))

    for win_idx, (start, end) in enumerate(windows):
        cropped = []
        for ch in channel_names:
            if ch in signal_data:
                cropped.append(signal_data[ch][start:end])
            else:
                cropped.append(np.zeros(end - start))
        cropped = np.array(cropped)

        fig, ax = plt.subplots(figsize=(20, len(channel_names) * 0.5))
        ax.set_xlim(0, window_size)
        ax.set_ylim(0, len(channel_names))
        ax.set_yticks([i + 0.5 for i in range(len(channel_names))])
        ax.set_yticklabels(channel_names, fontsize=18)
        ax.set_xlabel("Time(s)", fontsize=20)
        ax.set_ylabel("Channels", fontsize=20)

        # 绘制 bbox：ground truth（绿色） & 预测框（红色）
            
        for i, (ch, signal) in enumerate(zip(channel_names, cropped)):
            x = np.arange(window_size)
            y = signal + i + 0.5 # 垂直偏移
            
            # 使用深蓝色细线提高清晰度
            ax.plot(x, y, color='#1f77b4', linewidth=0.8, alpha=0.9)
            
            # 添加通道基线
            ax.axhline(i, color='gray', linestyle='-', linewidth=0.4, alpha=0.3)
        
        # 绘制事件标记
        for bbox_list, color, label in [(gt_bboxes, '#2ca02c', 'Ground Truth'), 
                                    (pred_bboxes, '#d62728', 'Prediction')]:
            for c, bbox, score in bbox_list.get(signal_id, []):
                if c not in channel_names:
                    continue
                ch_idx = channel_names.index(c)
                x, y, w, h = bbox
                
                bbox_start = x
                bbox_end = x + w
                win_start = start
                win_end = end

                inter_start = max(bbox_start, win_start)
                inter_end = min(bbox_end, win_end)

                if inter_start >= inter_end:
                    continue  # 无交集

                rel_x = inter_start - win_start
                rel_w = inter_end - inter_start

                
                if rel_w > 1:  # 忽略太小的框
                    # 使用带透明度的填充
                    ax.add_patch(
                        plt.Rectangle(
                            (rel_x, ch_idx), rel_w, 1,
                            facecolor=color,
                            alpha=0.2,  # 降低填充透明度
                            edgecolor=color,
                            linewidth=1.2,
                            linestyle='-'
                        )
                    )
                    # 添加边界线
                    ax.axvline(rel_x, ymin=ch_idx/(len(channel_names)), 
                            ymax=(ch_idx+1)/(len(channel_names)), 
                            color=color, linewidth=1.0, linestyle='--', alpha=0.7)
                    ax.axvline(rel_x + rel_w, ymin=ch_idx/(len(channel_names)), 
                            ymax=(ch_idx+1)/(len(channel_names)), 
                            color=color, linewidth=1.0, linestyle='--', alpha=0.7)
        
        # 添加网格和学术装饰
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.4)
        ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.2)
        
        # 添加图例
        gt_patch = plt.Rectangle((0,0), 1, 1, fc='#2ca02c', alpha=0.4, edgecolor='#2ca02c')
        pred_patch = plt.Rectangle((0,0), 1, 1, fc='#d62728', alpha=0.4, edgecolor='#d62728')
        ax.legend([gt_patch, pred_patch], ['Ground Truth', 'Prediction'], 
                loc='upper right', fontsize=18, frameon=True, framealpha=0.9)
        
        # 添加分隔线
        for i in range(1, len(channel_names)):
            ax.axhline(i, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
       
        # 优化布局并保存
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(output_dir, f"{signal_id}_win{win_idx}.png"), 
                    bbox_inches='tight', dpi=200)
        plt.close()

# ====================== 修改主循环 ====================== #
print(f"开始绘图（共 {len(selected_ids)} 个信号）")
for sid in tqdm(selected_ids):
    plot_signal_windows(sid)
# # ====================== 修改主循环 ====================== #
# print(f"仅绘制指定的 {len(target_signal_ids)} 个信号")
# for sid in tqdm(target_signal_ids):
#     if sid not in selected_ids.values:
#         # print(f"[跳过] {sid} 不在 top177 列表中")
#         continue
#     plot_signal_windows(sid)
