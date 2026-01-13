import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import pandas as pd
import re

# 设置中文字体，避免中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def get_duration(metadata):
    """从元数据中提取持续时间（处理带单位的情况）"""
    duration_str = metadata.get('duration', '600').strip()
    
    # 尝试提取数字部分
    if ' ' in duration_str:
        parts = duration_str.split()
        if parts:
            try:
                return float(parts[0])
            except ValueError:
                pass
    
    # 尝试直接转换为浮点数
    try:
        return float(duration_str)
    except ValueError:
        return 600.0  # 默认值

def parse_eeg_predictions(file_path):
    """解析EEG预测文件"""
    metadata = {}
    data_rows = []
    is_header_skipped = False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            stripped_line = line.strip()
            
            if not stripped_line or stripped_line in ['<Sheet1>', '</Sheet1>']:
                continue
                
            if stripped_line.startswith('#'):
                if not is_header_skipped and ',' in stripped_line:
                    is_header_skipped = True
                    continue
                
                if '=' in stripped_line:
                    key_value = stripped_line.split('=', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip('# ').strip()
                        value = key_value[1].strip()
                        metadata[key] = value
                continue
            
            if ',' in stripped_line:
                parts = stripped_line.split(',')
                if len(parts) == 5:
                    if parts[0] == 'channel' and parts[1] == 'start_time':
                        continue
                    data_rows.append(parts)
    
    if data_rows:
        columns = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
        df = pd.DataFrame(data_rows, columns=columns)
        df['start_time'] = pd.to_numeric(df['start_time'])
        df['stop_time'] = pd.to_numeric(df['stop_time'])
        df['confidence'] = pd.to_numeric(df['confidence'])
    else:
        df = pd.DataFrame()
    
    return metadata, df

def parse_doctor_annotations(file_path):
    """解析医生标注文件"""
    data_rows = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            stripped_line = line.strip()
            
            if not stripped_line or stripped_line in ['<Sheet1>', '</Sheet1>']:
                continue
                
            if stripped_line.startswith('onset'):
                continue
                
            parts = re.split(r'\s+', stripped_line)
            if len(parts) >= 4:
                data_rows.append({
                    'onset': float(parts[0]),
                    'duration': float(parts[1]),
                    'eventType': parts[2]
                })
    
    return pd.DataFrame(data_rows)

def plot_eeg_signal_with_annotations(h5_file_path, pred_file, true_file, 
                                    channel_name="C4-P4",
                                    start_time=0, end_time=None, 
                                    sampling_rate=200,
                                    seizure_density_factor=1.0, 
                                    normal_density_factor=1.0):
    """
    绘制HDF5文件中的脑电信号，并添加专家标注和预测事件的矩形框
    
    参数:
    h5_file_path: HDF5文件路径
    pred_file: 预测结果CSV文件路径
    true_file: 医生标注CSV文件路径
    channel_name: 要显示的通道名称
    start_time: 开始时间（秒）
    end_time: 结束时间（秒），如果为None则绘制到文件末尾
    sampling_rate: 采样率（Hz），默认200Hz
    seizure_density_factor: 癫痫发作区域的线条密度因子（>1表示更密集）
    normal_density_factor: 正常区域的线条密度因子（<1表示更稀疏）
    """
    
    # 解析预测结果
    try:
        metadata, pred_df = parse_eeg_predictions(pred_file)
        print(f"成功解析预测文件: {pred_file}")
        print(f"持续时间: {get_duration(metadata)} 秒")
        print(f"找到 {len(pred_df)} 条预测记录")
        
        # 筛选指定通道的预测数据
        channel_pred = pred_df[pred_df['channel'] == channel_name]
        predicted_seizure_periods = []
        for _, row in channel_pred.iterrows():
            predicted_seizure_periods.append((row['start_time'], row['stop_time']))
    except Exception as e:
        print(f"解析预测文件错误: {e}")
        predicted_seizure_periods = []
    
    # 解析医生标注
    try:
        true_df = parse_doctor_annotations(true_file)
        print(f"\n成功解析医生标注文件: {true_file}")
        print(f"找到 {len(true_df)} 条医生标注")
        
        expert_seizure_periods = []
        for _, row in true_df.iterrows():
            expert_seizure_periods.append((row['onset'], row['onset'] + row['duration']))
    except Exception as e:
        print(f"解析医生标注文件错误: {e}")
        expert_seizure_periods = []
    
    # 读取HDF5文件
    with h5py.File(h5_file_path, "r") as hf:
        signal = hf["signal"][()]
    
    # 如果信号是二维的，取第一个通道
    if signal.ndim == 2:
        signal = signal[0]  # 取第一个通道
    
    print(f"\n原始信号形状: {signal.shape}")
    print(f"信号总长度: {len(signal)} 个采样点，约 {len(signal)/sampling_rate:.1f} 秒")
    
    # 计算时间索引
    start_idx = int(start_time * sampling_rate)
    if end_time is not None:
        end_idx = int(end_time * sampling_rate)
    else:
        # 获取持续时间
        duration = get_duration(metadata) if 'metadata' in locals() else len(signal)/sampling_rate
        end_idx = int(duration * sampling_rate)
    
    # 确保索引在有效范围内
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    
    # 提取指定时间段的信号
    signal_segment = signal[start_idx:end_idx]
    
    # 创建时间轴（以秒为单位）
    time_axis = np.arange(len(signal_segment)) / sampling_rate + start_time
    
    # ====================== 创建图形 ====================== #
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # 设置标题和标签
    duration = (end_idx - start_idx) / sampling_rate
    ax.set_title(f"脑电信号波形 ({start_time:.1f}-{start_time+duration:.1f}秒)", 
                fontsize=16, pad=20)
    ax.set_xlabel("时间 (秒)", fontsize=14)
    ax.set_ylabel("信号幅值 (μV)", fontsize=14)
    
    # 设置X轴范围
    ax.set_xlim(start_time, start_time + duration)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 计算信号的动态范围用于Y轴缩放
    min_signal = np.min(signal_segment)
    max_signal = np.max(signal_segment)
    signal_range = max_signal - min_signal
    y_margin = signal_range * 0.1  # 10%的边距
    ax.set_ylim(min_signal - y_margin, max_signal + y_margin)
    
    # 计算信号的中心线
    center_y = (min_signal + max_signal) / 2
    
    # 合并所有癫痫发作区域（专家标注和预测）
    all_seizure_periods = []
    if expert_seizure_periods:
        all_seizure_periods.extend(expert_seizure_periods)
    if predicted_seizure_periods:
        all_seizure_periods.extend(predicted_seizure_periods)
    
    # 如果没有提供任何癫痫发作时间段，使用默认密度绘制
    if not all_seizure_periods:
        # 绘制信号 - 使用默认密度
        ax.plot(time_axis, signal_segment, color='#1e88e5', linewidth=0.8, alpha=0.9)
    else:
        # 分别绘制癫痫发作区域和非发作区域
        # 首先绘制非发作区域（稀疏）
        non_seizure_mask = np.ones(len(time_axis), dtype=bool)
        
        for seizure_start, seizure_end in all_seizure_periods:
            # 转换为索引
            seizure_start_idx = int((seizure_start - start_time) * sampling_rate)
            seizure_end_idx = int((seizure_end - start_time) * sampling_rate)
            
            # 确保索引在有效范围内
            seizure_start_idx = max(0, seizure_start_idx)
            seizure_end_idx = min(len(time_axis), seizure_end_idx)
            
            # 标记癫痫发作区域
            non_seizure_mask[seizure_start_idx:seizure_end_idx] = False
        
        # 对非发作区域进行降采样（稀疏化）
        non_seizure_indices = np.where(non_seizure_mask)[0]
        if len(non_seizure_indices) > 0:
            # 计算降采样步长
            downsampling_step = max(1, int(1 / normal_density_factor))
            non_seizure_indices_downsampled = non_seizure_indices[::downsampling_step]
            
            # 绘制非发作区域
            ax.plot(time_axis[non_seizure_indices_downsampled], 
                   signal_segment[non_seizure_indices_downsampled], 
                   color='#1e88e5', linewidth=0.8, alpha=0.9)
        
        # 绘制癫痫发作区域（密集）
        for seizure_start, seizure_end in all_seizure_periods:
            # 转换为索引
            seizure_start_idx = int((seizure_start - start_time) * sampling_rate)
            seizure_end_idx = int((seizure_end - start_time) * sampling_rate)
            
            # 确保索引在有效范围内
            seizure_start_idx = max(0, seizure_start_idx)
            seizure_end_idx = min(len(time_axis), seizure_end_idx)
            
            if seizure_start_idx < seizure_end_idx:
                # 对癫痫发作区域进行上采样（密集化）
                seizure_indices = np.arange(seizure_start_idx, seizure_end_idx)
                if len(seizure_indices) > 0:
                    # 计算上采样倍数
                    upsampling_factor = max(1, int(seizure_density_factor))
                    
                    # 如果上采样因子为1，直接绘制
                    if upsampling_factor == 1:
                        ax.plot(time_axis[seizure_indices], 
                               signal_segment[seizure_indices], 
                               color='#1e88e5', linewidth=1.2, alpha=0.9)
                    else:
                        # 创建更密集的时间轴和信号
                        dense_time = np.linspace(time_axis[seizure_start_idx], 
                                                time_axis[seizure_end_idx-1], 
                                                len(seizure_indices) * upsampling_factor)
                        
                        # 使用插值获取更密集的信号
                        from scipy import interpolate
                        interp_func = interpolate.interp1d(
                            time_axis[seizure_indices], 
                            signal_segment[seizure_indices], 
                            kind='linear'
                        )
                        dense_signal = interp_func(dense_time)
                        
                        # 绘制密集化的癫痫发作区域
                        ax.plot(dense_time, dense_signal, color='#1e88e5', linewidth=1.2, alpha=0.9)
    
    # 添加专家标注的癫痫发作区域（红色边框矩形框）
    if expert_seizure_periods:
        for seizure_start, seizure_end in expert_seizure_periods:
            # 计算矩形的高度和位置（真实框高度为信号范围的30%）
            rect_height_expert = signal_range * 0.9
            rect_bottom_expert = center_y - rect_height_expert / 2
            
            # 添加透明矩形框（只有边框）
            rect = Rectangle(
                (seizure_start, rect_bottom_expert), 
                seizure_end - seizure_start, 
                rect_height_expert,
                linewidth=2,  # 边框线宽
                edgecolor='red',  # 边框颜色
                facecolor='none',  # 透明填充
                alpha=0.8,  # 边框透明度
                zorder=3  # 关键修改：确保矩形在信号线上方
            )
            ax.add_patch(rect)

    # 添加预测的癫痫发作区域（黄色边框矩形框）
    if predicted_seizure_periods:
        for seizure_start, seizure_end in predicted_seizure_periods:
            # 计算矩形的高度和位置（预测框高度为信号范围的50%）
            rect_height_pred = signal_range
            rect_bottom_pred = center_y - rect_height_pred / 2
            
            # 检查是否有包含关系
            for exp_start, exp_end in expert_seizure_periods:
                # 如果预测框包含真实框或真实框包含预测框
                if (seizure_start <= exp_start and seizure_end >= exp_end) or \
                   (exp_start <= seizure_start and exp_end >= seizure_end):
                    # 调整位置使中心对齐
                    rect_bottom_pred = rect_bottom_expert + (rect_height_expert - rect_height_pred) / 2
                    break  # 找到第一个包含关系即可
            
            # 添加透明矩形框（只有边框）
            rect = Rectangle(
                (seizure_start, rect_bottom_pred), 
                seizure_end - seizure_start, 
                rect_height_pred,
                linewidth=2,  # 边框线宽
                edgecolor='goldenrod',  # 边框颜色
                facecolor='none',  # 透明填充
                alpha=0.8,  # 边框透明度
                zorder=3  # 关键修改：确保矩形在信号线上方
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()
    
    print(f"信号长度: {len(signal_segment)} 个采样点")
    print(f"时间范围: {start_time:.1f} - {start_time + len(signal_segment)/sampling_rate:.1f} 秒")
    print(f"采样率: {sampling_rate} Hz")

if __name__ == "__main__":
    # 文件路径
    h5_file_path = "/local2/undergraduate_home/fanxiaoya/dc_eeg/dino/dinolbh-1/singal/aaaaajru_s031_t003_C4-P4.h5"
    pred_file = "aaaaajru_s031_t003.csv"  # 预测结果CSV文件
    true_file = "doctor_annotations.csv"  # 医生标注CSV文件
    
    # 绘制脑电信号并添加标注
    plot_eeg_signal_with_annotations(
        h5_file_path=h5_file_path,
        pred_file=pred_file,
        true_file=true_file,
        channel_name="C4-P4",
        start_time=0,
        end_time=643,
        sampling_rate=200,
        seizure_density_factor=1,  # 癫痫区域密度因子
        normal_density_factor=0.05  # 正常区域密度因子
    )