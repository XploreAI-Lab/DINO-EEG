#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片脚本：将stft和bilabel文件按照200s窗口、50%重叠进行切片
处理dev、eval、train三种任务，并生成对应的筛选文件
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

tasks = [
    # "train",
    # "dev", 
    "eval",
]

# NQ与最短K的长度对应关系
NQ_cut = {
    1100: 829,
}

NQ = 1100

def slice_signal_and_label(args_tuple):
    """
    对单个文件进行切片处理（多进程版本）
    
    Args:
        args_tuple: (stft_file, bilabel_file, output_dir, window_size, overlap, do_filter)
    
    Returns:
        切片文件列表
    """
    stft_file, bilabel_file, output_dir, window_size, overlap, do_filter = args_tuple
    """
    对单个文件进行切片处理
    
    Args:
        stft_file: STFT信号文件路径
        bilabel_file: 重新生成的标签文件路径
        output_dir: 输出目录
        window_size: 窗口大小（秒），默认200s
        overlap: 重叠比例，默认0.5（50%）
    
    Returns:
        切片文件列表
    """
    sliced_files = []
    
    try:
        # 读取STFT信号数据
        with h5py.File(stft_file, 'r') as f_stft:
            signal = f_stft['signal'][:]
            stft_label = f_stft['label'][:] if 'label' in f_stft else None
        
        # 读取bilabel标签数据
        with h5py.File(bilabel_file, 'r') as f_bilabel:
            bilabel = f_bilabel['label'][:]
        
        # 获取文件基本信息
        base_name = os.path.splitext(os.path.basename(stft_file))[0]
        
        frames_total = signal.shape[-1]
        label_sample_rate = 200
        stft_fps = 2.0
        
        # 长度过滤（参考filter_and_sort逻辑）
        ls = signal.shape[-1]
        if do_filter and ls < NQ_cut[NQ]:
            return []
        
        # 计算切片参数
        step_size = int(window_size * (1 - overlap))
        step_frames_signal = int(round(step_size * stft_fps))
        step_samples_label = int(round(step_size * label_sample_rate))
        
        slice_idx = 0
        start_signal = 0
        start_label = 0
        
        while start_signal < frames_total and start_label < len(bilabel):
            rem_signal_points = frames_total - start_signal
            rem_label_points = len(bilabel) - start_label
            max_duration_seconds = min(rem_signal_points / stft_fps, rem_label_points / label_sample_rate)
            desired_seconds = min(window_size, max_duration_seconds)
            if desired_seconds <= 0:
                break
            signal_len_points = int(round(desired_seconds * stft_fps))
            label_len_points = int(round(desired_seconds * label_sample_rate))
            end_signal = start_signal + signal_len_points
            end_label = start_label + label_len_points
            signal_slice = signal[:, :, start_signal:end_signal]
            label_slice = bilabel[start_label:end_label]
            
            # 判断是否包含癫痫发作（参考filter_and_sort逻辑：如果所有标签都等于5则为非癫痫，否则为癫痫）
            has_seizure = not (label_slice == 5).all()
            
            # 生成切片文件名
            slice_name = f"{base_name}_slice_{slice_idx:03d}.h5"
            slice_path = os.path.join(output_dir, slice_name)
            
            # 保存切片
            with h5py.File(slice_path, 'w') as f_out:
                f_out.create_dataset('signal', data=signal_slice)
                f_out.create_dataset('label', data=label_slice)
                f_out.attrs['has_seizure'] = has_seizure
                f_out.attrs['start_time'] = start_signal
                f_out.attrs['window_size'] = window_size
                f_out.attrs['slice_index'] = slice_idx
                f_out.attrs['duration_seconds'] = signal_len_points / stft_fps
                f_out.attrs['signal_len_points'] = signal_len_points
                f_out.attrs['label_len_points'] = label_len_points
            
            sliced_files.append((slice_path, has_seizure, signal_len_points / stft_fps, label_len_points))
            
            # 更新起始位置
            start_signal += step_frames_signal
            start_label += step_samples_label
            slice_idx += 1
            
    except Exception as e:
        print(f"处理文件 {stft_file} 时出错: {e}")
        return []
    
    return sliced_files

def handle_task(stft_base_dir, bilabel_base_dir, output_base_dir, task, window_size=200, overlap=0.5, num_processes=None):
    """
    处理单个任务（dev/eval/train）- 多进程版本
    """
    stft_dir = os.path.join(stft_base_dir, task)
    bilabel_dir = os.path.join(bilabel_base_dir, task)
    output_dir = os.path.join(output_base_dir, task)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有STFT文件
    if not os.path.exists(stft_dir):
        print(f"警告: STFT目录不存在: {stft_dir}")
        return [], []
        
    stft_files = [f for f in os.listdir(stft_dir) if f.endswith('.h5')]
    
    if not stft_files:
        print(f"警告: {task} 任务中没有找到h5文件")
        return [], []
    
    print(f"\n处理 {task} 任务: {len(stft_files)} 个文件...")
    
    # 根据任务类型决定是否过滤
    do_filter = (task != "eval")
    
    # 准备多进程参数
    process_args = []
    for stft_filename in stft_files:
        stft_path = os.path.join(stft_dir, stft_filename)
        bilabel_path = os.path.join(bilabel_dir, stft_filename)
        
        # 检查对应的bilabel文件是否存在
        if not os.path.exists(bilabel_path):
            print(f"警告: 找不到对应的bilabel文件: {bilabel_path}")
            continue
            
        process_args.append((stft_path, bilabel_path, output_dir, window_size, overlap, do_filter))
    
    if not process_args:
        print(f"警告: {task} 任务中没有有效的文件对")
        return [], []
    
    # 设置进程数
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(process_args))
    
    print(f"使用 {num_processes} 个进程并行处理...")
    
    # 多进程处理
    all_seiz_files = []
    all_noseiz_files = []
    
    with mp.Pool(processes=num_processes) as pool:
        # 使用进度条显示处理进度
        results = list(tqdm(
            pool.imap(slice_signal_and_label, process_args),
            total=len(process_args),
            desc=f"Processing {task}"
        ))
    
    # 收集结果
    for sliced_files in results:
        for slice_path, has_seizure, duration_seconds, label_len_points in sliced_files:
            slice_name = os.path.basename(slice_path)
            time_length = duration_seconds
            signal_length = label_len_points
            
            # 格式化为与filter_and_sort相同的格式：文件名 时间长度 信号长度
            file_info = f"{slice_name} {time_length} {signal_length}"
            
            if has_seizure:
                all_seiz_files.append(file_info)
            else:
                all_noseiz_files.append(file_info)
    
    return all_seiz_files, all_noseiz_files

def main():
    parser = argparse.ArgumentParser(description='切片STFT和bilabel数据（多进程版本）')
    parser.add_argument('--stft_dir', default='/root/autodl-tmp/Siena_stft', help='STFT根目录')
    parser.add_argument('--bilabel_dir', default='/root/autodl-tmp/Siena_stft', help='bilabel根目录')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/Siena_sliced_200_data', help='输出根目录')
    parser.add_argument('--window_size', type=int, default=200, help='窗口大小（秒）')
    parser.add_argument('--overlap', type=float, default=0.5, help='重叠比例')
    parser.add_argument('--num_processes', type=int, default=None, help='进程数量（默认为CPU核心数）')
    
    args = parser.parse_args()
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始处理所有任务...")
    
    # 处理每个任务
    for task in tasks:
        seiz_files, noseiz_files = handle_task(
            args.stft_dir, args.bilabel_dir, args.output_dir, task,
            args.window_size, args.overlap, args.num_processes
        )
        
        # 生成筛选文件
        if task != "eval":
            # train和dev使用FS前缀
            seiz_file = os.path.join(args.output_dir, f'FS_{task}_NQ{NQ}_seiz.txt')
            noseiz_file = os.path.join(args.output_dir, f'FS_{task}_NQ{NQ}_noseiz.txt')
        else:
            # eval使用S前缀
            seiz_file = os.path.join(args.output_dir, f'S_{task}_NQ{NQ}_seiz.txt')
            noseiz_file = os.path.join(args.output_dir, f'S_{task}_NQ{NQ}_noseiz.txt')
        
        # 写入癫痫发作文件列表
        with open(seiz_file, 'w') as f:
            for file_info in sorted(seiz_files):
                f.write(file_info + '\n')
        
        # 写入非癫痫发作文件列表
        with open(noseiz_file, 'w') as f:
            for file_info in sorted(noseiz_files):
                f.write(file_info + '\n')
        
        print(f"{task} 任务完成: 癫痫切片 {len(seiz_files)} 个, 非癫痫切片 {len(noseiz_files)} 个")
        print(f"筛选文件: {os.path.basename(seiz_file)}, {os.path.basename(noseiz_file)}")
    
    print(f"\n所有任务处理完成!")
    print(f"切片数据保存在: {args.output_dir}")
    print(f"筛选文件保存在: {args.output_dir}")

if __name__ == '__main__':
    # 设置多进程启动方法（Windows兼容）
    mp.set_start_method('spawn', force=True)
    main()
