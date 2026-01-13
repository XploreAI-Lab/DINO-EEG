import sys

sys.path.append("..")

import argparse
import os
import h5py
from tqdm import tqdm
import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import minmax_scale
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
import gc
from functools import partial

FREQUENCY = 200

MAX_WORKERS = 10  # 线程数可根据机器核数调整
CHUNK_SIZE = 100  # 批处理大小
USE_PROCESS_POOL = True  # 是否使用进程池而非线程池


def process_single_file(
    h5_fn,
    data_dir,
    save_dir,
    time_step_size=1,
    db=False,
    scale=False,
    crop=False,
):
    """处理单个h5文件的函数 - 优化版本"""
    try:
        # 200
        physical_time_step_size = int(FREQUENCY * time_step_size)
        
        input_path = os.path.join(data_dir, h5_fn)
        output_path = os.path.join(save_dir, h5_fn)
        
        # 检查输出文件是否已存在，避免重复处理
        if os.path.exists(output_path):
            return f"[⚠] Skipped {h5_fn} (already exists)"
        
        # 使用上下文管理器确保文件正确关闭
        with h5py.File(input_path, "r") as f:
            # 使用内存映射读取大文件，减少内存占用
            signal = f["signal"][:]
            label = f["label"][:]

        # fourier_signal[C, F:1+physical_time_step_size//2, T:T//overlap+1] overlap默认为physical_time_step_size//2
        _, _, Zxx = stft(
            signal,
            fs=FREQUENCY,
            nperseg=physical_time_step_size,
            axis=-1,
            scaling="spectrum",
        )

        # [C(1), F, T] 取幅度值 [1, 101, T]
        signal = np.abs(Zxx)
        
        # 释放不再需要的内存
        del Zxx
        gc.collect()

        if crop:
            # [1, T, F]
            signal = signal.transpose((0, 2, 1))
            signal = signal[:, :, :64]
            signal = signal.transpose((0, 2, 1))

        if db:
            # 避免log 0
            signal[signal == 0.0] = 1e-8
            # 取对数，按公式 db=20*lg(abs(S)) 从amp/mag变成db
            # [1, F, T]
            signal = 20 * np.log10(signal)

        if scale:
            # [1, T, F]
            signal = signal.transpose((0, 2, 1))
            # [1, T, 1] 每个时间窗口内，每个数据的能量百分比
            signal_sum = np.sum(signal, axis=-1, keepdims=True)
            # 避免除零错误
            signal_sum[signal_sum == 0.0] = 1e-8
            # [1, T, F] scale
            signal = signal / signal_sum
            signal[signal == 0.0] = 1e-8
            # [1, F, T]
            signal = signal.transpose((0, 2, 1))

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用压缩和分块写入优化IO性能
        with h5py.File(output_path, "w") as shf:
            # 使用gzip压缩和分块存储
            shf.create_dataset(
                "signal", 
                data=signal.astype(np.float32),  # 使用float32减少存储空间
                compression="gzip",
                compression_opts=6,
                chunks=True
            )
            shf.create_dataset(
                "label", 
                data=label,
                compression="gzip",
                compression_opts=6,
                chunks=True
            )
            
        # 手动释放内存
        del signal, label
        gc.collect()
            
        return f"[✓] Processed {h5_fn}"
        
    except Exception as e:
        return f"[✗] Error processing {h5_fn}: {str(e)}"


def process_batch_files(file_batch, data_dir, save_dir, time_step_size, db, scale, crop):
    """批处理多个文件，减少线程创建开销"""
    results = []
    for h5_fn in file_batch:
        result = process_single_file(h5_fn, data_dir, save_dir, time_step_size, db, scale, crop)
        results.append(result)
    return results


def main_stft(
    data_dir,
    save_dir,
    time_step_size=1,
    db=False,
    scale=False,
    crop=False,
):
    files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    print(f"Found {len(files)} h5 files to process")
    print(f"Using {'ProcessPool' if USE_PROCESS_POOL else 'ThreadPool'} with {MAX_WORKERS} workers")
    print(f"Batch size: {CHUNK_SIZE}")
    
    # 检查是否有文件需要处理（跳过已存在的文件）
    files_to_process = []
    for f in files:
        output_path = os.path.join(save_dir, f)
        if not os.path.exists(output_path):
            files_to_process.append(f)
    
    if not files_to_process:
        print("All files already processed!")
        return
        
    print(f"Need to process {len(files_to_process)} files (skipping {len(files) - len(files_to_process)} existing files)")
    
    # 将文件分批处理，减少线程/进程创建开销
    file_batches = [files_to_process[i:i + CHUNK_SIZE] for i in range(0, len(files_to_process), CHUNK_SIZE)]
    
    # 创建部分函数以简化参数传递
    process_func = partial(
        process_batch_files,
        data_dir=data_dir,
        save_dir=save_dir,
        time_step_size=time_step_size,
        db=db,
        scale=scale,
        crop=crop
    )
    
    # 选择使用进程池还是线程池
    if USE_PROCESS_POOL:
        # 进程池适合CPU密集型任务，避免GIL限制
        executor_class = ProcessPoolExecutor
    else:
        # 线程池适合IO密集型任务
        executor_class = ThreadPoolExecutor
    
    # 批量处理文件
    with executor_class(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_func, batch) for batch in file_batches]
        
        # 显示进度
        processed_count = 0
        for future in tqdm(as_completed(futures), total=len(file_batches), desc="Processing batches"):
            batch_results = future.result()
            processed_count += len(batch_results)
            # 可选：显示详细结果
            # for result in batch_results:
            #     print(result)
    
    print(f"\nCompleted processing {processed_count} files!")
    
    # 清理内存
    gc.collect()


def main_dwt(data_dir, save_dir):
    pass


def main_minmax(data_dir, save_dir, time_step_size=1):
    # 先minmax，再stft，再minmax
    files = os.listdir(data_dir)
    # 200
    physical_time_step_size = int(FREQUENCY * time_step_size)

    for h5_fn in tqdm(files):
        with h5py.File(os.path.join(data_dir, h5_fn), "r") as f:
            # [1, L=seq_len*freq=30/60*200]
            signal = f["signal"][()]
            # [L,]
            label = f["label"][()]

        # 第一次minmax归一化
        signal = minmax_scale(signal, axis=1)

        # fourier_signal[C, F:1+physical_time_step_size//2, T:T//overlap+1] overlap默认为physical_time_step_size//2
        _, _, Zxx = stft(
            signal,
            fs=FREQUENCY,
            nperseg=physical_time_step_size,
            axis=-1,
            scaling="spectrum",
        )

        # [C(1), F, T] 取幅度值
        signal = np.abs(Zxx)

        # 第二次minmax归一化 形状不变 [1, F, T]
        signal = minmax_scale(signal.squeeze(), axis=1)[None, ...]

        with h5py.File(os.path.join(save_dir, h5_fn), "w") as shf:
            shf.create_dataset("signal", data=signal)
            shf.create_dataset("label", data=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/TUSZ_avg_stft/train",
        help="输入数据目录路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/TUSZ_avg_stft_processed/train",
        help="数据存储位置。(绝对路径)",
    )
    parser.add_argument(
        "--time_step_size",
        type=int,
        default=1,
        help="快速傅里叶变换步长",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="stft",
        choices=("stft", "dwt", "minmax"),
        help="进行傅里叶变换还是小波变换",
    )
    parser.add_argument(
        "--db",
        default=False,
        action="store_true",
        help="stft是否取log变为db单位",
    )
    parser.add_argument(
        "--scale",
        default=True,
        action="store_true",
        help="stft是否需要scale",
    )
    parser.add_argument(
        "--crop",
        default=True,
        action="store_true",
        help="stft是否裁剪",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=min(10, mp.cpu_count()),
        help="多线程/进程处理的最大worker数",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="批处理大小，减少线程创建开销",
    )
    parser.add_argument(
        "--use_process_pool",
        action="store_true",
        default=True,
        help="使用进程池而非线程池（适合CPU密集型任务）",
    )
    args = parser.parse_args()
    
    # 更新全局设置
    MAX_WORKERS = args.max_workers
    CHUNK_SIZE = args.chunk_size
    USE_PROCESS_POOL = args.use_process_pool
    
    print(f"Configuration:")
    print(f"  - Max workers: {MAX_WORKERS}")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print(f"  - Use process pool: {USE_PROCESS_POOL}")
    print(f"  - CPU count: {mp.cpu_count()}")

    if args.method == "stft":
        main_stft(
            args.data_dir,
            args.save_dir,
            args.time_step_size,
            args.db,
            args.scale,
            args.crop,
        )
    elif args.method == "minmax":
        main_minmax(
            args.data_dir,
            args.save_dir,
            args.time_step_size,
        )
    elif args.method == "dwt":
        main_dwt(args.data_dir, args.save_dir)
    else:
        raise NotImplementedError
