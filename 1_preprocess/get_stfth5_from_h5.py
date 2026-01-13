import os
import h5py
import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

FREQUENCY = 200
MAX_WORKERS = min(10, mp.cpu_count())  # 可根据机器调整
USE_PROCESS_POOL = True  # 使用进程池而非线程池

def process_file_stft(h5_fn, data_dir, save_dir, physical_time_step_size, db=False, scale=False, crop=False):
    try:
        with h5py.File(os.path.join(data_dir, h5_fn), "r") as f:
            signal = f["signal"][()]
            label = f["label"][()]
        signal = np.squeeze(signal)  # 去掉多余维度
        if signal.ndim == 1:
            signal = signal[None, :]  # 添加通道维度 (1, L)
        elif signal.ndim != 2:
            raise ValueError(f"Unsupported signal shape: {signal.shape}")

        _, _, Zxx = stft(signal, fs=FREQUENCY, nperseg=physical_time_step_size, axis=-1, scaling="spectrum")
        signal = np.abs(Zxx)  # [1, F, T]

        if crop:
            signal = signal.transpose((0, 2, 1))
            signal = signal[:, :, :64]
            signal = signal.transpose((0, 2, 1))

        if db:
            signal[signal == 0.0] = 1e-8
            signal = 20 * np.log10(signal)

        if scale:
            signal = signal.transpose((0, 2, 1))
            signal_sum = np.sum(signal, axis=-1, keepdims=True)
            signal = signal / signal_sum
            signal[signal == 0.0] = 1e-8
            signal = signal.transpose((0, 2, 1))

        with h5py.File(os.path.join(save_dir, h5_fn), "w") as shf:
            shf.create_dataset("signal", data=signal)
            shf.create_dataset("label", data=label)

        return f"[✓] {h5_fn}"

    except Exception as e:
        return f"[✗] {h5_fn} error: {str(e)}"


def run_stft_parallel(data_dir, save_dir, time_step_size=1, db=False, scale=True, crop=True):
    os.makedirs(save_dir, exist_ok=True)
    physical_time_step_size = int(FREQUENCY * time_step_size)
    files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    print(f"Found {len(files)} h5 files to process")
    print(f"Using {'ProcessPool' if USE_PROCESS_POOL else 'ThreadPool'} with {MAX_WORKERS} workers")
    
    # 选择使用进程池还是线程池
    executor_class = ProcessPoolExecutor if USE_PROCESS_POOL else ThreadPoolExecutor
    
    with executor_class(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_file_stft, h5_fn, data_dir, save_dir, physical_time_step_size, db, scale, crop)
            for h5_fn in files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            pass 

# STFT 示例
run_stft_parallel(
    data_dir="/root/autodl-tmp/TUSZ_avg_stft/eval",
    save_dir="/root/autodl-tmp/TUSZ_avg_stft_true/eval",
    time_step_size=1,
    db=False,
    scale=True,
    crop=True
)
