import sys

sys.path.append("..")

import argparse
import os
import h5py
from tqdm import tqdm
import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import minmax_scale

from datasets.constants import FREQUENCY


def main_stft(
    data_dir,
    save_dir,
    time_step_size=1,
    db=False,
    scale=False,
    crop=False,
):
    files = os.listdir(data_dir)
    # 200
    physical_time_step_size = int(FREQUENCY * time_step_size)

    for h5_fn in tqdm(files):
        with h5py.File(os.path.join(data_dir, h5_fn), "r") as f:
            # [1, L=seq_len*freq=30/60*200]
            signal = f["signal"][()]
            # [L,]
            label = f["label"][()]

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
            # # [1, T, F] scale，还可尝试除以最大值以及不scale
            signal = signal / signal_sum
            signal[signal == 0.0] = 1e-8
            # [1, F, T]
            signal = signal.transpose((0, 2, 1))

        with h5py.File(os.path.join(save_dir, h5_fn), "w") as shf:
            shf.create_dataset("signal", data=signal)
            shf.create_dataset("label", data=label)


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
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
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
        default=False,
        action="store_true",
        help="stft是否需要scale",
    )
    parser.add_argument(
        "--crop",
        default=False,
        action="store_true",
        help="stft是否裁剪",
    )
    args = parser.parse_args()

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
