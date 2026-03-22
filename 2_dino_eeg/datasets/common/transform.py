import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.constants import FREQUENCY


def main_stft(data_dir, save_dir, time_step_size=1, db=False, scale=False, crop=False):
    os.makedirs(save_dir, exist_ok=True)
    files = os.listdir(data_dir)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    for h5_fn in tqdm(files):
        with h5py.File(os.path.join(data_dir, h5_fn), 'r') as handle:
            signal = handle['signal'][()]
            label = handle['label'][()]

        _, _, zxx = stft(signal, fs=FREQUENCY, nperseg=physical_time_step_size, axis=-1, scaling='spectrum')
        signal = np.abs(zxx)

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
            signal_sum[signal_sum == 0.0] = 1.0
            signal = signal / signal_sum
            signal[signal == 0.0] = 1e-8
            signal = signal.transpose((0, 2, 1))

        with h5py.File(os.path.join(save_dir, h5_fn), 'w') as handle:
            handle.create_dataset('signal', data=signal)
            handle.create_dataset('label', data=label)


def main_dwt(data_dir, save_dir):
    del data_dir, save_dir
    raise NotImplementedError


def main_minmax(data_dir, save_dir, time_step_size=1):
    os.makedirs(save_dir, exist_ok=True)
    files = os.listdir(data_dir)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    for h5_fn in tqdm(files):
        with h5py.File(os.path.join(data_dir, h5_fn), 'r') as handle:
            signal = handle['signal'][()]
            label = handle['label'][()]

        signal = minmax_scale(signal, axis=1)
        _, _, zxx = stft(signal, fs=FREQUENCY, nperseg=physical_time_step_size, axis=-1, scaling='spectrum')
        signal = np.abs(zxx)
        signal = minmax_scale(signal.squeeze(), axis=1)[None, ...]

        with h5py.File(os.path.join(save_dir, h5_fn), 'w') as handle:
            handle.create_dataset('signal', data=signal)
            handle.create_dataset('label', data=label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--time_step_size', type=int, default=1)
    parser.add_argument('--method', type=str, default='stft', choices=('stft', 'dwt', 'minmax'))
    parser.add_argument('--db', default=False, action='store_true')
    parser.add_argument('--scale', default=False, action='store_true')
    parser.add_argument('--crop', default=False, action='store_true')
    args = parser.parse_args()

    if args.method == 'stft':
        main_stft(args.data_dir, args.save_dir, args.time_step_size, args.db, args.scale, args.crop)
    elif args.method == 'minmax':
        main_minmax(args.data_dir, args.save_dir, args.time_step_size)
    elif args.method == 'dwt':
        main_dwt(args.data_dir, args.save_dir)
