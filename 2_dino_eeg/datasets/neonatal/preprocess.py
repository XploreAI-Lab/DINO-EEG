import argparse
import os
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.constants import FREQUENCY, INCLUDED_CHANNELS


def main(raw_edf_dir: str, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    edf_files = [file_name for file_name in os.listdir(raw_edf_dir) if file_name.endswith('.edf')]
    annotations = loadmat(os.path.join(raw_edf_dir, 'annotations_2017.mat'))['annotat_new'].squeeze()

    seiz = []
    noseiz = []
    for edf_fn in edf_files:
        annotation_matrix = annotations[int(edf_fn.split('.')[0][3:]) - 1]
        result_and = np.bitwise_and.reduce(annotation_matrix).squeeze()
        result_or = np.bitwise_or.reduce(annotation_matrix).squeeze()
        if np.any(result_and == 1):
            annotation = result_and
            seiz.append(edf_fn.split('.edf')[0] + '\n')
        elif np.all(result_or == 0):
            annotation = result_or
            noseiz.append(edf_fn.split('.edf')[0] + '\n')
        else:
            continue

        raw = mne.io.read_raw_edf(os.path.join(raw_edf_dir, edf_fn), preload=True, verbose=False)
        pickChannels(raw, INCLUDED_CHANNELS)
        if raw.info['sfreq'] != FREQUENCY:
            raw.resample(FREQUENCY, verbose=False)
        raw.notch_filter(60, verbose=False)
        raw.filter(0.1, 70, verbose=False)
        signals, _ = rereference(raw)
        raw.close()

        for cname, signal in signals.items():
            label = np.repeat(annotation, FREQUENCY).astype(np.uint8)
            with h5py.File(os.path.join(save_dir, edf_fn.split('.edf')[0] + '_' + cname + '.h5'), 'w') as handle:
                handle.create_dataset('signal', data=signal)
                handle.create_dataset('label', data=label[: signal.shape[1]])

    with open(os.path.join(save_dir, 'seiz.txt'), 'w', encoding='utf-8') as handle:
        handle.writelines(seiz)
    with open(os.path.join(save_dir, 'noseiz.txt'), 'w', encoding='utf-8') as handle:
        handle.writelines(noseiz)


def pickChannels(raw, ordered_channel_names):
    labels = list(raw.info['ch_names'])
    channel_name_mapper = {}
    for label in labels:
        if 'EEG ' in label:
            channel_name_mapper[label] = label.split('-')[0].split(' ')[1]
    raw.rename_channels(channel_name_mapper)
    raw.pick(list(set(ordered_channel_names) & set(raw.info['ch_names'])))


def rereference(raw):
    signal_montage = {}
    labels_montage = [
        'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3',
        'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    ]
    bipolarPairs = [
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'), ('FP2', 'F8'), ('F8', 'T4'),
        ('T4', 'T6'), ('T6', 'O2'), ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
        ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ]
    for pair in bipolarPairs:
        try:
            signal_montage[pair[0] + '-' + pair[1]] = (raw.get_data(picks=[pair[0]]) - raw.get_data(picks=[pair[1]])).astype(np.float32)
        except ValueError:
            continue
    return signal_montage, labels_montage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_edf_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    main(args.raw_edf_dir, args.save_dir)
