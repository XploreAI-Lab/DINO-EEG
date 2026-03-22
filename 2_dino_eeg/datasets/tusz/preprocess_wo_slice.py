from collections import defaultdict
import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import mne
from pyprep.find_noisy_channels import NoisyChannels
from pandas import read_csv  # type: ignore
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.constants import FREQUENCY, INCLUDED_CHANNELS, N_CLASSES_TUSZ

label_mapper = {
    'fnsz': 1,
    'cpsz': 1,
    'spsz': 1,
    'gnsz': 1,
    'tnsz': 3,
    'tcsz': 3,
    'absz': 4,
}


def _deal_row(row, labels):
    label = row['label']
    if label == 'bckg':
        return
    try:
        labels[row['channel']][int(row['start_time'] * FREQUENCY): int(row['stop_time'] * FREQUENCY)] = label_mapper[label]
    except KeyError as exc:
        print(exc)


def main(raw_edf_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    edf_files = []
    for path, _, files in os.walk(raw_edf_dir):
        for name in files:
            if '.edf' in name:
                edf_files.append(os.path.join(path, name))

    montage = mne.channels.make_standard_montage('standard_1020')
    montage.rename_channels({'Fp1': 'FP1', 'Fp2': 'FP2', 'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ'})
    bad_channels_total = 0
    file_ignored = 0

    for idx, edf_fn in enumerate(edf_files):
        print(f'#####################Processing {idx + 1}th edf file, total {len(edf_files)} files.#####################')
        print(f'#####################Current filename: {Path(edf_fn).stem}.#####################')
        csv_file = read_csv(str(Path(edf_fn).with_suffix('.csv')), sep=',', comment='#')
        if csv_file['label'].isin(['seiz', 'mysz']).any():
            file_ignored += 1
            continue

        raw = mne.io.read_raw_edf(edf_fn, preload=True, verbose=False)
        pickChannels(raw, INCLUDED_CHANNELS)
        raw.set_montage(montage)
        if raw.info['sfreq'] != FREQUENCY:
            raw.resample(FREQUENCY)
        raw.notch_filter(60)
        raw.filter(0.1, 70)

        nd = NoisyChannels(raw, do_detrend=True)
        nd.find_bad_by_deviation()
        nd.find_bad_by_nan_flat()
        raw.info['bads'] = nd.get_bads()
        bad_channels_total += len(raw.info['bads'])
        print('Find bad channel(s) of current file:' + str(raw.info['bads']) + f'. Total {bad_channels_total} bad channel(s).')
        raw.interpolate_bads()
        signals, _ = rereference(raw)
        raw.close()

        labels = {cname: np.full_like(signal[0], N_CLASSES_TUSZ, dtype=np.uint8) for cname, signal in signals.items()}
        csv_file.apply(_deal_row, axis=1, labels=labels)

        for cname, signal in signals.items():
            file_path = os.path.join(save_dir, Path(edf_fn).stem + '_' + cname + '.h5')
            with h5py.File(file_path, 'w') as handle:
                handle.create_dataset('signal', data=signal)
                handle.create_dataset('label', data=labels[cname])

    print('Done.')


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
    parser = argparse.ArgumentParser('Get data sliced.')
    parser.add_argument('--raw_edf_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    main(args.raw_edf_dir, args.save_dir)
