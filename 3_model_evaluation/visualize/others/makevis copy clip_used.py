import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
from collections import defaultdict
from scipy.signal import resample


def load_bbox_by_signal(bbox_file):
    with open(bbox_file) as handle:
        data = json.load(handle)
    result = defaultdict(list)
    for item in data:
        img_id = item['image_id'].replace('.h5', '')
        signal_id = '_'.join(img_id.split('_')[:3])
        channel = img_id.split('_')[-1]
        result[signal_id].append((channel, item['bbox'], item.get('score', 1.0)))
    return result


def process_signals(signal_map, target_len):
    processed = {}
    for ch, signal in signal_map.items():
        detrended = signal - np.linspace(signal[0], signal[-1], len(signal))
        median = np.median(detrended)
        mad = np.median(np.abs(detrended - median))
        clipped = np.clip(detrended, median - 4 * mad, median + 4 * mad)
        min_val, max_val = np.min(clipped), np.max(clipped)
        normalized = 2 * (clipped - min_val) / (max_val - min_val) - 1 if max_val - min_val > 1e-6 else np.zeros_like(clipped)
        processed[ch] = resample(normalized, target_len)
    return processed


def main():
    parser = argparse.ArgumentParser(description='Visualize selected EEG signals with bbox overlays.')
    parser.add_argument('--h5_dir', required=True)
    parser.add_argument('--results_csv', required=True)
    parser.add_argument('--annotations_json', required=True)
    parser.add_argument('--pred_json', required=True)
    parser.add_argument('--gt_json', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    with open(args.annotations_json) as handle:
        annotations = json.load(handle)
    pred_bboxes = load_bbox_by_signal(args.pred_json)
    gt_bboxes = load_bbox_by_signal(args.gt_json)
    print(f"Loaded {len(df)} CSV rows, {len(annotations.get('images', []))} annotations, {len(pred_bboxes)} predicted signal groups, {len(gt_bboxes)} GT signal groups")
    print(f"H5 dir: {args.h5_dir}")
    print(f"Output dir: {args.output_dir}")


if __name__ == '__main__':
    main()
