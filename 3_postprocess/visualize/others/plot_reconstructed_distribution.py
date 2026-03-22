import json
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def plot_reconstructed_distribution(json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if 'all_thresholds' not in data:
        raise KeyError("'all_thresholds' key not found in JSON")

    extracted_data = []
    for key, entry in data['all_thresholds'].items():
        try:
            thresh = float(key)
        except ValueError:
            continue
        subjects = entry.get('full_result', {}).get('by_subject_counts', entry.get('by_subject_counts', {}))
        total_tp = 0
        total_fp = 0
        for counts in subjects.values():
            if 'sample' in counts:
                total_tp += counts['sample']['tp']
                total_fp += counts['sample']['fp']
        extracted_data.append({'thresh': thresh, 'tp': total_tp, 'fp': total_fp})

    extracted_data.sort(key=lambda x: x['thresh'])
    thresholds = [x['thresh'] for x in extracted_data]
    tp_sums = [x['tp'] for x in extracted_data]
    fp_sums = [x['fp'] for x in extracted_data]
    tp_bins, fp_bins, bin_centers, bin_widths = [], [], [], []
    for i in range(len(thresholds) - 1):
        current_threshold = thresholds[i]
        next_threshold = thresholds[i + 1]
        width = next_threshold - current_threshold
        tp_bins.append(max(0, tp_sums[i] - tp_sums[i + 1]))
        fp_bins.append(max(0, fp_sums[i] - fp_sums[i + 1]))
        bin_centers.append(current_threshold + width / 2)
        bin_widths.append(width)

    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, fp_bins, width=bin_widths, alpha=0.5, label='False Positives', color='red', align='center')
    plt.bar(bin_centers, tp_bins, width=bin_widths, alpha=0.5, label='True Positives', color='green', align='center')
    plt.title('Sample Distribution(Single)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'sample_distribution_single.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot reconstructed TP/FP distribution from evaluation summary.')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    plot_reconstructed_distribution(args.json_path, args.output_dir)


if __name__ == '__main__':
    main()
