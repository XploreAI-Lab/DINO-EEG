import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def calculate_f1(tp, fp, ref_true):
    sensitivity = tp / ref_true if ref_true > 0 else np.nan
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    if np.isnan(sensitivity) or np.isnan(precision):
        return np.nan
    if sensitivity + precision == 0:
        return 0
    return 2 * sensitivity * precision / (sensitivity + precision)


def plot_distribution(data, title, save_path, color='blue'):
    plt.figure(figsize=(10, 6))
    valid_data = [x for x in data if not np.isnan(x)]
    if not valid_data:
        plt.close()
        return
    plt.hist(valid_data, bins=20, alpha=0.7, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('F1 Score')
    plt.ylabel('Count (Subjects)')
    plt.grid(True, alpha=0.3)
    mean_val = np.mean(valid_data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
    ymin, ymax = plt.ylim()
    plt.text(mean_val, ymax * 0.9, ' Mean: {:.2f}'.format(mean_val), color='red')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot sample-level F1 distribution.')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.json_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    subjects = data['best_result']['by_subject_counts']
    sample_f1_scores = []
    for counts in subjects.values():
        if 'sample' in counts:
            sample = counts['sample']
            sample_f1_scores.append(calculate_f1(sample['tp'], sample['fp'], sample['refTrue']))
    plot_distribution(sample_f1_scores, 'Sample-level F1 Distribution', os.path.join(args.output_dir, 'sample_f1_distribution.png'))


if __name__ == '__main__':
    main()
