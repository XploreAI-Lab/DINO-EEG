import json
import matplotlib.pyplot as plt
import os
import argparse


def plot_tp_fp_vs_threshold(json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if 'all_thresholds' not in data:
        raise KeyError("'all_thresholds' key not found in JSON")

    thresholds = []
    tp_sums = []
    fp_sums = []
    sorted_keys = []
    for key in data['all_thresholds'].keys():
        try:
            sorted_keys.append((float(key), key))
        except ValueError:
            continue
    sorted_keys.sort(key=lambda x: x[0])

    for thresh, key in sorted_keys:
        entry = data['all_thresholds'][key]
        subjects = entry.get('full_result', {}).get('by_subject_counts', entry.get('by_subject_counts', {}))
        total_tp = 0
        total_fp = 0
        for counts in subjects.values():
            if 'sample' in counts:
                total_tp += counts['sample']['tp']
                total_fp += counts['sample']['fp']
        thresholds.append(thresh)
        tp_sums.append(total_tp)
        fp_sums.append(total_fp)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tp_sums, label='Total True Positives (TP)', marker='o', color='green')
    plt.plot(thresholds, fp_sums, label='Total False Positives (FP)', marker='x', color='red')
    plt.title('Total TP and FP (Sample-level) vs. Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'tp_fp_vs_threshold.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot TP/FP versus threshold.')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    plot_tp_fp_vs_threshold(args.json_path, args.output_dir)


if __name__ == '__main__':
    main()
