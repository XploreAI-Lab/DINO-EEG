import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_tp_fp_vs_threshold(json_path, output_dir):
    print(f"Loading {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return

    thresholds = []
    tp_sums = []
    fp_sums = []
    
    # Check if 'all_thresholds' exists
    if 'all_thresholds' not in data:
        print("Error: 'all_thresholds' key not found in JSON.")
        return
        
    all_thresholds_data = data['all_thresholds']
    
    sorted_keys = []
    for key in all_thresholds_data.keys():
        try:
            thresh = float(key)
            sorted_keys.append((thresh, key))
        except ValueError:
            continue
            
    # Sort by threshold value
    sorted_keys.sort(key=lambda x: x[0])
    
    print(f"Found {len(sorted_keys)} threshold entries in 'all_thresholds'.")

    for thresh, key in sorted_keys:
        entry = all_thresholds_data[key]
        subjects = {}
        
        # Navigate to by_subject_counts
        if 'full_result' in entry and 'by_subject_counts' in entry['full_result']:
            subjects = entry['full_result']['by_subject_counts']
        elif 'by_subject_counts' in entry:
            subjects = entry['by_subject_counts']
        else:
            # Skip if structure is unexpected
            continue
            
        total_tp = 0
        total_fp = 0
        
        for sub_id, counts in subjects.items():
            if 'sample' in counts:
                total_tp += counts['sample']['tp']
                total_fp += counts['sample']['fp']
        
        thresholds.append(thresh)
        tp_sums.append(total_tp)
        fp_sums.append(total_fp)

    if not thresholds:
        print("No valid threshold data found.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, tp_sums, label='Total True Positives (TP)', marker='o', color='green')
    plt.plot(thresholds, fp_sums, label='Total False Positives (FP)', marker='x', color='red')
    
    plt.title('Total TP and FP (Sample-level) vs. Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'tp_fp_vs_threshold.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    json_path = r'd:\python\dino_eval\evaluation_summary.json'
    output_dir = r'd:\python\dino_eval\score_distributions'
    plot_tp_fp_vs_threshold(json_path, output_dir)

if __name__ == "__main__":
    main()
