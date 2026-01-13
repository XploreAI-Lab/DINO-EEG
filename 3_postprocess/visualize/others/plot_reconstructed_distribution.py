import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_reconstructed_distribution(json_path, output_dir):
    print(f"Loading {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return

    # Check if 'all_thresholds' exists
    if 'all_thresholds' not in data:
        print("Error: 'all_thresholds' key not found in JSON.")
        return
        
    all_thresholds_data = data['all_thresholds']
    
    # Extract data: (threshold, total_tp, total_fp)
    extracted_data = []
    for key in all_thresholds_data.keys():
        try:
            thresh = float(key)
        except ValueError:
            continue
            
        entry = all_thresholds_data[key]
        subjects = {}
        
        # Navigate to by_subject_counts
        if 'full_result' in entry and 'by_subject_counts' in entry['full_result']:
            subjects = entry['full_result']['by_subject_counts']
        elif 'by_subject_counts' in entry:
            subjects = entry['by_subject_counts']
        else:
            continue
            
        total_tp = 0
        total_fp = 0
        
        for sub_id, counts in subjects.items():
            if 'sample' in counts:
                total_tp += counts['sample']['tp']
                total_fp += counts['sample']['fp']
        
        extracted_data.append({'thresh': thresh, 'tp': total_tp, 'fp': total_fp})
        
    if not extracted_data:
        print("No valid threshold data found.")
        return

    # Sort by threshold ascending
    extracted_data.sort(key=lambda x: x['thresh'])
    
    thresholds = [x['thresh'] for x in extracted_data]
    tp_sums = [x['tp'] for x in extracted_data]
    fp_sums = [x['fp'] for x in extracted_data]
    
    print(f"Found {len(thresholds)} thresholds from {min(thresholds)} to {max(thresholds)}")
    
    # Reconstruct bin counts
    # Bin i covers interval [thresholds[i], thresholds[i+1])
    # Count in bin i = Count(thresholds[i]) - Count(thresholds[i+1])
    
    tp_bins = []
    fp_bins = []
    bin_centers = []
    bin_widths = []
    
    for i in range(len(thresholds) - 1):
        t_current = thresholds[i]
        t_next = thresholds[i+1]
        
        # Counts are cumulative (>= threshold)
        # So count in [t_current, t_next) is diff
        tp_count = max(0, tp_sums[i] - tp_sums[i+1])
        fp_count = max(0, fp_sums[i] - fp_sums[i+1])
        
        width = t_next - t_current
        center = t_current + width / 2
        
        tp_bins.append(tp_count)
        fp_bins.append(fp_count)
        bin_centers.append(center)
        bin_widths.append(width)
        
    # Handle last bin: [thresholds[-1], 1.0]
    # Assuming remaining counts are all >= last_threshold
    
    # Plotting style mimicking analyze_scores.py
    plt.figure(figsize=(12, 6))
    
    # Plot FP first (usually more numerous, background)
    plt.bar(bin_centers, fp_bins, width=bin_widths, alpha=0.5, label='False Positives', color='red', align='center')
    
    # Plot TP
    plt.bar(bin_centers, tp_bins, width=bin_widths, alpha=0.5, label='True Positives', color='green', align='center')
    
    plt.title('Sample Distribution(Single)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'sample_distribution_single.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    json_path = r'd:\python\dino_eval\evaluation_summary.json'
    output_dir = r'd:\python\dino_eval\score_distributions'
    plot_reconstructed_distribution(json_path, output_dir)

if __name__ == "__main__":
    main()
