import json
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_f1(tp, fp, refTrue):
    # Sensitivity
    if refTrue > 0:
        sensitivity = tp / refTrue
    else:
        sensitivity = np.nan

    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan

    # F1 Score
    if np.isnan(sensitivity) or np.isnan(precision):
        f1 = np.nan
    elif (sensitivity + precision) == 0:
        f1 = 0
    else:
        f1 = 2 * sensitivity * precision / (sensitivity + precision)
    
    return f1

def plot_distribution(data, title, save_path, color='blue'):
    plt.figure(figsize=(10, 6))
    valid_data = [x for x in data if not np.isnan(x)]
    
    if len(valid_data) == 0:
        print(f"No valid data for {title}")
        plt.close()
        return
        
    plt.hist(valid_data, bins=20, alpha=0.7, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('F1 Score')
    plt.ylabel('Count (Subjects)')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    if len(valid_data) > 0:
        mean_val = np.mean(valid_data)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
        # Adjust text position to be within plot limits
        ymin, ymax = plt.ylim()
        plt.text(mean_val, ymax*0.9, ' Mean: {:.2f}'.format(mean_val), color='red')

    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    json_path = r'd:\python\dino_eval\evaluation_summary.json'
    output_dir = r'd:\python\dino_eval\score_distributions'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return
        
    if 'best_result' not in data or 'by_subject_counts' not in data['best_result']:
        print("Error: 'best_result' or 'by_subject_counts' not found in JSON.")
        return

    subjects = data['best_result']['by_subject_counts']
    
    sample_f1_scores = []
    
    print(f"Processing {len(subjects)} subjects...")
    
    for sub_id, counts in subjects.items():
        if 'sample' in counts:
            s = counts['sample']
            f1 = calculate_f1(s['tp'], s['fp'], s['refTrue'])
            sample_f1_scores.append(f1)
        else:
            print(f"Warning: No sample counts for {sub_id}")
            
    # Remove NaNs for statistics print
    valid_scores = [x for x in sample_f1_scores if not np.isnan(x)]
    
    print(f"Calculated F1 scores for {len(valid_scores)} subjects.")
    if len(valid_scores) > 0:
        print(f"Mean F1: {np.mean(valid_scores):.4f}")
        print(f"Std F1: {np.std(valid_scores):.4f}")
    
    # Verify against JSON summary
    if 'sample_results' in data['best_result']:
        summary_mean = data['best_result']['sample_results'].get('f1', 0)
        summary_std = data['best_result']['sample_results'].get('f1_std', 0)
        print(f"JSON Summary Mean F1: {summary_mean:.4f}")
        print(f"JSON Summary Std F1: {summary_std:.4f}")
    
    output_path = os.path.join(output_dir, 'sample_f1_distribution.png')
    plot_distribution(sample_f1_scores, 'Sample-level F1 Score Distribution per Subject', output_path, color='purple')

if __name__ == "__main__":
    main()
