import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import os
import sys

rcParams.update({'font.size': 20})

def plot(files, output_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        df = df.dropna(subset=['sensitivity', 'fpRate'])
        df = df.sort_values('fpRate')
        ax.plot(df['fpRate'], df['sensitivity'], marker='o', linewidth=2, markersize=6, label=name)

    ax.set_xlabel('FP / 24h')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity vs FP Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(output_path)

def main():
    if len(sys.argv) < 3:
        print('Usage: python plot_ses_fp_from_csv.py <csv1> <csv2> [output_png]')
        sys.exit(1)
    files = sys.argv[1:3]
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
    else:
        output_path = os.path.join(os.getcwd(), 'ses_fp_curves.png')
    plot(files, output_path)

if __name__ == '__main__':
    main()
