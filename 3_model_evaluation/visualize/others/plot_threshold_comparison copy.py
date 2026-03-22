import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the official threshold-comparison plotting script with an explicit working directory.')
    parser.add_argument('--base_path', required=True)
    args = parser.parse_args()

    script_path = Path(__file__).with_name('plot_threshold_comparison.py')
    subprocess.run([sys.executable, str(script_path)], cwd=args.base_path, check=True)
