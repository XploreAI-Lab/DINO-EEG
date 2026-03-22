import argparse
import os
import random
import numpy as np
from pathlib import Path

sys_path = str(Path(__file__).parent)
import sys
if sys_path not in sys.path:
    sys.path.append(sys_path)

from channel_dropout import DinoEEGEvaluator


def main():
    parser = argparse.ArgumentParser(description='Run channel-dropout evaluation.')
    parser.add_argument('--gt_path', required=True)
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--output_dir', default='dino_eeg_tcp_test_results')
    parser.add_argument('--meta_json_path', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluator = DinoEEGEvaluator()
    results = evaluator.run_complete_evaluation(
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        output_dir=args.output_dir,
        meta_json_path=args.meta_json_path,
    )
    print('\n=== Evaluation Complete ===')
    print(f"Global F1: {results['global_level'].get('best_f1', 0):.4f}")
    print(f"Baseline F1: {results['dropout_experiment']['baseline_f1']:.4f}")


if __name__ == '__main__':
    main()
