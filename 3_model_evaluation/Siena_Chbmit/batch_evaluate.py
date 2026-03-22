from pathlib import Path
import argparse
from integrated_evaluation import batch_evaluate


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate multiple JSON prediction files.')
    parser.add_argument('--json_files', nargs='+', required=True)
    parser.add_argument('--output_root', default='result')
    args = parser.parse_args()

    existing_files = []
    missing_files = []
    for file_path in args.json_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)

    if missing_files:
        print('Missing files:')
        for file_path in missing_files:
            print(f'  - {file_path}')

    if not existing_files:
        raise FileNotFoundError('No valid JSON files found for evaluation.')

    batch_evaluate(existing_files, args.output_root)
    print(f'Batch evaluation complete. Outputs saved under: {args.output_root}')


if __name__ == '__main__':
    main()
