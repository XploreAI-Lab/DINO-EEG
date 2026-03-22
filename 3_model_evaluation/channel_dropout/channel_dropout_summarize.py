import os
import shutil
from glob import glob
import json
import re
import math
import argparse


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_nan(x):
    return isinstance(x, float) and (math.isnan(x) or x != x)


def _avg_values(vals):
    for value in vals:
        if _is_nan(value):
            return float("nan")
    return sum(float(value) for value in vals) / len(vals)


def _avg_objects(objs):
    if all(isinstance(obj, dict) for obj in objs):
        keys = objs[0].keys()
        return {key: _avg_objects([obj.get(key) for obj in objs]) for key in keys}
    if all(isinstance(obj, list) for obj in objs):
        return [_avg_objects([obj[i] for obj in objs]) for i in range(len(objs[0]))]
    if all(_is_number(obj) for obj in objs):
        return _avg_values(objs)
    return objs[0]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sorted_group_files(base_dir, cls):
    files = glob(os.path.join(base_dir, f"{cls}_*.json"))
    def _idx(path):
        match = re.search(r"_(\d+)\.json$", os.path.basename(path))
        return int(match.group(1)) if match else 0
    return sorted(files, key=_idx)


def compute_channel_averages(base_dir):
    result = {}
    for cls in [2, 4, 8, 16, 21]:
        paths = _sorted_group_files(base_dir, cls)[:20]
        if not paths:
            continue
        result[str(cls)] = _avg_objects([_load_json(path) for path in paths])
    return result


def _fmt(value):
    if _is_nan(value):
        return "NaN"
    try:
        return f"{float(value):.10g}"
    except Exception:
        return str(value)


def generate_threshold_tables(avg_obj, base_dir):
    blocks = []
    for cls in ["2", "4", "8", "16", "21"]:
        if cls not in avg_obj:
            continue
        thresholds = avg_obj[cls].get("all_thresholds", {})
        items = []
        for key in sorted(thresholds.keys(), key=lambda x: float(x)):
            event_results = thresholds[key].get("full_result", {}).get("event_results", {})
            sensitivity = event_results.get("sensitivity", float("nan"))
            fp_rate = event_results.get("fpRate", float("nan"))
            items.append(f"{_fmt(fp_rate)},{_fmt(sensitivity)}")
        blocks.append("\\pgfplotstableread[col sep=comma]{\nFPRATE,Sensitivity\n" + "\n".join(items) + f"\n}}\\DINOChannel_{cls};\n")
    out_path = os.path.join(base_dir, "threshold_tables.tex")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(blocks))
    print(f"Threshold table written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize channel-dropout experiment outputs.")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir or os.path.join(root_dir, "merged_results")
    os.makedirs(output_dir, exist_ok=True)

    folders = glob(os.path.join(root_dir, "temp_dropout_*_*"))
    print(f"Found {len(folders)} result folders")
    for folder in folders:
        parts = os.path.basename(folder).split("_")
        if len(parts) != 4:
            continue
        channel, idx = parts[2], parts[3]
        src_file = os.path.join(folder, "evaluation_summary.json")
        if not os.path.exists(src_file):
            continue
        dst_file = os.path.join(output_dir, f"{channel}_{idx}.json")
        shutil.copy(src_file, dst_file)

    averages = compute_channel_averages(output_dir)
    avg_output_path = os.path.join(output_dir, "averages.json")
    with open(avg_output_path, "w", encoding="utf-8") as handle:
        json.dump(averages, handle, ensure_ascii=False, indent=2)
    print(f"Averages written to: {avg_output_path}")
    generate_threshold_tables(averages, output_dir)


if __name__ == "__main__":
    main()
