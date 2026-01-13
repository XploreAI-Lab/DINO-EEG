import os
import shutil
from glob import glob
import json
import re
import math

# 根目录（含 temp_dropout_x_y 文件夹）
root_dir = r"D:\python\dino_eval"

# 输出目录
output_dir = os.path.join(root_dir, "merged_results")
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 temp_dropout_*_* 文件夹
pattern = os.path.join(root_dir, "temp_dropout_*_*")
folders = glob(pattern)

print(f"共找到 {len(folders)} 个结果文件夹")

for folder in folders:
    folder_name = os.path.basename(folder)  # e.g. "temp_dropout_2_0"
    parts = folder_name.split("_")          # ["temp","dropout","2","0"]

    if len(parts) != 4:
        print(f"跳过异常目录：{folder}")
        continue

    channel = parts[2]     # 抽样通道数
    idx = parts[3]         # 第几组结果

    src_file = os.path.join(folder, "evaluation_summary.json")
    if not os.path.exists(src_file):
        print(f"未找到 evaluation_summary.json：{src_file}")
        continue

    dst_file = os.path.join(output_dir, f"{channel}_{idx}.json")

    shutil.copy(src_file, dst_file)
    print(f"复制：{src_file} → {dst_file}")

print("\n处理完毕！所有结果已保存到：", output_dir)

def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _is_nan(x):
    return isinstance(x, float) and (math.isnan(x) or x != x)

def _avg_values(vals):
    for v in vals:
        if _is_nan(v):
            return float("nan")
    return sum(float(v) for v in vals) / len(vals)

def _avg_objects(objs):
    if all(isinstance(o, dict) for o in objs):
        keys = objs[0].keys()
        return {k: _avg_objects([o.get(k) for o in objs]) for k in keys}
    if all(isinstance(o, list) for o in objs):
        n = len(objs[0])
        return [_avg_objects([o[i] for o in objs]) for i in range(n)]
    if all(_is_number(o) for o in objs):
        return _avg_values(objs)
    return objs[0]

def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _sorted_group_files(base_dir, cls):
    files = glob(os.path.join(base_dir, f"{cls}_*.json"))
    def _idx(p):
        m = re.search(r"_(\d+)\.json$", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(files, key=_idx)

def compute_channel_averages(base_dir):
    result = {}
    for cls in [2, 4, 8, 16, 21]:
        paths = _sorted_group_files(base_dir, cls)
        paths = paths[:20]
        if not paths:
            continue
        objs = [_load_json(p) for p in paths]
        result[str(cls)] = _avg_objects(objs)
    return result

averages = compute_channel_averages(output_dir)
avg_output_path = os.path.join(output_dir, "averages.json")
with open(avg_output_path, "w", encoding="utf-8") as f:
    json.dump(averages, f, ensure_ascii=False, indent=2)
print(f"均值结果已写入：{avg_output_path}")
def _fmt(v):
    if _is_nan(v):
        return "NaN"
    try:
        return f"{float(v):.10g}"
    except Exception:
        return str(v)

def generate_threshold_tables(avg_obj, base_dir):
    out = []
    for cls in ["2", "4", "8", "16", "21"]:
        if cls not in avg_obj:
            continue
        ch_obj = avg_obj[cls]
        ths = ch_obj.get("all_thresholds", {})
        items = []
        for k in sorted(ths.keys(), key=lambda x: float(x)):
            er = ths[k].get("full_result", {}).get("event_results", {})
            sens = er.get("sensitivity", float("nan"))
            fpr = er.get("fpRate", float("nan"))
            items.append(f"{_fmt(fpr)},{_fmt(sens)}")
        block = "\\pgfplotstableread[col sep=comma]{\nFPRATE,Sensitivity\n" + "\n".join(items) + f"\n}}\\DINOChannel_{cls};\n"
        out.append(block)
    text = "\n".join(out)
    out_path = os.path.join(base_dir, "threshold_tables.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"阈值表已写入：{out_path}")

generate_threshold_tables(averages, output_dir)
