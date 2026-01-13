import json
import shutil
from pathlib import Path


def remove_category0_inplace(json_path: str):
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")

    # 读取原始 JSON
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("文件内容不是 JSON 数组，无法处理。")

    # 过滤掉 category_id == 0 的条目
    filtered = [item for item in data if item.get("category_id") != 0]

    removed_count = len(data) - len(filtered)

    # 备份原文件到同目录
    backup_path = p.with_name(p.stem + ".backup.json")
    shutil.copyfile(p, backup_path)
    print(f"已备份原文件到：{backup_path}")

    # 写回原文件（带缩进，提升可读性）
    with p.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"总条目：{len(data)}；移除 category_id=0 条目：{removed_count}；保留：{len(filtered)}")
    print(f"已写回过滤后的内容到：{p}")


if __name__ == "__main__":
    remove_category0_inplace(r"d:\python\dino_0917\chbmit_ground_truth.bbox (7).json")