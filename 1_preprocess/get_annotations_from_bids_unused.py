import os
import csv
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional


def parse_recording_duration_from_tsv(tsv_path: str) -> Optional[float]:
    """
    从 BIDS 的 `_events.tsv` 文件中解析 `recordingDuration`。
    - 优先读取 `recordingDuration` 列的数值（忽略非数字，如 n/a）。
    - 如果该列缺失或无有效数值，则回退为 `max(onset + duration)` 作为近似总时长。

    返回：录制总时长（秒），若无法解析返回 None。
    """
    durations: List[float] = []
    max_end: float = 0.0
    found_header = False

    try:
        with open(tsv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = None
            for row in reader:
                # 跳过空行
                if not row or all(cell.strip() == '' for cell in row):
                    continue

                if headers is None:
                    headers = [h.strip() for h in row]
                    found_header = True
                    continue

                # 规范化行长度与内容
                values = [v.strip() for v in row]
                # 将缺失列填充为空字符串
                if len(values) < len(headers):
                    values += [''] * (len(headers) - len(values))

                # 建立列名 -> 值的映射
                record = {headers[i]: values[i] for i in range(len(headers))}

                # 记录 recordingDuration
                rd = record.get('recordingDuration')
                if rd is not None and rd.lower() != 'n/a':
                    try:
                        durations.append(float(rd))
                    except ValueError:
                        pass

                # 回退：max(onset + duration)
                onset = record.get('onset')
                duration = record.get('duration')
                try:
                    onset_f = float(onset) if onset is not None and onset.lower() != 'n/a' else 0.0
                    duration_f = float(duration) if duration is not None and duration.lower() != 'n/a' else 0.0
                    end = onset_f + duration_f
                    if end > max_end:
                        max_end = end
                except ValueError:
                    # 非数字内容，忽略
                    pass

        if not found_header:
            return None

        if durations:
            # 若文件行内有多个 recordingDuration，取最大值更安全
            return max(durations)
        # 使用回退近似值（若非零）
        return max_end if max_end > 0 else None
    except Exception:
        return None


def build_annotations_from_bids(bids_root: str) -> Dict[str, List[Dict]]:
    """
    遍历 BIDS 根目录下所有 `_events.tsv` 文件，生成适配
    `load_tusz_annotations` 所需的 JSON 数据结构：

    {
      "images": [
        {"file_name": "<base>.jpg", "width": <int>},
        ...
      ]
    }

    其中 `<base>` 为去掉 `_events.tsv` 后缀的文件基名，例如：
    `sub-00_ses-01_task-szMonitoring_run-00`。
    """
    images: Dict[str, int] = {}

    for root, _, files in os.walk(bids_root):
        for fname in files:
            if not fname.endswith('_events.tsv'):
                continue

            tsv_path = os.path.join(root, fname)
            base = fname[:-len('_events.tsv')]  # 去掉后缀
            file_name = f"{base}.jpg"  # 适配 convert_slice_to_original.py 中的转换逻辑

            rd = parse_recording_duration_from_tsv(tsv_path)
            if rd is None:
                # 若无法解析，跳过该文件
                continue

            width_int = int(round(rd))

            # 去重：若重复出现同一 base，保留最大时长（更稳妥）
            prev = images.get(file_name)
            if prev is None or width_int > prev:
                images[file_name] = width_int

    images_list = [{"file_name": k, "width": v} for k, v in sorted(images.items())]
    return {"images": images_list}


def main():
    parser = argparse.ArgumentParser(description="从 BIDS 事件文件生成 TUSZ 风格注释 JSON（仅 images+width）。")
    parser.add_argument("--bids_root", type=str, default=r"d:\python\dino_0917\siena",
                        help="BIDS 数据集根目录，例如 d:\\python\\dino_0917\\siena")
    parser.add_argument("--output", type=str, default=r"d:\python\dino_0917\TUSZ_tcp_test_annotations_full.json",
                        help="输出 JSON 文件路径")

    args = parser.parse_args()

    annotations = build_annotations_from_bids(args.bids_root)

    # 可选：包含一些元信息，load_tusz_annotations 仅使用 images 字段
    output_data = {
        "info": {
            "description": "Generated from BIDS _events.tsv",
            "bids_root": args.bids_root,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "images": annotations["images"],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"已生成注释 JSON：{args.output}")
    print(f"总计 images 数量：{len(output_data['images'])}")


if __name__ == "__main__":
    main()