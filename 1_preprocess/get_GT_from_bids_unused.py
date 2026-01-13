import os
import json
import csv
from pathlib import Path


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_events_tsv(tsv_path: Path):
    """
    解析单个 BIDS _events.tsv 文件，提取非 bckg 事件的 onset/duration。
    recordingDuration（宽度）必须存在：
    - 从任意行的 'recordingDuration' 列（支持大小写变体）收集值，若缺失则报错；
    - 当存在多值时，取最大值作为文件级宽度。
    返回: (rows, width)
    rows: 仅包含 eventType 不含 'bckg' 的事件，键有 'onset', 'duration'
    width: float
    """
    rows = []
    rec_durations = []

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            # 规范化键值（去除空白）
            row = { (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() }

            # 收集 recordingDuration（不受事件类型过滤影响）
            recd = (
                _to_float(row.get("recordingDuration"))
                or _to_float(row.get("RecordingDuration"))
                or _to_float(row.get("recording_duration"))
            )
            if recd is not None:
                rec_durations.append(recd)

            # 事件类型过滤：排除包含 bckg 的事件
            event_type = (
                row.get("eventType")
                or row.get("EventType")
                or row.get("event_type")
                or ""
            )
            if isinstance(event_type, str) and "bckg" in event_type.lower():
                continue

            # 仅保留有效 onset/duration 的事件行
            onset = _to_float(row.get("onset"))
            duration = _to_float(row.get("duration"))
            if onset is None or duration is None:
                continue

            rows.append({"onset": onset, "duration": duration})

    if not rec_durations:
        raise ValueError(f"{tsv_path} 缺少 recordingDuration 列或有效值")

    width = max(rec_durations)
    return rows, width


def traverse_bids_events(bids_root: Path):
    """
    Traverse BIDS root and collect entries from all *_events.tsv files.
    Each entry: {"tsv_id": <filename>, "onset": <float>, "duration": <float>, "width": <float>}
    """
    entries = []
    for tsv_path in Path(bids_root).rglob("*_events.tsv"):
        rows, width = parse_events_tsv(tsv_path)
        tsv_id = tsv_path.name
        for r in rows:
            entries.append({
                "tsv_id": tsv_id,
                "onset": r["onset"],
                "duration": r["duration"],
                "width": width,
            })
    return entries


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate JSON from BIDS _events.tsv files.")
    parser.add_argument("--bids_root", required=True, help="Path to BIDS root directory (e.g., siena)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    entries = traverse_bids_events(Path(args.bids_root))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()