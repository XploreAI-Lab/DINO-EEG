import argparse
import json
import re
from typing import Dict, List


def parse_slice_filename(slice_filename: str) -> tuple[str, int]:
    match = re.match(r"(.+)_slice_(\d+)\.h5$", slice_filename)
    if not match:
        raise ValueError(f"Cannot parse slice filename: {slice_filename}")
    return match.group(1) + ".h5", int(match.group(2))


def convert_slice_coordinates(bbox: List[float], slice_index: int, slice_width: float, original_width: int) -> List[float]:
    x, y, width, height = bbox
    del slice_width, original_width
    return [x + slice_index * 1800, y, width, height]


def merge_event_intervals(detections: List[Dict], merge_tolerance: float = 0.0) -> List[Dict]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["bbox"][0])
    merged = []
    current = None
    for det in sorted_dets:
        x, y, width, height = det["bbox"]
        start = float(x)
        end = float(x) + float(width)
        y_min = float(y)
        y_max = float(y) + float(height)
        if current is None:
            current = {
                "image_id": det["image_id"],
                "x_min": start,
                "x_max": end,
                "y_min": y_min,
                "y_max": y_max,
                "score": det.get("score"),
            }
            continue
        if start <= current["x_max"] + merge_tolerance:
            current["x_max"] = max(current["x_max"], end)
            current["y_min"] = min(current["y_min"], y_min)
            current["y_max"] = max(current["y_max"], y_max)
        else:
            merged.append({
                "image_id": current["image_id"],
                "bbox": [current["x_min"], current["y_min"], current["x_max"] - current["x_min"], current["y_max"] - current["y_min"]],
                **({"score": current["score"]} if current["score"] is not None else {}),
            })
            current = {
                "image_id": det["image_id"],
                "x_min": start,
                "x_max": end,
                "y_min": y_min,
                "y_max": y_max,
                "score": det.get("score"),
            }
    if current is not None:
        merged.append({
            "image_id": current["image_id"],
            "bbox": [current["x_min"], current["y_min"], current["x_max"] - current["x_min"], current["y_max"] - current["y_min"]],
            **({"score": current["score"]} if current["score"] is not None else {}),
        })
    return merged


def convert_slice_data_to_original(input_file: str, tusz_file: str, output_file: str):
    del tusz_file
    with open(input_file, "r", encoding="utf-8") as handle:
        slice_data = json.load(handle)

    converted_data = []
    skipped_count = 0
    for item in slice_data:
        try:
            original_filename, slice_index = parse_slice_filename(item["image_id"])
            original_bbox = convert_slice_coordinates(item["bbox"], slice_index, item.get("width", 3600.0), 0)
            converted_data.append({
                "image_id": original_filename,
                "bbox": original_bbox,
                "score": item.get("score"),
            })
        except Exception:
            skipped_count += 1

    image_groups = {}
    for item in converted_data:
        image_groups.setdefault(item["image_id"], []).append(item)

    final_data = []
    for detections in image_groups.values():
        final_data.extend(merge_event_intervals(detections, merge_tolerance=0.0))

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(final_data, handle, indent=2, ensure_ascii=False)

    print(f"Processed: {len(converted_data)}")
    print(f"Skipped: {skipped_count}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert slice-level ground truth back to original coordinates.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--tusz_file", default="")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    convert_slice_data_to_original(args.input_file, args.tusz_file, args.output_file)
