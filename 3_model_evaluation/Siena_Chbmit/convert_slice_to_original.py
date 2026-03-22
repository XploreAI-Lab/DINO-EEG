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


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    return 0.0 if union_area == 0 else intersection_area / union_area


def apply_nms(detections: List[Dict], iou_threshold: float = 0.0) -> List[Dict]:
    if not detections:
        return []
    sorted_detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    keep = []
    while sorted_detections:
        current = sorted_detections.pop(0)
        keep.append(current)
        sorted_detections = [
            detection for detection in sorted_detections
            if calculate_iou(current["bbox"], detection["bbox"]) <= iou_threshold
        ]
    return keep


def convert_slice_data_to_original(input_file: str, tusz_file: str, output_file: str):
    del tusz_file
    with open(input_file, "r", encoding="utf-8") as handle:
        slice_data = json.load(handle)

    converted_data = []
    skipped_count = 0
    for item in slice_data:
        try:
            original_filename, slice_index = parse_slice_filename(item["image_id"])
            original_bbox = convert_slice_coordinates(item["bbox"], slice_index, item.get("width", 3600), 0)
            converted_data.append({
                "image_id": original_filename,
                "bbox": original_bbox,
                "score": item["score"],
            })
        except Exception:
            skipped_count += 1

    image_groups = {}
    for item in converted_data:
        image_groups.setdefault(item["image_id"], []).append(item)

    final_data = []
    for detections in image_groups.values():
        final_data.extend(apply_nms(detections, iou_threshold=0.0))

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(final_data, handle, indent=2, ensure_ascii=False)

    print(f"Processed: {len(converted_data)}")
    print(f"Skipped: {skipped_count}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert slice-level predictions back to original coordinates.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--tusz_file", default="")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    convert_slice_data_to_original(args.input_file, args.tusz_file, args.output_file)
