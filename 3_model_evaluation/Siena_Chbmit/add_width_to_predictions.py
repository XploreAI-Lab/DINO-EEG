#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict


def load_width_mapping(annotations_file: str) -> Dict[str, int]:
    print(f"Loading width mapping from: {annotations_file}")
    with open(annotations_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    width_mapping = {}
    for image_info in data["images"]:
        width_mapping[image_info["file_name"]] = image_info["width"]
    print(f"Loaded width info for {len(width_mapping)} images")
    return width_mapping


def add_width_to_predictions(predictions_file: str, width_mapping: Dict[str, int], output_file: str) -> None:
    print(f"Processing predictions file: {predictions_file}")
    with open(predictions_file, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    unmatched_images = set()
    matched_count = 0
    for prediction in predictions:
        image_id = prediction["image_id"]
        parts = image_id.split("_")
        jpg_filename = "_".join(parts[:4]) + ".jpg" if len(parts) >= 4 else image_id
        if jpg_filename in width_mapping:
            prediction["width"] = width_mapping[jpg_filename]
            matched_count += 1
        else:
            unmatched_images.add(image_id)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)

    print(f"Saved updated predictions to: {output_file}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {len(unmatched_images)}")
    for image_id in sorted(unmatched_images):
        print(f"Unmatched image: {image_id}")


def main():
    parser = argparse.ArgumentParser(description="Add width metadata to prediction JSON.")
    parser.add_argument("--annotations_file", required=True)
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.annotations_file):
        raise FileNotFoundError(args.annotations_file)
    if not os.path.exists(args.predictions_file):
        raise FileNotFoundError(args.predictions_file)

    width_mapping = load_width_mapping(args.annotations_file)
    add_width_to_predictions(args.predictions_file, width_mapping, args.output_file)


if __name__ == "__main__":
    main()
