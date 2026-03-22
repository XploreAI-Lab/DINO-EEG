import os
import csv
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional


def parse_recording_duration_from_tsv(tsv_path: str) -> Optional[float]:
    durations: List[float] = []
    max_end: float = 0.0
    found_header = False

    try:
        with open(tsv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            headers = None
            for row in reader:
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if headers is None:
                    headers = [header.strip() for header in row]
                    found_header = True
                    continue

                values = [value.strip() for value in row]
                if len(values) < len(headers):
                    values += [""] * (len(headers) - len(values))
                record = {headers[i]: values[i] for i in range(len(headers))}

                recording_duration = record.get("recordingDuration")
                if recording_duration is not None and recording_duration.lower() != "n/a":
                    try:
                        durations.append(float(recording_duration))
                    except ValueError:
                        pass

                onset = record.get("onset")
                duration = record.get("duration")
                try:
                    onset_value = float(onset) if onset is not None and onset.lower() != "n/a" else 0.0
                    duration_value = float(duration) if duration is not None and duration.lower() != "n/a" else 0.0
                    max_end = max(max_end, onset_value + duration_value)
                except ValueError:
                    pass

        if not found_header:
            return None
        if durations:
            return max(durations)
        return max_end if max_end > 0 else None
    except Exception:
        return None


def build_annotations_from_bids(bids_root: str) -> Dict[str, List[Dict]]:
    images: Dict[str, int] = {}
    for root, _, files in os.walk(bids_root):
        for filename in files:
            if not filename.endswith("_events.tsv"):
                continue
            tsv_path = os.path.join(root, filename)
            base = filename[: -len("_events.tsv")]
            file_name = f"{base}.jpg"
            recording_duration = parse_recording_duration_from_tsv(tsv_path)
            if recording_duration is None:
                continue
            width_int = int(round(recording_duration))
            previous = images.get(file_name)
            if previous is None or width_int > previous:
                images[file_name] = width_int

    images_list = [{"file_name": key, "width": value} for key, value in sorted(images.items())]
    return {"images": images_list}


def main():
    parser = argparse.ArgumentParser(description="Build a TUSZ-style annotations JSON from BIDS *_events.tsv files.")
    parser.add_argument("--bids_root", required=True, help="BIDS root directory.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    annotations = build_annotations_from_bids(args.bids_root)
    output_data = {
        "info": {
            "description": "Generated from BIDS _events.tsv",
            "bids_root": args.bids_root,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "images": annotations["images"],
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, ensure_ascii=False, indent=2)

    print(f"Saved annotations JSON to: {args.output}")
    print(f"Image count: {len(output_data['images'])}")


if __name__ == "__main__":
    main()
