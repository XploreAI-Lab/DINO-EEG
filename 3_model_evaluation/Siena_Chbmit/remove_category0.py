import argparse
import json
import shutil
from pathlib import Path


def remove_category0_inplace(json_path: str):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON content is not a list.")

    filtered = [item for item in data if item.get("category_id") != 0]
    backup_path = path.with_name(path.stem + ".backup.json")
    shutil.copyfile(path, backup_path)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)

    print(f"Backup: {backup_path}")
    print(f"Removed: {len(data) - len(filtered)}")
    print(f"Remaining: {len(filtered)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove category_id=0 entries from a JSON list.")
    parser.add_argument("--json_path", required=True)
    args = parser.parse_args()
    remove_category0_inplace(args.json_path)
