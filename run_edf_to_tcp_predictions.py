#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

_MISSING_IMPORTS = {}

try:
    import h5py
except ModuleNotFoundError as exc:
    h5py = None
    _MISSING_IMPORTS["h5py"] = exc

try:
    import mne
except ModuleNotFoundError as exc:
    mne = None
    _MISSING_IMPORTS["mne"] = exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    np = None
    _MISSING_IMPORTS["numpy"] = exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    pd = None
    _MISSING_IMPORTS["pandas"] = exc

try:
    from scipy.signal import stft
except ModuleNotFoundError as exc:
    stft = None
    _MISSING_IMPORTS["scipy"] = exc

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    _MISSING_IMPORTS["torch"] = exc

try:
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    tqdm = None
    _MISSING_IMPORTS["tqdm"] = exc

try:
    from pyprep.find_noisy_channels import NoisyChannels
except ModuleNotFoundError:
    NoisyChannels = None


FREQUENCY = 200
N_CLASSES_TUSZ = 5
NQ_CUT_VAL = 829
INCLUDED_CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "A1",
    "A2",
    "FZ",
    "CZ",
    "PZ",
]
TCP_BIPOLAR_PAIRS = [
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("A1", "T3"),
    ("T3", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("C4", "T4"),
    ("T4", "A2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_torch_load(checkpoint_path: Path):
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def ensure_runtime_dependencies() -> None:
    if not _MISSING_IMPORTS:
        return
    missing = ", ".join(sorted(_MISSING_IMPORTS))
    raise ModuleNotFoundError(
        f"Missing required packages: {missing}. Install the project dependencies first, e.g. `pip install -r 2_dino_eeg/requirements.txt`."
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="One-click EDF -> TCP montage -> DINO-EEG prediction runner."
    )
    parser.add_argument("--edf_path", type=str, default=None)
    parser.add_argument("--edf_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional config .py. If omitted, the script will try to rebuild args from the checkpoint.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(repo_root / f"edf_infer_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--merge_iou_threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--disable_bad_channel_detection", action="store_true")
    parser.add_argument("--save_tcp_h5", action="store_true")
    parser.add_argument("--save_stft_h5", action="store_true")
    return parser.parse_args()


def resolve_edf_files(args: argparse.Namespace) -> list[Path]:
    if bool(args.edf_path) == bool(args.edf_dir):
        raise ValueError("Specify exactly one of --edf_path or --edf_dir.")
    if args.edf_path:
        edf_path = Path(args.edf_path).expanduser().resolve()
        if not edf_path.is_file():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")
        return [edf_path]
    edf_dir = Path(args.edf_dir).expanduser().resolve()
    if not edf_dir.is_dir():
        raise FileNotFoundError(f"EDF directory not found: {edf_dir}")
    edf_files = sorted(edf_dir.rglob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found under: {edf_dir}")
    return edf_files


def get_montage():
    montage = mne.channels.make_standard_montage("standard_1020")
    montage.rename_channels(
        {"Fp1": "FP1", "Fp2": "FP2", "Fz": "FZ", "Cz": "CZ", "Pz": "PZ"}
    )
    return montage


def pick_channels(raw) -> None:
    mapper = {}
    for label in list(raw.info["ch_names"]):
        if "EEG " in label:
            try:
                mapper[label] = label.split("-")[0].split(" ")[1]
            except Exception:
                pass
    raw.rename_channels(mapper)
    raw.pick(list(set(INCLUDED_CHANNELS) & set(raw.info["ch_names"])))


def rereference(raw) -> dict[str, np.ndarray]:
    signal_montage = {}
    for p0, p1 in TCP_BIPOLAR_PAIRS:
        try:
            s0 = raw.get_data(picks=[p0])
            s1 = raw.get_data(picks=[p1])
        except Exception:
            continue
        signal_montage[f"{p0}-{p1}"] = (s0 - s1).astype(np.float32)
    return signal_montage


def apply_batch_stft_and_transform(signals: np.ndarray) -> np.ndarray:
    _, _, zxx = stft(
        signals,
        fs=FREQUENCY,
        nperseg=FREQUENCY,
        axis=-1,
        scaling="spectrum",
    )
    zxx = np.abs(zxx)
    zxx = zxx[:, :64, :]
    zxx = zxx.transpose(0, 2, 1)
    denom = np.sum(zxx, axis=-1, keepdims=True)
    zxx = zxx / (denom + 1e-8)
    zxx[zxx == 0.0] = 1e-8
    zxx = zxx.transpose(0, 2, 1)
    return zxx.astype(np.float32)


def preprocess_single_edf(
    edf_path: Path,
    stft_eval_dir: Path,
    tcp_eval_dir: Path | None,
    detect_bad_channels: bool,
) -> tuple[list[dict], dict]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    try:
        pick_channels(raw)
        raw.set_montage(get_montage())

        if raw.info["sfreq"] != FREQUENCY:
            raw.resample(FREQUENCY)

        raw.notch_filter(60, verbose=False)
        raw.filter(0.1, 70, verbose=False)

        bad_channels = []
        if detect_bad_channels and NoisyChannels is not None:
            try:
                nd = NoisyChannels(raw, do_detrend=True)
                nd.find_bad_by_deviation()
                nd.find_bad_by_nan_flat()
                bad_channels = list(nd.get_bads())
                raw.info["bads"] = bad_channels
                if bad_channels:
                    raw.interpolate_bads(verbose=False)
            except Exception:
                bad_channels = []

        signals = rereference(raw)
        duration = raw.n_times / FREQUENCY
    finally:
        raw.close()

    if not signals:
        return [], {
            "edf_path": str(edf_path),
            "bad_channels": bad_channels,
            "tcp_channel_count": 0,
            "generated_h5_count": 0,
        }

    channel_names = list(signals.keys())
    signals_batch = np.concatenate([signals[c] for c in channel_names], axis=0)
    transformed_batch = apply_batch_stft_and_transform(signals_batch)

    stft_eval_dir.mkdir(parents=True, exist_ok=True)
    if tcp_eval_dir is not None:
        tcp_eval_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    base_name = edf_path.stem
    for idx, channel_name in enumerate(channel_names):
        transformed_signal = transformed_batch[idx : idx + 1]
        if transformed_signal.shape[-1] < NQ_CUT_VAL:
            continue

        file_name = f"{base_name}_{channel_name}.h5"
        signal_path = stft_eval_dir / file_name
        label_data = np.full(signals[channel_name].shape[1], N_CLASSES_TUSZ, dtype=np.uint8)

        with h5py.File(signal_path, "w") as handle:
            handle.create_dataset("signal", data=transformed_signal)
            handle.create_dataset("label", data=label_data)

        if tcp_eval_dir is not None:
            with h5py.File(tcp_eval_dir / file_name, "w") as handle:
                handle.create_dataset("signal", data=signals[channel_name])
                handle.create_dataset("label", data=label_data)

        generated_files.append(
            {
                "filename": file_name,
                "duration": duration,
                "time_dim": int(transformed_signal.shape[-1]),
                "is_background": True,
            }
        )

    summary = {
        "edf_path": str(edf_path),
        "bad_channels": bad_channels,
        "tcp_channel_count": len(channel_names),
        "generated_h5_count": len(generated_files),
    }
    return generated_files, summary


def write_eval_index_files(metadata_list: list[dict], txt_root: Path, nq: int) -> None:
    txt_root.mkdir(parents=True, exist_ok=True)
    seiz_path = txt_root / f"S_eval_NQ{nq}_seiz.txt"
    noseiz_path = txt_root / f"S_eval_NQ{nq}_noseiz.txt"

    with seiz_path.open("w", encoding="utf-8") as handle:
        handle.write("")

    with noseiz_path.open("w", encoding="utf-8") as handle:
        for item in sorted(metadata_list, key=lambda x: x["duration"]):
            handle.write(f'{item["filename"]} {item["duration"]} {item["time_dim"]}\n')


def build_args_from_sources(
    checkpoint_path: Path,
    config_path: Path | None,
    overrides: dict,
) -> types.SimpleNamespace:
    base = {}
    checkpoint = _safe_torch_load(checkpoint_path)
    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is not None:
        if isinstance(checkpoint_args, dict):
            base.update(checkpoint_args)
        else:
            base.update(vars(checkpoint_args))

    if config_path is not None:
        config_mod = _load_module(config_path, "dino_cfg")
        for key, value in config_mod.__dict__.items():
            if not key.startswith("_"):
                base.setdefault(key, value)

    if "modelname" not in base:
        raise ValueError(
            "Unable to rebuild model args from checkpoint. Provide --config_path explicitly."
        )

    base.update(overrides)
    return types.SimpleNamespace(**base)


def get_repo_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parent
    return {
        "repo_root": repo_root,
        "dino_root": repo_root / "2_dino_eeg",
        "post_root": repo_root / "3_postprocess",
    }


def ensure_dino_import_path(dino_root: Path) -> None:
    dino_root_str = str(dino_root)
    if dino_root_str not in sys.path:
        sys.path.insert(0, dino_root_str)


def build_model_and_loader(
    dino_args: types.SimpleNamespace,
    checkpoint_path: Path,
):
    from util.misc import collate_fn, load_checkpoint_mst
    from datasets import build_dataloader
    from models.dino.dino import build_dino

    dataloaders, _ = build_dataloader(collate_fn, dino_args, stage="full")
    model, criterion, postprocessors = build_dino(dino_args)
    checkpoint = load_checkpoint_mst(model, str(checkpoint_path), strict=False, logger=None)
    return model, criterion, postprocessors, dataloaders, checkpoint


def run_inference(
    model,
    postprocessors,
    dataloader,
    device: torch.device,
    threshold: float,
) -> list[dict]:
    raw_predictions = []
    model.eval()

    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc="Inference"):
            samples = samples.to(device)
            outputs = model(samples)
            orig_target_sizes = torch.stack(
                [t["orig_size"] for t in targets], dim=0
            ).to(device)
            results = postprocessors["bbox"](outputs, orig_target_sizes)

            for target, result in zip(targets, results):
                for score, label, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    score_value = float(score.item())
                    if score_value < threshold:
                        continue
                    bbox = box.tolist()
                    raw_predictions.append(
                        {
                            "image_id": target["image_id"],
                            "bbox": [bbox[0], 0, max(bbox[1] - bbox[0], 0), 63],
                            "score": score_value,
                            "category_id": int(label.item()),
                            "width": float(target["orig_size"].item()),
                        }
                    )

    return raw_predictions


def interval_iou(box_a: list[float], box_b: list[float]) -> float:
    a_start = float(box_a[0])
    a_end = float(box_a[0] + box_a[2])
    b_start = float(box_b[0])
    b_end = float(box_b[0] + box_b[2])
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0 else inter / union


def segment_id_from_image_id(image_id: str) -> str:
    return Path(image_id).stem.rsplit("_", 1)[0]


def merge_multichannel_predictions(
    predictions: list[dict],
    score_threshold: float,
    iou_threshold: float,
) -> list[dict]:
    grouped = {}
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        seg_id = segment_id_from_image_id(pred["image_id"])
        grouped.setdefault(seg_id, []).append(pred)

    merged = []
    for seg_id, preds in grouped.items():
        kept = []
        for pred in sorted(preds, key=lambda x: x["score"], reverse=True):
            if any(interval_iou(pred["bbox"], kept_pred["bbox"]) > iou_threshold for kept_pred in kept):
                continue
            merged_pred = dict(pred)
            merged_pred["image_id"] = seg_id
            merged.append(merged_pred)
            kept.append(pred)
    return merged


def predictions_to_csv_rows(predictions: list[dict]) -> list[dict]:
    rows = []
    for pred in predictions:
        onset = float(pred["bbox"][0])
        duration = float(pred["bbox"][2])
        rows.append(
            {
                "image_id": pred["image_id"],
                "onset_sec": onset,
                "offset_sec": onset + duration,
                "duration_sec": duration,
                "score": float(pred["score"]),
                "category_id": int(pred["category_id"]),
                "width_sec": float(pred.get("width", 0)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    paths = get_repo_paths()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = None
    if args.config_path:
        config_path = Path(args.config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = output_root / "data"
    txt_root = output_root / "txt"
    eval_out = output_root / "predictions"
    stft_eval_dir = data_root / "stft_amp_w_scale_w_crop" / "eval"
    tcp_eval_dir = (data_root / "tcp_montage" / "eval") if args.save_tcp_h5 else None
    eval_out.mkdir(parents=True, exist_ok=True)

    overrides = {
        "dataset": "tusz",
        "data_dir": str(data_root / "stft_amp_w_scale_w_crop"),
        "tusz_txt_dir": str(txt_root),
        "tusz_label_dir": None,
        "eval": True,
        "distributed": False,
        "amp": False,
        "num_workers": args.num_workers,
        "test_batch_size": args.test_batch_size,
        "device": args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        "downsample_seed": 42,
        "tusz_downsample_times": 0,
    }
    dino_args = build_args_from_sources(checkpoint_path, config_path, overrides)
    nq = int(dino_args.num_queries) + int(dino_args.dn_number) * 2

    edf_files = resolve_edf_files(args)
    preprocess_metadata = []
    preprocess_summary = []
    for edf_path in tqdm(edf_files, desc="Preprocess"):
        generated, summary = preprocess_single_edf(
            edf_path=edf_path,
            stft_eval_dir=stft_eval_dir,
            tcp_eval_dir=tcp_eval_dir,
            detect_bad_channels=not args.disable_bad_channel_detection,
        )
        preprocess_metadata.extend(generated)
        preprocess_summary.append(summary)

    if not preprocess_metadata:
        raise RuntimeError("No valid TCP/STFT H5 files were generated from the input EDF data.")

    write_eval_index_files(preprocess_metadata, txt_root, nq)

    ensure_dino_import_path(paths["dino_root"])
    model, _criterion, postprocessors, dataloaders, _checkpoint = build_model_and_loader(
        dino_args,
        checkpoint_path,
    )

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model.to(device)

    raw_predictions = run_inference(
        model=model,
        postprocessors=postprocessors,
        dataloader=dataloaders["test"],
        device=device,
        threshold=args.threshold,
    )
    merged_predictions = merge_multichannel_predictions(
        raw_predictions,
        score_threshold=args.threshold,
        iou_threshold=args.merge_iou_threshold,
    )

    raw_json = eval_out / "results.bbox.json"
    merged_json = eval_out / f"merged_predictions_nms_{args.threshold:.2f}.json"
    raw_csv = eval_out / "results.bbox.csv"
    merged_csv = eval_out / f"merged_predictions_nms_{args.threshold:.2f}.csv"
    summary_json = eval_out / "run_summary.json"

    with raw_json.open("w", encoding="utf-8") as handle:
        json.dump(raw_predictions, handle, ensure_ascii=False, indent=2)
    with merged_json.open("w", encoding="utf-8") as handle:
        json.dump(merged_predictions, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(predictions_to_csv_rows(raw_predictions)).to_csv(raw_csv, index=False)
    pd.DataFrame(predictions_to_csv_rows(merged_predictions)).to_csv(merged_csv, index=False)

    summary = {
        "edf_count": len(edf_files),
        "generated_h5_count": len(preprocess_metadata),
        "raw_prediction_count": len(raw_predictions),
        "merged_prediction_count": len(merged_predictions),
        "threshold": args.threshold,
        "merge_iou_threshold": args.merge_iou_threshold,
        "device": str(device),
        "paths": {
            "output_root": str(output_root),
            "stft_eval_dir": str(stft_eval_dir),
            "txt_root": str(txt_root),
            "raw_json": str(raw_json),
            "merged_json": str(merged_json),
            "raw_csv": str(raw_csv),
            "merged_csv": str(merged_csv),
        },
        "preprocess_summary": preprocess_summary,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
