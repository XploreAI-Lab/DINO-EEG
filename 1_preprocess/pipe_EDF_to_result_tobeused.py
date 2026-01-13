import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import mne
from scipy.signal import stft
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pyprep.find_noisy_channels import NoisyChannels
import warnings
import time

# ===============================
# Global config
# ===============================
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

FREQUENCY = 200
N_CLASSES_TUSZ = 5

INCLUDED_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ"
]

LABEL_MAPPER = {
    "fnsz": 1, "cpsz": 1, "spsz": 1, "gnsz": 1,
    "tnsz": 3, "tcsz": 3, "absz": 4,
}

TIME_STEP_SIZE = 1
PHYSICAL_TIME_STEP_SIZE = int(FREQUENCY * TIME_STEP_SIZE)

NQ = 1100
NQ_CUT_VAL = 829

# ===============================
# Helper functions
# ===============================
def get_montage():
    montage = mne.channels.make_standard_montage("standard_1020")
    montage.rename_channels({
        "Fp1": "FP1", "Fp2": "FP2",
        "Fz": "FZ", "Cz": "CZ", "Pz": "PZ"
    })
    return montage


def pick_channels(raw):
    labels = list(raw.info["ch_names"])
    mapper = {}
    for label in labels:
        if "EEG " in label:
            try:
                mapper[label] = label.split("-")[0].split(" ")[1]
            except:
                pass
    raw.rename_channels(mapper)
    raw.pick(list(set(INCLUDED_CHANNELS) & set(raw.info["ch_names"])))


def rereference(raw):
    signal_montage = {}
    bipolar_pairs = [
        ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
        ("C4", "T4"), ("T4", "A2"),
        ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ]

    for p0, p1 in bipolar_pairs:
        try:
            s0 = raw.get_data(picks=[p0])
            s1 = raw.get_data(picks=[p1])
            signal_montage[f"{p0}-{p1}"] = (s0 - s1).astype(np.float32)
        except:
            continue

    return signal_montage


# ===============================
# 🔥 Batch STFT (核心优化)
# ===============================
def build_batch(signals_dict):
    channel_names = list(signals_dict.keys())
    batch = np.concatenate([signals_dict[c] for c in channel_names], axis=0)
    return batch, channel_names


def apply_batch_stft_and_transform(signals, fs, nperseg):
    """
    signals: [C, L]
    return:  [C, 64, T]
    """
    _, _, Zxx = stft(
        signals,
        fs=fs,
        nperseg=nperseg,
        axis=-1,
        scaling="spectrum"
    )
    Zxx = np.abs(Zxx)
    Zxx = Zxx[:, :64, :]

    # normalize per time step
    Zxx = Zxx.transpose(0, 2, 1)
    denom = np.sum(Zxx, axis=-1, keepdims=True)
    Zxx = Zxx / (denom + 1e-8)
    Zxx[Zxx == 0.0] = 1e-8
    Zxx = Zxx.transpose(0, 2, 1)

    return Zxx


# ===============================
# Main worker
# ===============================
def process_single_file(edf_path, save_dir_root, task_split):
    try:
        base_path = os.path.splitext(edf_path)[0]
        csv_path = base_path + ".csv"

        if not os.path.exists(csv_path):
            return None

        try:
            csv_df = pd.read_csv(csv_path, sep=",", comment="#")
        except:
            return None

        if "label" in csv_df.columns:
            if csv_df["label"].isin(["seiz", "mysz"]).any():
                return None

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except:
            return None

        pick_channels(raw)
        raw.set_montage(get_montage())

        if raw.info["sfreq"] != FREQUENCY:
            raw.resample(FREQUENCY)

        raw.notch_filter(60, verbose=False)
        raw.filter(0.1, 70, verbose=False)

        try:
            nd = NoisyChannels(raw, do_detrend=True)
            nd.find_bad_by_deviation()
            nd.find_bad_by_nan_flat()
            raw.info["bads"] = nd.get_bads()
            raw.interpolate_bads(verbose=False)
        except:
            pass

        signals = rereference(raw)
        raw.close()
        del raw

        # labels
        labels_dict = {
            cname: np.full(signal.shape[1], N_CLASSES_TUSZ, dtype=np.uint8)
            for cname, signal in signals.items()
        }

        def _apply_row(row):
            if row["label"] == "bckg":
                return
            if row["channel"] not in labels_dict:
                return
            start = int(row["start_time"] * FREQUENCY)
            end = int(row["stop_time"] * FREQUENCY)
            mapped = LABEL_MAPPER.get(row["label"])
            if mapped is not None:
                labels_dict[row["channel"]][start:end] = mapped

        if "label" in csv_df.columns:
            csv_df.apply(_apply_row, axis=1)

        output_subdir = os.path.join(
            save_dir_root, "stft_amp_w_scale_w_crop", task_split
        )
        os.makedirs(output_subdir, exist_ok=True)

        base_name = os.path.basename(base_path)
        generated_files = []

        # ===== 批量 STFT =====
        signals_batch, channel_names = build_batch(signals)
        transformed_batch = apply_batch_stft_and_transform(
            signals_batch, FREQUENCY, PHYSICAL_TIME_STEP_SIZE
        )

        for idx, cname in enumerate(channel_names):
            transformed_signal = transformed_batch[idx:idx + 1]
            time_dim = transformed_signal.shape[-1]

            if time_dim < NQ_CUT_VAL:
                continue

            out_fn = f"{base_name}_{cname}.h5"
            out_path = os.path.join(output_subdir, out_fn)
            label_data = labels_dict[cname]

            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("signal", data=transformed_signal)
                hf.create_dataset("label", data=label_data)

            generated_files.append({
                "filename": out_fn,
                "duration": len(label_data) / FREQUENCY,
                "time_dim": time_dim,
                "is_background": (label_data == N_CLASSES_TUSZ).all(),
                "split": task_split
            })

        return generated_files

    except Exception as e:
        print(f"[ERROR] {edf_path}: {e}")
        return None


# ===============================
# Index writer
# ===============================
def write_index_files(metadata_list, save_dir_root):
    for task in ["train", "dev", "eval"]:
        seiz, bg = {}, {}

        for m in metadata_list:
            if m["split"] != task:
                continue
            target = bg if m["is_background"] else seiz
            target[m["filename"]] = (m["duration"], m["time_dim"])

        def dump(name, data):
            with open(name, "w") as f:
                for k, v in sorted(data.items(), key=lambda x: x[1][0]):
                    f.write(f"{k} {v[0]} {v[1]}\n")

        prefix = "FS_" if task != "eval" else "S_"
        dump(os.path.join(save_dir_root, f"{prefix}{task}_NQ{NQ}_seiz.txt"), seiz)
        dump(os.path.join(save_dir_root, f"{prefix}{task}_NQ{NQ}_noseiz.txt"), bg)

        print(f"[{task.upper()}] seiz={len(seiz)}, bg={len(bg)}")


# ===============================
# Entry
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tusz_root", type=str, default="/root/autodl-tmp/TUSZ_eval")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/TUSZ_eval3")
    parser.add_argument("--n_jobs", type=int, default=11)
    args = parser.parse_args()

    tasks = ["train", "dev", "eval"]
    file_list = []

    for task in tasks:
        task_dir = os.path.join(args.tusz_root, task)
        for root, _, files in os.walk(task_dir):
            for f in files:
                if f.endswith(".edf"):
                    file_list.append((os.path.join(root, f), args.save_dir, task))

    print(f"Found {len(file_list)} EDF files")

    all_metadata = []

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = [ex.submit(process_single_file, *f) for f in file_list]
        for fu in tqdm(as_completed(futures), total=len(futures)):
            res = fu.result()
            if res:
                all_metadata.extend(res)

    write_index_files(all_metadata, args.save_dir)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {(time.time() - start) / 60:.2f} min")
