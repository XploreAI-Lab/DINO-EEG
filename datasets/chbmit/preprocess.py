# 不切片 分通道单独保存
import argparse
import os
import sys

import mne
import h5py
import numpy as np

from tqdm import tqdm

sys.path.append("..")

from datasets.constants import FREQUENCY


def main(raw_edf_dir, preprocessed_label_dir, save_dir):
    edf_files: list[str] = []
    # 递归遍历raw_edf_dir, 返回的是一个三元组(root,dirs,files)
    for path, _, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name and ".edf.seizures" not in name:
                edf_files.append(os.path.join(path, name))

    total = len(edf_files)

    label_files = set(os.listdir(preprocessed_label_dir))
    ignored = []

    for idx in range(total):
        print("Processing {}th edf file, total {} files.".format(idx + 1, total))

        edf_fn = edf_files[idx]
        edf_jst_fn = edf_fn.split("/")[-1].split(".edf")[0]

        if not edf_jst_fn + ".h5" in label_files:
            print("Pass the {}th edf file.".format(idx + 1))
            ignored.append(edf_jst_fn + ".h5")
            continue

        # assert edf_jst_fn + ".h5" in label_files

        print("Current filename: {}".format(edf_jst_fn))

        raw = mne.io.read_raw_edf(edf_fn, preload=True, verbose=False)
        raw.resample(FREQUENCY, verbose=False)
        raw.notch_filter(60, verbose=False)
        raw.filter(0.1, 70, verbose=False)

        # 共5个这样的非法文件
        try:
            signals, _ = rereference(raw)
        except ValueError as e:
            print("No Channel Error, Pass")
            continue
        finally:
            raw.close()

        label_file_path = os.path.join(preprocessed_label_dir, edf_jst_fn + ".h5")
        with h5py.File(label_file_path, "r") as hf:
            boxes = hf["boxes"][()]

        if any(p in edf_jst_fn for p in {"chb20", "chb22"}):
            task = "dev"
        elif any(p in edf_jst_fn for p in {"chb23", "chb24"}):
            task = "eval"
        else:
            task = "train"

        save_dir_c = os.path.join(save_dir, task)

        for cname, signal in signals.items():
            label = np.full_like(signal[0], 0, dtype=np.uint8)
            for box in boxes:
                label[int(box[0] * FREQUENCY) : int(box[1] * FREQUENCY)] = 1

            file_path = os.path.join(save_dir_c, edf_jst_fn + "_" + cname + ".h5")

            with h5py.File(file_path, "w") as shf:
                shf.create_dataset("signal", data=signal)
                shf.create_dataset("label", data=label)

    print("Done.")
    print(f"{len(ignored)} filed ignored:")
    for file in ignored:
        print(file)


def rereference(raw):
    signal_montage = {}
    labels_montage = [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "T8-P8-0",
        "P8-O2",
        "FZ-CZ",
        "CZ-PZ",
        "P7-T7",
        "T7-FT9",
        "FT9-FT10",
        "FT10-T8",
    ]

    for montage in labels_montage:
        signal_montage[montage] = raw.get_data(picks=[montage]).astype(np.float32)

    return signal_montage, labels_montage


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get data sliced.")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--preprocessed_label_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="数据存储位置。(绝对路径)",
    )
    args = parser.parse_args()
    main(args.raw_edf_dir, args.preprocessed_label_dir, args.save_dir)
