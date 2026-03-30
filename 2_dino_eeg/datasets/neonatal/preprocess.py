import argparse
import os
import h5py
import mne
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

import sys

sys.path.append("..")

from datasets.constants import FREQUENCY, INCLUDED_CHANNELS


def main(raw_edf_dir: str, save_dir):
    edf_files = list(filter(lambda x: x.endswith(".edf"), os.listdir(raw_edf_dir)))

    # 加载注释文件
    Annotations = loadmat(os.path.join(raw_edf_dir, "annotations_2017.mat"))
    # 以秒为单位的注释 长度为79的元素为[3, L(秒)]的ndarray
    Annotations = Annotations["annotat_new"].squeeze()

    seiz = []
    noseiz = []

    for edf_fn in edf_files:
        # [3, L(秒)] 获取标注信息 将文件名称与数组下标联系起来
        annotations = Annotations[int(edf_fn.split(".")[0][3:]) - 1]
        # [L(秒), ]
        result_and = np.bitwise_and.reduce(annotations).squeeze()
        result_or = np.bitwise_or.reduce(annotations).squeeze()
        # 不是“与运算”结果含有1，“或运算”结果全为0的，说明没有达成共识
        if np.any(result_and == 1):
            annotation = result_and
            seiz.append(edf_fn.split(".edf")[0] + "\n")
        elif np.all(result_or == 0):
            annotation = result_or
            noseiz.append(edf_fn.split(".edf")[0] + "\n")
        else:
            continue

        # 加载edf文件
        raw = mne.io.read_raw_edf(
            os.path.join(raw_edf_dir, edf_fn), preload=True, verbose=False
        )
        pickChannels(raw, INCLUDED_CHANNELS)

        # 下采样至200Hz
        sample_freq = raw.info["sfreq"]
        if sample_freq != FREQUENCY:
            raw.resample(FREQUENCY, verbose=False)

        raw.notch_filter(60, verbose=False)
        raw.filter(0.1, 70, verbose=False)

        # TCP双极蒙太奇
        # {"FP1-F7": ndarray[1, L], ...}
        signals, _ = rereference(raw)
        raw.close()

        for cname, signal in signals.items():
            # 直接扩大FREQUENCY倍，即将以秒为单位的数组转换为以采样点为单位的数组
            label = np.repeat(annotation, FREQUENCY).astype(np.uint8)
            assert label.shape[-1] == signal.shape[-1], edf_fn

            file_path = os.path.join(
                save_dir, edf_fn.split(".edf")[0] + "_" + cname + ".h5"
            )

            with h5py.File(file_path, "w") as shf:
                shf.create_dataset("signal", data=signal)
                shf.create_dataset("label", data=label)

    with open(os.path.join(save_dir, "seiz.txt"), "w") as txt:
        txt.writelines(seiz)

    with open(os.path.join(save_dir, "noseiz.txt"), "w") as txt:
        txt.writelines(noseiz)

    print("Done.")


def pickChannels(raw, ordered_channel_names):
    labels = list(raw.info["ch_names"])
    channel_name_mapper = {}
    for label in labels:
        # EEG FP1-REF -> EEG FP1 / EEG FP1-LE -> EEG FP1 -> FP1 并且去掉如 PHOTIC-REF
        if "EEG " in label:
            channel_name_mapper[label] = label.split("-")[0].split(" ")[1]
    raw.rename_channels(channel_name_mapper)
    # Reorder channels. Channels that are not in ch_names are dropped
    # 后续使用get_data获取通道值，因此顺序不重要了，有些文件缺少部分通道，使用交集
    raw.pick(list(set(ordered_channel_names) & set(raw.info["ch_names"])))


# Neonatal 只能做不含A1，A2的20channel montage
def rereference(raw):
    # 最多22个通道
    signal_montage = {}
    labels_montage = [
        "FP1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "FP2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "T3-C3",
        "C3-CZ",
        "CZ-C4",
        "C4-T4",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
    ]

    bipolarPairs = [
        ("FP1", "F7"),
        ("F7", "T3"),
        ("T3", "T5"),
        ("T5", "O1"),
        ("FP2", "F8"),
        ("F8", "T4"),
        ("T4", "T6"),
        ("T6", "O2"),
        ("T3", "C3"),
        ("C3", "CZ"),
        ("CZ", "C4"),
        ("C4", "T4"),
        ("FP1", "F3"),
        ("F3", "C3"),
        ("C3", "P3"),
        ("P3", "O1"),
        ("FP2", "F4"),
        ("F4", "C4"),
        ("C4", "P4"),
        ("P4", "O2"),
    ]

    for pair in bipolarPairs:
        try:
            signal_montage[pair[0] + "-" + pair[1]] = (
                raw.get_data(picks=[pair[0]]) - raw.get_data(picks=[pair[1]])
            ).astype(np.float32)
        except ValueError:
            # 当通道不存在时，可能是AR参考或缺少，跳过
            continue

    return signal_montage, labels_montage


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get data sliced.")
    parser.add_argument(
        "--raw_edf_dir",
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
    main(args.raw_edf_dir, args.save_dir)
