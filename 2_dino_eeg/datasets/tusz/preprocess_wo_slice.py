# 测试集 不切片 分通道单独保存
from collections import defaultdict
import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import mne
from pyprep.find_noisy_channels import NoisyChannels
from pandas import read_csv  # type: ignore

import sys

sys.path.append("..")

from datasets.constants import FREQUENCY, INCLUDED_CHANNELS, N_CLASSES_TUSZ


# 这里直接合并了，不需要额外合并操作
label_mapper = {
    # 合并为CBSZ
    "fnsz": 1,
    "cpsz": 1,
    "spsz": 1,
    "gnsz": 1,
    # 合并为CTSZ
    "tnsz": 3,
    "tcsz": 3,
    "absz": 4,
}


def _deal_row(row, labels):
    label = row["label"]
    # 如果是背景类就忽略，label就是按照背景类初始化的
    if label == "bckg":
        return

    # print(40, (labels[row["channel"]]).shape, row["channel"])

    try:
        labels[row["channel"]][
            int(row["start_time"] * FREQUENCY) : int(row["stop_time"] * FREQUENCY)
        ] = label_mapper[label]
    except KeyError as e:
        print(e)
        return

def main(raw_edf_dir, save_dir):
    edf_files = []
    # 递归遍历raw_edf_dir, 返回的是一个三元组(root,dirs,files)
    for path, _, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    total = len(edf_files)

    bad_channels_total = 0
    # 设置montage，获取电极位置信息，方便坏导检测和插值重建
    montage = mne.channels.make_standard_montage("standard_1020")
    montage.rename_channels(
        {"Fp1": "FP1", "Fp2": "FP2", "Fz": "FZ", "Cz": "CZ", "Pz": "PZ"}
    )

    file_ignored = 0
    for idx in range(total):
        print(
            "#####################Processing {}th edf file, total {} files.#####################".format(
                idx + 1, total
            )
        )
        edf_fn = edf_files[idx]
        print(
            "#####################Current filename: {}.#####################".format(
                edf_fn.split(".edf")[0]
            )
        )

        # 先加载.csv文件
        csv_file = read_csv(edf_fn.split(".edf")[0] + ".csv", sep=",", comment="#")
        # 过滤非法文件label中含有[mysz,seiz]的文件
        if csv_file["label"].isin(["seiz", "mysz"]).any():
            file_ignored += 1
            continue

        # print(76, edf_fn)

        # 加载edf文件
        raw = mne.io.read_raw_edf(edf_fn, preload=True, verbose=False)
        pickChannels(raw, INCLUDED_CHANNELS)
        raw.set_montage(montage)

        # 下采样至200Hz
        sample_freq = raw.info["sfreq"]
        if sample_freq != FREQUENCY:
            raw.resample(FREQUENCY)

        raw.notch_filter(60)
        raw.filter(0.1, 70)

        # 坏导检测（经过1Hz高通滤波）
        nd = NoisyChannels(raw, do_detrend=True)
        nd.find_bad_by_deviation()
        nd.find_bad_by_nan_flat()
        # 更新坏导信息
        raw.info["bads"] = nd.get_bads()
        bad_channels_total += len(raw.info["bads"])
        print(
            "Find bad channel(s) of current file:"
            + str(raw.info["bads"])
            + ". Total {} bad channel(s).".format(bad_channels_total)
        )
        # 通过样条插值函数重建坏导
        raw.interpolate_bads()

        # TCP双极蒙太奇
        # {"FP1-F7": ndarray[1, L], ...}
        signals, _ = rereference(raw)
        raw.close()

        # 5代表bckg类 最大label_id为4，num_classes=5
        # {"FP1-F7": ndarray[L,], ...} signal有的key，label不一定有（通道没有事件）
        labels = {}
        for cname, signal in signals.items():
            # signal [1, L]
            labels[cname] = np.full_like(signal[0], N_CLASSES_TUSZ, dtype=np.uint8)
            # print(labels[cname].shape)

        csv_file.apply(_deal_row, axis=1, labels=labels)

        # 通道分开存储
        # 文件名 e.g. aaaaaaar_00000001_FP1-F7.h5
        for cname, signal in signals.items():
            # if (labels[cname] == N_CLASSES_TUSZ).all():
            #     continue

            file_path = os.path.join(
                save_dir, edf_fn.split("/")[-1].split(".edf")[0] + "_" + cname + ".h5"
            )

            with h5py.File(file_path, "w") as shf:
                shf.create_dataset("signal", data=signal)
                shf.create_dataset("label", data=labels[cname])

    print("Done.")


# Pick,rename and order channels name by ordered_channel_names
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
    # raw.reorder_channels(ordered_channel_names)


# 做Montage TUSZ有两种montage，其中03_tcp_ar_a不包含A1和A2电极相关的montage
# 且存在缺少通道的文件
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
        "A1-T3",
        "T3-C3",
        "C3-CZ",
        "CZ-C4",
        "C4-T4",
        "T4-A2",
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

    for pair in bipolarPairs:
        try:
            signal_montage[pair[0] + "-" + pair[1]] = (
                raw.get_data(picks=[pair[0]]) - raw.get_data(picks=[pair[1]])
            ).astype(np.float32)

            # print(205,(signal_montage[pair[0] + "-" + pair[1]]).shape)

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
