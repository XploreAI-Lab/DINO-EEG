import os
import mne
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

FREQUENCY = 200  # 目标采样率
MAX_WORKERS = 8
N_CLASSES_TUSZ = 5  # 0-4为癫痫类型, 5为bckg类

# ========== TCP 参考函数 ==========
def pickChannels(raw, ordered_channel_names):
    labels = list(raw.info["ch_names"])
    channel_name_mapper = {}
    for label in labels:
        channel_name_mapper[label] = label.split("-")[0]
    raw.rename_channels(channel_name_mapper)
    raw.pick(list(set(ordered_channel_names) & set(raw.info["ch_names"])))

def rereference(raw):
    signal_montage = {}
    labels_montage = [
        "FP1-F7","F7-T3","T3-T5","T5-O1",
        "FP2-F8","F8-T4","T4-T6","T6-O2",
        "A1-T3","T3-C3","C3-CZ","CZ-C4","C4-T4","T4-A2",
        "FP1-F3","F3-C3","C3-P3","P3-O1",
        "FP2-F4","F4-C4","C4-P4","P4-O2",
    ]
    bipolarPairs = [
        ("FP1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),
        ("FP2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
        ("A1","T3"),("T3","C3"),("C3","CZ"),("CZ","C4"),("C4","T4"),("T4","A2"),
        ("FP1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),
        ("FP2","F4"),("F4","C4"),("C4","P4"),("P4","O2"),
    ]
    for pair in bipolarPairs:
        try:
            signal_montage[pair[0]+"-"+pair[1]] = (
                raw.get_data(picks=[pair[0]]) - raw.get_data(picks=[pair[1]])
            ).astype(np.float32)
        except ValueError:
            continue
    return signal_montage, labels_montage

# 事件映射
def map_event_type(event_type: str):
    et = event_type.lower()
    if et.startswith("sz"):
        return 1  # 发作
    else:
        return None  # 忽略

def process_edf_file(sub_id, edf_path, events_path, save_dir):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.resample(FREQUENCY, verbose=False)
        raw.notch_filter(freqs=50, verbose=False)
        raw.filter(l_freq=0.1, h_freq=70, verbose=False)

        # 先做通道重命名+挑选
        ordered_channels = [
            "FP1","F7","T3","T5","O1",
            "FP2","F8","T4","T6","O2",
            "A1","T3","C3","CZ","C4","A2",
            "F3","P3","F4","P4"
        ]
        pickChannels(raw, ordered_channels)

        # TCP 双极参考
        signals, montage_names = rereference(raw)
        raw.close()

        # 初始化每个通道标签
        labels_dict = {ch: np.full(signals[ch].shape[1], N_CLASSES_TUSZ, dtype=np.uint8) 
                       for ch in signals}

        # 读取事件并填充标签
        events_df = pd.read_csv(events_path, sep='\t')
        for _, row in events_df.iterrows():
            cls = map_event_type(row['eventType'])
            if cls is None:
                continue
            onset = float(row['onset'])
            duration = float(row['duration'])
            start_sample = int(onset * FREQUENCY)
            end_sample = int((onset + duration) * FREQUENCY)
            for ch in labels_dict:
                labels_dict[ch][start_sample:end_sample] = cls

        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(edf_path))[0]

        # 保存每个TCP通道的信号与标签
        for cname, signal in signals.items():
            file_path = os.path.join(save_dir, f"{base}_{cname}.h5")
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("label", data=labels_dict[cname])

        return f"[✓] Finished {sub_id} - {os.path.basename(edf_path)}"

    except Exception as e:
        return f"[✗] Error in {sub_id} - {os.path.basename(edf_path)}: {str(e)}"

def main():
    root_dir = "/root/siena"
    save_dir = "/root/Siena_TCP"
    edf_tasks = []

    for sid in range(0, 18):
        sub_id = f"sub-{sid:02d}"
        eeg_dir = os.path.join(root_dir, sub_id, "ses-01", "eeg")
        if not os.path.isdir(eeg_dir):
            continue
        for file in os.listdir(eeg_dir):
            if file.endswith(".edf"):
                edf_path = os.path.join(eeg_dir, file)
                base_name = file.replace("_eeg.edf", "")
                events_file = f"{base_name}_events.tsv"
                events_path = os.path.join(eeg_dir, events_file)
                if os.path.exists(events_path):
                    edf_tasks.append((sub_id, edf_path, events_path, save_dir))
                else:
                    print(f"[✗] Missing events file for {sub_id} - {file}, skipping.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_edf_file, *task) for task in edf_tasks]
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
