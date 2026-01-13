import os
import mne
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

FREQUENCY = 200  # 目标采样率
MAX_WORKERS = 8  # 线程数可根据机器核数调整


# 事件类型映射函数
def map_event_type(event_type: str):
    et = event_type.lower()
    if et.startswith("sz"):
        return 1  # 发作类标为1
    else:
        return None  # 其他忽略


def process_edf_file(edf_path, events_path, save_dir, root_dir):
    try:
        # 解析 sub 和 ses
        rel_path = os.path.relpath(edf_path, root_dir)  # sub-00/ses-01/eeg/file.edf
        parts = rel_path.split(os.sep)
        sub_id = parts[0] if len(parts) > 0 else "unknown_sub"
        ses_id = parts[1] if len(parts) > 1 else "unknown_ses"

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.resample(FREQUENCY, verbose=False)
        raw.notch_filter(freqs=60, verbose=False)
        raw.filter(l_freq=0.1, h_freq=70, verbose=False)

        data = raw.get_data()
        channel_names = raw.ch_names
        n_samples = data.shape[1]

        labels = np.zeros(n_samples, dtype=np.uint8)

        events_df = pd.read_csv(events_path, sep="\t")
        for _, row in events_df.iterrows():
            cls = map_event_type(row["eventType"])
            if cls is None:
                continue
            onset = float(row["onset"])
            duration = float(row["duration"])
            start_sample = int(onset * FREQUENCY)
            end_sample = int((onset + duration) * FREQUENCY)
            labels[start_sample:end_sample] = cls

        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(edf_path))[0]

        for idx, channel in enumerate(channel_names):
            signal = data[idx]
            clean_channel = channel.replace(" SD", "").replace(" ", "_")
            file_name = f"{base}_{clean_channel}.h5"
            file_path = os.path.join(save_dir, file_name)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("signal", data=signal.astype(np.float32))
                hf.create_dataset("label", data=labels, dtype=np.uint8)

        raw.close()
        return f"[✓] Finished {sub_id}/{ses_id} - {os.path.basename(edf_path)}"

    except Exception as e:
        return f"[✗] Error in {os.path.basename(edf_path)}: {str(e)}"


def main():
    root_dir = "/root/autodl-tmp/TUSZ_avg/eval"
    save_dir = "/root/autodl-tmp/TUSZ_avg_stft/eval"
    edf_tasks = []

    for sub in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_dir) or not sub.startswith("sub-"):
            continue

        # 遍历所有 ses 文件夹
        for ses in os.listdir(sub_dir):
            ses_dir = os.path.join(sub_dir, ses, "eeg")
            if not os.path.isdir(ses_dir):
                continue

            for file in os.listdir(ses_dir):
                if file.endswith(".edf"):
                    edf_path = os.path.join(ses_dir, file)
                    base_name = file.replace("_eeg.edf", "")
                    events_file = f"{base_name}_events.tsv"
                    events_path = os.path.join(ses_dir, events_file)

                    if os.path.exists(events_path):
                        edf_tasks.append((edf_path, events_path, save_dir, root_dir))
                    else:
                        pass
                        # print(f"[✗] Missing events file for {edf_path}, skipping.")

    # 多线程执行处理任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_edf_file, *task) for task in edf_tasks]
        for future in as_completed(futures):
            pass
            # print(future.result())


if __name__ == "__main__":
    main()
