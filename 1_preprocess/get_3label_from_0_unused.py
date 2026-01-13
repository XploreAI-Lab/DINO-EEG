import os
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 参数
data_dir = "/root/autodl-tmp/SeizeIT2_stft_sliced/dev"   # 原始 h5 文件路径
save_dir = "/root/autodl-tmp/SeizeIT2_3label_sliced/dev"  # 新label保存路径
MAX_WORKERS = 8  # 可根据机器配置修改

os.makedirs(save_dir, exist_ok=True)

def process_label_file(h5_fn):
    try:
        src_path = os.path.join(data_dir, h5_fn)
        dst_path = os.path.join(save_dir, h5_fn)

        with h5py.File(src_path, "r") as hf:
            label = hf["label"][()]  # shape: (L,)

        # ✅ label修改策略
        label[label == 0] = 5  # 将背景类 0 改为 5

        # ✅ 只保存新的 label，不包含 signal
        with h5py.File(dst_path, "w") as hf:
            hf.create_dataset("label", data=label, dtype=np.uint8)

        return f"[✓] Processed {h5_fn}"
    except Exception as e:
        return f"[✗] Failed {h5_fn}: {str(e)}"

# 获取文件列表
h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]

# 多线程执行
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_label_file, h5_fn) for h5_fn in h5_files]
    for result in tqdm(as_completed(futures), total=len(futures)):
        print(result.result())
