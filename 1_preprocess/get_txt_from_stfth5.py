import os
import h5py
from tqdm import tqdm

# 任务集（对应子文件夹名）
tasks = ["dev"]
# tasks = ["eval"]
# 设置常量
FREQUENCY = 200  # 信号采样率（Hz）
NQ = 1100  # 查询个数
NQ_cut = {
    1100: 829,  # 用于筛选最短长度，单位为采样点数
}

# 输出信息格式化函数
def _func(item):
    return item[0] + " " + str(item[1][0]) + " " + str(item[1][1]) + "\n"

# 核心处理函数
def handle(path: str, do_filter: bool = True):
    files = os.listdir(path)
    seiz = {}
    noseiz = {}

    for file in tqdm(files, desc=f"Processing {path}"):
        if not file.endswith(".h5"):
            continue

        with h5py.File(os.path.join(path, file), "r") as hf:
            signal = hf["signal"][()]
            y = hf["label"][()]

        ls = signal.shape[-1]
        if do_filter and ls < NQ_cut[NQ]:
            continue

        # 判断是否为背景类（全 0）
        if (y == 0).all():
            noseiz[file] = (len(y) / FREQUENCY, ls)
        else:
            seiz[file] = (len(y) / FREQUENCY, ls)

    # 按时长升序排序
    seiz = dict(sorted(seiz.items(), key=lambda item: item[1][0]))
    noseiz = dict(sorted(noseiz.items(), key=lambda item: item[1][0]))

    seiz = list(map(_func, seiz.items()))
    noseiz = list(map(_func, noseiz.items()))

    return seiz, noseiz

# 主函数
def run_main(base_path):
    for task in tasks:
        task_path = base_path
        do_filter = task != "eval"

        seiz, noseiz = handle(task_path, do_filter)

        prefix = "FS_" if do_filter else "S_"
        out_seiz = os.path.join(base_path, f"{prefix}{task}_NQ{NQ}_seiz.txt")
        out_noseiz = os.path.join(base_path, f"{prefix}{task}_NQ{NQ}_noseiz.txt")

        with open(out_seiz, "w") as f:
            f.writelines(seiz)

        with open(out_noseiz, "w") as f:
            f.writelines(noseiz)

    print("✅ 所有任务完成，已保存输出文件。")

# 使用示例（修改为你自己的路径）
base_path = "/root/autodl-tmp/TUSZ_avg_stft/dev"  # 修改为你的数据根路径
run_main(base_path)
