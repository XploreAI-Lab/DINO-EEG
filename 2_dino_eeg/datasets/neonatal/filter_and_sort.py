import os
import sys
import h5py
from tqdm import tqdm

tasks = [
    "train",
    "dev",
    "eval",
]

# NQ与最短K的长度对应关系
# lengthT >= NQ/((1/(8**2)+1/(16**2)+1/(32**2)+1/(64**2))*101)
# T=101 NQ=1100 lengthT>=1178
# T=101 NQ=900 lengthT>=430
# T=64 NQ=1100 lengthT>=829
NQ_cut = {
    # 900: 430,
    1100: 829,
}

NQ = 1100


def _func(item):
    return item[0] + " " + str(item[1][0]) + " " + str(item[1][1]) + "\n"


def handle(path: str, do_filter: bool = True):
    files = os.listdir(path)

    seiz = {}
    noseiz = {}

    for file in tqdm(files):
        with h5py.File(os.path.join(path, file), "r") as hf:
            signal = hf["signal"][()]
            y = hf["label"][()]

        ls = signal.shape[-1]
        if do_filter and ls < NQ_cut[NQ]:
            continue

        # 将seiz和noseiz的文件分开为2个txt文件
        if (y == 0).all():
            noseiz[file] = (len(y) / 200, ls)
        else:
            seiz[file] = (len(y) / 200, ls)

    # 将文件按长度从大到小排列
    seiz = dict(sorted(seiz.items(), key=lambda item: item[1][0]))
    noseiz = dict(sorted(noseiz.items(), key=lambda item: item[1][0]))

    seiz = list(map(_func, seiz.items()))
    noseiz = list(map(_func, noseiz.items()))

    return seiz, noseiz


def main(path):
    for task in tasks:
        if task != "eval":
            seiz, noseiz = handle(
                os.path.join(path, "stft_amp_w_scale_w_crop", task), True
            )
            with open(
                os.path.join(path, "FS_" + task + "_NQ" + str(NQ) + "_seiz.txt"), "w"
            ) as txt:
                txt.writelines(seiz)
            with open(
                os.path.join(path, "FS_" + task + "_NQ" + str(NQ) + "_noseiz.txt"), "w"
            ) as txt:
                txt.writelines(noseiz)
        else:
            seiz, noseiz = handle(
                os.path.join(path, "stft_amp_w_scale_w_crop", task), False
            )
            with open(
                os.path.join(path, "S_" + task + "_NQ" + str(NQ) + "_seiz.txt"), "w"
            ) as txt:
                txt.writelines(seiz)
            with open(
                os.path.join(path, "S_" + task + "_NQ" + str(NQ) + "_noseiz.txt"), "w"
            ) as txt:
                txt.writelines(noseiz)


if __name__ == "__main__":
    main(sys.argv[1])
