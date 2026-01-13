# 针对新的分类数量，制作新的label数组单独保存。不直接加在原始数据后面
import argparse
import os
import h5py

from tqdm import tqdm


# 读取切片后的文件，补充二|N分类的label
def main(data_dir, save_dir):
    files = os.listdir(data_dir)
    for h5_fn in tqdm(files):
        with h5py.File(os.path.join(data_dir, h5_fn), "r") as hf:
            # [L,]
            label = hf["label"][()]

        # ----------随着label合并策略的不同，每次都需要手动重写具体的做法----------

        # label[label != 7] = 1 # TUEV
        
        label[label != 5] = 1 # TUSZ

        # ---------------------------------------------------------------------

        with h5py.File(os.path.join(save_dir, h5_fn), "w") as hf:
            hf.create_dataset("label", data=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
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
    main(args.data_dir, args.save_dir)
