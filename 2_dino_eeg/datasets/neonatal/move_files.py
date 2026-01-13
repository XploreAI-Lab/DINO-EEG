import os
import sys
import shutil

from tqdm import tqdm


def main(txt_dir, source_dir, dest_dir):
    # 打开文本文件
    with open(os.path.join(txt_dir, "trainSet.txt"), "r") as txt:
        train_files = set(map(lambda x: x.strip("\n"), txt.readlines()))
    with open(os.path.join(txt_dir, "devSet.txt"), "r") as txt:
        dev_files = set(map(lambda x: x.strip("\n"), txt.readlines()))
    with open(os.path.join(txt_dir, "evalSet.txt"), "r") as txt:
        eval_files = set(map(lambda x: x.strip("\n"), txt.readlines()))

    files = os.listdir(source_dir)

    for file in tqdm(files):
        if file in train_files:
            shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, "train"))
        elif file in dev_files:
            shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, "dev"))
        elif file in eval_files:
            shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, "eval"))
        else:
            if not '.txt' in file:
                raise ValueError(file)
            continue


if __name__ == "__main__":
    main(*sys.argv[1:])
