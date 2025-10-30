# 按受试者划分验证集和测试集，8:1:1，发作比例相同
import argparse
import os

from sklearn.model_selection import train_test_split


def main(path, seed, save_dir):
    files = os.listdir(path)
    # 对于Neonatal，每个eeg就是一个patient，39+22

    with open(os.path.join(path, "seiz.txt"), "r") as txt:
        Patients_seiz = set(map(lambda x: x.strip("\n"), txt.readlines()))

    with open(os.path.join(path, "noseiz.txt"), "r") as txt:
        Patients_noseiz = set(map(lambda x: x.strip("\n"), txt.readlines()))

    Patients = Patients_seiz.union(Patients_noseiz)
    stratify = [1] * len(Patients_seiz) + [0] * len(Patients_noseiz)

    print(len(Patients))

    # X_train, X_test, y_train, y_test
    train_patients, dev_eval_patients, _, stratify = train_test_split(
        list(Patients), stratify, test_size=0.3, random_state=seed, stratify=stratify
    )

    print(len(dev_eval_patients))

    dev_patients, eval_patients = train_test_split(
        dev_eval_patients, test_size=0.66, random_state=seed, stratify=stratify
    )

    print(len(dev_patients), len(eval_patients))

    train_patients = set(train_patients)
    dev_patients = set(dev_patients)
    eval_patients = set(eval_patients)
 
    X = list(map(lambda x: x + "\n", files))
    X_train = list(filter(lambda x: any([tp in x for tp in train_patients]), X))
    X_dev = list(filter(lambda x: any([dp in x for dp in dev_patients]), X))
    X_eval = list(filter(lambda x: any([ep in x for ep in eval_patients]), X))

    with open(os.path.join(save_dir, "trainSet.txt"), "w") as txt:
        txt.writelines(X_train)

    with open(os.path.join(save_dir, "devSet.txt"), "w") as txt:
        txt.writelines(X_dev)

    with open(os.path.join(save_dir, "evalSet.txt"), "w") as txt:
        txt.writelines(X_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default=None,
        help="预处理完整文件绝对路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        default=None,
        help="txt文件保存绝对路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    main(args.data_dir, args.seed, args.save_dir)
