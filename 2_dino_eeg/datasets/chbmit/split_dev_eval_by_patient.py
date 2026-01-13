# 按受试者划分验证集和测试集，其中chb21和chb1是同一个受试者
# 按BIOT同款处理方式，20，22做验证集，23，24做测试集，其他做训练集
import argparse
import os


def main(path, save_dir):
    files = os.listdir(path)
    X = list(map(lambda x: x + "\n", files))

    # e.g. chb01_18_P7-O1.h5
    Patients = set(map(lambda x: x.split("_", 1)[0], files))

    dev_patients = {"chb20", "chb22"}
    eval_patients = {"chb23", "chb24"}
    train_patients = Patients - dev_patients - eval_patients

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
    args = parser.parse_args()
    main(args.data_dir, args.save_dir)
