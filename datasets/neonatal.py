import os
import h5py
import numpy as np
from pandas import concat, read_csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)

from util import box_ops as bop
from .constants import FREQUENCY


def create_annotations(y):
    """
    把[L,]的标签转换成包含分类和cxw的标签的字典
    """
    i = 0
    end = len(y)
    bboxes = []
    labels = []
    while i < end:
        left = i
        label = y[i]

        # 直到label发生变换之前都是同一个事件
        while i < end and y[i] == label:
            i += 1
        right = i

        if label == 0:
            continue

        # 在coco中，label也是从1开始算的，如coco一共90个类，因此label为1~90；tuev的标签也是从1开始算
        labels.append(torch.tensor(label))

        # 归一化，将事件长度放缩到0~1之间
        box = torch.Tensor([left / float(end), right / float(end)])
        bboxes.append(box)

    # assert len(bboxes) > 0

    if len(bboxes) > 0:
        # [N（事件数量）, 2（onset和offset（归一化的[0,1]））]
        boxes = torch.stack(bboxes)
        # [N（事件数量）, 2（onset和offset（真实坐标））]
        boxes_eval = boxes * (end / FREQUENCY)
        # [N（事件数量）, 4（center和width），y轴直接为整个y]
        boxes = bop.box_xyxy_to_cxcywh(bop.box_cxw_to_xyxy(bop.box_x0x1_to_cxw(boxes)))
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        boxes_eval = torch.empty((0, 4), dtype=torch.float32)

    return {
        "boxes": boxes,
        "labels": torch.as_tensor(labels, dtype=torch.int),
        "boxes_eval": boxes_eval,
    }


class NEONATALDataSet(Dataset):
    def __init__(self, task, args):
        super(NEONATALDataSet, self).__init__()

        self.data_dir = os.path.join(args.data_dir, task)
        self.txt_dir = args.neonatal_txt_dir
        self.nq = str(int(args.num_queries) + int(args.dn_number) * 2)  # 1100

        if task == "train":
            path_s = os.path.join(self.txt_dir, "FS_train_NQ" + self.nq + "_seiz.txt")
            path_ns = os.path.join(
                self.txt_dir, "FS_train_NQ" + self.nq + "_noseiz.txt"
            )
        elif task == "dev":
            path_s = os.path.join(self.txt_dir, "FS_dev_NQ" + self.nq + "_seiz.txt")
            path_ns = os.path.join(self.txt_dir, "FS_dev_NQ" + self.nq + "_noseiz.txt")
        elif task == "eval":
            path_s = os.path.join(self.txt_dir, "S_eval_NQ" + self.nq + "_seiz.txt")
            path_ns = os.path.join(self.txt_dir, "S_eval_NQ" + self.nq + "_noseiz.txt")
        else:
            raise ValueError("非法的task")

        df_s = read_csv(
            path_s, sep=" ", header=None, names=["fn", "sec", "T"], index_col=None
        )
        df_ns = read_csv(
            path_ns, sep=" ", header=None, names=["fn", "sec", "T"], index_col=None
        )

        self.data = self._downsample(
            df_s,
            df_ns,
            seed=args.downsample_seed,
            ratio=0,  # Neonatal不需要下采样
        )

        self.data_fn_c = self.data.columns.get_loc("fn")
        self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        current_data: str = self.data.iloc[index, self.data_fn_c]
        with h5py.File(os.path.join(self.data_dir, current_data), "r") as hf:
            signal = hf["signal"][()]
            label = hf["label"][()]

        # [C(1), F, T] [1, 101, T]
        x = torch.as_tensor(signal).float()

        # [T*F, ]
        y = create_annotations(label)
        # 直接使用index作为image_id（切片id）
        # 因为测试时不打乱样本顺序，index就是逐个遍历range(n)（即整个验证集/测试集）所得
        # y["image_id"] = torch.tensor(index)
        y["image_id"] = current_data
        y["orig_size"] = torch.tensor(label.shape[0] / FREQUENCY)
        y["patient"] = current_data.split("_", maxsplit=1)[0]

        return x, y

    def _downsample(self, df_s, df_ns, seed, ratio=1.0):
        if ratio == 0:  # 测试集和验证集不需要下采样
            filtered_df_ns = df_ns
        else:
            # 统计发作文件的最长、最短文件时长、发作文件数量
            _max_l, _min_l, _num_s = df_s["sec"].max(), df_s["sec"].min(), df_s.shape[0]
            # 仅在发作文件的长度中下采样
            filtered_df_ns = df_ns[(df_ns["sec"] >= _min_l) & (df_ns["sec"] <= _max_l)]
            # 进行下采样后的未发作文件
            assert int(_num_s * ratio) <= filtered_df_ns.shape[0]
            filtered_df_ns = filtered_df_ns.sample(
                n=int(_num_s * ratio), random_state=seed
            )
        # 合并发作和未发作文件
        combined_df = concat(
            [df_s, filtered_df_ns],
            names=["fn", "sec", "T"],
            ignore_index=True,
        )
        # 重新排序
        combined_df = combined_df.sort_values(by="sec")
        combined_df = combined_df.reset_index(drop=True)

        return combined_df


def build_neonatal_dataloader(collate_fn, args):
    dataloaders = {}
    datasets = {}

    dataset_train = NEONATALDataSet("train", args)
    dataset_validation = NEONATALDataSet("dev", args)
    dataset_test = NEONATALDataSet("eval", args)

    sampler_train = SequentialSampler(dataset_train)
    sampler_validation = SequentialSampler(dataset_validation)
    sampler_test = SequentialSampler(dataset_test)
    batch_sampler_train = BatchSampler(
        sampler_train, args.train_batch_size, drop_last=True
    )

    dataloader_validation = DataLoader(
        dataset_validation,
        args.test_batch_size,
        sampler=sampler_validation,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dataloader_test = DataLoader(
        dataset_test,
        args.test_batch_size,
        sampler=sampler_test,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    dataloaders["train"] = dataloader_train
    datasets["train"] = dataset_train
    dataloaders["validation"] = dataloader_validation
    datasets["validation"] = dataset_validation
    dataloaders["test"] = dataloader_test
    datasets["test"] = dataset_test

    return dataloaders, datasets
