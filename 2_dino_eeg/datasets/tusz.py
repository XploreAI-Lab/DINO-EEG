import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)
from torch.utils.data.distributed import DistributedSampler

from util import box_ops as bop
from .constants import SEGMENT_LEN, N_CLASSES_TUSZ, FREQUENCY
from pandas import read_csv, concat


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

        # label中的N_CLASSES_TUSZ代表背景类，不应该存在于gt标签中；去寻找下一个事件
        if label == N_CLASSES_TUSZ:
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


# TUSZ本来就是跨受试者的
class TUSZDataSet(Dataset):
    def __init__(self, task, args, stage="full"):
        """
        Args:
            task: "train", "dev", "eval"
            args: 训练参数
            stage: "seizure_only" - 仅癫痫数据（第一阶段）
                   "full" - 完整数据（第二阶段，包含下采样的非癫痫数据）
        """
        super(TUSZDataSet, self).__init__()
        self.data_dir = os.path.join(args.data_dir, task)
        self.txt_dir = args.tusz_txt_dir
        self.nq = str(int(args.num_queries) + int(args.dn_number) * 2)  # 1100
        self.stage = stage

        if args.tusz_label_dir is not None:
            self.label_dir = os.path.join(args.tusz_label_dir, task)
        else:
            self.label_dir = None

        # 训练集和验证集需要下采样，测试集不能下采样
        if task == "train":
            # train文件夹的训练文件
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

        # 根据训练阶段选择数据
        if stage == "seizure_only":
            # 第一阶段：仅使用癫痫数据
            self.data = df_s
        else:
            # 第二阶段：使用完整数据（含下采样的非癫痫数据）
            self.data = self._downsample(
                df_s,
                df_ns,
                seed=args.downsample_seed,
                ratio=args.tusz_downsample_times if task != "eval" else 0,
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

        if self.label_dir is not None:
            with h5py.File(os.path.join(self.label_dir, current_data), "r") as hf:
                label = hf["label"][()]

        # [C(1), F, T] [1, 101, T]
        x = torch.as_tensor(signal).float()

        # [T*F, ]
        y = create_annotations(label)
        # 因为测试时不打乱样本顺序，index就是逐个遍历range(n)（即整个验证集/测试集）所得
        # y["image_id"] = torch.tensor(index)
        y["image_id"] = current_data
        y["orig_size"] = torch.tensor(label.shape[0] / FREQUENCY)
        y["patient"] = current_data.split("_", maxsplit=1)[0]
        # y["ch_name"] = current_data.split("_")[-1].split(".h5")[0]

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
        combined_df = combined_df.sort_values(by="sec", ascending=False)
        combined_df = combined_df.reset_index(drop=True)

        return combined_df


def build_tusz_dataloader(collate_fn, args, stage="full"):
    """
    构建TUSZ数据加载器
    
    Args:
        collate_fn: 数据整理函数
        args: 训练参数
        stage: "seizure_only" - 第一阶段（仅癫痫数据）
               "full" - 第二阶段（完整数据）
    """
    dataloaders = {}
    datasets = {}

    # 在评估模式下，只创建测试数据集
    if getattr(args, 'eval', False):
        dataset_test = TUSZDataSet("eval", args, stage="full")  # 测试集始终使用完整数据
    else:
        dataset_train = TUSZDataSet("train", args, stage=stage)
        dataset_validation = TUSZDataSet("dev", args, stage="full")  # 验证集始终使用完整数据
        dataset_test = TUSZDataSet("eval", args, stage="full")  # 测试集始终使用完整数据

    # 分布式训练支持
    if getattr(args, 'eval', False):
        # 评估模式下只创建测试采样器
        if args.distributed:
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_test = SequentialSampler(dataset_test)
    else:
        # 训练模式下创建所有采样器
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            sampler_validation = DistributedSampler(dataset_validation, shuffle=False)
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
            batch_sampler_train = None  # 使用DistributedSampler时不需要BatchSampler
        else:
            sampler_train = RandomSampler(dataset_train)  # 单卡训练时使用随机采样
            sampler_validation = SequentialSampler(dataset_validation)
            sampler_test = SequentialSampler(dataset_test)
            batch_sampler_train = BatchSampler(
                sampler_train, args.train_batch_size, drop_last=True
            )

    # 创建测试数据加载器（训练和评估模式都需要）
    dataloader_test = DataLoader(
        dataset_test,
        args.test_batch_size,
        sampler=sampler_test,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,  # 避免两阶段切换时worker冲突
    )
    
    if not getattr(args, 'eval', False):
        # 训练模式下创建训练和验证数据加载器
        dataloader_validation = DataLoader(
            dataset_validation,
            args.test_batch_size,
            sampler=sampler_validation,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=False,  # 避免两阶段切换时worker冲突
        )
        
        # 训练数据加载器配置
        if args.distributed:
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args.train_batch_size,
                sampler=sampler_train,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=False,  # 关闭pin_memory节省显存
                drop_last=True,
                persistent_workers=False,  # 避免两阶段切换时worker冲突
            )
        else:
            dataloader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=False,
                persistent_workers=False,  # 避免两阶段切换时worker冲突
            )

        dataloaders["train"] = dataloader_train
        datasets["train"] = dataset_train
        dataloaders["validation"] = dataloader_validation
        datasets["validation"] = dataset_validation
    dataloaders["test"] = dataloader_test
    datasets["test"] = dataset_test

    return dataloaders, datasets
