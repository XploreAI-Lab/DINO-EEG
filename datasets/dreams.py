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

from util import box_ops as bop
from .constants import SEGMENT_LEN, N_CLASSES_DREAMS


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
        if label == N_CLASSES_DREAMS:
            continue

        # 在coco中，label也是从1开始算的，如coco一共90个类，因此label为1~90；tuev的标签也是从1开始算
        labels.append(torch.tensor(label))

        # 归一化，将事件长度放缩到0~1之间
        box = torch.Tensor([left / float(end), right / float(end)])
        bboxes.append(box)

    # assert len(bboxes) > 0
    # [N（事件数量）, 2（onset和offset（归一化的[0,1]））]
    # boxes = torch.stack(bboxes)

    if len(bboxes) > 0:
        # [N（事件数量）, 2（center和width）]
        boxes = torch.stack(bboxes)
        # [N（事件数量）, 2（onset和offset（真实坐标））]
        boxes_eval = boxes * SEGMENT_LEN
        # [N（事件数量）, 4（center和width），y轴直接为整个y]
        boxes = bop.box_xyxy_to_cxcywh(bop.box_cxw_to_xyxy(bop.box_x0x1_to_cxw(boxes)))
    else:
        boxes = torch.Tensor()

    return {
        "boxes": boxes,
        "labels": torch.as_tensor(labels, dtype=torch.int),
        "boxes_eval": boxes_eval,
    }


class DREAMSDataSet(Dataset):
    def __init__(self, data_dir, txt_dir, task):
        super(DREAMSDataSet, self).__init__()
    