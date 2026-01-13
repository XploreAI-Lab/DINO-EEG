# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os

import torch
from torch import nn
from typing import Dict, List

from util.misc import NestedTensor, clean_state_dict
from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer
from .modified_swin_transformer import build_swin_transformer_mst


class Joiner(nn.Sequential):
    """
    将backbone和position encoding集成在一个nn.Module里,使得向前过程中更方便的使用两者的功能
    Joiner是nn.Sequential的子类，通过初始化，使得self[0]是backbone，self[1]是position encoding。
    前向过程就是对backbone的每层输出都进行位置编码，最终返回backbone的输出及对应的位置编码结果。
    """

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """
        tensor_list: pad预处理之后的图像信息
        tensor_list.tensors: [bs, 3, 608, 810]预处理后的图片数据 对于小图片而言多余部分用0填充
        tensor_list.mask: [bs, 608, 810] 用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
        """
        # backbone的输出
        # 原图经过backbone前向传播
        # xs: '0' = NestedTensor: tensors[bs, 2048, 19, 26] + mask[bs, 19, 26]
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        # out: list{0: tensor=[bs,2048,19,26] + mask=[bs,19,26]}  经过backbone resnet50 block5输出的结果
        # pos: list{0: [bs,256,19,26]}  位置编码
        return out, pos


class _NestedTensorConv2d(nn.Module):
    def __init__(self, n_backbone_channel=3) -> None:
        super().__init__()
        # 1*1卷积，不会影响大小
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=n_backbone_channel, kernel_size=1
        )

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        x = self.conv(x)
        tensor_list.tensors = x
        return tensor_list


class _NestedTensorConv1d(nn.Module):
    def __init__(self, n_backbone_channel=3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=n_backbone_channel, kernel_size=1
        )

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        x = self.conv(x)
        tensor_list.tensors = x
        return tensor_list


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now
    """
    # 对backbone输出的特征图进行位置编码,用于后续Transformer部分
    position_embedding = build_position_encoding(args)
    # 是否需要训练backbone(即是否采用预训练backbone)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = nn.Sequential()
        # 将单通道数据变成3通道数据以匹配预训练的swin transformer
        backbone.add_module("_conv0", _NestedTensorConv2d(n_backbone_channel=3))

        swin_transformer = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=args.dilation,
            use_checkpoint=use_checkpoint,
        )

        # swin_transformer = build_swin_transformer_mst(
        #     args.backbone,
        #     pretrain_img_size=pretrain_img_size,
        #     out_indices=tuple(return_interm_indices),
        #     dilation=args.dilation,
        #     use_checkpoint=use_checkpoint,
        # )

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in swin_transformer.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        print("no finetune:", name)
                        break

        pretrained_dir = args.backbone_dir
        PTDICT = {
            "swin_T_224_1k": "swin_tiny_patch4_window7_224.pth",
            "swin_B_384_22k": "swin_base_patch4_window12_384.pth",
            "swin_L_384_22k": "swin_large_patch4_window12_384_22k.pth",
        }
        pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
        checkpoint = torch.load(pretrainedpath, map_location="cpu")["model"]
        # checkpoint = torch.load(pretrainedpath, map_location="cpu")

        from collections import OrderedDict

        def key_select_function(keyname):
            if "head" in keyname:
                return False
            if args.dilation and "layers.3" in keyname:
                return False
            return True

        _tmp_st = OrderedDict(
            {
                k: v
                for k, v in clean_state_dict(checkpoint).items()
                if key_select_function(k)
            }
        )
        _tmp_st_output = swin_transformer.load_state_dict(_tmp_st, strict=False)
        print(str(_tmp_st_output))
        bb_num_channels = swin_transformer.num_features[
            4 - len(return_interm_indices) :
        ]

        backbone.add_module("_swin", swin_transformer)

    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    # 将backbone和位置编码集合在一个model
    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model
