# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    根据采样点位置从所有点的value中拿出对应的value，并且和对应的注意力权重进行weighted sum
    -----
    value: 通过线性变换将输入的特征图变换成value  [B, K, C] -> [B, K, 8, 32]
    value_spatial_shapes: 4个flatten后特征图的shape [4, 2]
    sampling_locations: 采样点 [bs, Len_q, n_head, n_levels, n_points, 2] [B, K, 8, 4, 4, 2]
    attention_weights: 注意力权重 [bs, Len_q, 8, 4, 4] [B, K , ...]
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    # 把value分割到各个特征层上得到对应的 list value 即把拼接而成的value给分开，长度为4的tuple，里面是(B, H*W, 8, 32) value被拆分为8个头
    # 这里是吧拼接的K拆开
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # 采样点坐标从[0,1] -> [-1, 1]  F.grid_sample要求采样坐标归一化到[-1, 1] [B, K, n_head, n_levels, n_points, 2]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # value_l_ [N_*M_, D_, H_, W_] [B*8, 32, H, W]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)  # 得到每个特征层的value list
        # 就是根据给定的位置进行采样
        """
        sampling_grids [B, K, n_head:8, n_levels:4, n_points:4, 2] 
        sampling_grids[:, :, :, lid_]代表第lid_个level特征层(B, K, 8, 4, 2)
        (N_, Lq_, M_, P_, 2) -> (N_, M_, Lq_, P_, 2) -> sampling_grid_l_:[N_*M_, Lq_, P_, 2]
        sampling_grid_l_ [N_*M_, Lq_, P_, 2] [B*8, K, 4, 2]
        """
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)  # 得到每个特征层的采样点 list
        # N_*M_, D_, Lq_, P_  采样算法  根据每个特征层采样点到每个特征层的value进行采样  非采样点用0填充
        """
        对于每个head的每个特征层，使用采样点sampling_grid_l_里面的点坐标去value_l_里面对应位置的四邻域使用双线性插值采样
        F.grid_sample对于input [N, C, H_in, W_in]，grid [N, H_out, W_out, 2]输出形状为 [N, C, H_out, W_out]
        sampling_value_l_(N_*M_, D_, Lq_, P_ ) [B*8, 32, K, 4] 采样算法  根据每个特征层采样点到每个特征层的value进行采样  非采样点用0填充
        """
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_) [B*8, 1, K, 4*4]
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # 注意力权重 和 采样后的value 进行 weighted sum
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    # [B, K, C]
    return output.transpose(1, 2).contiguous()
