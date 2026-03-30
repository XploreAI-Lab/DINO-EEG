# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
import setproctitle

import numpy as np
import torch

import util.misc as utils
from datasets import build_dataloader
from engine import evaluate, train_one_epoch
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from util.early_stopping import TwoStageTrainer


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument(
        "--dataset",
        default="tusz",
        type=str,
        choices=["tusz", "neonatal", "chbmit"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dir to sliced train EEG signals (.h5 files).",
    )
    parser.add_argument(
        "--downsample_seed",
        type=int,
        default=42,
        help="数据下采样随机种子，保证训练数据相同所以不随模型随机种子改变",
    )
    # ---------------------------------tuev数据集相关---------------------------------
    parser.add_argument(
        "--tuev_cross_subject", action="store_true", help="tuev是否跨受试者"
    )
    parser.add_argument(
        "--tuev_trainSet_txt_path",
        type=str,
        default=None,
        help="tuev跨受试者训练集txt文件",
    )
    parser.add_argument(
        "--tuev_devSet_txt_path",
        type=str,
        default=None,
        help="tuev跨受试者验证集txt文件",
    )
    parser.add_argument(
        "--tuev_label_dir",
        type=str,
        default=None,
        help="tuev额外label存储目录",
    )

    # ---------------------------------tusz数据集相关---------------------------------
    parser.add_argument(
        "--tusz_txt_dir",
        type=str,
        default=None,
        help="tusz txt文件目录",
    )
    parser.add_argument(
        "--tusz_label_dir",
        type=str,
        default=None,
        help="tusz额外label存储目录",
    )
    parser.add_argument(
        "--tusz_downsample_times",
        type=float,
        default=1,
        help="tusz多数类下采样倍率，指为少数类的几倍",
    )
    # parser.add_argument(
    #     "--patient",
    #     type=str,
    #     default=None,
    #     help="按病人评估，病人名称",
    # )

    # ---------------------------------tuar数据集相关---------------------------------
    parser.add_argument(
        "--tuar_txt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tuar_downsample_ratio",
        type=float,
        default=0.2,
        help="tuar下采样倍率，0~1之间，指使用全部数据的多少[共10274个30s切片]",
    )
    # ---------------------------------chbmit数据集相关---------------------------------
    parser.add_argument(
        "--chbmit_txt_dir",
        type=str,
        default=None,
        help="chbmit的txt文件目录",
    )
    parser.add_argument(
        "--chbmit_downsample_times",
        type=float,
        default=1,
        help="chbmit多数类下采样倍率，指为少数类的几倍",
    )
    # ---------------------------------neonatal数据集相关---------------------------------
    parser.add_argument(
        "--neonatal_txt_dir",
        type=str,
        default=None,
        help="neonatal的txt文件目录",
    )

    # training parameters
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_validation", action="store_true", 
                       help="使用validation数据集进行评估，而不是test数据集")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # -------------------------distributed training parameters-------------------------
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", "--local-rank", type=int, default=0, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")

    # -------------------------两阶段训练参数-------------------------
    parser.add_argument("--two_stage_training", action="store_true", 
                       help="启用两阶段训练：先仅癫痫数据，再全部数据")
    parser.add_argument("--stage1_epochs", default=50, type=int,
                       help="第一阶段最大训练轮数")
    parser.add_argument("--stage2_epochs", default=100, type=int,
                       help="第二阶段最大训练轮数")
    parser.add_argument("--stage1_patience", default=10, type=int,
                       help="第一阶段早停耐心值")
    parser.add_argument("--stage2_patience", default=15, type=int,
                       help="第二阶段早停耐心值")
    parser.add_argument("--stage2_lr_factor", default=0.1, type=float,
                       help="第二阶段学习率缩放因子")
    parser.add_argument("--start_stage2_from_checkpoint", type=str, default=None,
                       help="从指定检查点开始第二阶段训练")
    parser.add_argument("--eval_both_sets", action="store_true",
                       help="评估时同时测试验证集和测试集")
    
    # -------------------------显存优化参数-------------------------
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                       help="梯度累积步数，用于模拟更大的batch size")

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def load_checkpoint(checkpoint_path, model_without_ddp, logger=None):
    """加载检查点文件，支持多种格式"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 记录检查点中的键
    if logger:
        logger.info(f"检查点文件中的键: {list(checkpoint.keys())}")
    
    # 尝试不同的键来加载模型状态
    state_dict = None
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # 如果没有找到标准键，假设整个文件就是状态字典
        state_dict = checkpoint
    
    # 加载模型状态
    load_result = model_without_ddp.load_state_dict(state_dict, strict=False)
    
    if logger:
        logger.info(f"模型加载结果: {load_result}")
    
    return checkpoint


def main(args):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    setproctitle.setproctitle("DINO")
    # ---------------------------------------基本参数准备------------------------------------------------
    # 初始化分布式训练
    utils.init_distributed_mode(args)

    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False

    # setup logger
    # 设置日志
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"),
        distributed_rank=args.rank,
        color=False,
        name="detr",
    )
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + " ".join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info("world size: {}".format(args.world_size))
    logger.info("rank: {}".format(args.rank))
    logger.info("local_rank: {}".format(args.local_rank))
    logger.info("args: " + str(args) + "\n")

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # 设置CUDA设备
    if args.distributed:
        # 分布式训练：使用local_rank设置设备
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.local_rank)
    else:
        # 单卡训练：使用指定设备
        device = torch.device(args.device)
        if device.type == 'cuda':
            # 如果设备字符串包含索引（如cuda:0），提取索引；否则使用0
            if ':' in args.device:
                device_id = int(args.device.split(':')[1])
            else:
                device_id = 0
            torch.cuda.set_device(device_id)


    # fix the seed for reproducibility  固定随机数种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ----------------------------------------------模型部分----------------------------------
    # build model 搭建模型
    # model 整体模型
    # criterion 损失函数
    # postprocessors bbox后处理  调用coco api
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    # DDP分布式训练
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module

    # 打印模型参数
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of params:" + str(n_parameters))
    logger.info(
        "params:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            indent=2,
        )
    )

    # 分别设置学习率
    param_dicts = get_param_dict(args, model_without_ddp)

    # 优化器和学习率调整策略
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    # -------------------------------------------数据集部分---------------------------------------------
    # 创建训练和验证数据集，定义数据集采样策略
    logger.info("Building dataset...")
    
    # 如果启用两阶段训练，则需要分别构建两个阶段的数据集
    if args.two_stage_training:
        # 第一阶段：仅癫痫数据
        dataloaders_stage1, _ = build_dataloader(utils.collate_fn, args, stage="seizure_only")
        # 第二阶段：完整数据
        dataloaders_stage2, _ = build_dataloader(utils.collate_fn, args, stage="full")
        
        # 如果从检查点开始第二阶段，直接使用第二阶段数据
        if args.start_stage2_from_checkpoint:
            dataloaders = dataloaders_stage2
            logger.info("直接使用第二阶段完整数据集")
        else:
            dataloaders = dataloaders_stage1  # 先使用第一阶段数据
            logger.info("使用第一阶段癫痫数据集")
    else:
        dataloaders, _ = build_dataloader(utils.collate_fn, args)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(dataloaders["train"]),
            epochs=args.epochs,
            pct_start=0.2,
        )
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_drop_list
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        # 使用新的加载函数
        load_checkpoint(args.frozen_weights, model_without_ddp.detr, logger)

    # ------------------------------------------------训练前准备--------------------------------------------
    # 设置实验数据保存路径
    output_dir = Path(args.output_dir)

    # 需要断点续训或者加载模型进行评估
    if args.resume:
        logger.info(f"尝试从检查点恢复: {args.resume}")
        
        # 使用新的加载函数
        checkpoint = load_checkpoint(args.resume, model_without_ddp, logger)

        # 评估也会resume，确保不是eval
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            logger.info("恢复优化器和学习率调度器状态")
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    # 是否使用预训练参数，指的是预训练的大模型，不是说训练好需要评估的模型
    if (not args.resume) and args.pretrain_model_path:
        logger.info(f"加载预训练模型: {args.pretrain_model_path}")
        # 使用新的加载函数
        checkpoint = load_checkpoint(args.pretrain_model_path, model_without_ddp, logger)
        
        # 处理需要忽略的参数
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        # 这里不需要再次加载，因为load_checkpoint已经完成了加载
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))

    # 是否需要验证
    if args.eval:
        os.environ["EVAL_FLAG"] = "TRUE"

        try:
            select_thresholds_index = (checkpoint["select_thresholds_index"]).numpy()
            print(select_thresholds_index)
        except:
            select_thresholds_index = None

        # 评估验证集或测试集
        eval_dataset = "validation" if args.eval_validation else "test"
        logger.info(f"开始评估{eval_dataset}集...")
        val_stats, _ = evaluate(
            model,
            criterion,
            postprocessors,
            dataloaders[eval_dataset],
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            select_thresholds_index=select_thresholds_index,
            args=args,
            logger=logger,
        )

        log_stats = {**{f"val_{k}": v for k, v in val_stats.items()}}
        
        logger.info("="*60)
        logger.info("评估结果:")
        logger.info(f"{eval_dataset}集 mAP: {val_stats.get('eval_bbox', 'N/A')}")
        logger.info(f"{eval_dataset}集 F1: {val_stats.get('eval_f1', 'N/A')}")
        logger.info("="*60)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # --------------------------------------------开始训练--------------------------------------------------
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    
    # 初始化两阶段训练器（如果启用）
    if args.two_stage_training:
        # 检查是否从指定检查点开始第二阶段
        if args.start_stage2_from_checkpoint:
            logger.info(f"从指定检查点开始第二阶段训练: {args.start_stage2_from_checkpoint}")
            
            # 加载指定的检查点
            checkpoint = load_checkpoint(args.start_stage2_from_checkpoint, model_without_ddp, logger)
            
            # 恢复优化器和学习率调度器状态
            if "optimizer" in checkpoint and "lr_scheduler" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logger.info("恢复优化器和学习率调度器状态")
            
            # 从检查点恢复epoch
            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"] + 1
                logger.info(f"从epoch {args.start_epoch}开始第二阶段训练")
            
            # 调整学习率为第二阶段
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.stage2_lr_factor
            logger.info(f"学习率调整为: {optimizer.param_groups[0]['lr']}")
            
            # 初始化两阶段训练器，直接设置为第二阶段
            two_stage_trainer = TwoStageTrainer(
                model=model_without_ddp,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                stage1_patience=args.stage1_patience,
                stage2_patience=args.stage2_patience,
                output_dir=args.output_dir
            )
            
            # 强制设置为第二阶段
            two_stage_trainer.stage1_completed = True
            two_stage_trainer.current_stage = 2
            
            # 计算剩余训练轮数
            total_epochs = args.start_epoch + args.stage2_epochs
            logger.info(f"直接开始第二阶段，预计训练到epoch {total_epochs}")
            
        else:
            # 正常的两阶段训练，从第一阶段开始
            two_stage_trainer = TwoStageTrainer(
                model=model_without_ddp,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                stage1_patience=args.stage1_patience,
                stage2_patience=args.stage2_patience,
                output_dir=args.output_dir
            )
            total_epochs = args.stage1_epochs + args.stage2_epochs
            logger.info(f"两阶段训练模式：第一阶段{args.stage1_epochs}轮，第二阶段{args.stage2_epochs}轮")
        
        logger.info(f"注意：配置文件中的epochs={args.epochs}被两阶段训练覆盖为{total_epochs}")
        
        # 调整学习率调度器参数以适应两阶段训练
        if hasattr(args, 'lr_drop') and args.lr_drop >= total_epochs:
            logger.info(f"警告：配置文件中lr_drop={args.lr_drop}大于总训练轮数{total_epochs}，学习率不会下降")
    else:
        total_epochs = args.epochs
    
    for epoch in range(args.start_epoch, total_epochs):
        # 记录训练时间
        epoch_start_time = time.time()
        
        # 两阶段训练逻辑
        if args.two_stage_training:
            # 检查是否需要切换到第二阶段（达到第一阶段最大轮数）
            if epoch >= args.stage1_epochs and not two_stage_trainer.stage1_completed:
                logger.info(f"第一阶段达到最大轮数 {args.stage1_epochs}，切换到第二阶段")
                
                # 清理旧的DataLoader worker状态
                if hasattr(dataloaders["train"], '_iterator'):
                    del dataloaders["train"]._iterator
                if hasattr(dataloaders["validation"], '_iterator'):
                    del dataloaders["validation"]._iterator
                
                # 强制垃圾回收和清理GPU缓存
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # 加载第一阶段最佳权重并切换到第二阶段
                two_stage_trainer.complete_stage1()
                
                # 重新创建第二阶段数据集和DataLoader（确保DistributedSampler一致性）
                logger.info("重新创建第二阶段DataLoader以确保分布式一致性")
                dataloaders_stage2, _ = build_dataloader(utils.collate_fn, args, stage="full")
                dataloaders = dataloaders_stage2
                
                # 调整学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.stage2_lr_factor
                logger.info(f"第一阶段完成，切换到第二阶段，学习率调整为: {optimizer.param_groups[0]['lr']}")
                
                # 如果使用OneCycleLR，需要重建调度器
                if args.onecyclelr:
                    remaining_epochs = total_epochs - epoch
                    logger.info(f"重建OneCycleLR调度器，剩余训练轮数: {remaining_epochs}")
                    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=optimizer.param_groups[0]['lr'],
                        steps_per_epoch=len(dataloaders["train"]),
                        epochs=remaining_epochs,
                        pct_start=0.2,
                    )
                
                # 分布式训练：确保所有进程在阶段切换后同步
                if args.distributed:
                    torch.distributed.barrier()
                    logger.info(f"Rank {args.rank}: 所有进程已同步完成第一阶段到第二阶段的切换")
            
            # 记录当前阶段信息
            stage_info = two_stage_trainer.get_stage_info()
            current_stage = stage_info['current_stage']
            logger.info(f"训练阶段: {current_stage}, Epoch: {epoch}/{total_epochs}")
        
        # 分布式训练：设置epoch以确保每个epoch的数据打乱不同
        if args.distributed:
            if hasattr(dataloaders["train"].sampler, 'set_epoch'):
                dataloaders["train"].sampler.set_epoch(epoch)
                logger.debug(f"设置训练集sampler epoch: {epoch}")
            if hasattr(dataloaders["validation"].sampler, 'set_epoch'):
                dataloaders["validation"].sampler.set_epoch(epoch)
                logger.debug(f"设置验证集sampler epoch: {epoch}")
            
            # 确保所有进程在开始训练前同步
            torch.distributed.barrier()
            logger.debug(f"Rank {args.rank}: 所有进程已同步，开始epoch {epoch}训练")
        # 训练一个epoch
        train_stats = train_one_epoch(
            model,
            criterion,
            dataloaders["train"],
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
            ema_m=ema_m,
        )

        # 调整学习率
        if not args.onecyclelr:
            lr_scheduler.step()
        
        # 分布式训练：在保存模型前同步所有进程
        if args.distributed:
            torch.distributed.barrier()
            logger.debug(f"Rank {args.rank}: 所有进程已同步，准备保存模型")
        
        # 每个 epoch 保存一个模型文件（只在主进程中保存）
        if args.output_dir and epoch % 10 == 0 and utils.is_main_process():
            checkpoint_paths = [
                output_dir / f"checkpoint{epoch:04}.pth",     # 独立保存每个 epoch
                output_dir / "checkpoint.pth"                 # 最新的普通 checkpoint（始终覆盖）
            ]
            for checkpoint_path in checkpoint_paths:
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                utils.save_on_master(weights, checkpoint_path)

        # 验证集评估
        logger.info(f"Epoch {epoch}: 开始验证集评估...")
        val_stats, select_thresholds_index = evaluate(
            model,
            criterion,
            postprocessors,
            dataloaders["validation"],
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            select_thresholds_index=None,
            args=args,
            logger=(logger if args.save_log else None),
        )
        map_regular = val_stats["eval_bbox"]
        f1_regular = val_stats.get("eval_f1", 0.0)  # 获取F1分数
        
        # 记录验证集性能
        logger.info(f"Epoch {epoch} 验证集 mAP: {map_regular:.4f}, F1: {f1_regular:.4f}")
        
        # 使用F1分数进行模型选优
        _isbest = best_map_holder.update(f1_regular, epoch, is_ema=False)
        
        # 两阶段训练的早停逻辑
        should_stop = False
        if args.two_stage_training:
            if current_stage == 1:
                # 第一阶段早停检查
                should_stop = two_stage_trainer.should_stop_stage1(f1_regular, epoch)
                if should_stop:
                    logger.info(f"第一阶段早停，epoch: {epoch}, 最佳F1: {f1_regular:.4f}")
                    
                    # 清理旧的DataLoader worker状态
                    if hasattr(dataloaders["train"], '_iterator'):
                        del dataloaders["train"]._iterator
                    if hasattr(dataloaders["validation"], '_iterator'):
                        del dataloaders["validation"]._iterator
                    
                    # 强制垃圾回收和清理GPU缓存
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # 强制进入第二阶段，加载第一阶段最佳权重
                    two_stage_trainer.complete_stage1()
                    
                    # 重新创建第二阶段DataLoader（确保DistributedSampler一致性）
                    logger.info("早停触发：重新创建第二阶段DataLoader以确保分布式一致性")
                    dataloaders_stage2, _ = build_dataloader(utils.collate_fn, args, stage="full")
                    dataloaders = dataloaders_stage2
                    
                    # 调整学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.stage2_lr_factor
                    logger.info(f"第一阶段早停，切换到第二阶段，学习率调整为: {optimizer.param_groups[0]['lr']}")
                    
                    # 如果使用OneCycleLR，需要重建调度器
                    if args.onecyclelr:
                        remaining_epochs = total_epochs - epoch
                        logger.info(f"早停触发：重建OneCycleLR调度器，剩余训练轮数: {remaining_epochs}")
                        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=optimizer.param_groups[0]['lr'],
                            steps_per_epoch=len(dataloaders["train"]),
                            epochs=remaining_epochs,
                            pct_start=0.2,
                        )
                    
                    # 分布式训练：确保所有进程在早停切换后同步
                    if args.distributed:
                        torch.distributed.barrier()
                        logger.info(f"Rank {args.rank}: 所有进程已同步完成早停触发的阶段切换")
                    
                    should_stop = False  # 继续训练第二阶段
            else:
                # 第二阶段早停检查
                should_stop = two_stage_trainer.should_stop_stage2(f1_regular, epoch)
                if should_stop:
                    logger.info(f"第二阶段早停，epoch: {epoch}, 最佳F1: {f1_regular:.4f}")
        
        if _isbest and utils.is_main_process():
            checkpoint_path = output_dir / "checkpoint_best_regular.pth"
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "select_thresholds_index": torch.asarray(select_thresholds_index),
                    "args": args,
                },
                checkpoint_path,
            )
        
        # 如果早停，则退出训练循环
        if should_stop:
            logger.info("早停触发，结束训练")
            break
        # 保存日志
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},  # 验证集结果
        }
        
        # 添加F1分数的特别记录
        log_stats["val_f1_score"] = f1_regular
        
        log_stats.update(best_map_holder.summary())

        ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
        log_stats.update(ep_paras)
        
        # 添加两阶段训练信息到日志
        if args.two_stage_training:
            stage_info = two_stage_trainer.get_stage_info()
            log_stats.update({
                "training_stage": stage_info['current_stage'],
                "stage1_completed": stage_info['stage1_completed'],
                "stage1_best_score": stage_info['stage1_best_score'],
                "stage2_best_score": stage_info['stage2_best_score']
            })
        try:
            log_stats.update({"now_time": str(datetime.datetime.now())})
        except:
            pass

        # 计算运行时间
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats["epoch_time"] = epoch_time_str
        # 保存模型
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 计算总运行时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    # ---------------------------------------------训练结束---------------------------------------
    # remove the copied files.
    copyfilelist = vars(args).get("copyfilelist")
    if copyfilelist and args.local_rank == 0:
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            utils.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)