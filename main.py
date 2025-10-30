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
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")


    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


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

    device = torch.device(args.device)
    torch.cuda.set_device(device)

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
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
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
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    # ------------------------------------------------训练前准备--------------------------------------------
    # 设置实验数据保存路径
    output_dir = Path(args.output_dir)
    # 是否断点继续训练
    # if (
    #     os.path.exists(os.path.join(args.output_dir, "checkpoint.pth"))
    #     and not args.eval
    #     and not args.resume
    # ):
    #     args.resume = os.path.join(args.output_dir, "checkpoint.pth")

    # 需要断点续训或者加载模型进行评估
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            assert os.path.exists(args.resume)
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

        # 评估也会resume，确保不是eval
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            print(checkpoint["lr_scheduler"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    # 是否使用预训练参数，指的是预训练的大模型，不是说训练好需要评估的模型
    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
        from collections import OrderedDict

        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        _tmp_st = OrderedDict(
            {
                k: v
                for k, v in utils.clean_state_dict(checkpoint).items()
                if check_keep(k, _ignorekeywordlist)
            }
        )
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    # 是否需要验证
    if args.eval:
        os.environ["EVAL_FLAG"] = "TRUE"

        try:
            select_thresholds_index = (checkpoint["select_thresholds_index"]).numpy()
            print(select_thresholds_index)
        except:
            select_thresholds_index = None

        test_stats, _ = evaluate(
            model,
            criterion,
            postprocessors,
            dataloaders["test"],
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            select_thresholds_index=select_thresholds_index,
            args=args,
            logger=logger,
        )

        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # --------------------------------------------开始训练--------------------------------------------------
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):
        # 记录训练时间
        epoch_start_time = time.time()
        # 分布式训练
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
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
        # 保存模型
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
        # 调整学习率
        if not args.onecyclelr:
            lr_scheduler.step()
        # 保存模型
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 'lr_drop' epochs
            if (epoch + 1) % args.lr_drop == 0 or (
                epoch + 1
            ) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                utils.save_on_master(weights, checkpoint_path)

        # 验证
        test_stats, select_thresholds_index = evaluate(
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
        map_regular = test_stats["eval_bbox"]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
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
        # 保存日志
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        log_stats.update(best_map_holder.summary())

        ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
        log_stats.update(ep_paras)
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
