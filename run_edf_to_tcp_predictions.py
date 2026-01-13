#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import types
from pathlib import Path
import importlib.util

def _load_module(module_path, name):
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _build_args(config_mod, overrides):
    ns = types.SimpleNamespace()
    for k, v in config_mod.__dict__.items():
        if not k.startswith("_"):
            setattr(ns, k, v)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", type=str, default=str(Path(__file__).resolve().parents[2] / "2_dinoeeg" / "dino_eeg_0827" / "config" / "DINO" / "DINO_4scale_swin_tusz_multi_gpu.py"))
    parser.add_argument("--output_root", type=str, default=str(Path.cwd() / f"edf_infer_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    edf_path = Path(args.edf_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    config_path = Path(args.config_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = output_root / "data"
    txt_root = output_root / "txt"
    eval_out = output_root / "eval_outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    txt_root.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    pre_mod_path = str(Path(__file__).resolve().parents[2] / "1_preprocess" / "pipe_EDF_to_result_tobeused.py")
    pre = _load_module(pre_mod_path, "pipe_EDF")
    meta = pre.process_single_file(str(edf_path), str(data_root), "eval")
    if not meta:
        raise RuntimeError("预处理失败或未生成任何文件")
    pre.write_index_files(meta, str(txt_root))

    cfg = _load_module(str(config_path), "dino_cfg")
    overrides = {
        "dataset": "tusz",
        "data_dir": str(data_root / "stft_amp_w_scale_w_crop"),
        "tusz_txt_dir": str(txt_root),
        "tusz_label_dir": None,
        "device": args.device,
        "num_workers": args.num_workers,
        "eval": True,
        "distributed": False,
        "amp": False,
        "downsample_seed": 42,
        "tusz_downsample_times": 0,
    }
    dino_args = _build_args(cfg, overrides)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2_dinoeeg" / "dino_eeg_0827"))
    from util.misc import collate_fn, load_checkpoint_mst
    from datasets import build_dataloader
    from models.dino.dino import build_dino
    import torch
    import engine

    dataloaders, _datasets = build_dataloader(collate_fn, dino_args, stage="full")
    model, criterion, postprocessors = build_dino(dino_args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    load_checkpoint_mst(model, str(checkpoint_path), strict=False, logger=None)

    os.chdir(str(eval_out))
    _stats, _sel = engine.evaluate(
        model=model,
        criterion=criterion,
        postprocessors=postprocessors,
        data_loader=dataloaders["test"],
        device=device,
        output_dir=str(eval_out),
        wo_class_error=False,
        select_thresholds_index=None,
        args=dino_args,
        logger=None,
    )

    gt_json = eval_out / "ground_truth.bbox.json"
    pred_json = eval_out / "results.bbox.json"
    if not gt_json.exists():
        gt_json = Path(eval_out / "ground_truth.bbox.json")
    if not pred_json.exists():
        pred_json = Path(eval_out / "results.bbox.json")
    if not gt_json.exists() or not pred_json.exists():
        raise RuntimeError("评估未产生必要的JSON文件")

    post_mod_path = str(Path(__file__).resolve().parents[0] / "integrated_evaluation.py")
    post = _load_module(post_mod_path, "integrated_evaluation")
    evaluator = post.IntegratedEvaluator()

    merged_path = eval_out / f"merged_predictions_nms_{args.threshold:.2f}.json"
    evaluator.merge_multichannel_predictions(
        str(pred_json),
        str(merged_path),
        merge_strategy="nms",
        score_threshold=args.threshold,
        iou_threshold=0.0,
    )

    tsv_dir = eval_out / f"tsv_threshold_{args.threshold:.2f}"
    gt_dir = tsv_dir / "gt"
    hyp_dir = tsv_dir / "hyp"
    evaluator.json_to_tsv(
        str(gt_json),
        str(merged_path),
        None,
        str(gt_dir),
        str(hyp_dir),
        score_threshold=args.threshold,
        max_predictions=30,
        channelwise=False,
    )

    with open(pred_json, "r", encoding="utf-8") as f:
        preds_before = json.load(f)
    with open(merged_path, "r", encoding="utf-8") as f:
        preds_after = json.load(f)
    nms_summary = {
        "threshold": args.threshold,
        "input_predictions": len(preds_before),
        "merged_predictions": len(preds_after),
        "output_dirs": {
            "eval_out": str(eval_out),
            "tsv_root": str(tsv_dir),
            "gt_dir": str(gt_dir),
            "hyp_dir": str(hyp_dir),
        },
        "files": {
            "gt_json": str(gt_json),
            "pred_json": str(pred_json),
            "merged_json": str(merged_path),
        },
    }
    with open(eval_out / "nms_summary.json", "w", encoding="utf-8") as f:
        json.dump(nms_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(nms_summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
