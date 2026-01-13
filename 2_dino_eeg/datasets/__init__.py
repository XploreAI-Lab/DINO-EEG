from .tusz import build_tusz_dataloader
from .evaluator import get_coco_metrics as get_metrics, get_coco_summary as get_summary
from .metrics import get_event_metrics, get_patient_metircs
from .chbmit import build_chbmit_dataloader
from .neonatal import build_neonatal_dataloader

__all__ = [
    "build_dataloader",
    "get_metrics",
    "get_summary",
    "get_event_metrics",
    "get_patient_metircs",
]


def build_dataloader(collate_fn, args, stage="full"):
    """
    构建数据加载器
    
    Args:
        collate_fn: 数据整理函数
        args: 训练参数
        stage: "seizure_only" - 第一阶段（仅癫痫数据）
               "full" - 第二阶段（完整数据）
    """
    if args.dataset == "tusz":
        return build_tusz_dataloader(collate_fn, args, stage=stage)
    elif args.dataset == "chbmit":
        return build_chbmit_dataloader(collate_fn, args)
    elif args.dataset == "neonatal":
        return build_neonatal_dataloader(collate_fn, args)

    raise ValueError(f"dataset {args.dataset} not supported")
