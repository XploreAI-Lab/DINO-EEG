from .tuev import build_tuev_dataloader, build_tuev_dataloader_cross_subject
from .tusz import build_tusz_dataloader
from .tuar import build_tuar_dataloader
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


def build_dataloader(collate_fn, args):
    if args.dataset == "tuev":
        if args.tuev_cross_subject:
            return build_tuev_dataloader_cross_subject(collate_fn, args)
        else:
            return build_tuev_dataloader(collate_fn, args)
    elif args.dataset == "tusz":
        return build_tusz_dataloader(collate_fn, args)
    elif args.dataset == "tuar":
        return build_tuar_dataloader(collate_fn, args)
    elif args.dataset == "chbmit":
        return build_chbmit_dataloader(collate_fn, args)
    elif args.dataset == "neonatal":
        return build_neonatal_dataloader(collate_fn, args)

    raise ValueError(f"dataset {args.dataset} not supported")
