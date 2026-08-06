"""Train or validate the tree-crown YOLO11-seg model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --mode validate --weights <best.pt>
    python train.py --config configs/pretrain-oamtcd.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("tree-crown.train")

# Keys forwarded verbatim to ultralytics train()/val(). Falsy values (0, False)
# are meaningful (e.g. batch=0 AutoBatch, workers=0, amp=False), so resolution
# uses membership, never `X or default`.
FORWARDED_KEYS = (
    "data",
    "imgsz",
    "batch",
    "device",
    "seed",
    "verbose",
    "deterministic",
    "workers",
    "cache",
    "amp",
    "patience",
    "project",
    "name",
    "plots",
    "single_cls",
)

DEFAULT_MODEL = "yolo11n-seg.pt"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(config: dict, overrides: dict, key: str, default):
    """Resolve a config key: an explicit CLI override beats the config file."""
    if key in overrides:
        return overrides[key]
    if key in config:
        return config[key]
    return default


def _run_training(config: dict, mode: str, **overrides) -> None:
    # Import ultralytics lazily so `--help` works without the heavy dependency.
    from ultralytics import YOLO

    args = {
        key: _resolve(config, overrides, key, default)
        for key, default in {
            "data": "data.yaml",
            "imgsz": 1280,
            "batch": 16,
            "device": 0,
            "seed": 0,
            "verbose": True,
            "deterministic": True,
            "workers": 8,
            "cache": False,
            "amp": True,
            "patience": 100,
            "project": "",
            "name": "train",
            "plots": True,
            "single_cls": False,
        }.items()
    }

    if mode == "validate":
        weights = overrides.get("weights")
        if not weights:
            raise ValueError("--weights is required for validate mode")
        logger.info("Validating model: %s", weights)
        model = YOLO(weights)
        results = model.val(
            data=args["data"],
            imgsz=args["imgsz"],
            batch=args["batch"],
            device=args["device"],
            verbose=args["verbose"],
        )
        box_map = getattr(results, "box", None) and results.box.map
        mask_map = getattr(results, "seg", None) and results.seg.map
        logger.info("Validation done. box mAP50-95: %.4f | mask mAP50-95: %.4f", box_map, mask_map)
        return

    model_path = config.get("model", DEFAULT_MODEL)
    resume = _resolve(config, overrides, "resume", None)
    if resume:
        # ultralytics: resume=True resumes from {project}/{name}/weights/last.pt;
        # resume=<path> resumes from an explicit checkpoint.
        ckpt = "last.pt" if resume is True else str(resume)
        logger.info("Resuming training from checkpoint: %s", ckpt)
        model = YOLO(ckpt)
        model.train(epochs=_resolve(config, overrides, "epochs", 100), resume=True, **args)
    else:
        logger.info("Loading model: %s", model_path)
        model = YOLO(model_path)
        logger.info(
            "Starting training: data=%s imgsz=%d batch=%s device=%s epochs=%d",
            args["data"],
            args["imgsz"],
            args["batch"],
            args["device"],
            _resolve(config, overrides, "epochs", 100),
        )
        model.train(epochs=_resolve(config, overrides, "epochs", 100), **args)

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        save_dir = getattr(trainer, "save_dir", None)
        best = getattr(trainer, "best", None)
        logger.info("Training finished. Results: %s", save_dir)
        logger.info("Best weights: %s", best)
    else:
        logger.info("Training finished. Best weights at runs/segment/train/weights/best.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or validate tree-crown YOLO11-seg.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["train", "validate"], default="train", help="Run training or validation.")
    parser.add_argument("--weights", help="Path to weights (required for validate mode).")
    parser.add_argument("--data", help="Override dataset YAML.")
    parser.add_argument("--imgsz", type=int, help="Override input resolution.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch", type=int, help="Override batch size (0 for AutoBatch).")
    parser.add_argument("--device", help="Override device (e.g. 0, 'cpu').")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume training. Bare flag resumes from {project}/{name}/weights/last.pt; "
        "or give a path to resume from that checkpoint.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        config = _load_config(Path(args.config))
        overrides = {
            k: v
            for k, v in {
                "weights": args.weights,
                "data": args.data,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "device": args.device,
                "seed": args.seed,
                "resume": args.resume,
            }.items()
            if v is not None
        }
        _run_training(config, args.mode, **overrides)
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
