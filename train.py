"""Train or validate the tree-crown YOLO11-seg model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --mode validate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("tree-crown.train")


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


def _run_training(config: dict, mode: str, **overrides) -> None:
    # Import ultralytics lazily so `--help` works without the heavy dependency.
    from ultralytics import YOLO

    model_path = config.get("model", "yolov11n-seg.pt")
    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    data = config.get("data", "data.yaml")
    imgsz = overrides.get("imgsz") or config.get("imgsz", 1280)
    epochs = overrides.get("epochs") or config.get("epochs", 100)
    batch = overrides.get("batch") or config.get("batch", 16)
    device = overrides.get("device") or config.get("device", 0)
    seed = overrides.get("seed") or config.get("seed", 0)

    common = {
        "data": data,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "seed": seed,
        "verbose": True,
    }

    if mode == "validate":
        weights = overrides.get("weights")
        if not weights:
            raise ValueError("--weights is required for validate mode")
        logger.info("Validating model: %s", weights)
        model = YOLO(weights)
        results = model.val(data=data, imgsz=imgsz, batch=batch, device=device, verbose=True)
        logger.info("Validation done. mAP50-95: %.4f", getattr(results, "box", None) and results.box.map)
        return

    logger.info("Starting training: epochs=%d imgsz=%d batch=%d device=%s", epochs, imgsz, batch, device)
    model.train(epochs=epochs, **common)
    logger.info("Training finished. Best weights at runs/segment/train/weights/best.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or validate tree-crown YOLO11-seg.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["train", "validate"], default="train", help="Run training or validation.")
    parser.add_argument("--weights", help="Path to weights (required for validate mode).")
    parser.add_argument("--imgsz", type=int, help="Override input resolution.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--device", help="Override device (e.g. 0, 'cpu').")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        config = _load_config(Path(args.config))
        overrides = {
            k: v
            for k, v in {
                "weights": args.weights,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "device": args.device,
                "seed": args.seed,
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
