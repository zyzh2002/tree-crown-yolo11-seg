"""Train or validate the tree-crown YOLO11-seg model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --mode validate --weights <best.pt>
    python train.py --config configs/pretrain-oamtcd.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import subprocess
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
    "optimizer",
    "lr0",
    "lrf",
    "cos_lr",
    "freeze",
    "warmup_epochs",
    "close_mosaic",
    "mosaic",
    "degrees",
    "flipud",
    "fliplr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "scale",
    "translate",
    "weight_decay",
    "save_period",
    "max_det",
)

VALIDATION_KEYS = (
    "data",
    "imgsz",
    "batch",
    "device",
    "verbose",
    "split",
    "max_det",
    "conf",
    "iou",
    "save_json",
    "plots",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha256(path)}


def _git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current") or "detached",
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def _package_versions() -> dict:
    packages = ("ultralytics", "torch", "torchvision", "onnx")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _capture_provenance(
    *,
    config: dict,
    resolved: dict,
    model_path: str,
    lock_path: Path = Path("uv.lock"),
) -> dict:
    """Capture reproducibility metadata before training starts."""
    provenance: dict = {
        "git": _git_state(),
        "config": config,
        "resolved_args": resolved,
        "packages": _package_versions(),
    }
    model = Path(model_path)
    if model.is_file():
        provenance["initial_weights"] = _file_record(model)
    if "data" in resolved:
        data_yaml = Path(str(resolved["data"]))
        manifest = data_yaml.parent / "manifest.json"
        if manifest.is_file():
            provenance["dataset_manifest"] = _file_record(manifest)
    if lock_path.is_file():
        provenance["uv_lock"] = _file_record(lock_path)
    return provenance


def _write_provenance(save_dir: Path, provenance: dict) -> None:
    """Write a pre-training provenance snapshot next to the completed run."""
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "provenance.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


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

    if config.get("task", "segment") != "segment":
        raise ValueError("task must be 'segment'")

    defaults = {
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
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "freeze": None,
        "warmup_epochs": 3.0,
        "close_mosaic": 10,
        "mosaic": 1.0,
        "degrees": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "scale": 0.5,
        "translate": 0.1,
        "weight_decay": 0.0005,
        "save_period": -1,
        "max_det": 300,
        "split": "val",
        "conf": None,
        "iou": 0.7,
        "save_json": False,
    }
    args = {key: _resolve(config, overrides, key, default) for key, default in defaults.items()}

    if mode == "validate":
        weights = overrides.get("weights")
        if not weights:
            raise ValueError("--weights is required for validate mode")
        logger.info("Validating model: %s", weights)
        model = YOLO(weights)
        results = model.val(**{key: args[key] for key in VALIDATION_KEYS})
        box_map = getattr(results, "box", None) and results.box.map
        mask_map = getattr(results, "seg", None) and results.seg.map
        logger.info("Validation done. box mAP50-95: %.4f | mask mAP50-95: %.4f", box_map, mask_map)
        return

    model_path = config.get("model", DEFAULT_MODEL)
    resume = _resolve(config, overrides, "resume", None)
    if resume is True:
        raise ValueError("--resume requires an explicit checkpoint path")
    provenance_args = (
        {"resume": True, "checkpoint": str(resume)}
        if resume
        else {"epochs": _resolve(config, overrides, "epochs", 100), **args}
    )
    provenance = _capture_provenance(
        config=config,
        resolved=provenance_args,
        model_path=model_path if not resume else str(resume),
    )
    if resume:
        ckpt = str(resume)
        logger.info("Resuming training from checkpoint: %s", ckpt)
        model = YOLO(ckpt)
        model.train(
            epochs=_resolve(config, overrides, "epochs", 100),
            resume=True,
            **{key: args[key] for key in FORWARDED_KEYS},
        )
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
        model.train(epochs=_resolve(config, overrides, "epochs", 100), **{key: args[key] for key in FORWARDED_KEYS})

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        save_dir = getattr(trainer, "save_dir", None)
        best = getattr(trainer, "best", None)
        logger.info("Training finished. Results: %s", save_dir)
        logger.info("Best weights: %s", best)
        if isinstance(save_dir, (str, Path)):
            _write_provenance(Path(save_dir), provenance)
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
        help="Resume training from an explicit last.pt checkpoint path.",
    )
    parser.add_argument("--max-det", type=int, help="Override maximum detections retained during validation.")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Override validation split.")
    parser.add_argument("--conf", type=float, help="Override validation confidence threshold.")
    parser.add_argument("--iou", type=float, help="Override validation NMS IoU threshold.")
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
                "max_det": args.max_det,
                "split": args.split,
                "conf": args.conf,
                "iou": args.iou,
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
