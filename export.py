"""Export a trained YOLO11-seg model to ONNX with fixed input/output naming.

The exported ONNX is the ABI contract for the onboard TensorRtEngine. Onboard
baseline is TensorRT 8.5.2, so the export must stay compatible with its operators.

Usage:
    python export.py --weights runs/segment/train/weights/best.pt --imgsz 1280
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("tree-crown.export")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _export(weights: str, imgsz: int, target_trt: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        dynamic=False,
        half=False,
    )
    path = Path(onnx_path)
    logger.info("Exported ONNX: %s", path)
    logger.info("Target TensorRT baseline: %s", target_trt)
    logger.info("Input contract: images [1, 3, %d, %d] float32 NCHW", imgsz, imgsz)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO11-seg to ONNX.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input resolution (fixed contract).")
    parser.add_argument("--target-trt", default="8.5.2", help="Onboard TensorRT baseline version.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        _export(args.weights, args.imgsz, args.target_trt)
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Export failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
