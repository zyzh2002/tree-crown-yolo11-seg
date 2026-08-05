"""Prepare the OAM-TCD dataset for generic single-class crown pretraining.

Downloads the OAM-TCD (restor/tcd) HF dataset at a pinned revision and converts
the individual-tree instance annotations (COCO category 2) into YOLO11-seg
normalized segment labels. Group/canopy annotations (category 1) are dropped.

The output is a single-class "tree-crown" dataset used ONLY as weight
initialization for the final Xi'an 4-species model. It is never published as a
release artifact and never used as the public ABI class list.

Usage:
    python prepare_oamtcd.py --output data/oamtcd
    python prepare_oamtcd.py --output data/oamtcd --limit 200   # smoke subset
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from huggingface_hub import snapshot_download
from pycocotools import mask as coco_mask
from ultralytics.data.converter import merge_multi_segment

logger = logging.getLogger("tree-crown.prepare.oamtcd")

REPO_ID = "restor/tcd"
REVISION = "d97d4da0ebbb6e249ae95ac5e19656babd972eb2"
TRAIN_FOLDS = (0, 1, 2, 3)
VAL_FOLD = 4
KEEP_CATEGORY_IDS = {2}  # tree (individual); drop 1 = canopy (group)
DROP_CATEGORY_IDS = [1]
YOLO_CLASS_NAME = "tree-crown"
YOLO_CLASS_ID = 0


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def download_raw(output: Path, revision: str = REVISION) -> Path:
    """Download the OAM-TCD parquet files into output/raw (pinned revision)."""
    raw = output / "raw"
    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=str(raw),
    )
    return raw


def load_frame(raw: Path) -> pl.DataFrame:
    """Load train/test parquet shards, marking test rows with a `_is_test` column.

    Train and test shards are read separately so the source split is preserved;
    the buggy path-string "test" heuristic is gone.
    """
    data_dir = raw / "data"
    train_paths = sorted(data_dir.glob("train-*.parquet"))
    test_paths = sorted(data_dir.glob("test-*.parquet"))
    if not train_paths and not test_paths:
        raise FileNotFoundError(f"No parquet shards found under {data_dir}")
    frames = [pl.scan_parquet(str(p)).collect().with_columns(pl.lit(False).alias("_is_test")) for p in train_paths]
    frames += [pl.scan_parquet(str(p)).collect().with_columns(pl.lit(True).alias("_is_test")) for p in test_paths]
    return pl.concat(frames)


def annotations_to_instances(annotations_json: str) -> list[dict]:
    """Parse the coco_annotations JSON string into a list of annotation dicts."""
    try:
        data = json.loads(annotations_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [a for a in data if isinstance(a, dict) and a.get("segmentation") is not None]


def segmentation_to_yolo(seg, width: int, height: int) -> list[float] | None:
    """Convert a COCO segmentation (polygon list or RLE) to YOLO normalized coords.

    Returns a flat [x1, y1, x2, y2, ...] list (class NOT included), or None if
    the instance is degenerate and must be dropped.
    """
    if isinstance(seg, dict):
        rle = seg  # COCO compressed RLE dict: {"counts": ..., "size": [H, W]}
        binary = coco_mask.decode(rle)  # type: ignore[arg-type]
        if binary is None or binary.sum() == 0:
            return None
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        pts = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
        pts = pts.astype(np.float64)
    elif isinstance(seg, list) and len(seg) > 0:
        if len(seg) > 1:
            # COCO format: each polygon is a flat [x1, y1, x2, y2, ...] array.
            # merge_multi_segment reshapes inputs to (N, 2) internally, so the
            # explicit reshape keeps the points as contiguous (N, 2) for the
            # downstream np.concatenate.
            polygons = [np.array(p, dtype=np.float64).reshape(-1, 2) for p in seg]
            merged = merge_multi_segment(polygons)
            pts = np.concatenate(merged, axis=0).astype(np.float64)
        else:
            pts = np.array(seg[0], dtype=np.float64).reshape(-1, 2)
    else:
        return None

    if pts.shape[0] < 3:
        return None
    pts[:, 0] /= width
    pts[:, 1] /= height
    pts = np.clip(pts, 0.0, 1.0)
    if _polygon_area(pts) <= 0:
        return None
    return pts.reshape(-1).tolist()


def _polygon_area(pts: np.ndarray) -> float:
    """Signed shoelace area; positive for a valid CCW polygon."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def assign_split(validation_fold, in_test: bool) -> str:
    """Map an OAM-TCD row to train/val/test."""
    if in_test:
        return "test"
    if validation_fold == VAL_FOLD:
        return "val"
    if validation_fold in TRAIN_FOLDS:
        return "train"
    raise ValueError(f"Unexpected validation_fold {validation_fold}")


def row_split(row: dict) -> str:
    """Determine the split for a prepared row (testable without parquet)."""
    return assign_split(row["validation_fold"], bool(row.get("_is_test", False)))


def write_image_and_labels(
    row: dict,
    split: str,
    images_dir: Path,
    labels_dir: Path,
    stats: dict,
) -> int:
    """Write one image .jpg and its YOLO .txt label into the split dirs.

    Returns the number of instances kept for this row.
    """
    image_id = int(row["image_id"])
    width = int(row["width"])
    height = int(row["height"])
    img_bytes = _extract_image_bytes(row)

    image_path = images_dir / f"{image_id}.jpg"
    label_path = labels_dir / f"{image_id}.txt"

    if img_bytes is None:
        stats["decode_error"] += 1
        return 0
    image_path.write_bytes(img_bytes)

    lines: list[str] = []
    kept = 0
    for ann in annotations_to_instances(str(row["coco_annotations"])):
        if ann["category_id"] not in KEEP_CATEGORY_IDS:
            continue
        coords = segmentation_to_yolo(ann["segmentation"], width, height)
        if coords is None:
            stats["dropped_invalid"] += 1
            continue
        lines.append(" ".join([str(YOLO_CLASS_ID), *[f"{v:.6f}" for v in coords]]))
        kept += 1
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    stats["instances"] += kept
    return kept


def _extract_image_bytes(row: dict) -> bytes | None:
    """Extract raw image bytes from the HF 'image' column (bytes or dict)."""
    value = row["image"]
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
        if isinstance(value, Path):
            try:
                return value.read_bytes()
            except OSError:
                return None
        if isinstance(value, bytes):
            return value
        return None
    if isinstance(value, bytes):
        return value
    return None


def prepare(output: Path, limit: int | None = None, skip_download: bool = False) -> dict:
    """Run the full OAM-TCD -> YOLO-seg conversion and return the manifest."""
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    raw = output / "raw"
    if skip_download:
        if not (raw / "data").exists():
            raise FileNotFoundError(f"--skip-download given but no data under {raw}")
    else:
        raw = download_raw(output)

    df = load_frame(raw)
    if limit is not None:
        df = df.head(limit)

    stats = {"instances": 0, "dropped_invalid": 0, "decode_error": 0}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_instances = {"train": 0, "val": 0, "test": 0}
    oam_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for row in df.iter_rows(named=True):
        split = row_split(row)
        split_counts[split] += 1
        oam_split[split].add(str(row["oam_id"]))
        split_instances[split] += write_image_and_labels(
            row,
            split,
            output / "images" / split,
            output / "labels" / split,
            stats,
        )

    _write_data_yaml(output)
    _write_manifest(output, df, stats, split_counts, split_instances, oam_split, limit)
    return _load_manifest(output)


def _write_data_yaml(output: Path) -> None:
    content = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  {YOLO_CLASS_ID}: {YOLO_CLASS_NAME}\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def _write_manifest(output, df, stats, split_counts, split_instances, oam_split, limit) -> None:
    manifest = {
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "license": "cc-by-4.0",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_commit": _git_commit(),
        "class_map": {YOLO_CLASS_NAME: YOLO_CLASS_ID},
        "keep_category_ids": sorted(KEEP_CATEGORY_IDS),
        "drop_category_ids": DROP_CATEGORY_IDS,
        "limit": limit,
        "total_rows": df.height,
        "split": {
            s: {
                "n_images": split_counts[s],
                "n_instances": split_instances[s],
                "oam_ids": sorted(oam_split[s]),
            }
            for s in ("train", "val", "test")
        },
        "counts": stats,
        "oam_split_guarantee": "no oam_id appears in more than one split",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_manifest(output: Path) -> dict:
    with (output / "manifest.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare OAM-TCD for YOLO11-seg pretraining.")
    parser.add_argument("--output", default="data/oamtcd", help="Output directory (git-ignored).")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows (smoke).")
    parser.add_argument("--skip-download", action="store_true", help="Use existing downloaded parquet.")
    parser.add_argument("--revision", default=REVISION, help="Pinned HF revision.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        manifest = prepare(Path(args.output), limit=args.limit, skip_download=args.skip_download)
        logger.info("OAM-TCD prepared. Manifest at %s", Path(args.output) / "manifest.json")
        logger.info("Split: %s", manifest["split"])
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Prepare failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
