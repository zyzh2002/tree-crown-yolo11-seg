"""Prepare the OAM-TCD dataset for generic single-class crown pretraining.

Downloads the OAM-TCD (restor/tcd) HF dataset at a pinned revision and converts
the individual-tree instance annotations (COCO category 2) into YOLO11-seg
normalized segment labels. Group/canopy annotations (category 1) are dropped.

The output is a single-class "tree-crown" dataset used ONLY as weight
initialization for the final Xi'an model. It is never a deployable artifact and
never used as the public ABI class list.

Usage:
    python prepare_oamtcd.py --output data/oamtcd-rgb
    python prepare_oamtcd.py --output data/oamtcd-smoke --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from huggingface_hub import snapshot_download
from pycocotools import mask as coco_mask

logger = logging.getLogger("tree-crown.prepare.oamtcd")

REPO_ID = "restor/tcd"
REVISION = "d97d4da0ebbb6e249ae95ac5e19656babd972eb2"
TRAIN_FOLDS = (0, 1, 2, 3)
VAL_FOLD = 4
KEEP_CATEGORY_IDS = {2}  # tree (individual); drop 1 = canopy (group)
DROP_CATEGORY_IDS = [1]
YOLO_CLASS_NAME = "tree-crown"
YOLO_CLASS_ID = 0
MIN_DOMINANT_COMPONENT_RATIO = 0.9


@dataclass(frozen=True)
class WriteResult:
    """Result of converting one source row."""

    written: bool
    instances: int


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
    if raw.exists():
        shutil.rmtree(raw)
    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=str(raw),
    )
    _write_raw_source(raw, revision)
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
        rle = seg  # COCO RLE dict: {"counts": ..., "size": [H, W]}
        # counts may be a compressed str/bytes (coco_mask.decode handles it) or
        # an uncompressed list of run-lengths (decode needs frPyObjects first).
        if not isinstance(rle.get("counts"), (str, bytes)):
            # pycocotools frPyObjects/decode are C-extensions with weak stubs;
            # pyright cannot infer the dict-RLE overload.
            rle = coco_mask.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]  # type: ignore[arg-type]
        binary = coco_mask.decode(rle)  # type: ignore[arg-type]
        if binary is None or binary.sum() == 0:
            return None
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # YOLO-seg stores one polygon per instance. Reject truly disconnected
        # instances rather than creating an artificial bridge between regions.
        areas = [cv2.contourArea(contour) for contour in contours]
        largest = max(range(len(contours)), key=areas.__getitem__)
        if sum(areas) > 0 and areas[largest] / sum(areas) < MIN_DOMINANT_COMPONENT_RATIO:
            return None
        contour = contours[largest]
        pts = contour.reshape(-1, 2).astype(np.float64)
    elif isinstance(seg, list) and len(seg) > 0:
        polygons = [np.array(p, dtype=np.float64).reshape(-1, 2) for p in seg if len(p) >= 6]
        if not polygons:
            return None
        areas = [_polygon_area(polygon) for polygon in polygons]
        largest = max(range(len(polygons)), key=areas.__getitem__)
        if sum(areas) > 0 and areas[largest] / sum(areas) < MIN_DOMINANT_COMPONENT_RATIO:
            return None
        pts = polygons[largest]
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
) -> WriteResult:
    """Write one image and its YOLO label into the split dirs.

    Rows containing only excluded canopy-group annotations are omitted because
    treating them as empty labels would incorrectly supervise visible trees as
    background.
    """
    image_id = int(row["image_id"])
    width = int(row["width"])
    height = int(row["height"])
    annotations = annotations_to_instances(str(row["coco_annotations"]))
    tree_annotations = [ann for ann in annotations if ann.get("category_id") in KEEP_CATEGORY_IDS]
    has_canopy = any(ann.get("category_id") in DROP_CATEGORY_IDS for ann in annotations)

    if has_canopy and not tree_annotations:
        stats["dropped_canopy_only_images"] = stats.get("dropped_canopy_only_images", 0) + 1
        return WriteResult(written=False, instances=0)

    lines: list[str] = []
    seen: set[str] = set()
    for ann in tree_annotations:
        coords = segmentation_to_yolo(ann["segmentation"], width, height)
        if coords is None:
            stats["dropped_invalid"] += 1
            continue
        line = " ".join([str(YOLO_CLASS_ID), *[f"{v:.6f}" for v in coords]])
        if line in seen:
            stats["dropped_duplicate"] = stats.get("dropped_duplicate", 0) + 1
            continue
        seen.add(line)
        lines.append(line)

    if tree_annotations and not lines:
        stats["dropped_no_valid_tree_images"] = stats.get("dropped_no_valid_tree_images", 0) + 1
        return WriteResult(written=False, instances=0)

    img_bytes = _extract_image_bytes(row)
    if img_bytes is None:
        stats["decode_error"] += 1
        return WriteResult(written=False, instances=0)

    if has_canopy:
        img_bytes = _redact_canopy_regions(
            img_bytes,
            [ann for ann in annotations if ann.get("category_id") == 1],
            tree_annotations,
        )
        if img_bytes is None:
            stats["decode_error"] += 1
            return WriteResult(written=False, instances=0)
        stats["redacted_canopy_images"] = stats.get("redacted_canopy_images", 0) + 1
    else:
        img_bytes = _encode_rgb_jpeg(img_bytes)
        if img_bytes is None:
            stats["decode_error"] += 1
            return WriteResult(written=False, instances=0)

    image_path = images_dir / f"{image_id}.jpg"
    label_path = labels_dir / f"{image_id}.txt"
    image_path.write_bytes(img_bytes)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    stats["instances"] += len(lines)
    return WriteResult(written=True, instances=len(lines))


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


def _redact_canopy_regions(
    image_bytes: bytes,
    canopy_annotations: list[dict],
    tree_annotations: list[dict],
) -> bytes | None:
    """Black out excluded group-canopy regions so they are not supervised as background."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    canopy_mask = _annotation_mask(canopy_annotations, width, height)
    tree_mask = _annotation_mask(tree_annotations, width, height)
    canopy_mask[tree_mask.astype(bool)] = 0
    image[canopy_mask.astype(bool)] = 0
    return _encode_rgb_jpeg(image)


def _encode_rgb_jpeg(image_or_bytes: np.ndarray | bytes) -> bytes | None:
    """Encode a source image as a three-channel JPEG for Ultralytics Mosaic."""
    if isinstance(image_or_bytes, bytes):
        image = cv2.imdecode(np.frombuffer(image_or_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        image = image_or_bytes
    if image is None:
        return None
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes() if ok else None


def _annotation_mask(annotations: list[dict], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, dict):
            rle = segmentation
            if not isinstance(rle.get("counts"), (str, bytes)):
                rle = coco_mask.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]  # type: ignore[arg-type]
            decoded = coco_mask.decode(rle)  # type: ignore[arg-type]
            if decoded is not None:
                mask |= decoded.astype(np.uint8)
        elif isinstance(segmentation, list):
            polygons = [np.array(p, dtype=np.float64).reshape(-1, 2) for p in segmentation if len(p) >= 6]
            if polygons:
                cv2.fillPoly(mask, [polygon.round().astype(np.int32) for polygon in polygons], 1)
    return mask


def prepare(output: Path, limit: int | None = None, skip_download: bool = False, revision: str = REVISION) -> dict:
    """Run the full OAM-TCD -> YOLO-seg conversion and return the manifest."""
    output.mkdir(parents=True, exist_ok=True)
    generated = (output / "images", output / "labels", output / "data.yaml", output / "manifest.json")
    if any(path.exists() for path in generated):
        raise FileExistsError(f"Generated output already exists under {output}; use a new output directory")

    raw = output / "raw"
    if skip_download:
        if not (raw / "data").exists():
            raise FileNotFoundError(f"--skip-download given but no data under {raw}")
        source = _load_raw_source(raw)
        if source.get("revision") != revision:
            raise ValueError(
                f"Existing raw revision {source.get('revision')!r} does not match requested revision {revision!r}"
            )
    else:
        raw = download_raw(output, revision)

    df = load_frame(raw)
    if limit is not None:
        df = df.head(limit)

    stats = {
        "instances": 0,
        "dropped_invalid": 0,
        "dropped_duplicate": 0,
        "dropped_canopy_only_images": 0,
        "dropped_no_valid_tree_images": 0,
        "redacted_canopy_images": 0,
        "decode_error": 0,
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_instances = {"train": 0, "val": 0, "test": 0}
    split_max_instances = {"train": 0, "val": 0, "test": 0}
    oam_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    staging = Path(tempfile.mkdtemp(prefix=".oamtcd-convert-", dir=output))
    try:
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)

        for row in df.iter_rows(named=True):
            split = row_split(row)
            result = write_image_and_labels(
                row,
                split,
                staging / "images" / split,
                staging / "labels" / split,
                stats,
            )
            if not result.written:
                continue
            split_counts[split] += 1
            split_instances[split] += result.instances
            split_max_instances[split] = max(split_max_instances[split], result.instances)
            oam_split[split].add(str(row["oam_id"]))

        _validate_oam_splits(oam_split)
        _write_data_yaml(staging)
        _write_manifest(
            staging,
            df,
            stats,
            split_counts,
            split_instances,
            oam_split,
            limit,
            revision=revision,
            split_max_instances=split_max_instances,
            raw_files=_raw_files(raw),
        )
        _install_generated(output, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return _load_manifest(output)


def _write_data_yaml(output: Path) -> None:
    content = f"train: images/train\nval: images/val\ntest: images/test\nnames:\n  {YOLO_CLASS_ID}: {YOLO_CLASS_NAME}\n"
    (output / "data.yaml").write_text(content, encoding="utf-8")


def _write_manifest(
    output,
    df,
    stats,
    split_counts,
    split_instances,
    oam_split,
    limit,
    revision: str = REVISION,
    split_max_instances: dict[str, int] | None = None,
    raw_files: list[dict] | None = None,
) -> None:
    split_max_instances = split_max_instances or {"train": 0, "val": 0, "test": 0}
    manifest = {
        "source": {
            "repo_id": REPO_ID,
            "revision": revision,
            "license": "cc-by-4.0",
            "files": raw_files or [],
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
                "max_instances_per_image": split_max_instances[s],
                "oam_ids": sorted(oam_split[s]),
            }
            for s in ("train", "val", "test")
        },
        "counts": stats,
        "oam_split_guarantee": "no oam_id appears in more than one split",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_oam_splits(oam_split: dict[str, set[str]]) -> None:
    """Reject source groups assigned to more than one split."""
    conflicts = (
        (oam_split["train"] & oam_split["val"])
        | (oam_split["train"] & oam_split["test"])
        | (oam_split["val"] & oam_split["test"])
    )
    if conflicts:
        sample = ", ".join(sorted(conflicts)[:10])
        raise ValueError(f"oam_id appears in multiple splits: {sample}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_files(raw: Path) -> list[dict]:
    return [
        {"path": str(path.relative_to(raw)), "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted((raw / "data").glob("*.parquet"))
    ]


def _write_raw_source(raw: Path, revision: str) -> None:
    source = {"repo_id": REPO_ID, "revision": revision, "files": _raw_files(raw)}
    (raw / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")


def _load_raw_source(raw: Path) -> dict:
    path = raw / "source.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw provenance missing: {path}. Download the pinned revision again instead of using --skip-download."
        )
    source = json.loads(path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in source.get("files", [])}
    actual = {item["path"]: item for item in _raw_files(raw)}
    if expected != actual:
        raise ValueError("Existing raw parquet files do not match raw/source.json checksums")
    return source


def _install_generated(output: Path, staging: Path) -> None:
    """Replace generated dataset files only after conversion succeeds."""
    for name in ("images", "labels", "data.yaml", "manifest.json"):
        destination = output / name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        shutil.move(str(staging / name), str(destination))


def _load_manifest(output: Path) -> dict:
    with (output / "manifest.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare OAM-TCD for YOLO11-seg pretraining.")
    parser.add_argument("--output", default="data/oamtcd-rgb", help="Output directory (git-ignored).")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows (smoke).")
    parser.add_argument("--skip-download", action="store_true", help="Use existing downloaded parquet.")
    parser.add_argument("--revision", default=REVISION, help="Pinned HF revision.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        manifest = prepare(
            Path(args.output),
            limit=args.limit,
            skip_download=args.skip_download,
            revision=args.revision,
        )
        logger.info("OAM-TCD prepared. Manifest at %s", Path(args.output) / "manifest.json")
        logger.info("Split: %s", manifest["split"])
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Prepare failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
