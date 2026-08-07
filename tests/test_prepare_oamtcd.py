"""Tests for prepare_oamtcd.py OAM-TCD conversion and split integrity."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import polars as pl
import pytest
from pycocotools import mask as coco_mask

import prepare_oamtcd

WIDTH = 100
HEIGHT = 100


# --- segmentation_to_yolo ---


def test_single_polygon_normalization() -> None:
    seg = [[0, 0, 100, 0, 100, 100, 0, 100]]  # flat COCO polygon
    coords = prepare_oamtcd.segmentation_to_yolo(seg, WIDTH, HEIGHT)
    assert coords == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_disconnected_multi_polygon_is_dropped() -> None:
    seg = [
        [0, 0, 20, 0, 20, 20, 0, 20],
        [80, 80, 100, 80, 100, 100, 80, 100],
    ]
    coords = prepare_oamtcd.segmentation_to_yolo(seg, WIDTH, HEIGHT)
    assert coords is None


def test_rle_decode_contour_normalization() -> None:
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[2:8, 2:8] = 1
    rle = coco_mask.encode(np.asfortranarray(binary))
    counts = rle["counts"]
    rle = {"counts": counts.decode("utf-8") if isinstance(counts, bytes) else counts, "size": [10, 10]}
    coords = prepare_oamtcd.segmentation_to_yolo(rle, 10, 10)
    assert coords is not None
    assert len(coords) % 2 == 0
    assert len(coords) >= 6
    assert all(0.0 <= v <= 1.0 for v in coords)


def test_uncompressed_rle_decode() -> None:
    # Build a known binary mask, then express it as an UNCOMPRESSED RLE
    # (counts as a list of alternating run lengths from 0) to exercise the
    # frPyObjects path that the OAM-TCD data uses.
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[2:8, 2:8] = 1
    flat = binary.reshape(-1, order="F")
    runs: list[int] = [0]
    prev = 0
    for bit in flat:
        if bit != prev:
            runs.append(0)
            prev = bit
        runs[-1] += 1
    rle = {"counts": runs, "size": [10, 10]}
    coords = prepare_oamtcd.segmentation_to_yolo(rle, 10, 10)
    assert coords is not None
    assert len(coords) % 2 == 0
    assert len(coords) >= 6
    assert all(0.0 <= v <= 1.0 for v in coords)


def test_disconnected_rle_is_dropped_instead_of_bridged() -> None:
    binary = np.zeros((20, 20), dtype=np.uint8)
    binary[2:7, 2:7] = 1
    binary[13:18, 13:18] = 1
    rle = coco_mask.encode(np.asfortranarray(binary))
    counts = rle["counts"]
    rle = {"counts": counts.decode("utf-8") if isinstance(counts, bytes) else counts, "size": [20, 20]}

    coords = prepare_oamtcd.segmentation_to_yolo(rle, 20, 20)

    assert coords is None


def test_less_than_three_points_dropped() -> None:
    seg = [[0, 0, 10, 0]]  # flat, 2 points
    assert prepare_oamtcd.segmentation_to_yolo(seg, WIDTH, HEIGHT) is None


def test_zero_area_dropped() -> None:
    seg = [[0, 0, 10, 0, 20, 0]]  # collinear -> area 0
    assert prepare_oamtcd.segmentation_to_yolo(seg, WIDTH, HEIGHT) is None


def test_coords_clipped_to_unit() -> None:
    seg = [[-5, -5, 5, -5, 5, 5]]
    coords = prepare_oamtcd.segmentation_to_yolo(seg, 10, 10)
    assert coords is not None
    assert all(0.0 <= v <= 1.0 for v in coords)
    assert coords[0] == 0.0  # -0.5 clipped up to 0
    assert coords[1] == 0.0


# --- annotations_to_instances ---


def test_annotations_keeps_only_dicts_with_segmentation() -> None:
    anns = prepare_oamtcd.annotations_to_instances(
        json.dumps(
            [
                {"category_id": 2, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                {"category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                {"category_id": 2, "segmentation": None},
                "not-a-dict",
            ]
        )
    )
    assert len(anns) == 2
    assert all(ann["segmentation"] is not None for ann in anns)


# --- write_image_and_labels ---


def _row(coco_annotations: str, image: bytes | None = b"fakejpeg") -> dict:
    return {
        "image_id": 1,
        "width": WIDTH,
        "height": HEIGHT,
        "image": image,
        "coco_annotations": coco_annotations,
    }


def _png_bytes() -> bytes:
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _grayscale_tiff_bytes() -> bytes:
    image = np.full((HEIGHT, WIDTH), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".tif", image)
    assert ok
    return encoded.tobytes()


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def test_category_2_kept_category_1_dropped(tmp_path: Path) -> None:
    anns = [
        {"category_id": 1, "segmentation": [[0, 0, 70, 0, 70, 70, 0, 70]]},
        {"category_id": 2, "segmentation": [[60, 60, 100, 60, 100, 100, 60, 100]]},
    ]
    stats = {"instances": 0, "dropped_invalid": 0, "decode_error": 0, "redacted_canopy_images": 0}
    images_dir, labels_dir = _make_dirs(tmp_path)
    prepare_oamtcd.write_image_and_labels(
        _row(json.dumps(anns), image=_png_bytes()), "train", images_dir, labels_dir, stats
    )
    lines = (labels_dir / "1.txt").read_text(encoding="ascii").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("0 ")  # single class id 0
    assert stats["instances"] == 1
    output_image = cv2.imread(str(next(images_dir.iterdir())))
    assert output_image is not None
    assert np.all(output_image[20, 20] < 10)
    assert np.all(output_image[65, 65] > 245)
    assert np.all(output_image[80, 80] > 245)
    assert stats["redacted_canopy_images"] == 1


def test_true_background_writes_empty_txt(tmp_path: Path) -> None:
    stats = {"instances": 0, "dropped_invalid": 0, "dropped_duplicate": 0, "decode_error": 0}
    images_dir, labels_dir = _make_dirs(tmp_path)
    result = prepare_oamtcd.write_image_and_labels(
        _row("[]", image=_png_bytes()), "train", images_dir, labels_dir, stats
    )
    assert (labels_dir / "1.txt").read_text(encoding="ascii") == ""
    assert stats["instances"] == 0
    assert result.written is True
    assert result.instances == 0


def test_grayscale_source_is_written_as_three_channel_jpeg(tmp_path: Path) -> None:
    stats = {"instances": 0, "dropped_invalid": 0, "dropped_duplicate": 0, "decode_error": 0}
    images_dir, labels_dir = _make_dirs(tmp_path)

    result = prepare_oamtcd.write_image_and_labels(
        _row("[]", image=_grayscale_tiff_bytes()), "train", images_dir, labels_dir, stats
    )

    image_path = next(images_dir.iterdir())
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert result.written is True
    assert image_path.suffix == ".jpg"
    assert image is not None
    assert image.shape == (HEIGHT, WIDTH, 3)


def test_canopy_only_row_is_not_written_as_background(tmp_path: Path) -> None:
    anns = [{"category_id": 1, "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]}]
    stats = {
        "instances": 0,
        "dropped_invalid": 0,
        "dropped_duplicate": 0,
        "dropped_canopy_only_images": 0,
        "decode_error": 0,
    }
    images_dir, labels_dir = _make_dirs(tmp_path)

    result = prepare_oamtcd.write_image_and_labels(_row(json.dumps(anns)), "train", images_dir, labels_dir, stats)

    assert result.written is False
    assert not list(images_dir.iterdir())
    assert not list(labels_dir.iterdir())
    assert stats["dropped_canopy_only_images"] == 1


def test_decode_error_returns_zero(tmp_path: Path) -> None:
    stats = {"instances": 0, "dropped_invalid": 0, "dropped_duplicate": 0, "decode_error": 0}
    images_dir, labels_dir = _make_dirs(tmp_path)
    row = _row("[]", image=None)  # undecodable image
    result = prepare_oamtcd.write_image_and_labels(row, "train", images_dir, labels_dir, stats)
    assert result.written is False
    assert stats["decode_error"] == 1


def test_duplicate_labels_are_removed(tmp_path: Path) -> None:
    ann = {"category_id": 2, "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]}
    stats = {"instances": 0, "dropped_invalid": 0, "dropped_duplicate": 0, "decode_error": 0}
    images_dir, labels_dir = _make_dirs(tmp_path)

    result = prepare_oamtcd.write_image_and_labels(
        _row(json.dumps([ann, ann]), image=_png_bytes()), "train", images_dir, labels_dir, stats
    )

    assert result.instances == 1
    assert stats["dropped_duplicate"] == 1
    assert len((labels_dir / "1.txt").read_text(encoding="ascii").splitlines()) == 1


# --- split assignment ---


def test_assign_split_folds_and_test() -> None:
    for fold in (0, 1, 2, 3):
        assert prepare_oamtcd.assign_split(fold, False) == "train"
    assert prepare_oamtcd.assign_split(4, False) == "val"
    assert prepare_oamtcd.assign_split(0, True) == "test"
    assert prepare_oamtcd.assign_split(4, True) == "test"
    with pytest.raises(ValueError):
        prepare_oamtcd.assign_split(5, False)


def test_row_split_uses_is_test_column() -> None:
    assert prepare_oamtcd.row_split({"validation_fold": 0, "_is_test": True}) == "test"
    assert prepare_oamtcd.row_split({"validation_fold": 0, "_is_test": False}) == "train"
    assert prepare_oamtcd.row_split({"validation_fold": 4, "_is_test": False}) == "val"


def test_no_oam_id_appears_in_multiple_splits() -> None:
    rows = [
        {"oam_id": "a", "validation_fold": 0, "_is_test": False},
        {"oam_id": "b", "validation_fold": 4, "_is_test": False},
        {"oam_id": "c", "validation_fold": 2, "_is_test": True},
        {"oam_id": "d", "validation_fold": 1, "_is_test": False},
        {"oam_id": "e", "validation_fold": 3, "_is_test": False},
    ]
    splits = {r["oam_id"]: prepare_oamtcd.row_split(r) for r in rows}
    assert len(splits) == len(rows)  # every oam_id assigned exactly once
    assert splits["c"] == "test"
    by_split: dict[str, set[str]] = defaultdict(set)
    for oid, sp in splits.items():
        by_split[sp].add(oid)
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])


def test_split_validation_rejects_cross_split_oam_id() -> None:
    oam_split = {"train": {"shared"}, "val": {"shared"}, "test": set()}
    with pytest.raises(ValueError, match="shared"):
        prepare_oamtcd._validate_oam_splits(oam_split)


# --- manifest & data.yaml ---


def test_manifest_fields_complete(tmp_path: Path) -> None:
    df = pl.DataFrame({"oam_id": ["a", "b"], "validation_fold": [0, 4]})
    stats = {"instances": 3, "dropped_invalid": 1, "decode_error": 0}
    split_counts = {"train": 1, "val": 0, "test": 1}
    split_instances = {"train": 2, "val": 0, "test": 1}
    oam_split = {"train": {"a"}, "val": set(), "test": {"b"}}
    prepare_oamtcd._write_manifest(
        tmp_path,
        df,
        stats,
        split_counts,
        split_instances,
        oam_split,
        None,
        revision="test-revision",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["repo_id"] == prepare_oamtcd.REPO_ID
    assert manifest["source"]["revision"] == "test-revision"
    assert manifest["class_map"] == {"tree-crown": 0}
    assert manifest["keep_category_ids"] == [2]
    assert manifest["drop_category_ids"] == [1]
    assert manifest["total_rows"] == 2
    assert "created_at" in manifest
    assert "train_commit" in manifest
    assert manifest["oam_split_guarantee"] == "no oam_id appears in more than one split"
    assert manifest["split"]["test"]["oam_ids"] == ["b"]
    assert manifest["split"]["train"]["n_instances"] == 2
    assert manifest["split"]["test"]["n_instances"] == 1


def test_data_yaml_single_class(tmp_path: Path) -> None:
    prepare_oamtcd._write_data_yaml(tmp_path)
    content = (tmp_path / "data.yaml").read_text(encoding="utf-8")
    assert "names:" in content
    assert "  0: tree-crown" in content
    assert "1:" not in content
    assert "path:" not in content


def test_prepare_forwards_revision_to_download(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "image_id": [1],
            "width": [100],
            "height": [100],
            "image": [b"image"],
            "coco_annotations": ["[]"],
            "validation_fold": [0],
            "oam_id": ["a"],
            "_is_test": [False],
        }
    )
    raw = tmp_path / "raw-source"
    with (
        mock.patch.object(prepare_oamtcd, "download_raw", return_value=raw) as download,
        mock.patch.object(prepare_oamtcd, "load_frame", return_value=frame),
    ):
        manifest = prepare_oamtcd.prepare(tmp_path / "out", revision="custom-revision")

    download.assert_called_once_with(tmp_path / "out", "custom-revision")
    assert manifest["source"]["revision"] == "custom-revision"


def test_download_raw_removes_stale_shards_before_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    stale = output / "raw" / "data" / "stale.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    def fake_snapshot_download(**kwargs) -> None:
        data_dir = Path(kwargs["local_dir"]) / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "train-00000.parquet").write_bytes(b"new")

    monkeypatch.setattr(prepare_oamtcd, "snapshot_download", fake_snapshot_download)
    prepare_oamtcd.download_raw(output, revision="new-revision")

    assert not stale.exists()
    source = json.loads((output / "raw" / "source.json").read_text(encoding="utf-8"))
    assert source["revision"] == "new-revision"
    assert [item["path"] for item in source["files"]] == ["data/train-00000.parquet"]


def test_prepare_refuses_existing_generated_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    (output / "images" / "train").mkdir(parents=True)
    (output / "images" / "train" / "stale.jpg").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="new output directory"):
        prepare_oamtcd.prepare(output, skip_download=True)


def test_polygon_area_positive() -> None:
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert prepare_oamtcd._polygon_area(pts) == pytest.approx(1.0)
