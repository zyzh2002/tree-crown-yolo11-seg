"""Tests for staging non-deployable training artifacts on Hugging Face."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

import stage


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("model: yolo11n-seg.pt\n", encoding="utf-8")
    args = tmp_path / "args.yaml"
    args.write_text("model: yolo11n-seg.pt\nimgsz: 1280\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text("names:\n  0: tree-crown\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"source": {"repo_id": "restor/tcd", "revision": "abc"}}\n', encoding="utf-8")
    results = tmp_path / "results.csv"
    with results.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "metrics/mAP50-95(B)": 0.2, "metrics/mAP50-95(M)": 0.1})
        writer.writerow({"epoch": 2, "metrics/mAP50-95(B)": 0.3, "metrics/mAP50-95(M)": 0.25})
    return {
        "checkpoint": checkpoint,
        "config": config,
        "args": args,
        "data": data,
        "manifest": manifest,
        "results": results,
    }


def test_staging_rejects_production_semver_tag() -> None:
    with pytest.raises(ValueError, match="reserved for production"):
        stage._validate_staging_tag("v1.0.0")


def test_staging_tag_must_match_lifecycle() -> None:
    with pytest.raises(ValueError, match="experimental tags must start"):
        stage._validate_staging_tag("cand-stage2-y11n-r1", lifecycle="experimental")


def test_staging_requires_full_training_commit() -> None:
    with pytest.raises(ValueError, match="40-character"):
        stage._validate_train_commit("abc123")


def test_staging_repo_cannot_be_overridden_to_production() -> None:
    with pytest.raises(ValueError, match="dedicated staging repo"):
        stage._validate_staging_repo("zyzh0/tree-crown-yolo11-seg")


def test_build_bundle_is_explicitly_non_deployable(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    bundle = tmp_path / "bundle"

    metadata = stage._build_bundle(
        bundle=bundle,
        tag="exp-stage1a-oamtcd-y11n-r1",
        lifecycle="experimental",
        stage_name="stage1a",
        architecture="yolo11n-seg",
        checkpoint=inputs["checkpoint"],
        config=inputs["config"],
        args=inputs["args"],
        results=inputs["results"],
        data_yaml=inputs["data"],
        dataset_manifest=inputs["manifest"],
        train_commit="a" * 40,
    )

    assert metadata["deployable"] is False
    assert metadata["intended_use"] == "initialization-only"
    assert metadata["classes"] == ["tree-crown"]
    assert metadata["metrics"]["best_mask_map50_95"] == pytest.approx(0.25)
    assert not (bundle / "model.yaml").exists()
    assert (bundle / "model.pt").exists()
    artifact = yaml.safe_load((bundle / "artifact.yaml").read_text(encoding="utf-8"))
    assert artifact["lifecycle"] == "experimental"
    checksum_lines = (bundle / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert any(line.endswith("  model.pt") for line in checksum_lines)
    assert any(line.endswith("  artifact.yaml") for line in checksum_lines)


def test_candidate_bundle_uses_evaluation_only_intent(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    metadata = stage._build_bundle(
        bundle=tmp_path / "bundle",
        tag="cand-stage2-xian-y11n-r1",
        lifecycle="candidate",
        stage_name="stage2",
        architecture="yolo11n-seg",
        checkpoint=inputs["checkpoint"],
        config=inputs["config"],
        args=inputs["args"],
        results=inputs["results"],
        data_yaml=inputs["data"],
        dataset_manifest=inputs["manifest"],
        train_commit="a" * 40,
    )
    assert metadata["intended_use"] == "evaluation-only"
