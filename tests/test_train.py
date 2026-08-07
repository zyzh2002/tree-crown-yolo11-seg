"""Tests for train.py config forwarding, override precedence, and metrics."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml

import train


def _run(config: dict, mode: str = "train", **overrides) -> mock.MagicMock:
    """Run _run_training with a mocked ultralytics.YOLO and return the mock."""
    # Patch the lazily imported YOLO class inside _run_training.
    with mock.patch("ultralytics.YOLO") as yolo_cls:
        yolo = yolo_cls.return_value
        train._run_training(config, mode, **overrides)
    return yolo


def test_train_forwarded_kwargs_include_all_keys() -> None:
    config = {
        "model": "yolo11n-seg.pt",
        "data": "data.yaml",
        "imgsz": 1280,
        "epochs": 100,
        "batch": 4,
        "device": 0,
        "seed": 0,
        "deterministic": True,
        "workers": 2,
        "cache": False,
        "amp": True,
        "patience": 20,
        "project": "runs",
        "name": "train",
        "plots": True,
        "single_cls": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "freeze": 10,
        "warmup_epochs": 2,
        "close_mosaic": 10,
        "max_det": 1000,
    }
    yolo = _run(config)
    kwargs = yolo.train.call_args.kwargs
    for key in (
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
        "max_det",
    ):
        assert key in kwargs
    assert kwargs["deterministic"] is True
    assert kwargs["workers"] == 2
    assert kwargs["single_cls"] is False


def test_override_precedence_cli_wins() -> None:
    config = {"model": "yolo11n-seg.pt", "data": "data.yaml", "imgsz": 1280, "batch": 16, "workers": 8, "amp": True}
    yolo = _run(config, imgsz=640, batch=4)
    kwargs = yolo.train.call_args.kwargs
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 4
    assert kwargs["workers"] == 8
    assert kwargs["amp"] is True


def test_falsy_overrides_are_preserved() -> None:
    config = {"model": "yolo11n-seg.pt", "data": "data.yaml", "batch": 16, "workers": 8, "amp": True}
    yolo = _run(
        config,
        batch=0,
        workers=0,
        amp=False,
        cache=False,
        patience=0,
        plots=False,
        deterministic=False,
        seed=0,
        device=0,
    )
    kwargs = yolo.train.call_args.kwargs
    assert kwargs["batch"] == 0
    assert kwargs["workers"] == 0
    assert kwargs["amp"] is False
    assert kwargs["cache"] is False
    assert kwargs["patience"] == 0
    assert kwargs["plots"] is False
    assert kwargs["deterministic"] is False
    assert kwargs["seed"] == 0
    assert kwargs["device"] == 0


def test_defaults_when_config_missing() -> None:
    yolo = _run({"model": "yolo11n-seg.pt"})
    kwargs = yolo.train.call_args.kwargs
    assert kwargs["imgsz"] == 1280
    assert kwargs["batch"] == 16
    assert kwargs["workers"] == 8
    assert kwargs["deterministic"] is True
    # Empty project lets ultralytics prepend runs/<task>/ without a redundant
    # nested runs/segment/runs/... path.
    assert kwargs["project"] == ""


def test_validate_forwarded_kwargs() -> None:
    config = {
        "data": "data.yaml",
        "imgsz": 1280,
        "batch": 4,
        "device": 0,
        "split": "test",
        "max_det": 1000,
        "iou": 0.6,
        "conf": 0.2,
        "save_json": True,
        "plots": False,
    }
    yolo = _run(config, mode="validate", weights="runs/x/weights/best.pt")
    vkwargs = yolo.val.call_args.kwargs
    assert vkwargs["data"] == "data.yaml"
    assert vkwargs["imgsz"] == 1280
    assert vkwargs["batch"] == 4
    assert vkwargs["device"] == 0
    assert vkwargs["verbose"] is True
    assert vkwargs["split"] == "test"
    assert vkwargs["max_det"] == 1000
    assert vkwargs["iou"] == 0.6
    assert vkwargs["conf"] == 0.2
    assert vkwargs["save_json"] is True
    assert vkwargs["plots"] is False


def test_validate_requires_weights() -> None:
    with mock.patch("ultralytics.YOLO"), pytest.raises(ValueError, match="--weights is required"):
        train._run_training({"data": "data.yaml"}, "validate")


def test_validate_logs_box_and_mask_map(caplog: pytest.LogCaptureFixture) -> None:
    with mock.patch("ultralytics.YOLO") as yolo_cls:
        results = mock.MagicMock()
        results.box.map = 0.5
        results.seg.map = 0.4
        yolo_cls.return_value.val.return_value = results
        with caplog.at_level("INFO"):
            train._run_training({"data": "data.yaml"}, "validate", weights="w.pt")
    assert "box mAP50-95: 0.5000" in caplog.text
    assert "mask mAP50-95: 0.4000" in caplog.text


def test_train_reports_real_save_dir(caplog: pytest.LogCaptureFixture) -> None:
    with mock.patch("ultralytics.YOLO") as yolo_cls:
        yolo = yolo_cls.return_value
        yolo.trainer.save_dir = "runs/segment/oamtcd-pretrain"
        yolo.trainer.best = "runs/segment/oamtcd-pretrain/weights/best.pt"
        with caplog.at_level("INFO"):
            train._run_training({"model": "yolo11n-seg.pt", "data": "data.yaml"}, "train")
    assert "runs/segment/oamtcd-pretrain" in caplog.text
    assert "weights/best.pt" in caplog.text


def test_model_fallback_name_is_yolo11n_seg() -> None:
    with mock.patch("ultralytics.YOLO") as yolo_cls:
        train._run_training({"data": "data.yaml"}, "train")
    assert yolo_cls.call_args.args[0] == "yolo11n-seg.pt"


def test_resume_requires_explicit_checkpoint_path() -> None:
    with pytest.raises(ValueError, match="explicit checkpoint"):
        _run({"model": "yolo11n-seg.pt", "data": "data.yaml"}, resume=True)


def test_resume_loads_explicit_checkpoint_path() -> None:
    with mock.patch("ultralytics.YOLO") as yolo_cls:
        train._run_training(
            {"model": "yolo11n-seg.pt", "data": "data.yaml"},
            "train",
            resume="runs/segment/x/weights/last.pt",
        )
    assert yolo_cls.call_args.args[0] == "runs/segment/x/weights/last.pt"
    assert yolo_cls.return_value.train.call_args.kwargs["resume"] is True


def test_resume_provenance_does_not_claim_current_config_is_effective() -> None:
    captured: dict = {}
    with (
        mock.patch("ultralytics.YOLO"),
        mock.patch.object(
            train,
            "_capture_provenance",
            side_effect=lambda **kwargs: captured.update(kwargs["resolved"]) or {},
        ),
        mock.patch.object(train, "_write_provenance"),
    ):
        train._run_training(
            {"model": "yolo11n-seg.pt", "data": "new-data.yaml", "lr0": 0.123},
            "train",
            resume="runs/segment/x/weights/last.pt",
        )
    assert captured == {"resume": True, "checkpoint": "runs/segment/x/weights/last.pt"}


def test_rejects_non_segment_task() -> None:
    with pytest.raises(ValueError, match="task must be 'segment'"):
        _run({"task": "detect", "model": "yolo11n.pt"})


def test_capture_provenance_records_start_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "base.pt"
    model.write_bytes(b"base")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    data_yaml = data_dir / "data.yaml"
    data_yaml.write_text("path: .\nnames:\n  0: tree-crown\n", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("lock\n", encoding="utf-8")
    monkeypatch.setattr(train, "_git_state", lambda: {"commit": "abc", "branch": "feat/x", "dirty": True})
    monkeypatch.setattr(train, "_package_versions", lambda: {"ultralytics": "8.4.115"})

    provenance = train._capture_provenance(
        config={"model": str(model), "data": str(data_yaml), "epochs": 50},
        resolved={"data": str(data_yaml), "batch": 4, "max_det": 1000},
        model_path=str(model),
        lock_path=lock,
    )
    assert provenance["git"]["dirty"] is True
    assert provenance["config"]["epochs"] == 50
    assert provenance["resolved_args"]["max_det"] == 1000
    assert len(provenance["initial_weights"]["sha256"]) == 64
    assert len(provenance["dataset_manifest"]["sha256"]) == 64
    assert len(provenance["uv_lock"]["sha256"]) == 64


def test_write_provenance_uses_pretraining_snapshot(tmp_path: Path) -> None:
    snapshot = {"git": {"commit": "before", "dirty": False}, "resolved_args": {"batch": 4}}
    train._write_provenance(tmp_path / "run", snapshot)
    written = yaml.safe_load((tmp_path / "run" / "provenance.yaml").read_text(encoding="utf-8"))
    assert written == snapshot


def test_training_captures_provenance_before_model_train() -> None:
    events: list[str] = []
    with (
        mock.patch("ultralytics.YOLO") as yolo_cls,
        mock.patch.object(
            train, "_capture_provenance", side_effect=lambda **_: events.append("capture") or {}
        ) as capture,
        mock.patch.object(train, "_write_provenance"),
    ):
        yolo_cls.return_value.train.side_effect = lambda **_: events.append("train")
        train._run_training({"model": "yolo11n-seg.pt", "data": "data.yaml"}, "train")
    assert capture.called
    assert events[:2] == ["capture", "train"]
