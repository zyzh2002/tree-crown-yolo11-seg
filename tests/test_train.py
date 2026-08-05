"""Tests for train.py config forwarding, override precedence, and metrics."""

from __future__ import annotations

from unittest import mock

import pytest

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
    config = {"data": "data.yaml", "imgsz": 1280, "batch": 4, "device": 0}
    yolo = _run(config, mode="validate", weights="runs/x/weights/best.pt")
    vkwargs = yolo.val.call_args.kwargs
    assert vkwargs["data"] == "data.yaml"
    assert vkwargs["imgsz"] == 1280
    assert vkwargs["batch"] == 4
    assert vkwargs["device"] == 0
    assert vkwargs["verbose"] is True


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
