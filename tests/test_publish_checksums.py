"""Tests for the publish.py checksum manifest generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from publish import (
    _build_meta,
    _build_release_bundle,
    _sha256,
    _validate_production_repo,
    _validate_release,
    _validate_train_commit,
    _write_checksums,
)


@pytest.fixture()
def release_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    onnx = tmp_path / "model.onnx"
    model_yaml = tmp_path / "model.yaml"
    checksums = tmp_path / "SHA256SUMS"
    onnx.write_bytes(b"ONNXBYTES\x00fake-model")
    model_yaml.write_text("model_name: tree-crown-yolo11-seg\nversion: v1.0.0\n", encoding="utf-8")
    return onnx, model_yaml, checksums


def test_manifest_has_two_expected_lines(release_files: tuple[Path, Path, Path]) -> None:
    onnx, model_yaml, checksums = release_files
    _write_checksums(onnx, model_yaml, checksums)
    lines = checksums.read_text(encoding="ascii").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("  model.onnx")
    assert lines[1].endswith("  model.yaml")


def test_manifest_digests_are_lowercase_hex_64(release_files: tuple[Path, Path, Path]) -> None:
    onnx, model_yaml, checksums = release_files
    _write_checksums(onnx, model_yaml, checksums)
    for line in checksums.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")
        assert name in ("model.onnx", "model.yaml")


def test_sha256_matches_sha256sum(release_files: tuple[Path, Path, Path]) -> None:
    onnx, model_yaml, checksums = release_files
    _write_checksums(onnx, model_yaml, checksums)
    lines = checksums.read_text(encoding="ascii").splitlines()
    expected = {
        f"{_sha256(onnx)}  model.onnx",
        f"{_sha256(model_yaml)}  model.yaml",
    }
    assert set(lines) == expected
    # Verify the manifest is a valid sha256sum --check input.
    import subprocess

    result = subprocess.run(
        ["sha256sum", "--check", "--strict", str(checksums)],
        cwd=onnx.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_byte_change_fails_verification(release_files: tuple[Path, Path, Path]) -> None:
    onnx, model_yaml, checksums = release_files
    _write_checksums(onnx, model_yaml, checksums)
    # Corrupt the ONNX after manifest generation.
    onnx.write_bytes(onnx.read_bytes() + b"X")
    import subprocess

    result = subprocess.run(
        ["sha256sum", "--check", "--strict", str(checksums)],
        cwd=onnx.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_release_requires_semver_tag() -> None:
    with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
        _validate_release("exp-stage1a-oamtcd-y11n-r1", ["tree-crown"])


def test_release_rejects_single_class_pretraining_abi() -> None:
    with pytest.raises(ValueError, match="exactly two classes"):
        _validate_release("v1.0.0", ["tree-crown"])


def test_release_requires_platanus_other_tree_class_order() -> None:
    with pytest.raises(ValueError, match="platanus.*other-tree"):
        _validate_release("v1.0.0", ["other-tree", "platanus"])


def test_release_metadata_is_explicitly_deployable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "publish._inspect_onnx",
        lambda _: {
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 1280, 1280]}],
            "outputs": [
                {"name": "output0", "dtype": "float32", "shape": [1, 38, 33600]},
                {"name": "output1", "dtype": "float32", "shape": [1, 32, 320, 320]},
            ],
        },
    )
    meta = _build_meta("v1.0.0", ["platanus", "other-tree"], "a" * 40, tmp_path / "model.onnx", "8.5.2")
    assert meta["lifecycle"] == "release"
    assert meta["deployable"] is True


def test_release_rejects_wrong_fixed_input_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "publish._inspect_onnx",
        lambda _: {
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 640, 640]}],
            "outputs": [{"name": "output0", "dtype": "float32", "shape": [1, 38, 8400]}],
        },
    )
    with pytest.raises(ValueError, match="fixed input contract"):
        _build_meta("v1.0.0", ["platanus", "other-tree"], "a" * 40, tmp_path / "model.onnx", "8.5.2")


def test_release_rejects_wrong_detection_channel_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "publish._inspect_onnx",
        lambda _: {
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 1280, 1280]}],
            "outputs": [
                {"name": "output0", "dtype": "float32", "shape": [1, 40, 33600]},
                {"name": "output1", "dtype": "float32", "shape": [1, 32, 320, 320]},
            ],
        },
    )
    with pytest.raises(ValueError, match="38 channels"):
        _build_meta("v1.0.0", ["platanus", "other-tree"], "a" * 40, tmp_path / "model.onnx", "8.5.2")


def test_release_rejects_non_float32_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "publish._inspect_onnx",
        lambda _: {
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 1280, 1280]}],
            "outputs": [
                {"name": "output0", "dtype": "float16", "shape": [1, 38, 33600]},
                {"name": "output1", "dtype": "float16", "shape": [1, 32, 320, 320]},
            ],
        },
    )
    with pytest.raises(ValueError, match="two float32 outputs"):
        _build_meta("v1.0.0", ["platanus", "other-tree"], "a" * 40, tmp_path / "model.onnx", "8.5.2")


def test_release_rejects_missing_mask_prototype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "publish._inspect_onnx",
        lambda _: {
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 1280, 1280]}],
            "outputs": [
                {"name": "output0", "dtype": "float32", "shape": [1, 38, 33600]},
                {"name": "output1", "dtype": "float32", "shape": [1, 16, 320, 320]},
            ],
        },
    )
    with pytest.raises(ValueError, match="mask prototype"):
        _build_meta("v1.0.0", ["platanus", "other-tree"], "a" * 40, tmp_path / "model.onnx", "8.5.2")


def test_train_commit_must_be_full_git_sha() -> None:
    with pytest.raises(ValueError, match="40-character"):
        _validate_train_commit("abc123")
    _validate_train_commit("a" * 40)


def test_release_bundle_renames_best_onnx_to_model_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "best.onnx"
    source.write_bytes(b"onnx")
    monkeypatch.setattr(
        "publish._build_meta",
        lambda *args: {
            "model_name": "tree-crown-yolo11-seg",
            "version": "v1.0.0",
            "lifecycle": "release",
            "deployable": True,
        },
    )

    bundle = _build_release_bundle(
        source,
        "v1.0.0",
        ["platanus", "other-tree"],
        "a" * 40,
        "8.5.2",
        tmp_path / "bundle",
    )

    assert bundle.onnx.name == "model.onnx"
    assert bundle.onnx.read_bytes() == b"onnx"
    lines = bundle.checksums.read_text(encoding="ascii").splitlines()
    assert lines[0].endswith("  model.onnx")


def test_release_rejects_prerelease_tag() -> None:
    with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
        _validate_release("v1.0.0-rc1", ["platanus", "other-tree"])


def test_production_repo_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="configured production"):
        _validate_production_repo("zyzh0/tree-crown-yolo11-seg-staging", "git@hf.co:other/repo")
