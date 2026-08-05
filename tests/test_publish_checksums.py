"""Tests for the publish.py checksum manifest generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from publish import _sha256, _write_checksums


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
