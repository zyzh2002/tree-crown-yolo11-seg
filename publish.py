"""Publish a trained YOLO11-seg ONNX + model.yaml to a Hugging Face private repo.

The ONNX + model.yaml form the external ABI contract consumed by the onboard
repo. Publishing uses a Hugging Face token (huggingface_hub) by default, read
from .local/credentials.env (mode 600, git-ignored) unless --token is given.
An SSH fallback (git push to git@hf.co) is available via --ssh.

ONNX files are large and stored on HF via Git LFS (the repo .gitattributes
already maps *.onnx to LFS). The token path handles LFS automatically; the SSH
path requires git-lfs installed and 'git lfs install' run once.

Usage:
    python publish.py --tag v1.0.0 --weights <best.pt> --train-commit <git-sha>
    python publish.py --tag v1.0.0 --origin <model.onnx> --train-commit <git-sha>
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("tree-crown.publish")

MODEL_NAME = "tree-crown-yolo11-seg"
DEFAULT_TARGET_TRT = "8.5.2"
DEFAULT_HF_REPO = "zyzh0/tree-crown-yolo11-seg"
DEFAULT_HF_SSH = "git@hf.co:zyzh0/tree-crown-yolo11-seg"
DEFAULT_CREDENTIALS = ".local/credentials.env"
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
RELEASE_CLASSES = ["platanus", "other-tree"]


@dataclass(frozen=True)
class ReleaseBundle:
    onnx: Path
    model_yaml: Path
    checksums: Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_token_from_env(env_path: str) -> str:
    """Read HF_TOKEN from a credentials env file (git-ignored)."""
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing credentials file {path}. Create it with mode 600 containing HF_TOKEN=<token>."
        )
    token: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and not line.startswith("#"):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not token:
        raise RuntimeError(f"HF_TOKEN not found in {path}.")
    return token


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _load_classes(data_yaml: str) -> list[str]:
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with Path(data_yaml).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    names = cfg.get("names") or {}
    # YAML may parse numeric class keys as int; normalize to str for lookup.
    normalized = {str(k): v for k, v in names.items()}
    return [normalized[str(i)] for i in sorted(int(k) for k in normalized)]


def _inspect_onnx(onnx_path: Path) -> dict:
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inputs = []
    for inp in graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)).name
        inputs.append({"name": inp.name, "dtype": dtype, "shape": dims})
    outputs = []
    for out in graph.output:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(out.type.tensor_type.elem_type)).name
        outputs.append({"name": out.name, "dtype": dtype, "shape": dims})
    return {"inputs": inputs, "outputs": outputs}


def _write_model_yaml(meta: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=False)


def _sha256(path: Path) -> str:
    """Streaming SHA-256 of a file, safe for large model weights."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(onnx: Path, model_yaml: Path, output: Path) -> None:
    """Write a SHA256SUMS manifest for the two release artifacts."""
    output.write_text(
        f"{_sha256(onnx)}  {onnx.name}\n{_sha256(model_yaml)}  {model_yaml.name}\n",
        encoding="ascii",
    )


def _build_meta(tag: str, classes: list[str], train_commit: str, onnx_path: Path, target_trt: str) -> dict:
    info = _inspect_onnx(onnx_path)
    expected_input = {"name": "images", "dtype": "float32", "shape": [1, 3, 1280, 1280]}
    if info["inputs"] != [expected_input]:
        raise ValueError(f"ONNX does not match fixed input contract: expected {expected_input!r}")
    if len(info["outputs"]) != 2 or any(output["dtype"] != "float32" for output in info["outputs"]):
        raise ValueError("YOLO11-seg ONNX must expose exactly two float32 outputs")
    if not any(output["shape"] == [1, 38, 33600] for output in info["outputs"]):
        raise ValueError("Two-class YOLO11-seg detection output must contain 38 channels")
    if not any(output["shape"] == [1, 32, 320, 320] for output in info["outputs"]):
        raise ValueError("YOLO11-seg ONNX must expose the [1, 32, 320, 320] mask prototype output")
    return {
        "model_name": MODEL_NAME,
        "version": tag,
        "lifecycle": "release",
        "deployable": True,
        "input": info["inputs"][0],
        "outputs": info["outputs"],
        "classes": classes,
        "train_commit": train_commit,
        "target_trt": target_trt,
    }


def _validate_release(tag: str, classes: list[str]) -> None:
    """Enforce the production ABI and keep experimental artifacts in staging."""
    if not RELEASE_TAG.fullmatch(tag):
        raise ValueError("Production tag must use vMAJOR.MINOR.PATCH format")
    if len(classes) != 2:
        raise ValueError("The first deployable release must contain exactly two classes")
    if classes != RELEASE_CLASSES:
        raise ValueError(f"Production classes must be ordered as {RELEASE_CLASSES!r}")


def _validate_train_commit(train_commit: str) -> None:
    if not GIT_SHA.fullmatch(train_commit):
        raise ValueError("train_commit must be a full 40-character Git SHA")


def _validate_production_repo(hf_repo: str, hf_ssh: str) -> None:
    if hf_repo != DEFAULT_HF_REPO or hf_ssh != DEFAULT_HF_SSH:
        raise ValueError("Production releases must use the configured production Hugging Face repository")


def _build_release_bundle(
    source_onnx: Path,
    tag: str,
    classes: list[str],
    train_commit: str,
    target_trt: str,
    bundle_dir: Path,
) -> ReleaseBundle:
    """Create the fixed-name production bundle consumed by the onboard repo."""
    bundle_dir.mkdir(parents=True, exist_ok=False)
    onnx = bundle_dir / "model.onnx"
    shutil.copy2(source_onnx, onnx)
    model_yaml = bundle_dir / "model.yaml"
    _write_model_yaml(_build_meta(tag, classes, train_commit, onnx, target_trt), model_yaml)
    checksums = bundle_dir / "SHA256SUMS"
    _write_checksums(onnx, model_yaml, checksums)
    return ReleaseBundle(onnx=onnx, model_yaml=model_yaml, checksums=checksums)


def _publish_ssh(
    onnx: Path,
    model_yaml: Path,
    checksums: Path,
    hf_ssh: str,
    tag: str,
) -> None:
    """Clone the HF repo, copy the artifacts, commit, tag, and push over SSH."""
    with tempfile.TemporaryDirectory(prefix="hf-publish-") as tmp:
        work = Path(tmp)
        _run_git(["clone", hf_ssh, str(work / "repo")])
        repo = work / "repo"
        shutil.copy2(onnx, repo / onnx.name)
        shutil.copy2(model_yaml, repo / model_yaml.name)
        shutil.copy2(checksums, repo / checksums.name)
        _run_git(["add", onnx.name, model_yaml.name, checksums.name], cwd=repo)
        _run_git(["commit", "-m", f"publish {tag}: {onnx.name} + {model_yaml.name} + {checksums.name}"], cwd=repo)
        # Refuse to move an existing tag.
        try:
            _run_git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=repo)
            raise RuntimeError(f"tag {tag} already exists; refusing to overwrite")
        except subprocess.CalledProcessError:
            pass
        _run_git(["tag", tag], cwd=repo)
        _run_git(["push", "origin", "main", "--tags"], cwd=repo)
        logger.info("Pushed tag %s to %s", tag, hf_ssh)


def _publish_token(
    onnx: Path,
    model_yaml: Path,
    checksums: Path,
    hf_repo: str,
    tag: str,
    token: str,
) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    logger.info("Publishing via token to %s (tag %s)", hf_repo, tag)
    create_repo(repo_id=hf_repo, token=token, private=True, exist_ok=True)
    api = HfApi(token=token)

    # Guard against moving an existing immutable tag.
    existing = [t.name for t in api.list_repo_refs(repo_id=hf_repo, repo_type="model").tags]
    if tag in existing:
        raise RuntimeError(f"tag {tag} already exists; refusing to overwrite")

    commit = api.create_commit(
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"publish {tag}",
        operations=[
            CommitOperationAdd(path_in_repo=onnx.name, path_or_fileobj=str(onnx)),
            CommitOperationAdd(path_in_repo=model_yaml.name, path_or_fileobj=str(model_yaml)),
            CommitOperationAdd(path_in_repo=checksums.name, path_or_fileobj=str(checksums)),
        ],
    )
    logger.info("Committed %s + %s + %s (oid %s)", onnx.name, model_yaml.name, checksums.name, commit.oid)
    api.create_tag(
        repo_id=hf_repo,
        repo_type="model",
        tag=tag,
        revision=commit.oid,
    )
    logger.info("Publish complete via token. Handoff: repo=%s tag=%s commit=%s", hf_repo, tag, commit.oid)


def _resolve_onnx(weights: str) -> Path:
    """If --weights is a .pt, find the sibling exported .onnx; else use it directly."""
    p = Path(weights)
    if p.suffix == ".onnx":
        return p
    # ultralytics exports best.onnx next to best.pt
    onnx_candidate = p.with_suffix(".onnx")
    if onnx_candidate.exists():
        return onnx_candidate
    raise FileNotFoundError(
        f"No ONNX found for {p}. Export it first (python export.py --weights {p}) or pass a path to an existing .onnx."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish ONNX + model.yaml to an HF private repo.")
    parser.add_argument("--tag", required=True, help="Version tag, e.g. v1.0.0.")
    parser.add_argument("--weights", help="Path to trained .pt (resolves sibling .onnx) or a .onnx directly.")
    parser.add_argument("--origin", help="Explicit path to the exported ONNX (alias for --weights).")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml for the class list.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HF private repo id (token path).")
    parser.add_argument("--hf-ssh", default=DEFAULT_HF_SSH, help="HF SSH URL (SSH path).")
    parser.add_argument("--target-trt", default=DEFAULT_TARGET_TRT, help="Onboard TensorRT baseline version.")
    parser.add_argument("--train-commit", required=True, help="Git commit used by the released training run.")
    parser.add_argument("--token", help="HF_TOKEN. Overrides token from --env.")
    parser.add_argument(
        "--env", default=DEFAULT_CREDENTIALS, help="File with HF_TOKEN (default: .local/credentials.env)."
    )
    parser.add_argument("--ssh", action="store_true", help="Use SSH git push instead of token.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        _validate_production_repo(args.hf_repo, args.hf_ssh)
        onnx = _resolve_onnx(args.origin or args.weights)
        if not onnx.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx}")
        classes = _load_classes(args.data)
        _validate_release(args.tag, classes)
        _validate_train_commit(args.train_commit)
        train_commit = args.train_commit
        with tempfile.TemporaryDirectory(prefix="tree-crown-release-") as tmp:
            bundle = _build_release_bundle(
                onnx,
                args.tag,
                classes,
                train_commit,
                args.target_trt,
                Path(tmp) / args.tag,
            )
            logger.info("Publishing to HF (tag %s): %s", args.tag, bundle.onnx.name)
            logger.info("Classes: %s", classes)
            logger.info("train_commit: %s", train_commit)

            if args.ssh:
                _publish_ssh(bundle.onnx, bundle.model_yaml, bundle.checksums, args.hf_ssh, args.tag)
            else:
                token = args.token or _load_token_from_env(args.env)
                _publish_token(bundle.onnx, bundle.model_yaml, bundle.checksums, args.hf_repo, args.tag, token)
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Publish failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
