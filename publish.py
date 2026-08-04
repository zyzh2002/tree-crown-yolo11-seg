"""Publish a trained YOLO11-seg ONNX + model.yaml to a Hugging Face private repo.

The ONNX + model.yaml form the external ABI contract consumed by the onboard
repo. Publishing is done over SSH (git push) by default, which uses the local
SSH key and needs no token. A token-based path (huggingface_hub) is available
as a fallback via --token or HF_TOKEN in .local/credentials.env.

ONNX files are large and stored on HF via Git LFS (the repo .gitattributes
already maps *.onnx to LFS). git-lfs must be installed and 'git lfs install'
run once.

Usage:
    python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt
    python publish.py --tag v1.0.0 --origin <path-to-onnx>  # publish an existing ONNX
    python publish.py --tag v1.0.0 --weights <onnx> --token <token>  # token fallback
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger("tree-crown.publish")

MODEL_NAME = "tree-crown-yolo11-seg"
DEFAULT_TARGET_TRT = "8.5.2"
DEFAULT_HF_REPO = "zyzh0/tree-crown-yolo11-seg"
DEFAULT_HF_SSH = "git@hf.co:zyzh0/tree-crown-yolo11-seg"


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


def _git_commit() -> str:
    try:
        return _run_git(["rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        logger.warning("Could not determine git HEAD; using 'unknown'.")
        return "unknown"


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

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inputs = []
    for inp in graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim]
        inputs.append({"name": inp.name, "dtype": "float32", "shape": dims})
    outputs = []
    for out in graph.output:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in out.type.tensor_type.shape.dim]
        outputs.append({"name": out.name, "dtype": "float32", "shape": dims})
    return {"inputs": inputs, "outputs": outputs}


def _write_model_yaml(meta: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=False)


def _build_meta(tag: str, classes: list[str], train_commit: str, onnx_path: Path, target_trt: str) -> dict:
    info = _inspect_onnx(onnx_path)
    return {
        "model_name": MODEL_NAME,
        "version": tag,
        "input": info["inputs"][0],
        "outputs": info["outputs"],
        "classes": classes,
        "train_commit": train_commit,
        "target_trt": target_trt,
    }


def _publish_ssh(
    onnx: Path,
    model_yaml: Path,
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
        _run_git(["add", onnx.name, model_yaml.name], cwd=repo)
        _run_git(["commit", "-m", f"publish {tag}: {onnx.name} + {model_yaml.name}"], cwd=repo)
        _run_git(["tag", tag], cwd=repo)
        _run_git(["push", "origin", "main", "--tags"], cwd=repo)
        logger.info("Pushed tag %s to %s", tag, hf_ssh)


def _publish_token(
    onnx: Path,
    model_yaml: Path,
    hf_repo: str,
    tag: str,
    token: str,
) -> None:
    from huggingface_hub import HfApi, create_repo

    logger.info("Publishing via token to %s (tag %s)", hf_repo, tag)
    create_repo(repo_id=hf_repo, token=token, private=True, exist_ok=True)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(onnx),
        path_in_repo=onnx.name,
        repo_id=hf_repo,
        repo_type="model",
    )
    logger.info("Uploaded %s", onnx.name)
    api.upload_file(
        path_or_fileobj=str(model_yaml),
        path_in_repo=model_yaml.name,
        repo_id=hf_repo,
        repo_type="model",
    )
    logger.info("Uploaded %s", model_yaml.name)
    logger.info("Publish complete via token. Handoff: repo=%s tag=%s", hf_repo, tag)


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
    parser.add_argument("--token", help="HF_TOKEN. When set, uses token upload instead of SSH.")
    parser.add_argument("--env", help="Read HF_TOKEN from an env file (e.g. .local/credentials.env).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        onnx = _resolve_onnx(args.origin or args.weights)
        if not onnx.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx}")
        classes = _load_classes(args.data)
        train_commit = _git_commit()
        meta = _build_meta(args.tag, classes, train_commit, onnx, args.target_trt)
        model_yaml = onnx.parent / "model.yaml"
        _write_model_yaml(meta, model_yaml)

        logger.info("Publishing to HF (tag %s): %s", args.tag, onnx.name)
        logger.info("Classes: %s", classes)
        logger.info("train_commit: %s", train_commit)

        if args.token:
            _publish_token(onnx, model_yaml, args.hf_repo, args.tag, args.token)
        elif args.env:
            token = _load_token_from_env(args.env)
            _publish_token(onnx, model_yaml, args.hf_repo, args.tag, token)
        else:
            _publish_ssh(onnx, model_yaml, args.hf_ssh, args.tag)
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Publish failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
