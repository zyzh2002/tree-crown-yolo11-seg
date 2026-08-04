"""Publish a trained YOLO11-seg ONNX + model.yaml to a Hugging Face private repo.

The ONNX + model.yaml form the external ABI contract consumed by the onboard
repo. Publish requires an HF_TOKEN in the git-ignored .local/credentials.env.

Usage:
    python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("tree-crown.publish")

MODEL_NAME = "tree-crown-yolo11-seg"
CREDENTIALS_PATH = Path(".local/credentials.env")
DEFAULT_TARGET_TRT = "8.5.2"
DEFAULT_HF_REPO = "zyzh2002/tree-crown-yolo11-seg"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_hf_token() -> str:
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Missing credentials file {CREDENTIALS_PATH}. Create it with mode 600 containing HF_TOKEN=<token>."
        )
    token = None
    for line in CREDENTIALS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and not line.startswith("#"):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not token:
        raise RuntimeError(f"HF_TOKEN not found in {CREDENTIALS_PATH}.")
    return token


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError:
        logger.warning("Could not determine git HEAD; using 'unknown'.")
        return "unknown"


def _load_classes(data_yaml: str) -> list[str]:
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with Path(data_yaml).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    names = cfg.get("names") or {}
    return [names[str(i)] for i in sorted(int(k) for k in names)]


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


def _build_model_yaml(
    tag: str,
    classes: list[str],
    train_commit: str,
    onnx_path: Path,
    target_trt: str,
) -> dict:
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


def _publish(
    tag: str,
    onnx_path: str,
    data_yaml: str,
    hf_repo: str,
    target_trt: str,
    token: str,
) -> None:
    from huggingface_hub import HfApi, create_repo

    onnx = Path(onnx_path)
    if not onnx.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx}")

    classes = _load_classes(data_yaml)
    train_commit = _git_commit()
    meta = _build_model_yaml(tag, classes, train_commit, onnx, target_trt)
    model_yaml_path = onnx.with_suffix(".yaml").with_name("model.yaml")

    logger.info("Publishing to HF repo: %s (tag %s)", hf_repo, tag)
    logger.info("Classes: %s", classes)
    logger.info("train_commit: %s", train_commit)

    create_repo(repo_id=hf_repo, token=token, private=True, exist_ok=True)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(onnx),
        path_in_repo=onnx.name,
        repo_id=hf_repo,
        repo_type="model",
    )
    logger.info("Uploaded %s", onnx.name)

    with model_yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=False)
    api.upload_file(
        path_or_fileobj=str(model_yaml_path),
        path_in_repo=model_yaml_path.name,
        repo_id=hf_repo,
        repo_type="model",
    )
    logger.info("Uploaded %s", model_yaml_path.name)
    logger.info("Publish complete. Handoff: repo=%s tag=%s", hf_repo, tag)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish ONNX + model.yaml to an HF private repo.")
    parser.add_argument("--tag", required=True, help="Version tag, e.g. v1.0.0.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights (or exported .onnx).")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml for the class list.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HF private repo id.")
    parser.add_argument("--target-trt", default=DEFAULT_TARGET_TRT, help="Onboard TensorRT baseline version.")
    parser.add_argument("--token", help="HF_TOKEN (overrides .local/credentials.env).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        token = args.token or _load_hf_token()
        # sanity: never log the token
        os.environ.setdefault("HF_TOKEN", token)
        _publish(
            tag=args.tag,
            onnx_path=args.weights,
            data_yaml=args.data,
            hf_repo=args.hf_repo,
            target_trt=args.target_trt,
            token=token,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Publish failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
