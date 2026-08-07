"""Stage non-deployable training checkpoints in a private Hugging Face repo.

Staging artifacts are for checkpoint backup, comparison, and later
fine-tuning. They deliberately use artifact.yaml instead of the production
model.yaml ABI contract and are always marked deployable: false.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from publish import DEFAULT_CREDENTIALS, _load_classes, _load_token_from_env, _sha256, _validate_train_commit

logger = logging.getLogger("tree-crown.stage")

DEFAULT_HF_STAGING_REPO = "zyzh0/tree-crown-yolo11-seg-staging"
PRODUCTION_TAG = re.compile(r"^v\d+\.\d+\.\d+(?:[-+].+)?$")
TAG_PREFIX = {"experimental": "exp-", "candidate": "cand-"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _validate_staging_tag(tag: str, lifecycle: str = "experimental") -> None:
    if PRODUCTION_TAG.fullmatch(tag):
        raise ValueError(f"tag {tag} is reserved for production releases")
    prefix = TAG_PREFIX[lifecycle]
    if not tag.startswith(prefix):
        raise ValueError(f"{lifecycle} tags must start with {prefix}")


def _best_metrics(results: Path) -> dict:
    with results.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    def best(metric: str) -> tuple[int, float] | None:
        values = [(int(float(row["epoch"])), float(row[metric])) for row in rows if row.get(metric)]
        return max(values, key=lambda item: item[1]) if values else None

    metrics: dict[str, float | int] = {}
    for key, output_name in (
        ("metrics/mAP50-95(B)", "best_box_map50_95"),
        ("metrics/mAP50-95(M)", "best_mask_map50_95"),
    ):
        value = best(key)
        if value is not None:
            metrics[output_name] = value[1]
            metrics[f"{output_name}_epoch"] = value[0]
    return metrics


def _write_checksums(bundle: Path) -> None:
    files = sorted(path for path in bundle.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    content = "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    (bundle / "SHA256SUMS").write_text(content, encoding="ascii")


def _build_bundle(
    *,
    bundle: Path,
    tag: str,
    lifecycle: str,
    stage_name: str,
    architecture: str,
    checkpoint: Path,
    config: Path,
    args: Path,
    results: Path,
    data_yaml: Path,
    dataset_manifest: Path,
    train_commit: str,
) -> dict:
    _validate_staging_tag(tag, lifecycle)
    _validate_train_commit(train_commit)
    for path in (checkpoint, config, args, results, data_yaml, dataset_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Staging input not found: {path}")

    bundle.mkdir(parents=True, exist_ok=False)
    copies = {
        checkpoint: "model.pt",
        config: "train-config.yaml",
        args: "train-args.yaml",
        results: "results.csv",
        data_yaml: "data.yaml",
        dataset_manifest: "dataset-manifest.json",
    }
    for source, name in copies.items():
        shutil.copy2(source, bundle / name)

    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    metadata = {
        "model_name": "tree-crown-yolo11-seg",
        "artifact_version": tag,
        "artifact_type": "pytorch-checkpoint",
        "lifecycle": lifecycle,
        "deployable": False,
        "stage": stage_name,
        "architecture": architecture,
        "checkpoint": {
            "file": "model.pt",
            "sha256": _sha256(bundle / "model.pt"),
        },
        "intended_use": "initialization-only" if lifecycle == "experimental" else "evaluation-only",
        "classes": _load_classes(str(data_yaml)),
        "dataset": dataset.get("source", {}),
        "train_commit": train_commit,
        "metrics": _best_metrics(results),
    }
    (bundle / "artifact.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    _write_checksums(bundle)
    return metadata


def _upload_bundle(bundle: Path, hf_repo: str, tag: str, token: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    create_repo(repo_id=hf_repo, token=token, private=True, exist_ok=True)
    api = HfApi(token=token)
    refs = api.list_repo_refs(repo_id=hf_repo, repo_type="model")
    if tag in {ref.name for ref in refs.tags}:
        raise RuntimeError(f"tag {tag} already exists; refusing to overwrite")

    prefix = f"artifacts/{tag}"
    operations = [
        CommitOperationAdd(path_in_repo=f"{prefix}/{path.name}", path_or_fileobj=str(path))
        for path in sorted(bundle.iterdir())
        if path.is_file()
    ]
    commit = api.create_commit(
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"stage {tag}",
        operations=operations,
    )
    api.create_tag(repo_id=hf_repo, repo_type="model", tag=tag, revision=commit.oid)
    logger.info("Staged artifact: repo=%s tag=%s commit=%s", hf_repo, tag, commit.oid)


def _validate_staging_repo(hf_repo: str) -> None:
    if hf_repo != DEFAULT_HF_STAGING_REPO:
        raise ValueError(f"Staging artifacts must use the dedicated staging repo {DEFAULT_HF_STAGING_REPO}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a non-deployable checkpoint in a private HF repository.")
    parser.add_argument("--tag", required=True, help="Immutable exp-* or cand-* staging tag.")
    parser.add_argument("--lifecycle", choices=sorted(TAG_PREFIX), default="experimental")
    parser.add_argument("--stage", required=True, help="Training stage, e.g. stage1a or stage2.")
    parser.add_argument("--architecture", required=True, help="Model architecture, e.g. yolo11n-seg.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--args", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--train-commit", required=True, help="Git commit used by the staged training run.")
    parser.add_argument("--token", help="HF token. Overrides token from --env.")
    parser.add_argument("--env", default=DEFAULT_CREDENTIALS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    try:
        _validate_staging_repo(DEFAULT_HF_STAGING_REPO)
        with tempfile.TemporaryDirectory(prefix="tree-crown-stage-") as tmp:
            bundle = Path(tmp) / args.tag
            metadata = _build_bundle(
                bundle=bundle,
                tag=args.tag,
                lifecycle=args.lifecycle,
                stage_name=args.stage,
                architecture=args.architecture,
                checkpoint=args.checkpoint,
                config=args.config,
                args=args.args,
                results=args.results,
                data_yaml=args.data,
                dataset_manifest=args.dataset_manifest,
                train_commit=args.train_commit,
            )
            logger.info("Staging %s (%s, deployable=%s)", args.tag, metadata["stage"], metadata["deployable"])
            token = args.token or _load_token_from_env(args.env)
            _upload_bundle(bundle, DEFAULT_HF_STAGING_REPO, args.tag, token)
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level error funnel
        logger.error("Staging failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
