# AGENTS.md

Instructions for AI coding agents working on this repository.

## Scope

This repository is the **training** side of the tree-crown detection stack
(`tree-crown-training`). It is the PC-side **producer** of the YOLO11-seg model
artifact: dataset → train/validate → export ONNX → publish to a Hugging Face (HF)
private model repo. It does **not** do any onboard inference.

The onboard consumer is the separate C++ project
[manifold-3-vision-detect](https://github.com/zyzh2002/manifold-3-vision-detect).
The two repositories share no code. The only handoff point is the **HF private model
repo**: this repo produces ONNX + `model.yaml`, the onboard repo pulls them via
`fetch_model.sh` and builds a device `.engine` with `trtexec`.

## Workflow Boundary (must follow)

```
training data -> train/validate -> optional staging backup -> export ONNX -> publish to HF private repo
    -> onboard fetch_model.sh pulls -> trtexec builds .engine
```

- This repo's endpoint is the **HF private repo**. It never deploys to a device.
- Non-deployable checkpoints may be uploaded to the separate private staging
  repo `zyzh0/tree-crown-yolo11-seg-staging` with `stage.py`. Staging artifacts
  must use `deployable: false` and must never be consumed onboard.
- **Datasets never enter git.** `/data` is git-ignored (or managed with DVC).
- **ONNX weights never enter git.** Final artifacts are uploaded by `publish.py`
  to HF; never committed to the repo.
- Once published, ONNX + `model.yaml` form the external contract. Any change must
  **bump the version tag**; never overwrite silently.

## Language & Style

- Use English for internal reasoning and Chinese for all user-facing communication.
- **All code comments, commit messages, and agent-facing documentation must be in
  English.** Chinese characters are forbidden in source files, comments, commit
  messages, and agent-facing docs.
- Human-facing documentation under `docs/` is written in Chinese (see below). It is
  the human handover language and is exempt from the English-only rule.
- Python formatting follows `ruff` (config in `pyproject.toml`). Do not carry over the
  onboard C++ `.clang-format`.

## Documentation Layout

- `docs/` — **Human-facing, Chinese.** Onboarding, project status, development,
  and publishing workflows. Start at `docs/README.md`.
- `.agents/docs/` — **Agent-facing, English.** Working artifacts: design specs
  (`specs/`), implementation plans (`plans/`), and handoffs (`handoffs/`).
  Training and experiment records live under `experiments/` and are the
  authoritative source for run metrics and artifact hashes; specs reference
  them instead of duplicating performance numbers.
- `AGENTS.md` / `CLAUDE.md` — Agent instructions at the repository root.
- `README.md` — Human quick start, kept short; links into `docs/`.

Agents write working artifacts under `.agents/docs/`, never under `docs/`. Human
docs are updated deliberately, in Chinese.

## Skills

This repository ships agent skills under `.agents/skills/`. Agents are encouraged
to consult them when a task matches a skill's description:

- Creative work / new features: `brainstorming`, `writing-plans`
- Bug fixes: `systematic-debugging`
- Commits: `git-commit` / `conventional-commit`
- Hugging Face operations (publish/stage/download): `hf-cli`
- Before claiming completion: `verification-before-completion`

Loading a matching skill before starting the work is preferred, but user
instructions always take precedence.

## Commands

### Setup

```bash
uv sync                     # or: pip install -e .
```

- Python is pinned to 3.12 (`.python-version`); `torch`/`torchvision` resolve from the
  PyTorch **cu126** index via `[tool.uv.sources]` in `pyproject.toml`.
- **V100 (compute capability 7.0) requirement:** torch builds for cu128/cu130 do NOT ship
  Volta (sm_70) kernels and fail with `no kernel image is available`. Only cu126 (and older)
  builds support V100. Do not remove the cu126 index pin.
- After editing `pyproject.toml`, rebuild from scratch: `rm -rf .venv uv.lock && uv sync --extra dev --extra data`.

### Train

```bash
python train.py --config configs/default.yaml
```

### Validate

```bash
python train.py --config configs/default.yaml --mode validate
```

### Export ONNX

```bash
python export.py --weights runs/segment/train/weights/best.pt --imgsz 1280
```

### Publish to HF

```bash
python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt --train-commit <git-sha>
# Token (default): reads HF_TOKEN from .local/credentials.env.
# SSH alternative: python publish.py --tag v1.0.0 --weights <onnx> --train-commit <git-sha> --ssh
```

### Stage a checkpoint

```bash
python stage.py --tag exp-stage1a-oamtcd-y11n-r1 --stage stage1a \
  --architecture yolo11n-seg --checkpoint <best.pt> --config <config.yaml> \
  --args <args.yaml> --results <results.csv> --data <data.yaml> \
  --dataset-manifest <manifest.json> --train-commit <training-git-sha>
```

- Staging is checkpoint storage, not deployment publication.
- Only `publish.py` may create a deployable `vX.Y.Z` production artifact.

### Lint

```bash
ruff check .
ruff format --check .
```

## Credential Management

- Publishing to HF uses a **token by default** (`huggingface_hub` upload), read
  from the git-ignored `.local/credentials.env` (mode `600`, never committed).
  Override with `--token` or `--env`.
- An SSH fallback is available via `--ssh` (`git push` to `git@hf.co`), which
  needs `git-lfs` installed and `git lfs install` run once (HF stores `*.onnx`
  via LFS).
- Never log or commit a token.

## Git Workflow

- Commit messages in English, prefixed with Conventional Commits:
  `chore:`, `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Rebase over merge for linear history.

### Branch Policy

- Lightweight trunk-based workflow. `main` must always be usable.
- Normal implementation work updates `main` with one short-lived branch per objective.
- Branch prefixes: `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`, `test/`, followed
  by a short kebab-case topic, e.g. `feat/train-yolo11-seg`.
- Do not create long-lived `develop`, `integration`, `release`, or agent-specific branches.
- Rebase the development branch onto the latest `main` before integration; do not create
  merge commits.
- Agents may create local branches and commits as needed for an approved task.
- Agents must obtain explicit user approval before pushing, creating a pull request,
  merging into `main`, or deleting a remote branch.
- Direct commits to `main` require explicit user authorization for the specific task.
- Delete short-lived branches after their changes are integrated.

## Model Artifact Contract

`publish.py` uploads `model.yaml` alongside the ONNX. It is the machine-readable ABI
contract for the onboard `TensorRtEngine` (which replaces the hardcoded
`synthetic_engine_contract.h` in the onboard repo). Required fields:

```yaml
model_name: tree-crown-yolo11-seg
version: "<tag>"
input: { name: "images", dtype: "float32", shape: [1, 3, 1280, 1280] }
outputs: { name: "...", dtype: "...", shape: [...] }   # per actual export
classes: [ ... ]          # identical to data.yaml
train_commit: "<git sha>" # traceable to training config
target_trt: "8.5.2"       # onboard TensorRT version
```

- `classes` in `model.yaml` must match `data.yaml` exactly.
- `train_commit` must be the git SHA of the training run so the config is reproducible.
- `target_trt` documents the onboard baseline (TensorRT 8.5.2 / CUDA 11.4.19 /
  cuDNN 8.6.0). Export must stay compatible with TensorRT 8.5.2 operators.

## Target Device Constraint

- Onboard baseline: TensorRT 8.5.2 / CUDA 11.4.19 / cuDNN 8.6.0.
- The `.engine` is built on-device by `trtexec` from the ONNX; this repo never builds
  a `.engine`.
- Export must avoid operators unsupported by TensorRT 8.5.2.

## Handoff to Onboard Repo

After each publish, record clearly in the commit / release notes:
1. HF repo id + version tag.
2. `model.yaml` input/output names, dtype, shape.
3. Class list.
4. Training git commit (for traceability).

The onboard `fetch_model.sh` pulls the artifact and validates the by-name ABI against
`model.yaml`.
