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

## Skills (Mandatory)

This repository ships agent skills under `.agents/skills/`. Skills are process
controls, not optional reading. The following rules are mandatory:

- Before taking action on a task, inspect the available skills and load every
  skill that matches the task. A matching skill must be loaded even when the
  change appears small, obvious, or documentation-only.
- Load `using-superpowers` first when it is available. It defines the required
  skill-discovery and hard-gate workflow for the session.
- Read the loaded `SKILL.md` completely and follow its hard gates, required
  questions, approval checkpoints, tool restrictions, and verification steps.
  Do not replace a required skill workflow with personal judgment or a shorter
  equivalent process.
- If multiple skills apply, use process skills before implementation skills and
  use verification skills before making completion claims. Do not skip a skill
  because another skill appears to cover part of the same work.
- Announce the relevant skill and its purpose in the progress update before
  beginning the associated work. Never claim to have used a skill that was not
  actually loaded with the skill tool.
- User instructions take precedence over repository skills only when they
  explicitly conflict. Record the conflict and follow the user's instruction;
  otherwise the skill workflow remains mandatory.

### Required Skill Mapping

- New behavior, creative work, or feature design: `brainstorming` before
  implementation; obtain the approval required by that skill before editing.
- A written specification or multi-step requirement: `writing-plans` before
  touching implementation files; keep the plan in `.agents/docs/plans/`.
- Executing an approved plan: `executing-plans` or
  `subagent-driven-development`, as selected by the user or plan workflow.
- Bug reports, regressions, unexpected behavior, or failed tests:
  `systematic-debugging` before proposing or applying a fix.
- Any feature or bugfix implementation: `test-driven-development` before
  writing production code unless the user explicitly waives it.
- Hugging Face authentication, download, upload, staging, or publication:
  `hf-cli` before using the HF CLI or API.
- Architecture changes or architecture documentation:
  `architecture-blueprint-generator` when its documented scope applies.
- Documentation authoring or restructuring: `documentation-writer` when its
  documented scope applies; human-facing documentation remains Chinese and
  agent-facing documentation remains English.
- Code review requests or major changes before integration:
  `requesting-code-review`; receiving review feedback requires
  `receiving-code-review` before applying suggestions.
- Commits or commit-message generation: `make-repo-contribution` first when
  repository contribution guidance applies, then `git-commit` or
  `conventional-commit` as appropriate. Never commit merely because a task is
  complete.
- Before claiming that work is fixed, complete, verified, or passing:
  `verification-before-completion` with fresh command evidence.

### Skill Stop Conditions

- If a loaded skill requires user clarification or approval, stop and ask one
  focused question; do not continue implementation on an assumption.
- If a loaded skill requires a design or plan review, do not edit production
  files until that checkpoint is satisfied.
- If verification fails, report the actual failure and keep the task open;
  do not claim completion based on intent, partial output, or an earlier run.
- If no available skill matches the task, state that explicitly and proceed
  using the repository instructions and the verification requirements above.

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
