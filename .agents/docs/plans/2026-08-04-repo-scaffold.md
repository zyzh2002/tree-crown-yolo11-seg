# Implementation Plan: Repo Scaffold

## Objective

Scaffold the empty `tree-crown-training` repository so that the training → export →
publish workflow is runnable end-to-end, mirroring the onboard repo conventions.

## Steps

1. [x] `git init -b main`; create directory structure.
2. [x] Write `.gitignore` (data, weights, ONNX, venv, `.local/`, runs).
3. [x] Write `AGENTS.md`, `CLAUDE.md`.
4. [x] Write `pyproject.toml` (ultralytics, onnx, huggingface_hub, pyyaml, ruff).
5. [x] Write `data.yaml` (placeholder class list).
6. [x] Write `README.md`, `docs/` (Chinese).
7. [x] Write `.agents/docs/` (spec + plan).
8. [ ] Write `configs/default.yaml`.
9. [ ] Scaffold `train.py`, `export.py`, `publish.py`.
10. [ ] Verify scripts run (`--help`) and `ruff check` passes.
11. [ ] Commit with `chore: scaffold training repo`.

## Follow-ups

- Confirm real class list and update `data.yaml`.
- Confirm HF repo id + owner namespace; wire into `publish.py`.
- First real training run to lock `configs/default.yaml`.