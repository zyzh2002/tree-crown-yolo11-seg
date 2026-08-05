# Implementation Plan: Repo Scaffold

## Objective

Scaffold the empty `tree-crown-training` repository so that the training → export →
publish workflow is runnable end-to-end, mirroring the onboard repo conventions.

## Steps

1. [x] `git init -b main`; create directory structure.
2. [x] Write `.gitignore` (data, weights, ONNX, venv, `.local/`, runs).
3. [x] Write `AGENTS.md`, `CLAUDE.md`.
4. [x] Write `pyproject.toml` (ultralytics, onnx, huggingface_hub, pyyaml, ruff).
5. [x] Write `data.yaml` (single-class `tree-crown`, confirmed).
6. [x] Write `README.md`, `docs/` (Chinese).
7. [x] Write `.agents/docs/` (spec + plan).
8. [x] Write `configs/default.yaml`.
9. [x] Scaffold `train.py`, `export.py`, `publish.py`.
10. [x] Verify scripts run (`--help`) and `ruff check` passes.
11. [x] Commit with `chore: scaffold training repo`.

## Follow-ups

- First real training run to lock `configs/default.yaml`.
- Implement `prepare_oamtcd.py` for OAM-TCD single-class tree-crown pretraining
  (separate task, per the age-estimation design spec). `prepare_geotree.py` is
  **rejected**: geotree labels are detection boxes without segmentation polygons
  and too many near-empty placeholder tiles, so it cannot contribute crowns.
- Align training config with the scripts' actual behavior (unused `imgsz_batch` / `task`,
  `deterministic` passthrough, fixed export naming).