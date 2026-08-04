# Tree Crown Age Estimation Design

## Goal

Extend the tree-crown training pipeline from a single-stage crown segmentation model to a
two-stage task: (1) segment individual tree crowns from top-down aerial imagery, then (2)
estimate each crown's age. The age model lives on the onboard device and runs after
segmentation to produce a tree-age at inference time.

This is a design document only. No code is written in this pass.

## Scope & Non-Goals

- **In scope:** overall two-stage architecture, data sourcing for segmentation, age
  estimation approach (allometric model), deployment split between training and onboard.
- **Out of scope:** implementation of the segmentation model, the allometric model, or any
  code. Species-level classification (distinguishing tree species) is **not** a goal; the
  crown segmentation is generic (single class `tree-crown`).

## Confirmed Decisions (user-aligned)

- **Task:** crown segmentation → age estimation (two stages).
- **Segmentation:** generic single-class crown segmentation (existing `data.yaml` placeholder
  `tree-crown`). Public datasets (geotree + OAM-TCD) are sufficient.
- **Age estimation method:** allometric / rule-based model from crown geometry (NOT a trained
  deep regression model).
- **Age granularity:** 5-year bins for young/mid-age (<~40 years); mature trees are reported
  as a "mature" bucket (crown-diameter saturation limit).
- **Age model location:** onboard device (C++), runs after segmentation, computes crown
  diameter from the mask and looks up age.
- **Allometric parameters:** start from published Platanus growth parameters (e.g.
  Chapman-Richards), calibrate later with Xi'an in-situ measurements.
- **Research only, non-commercial:** CC-BY-NC datasets are usable.
- **This deliverable:** design document only.

## System Architecture

```
Top-down aerial imagery
        │
        ▼
┌───────────────────────────────┐
│  STAGE A: Crown segmentation  │  YOLO11-seg (generic "tree-crown")
│  (PC training, HF publish)    │  weights published to HF private repo
└───────────────────────────────┘
        │  per-instance mask
        ▼
┌───────────────────────────────┐
│  STAGE B: Age estimation      │  ONBOARD (C++), after segmentation
│  (allometric / rule-based)    │  crown size → age lookup
└───────────────────────────────┘
        │
        ▼
   per-tree age estimate
```

### Stage handoff

- **Stage A** follows the existing pipeline: train → export ONNX → publish to HF private
  repo (`zyzh0/tree-crown-yolo11-seg`). The onboard `fetch_model.sh` pulls it and builds a
  `.engine` with `trtexec`.
- **Stage B** is a small onboard component (C++). It consumes the segmentation mask
  (crown area/diameter) and applies an allometric age model. It is **not** part of the
  publishable ONNX artifact; it is a lightweight lookup/rule applied on-device.

## Stage A: Crown Segmentation

### Data

Two public datasets, both generic crown segmentation (no species labels):

| Dataset | Source | Format | Notes |
|---|---|---|---|
| `the-shoaib2/geotree` | HF | YOLO-ready | Single class `tree`, 132 imgs, 697 MB. Zero conversion. |
| `restor/tcd` (OAM-TCD) | HF | GeoTIFF + COCO | 5072 imgs, 2048², ~280k trees. CC-BY subset. Needs COCO→YOLO conversion. |

- `data.yaml` keeps single class `tree-crown` (the existing placeholder).
- geotree is used first as a quick pipeline smoke test (zero conversion); OAM-TCD is the
  higher-quality, more diverse upstream for the real model.
- Both are CC-BY-compatible (OAM-TCD used only from the pure-CC-BY `restor/tcd` repo).
- **Known limitation:** both datasets under-label closed-canopy (touching crowns); they are
  not ideal for dense crowns but acceptable for a first generic segmentation model.

### Recommended flow

1. `prepare_geotree.py` — download geotree, layout to `data/` (near-zero conversion).
2. `prepare_oamtcd.py` — download `restor/tcd`, convert COCO→YOLO instance segmentation.
3. Merge `tree` + `tree-canopy` OAM-TCD classes into a single `tree-crown`.
4. Train `yolov11n-seg` on 1280² input (existing contract), validate, export, publish.

## Stage B: Age Estimation

### Core idea

Age is not directly readable from a top-down image. Published literature uses an indirect
chain:

```
crown diameter (CD)  ──►  (allometric)  ──►  age
```

- **CD ↔ DBH:** strong correlation (R² > 0.80).
- **DBH ↔ age:** weaker (R² ≈ 0.7), species-specific, affected by pruning and site.
- **CD growth saturates around ~40 years** (Platanus × acerifolia reaches ~13 m crown
  diameter, then stabilizes). Beyond saturation, age cannot be distinguished from crown size
  alone.

### Age model (allometric, rule-based)

- Feature: crown diameter, computed from the segmentation mask (equivalent diameter of the
  mask area, or max-axis).
- Model: Chapman-Richards (or Peper-style) growth curve fitted to published Platanus
  parameters.
- Output: 5-year bins for ages <~40 years; a single "mature" bucket for ≥ ~40 years.
- Expected accuracy: on the order of ±15% (from literature), driven mostly by crown size.

### Deployment (onboard)

- Runs on-device in C++ after segmentation.
- Input: crown mask (area / diameter) from Stage A.
- Lookup: map crown diameter → age via the fitted curve.
- No neural network re-training; a small table/curve + interpolation.

### Calibration plan

- Start with published Platanus allometric parameters (literature).
- Later calibrate with Xi'an in-situ measurements: trees with known planting age (from
  municipal archives) paired with measured crown diameter. Adjust curve parameters to the
  local climate/site.

## Data & Labeling Requirements

- **Segmentation:** covered by public datasets; no manual labeling needed.
- **Age calibration:** needs (crown size, true age) pairs for Xi'an Platanus. Sources:
  - Municipal planting archives (trees with known planting year).
  - Tree-ring analysis (dendrochronology) for validation.
  - Open-grown trees only (avoid heavy pruning/competition, which distort the CD–age curve).

## Risks & Assumptions

- **Top-down RGB only, no height:** tree height (needs LiDAR/CHM) is unavailable onboard;
  age relies on crown diameter alone, capping achievable accuracy.
- **Crown saturation:** mature trees (>~40 y) cannot be aged precisely; only "mature" is
  reported. This is an accepted limitation.
- **Species specificity:** the allometric curve is species-specific. For a generic
  segmentation, the age model must assume a species (Platanus) or be extended per species.
- **Pruning/site effects:** urban pruning and site conditions distort the CD–age relation;
  keep calibration to open-grown trees.

## Open Items

- Decide whether the age model assumes Platanus only, or per-species curves.
- Confirm Xi'an local calibration data availability and access.
- Whether to include a light height cue (e.g. from a CHM) if available on some onboard paths.
- C++ implementation scope for the age component (onboard repo, not this repo).

## Acceptance Criteria (design)

- Document specifies the two-stage architecture, data sources, and age estimation approach.
- Age model is allometric/rule-based, onboard, with 5-year bins + mature bucket.
- Segmenting data plan (geotree + OAM-TCD) and age calibration plan are explicit.