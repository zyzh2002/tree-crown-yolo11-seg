# Staged-Model Visualization Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Implementation Status (2026-08-14)

- Task 1 complete: `jupyter` added to the dev extra and installed with
  `uv add --dev jupyter`; torch stayed at the cu126 build; CPU fallback
  verified (`cuda available: False`).
- Task 2 complete: `notebooks/visualize-staged-model.ipynb` created with all
  logic inline and executed headless successfully.
- OOM fix (2026-08-14): the first headless run OOM'd on an 8 GB machine.
  Root cause: source images are 2048x2048 and the notebook held 20 full-res
  results at once with ultralytics' default `max_det=300` (~5 GB of float32
  masks per image) plus ~1.3 GB of retained parquet frames. Fix: per-image
  inference with immediate render + `del`/`gc.collect()`, `MAX_DET=50`
  guard, releasing source frames right after sampling, and 512 px display
  thumbnails. Verified peak fits well under 8 GB; headless execution passed
  with 0 errors.
- Fidelity fix (2026-08-14): the row filter initially kept only rows with at
  least one valid tree polygon (366 of 439), but the training pipeline also
  keeps the 51 rows with NO annotations as background/negative samples
  (empty label files) - 417 written images in the staged manifest. The
  filter was corrected to mirror `write_image_and_labels` exactly; the
  notebook now reports 417 accepted test rows, matching the manifest.

**Goal:** Add a permanent CPU-only Jupyter notebook that visualizes the staged Stage 1a checkpoint `exp-stage1a-oamtcd-y11n-r1` on held-out OAM-TCD test images.

**Architecture:** A single self-contained notebook holds all logic inline (download + SHA256 gate, test-row loading/filtering/sampling, training-time image reconstruction with canopy redaction, GT polygon extraction, CPU inference, and a 3-column comparison grid). No helper module and no unit tests, per the recorded user override.

**Tech Stack:** Python 3.12, ultralytics 8.4.115 (CPU), polars, huggingface_hub, pycocotools, opencv, matplotlib, jupyter (new dev dep). Reuses existing `prepare_oamtcd.py` and `publish.py` internals.

## Global Constraints

- Design spec: `.agents/docs/specs/2026-08-14-staged-model-visualization-notebook-design.md` — it is authoritative; every fact and behavior below is drawn from it.
- User overrides (recorded): (1) no unit tests and no helper module — the notebook is a debugging aid with all logic inline; (2) reuse the current venv — install jupyter with `uv add --dev jupyter`, not the AGENTS.md from-scratch rebuild.
- Do not modify `prepare_oamtcd.py`, `publish.py`, `data.yaml`, training configs, or the onboard repository. Importing their existing functions is allowed.
- Code comments and identifiers in English; markdown cells in the notebook may be Chinese (human-facing exemption).
- Never commit datasets, checkpoints, PNGs, or parquet files; only the notebook and source code are committed. `runs/` and `*.png` are already git-ignored.
- Do not add any runtime dependency beyond `jupyter` (dev extra). The V100 cu126 torch pin must not change.
- Work happens on the current branch `feat/oamtcd-pretraining`. Local commits are allowed; pushing requires explicit user approval.

---

### Task 1: Add jupyter to the dev extra and install it incrementally

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: a venv where `jupyter nbconvert --execute` and the notebook kernel work, with the torch cu126 pin unchanged.

- [ ] **Step 1: Add `"jupyter"` to the `dev` extra in `pyproject.toml`**

```toml
dev = [
    "ruff>=0.6.0",
    "pytest>=8.0.0",
    "jupyter",
]
```

- [ ] **Step 2: Install incrementally with uv add**

Run: `uv add --dev jupyter`
Expected: resolves and installs jupyter/nbconvert/ipykernel without touching
the existing torch installation. (User-approved route; the AGENTS.md rebuild
is explicitly waived for this task.)

- [ ] **Step 3: Verify jupyter is usable**

Run: `.venv/bin/jupyter --version`
Expected: version output including `jupyter`, `nbconvert`, `ipykernel`.

- [ ] **Step 4: Confirm existing deps still work and the pin is unchanged**

Run: `.venv/bin/python -c "import torch, ultralytics, polars, pycocotools, cv2, matplotlib; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.13.0+cu126` (or whatever was installed) and `False` for cuda
availability — proving CPU fallback on this GPU-less machine.

- [ ] **Step 5: Commit**

Run: `git add pyproject.toml uv.lock && git commit -m "chore: add jupyter dev dependency for notebook tooling"`

---

### Task 2: Create the notebook

**Files:**
- Create: `notebooks/visualize-staged-model.ipynb`

**Interfaces:**
- Consumes: `prepare_oamtcd.REPO_ID`, `REVISION`, `KEEP_CATEGORY_IDS`, `DROP_CATEGORY_IDS`, `YOLO_CLASS_NAME`, `annotations_to_instances`, `segmentation_to_yolo`, `_extract_image_bytes`, `_redact_canopy_regions`, `_encode_rgb_jpeg`; `publish._load_token_from_env`; `ultralytics.YOLO`; matplotlib; polars; huggingface_hub.
- Produces: an executable notebook rendering a 3-column x N-row grid (reconstructed input | GT overlay | prediction overlay).

- [ ] **Step 1: Create the notebook cells**

Follow the cell-by-cell layout in the design spec (Sections "Inline Logic
Blocks" and "Notebook Specification"), in this order:
1. Markdown (Chinese): purpose, prerequisites (credentials env, `uv add
   --dev jupyter`), runtime expectations (CPU, ~1-3 minutes for 20 images).
2. Parameters: `N_IMAGES = 20`, `SEED = 0`, `IMGSZ = 1280`, `CONF = 0.25`,
   `DEVICE = "cpu"`, `SAVE_PNG = False`; token via
   `publish._load_token_from_env(".local/credentials.env")` (never printed).
3. Download checkpoint via `hf_hub_download` + inline SHA256 gate against
   `CHECKPOINT_SHA256` (full constant inline).
4. Load `data/test-00000-of-00001.parquet` via `hf_hub_download` +
   `pl.read_parquet`; inline filter (drop canopy-only rows and rows without
   a valid individual-tree polygon, reusing `prepare_oamtcd` helpers); seeded
   `pl.DataFrame.sample`; print sampled `image_id`s and note the sample draws
   from the 417 training-pipeline-accepted test images.
5. GT overlays: per row, reconstruct the training-time RGB image
   (`_extract_image_bytes` + `_redact_canopy_regions` + `_encode_rgb_jpeg` +
   `cv2.cvtColor`), draw `gt_polygons` (category-2 only, via
   `segmentation_to_yolo` scaled back to pixel coordinates) in one distinct
   color on a copy.
6. Inference: `YOLO(str(checkpoint))`;
   `results = model.predict(source=<list of reconstructed RGB arrays>,
   device=DEVICE, imgsz=IMGSZ, conf=CONF, verbose=False)`; overlays via
   `result.plot()`. Comment that the model is single-class `tree-crown`.
7. Grid: `fig, axes = plt.subplots(N_IMAGES, 3, ...)`, one row per image
   (reconstructed | GT | prediction); `plt.tight_layout()`; if `SAVE_PNG`,
   save each PNG to `runs/visualize/exp-stage1a-oamtcd-y11n-r1/` (git-ignored).
8. Markdown (Chinese): HF cache cleanup note.

English code comments, Chinese markdown cells.

- [ ] **Step 2: Execute the notebook headless**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/visualize-staged-model.ipynb`
Expected: all cells execute without error; the final grid contains 3 x
`N_IMAGES` panels with plausible mask overlays. (CPU inference for 20 images
takes roughly 1-3 minutes.)

- [ ] **Step 3: Restore the executed notebook to a clean state**

Clear outputs so no binary blobs are committed:
Run: `.venv/bin/jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --inplace notebooks/visualize-staged-model.ipynb`
Expected: notebook has no stored outputs in the working tree.

- [ ] **Step 4: Commit**

Run: `git add notebooks/visualize-staged-model.ipynb && git commit -m "feat: add CPU visualization notebook for staged OAM-TCD checkpoint (debugging aid, logic inline)"`

---

### Task 3: End-to-end verification and review

**Files:**
- Verify: `notebooks/visualize-staged-model.ipynb`

**Interfaces:**
- Consumes: the completed Tasks 1-2.
- Produces: evidence that the notebook meets the acceptance criteria.

- [ ] **Step 1: Fresh headless execution**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute notebooks/visualize-staged-model.ipynb`
Expected: success; spot-check that predictions overlap GT polygons
qualitatively. Then clear outputs again with the command from Task 2 Step 3.

- [ ] **Step 2: Repo hygiene check**

Run: `git diff --check && git status --short`
Expected: only `pyproject.toml`, `uv.lock`, the notebook, and the spec/plan
docs are tracked changes; no `*.png`, `*.pt`, `*.parquet`, or `runs/` files
staged or committed.

- [ ] **Step 3: Confirm the SHA256 gate fires on a wrong path**

Run a short inline check in the venv that hashes an unrelated file with the
wrong expected digest and confirms the mismatch raises.

- [ ] **Step 4: Final commit if anything changed**

Run: `git status --short`; commit remaining tweaks with a `fix:` or `docs:`
prefix only if the working tree is non-empty. Do not push; pushing requires
user approval.

---

## Acceptance Criteria (recap)

- Notebook runs end-to-end on CPU without GPU access.
- GT overlays and inference inputs come from the same reconstruction path as
  the training labels (canopy redaction included).
- Checkpoint integrity is verified before inference.
- Instructions in the first markdown cell suffice for a new user with
  credentials set up.
