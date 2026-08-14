# Staged-Model Visualization Notebook Design

- Created: 2026-08-14
- Status: approved design (as designed in the opencode session
  "Stage model test set and CPU inference visualization"), revised after
  fact-checking, revised again on 2026-08-14 per user decisions; approved
  for implementation
- Scope: qualitative visualization of a staged checkpoint on OAM-TCD test
  images using CPU inference. No metrics, no deployment, no ABI changes.
- User overrides (explicit, recorded per AGENTS.md conflict rule):
  - This notebook is a debugging aid, NOT a test deliverable. No separate
    unit-test module and no pytest coverage are required (TDD waived). All
    logic lives inline in the notebook.
  - Reuse the current venv: install jupyter incrementally with
    `uv add --dev jupyter` instead of the AGENTS.md from-scratch rebuild.

## Goal

Provide a permanent Jupyter notebook that visualizes the staged Stage 1a
checkpoint `exp-stage1a-oamtcd-y11n-r1` on held-out OAM-TCD test images, with
CPU-only inference, to inspect segmentation quality qualitatively.

## Context and Facts

All facts below were re-verified against the repository and the HF staging
repo on 2026-08-14.

- Staged model: HF repo `zyzh0/tree-crown-yolo11-seg-staging`, tag
  `exp-stage1a-oamtcd-y11n-r1`, HF commit
  `1204de0583064cbfd28ac1dcbbc0a6aae97ac1b1`.
  - Checkpoint path inside the repo:
    `artifacts/exp-stage1a-oamtcd-y11n-r1/model.pt` (verified present,
    6,161,508 bytes).
  - `model.pt` SHA256: `83e2a1c11a8ccfd41207bbd127f4ea69eac270d1dcb54039ec4e71211aef73b9`
    (authoritative: `.agents/docs/experiments/oamtcd-stage1a-r1.md`).
  - Single-class `tree-crown` (class id 0), `single_cls: true`, trained at
    imgsz 1280, `yolo11n-seg`, `deployable: false`.
- Test set: the OAM-TCD source test split is a true held-out test set. The
  prepared dataset (`prepare_oamtcd.py`) maps source `test-*.parquet` rows to
  `test`; val comes from fold 4, train from folds 0-3. The staged
  `dataset-manifest.json` records per-split counts. The test split has never
  been evaluated by any Stage 1a run.
- Data source for the notebook: `restor/tcd` HF dataset (license cc-by-4.0),
  pinned revision `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`, config `default`.
  - The source test shard is `data/test-00000-of-00001.parquet` with 439 rows.
  - After the training-pipeline conversion, only **417** test images were
    written (verified from the staged `dataset-manifest.json`): 22 source rows
    are dropped (canopy-only images, invalid/duplicate polygons, decode
    errors). The notebook must reproduce the same drop logic.
- Training-pipeline image transform (mandatory for GT/prediction consistency):
  `write_image_and_labels` redacts group-canopy (category 1) regions to black
  on every image that mixes canopy and individual-tree annotations
  (`_redact_canopy_regions`), and drops canopy-only rows. Manifest counts:
  `redacted_canopy_images = 3755`, `dropped_canopy_only_images = 375`
  (dataset-wide). Showing raw unredacted images would display canopy regions
  the model never saw during training; the notebook must reconstruct the
  training-time image before inference and overlay.
- Current machine has no GPU; CPU inference is mandatory and acceptable.
  The installed `torch` is the cu126 wheel; on a GPU-less machine CUDA
  libraries are lazily loaded and never required, `torch.cuda.is_available()`
  reports `False`, and all computation automatically falls back to CPU.
  Reusing the venv for jupyter does not change this. The notebook also
  passes `device="cpu"` explicitly to ultralytics as a belt-and-braces guard.
- Environment gap: `jupyter`/`ipykernel`/`nbconvert` are NOT installed in the
  venv (verified). `matplotlib`, `ultralytics` 8.4.115, `datasets`, `pycocotools`,
  `polars`, `torch` (cu126 build, works on CPU), `huggingface_hub`, `cv2`,
  `numpy` are installed (verified).
- Credentials: `.local/credentials.env` exists with `HF_TOKEN` set (verified).
  `publish._load_token_from_env` reads it; never hardcode tokens.

## Non-Goals

- No quantitative metrics (mAP/precision/recall) in this notebook.
- No full-dataset materialization; no re-run of `prepare_oamtcd.py`.
- No conf/IoU sweeps, no export, no publication.
- No changes to `data.yaml`, `model.yaml`, or the onboard ABI.
- No separate helper module and no unit tests (user override: debugging aid).

## Deliverable

1. `notebooks/visualize-staged-model.ipynb` — self-contained notebook (new
   directory `notebooks/`, committed to git). All logic inline.
2. `pyproject.toml` — add `jupyter` (>= nbconvert + ipykernel) to the `dev`
   extra so the notebook can be executed headless; installed incrementally
   with `uv add --dev jupyter` (user-approved reuse of the current venv).

## Inline Logic Blocks (notebook cells)

All logic lives in notebook cells as small functions or short snippets
(English code comments). Key pieces, in order of appearance:

- Staging constants: `STAGING_REPO_ID = "zyzh0/tree-crown-yolo11-seg-staging"`,
  `STAGING_TAG = "exp-stage1a-oamtcd-y11n-r1"`,
  `CHECKPOINT_PATH = "artifacts/exp-stage1a-oamtcd-y11n-r1/model.pt"`,
  `CHECKPOINT_SHA256 = "83e2a1c11a8ccfd41207bbd127f4ea69eac270d1dcb54039ec4e71211aef73b9"`
  (comment pointing at `.agents/docs/experiments/oamtcd-stage1a-r1.md`).
- Constants imported from repo modules to avoid duplication:
  `REPO_ID`, `REVISION`, `KEEP_CATEGORY_IDS`, `YOLO_CLASS_NAME` from
  `prepare_oamtcd`.
- Checkpoint download: `hf_hub_download(STAGING_REPO_ID,
  filename=CHECKPOINT_PATH, revision=STAGING_TAG, token=token)` followed by a
  SHA256 check against `CHECKPOINT_SHA256`; raise on mismatch with an
  actionable message.
- Test rows: download `data/test-00000-of-00001.parquet` from `REPO_ID` at
  `REVISION` with `hf_hub_download`, read with `pl.read_parquet`.
  Chosen over `datasets` streaming because the parquet `image` column holds
  the original bytes, which feed the training pipeline's redact path
  byte-for-byte. (`datasets` 5.x decodes to PIL and diverges from the
  training-time transform.)
- Filter: exact mirror of the training drop rule — drop a row only when it
  has canopy annotations but no individual-tree annotations (canopy-only),
  or when tree annotations exist but none convert to a valid
  `segmentation_to_yolo` polygon. Rows with NO annotations at all are KEPT
  as background/negative samples (the pipeline writes empty label files for
  them), so the accepted count is 417, matching the staged manifest.
- Sampling: deterministic sample of `N_IMAGES` rows from the filtered frame
  (`pl.DataFrame.sample(n=..., seed=SEED)`); raise a clear error if fewer
  rows remain after filtering. Release the source frames right after
  sampling (439 x 2048x2048 JPEG bytes ~= 1.3 GB).
- Training-image reconstruction: reuse `prepare_oamtcd._extract_image_bytes`,
  `prepare_oamtcd.annotations_to_instances`,
  `prepare_oamtcd._redact_canopy_regions`, `prepare_oamtcd._encode_rgb_jpeg`,
  and the same category split as `write_image_and_labels`; return the RGB
  ndarray the model was actually supervised on (canopy regions blacked out).
- GT polygons: individual-tree annotations via `annotations_to_instances`
  filtered by `KEEP_CATEGORY_IDS`, converted with `segmentation_to_yolo` at
  full image resolution, scaled back to pixel coordinates for matplotlib
  drawing. This is the exact training-time conversion path. Background rows
  yield an empty polygon list (correct: the GT column is intentionally blank).
- Memory guards (OOM fix, 2026-08-14): per-image inference with immediate
  render + `del`/`gc.collect()`, `MAX_DET=50` cap (ultralytics default 300
  x full-res 2048x2048 float32 masks ~= 5 GB per image), and 512 px display
  thumbnails. Verified peak well below 8 GB on the dev machine.

## Notebook Specification (cell-by-cell)

1. **Markdown (Chinese)**: purpose, prerequisites (credentials env, dev venv
   with jupyter), runtime expectations (CPU, ~1-3 minutes for 20 images),
   memory notes (2048x2048 source images, MAX_DET cap, per-image loop, and a
   warning that this notebook must not run on the Orin NX - onboard inference
   uses the TensorRT engine instead).
2. **Setup and parameters**
   - Parameters at top, editable without touching logic:
     `N_IMAGES = 20`, `SEED = 0`, `IMGSZ = 1280`, `CONF = 0.25`,
     `DEVICE = "cpu"`, `SAVE_PNG = False`, `MAX_DET = 50`,
     `DISPLAY_SIZE = 512`.
   - Load the HF token via
     `publish._load_token_from_env(".local/credentials.env")`; never print it.
   - Import constants from `prepare_oamtcd` and reuse its helpers; all logic
     is inline (no helper module, per the recorded user override).
3. **Download and verify checkpoint** — `hf_hub_download` with a local
   streaming SHA256 gate inside the cell.
4. **Sample test images** — download the parquet, filter with the exact
   pipeline mirror, sample deterministically; print the sampled image ids
   and note the sample is drawn from the 417 training-pipeline-accepted
   test images; free the source frames.
5. **Ground-truth overlay** — for each row, rebuild the training-time image
   (redaction included), then draw the GT polygons on it with matplotlib
   (one distinct color).
6. **CPU inference** — `YOLO(<downloaded model.pt>)`; loop over images ONE
   at a time with `model.predict(source=<single image>, device="cpu",
   imgsz=IMGSZ, conf=CONF, max_det=MAX_DET, verbose=False)`, render
   `result.plot()` immediately, drop the result, and `gc.collect()` before
   the next image. Comment: single-class `tree-crown` even if `names`
   prints generically.
7. **Visualization grid** — one row per image, three columns:
   reconstructed input | GT overlay | prediction overlay, all displayed as
   `DISPLAY_SIZE` thumbnails. Inline matplotlib display. If `SAVE_PNG`,
   write to `runs/visualize/exp-stage1a-oamtcd-y11n-r1/` (git-ignored).
8. **Markdown (Chinese)**: cleanup note for the HF cache copies of the
   checkpoint and parquet if disk space is needed.

## Error Handling

- Missing `.local/credentials.env` or missing `HF_TOKEN` -> clear error
  pointing to `docs/getting-started.md` / `credentials.env.example`.
- Download failure or SHA256 mismatch -> raise; do not proceed with
  inference.
- Fewer than `N_IMAGES` trainable test rows -> explicit error.

## Conventions

- Code comments and identifiers in English; markdown cells may be Chinese
  (human-facing exemption in AGENTS.md).
- No new Python dependencies for the notebook logic itself; `jupyter` is the
  only new dev dependency (environment, not logic).
- Environment note (markdown cell): run `uv add --dev jupyter` once to make
  the kernel available in the current venv (user-approved reuse of the
  current venv; the torch cu126 pin is untouched).

## Verification (for the implementing agent)

1. Execute the notebook headless:
   `jupyter nbconvert --to notebook --execute
   notebooks/visualize-staged-model.ipynb` on CPU; all cells complete and the
   final grid contains 3 x N panels with plausible masks.
2. Confirm the SHA256 gate fires correctly (test with a truncated or wrong
   file path that raises).
3. Confirm no git-ignored artifacts (PNGs, checkpoints, parquet) are
   committed; the notebook itself is the only new committed file under
   `notebooks/`.
4. Confirm the current venv still imports `torch` after the jupyter install
   and that `uv add` did not change the cu126 pin.

## Acceptance Criteria

- Notebook runs end-to-end on CPU without GPU access.
- GT overlays and inference inputs come from the same reconstruction path as
  the training labels (canopy redaction included).
- Checkpoint integrity is verified before inference.
- Instructions in the first markdown cell suffice for a new user with
  credentials set up.
