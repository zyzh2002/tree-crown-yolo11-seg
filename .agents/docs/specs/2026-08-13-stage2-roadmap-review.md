# Stage 2 Roadmap Review Record

## Status

Independent review of the Stage 2 roadmap performed on 2026-08-13,
including direct dataset downloads, live municipal-API queries, and source
inspection of the onboard preprocessing code and the pinned ultralytics
8.4.115 training pipeline.

One finding was fixed in this pass: the Paris `les-arbres` + IGN BD ORTHO
route was added to the data-source investigation spec and the acquisition
plan. All other findings below are recorded for individual user
verification; no further changes were made.

## Verified Facts

- NYC 2015 Street Tree Census (live Socrata query, 2026-08-13): London
  planetree = 87,014; top negative species honeylocust 64,264, Callery
  pear 58,931, pin oak 53,185, Norway maple 34,189, ginkgo 29,258.
- NEON DTA gpkg: downloaded from HF, SHA-256 matches the pinned
  `eeffb0c06781b452a4a54eac5c80bfc257084cc467cc7339ce1e67d84543c61f`
  byte-for-byte; direct sqlite3 audit confirms PLOC = 103 rows across the
  9 documented sites.
- OAM-TCD pinned revision `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`
  exists with 7 train + 1 test parquet shards, matching the converter.
- DJI Matrice 4T wide camera: 1/1.3-inch 48 MP, 82 degrees diagonal FOV,
  24 mm equivalent focal length (DJI official specs); liveview 1080p/30.
- Onboard preprocessing (`manifold-3-vision-detect`
  `src/inference/preprocess.cpp`): letterbox scale 0.8889 for 1440x1080,
  content 1280x960 centered, 80 px zero padding per side (160 px total),
  nearest-neighbor NV12 sampling. The spec wording "160 px top and
  bottom" overstates the per-side padding.
- Two-class output contract [1, 38, 33600] detections and
  [1, 32, 320, 320] prototypes is consistent with yolo11n-seg at 1280
  (grid 160^2 + 80^2 + 40^2 = 33600); `publish.py` checks match.
- TensorRT 8.5.2 supports ONNX opset <= 17; known Jetson build issues
  with YOLO11 are INT8/Swish-fusion related. The FP32 opset-17 export is
  unaffected; engine build time at 1280x1280 should be probed early.
- Repo test suite: 49/63 passed in an ad-hoc minimal venv; the 14
  failures were all `ultralytics` import errors in that venv, not code
  defects. The repo's own record (63 passed) stands.

## Findings Pending User Verification

### P1-1: Esri tile source conflicts with Esri ToS

Evidence: Esri terms prohibit systematic harvesting of basemap tiles
except via Esri Content Packages with an ArcGIS organizational account,
prohibit redistribution, and restrict data derived from World Imagery to
non-commercial use within ArcGIS.

Suggested fix to verify: switch the NYC tile source to NYC's own
6-inch (15 cm) orthoimagery tile services (`maps.nyc.gov` TMS/XYZ, CC BY
4.0). The per-year resolution table shows 6-inch coverage since 2001;
2014/2016 vintages match the 2015 census year, which also removes the
documented time-offset risk.

### P1-2: NYC volume route vs exhaustive-annotation gate

The 2026-08-07 strategy spec requires every visible, distinguishable
crown in a supervised frame to be labeled. The NYC plan labels inventory
points only; non-inventory crowns (parks, yards, new plantings) in a
crop would remain unlabeled and violate the gate.

Options to verify: (a) accept auto-delineated + inventory-labeled crops
as noisy supervision under an explicit written waiver, or (b) downgrade
NYC/Paris volume data to mask-head pretraining and keep two-class
supervision for the local Xi'an set only.

### P2-2: BAMFORESTS license ambiguity

The paper states CC BY 4.0; the DLR download page footer shows
"DLR (CC BY-NC-ND 3.0)". Obtain written confirmation from the DLR
authors before any Stage 1b run, because NC-ND would block deriving a
published model.

### P2-3: Train/deploy preprocessing mismatch

Training (ultralytics 8.4.115) uses bilinear resize with 114 gray
letterbox padding (mosaic epochs stretch to 1280); the device uses
nearest-neighbor NV12 sampling with zero padding. Suggested check: run
validation with nearest-neighbor resizing once to quantify the mAP
delta.

### P3-1: Stale checkpoint path in fine-tune config

`configs/finetune-platanus-other.yaml` points `model:` at
`runs/segment/oamtcd-pretrain-fixed/weights/best.pt`; the actual Stage 1a
artifact is `runs/segment/oamtcd-stage1a-rgb-3/weights/best.pt`.

### P3-2: GSD math

Correct value: 0.965 mm/px per meter of altitude, i.e. 9.65 cm/px at
100 m (docs state ~9.4 cm/px, ~3% low). The liveview FOV coverage
assumption remains unverified until the first-flight calibration; the
existing risk note is appropriate.

### P3-3: ultralytics version range

`pyproject.toml` allows `ultralytics>=8.3.0` (uv.lock pins 8.4.115).
Fine-tuning must run on the same version as pretraining or the head
structure may change; consider asserting the version in the fine-tune
config or narrowing the constraint.

### P3-4: Square-input compute waste

The fixed 1280x1280 contract letterboxes 1440x1080 to 1280x960, wasting
~25% of compute on padding. A 1280x960 static rectangular input would
have identical content scale; it is a cross-repo ABI change and belongs
to a future major version bump, not the current release.
