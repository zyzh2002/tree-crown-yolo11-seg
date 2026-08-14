# Stage 2 Roadmap Review Record

## Status

Independent review of the Stage 2 roadmap performed on 2026-08-13,
including direct dataset downloads, live municipal-API queries, and source
inspection of the onboard preprocessing code and the pinned Ultralytics
8.4.115 training pipeline.

The Paris `les-arbres` + IGN BD ORTHO route was added to the data-source
investigation spec and acquisition plan during the original review. This
record was fact-checked again on 2026-08-14. The corrections below update
the evidence and wording of this record; they do not change the training
configuration or start Stage 2 training.

## Verified Facts

- NYC 2015 Street Tree Census (live Socrata query, 2026-08-13): London
  planetree = 87,014. Selected negative species counts are honeylocust
  64,264, Callery pear 58,931, pin oak 53,185, Norway maple 34,189, and
  ginkgo 21,024. Japanese zelkova has 29,258 records.
- NEON DTA gpkg: downloaded from HF, SHA-256 matches the pinned
  `eeffb0c06781b452a4a54eac5c80bfc257084cc467cc7339ce1e67d84543c61f`
  byte-for-byte; direct sqlite3 audit confirms PLOC = 103 rows across the
  9 documented sites.
- OAM-TCD pinned revision `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`
  exists with 7 train + 1 test parquet shards, matching the converter.
- DJI Matrice 4T wide camera: 1/1.3-inch CMOS, 48 MP effective pixels,
  82-degree FOV, and 24 mm equivalent focal length (DJI official specs).
  The onboard PSDK ImageStream was measured separately at 1440x1080 NV12
  and approximately 30 fps.
- Onboard preprocessing (`manifold-3-vision-detect`
  `src/inference/preprocess.cpp`): letterbox scale 0.8889 for 1440x1080,
  content 1280x960 centered in a 1280x1280 tensor, 160 px zero padding on
  each vertical side (320 px total), and nearest-neighbor NV12 sampling.
- For a two-class YOLO11n-seg model at static 1280x1280 input, the
  architecture-derived output shapes are `[1, 38, 33600]` detections and
  `[1, 32, 320, 320]` prototypes. The candidate count is
  `160^2 + 80^2 + 40^2 = 33600`; `publish.py` enforces these shapes against
  the inspected ONNX graph. This does not prove consumer-side integration.
- TensorRT 8.5 supports ONNX operators through opset 17, subject to the
  individual operator, data type, attribute, and shape restrictions in its
  support matrix. Target-device parser/build and numerical validation are
  still required for the specific exported graph.
- Fresh verification in the complete repository environment on 2026-08-14:
  `63 passed, 3 warnings`; `ruff check .` passed; `ruff format --check .`
  passed. The earlier 49/63 minimal-venv result has no retained log and is
  not independently auditable.

## Findings Pending Resolution

### P1-1: Esri imagery has material terms-of-use risk

Esri terms and World Imagery guidance create a material compliance risk for
systematic tile harvesting, redistribution, and automated extraction. The
exact permission depends on the service, account, export workflow, and use
case; this record should not make a universal legal determination.

The safer request-free alternative is NYC's official 6-inch orthoimagery
through `maps.nyc.gov` TMS/XYZ/WMTS, explicitly licensed CC BY 4.0. NYC's
2001 and 2002 6-inch captures are complementary partial coverage; citywide
6-inch coverage begins in 2004. The 2014 and 2016 vintages are available and
can reduce and measure, but cannot eliminate, the census time-offset risk:
the census collection spans 2015-2016, has varying per-tree dates, and does
not include a 2015 citywide capture. A leaf-state and tree-change pilot is
still required.

### P1-2: Inventory-point volume data does not satisfy the exhaustive gate

The 2026-08-07 strategy spec requires every visible and distinguishable crown
in a supervised frame to be labeled. The NYC and Paris workflows label
inventory points only; non-inventory crowns in parks, yards, or new plantings
can remain unlabeled and violate the gate.

An explicit waiver may classify auto-delineated inventory crops as noisy
supervision, but that does not make them exhaustive two-class supervision.
Calling the same windows "mask-head pretraining" also does not make unlabeled
crowns ignored under the current YOLO training path. Valid alternatives are
manual exhaustive annotation, an explicitly implemented ignore/noisy-label
loss, or retaining the data outside the formal supervised two-class split.

### P1-3: Published shape contract is not consumer integration evidence

The producer-side contract is a real two-output YOLO11-seg graph, while the
separate consumer implementation inspected during this review still uses a
synthetic three-output contract. The consumer-side transition is therefore a
cross-repository prerequisite for deployment and must not be inferred from a
successful `publish.py` shape check. This is recorded as an integration
blocker, not changed by the training-side review.

### P2-2: BAMFORESTS package-level license provenance

The BAMFORESTS paper explicitly states that the dataset is available under
CC BY 4.0. The DLR page's `DLR (CC BY-NC-ND 3.0)` text appears as a credit
under page figures and tables; it is not an explicit package-level license
statement for the downloadable ZIP files.

The remaining concern is provenance: the unofficial HF conversion does not
provide complete license metadata or a hash relationship to the official DLR
package. Prefer the official package, preserve the paper citation and
attribution, and obtain clarification if the HF conversion is to be treated
as the authoritative source. Do not state that NC-ND has been established
for the dataset or that it automatically blocks every trained-model use.

### P2-3: Train/deploy preprocessing mismatch

The pinned Ultralytics 8.4.115 normal letterbox path uses aspect-ratio-
preserving `INTER_LINEAR` resizing and padding value 114. Mosaic composes
aspect-ratio-preserved images on a larger canvas and then applies geometric
transformation/cropping; it is not direct non-uniform stretching of one
image to 1280.

The onboard path uses nearest-neighbor NV12 sampling and zero padding. A
useful validation must compare interpolation, padding value, NV12 color
range and conversion matrix, integer sampling coordinates, channel order,
normalization, and the final tensor for the same raw frame. A single
nearest-neighbor mAP run is not sufficient to isolate all differences.

### P3-1: Stale checkpoint path in fine-tune config

`configs/finetune-platanus-other.yaml` points `model:` at
`runs/segment/oamtcd-pretrain-fixed/weights/best.pt`; the recorded Stage 1a
artifact is `runs/segment/oamtcd-stage1a-rgb-3/weights/best.pt`. This remains
an actionable configuration issue for the training-machine agent.

### P3-2: Conditional GSD calculation

Assuming the 82-degree camera specification is the effective diagonal FOV
of the full 4:3 PSDK frame, the derived FOV is approximately 69.632 degrees
horizontal and 55.091 degrees vertical. This gives approximately 0.9659
mm/source-pixel per metre of camera-to-plane distance, or 9.659 cm/source-
pixel at 100 m. After resizing to the 1280x960 model content area, the
corresponding model-content scale is approximately 10.87 cm/model-pixel at
100 m.

The liveview FOV assumption remains uncalibrated. Camera-to-plane distance,
crown height, terrain, distortion, cropping, and off-nadir viewing must be
checked during the first-flight calibration.

### P3-3: Ultralytics version range affects reproducibility

`pyproject.toml` allows `ultralytics>=8.3.0`, while `uv.lock` currently
resolves 8.4.115. A cross-version run is not automatically invalid and does
not prove that the segmentation head changes. It can change checkpoint
loading coverage, augmentation behavior, defaults, losses, exporter behavior,
and reproducibility. The training-machine agent should use the committed
lockfile or record and verify the exact package version.

### P3-4: Square-input padding overhead

The fixed 1280x1280 tensor contains 1280x320 padding, exactly 25% of the
input tensor area. A static 1280x960 input would preserve the same 0.8889
source-image resize scale and remove that padding.

This area ratio is not a measured 25% compute or latency saving. Actual
TensorRT arithmetic, tactics, memory transfers, attention operations, and
end-to-end timing require a real rectangular graph and target-device
benchmark. The rectangular graph would also change the output shapes to
approximately `[1, 38, 25200]` and `[1, 32, 240, 320]`, so it is a future
cross-repository ABI change rather than a current-release optimization.

## Review Outcome

- The factual corrections in this record are documentation corrections only.
- The exhaustive annotation issue, license provenance, preprocessing parity,
  checkpoint path, version reproducibility, and first-flight calibration
  remain implementation or decision work outside this record.
- No Stage 2 training was started by this review.
- The current repository test and lint results are reproducible as recorded
  above; the historical minimal-venv result is retained only as unaudited
  context.
