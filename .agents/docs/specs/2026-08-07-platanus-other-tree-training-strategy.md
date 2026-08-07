# Platanus and Other-Tree Training Strategy

## Status

Proposed training-data and model-ABI strategy. This document supersedes the
four-species V1 class decision in
`2026-08-05-tree-crown-age-estimation-design.md` for the first published
model. The earlier document remains authoritative for deployment boundaries,
ONNX constraints, artifact versioning, and the definition of instance masks.

## Problem Statement

The initial generic tree-crown pretraining stage on OAM-TCD produced a usable
pre-fix baseline checkpoint. A later code audit found canopy-only images that
were incorrectly converted to background and disconnected RLE components that
could produce distorted polygons. The corrected Stage 1a must therefore be
rerun before its initializer is selected for downstream training. The
repository has no locally captured and species-annotated Xi'an training data.
The publicly accessible top-down tree datasets assessed for the next stage do
not supply instance-segmentation labels for the four originally proposed Xi'an
species: platanus, Styphnolobium japonicum, Ginkgo biloba, and Koelreuteria
paniculata.

The operational age-estimation workflow requires platanus predictions. A
four-species deployment model would therefore require a new local data
collection and expert annotation campaign for all four species before a valid
fine-tuning run could start. That cost is disproportionate to the immediate
product requirement.

## Decision

The first deployable model uses the following ordered two-class ABI:

```yaml
names:
  0: platanus
  1: other-tree
```

`platanus` means a reliably identified Platanus crown, at genus level.
`other-tree` means a reliably identified, individually distinguishable crown
that is not platanus. It is a negative class for the platanus workflow, not a
catch-all for unknown vegetation or background.

The class order is fixed for the first published tag. A later restoration of
the four-species taxonomy is a new ABI and must use a new model version tag.

## Scope

This strategy covers:

- Generic instance-segmentation pretraining before local species fine-tuning.
- Local Xi'an data requirements for platanus and other-tree labels.
- A two-class YOLO11-seg fine-tuning and evaluation flow.
- The external model ABI and release acceptance criteria.

This strategy does not cover:

- Device-side inference, NMS, mask reconstruction, or age-estimation policy.
- Tree-height, GSD, GPS, camera-intrinsic, or crown-diameter calculation.
- Four-species classification data acquisition or model training.
- Automatic species identification from unverified imagery or annotations.

## Dataset Assessment and Roles

### OAM-TCD: Stage 1a Baseline and Required Rerun

OAM-TCD supplies generic individual-tree masks and is already converted in
this repository as a single `tree-crown` class. Its `tree-canopy` group labels
remain excluded. Its generic labels cannot be mapped to `other-tree` because a
generic tree may be platanus.

The pre-fix OAM-TCD checkpoint may be retained only as an experimental baseline.
The corrected converter must omit canopy-only rows, redact excluded canopy
regions in mixed rows while preserving individual-tree pixels, reject
multi-component instances that cannot be represented faithfully by one YOLO
polygon, deduplicate labels, validate
split disjointness, and use `max_det: 1000` during dense validation. The corrected Stage 1a starts again from the official
`yolo11n-seg.pt`; it must not continue from the pre-fix checkpoint.

Stage 1a checkpoints are initialization artifacts, not release artifacts. They
may be uploaded to `zyzh0/tree-crown-yolo11-seg-staging` with
`deployable: false`, but they must never be published as two-class models.

### BAMFORESTS: Optional Stage 1b

BAMFORESTS is the preferred optional extension to generic pretraining because
it provides publicly available, CC BY 4.0, COCO instance-crown polygons from
very-high-resolution UAV imagery. It has 27,160 delineated crowns across
forest and city-park scenes, including the Hain city-park area.

The publicly released data provides crown shapes but not species labels.
Consequently it is usable only as a single-class `tree-crown` dataset. It must
not provide platanus or other-tree labels.

The Stage 1b model must initialize from the selected OAM-TCD `best.pt`, not
`last.pt`. Stage 1b remains conditional: use it for Stage 2 only if an
ablation against the OAM-TCD-only initializer improves the frozen local
validation metrics without increasing the other-tree-to-platanus error rate.

### Rejected and Reference-Only Sources

- FIRC, Kaggle Aerial Tree Detection, and similar box-only sources cannot
  supervise an instance-segmentation mask head.
- TreeSatAI and HRVQA patch labels cannot provide per-crown instances.
- PureForest and For-species20k are point-cloud or multimodal datasets rather
  than compatible top-down RGB instance-mask sources.
- Forest Inspection provides simulated semantic masks, not species-specific
  crown instances.
- WHU-STree is a street-level panoramic-image and point-cloud dataset, not a
  top-down RGB training source. It may be inspected as a species-reference
  source but must not be mixed into the YOLO11-seg training data.
- Sources with unavailable access or unverified license terms remain excluded
  until their terms, viewpoint, annotation completeness, and labels are
  independently verified.

## Local Xi'an Dataset Requirements

### Capture Requirements

Local imagery is mandatory for any deployable Stage 2 model. Capture top-down
or orthomosaic RGB imagery that matches the deployment camera geometry as
closely as practical. Store the following metadata for each capture group:

```text
capture group or road-corridor identifier
capture date and season
location or spatial grouping reference
image dimensions and GSD when available
source and image-use authorization
```

The final model does not consume this metadata. It is required to construct
spatially independent splits and diagnose domain shift.

### Exhaustive Two-Class Annotation Gate

A frame is eligible for the two-class training, validation, or test set only
when every visible and distinguishable tree crown in that frame is annotated
with one of the two classes:

- `platanus` for a crown verified as Platanus.
- `other-tree` for a crown verified not to be Platanus.

Each accepted crown requires one non-self-intersecting polygon or mask. A
crown that is heavily occluded, truncated, or inseparable may be excluded only
under a written, consistently applied exclusion rule.

If a non-platanus crown cannot be reliably determined as non-platanus, the
frame is ineligible for supervised two-class training. Do not leave that crown
unlabeled in an otherwise annotated training frame: YOLO training would treat
the visible but unlabeled crown as background, conflicting with the
`other-tree` objective. Such imagery may instead be retained as an unlabelled
qualitative-inference set.

The label source must be traceable. Acceptable evidence includes an authoritative
municipal inventory linked by location, onsite expert verification, or a
documented field-survey procedure. Appearance-only guesses are not acceptable.

### Split Requirements

Partition data by road corridor, capture group, or non-overlapping spatial
area before training. No group may appear in more than one of train, validation,
and test. The test set must remain untouched until candidate-release
evaluation.

The initial data-collection target is at least 300--500 eligible frames and
2,000 platanus instances. This is a starting threshold for a baseline, not a
release-quality guarantee. Collection must deliberately include confusable
non-platanus crowns, different seasons, illumination, occlusion levels, and
capture scales.

## Training Flow

```text
OAM-TCD corrected generic single-class checkpoint
    |
    +-- local validation baseline ----------------------------------+
    |                                                                |
    +--> optional BAMFORESTS generic single-class pretraining -------+-->
                                                                     |
                                                                     v
Xi'an exhaustive platanus + other-tree two-class fine-tuning
    |
    v
Spatially independent Xi'an evaluation and error analysis
    |
    v
Fixed-shape ONNX export and immutable HF release
```

Stage 2 initializes from the `best.pt` selected by the local-validation
ablation. It trains with `single_cls: false` because the dataset has two
semantic classes. Training preserves the fixed 1280 x 1280 input and starts
with a conservative batch size of 2 for the known 8 GiB GPU constraint.

Fine-tuning configuration must expose low-learning-rate transfer controls,
including `lr0`, `lrf`, `cos_lr`, `freeze`, `warmup_epochs`, and
`close_mosaic`. The training wrapper must forward those values unchanged to
Ultralytics so they can be versioned in the YAML configuration.

## Evaluation and Release Gates

Evaluate candidate models only on the spatially independent local test split.
Every candidate release report must contain:

- Per-class and macro box mAP50-95.
- Per-class and macro mask mAP50-95.
- Per-class precision and recall at a predeclared operating threshold.
- A two-class confusion matrix.
- The rate of `other-tree` instances predicted as `platanus` at that threshold.
- False-positive examples from non-platanus crowns and non-tree background.
- Metric strata by capture group, season, illumination, occlusion, and
  resolution or GSD when those fields are available.

BAMFORESTS is adopted as the Stage 1b initializer only if it improves the
chosen validation score and does not worsen the other-tree-to-platanus error
rate relative to the OAM-TCD-only baseline. The experiment record must state
the initialization checkpoint, configuration, local split manifest, and source
dataset revisions.

## Artifact Contract

The release retains the existing fixed-input constraints:

```yaml
input:
  name: images
  dtype: float32
  shape: [1, 3, 1280, 1280]
classes:
  - platanus
  - other-tree
target_trt: "8.5.2"
```

For two classes and the standard 32 mask-prototype coefficients, the raw
detection tensor has $4 + 2 + 32 = 38$ values per candidate. Actual output
names and shapes must be inspected from the exported ONNX graph and written
verbatim into `model.yaml`; the calculated channel count is a consistency
check, not a substitute for graph inspection.

Each production release uploads ONNX, `model.yaml`, and `SHA256SUMS` to a new immutable
Hugging Face tag. The `model.yaml` class list must exactly match the root
`data.yaml` class order. Checkpoints, datasets, ONNX files, and TensorRT
engines remain outside Git.

## Implementation Boundaries

The follow-on implementation plan must provide:

- A BAMFORESTS converter with a small `--limit` smoke path, a source manifest,
  and single-class YOLO-seg output.
- A local-data validation/conversion tool that rejects non-exhaustive labels
  before placing frames into a supervised split.
- A two-class fine-tuning configuration and the required training-wrapper
  forwarding support.
- Tests for polygon conversion, class-map correctness, split disjointness, and
  local annotation eligibility.
- Chinese operator documentation for capture, annotation, split, and release
  procedures under `docs/`.

Implementation must not download datasets into Git or silently overwrite an
existing artifact tag. Non-deployable checkpoints may be uploaded only to the
separate staging repository with `artifact.yaml` and `deployable: false`.
Only `publish.py` may create a deployable production artifact.
