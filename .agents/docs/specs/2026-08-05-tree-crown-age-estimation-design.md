# Xi'an Street Tree Species Instance Segmentation Design

## Goal

Build and publish a YOLO11-seg model that detects, identifies, and delineates
individual crowns of a closed set of common Xi'an street-tree classes from
top-down RGB imagery.

This training repository (`tree-crown-training`) owns the model-training,
validation, ONNX-export, and HF-publish pipeline. The main repository
(`manifold-3-vision-detect`, C++) owns device inference, result post-processing,
business-category mapping, geometry derivation, and any age-estimation policy.

This is a design document only. No training code is written (or changed) in
this pass; the scaffolded `train.py` / `export.py` / `publish.py` remain as-is
until an implementation plan is approved.

## Confirmed Decisions

- **Task:** multi-class tree species instance segmentation from top-down RGB
  imagery.
- **Model:** `yolo11n-seg` initially; use a larger YOLO11 segmentation variant
  only after a measured accuracy-versus-device-latency comparison.
- **Input contract:** one fixed-size RGB image, `float32`, NCHW,
  `[1, 3, 1280, 1280]`.
- **Target classes:** a closed set of four Xi'an street-tree species for V1.
- **`other-tree` (conditional):** V1 includes an `other-tree` instance class
  **only if** the final top-down training data is verified to contain
  exhaustive, species-reliable annotations for non-target crowns. When enabled
  it is the fifth ordered ABI class and requires a new model version tag. If
  the data-eligibility gate is not met, the model is released with the four
  target classes only and `other-tree` is omitted.
- **Artifact versioning:** a class-list addition, removal, reordering, or
  semantic change is an ABI change and requires a new HF version tag.
- **Repository boundary:** this repo does not process tree height, GSD, GPS,
  camera intrinsics, crown diameter in meters, age, or business rules.
- **Age estimation:** remains in the main repo. The main repo decides whether
  a predicted `platanus` instance is eligible for age estimation, using the
  model confidence and any external metadata or policy it requires.

## V1 Class ABI

The class list is ordered and immutable within a published model version. The
four target classes are always present. `other-tree` is included only when the
data-eligibility gate passes.

Target classes:

```yaml
names:
  0: platanus
  1: styphnolobium-japonicum
  2: ginkgo-biloba
  3: koelreuteria-paniculata
```

With `other-tree` enabled:

```yaml
names:
  0: platanus
  1: styphnolobium-japonicum
  2: ginkgo-biloba
  3: koelreuteria-paniculata
  4: other-tree
```

| ID | Stable model label | Common Chinese name | Labeling policy |
|---:|---|---|---|
| 0 | `platanus` | 悬铃木 | Genus-level label. Do not force uncertain one-ball, two-ball, or three-ball records into species-level labels. |
| 1 | `styphnolobium-japonicum` | 国槐 | Use the accepted scientific name consistently in datasets and artifacts. |
| 2 | `ginkgo-biloba` | 银杏 | Use one class for both male and female trees. |
| 3 | `koelreuteria-paniculata` | 栾树 | Do not merge visually or botanically distinct related species without an explicit future ABI decision. |
| 4 | `other-tree` (conditional) | 其他树种 | A single, individually identifiable crown verified to be a non-target species. See the `other-tree` rules below. |

The labels are model-facing visual semantics. The main repo may map them to
business enums, Chinese display labels, workflows, and age-estimation policies,
but it must not reinterpret a class ID as a different visual species.

## `other-tree` Semantics

When enabled, `other-tree` means:

> An individually identifiable crown, visible in the image, that is confirmed
> to be a species other than the four target classes.

It is **not**:

- Background (buildings, vehicles, grass, shrubs, ground).
- A `tree-canopy` group mask where multiple touching crowns cannot be separated.
- A tree whose species is unknown or whose label source is unreliable.
- A substitute for an unannotated target tree.

The main repo may map `other-tree` to "other species", "age estimation not
supported", or "excluded from a given workflow", but it must not change the
model semantics of the four target classes.

## System Architecture

```text
Top-down RGB image
        |
        v
+-----------------------------------------------+
| Training repo                                  |
| YOLO11-seg multi-class instance segmentation   |
| 4 (or 5) species -> boxes + species scores + masks
+-----------------------------------------------+
        |
        | ONNX + model.yaml + SHA256SUMS
        v
HF private model repo (immutable version tag)
        |
        v
+-----------------------------------------------+
| Main repo / onboard C++                        |
| fetch + checksum/ABI validation + trtexec      |
| inference + NMS + mask reconstruction          |
| class-to-business mapping                      |
| geometry / tree height / GSD / age policy      |
+-----------------------------------------------+
```

The training repository endpoint is the HF private model repo. It never deploys
to a device and never builds a TensorRT `.engine`.

## Model Interface

### Input

```yaml
input:
  name: images
  dtype: float32
  shape: [1, 3, 1280, 1280]
```

The main repo must apply the same deployed preprocessing as training/export:

- Decode image pixels as RGB.
- Letterbox while preserving aspect ratio to `1280 x 1280`.
- Track scale and padding so boxes and masks can be transformed back to original
  image coordinates.
- Convert to NCHW `float32`.
- Normalize pixel values to `[0.0, 1.0]`.

The model does not accept tree height, GSD, GPS, species metadata, camera
parameters, or age-related data.

### Raw ONNX Outputs

YOLO11-seg produces raw detection and mask-prototype tensors, not final binary
mask images and not age estimates.

With `nc` classes and the standard 32 mask-prototype channels, the per-candidate
detection channel count is:

```text
4 box values + nc class scores + 32 mask coefficients = 36 + nc
```

For the four-class release:

```text
40 values per candidate             (4 + 4 + 32)
detection tensor:      [1, 40, 33600]
mask prototype tensor: [1, 32, 320, 320]
```

For the five-class release (if `other-tree` is enabled):

```text
41 values per candidate             (4 + 5 + 32)
detection tensor:      [1, 41, 33600]
mask prototype tensor: [1, 32, 320, 320]
```

The `33600` and `320` figures correspond to the fixed `1280 x 1280` input and
are illustrative. Output names and final shapes are **not hardcoded by this
specification**. They must be read from the actual exported ONNX graph and
written exactly into the released `model.yaml`. A consumer must validate the
released ABI by name, dtype, and shape before installation.

### Main-Repo Post-Processing

The main repo is responsible for:

1. Reading raw class scores and selecting the winning class.
2. Applying per-class confidence filtering and class-aware NMS.
3. Combining each accepted detection's 32 mask coefficients with the mask
   prototypes.
4. Applying sigmoid, bounding-box crop, resize, thresholding, and inverse
   letterbox transformation.
5. Producing each instance in original-image coordinates:

```text
class_id
class_name
confidence
bbox_xyxy
binary_instance_mask
```

The model does not output tree height, GSD, crown diameter in meters, species
confirmation metadata, age, or an age bucket.

## Training Data Strategy

### Local Xi'an Data Is Mandatory for Deployment

Public data may provide initialization and coverage, but a deployable model must
be fine-tuned and independently evaluated on top-down imagery from Xi'an or an
operationally equivalent local capture domain.

Every local target instance must have:

```text
image
+ one crown polygon or mask
+ one of the V1 labels (target species, or `other-tree` when enabled)
+ source / capture group identifier
+ acquisition date or season
+ image resolution or GSD metadata when available
```

The final model does not consume GSD, but retaining capture metadata is required
to diagnose domain shift and to construct valid spatial splits.

### Annotation Rules

- Annotate every visible instance of a target species in an annotated frame.
  Do not selectively annotate only large or easy crowns.
- When `other-tree` is enabled, also annotate every visible non-target crown
  that can be reliably identified. Keep non-target trees that cannot be
  reliably labeled as un-annotated negative context; do not guess their class.
- Exclude instances whose species cannot be verified from the label source.
  Do not guess species from appearance during annotation.
- Use one complete, non-self-intersecting polygon per visible crown.
- For heavily occluded, truncated, or inseparable crowns, use a documented
  exclusion rule consistently rather than generating unreliable masks.
- Record source, license, label authority, annotation tool/version, and any
  conversion parameters in a dataset manifest. Image data and labels remain
  outside Git.

### Data Eligibility Gate for `other-tree`

`other-tree` participates in the final training set only if the data source
satisfies **all** of:

1. Top-down / orthophoto RGB imagery with a viewpoint close to deployment.
2. An instance polygon or mask for every distinguishable crown.
3. A verifiable species label for every crown, or reliable confirmation that it
   is a non-target species.
4. Both target and non-target trees are exhaustively annotated, so unlabeled
   trees are not implicitly treated as background.
5. `other-tree` is composed of multiple non-target species, including
   Xi'an-common confusable species, not a single species.

If the gate is not met, release the four-class model and omit `other-tree`.

### Disallowed Mappings

The following sources cannot contribute `other-tree` labels:

| Source | Reason |
|---|---|
| OAM-TCD / `restor/tcd` | Annotates generic `tree` / `tree-canopy`; whether a crown is a target species is unknown. Mapping to `other-tree` would mislabel potential target trees. |
| geotree | Generic crown class only, no species labels. |
| Urban Street Tree Dataset | Has species labels but is ground/side-view; must not be mixed into the final top-down YOLO11-seg training set. |
| TreeAI partially labeled images | Unlabeled instances are treated as background; cannot contribute to complete multi-class instance-segmentation training. |
| Local imagery with unverifiable species | Cannot be filled with appearance-based guesses. |

### Public Data Usage

| Source | Allowed role | Restrictions |
|---|---|---|
| OAM-TCD / `restor/tcd` | Generic single-class crown-instance pretraining | Use its `tree` individual-instance labels only for instance segmentation. Do not merge `tree-canopy` group masks into individual-tree labels. |
| TreeAI Global Initiative | Candidate top-down species-transfer source | Use only after access, label format, target-species coverage, and license terms are verified. The current CC BY-NC-ND-style terms require explicit approval before a trained artifact is published to HF. |
| Urban Street Tree Dataset | Optional species appearance reference or separate side-view classification experiment | Do not directly mix its ground/side-view images into the final top-down YOLO11-seg training set. |
| BAMFORESTS or similar crown datasets | Optional generic instance-segmentation pretraining | Do not treat forest-domain species labels as substitutes for Xi'an street-tree labels. |

A public source must not be used merely because its species names overlap with
the V1 classes. Its viewpoint, annotation completeness, license, and label
semantics must first match the intended use.

## Training Flow

```text
Optional generic crown-instance pretraining
    |
    | OAM-TCD individual `tree` instances only
    v
Top-down multi-class species fine-tuning
    |
    | local Xi'an target masks + verified V1 species labels
    | optional approved TreeAI transfer data
    | optional `other-tree` if the data-eligibility gate passes
    v
Spatially independent Xi'an validation and test
    |
    v
Fixed-shape ONNX export
    |
    v
HF model release
```

The final fine-tuning dataset must use exactly the V1 class IDs and order. A
generic single-class checkpoint is used only as initialization; it is not
combined with unlabeled or semantically incompatible public data in the final
multi-class YOLO training run.

## Evaluation Requirements

Every candidate release must report:

- Per-class and macro `box` mAP50-95.
- Per-class and macro `mask` mAP50-95.
- Per-class precision and recall at the chosen operating confidence threshold.
- Confusion matrix across the released classes (four, or five with `other-tree`).
- Results on the spatially independent Xi'an test split.
- False-positive analysis on the non-target-tree negative subset.
- When `other-tree` is enabled, its false-positive rate into any of the four
  target classes.
- Results stratified by capture condition when available: season, illumination,
  occlusion, GSD/resolution, and road corridor.

Before release-candidate training starts, the project must record its acceptance
thresholds and confidence/NMS operating point in a versioned evaluation
configuration. A release result must not define its own threshold after testing.

## Export and Publish Contract

The exported ONNX must remain compatible with the onboard baseline:

```text
TensorRT 8.5.2
CUDA 11.4.19
cuDNN 8.6.0
```

Export requirements:

```text
input size: 1280 x 1280
layout: NCHW
dtype: float32
dynamic axes: disabled
opset: 17
half precision: disabled
```

Each HF release uploads the following immutable artifact set under a new tag:

```text
<model>.onnx
model.yaml
SHA256SUMS
```

`model.yaml` must include the exact inspected ONNX input/output ABI and the
released class list. Four-class example:

```yaml
model_name: tree-crown-yolo11-seg
version: vX.Y.Z
input:
  name: images
  dtype: float32
  shape: [1, 3, 1280, 1280]
outputs:
  # Inspected from the actual ONNX graph.
classes:
  - platanus
  - styphnolobium-japonicum
  - ginkgo-biloba
  - koelreuteria-paniculata
train_commit: <git-sha>
target_trt: "8.5.2"
```

With `other-tree` enabled, `classes` appends `other-tree` as the fifth entry.

A published tag is immutable. Any model, class ABI, preprocessing, output ABI,
or operating-policy change requires a new tag. ONNX files, checkpoints, datasets,
and `.engine` files never enter this Git repository.

## Main-Repo Responsibilities

The main repo owns all processing after raw model inference:

- ONNX artifact download, SHA256 verification, and ABI validation.
- Device-local TensorRT `.engine` creation with `trtexec`.
- Image preprocessing and inverse coordinate transforms.
- Detection filtering, NMS, and mask reconstruction.
- Mapping the stable model labels to business types and display names.
- Deciding how no-detection, low-confidence, or unsupported-tree cases are
  represented in the product.
- Deriving tree height, GSD, position, camera geometry, and physical crown
  measurements.
- Deciding whether a predicted `platanus` instance is sufficiently trusted and
  eligible for allometric age estimation.
- Computing and presenting any age estimate or `unknown` result.

## Non-Goals

This repository does not:

- Train a tree-age regression model.
- Train or run a separate tree-height model.
- Calculate physical crown diameter or meters-per-pixel.
- Classify trees beyond the released V1 classes.
- Determine business-policy meaning for a class ID.
- Build or distribute TensorRT `.engine` files.
- Deploy models to an onboard device.

## Risks

- **Top-down RGB limits:** similar crown color, morphology, pruning patterns,
  shadows, and seasonal changes can make species visually ambiguous.
- **Closed canopy:** touching crowns remain difficult to separate reliably from
  RGB imagery alone.
- **Domain shift:** public data, especially ground-view imagery or non-Xi'an
  forest imagery, cannot establish deployment accuracy in Xi'an.
- **Class imbalance:** the four target species may have very different local
  prevalence; collection and split design must prevent rare-class collapse.
- **Label noise:** inaccurate municipal records or visually guessed labels will
  cap model accuracy more severely than additional unverified data can improve it.
- **`other-tree` risk:** if enabled without exhaustive, species-reliable
  non-target annotations, it becomes a junk class that absorbs unlabeled trees
  and degrades the four target classes.
- **License restrictions:** a source license may prohibit a trained-model
  artifact's distribution even when download and local research use are allowed.

## Acceptance Criteria

- The released model recognizes and instance-segments exactly the ordered V1
  classes defined in this specification (four, or five with enabled `other-tree`).
- `data.yaml`, training labels, ONNX `model.yaml`, and the main-repo ABI
  configuration use exactly the same class IDs, order, and stable names.
- Training and test splits are spatially/capture-group independent.
- The deployment candidate is fine-tuned and evaluated on verified local Xi'an
  top-down imagery.
- `other-tree`, if enabled, meets the data-eligibility gate and is reported with
  its own class metrics and target-class false-positive rate.
- The release report contains all required per-class, macro, spatial-holdout,
  and non-target false-positive metrics.
- The exported static ONNX is inspected, recorded in `model.yaml`, checksumed,
  and successfully built by the target TensorRT 8.5.2 environment.
- No training data, labels, weights, ONNX artifacts, or TensorRT engine enters
  Git.

## Traceability

- The four target species and their stable labels are grounded in Xi'an official
  greening guidance: 悬铃木 (`platanus`), 国槐 (`styphnolobium-japonicum`),
  银杏 (`ginkgo-biloba`), and 栾树 (`koelreuteria-paniculata`) appear in the
  Xi'an street-tree planting/maintenance technical specification and the
  city greening plant-configuration design guidance.
- Candidate public datasets are documented in `.agents/docs/` with their source,
  viewpoint, annotation granularity, license, and the reason for their role
  (pretraining, transfer, classification reference, or exclusion).