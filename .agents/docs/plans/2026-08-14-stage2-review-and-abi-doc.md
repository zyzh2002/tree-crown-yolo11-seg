# Stage 2 Review and Producer ABI Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the factual Stage 2 review record and add a producer-only model artifact ABI convention without changing training behavior or onboard implementation.

**Architecture:** Keep the review record as an evidence and finding log. Add `docs/model-abi.md` as the human-facing Chinese source for the training repository's published ONNX and `model.yaml` contract. The ABI document describes only producer-owned artifact semantics and remains independent of any consumer repository implementation.

**Tech Stack:** Markdown documentation, existing `publish.py` contract, static YOLO11-seg ONNX metadata.

## Global Constraints

- Modify only `.agents/docs/specs/2026-08-13-stage2-roadmap-review.md` and create `docs/model-abi.md` for the requested deliverables.
- Do not modify `pyproject.toml`, training configs, data acquisition plans, training code, or the onboard repository.
- Do not describe onboard code, implementation gaps, phases, or repository paths in `docs/model-abi.md`.
- Keep producer classes ordered as `platanus`, `other-tree`.
- Preserve the fixed producer input contract `[1, 3, 1280, 1280]` float32 NCHW.
- Do not commit or push any file, including the local-only `STAGE2-ROADMAP-REVIEW-CONCLUSION.md`.

---

### Task 1: Correct Stage 2 review record

**Files:**
- Modify: `.agents/docs/specs/2026-08-13-stage2-roadmap-review.md`

**Interfaces:**
- Consumes: Existing evidence, local source inspections, and the approved review conclusion.
- Produces: A self-consistent review record that distinguishes verified facts, conditional calculations, unsupported claims, and unresolved decisions.

- [x] **Step 1: Correct verified facts**

  Update the NYC count to `ginkgo = 21,024` and identify `29,258` as Japanese zelkova. Separate DJI official camera specifications from the onboard measured `1440x1080 NV12 @ approximately 30 fps` stream. Correct preprocessing to `160 px` padding on each vertical side and `320 px` total.

- [x] **Step 2: Qualify architecture and TensorRT claims**

  Describe `[1, 38, 33600]` plus `[1, 32, 320, 320]` as an architecture-derived and publisher-enforced contract. Retain the general TensorRT 8.5 opset-17 support statement, remove the claim that FP32 is unaffected, and require target-device parser/build and numerical validation.

- [x] **Step 3: Correct each pending finding**

  Revise P1-1 to describe Esri as a compliance risk and state that NYC 2014/2016 imagery reduces but does not eliminate temporal mismatch. Revise P1-2 so inventory-only crops are not presented as a valid mask-head pretraining workaround. Reframe P2-2 as package-level license provenance. Remove the mosaic `stretch` wording from P2-3. Make P3-2 conditional and distinguish source-frame `9.659 cm/px` from model-content `10.87 cm/px`. Reframe P3-3 as reproducibility risk. State that P3-4's 25% is input-area padding, not measured compute or latency savings.

- [x] **Step 4: Record the omitted producer-side conclusion**

  Add a finding that the published artifact contract is architecture-derived and not evidence that a consumer implementation has completed integration. Keep this wording generic and do not describe the onboard repository or its code.

- [x] **Step 5: Replace unauditable test history**

  Replace the unsupported historical `49/63` claim with the currently reproducible `63 passed, 3 warnings` result, or explicitly label the old run as unaudited if historical context is retained.

### Task 2: Add producer-only ABI convention

**Files:**
- Create: `docs/model-abi.md`

**Interfaces:**
- Consumes: `publish.py` checks at lines 141-162, root `data.yaml`, and the existing publishing/version rules.
- Produces: A Chinese human-facing producer contract for `model.onnx`, `model.yaml`, and `SHA256SUMS`.

- [x] **Step 1: Define document boundary and status**

  State that the document defines the model artifact produced by this repository. Explicitly exclude consumer implementation, runtime code, deployment status, and consumer repository details.

- [x] **Step 2: Document the fixed artifact contract**

  Specify the input as `images`, float32, `[1, 3, 1280, 1280]`, NCHW. Specify two float32 outputs: a detection tensor `[1, 38, 33600]` and a prototype tensor `[1, 32, 320, 320]`. State that output names must be copied from the inspected ONNX graph rather than guessed.

- [x] **Step 3: Document output semantics**

  Define detection channels `0..3` as decoded `cx, cy, w, h`, `4..5` as the ordered class probabilities, and `6..37` as 32 mask coefficients. Define the P3/8, P4/16, and P5/32 candidate count calculation. State class-score activation, mask coefficient/prototype activation, and mask reconstruction semantics without prescribing consumer implementation.

- [x] **Step 4: Document preprocessing and coordinate convention**

  Record RGB, NCHW, and `[0, 1]` normalization. Require every released source-to-model transform and coordinate convention to be recorded with the artifact, while keeping camera-specific calibration outside the tensor ABI and not silently freezing an unvalidated interpolation or padding choice.

- [x] **Step 5: Document metadata, immutability, and release gates**

  Include a `model.yaml` example with `schema_version`, input, output roles, classes, `train_commit`, and `target_trt`. Require exact class order, actual graph names, immutable version tags, and a matching `SHA256SUMS` file. Do not claim that any external consumer currently validates or supports the contract.

### Task 3: Verify documentation-only changes

**Files:**
- Test: `.agents/docs/specs/2026-08-13-stage2-roadmap-review.md`
- Test: `docs/model-abi.md`

- [x] **Step 1: Check the diff for whitespace errors**

  Run `git diff --check` and confirm it reports no errors.

- [x] **Step 2: Search for forbidden stale claims**

  Search the two changed documents for `80 px`, `ginkgo 29,258`, `FP32 export is unaffected`, `mosaic epochs stretch`, and any onboard repository or runtime implementation references in `docs/model-abi.md`.

- [x] **Step 3: Read both documents end to end**

  Confirm that the review record and producer ABI document do not contradict each other, and that no training configuration or dependency claim was silently changed.

- [x] **Step 4: Confirm no commit operation occurred**

  Run `git status --short` and verify the requested files are only working-tree changes; leave `STAGE2-ROADMAP-REVIEW-CONCLUSION.md` uncommitted.
