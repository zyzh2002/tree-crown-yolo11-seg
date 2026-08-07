# Training Staging and Finalization Implementation Plan

## Goal

Correct the OAM-TCD Stage 1a data path, separate non-deployable checkpoint
storage from production publication, and prepare the two-class Stage 2 training
path without starting training.

## Implemented Scope

1. Reject canopy-only source rows instead of supervising them as background.
2. Prevent disconnected RLE components from being joined by artificial polygon
   bridges.
3. Deduplicate labels, verify split disjointness, record raw-file checksums, and
   refuse in-place replacement so a corrected dataset is always generated in a
   new output directory.
4. Forward dense-validation and transfer-learning parameters through
   `train.py`; require an explicit checkpoint for resume.
5. Record run provenance beside future completed training runs.
6. Add `stage.py` for immutable `exp-*` and `cand-*` artifacts in
   `zyzh0/tree-crown-yolo11-seg-staging`.
7. Restrict `publish.py` to deployable `vX.Y.Z` two-class production releases.
8. Change the first production ABI to ordered classes `platanus`, `other-tree`.
9. Add the Stage 2 transfer-learning baseline configuration.

## Required Follow-Up Outside This Change

1. Reconvert OAM-TCD into a new output directory with the corrected converter.
2. Review the new manifest and compare removed rows/instances with the pre-fix
   dataset.
3. Run a one-epoch smoke test on the corrected data.
4. Rerun Stage 1a from official `yolo11n-seg.pt`.
5. Stage the corrected checkpoint with `stage.py`.
6. Implement and validate the Xi'an exhaustive two-class annotation pipeline.
7. Run downstream initializer and model-size ablations before production
   release.

No training command is part of this implementation change.
