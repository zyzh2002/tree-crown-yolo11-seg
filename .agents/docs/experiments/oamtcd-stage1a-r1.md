# OAM-TCD Stage 1a Run r1

Experiment record for the corrected OAM-TCD single-class pretraining run.
This record is the authoritative source for Stage 1a metrics and artifacts;
the training strategy spec references it instead of duplicating numbers.

## Run Metadata

- Created: 2026-08-08
- Training commit: `24e8baa1b8cea42b199fce01cb094e26e291e776`
- Data conversion commit: `a28b662df0d1cc7342167876cbbb4df602636be1`
- Config: `configs/pretrain-oamtcd.yaml`
- Data: `data/oamtcd-rgb` (all RGB JPEG)
- Model init: official `yolo11n-seg.pt`
- Task: `segment`, `single_cls: true`, `max_det: 1000`
- Device: V100 GPUs 5,6 (DDP), batch 4, epochs 50
- Duration: 7185.3 s (~2.0 h)
- Output: `runs/segment/oamtcd-stage1a-rgb-3`

## Metrics

Final epoch 50 and best values with the achieving epoch.

| Metric | Final (epoch 50) | Best | Best epoch |
|---|---:|---:|---:|
| Box precision | 0.75443 | 0.76144 | 46 |
| Box recall | 0.64439 | 0.64439 | 50 |
| Box mAP50 | 0.72566 | 0.72566 | 50 |
| Box mAP50-95 | 0.41959 | 0.41959 | 50 |
| Mask precision | 0.75623 | 0.75693 | 35 |
| Mask recall | 0.61358 | 0.61504 | 45 |
| Mask mAP50 | 0.69637 | 0.69637 | 50 |
| Mask mAP50-95 | 0.35077 | 0.35131 | 45 |

## Artifacts

- `best.pt` SHA256: `83e2a1c11a8ccfd41207bbd127f4ea69eac270d1dcb54039ec4e71211aef73b9`
- Dataset manifest: `data/oamtcd-rgb/manifest.json`
  - SHA256: `3bb65f53dbffa7efe95b2f82c6d54a8193f20f8f56ed7d8e16f675a606e70575`
  - Records raw parquet SHA256s and conversion commit.
- Run provenance: `runs/segment/oamtcd-stage1a-rgb-3/provenance.yaml`
- HF staging: `zyzh0/tree-crown-yolo11-seg-staging`
  - Tag: `exp-stage1a-oamtcd-y11n-r1`
  - HF commit: `1204de0583064cbfd28ac1dcbbc0a6aae97ac1b1`
  - `deployable: false`, lifecycle `experimental`

## Conclusion

The checkpoint is a usable single-class `tree-crown` initializer for Stage 2.
It is not a deployable two-class model and must never be published via
`publish.py`.
