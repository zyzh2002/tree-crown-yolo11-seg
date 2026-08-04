# Training Repository Design (tree-crown-training)

## Goal

Establish a separate, GitHub-public training repository for the YOLO11-seg tree-crown
detection model. It is the PC-side "producer" of the model artifact; the existing
onboard repository (manifold-3-vision-detect) is the "consumer". The two repositories
share no code; their only handoff point is a versioned Hugging Face private model repo.

## Decisions (user-confirmed)

- **Training code lives in its own GitHub public repo** (`tree-crown-training`), separate
  from the C++ onboard project. Different language, toolchain, and lifecycle justify the split.
- **Model artifacts are NOT in git.** The ONNX source model is versioned and published to a
  **Hugging Face private model repo**; the dataset never enters git (`.gitignore` or DVC).
- **No HF MCP.** The onboard repo consumes the model via a download script
  (`fetch_model.sh`), not through an agent-facing MCP. Model fetching is a build/deploy path.
- **Runtime `.engine` is built on-device** with `trtexec` from the ONNX (target TensorRT 8.5.2),
  never committed. The `.engine` is device-specific (GPU + TRT version bound) and not reusable.
- **Training repo conventions mirror the onboard repo** where applicable: trunk-based branching,
  Conventional Commits, English code/comments/agent docs + Chinese human docs, git-ignored
  `.local/credentials.env` for secrets (HF_TOKEN), git-ignored data/artifacts. Python formatting
  uses ruff/black (NOT the onboard C++ `.clang-format`).

## Repository Handoff Point

```
┌──────────────────┐   ONNX + model.yaml    ┌──────────────────────┐
│  Training repo   │ ─────────────────────> │  HF 私有模型仓库      │
│  (GitHub 公有)    │    (版本化发布)         │  tree-crown-yolo11-seg│
└──────────────────┘                        └──────────────────────┘
                                                    │
                                             fetch_model.sh 拉取
                                                    ▼
┌──────────────────┐  ONNX + model.yaml   ┌──────────────────────┐
│  Onboard repo    │ ───────────────────> │ Manifold 3 设备        │
│  (本项目)         │    trtexec 构建 .engine│  (TensorRT 8.5.2)    │
└──────────────────┘                      └──────────────────────┘
```

## Model Artifact Contract (model.yaml)

`publish.py` uploads `model.yaml` alongside the ONNX. It is the machine-readable ABI
contract for the onboard `TensorRtEngine` (replaces the hardcoded
`synthetic_engine_contract.h`):

```yaml
model_name: tree-crown-yolo11-seg
version: "<tag>"
input: { name: "images", dtype: "float32", shape: [1, 3, 1280, 1280] }
outputs: { name: "...", dtype: "...", shape: [...] }   # per actual export
classes: [ ... ]          # identical to data.yaml
train_commit: "<git sha>" # traceable to training config
target_trt: "8.5.2"       # onboard TensorRT version
```

## Initialization Sequence

1. `git init` if needed.
2. Mirror onboard conventions into `AGENTS.md`, `CLAUDE.md`, `docs/`, `docs/README.md`.
3. Create directory structure and `.gitignore` (data, weights, ONNX, venv, `.local/`).
4. Write `README.md`, `pyproject.toml`, `data.yaml` (placeholder classes).
5. Scaffold `train.py`, `export.py`, `publish.py` skeletons.
6. Commit with an English conventional commit (e.g. `chore: scaffold training repo`).

## Acceptance Criteria

- Initialization commit present; contains no data, weights, or ONNX.
- `train.py` runs and reproduces; `export.py` produces fixed-name ONNX; `publish.py` uploads
  ONNX + `model.yaml` with a version tag.
- `model.yaml` classes match `data.yaml`; includes `train_commit`.
- Onboard `fetch_model.sh` can pull the artifact and build a `.engine` on-device.

## Open Items

- Actual tree-crown class list (placeholder in `data.yaml` until dataset is confirmed).
- Dataset size / provenance / versioning (DVC vs `.gitignore`).

## Resolved Decisions

- **HF repo**: `zyzh0/tree-crown-yolo11-seg` (private, owner `zyzh0`). Verified
  read/write access via both token and SSH (`git@hf.co:zyzh0/tree-crown-yolo11-seg`).
- **Publish mechanism**: token-based `huggingface_hub` upload by default
  (`HF_TOKEN` from `.local/credentials.env`, mode 600); SSH `git push` is the
  fallback via `--ssh`.
- `git-lfs` required for the SSH path (`*.onnx`); the HF repo `.gitattributes`
  already maps `*.onnx` to LFS.