# 上手与开发

本文介绍本仓库的完整工作流：环境准备、训练、验证、导出、发布与排障。

## 环境准备

- Python 3.12（由 `.python-version` 固定）。
- 推荐使用 `uv`（或 `pip`）：

```bash
uv sync --extra dev --extra data
uv run python train.py --help
```

依赖清单见 `pyproject.toml`（`ultralytics`、`onnx`、`huggingface_hub`、`pyyaml` 等）。
开发依赖（`ruff`、`pytest`）通过 `[project.optional-dependencies] dev` 安装。

## 数据集

- 数据集放在 `data/` 下，**不进 git**（`.gitignore` 已忽略）。
- 目录结构（与 `data.yaml` 一致）：

```
data/
├── images/
│   ├── train/   # 图像 + 同名 .txt 分割标注（YOLO 格式）
│   ├── val/
│   └── test/
└── labels/      # 对应标注（若使用 ultralytics 默认布局）
```

- `data.yaml` 的 `names` 类别列表是**对外契约**，必须与发布时 `model.yaml` 的类别完全一致。
  首个可部署版本固定为两类：`platanus`（悬铃木）和 `other-tree`（已可靠确认不是悬铃木的
  可区分单树树冠），详见 `.agents/docs/specs/2026-08-07-platanus-other-tree-training-strategy.md`。

## 公开预训练数据（OAM-TCD）

在本地西安两类标注就绪前，先用 OAM-TCD（`restor/tcd`）做**单类树冠预训练**，生成初始化权重。

- OAM-TCD 只用于**单类 `tree-crown` 预训练**，产物可以用 `stage.py` 备份到 HF 暂存仓库，
  但必须保持 `deployable: false`，绝不能用 `publish.py` 发布。
- 转换脚本只保留 `tree` 单树实例（COCO `category_id=2`）；纯 `canopy` 图像被排除，混合图中的
  `canopy` 群体区域会在保留单树像素的前提下被遮蔽，避免未标注多树区域被错误监督为背景。
- 无法由一个 YOLO polygon 忠实表达的多连通实例会被丢弃并计入 `dropped_invalid`。
- 根目录 `data.yaml` 是最终两类 ABI，**不会**被 OAM 单类 `data.yaml` 覆盖（两者是不同文件）。
- 旧 `oamtcd-pretrain-4` 使用的转换数据存在 canopy-only 图像被当作背景、部分多连通 RLE
  轮廓失真的问题，只能作为修复前 baseline。正式 Stage 1a 必须在修复后的新目录重新转换并训练。

```bash
# 1. 转换（下载约 3.5 GB，预留 10-20 GB 磁盘）
.venv/bin/python prepare_oamtcd.py --output data/oamtcd-rgb

# 2. 预训练（batch 按平台显存实测，详见下方说明）
.venv/bin/python train.py --config configs/pretrain-oamtcd.yaml
```

- 预训练配置 `configs/pretrain-oamtcd.yaml` 的 `batch` 是**多卡全局 batch**：
  - 8GB 单卡（RTX 2070 Super）用 `batch=2`。
  - 双 V100 32GB 用全局 `batch=4`（每卡 2），`device: "4,5"`（同 NUMA 节点，避免与
    其他用户共享 GPU 0/1 造成 DDP rank 同步变慢）。
- 换新平台**必须先跑 1 epoch** 确认显存和验证阶段不 OOM，再跑完整训练。
- OAM-TCD 验证使用 `max_det: 1000`，避免高密度图像被默认 300 个检测上限截断。
- 转换器拒绝覆盖已有输出；需要重新生成时使用新的输出目录。使用 `--skip-download` 前必须存在
  `raw/source.json`，且 parquet SHA256 与记录一致。
- 转换产物含 `data.yaml`（单类 `tree-crown`）与 `manifest.json`（split 统计、`oam_id`
  跨 split 保证、来源 revision）。
- 用 `--limit N` 可先做小子集冒烟：`.venv/bin/python prepare_oamtcd.py --output data/oamtcd-smoke --limit 40`。

## 训练

```bash
python train.py --config configs/default.yaml
```

- 超参数在 `configs/default.yaml` 配置（默认输入 1280x1280、YOLO11-seg）。
- 训练输出写入 `runs/segment/train/`（git 忽略），最佳权重在 `weights/best.pt`。

## 验证

```bash
python train.py --config configs/default.yaml --mode validate --weights runs/segment/train/weights/best.pt
```

## 导出 ONNX

```bash
python export.py --weights runs/segment/train/weights/best.pt --imgsz 1280
```

- 输入固定为 1280x1280、3 通道 NCHW，输入名 `images`。
- 导出命名固定，保证机载 `TensorRtEngine` 能按名做 ABI 校验。
- 导出 DO 需兼容目标机 TensorRT 8.5.2 算子，避免不支持的层。

## 发布到 HF 私有仓库

### 暂存 checkpoint

未完成最终训练、尚未通过发布门槛或仅作为初始化的模型上传到独立私有仓库：

```bash
python stage.py \
  --tag exp-stage1a-oamtcd-y11n-r1 \
  --stage stage1a \
  --architecture yolo11n-seg \
  --checkpoint runs/segment/oamtcd-pretrain-fixed/weights/best.pt \
  --config configs/pretrain-oamtcd.yaml \
  --args runs/segment/oamtcd-pretrain-fixed/args.yaml \
  --results runs/segment/oamtcd-pretrain-fixed/results.csv \
  --data data/oamtcd-rgb/data.yaml \
  --dataset-manifest data/oamtcd-rgb/manifest.json \
  --train-commit <训练运行使用的 Git SHA>
```

- 暂存仓库固定为 `zyzh0/tree-crown-yolo11-seg-staging`。
- 实验 tag 使用 `exp-*`，两类候选模型使用 `cand-*`。
- 暂存包生成 `artifact.yaml`，始终设置 `deployable: false`，不会生成生产 `model.yaml`。
- 暂存 tag 不可覆盖；新的运行使用新的 `rN` tag。
- 暂存时必须显式提供训练运行实际使用的 Git SHA，不能用当前工作区 HEAD 代替历史运行。

### 正式发布

```bash
python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt --train-commit <训练运行 Git SHA>
```

- 默认走 token（`huggingface_hub`），`HF_TOKEN` 从 `.local/credentials.env`（权限 600）读取。
- SSH 方式可选：`--ssh`（`git push` 到 `git@hf.co`），需要 `git-lfs`（`git lfs install` 一次）。
- 发布时自动生成并上传 `model.yaml`（含版本、输入/输出、类别、`train_commit`、`target_trt`）。
- `publish.py` 只接受 `vX.Y.Z` tag 和 `[platanus, other-tree]` 两类 ABI，并写入
  `lifecycle: release`、`deployable: true`。单类预训练 checkpoint 会被拒绝。
- 详情见 [publishing.md](publishing.md)。

## 代码风格

```bash
ruff check .
ruff format --check .
```

## 排障

- **导入失败 / 缺依赖**：`uv sync` 后仍未解决，检查 Python 版本与网络。
- **发布失败（token）**：确认 `.local/credentials.env` 存在且含有效的 `HF_TOKEN`，且 token 对该私有仓库有写权限。
- **发布失败（SSH）**：确认本地 SSH key 已加入 HF 账号、git-lfs 已安装，且对 `zyzh0/tree-crown-yolo11-seg` 有写权限。
- **类别不一致**：确认 `data.yaml` 与最新发布 `model.yaml` 的 `classes` 完全一致。
