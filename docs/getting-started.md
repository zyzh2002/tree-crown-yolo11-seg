# 上手与开发

本文介绍本仓库的完整工作流：环境准备、训练、验证、导出、发布与排障。

## 环境准备

- Python ≥ 3.10。
- 推荐使用 `uv`（或 `pip`）：

```bash
uv sync            # 创建虚拟环境并安装依赖
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
  V1 固定为四种西安常见行道树：`platanus`（悬铃木）、`styphnolobium-japonicum`（国槐）、
  `ginkgo-biloba`（银杏）、`koelreuteria-paniculata`（栾树）。若数据资格门满足，可追加
  第 5 类 `other-tree`（其他树种），详见 `.agents/docs/specs/2026-08-05-tree-crown-age-estimation-design.md`。

## 公开预训练数据（OAM-TCD）

在本地西安四类标注就绪前，先用 OAM-TCD（`restor/tcd`）做**单类树冠预训练**，生成初始化权重。

- OAM-TCD 只用于**单类 `tree-crown` 预训练**，产物**绝不能**作为四类模型发布，也**绝不能**
  用 `publish.py` 发布。
- 转换脚本只保留 `tree` 单树实例（COCO `category_id=2`），丢弃 `canopy` 群体（`category_id=1`）。
- 根目录 `data.yaml` 是最终四类 ABI，**不会**被 OAM 单类 `data.yaml` 覆盖（两者是不同文件）。

```bash
# 1. 转换（下载约 3.5 GB，预留 10-20 GB 磁盘）
.venv/bin/python prepare_oamtcd.py --output data/oamtcd

# 2. 预训练（batch 从 1 起步，实测后升 2/4）
.venv/bin/python train.py --config configs/pretrain-oamtcd.yaml
```

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

```bash
python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt
```

- 默认走 token（`huggingface_hub`），`HF_TOKEN` 从 `.local/credentials.env`（权限 600）读取。
- SSH 方式可选：`--ssh`（`git push` 到 `git@hf.co`），需要 `git-lfs`（`git lfs install` 一次）。
- 发布时自动生成并上传 `model.yaml`（含版本、输入/输出、类别、`train_commit`、`target_trt`）。
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