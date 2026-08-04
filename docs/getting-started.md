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
  当前为占位（`tree-crown`），真实类别待确认后填入。

## 训练

```bash
python train.py --config configs/default.yaml
```

- 超参数在 `configs/default.yaml` 配置（默认输入 1280x1280、YOLO11-seg）。
- 训练输出写入 `runs/segment/train/`（git 忽略），最佳权重在 `weights/best.pt`。

## 验证

```bash
python train.py --config configs/default.yaml --mode validate
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

- 默认走 SSH（`git@hf.co`），使用本地 SSH key，无需 token。
- 需要 `git-lfs`（`git lfs install` 一次）。
- token 方式可选（`--token` 或 `--env` 读取 `.local/credentials.env` 中的 `HF_TOKEN`）。
- 发布时自动生成并上传 `model.yaml`（含版本、输入/输出、类别、`train_commit`、`target_trt`）。
- 详情见 [publishing.md](publishing.md)。

## 代码风格

```bash
ruff check .
ruff format --check .
```

## 排障

- **导入失败 / 缺依赖**：`uv sync` 后仍未解决，检查 Python 版本与网络。
- **发布失败（SSH）**：确认本地 SSH key 已加入 HF 账号、git-lfs 已安装，且对 `zyzh0/tree-crown-yolo11-seg` 有写权限。
- **发布失败（token）**：确认 `--token` 正确，或 `.local/credentials.env` 存在且含 `HF_TOKEN`。
- **类别不一致**：确认 `data.yaml` 与最新发布 `model.yaml` 的 `classes` 完全一致。