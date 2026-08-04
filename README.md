# tree-crown-training

YOLO11-seg 树冠目标检测的**训练仓库**（PC 端）。负责"数据集 → 训练/验证 → 导出 ONNX →
发布到 Hugging Face 私有模型仓库"的完整生产链路，**不涉及任何机载推理**。

机载推理由同框架的 C++ 项目 [manifold-3-vision-detect](https://github.com/zyzh2002/manifold-3-vision-detect)
承担。两个仓库共享的唯一交接点是 **HF 私有模型仓库**：本仓库是产物的"生产者"，机载仓库是"消费者"。

```
训练数据 → 训练/验证 → 导出 ONNX → 发布到 HF 私有仓库
    → 机载 fetch_model.sh 拉取 → trtexec 构建 .engine
```

## 快速开始

```bash
uv sync              # 或: pip install -e .
python train.py --config configs/default.yaml
python train.py --config configs/default.yaml --mode validate
python export.py --weights runs/segment/train/weights/best.pt --imgsz 1280
python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt
```

## 文档

- [docs/README.md](docs/README.md) — 人类文档入口（中文）
- [AGENTS.md](AGENTS.md) — AI 代理工作指南（英文）
- [.agents/docs/](.agents/docs/) — 设计 spec 与实施计划（英文，agent 产物）

## 关键约束

- **数据集 / 权重 / ONNX 一律不进 git**（见 `.gitignore`）。
- 目标机基线：**TensorRT 8.5.2 / CUDA 11.4.19 / cuDNN 8.6.0**。`.engine` 由设备端 `trtexec`
  从 ONNX 构建，本仓库不产 `.engine`。
- 发布即契约：`model.yaml` 与 `data.yaml` 类别必须一致；改动必须升版本号，不能静默覆盖。
- 输入固定 1280x1280、3 通道 NCHW，导出命名固定，保证机载 `TensorRtEngine` 按名校验。