# 发布流程

本文介绍如何把训练好的模型发布到 Hugging Face 私有模型仓库，以及机载仓库如何消费该产物。

## 作用

本仓库是产物"生产者"，机载仓库（`manifold-3-vision-detect`）是"消费者"。两者唯一的交接点是
**HF 私有模型仓库**。本仓库发布 ONNX + `model.yaml`，机载侧的 `fetch_model.sh` 拉取后由
`trtexec` 在设备端构建 `.engine`。

## 发布

```bash
python publish.py --tag v1.0.0 --weights runs/segment/train/weights/best.pt
```

- 需要 `HF_TOKEN`，从 git 忽略的 `.local/credentials.env`（权限 600）读取。
- `--tag` 为版本号（如 `v1.0.0`），写入 `model.yaml` 的 `version` 字段。
- 发布产物含：
  - ONNX 权重（`best.onnx` 或显式导出名）。
  - `model.yaml`（模型 ABI 契约）。

## 版本契约

- **发布即契约**：一旦发布，ONNX + `model.yaml` 就是对外契约。任何改动必须**升版本号**，
  不能静默覆盖。
- 每次发布应记录：HF 仓库 id + 版本 tag、`model.yaml` 输入/输出名与 dtype/shape、类别列表、
  训练 git commit（`train_commit`）。

## model.yaml 字段

```yaml
model_name: tree-crown-yolo11-seg
version: "<tag>"
input: { name: "images", dtype: "float32", shape: [1, 3, 1280, 1280] }
outputs: { name: "...", dtype: "...", shape: [...] }   # 按真实导出填写
classes: [ ... ]          # 与 data.yaml 完全一致
train_commit: "<git sha>" # 可回溯到训练配置
target_trt: "8.5.2"       # 目标机 TensorRT 版本
```

- `classes` 必须与 `data.yaml` 完全一致。
- `train_commit` 必须为训练运行时的 git SHA，保证可回溯。
- `target_trt` 记录机载基线（TensorRT 8.5.2 / CUDA 11.4.19 / cuDNN 8.6.0）。

## 目标机约束

- 机载基线：TensorRT 8.5.2 / CUDA 11.4.19 / cuDNN 8.6.0。
- 输出 `.engine` 由设备端 `trtexec` 从 ONNX 构建，本仓库不产 `.engine`。
- 导出时注意算子兼容 TensorRT 8.5.2，避免不支持的层。

## 交接给机载仓库的信息

完成一次发布后，请在提交 / 说明里明确记录：

1. HF 仓库地址 + 版本 tag。
2. `model.yaml` 的输入/输出名、dtype、shape。
3. 类别列表。
4. 训练的 git commit（用于回溯）。

机载仓库的 `fetch_model.sh` 会按此拉取，机载侧 `TensorRtEngine` 读 `model.yaml` 做按名 ABI 校验，
替换掉目前硬编码的合成契约（`synthetic_engine_contract.h`）。