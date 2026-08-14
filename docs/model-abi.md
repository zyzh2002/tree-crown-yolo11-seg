# 模型产物 ABI 约定

## 文档边界

本文只定义本仓库生产的模型产物契约。适用对象是：

- `model.onnx`：固定输入输出的模型文件；
- `model.yaml`：模型名称、版本、输入输出和类别语义；
- `SHA256SUMS`：发布文件校验清单。

本文不定义任何外部程序的代码结构、加载流程、运行时实现或部署状态。外部消费者只需要依据发布产物中的契约读取模型，不应反向改变本仓库的产物语义。

## 契约原则

- 发布的 ONNX、`model.yaml` 和 `SHA256SUMS` 必须属于同一个不可变版本 tag。
- `model.yaml` 必须记录实际导出 ONNX graph 中的输入输出名称、dtype 和 shape。
- 类别顺序是模型语义的一部分，不能在不升版本的情况下调整。
- 输入输出 shape、输出布局、类别语义或预处理规则发生变化时，必须创建新的版本 tag。
- 训练 checkpoint、数据集和 ONNX 文件不进入本仓库 Git。
- `target_trt` 只记录发布时的目标兼容基线，不代表本仓库构建 TensorRT engine。

## 固定输入

第一版可部署两类模型使用以下模型张量输入：

| 字段 | 约定值 |
|---|---|
| Name | `images` |
| Dtype | `float32` |
| Shape | `[1, 3, 1280, 1280]` |
| Layout | NCHW |
| Channel order | RGB |
| Value range | `[0, 1]` |

模型张量坐标以固定的 `1280x1280` 输入为基准。任何源图到模型张量的缩放、padding 和坐标映射都必须在发布前固定并记录；不能只依赖调用方的默认行为。

## 固定输出

标准两类 YOLO11-seg 模型包含两个 float32 输出。输出名称必须从实际 ONNX graph 读取，不能凭经验猜测；下面的 `output0` 和 `output1` 仅为格式示例。

| Role | 示例名称 | Dtype | Shape | Layout |
|---|---|---|---|---|
| `detection` | `output0` | `float32` | `[1, 38, 33600]` | BCH |
| `prototypes` | `output1` | `float32` | `[1, 32, 320, 320]` | BCHW |

### Detection tensor

检测输出按 `[batch, channel, candidate]` 排列，通道约定如下：

| Channel range | Meaning |
|---|---|
| `0..3` | 已解码的 `cx, cy, w, h` |
| `4..5` | 按 `classes` 顺序排列的类别分数 |
| `6..37` | 32 个 mask coefficient |

候选点来自三个检测尺度：

```text
P3/8:  160 x 160 = 25,600
P4/16:  80 x 80  =  6,400
P5/32:  40 x 40  =  1,600
总计:              33,600
```

box 坐标是模型输入坐标系中的已解码像素值，不是归一化到 `[0, 1]` 的值。类别分数是模型导出输出中的类别概率；发布验证必须确认其激活语义，不能在不同实现之间默认重复应用 sigmoid。

### Prototype tensor

Prototype 输出按 `[batch, coefficient, height, width]` 排列：

```text
[1, 32, 320, 320]
```

单个实例的 mask logits 由 32 个 coefficient 与对应 prototype 平面线性组合得到，再对 logits 应用 sigmoid。mask 的阈值、裁剪、上采样和坐标映射必须与发布验证所采用的后处理定义保持一致。

## 类别 ABI

第一版可部署模型的类别顺序固定为：

```yaml
classes:
  - platanus
  - other-tree
```

语义定义：

- `platanus`：经过可靠证据确认属于 Platanus 属的树冠；
- `other-tree`：经过可靠证据确认不是 Platanus 的、可单独区分的树冠；
- 未知植被、背景和无法可靠分类的对象不能被随意归入 `other-tree`。

`data.yaml` 的 `names` 顺序必须与发布的 `model.yaml` 的 `classes` 完全一致。

## 发布 metadata

生产 `model.yaml` 至少应表达以下信息：

```yaml
schema_version: 1
abi_version: 1
model_name: tree-crown-yolo11-seg
version: "vX.Y.Z"
lifecycle: release
deployable: true

input:
  name: images
  dtype: float32
  shape: [1, 3, 1280, 1280]

outputs:
  - name: "<actual detection output name>"
    role: detection
    dtype: float32
    shape: [1, 38, 33600]
    layout: BCH
  - name: "<actual prototype output name>"
    role: prototypes
    dtype: float32
    shape: [1, 32, 320, 320]
    layout: BCHW

classes:
  - platanus
  - other-tree

train_commit: "<full training git sha>"
target_trt: "8.5.2"
```

其中：

- `schema_version` 表示 metadata 文件格式版本；
- `abi_version` 表示输入输出和语义契约版本；
- `version` 必须与发布 tag 一致；
- `outputs[].name` 必须来自实际 ONNX graph；
- `outputs[].role` 用于区分 detection 和 prototypes，不能只依赖输出排列顺序；
- `train_commit` 必须能回溯到训练配置、数据版本和训练环境；
- `target_trt` 记录发布目标 TensorRT 基线。

## 版本与校验

一次正式发布必须包含：

```text
model.onnx
model.yaml
SHA256SUMS
```

`SHA256SUMS` 必须同时覆盖 `model.onnx` 和 `model.yaml`。已发布 tag 不得移动或覆盖；以下任一变化都必须新建版本：

- 输入名称、dtype 或 shape；
- 输出数量、名称、dtype、shape、layout 或 role；
- box、class、mask 的数值语义；
- 类别名称或类别顺序；
- 模型输入的预处理和坐标约定；
- 目标兼容基线或其他会影响产物解释的 metadata。

## 发布前检查

- ONNX graph 实际输入为 `images`、float32、`[1,3,1280,1280]`。
- ONNX graph 恰好包含两个 float32 输出。
- 一个输出为 `[1,38,33600]`，另一个输出为 `[1,32,320,320]`。
- 输出名称、role、dtype 和 shape 已写入 `model.yaml`。
- `model.yaml.classes` 与根目录 `data.yaml` 完全一致。
- `train_commit` 是训练运行实际使用的完整 Git SHA。
- `SHA256SUMS` 在最终文件生成后计算。
- 版本 tag 未存在且不会覆盖已有产物。
- 目标 TensorRT 兼容性已按发布流程单独验证；仅有 opset 数字匹配不能替代实际 graph 验证。
