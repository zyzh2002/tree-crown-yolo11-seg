# OAM-TCD 预训练实现交接文档

> **交接对象**：接手的 AI agent 或开发者。
> **文档语言**：英文代码注释/提交信息，本文档为交接说明使用中文（属 agent-facing 工作产物，放在仓库根目录，暂停后由新 agent 接手）。

## 1. 背景与目标

本项目（`tree-crown-yolo11-seg`）最终目标是训练一个**西安 4 类行道树实例分割模型**：

```
names:
  0: platanus          悬铃木
  1: styphnolobium-japonicum  国槐
  2: ginkgo-biloba      银杏
  3: koelreuteria-paniculata  栾树
```

当前没有西安本地标注数据，因此先做**单类通用树冠预训练**（OAM-TCD），生成一个初始化权重，供未来西安 4 类微调使用。

**关键约束（不可违反）：**
- OAM-TCD 只用于**单类 `tree-crown` 预训练**，作为初始化权重。
- OAM-TCD 预训练产物**绝不能**作为 4 类模型发布，**绝不能**用 `publish.py` 发布。
- 根目录 `data.yaml` 是最终 4 类 ABI，**不能**被 OAM 单类数据覆盖。
- OAM-TCD 只用 `tree` 单树实例（COCO `category_id=2`），丢弃 `canopy` 群体（`category_id=1`）。

## 2. 当前分支与状态

```
当前分支：feat/oamtcd-pretraining
基于：main（含已提交的 7c5e4b3 配置修复）
```

### 工作区改动（未提交）
```
?? HANDOFF-OAMTCD.md          # 本交接文档（git 忽略，工作产物）
```

所有代码改动均已提交（见下）。工作区干净，仅剩本交接文档未跟踪。

### 已完成的提交（在 feat/oamtcd-pretraining 上）
```
6c36ff1 chore: tune OAM-TCD pretrain config after full-data validation
df2848d chore: set OAM-TCD pretrain batch to 4 after GPU measurement
771f9ff fix: default empty project so output lands in runs/<task>/
4318911 fix: decode uncompressed COCO RLE in OAM-TCD conversion
c1641f7 chore: add pyright config pointing at venv
06e45d3 fix: harden training config forwarding and metrics
f53960b chore: add data deps and scope ruff to project source
b621916 style: apply ruff format to publish.py and checksum tests
d8e675d docs: document OAM-TCD pretraining workflow and reject geotree
e0f2858 feat: add OAM-TCD single-class tree-crown pretraining config
1180731 fix: return zero on image decode error in manifest tally
6b4970d fix: report per-split instance counts in OAM-TCD manifest
2bd121a fix: correctly identify OAM-TCD test split in prepare_oamtcd
7c5e4b3 fix: correct YOLO11-seg model name and conservative default batch
```

**测试**：`pytest` 全绿（30 passed）。`ruff check .` / `ruff format --check .` 干净。

## 3. 已完成的工作

### 3.1 `train.py` 加固（已完成，测试通过）
- 修复 fallback 模型名：`yolov11n-seg.pt` → `yolo11n-seg.pt`。
- 修复 falsy override 被 `or` 吞掉的 bug：改用 `in` 成员判断优先级（使 `batch=0`、`workers=0`、`amp=False` 等生效）。
- 透传 `deterministic / workers / cache / amp / patience / project / name / plots / single_cls`。
- validate 模式同时输出 `box mAP50-95` 和 `mask mAP50-95`。
- 训练结束报告真实 `trainer.save_dir` / `trainer.best`，不再硬编码路径。
- 新增 `--data` CLI flag。

**验证**：`pytest tests/test_train.py -q` → 9 passed。

### 3.2 LSP / pyright 配置（已完成）
- 全局 `~/.config/opencode/opencode.jsonc` 新增 `lsp: { pyright: { enabled: true } }`。
- 全局安装 `pyright`（通过 `npm i -g pyright`）。
- 项目新增 `pyrightconfig.json` 指向 `.venv`，使 pyright 能解析项目依赖。
- **注意**：改全局 opencode 配置后需**重启 opencode** 才生效。

### 3.3 依赖（已完成）
`pyproject.toml` 新增 `data` 可选依赖组：
```toml
data = [
    "datasets>=2.20.0",
    "pycocotools>=2.0.0",
    "polars>=1.0.0",
]
```
已执行 `uv sync --extra dev --extra data`，`.venv` 已含 polars/pycocotools/pytest/ruff。

## 4. 已完成的工作（原"未完成"项已全部完成并提交）

原 §4.1-§4.4 的待办项已在 `feat/oamtcd-pretraining` 分支全部完成：

- **4.1 split bug** → 已修复（commit `2bd121a`）：`load_frame()` 分别读 train/test shard 并加 `_is_test` 列，新增纯函数 `row_split()`，去除路径字符串 hack。
- **4.2 测试** → 已创建 `tests/test_prepare_oamtcd.py`（16 个测试，合成数据不联网）。后又补了 uncompressed RLE 测试。全套 30 passed。
- **4.3 预训练配置** → 已创建 `configs/pretrain-oamtcd.yaml`，并经过实测调整为 **batch=2、workers=4**（commit `6c36ff1`，见下）。
- **4.4 文档同步** → 已提交（commit `d8e675d`）：plan 保留 prepare_oamtcd 否决 geotree；2026-08-05 spec 整体否决 geotree；2026-08-04 spec 标记被取代；getting-started 补预训练流程。
- **4.5 步骤 5-8** → 已执行：coco8-seg smoke ✅、OAM-TCD 小子集转换 ✅、OAM-TCD 1 epoch smoke ✅、batch 实测 ✅。

### 实测发现的两个真实数据 bug（已修复）
1. **uncompressed RLE**（commit `4318911`）：OAM-TCD 的 RLE 标注 `counts` 是 run-length **list**（非压缩字符串），`coco_mask.decode` 无法处理，需先过 `frPyObjects`。
2. **project 路径冗余**（commit `771f9ff`）：ultralytics 会给 `project` 加 `runs/<task>/` 前缀，默认 `"runs"` 导致 `runs/segment/runs/...` 冗余路径。改默认 `project: ""`。

### 完整数据转换（已完成，未提交——数据不进 git）
- `data/oamtcd/`：4608 图 / 266643 实例 / 4608 标签，train 3492 / val 677 / test 439，split 无 oam_id 交叉，仅 20 个 dropped_invalid。
- `data/oamtcd/data.yaml`（单类 tree-crown）+ `manifest.json` 已生成。

### batch/workers 实测结论（重要，换平台后参考）
- **batch=4 在完整数据下验证阶段 OOM**（验证复用训练 batch 且优化器状态未释放）；**batch=2 稳定跑完 train+val**（EXIT 0，每 epoch ≈ 0.32 h）。
- **AMP 已确认开启**（"AMP: checks passed"），不是瓶颈。
- 原瓶颈是**系统内存**：7.7G 时 DataLoader prefetch 导致 swap 颠簸、GPU 饿死（利用率 0-4%）。WSL 内存加到 12G + `workers=4` 后 GPU 利用率稳定 90-100%。

## 4.5 剩余工作（换更高性能 GPU 平台后执行）

**当前这台 RTX 2070 Super 8GB + 11GB 系统内存跑 1280² 的 50-epoch 预训练太慢**（每 epoch ≈0.32h，50 epoch ≈16h，且显存余量极小），决定换更高性能 GPU 平台。

```text
1. 在新平台 uv sync --extra dev --extra data
2. 转换：python prepare_oamtcd.py --output data/oamtcd   （或复用已转换的 data/oamtcd）
3. 完整预训练：python train.py --config configs/pretrain-oamtcd.yaml  （batch=2, workers=4）
   - 新平台显存更大可尝试 batch=4/8（先跑 1 epoch 验证 val 不 OOM 再升）
4. 产物：runs/segment/oamtcd-pretrain/weights/best.pt（仅作初始化权重，绝不 publish）
```

## 5. 硬件 / 环境约束

- GPU：RTX 2070 Super，8 GiB（**本机限制**，换平台后解除）。
- 系统内存：WSL 已加到 12 GiB（原 7.7 GiB 是主要瓶颈）。
- 磁盘：约 868 GiB 可用（充足）。OAM-TCD 转换后 `data/oamtcd/` 约 3.4G 原始 + 3.0G 图像。
- 训练用 CUDA 版 torch（2.13.0+cu130）。
- 换平台后：`workers` 可按 CPU 核数上调（本机 16 核用 4）；`batch` 按目标 GPU 显存重测。

## 6. 关键代码位置

- `train.py`：训练/验证入口（已加固）。
- `prepare_oamtcd.py`：OAM-TCD 转换（**有 split bug 待修**）。
- `pyproject.toml`：依赖 + `data` 可选组。
- `pyrightconfig.json`：pyright → `.venv`。
- `tests/test_train.py`：train.py 测试（9 passed）。
- `tests/test_publish_checksums.py`：既有测试（publish 校验）。
- 根 `data.yaml`：最终 4 类 ABI（勿动）。
- `.agents/docs/specs/`、`.agents/docs/plans/`：spec 与计划。

## 7. 常用命令

```bash
# 跑测试
.venv/bin/python -m pytest -q

# 类型检查
pyright train.py prepare_oamtcd.py

# lint / format
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# 训练 smoke（coco8-seg）
.venv/bin/python train.py --config configs/default.yaml --data coco8-seg.yaml --imgsz 512 --epochs 1 --batch 2

# OAM-TCD 完整转换（复用已下载的 raw 用 --skip-download）
.venv/bin/python prepare_oamtcd.py --output data/oamtcd --skip-download

# OAM-TCD 完整预训练（batch=2, workers=4, 已调优）
.venv/bin/python train.py --config configs/pretrain-oamtcd.yaml
```

## 8. 提交建议（English Conventional Commits）

已按此拆分提交到 `feat/oamtcd-pretraining` 分支（见 §2 提交列表）。
```
fix: harden training config forwarding and metrics
feat: add OAM-TCD single-class pretraining data conversion
test: cover OAM-TCD conversion and split integrity
feat: add OAM-TCD pretraining config
docs: document OAM-TCD pretraining workflow and reject geotree
```

## 9. 风险提示

- ✅ **split bug 已修复**（commit `2bd121a`），完整转换已验证 train/val/test 无 oam_id 交叉。
- ✅ **uncompressed RLE 已修复**（commit `4318911`），完整转换 4608 图仅 20 个 dropped_invalid、0 decode_error。
- ✅ **`image` 列格式已验证**：为 `Struct{bytes, path}`，`_extract_image_bytes` 兼容；测试覆盖。
- ⚠️ **batch=4 在完整数据验证阶段 OOM**（batch=2 稳定）。换更大显存平台后再试 batch 4/8，需先跑 1 epoch 确认验证不 OOM。
- ⚠️ **AMP 已开启**，非瓶颈；若换平台后想进一步提速可对比 `amp: false`，但一般不推荐。
- ⚠️ **根 `data.yaml` 与 OAM 单类 `data.yaml` 是两份不同文件**，切勿混淆。
- ⚠️ **OAM-TCD 预训练产物绝不能 publish**（仅作 4 类模型的初始化权重）。
- ⚠️ **`data/oamtcd/` 不进 git**，换平台后需重新转换或迁移原始 parquet。