# 开发工作流

本文介绍给本仓库做贡献时的工作流约定：分支、提交、代码风格、文档分层。涉及 AI 代理的完整指令
见根目录 `AGENTS.md`（英文）。

## 分支策略

- 采用轻量级主干开发（trunk-based）：`main` 必须始终可用。
- 常规实现工作：在 `main` 上开短生命周期分支，一个目标一个分支。
- 分支前缀：`feat/`、`fix/`、`docs/`、`chore/`、`refactor/`、`test/`，后跟 kebab-case 主题，
  例如 `feat/train-yolo11-seg`。
- 不创建长期分支（`develop`、`integration`、`release` 等）。
- 集成前先把开发分支 rebase 到最新 `main`，不产生 merge commit。
- 合入 `main` 前需要明确授权；不要直接向 `main` 提交。

## 提交规范

- 提交信息使用英文，采用 Conventional Commits 前缀：`chore:`、`feat:`、`fix:`、`docs:`、
  `refactor:`、`test:`。
- Rebase 优先，保持线性历史。

## 代码风格

- Python 使用 `ruff` 格式化与静态检查（配置在 `pyproject.toml`），不沿用机载仓库的 C++ `.clang-format`。
- 代码注释、提交信息与 agent 文档一律英文；`docs/` 下的人类文档使用中文。
- 源代码中禁止中文注释与非英文标识符。

## 文档分层

- `docs/`：人类文档（中文），面向维护者与接手者。
- `.agents/docs/`：agent 工作产物（英文），含设计 spec（`specs/`）与实施计划（`plans/`）。
- 根目录 `AGENTS.md` / `CLAUDE.md`：agent 指令。

## 凭据管理

- 密钥存 git 忽略的 `.local/credentials.env`（权限 600），不入库。
- `HF_TOKEN` 从该文件读取；缺失时发布命令报错退出，不以占位符继续。

## 代码评审预期

- 每个可评审目标对应一个短生命周期分支；完成后提出评审。
- 评审关注：行为正确性、接口边界、`data.yaml` 与发布 `model.yaml` 的一致性、对 `docs/`（人类）
  与 `.agents/docs/`（agent）的同步更新。
- 合入 `main` 前确保：`ruff check` 通过、脚本可运行、文档与实现一致。