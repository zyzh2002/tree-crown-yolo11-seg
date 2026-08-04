# 项目文档

本文档库面向**人类**维护者与接手者，使用中文编写。AI 代理的工作指南与内部产物位于
`AGENTS.md` 与 `.agents/docs/`（英文），不在此列。

## 文档清单

| 文档 | 内容 | 何时阅读 |
|---|---|---|
| [getting-started.md](getting-started.md) | 环境准备、训练、验证、导出、发布、排障 | 新人首次上手 / 搭建环境 |
| [development.md](development.md) | 分支策略、提交规范、代码风格、文档分层 | 开始贡献代码前 |
| [publishing.md](publishing.md) | 发布流程、版本契约、交接给机载仓库的信息 | 发布 / 更新模型前 |

## 阅读路径

**新人 / 接手者：** `getting-started.md` → `development.md` → `publishing.md`

**日常开发：** `development.md` + 对应功能模块

**详细技术决策与完整验证数据：** 见 `.agents/docs/`（AI 工作产物，英文，包含设计 spec 与实施计划）。

## 仓库根文档

- [README.md](../README.md) — 仓库快速入口（中文）
- [AGENTS.md](../AGENTS.md) — AI 代理工作指南（英文，人类无需阅读）