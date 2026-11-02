# Apex.md

本文件是 Apex Harness 项目的**全局指令文件**（类似 Claude Code 的 `CLAUDE.md`）。
服务启动时会自动读取，并作为系统提示注入到每次 LLM 对话中。

## 项目简介

Apex Harness 是一个插件式 AI Agent 框架，核心组件包括：

- **core**：Kernel / EventBus / ServiceRegistry / Context / Plugin（插件解耦基石）
- **llm**：LLMProvider 接口 + OpenAI/DeepSeek 兼容实现（支持流式）
- **tools**：文件系统 / Shell / Web 抓取工具
- **agent**：Turn / Loop / Planner / Orchestrator（ReAct + Plan-and-Execute）
- **session**：消息历史 + 事件日志
- **server**：FastAPI 流式聊天接口

## 行为准则

- 回答用中文
- 涉及代码时给出具体文件路径和行号
- 优先使用已有的工具（读文件、列目录、执行命令）来获取真实信息
- 不确定时承认不确定，不要编造

## 工作目录

当前工作目录下的文件可以通过 `read_file`、`list_dir` 等工具访问。
