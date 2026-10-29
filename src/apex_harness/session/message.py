"""消息类型定义（system / human / ai / tool）。

直接复用 LangChain 的消息类，保持与 LangChain 生态兼容。
为了项目内命名习惯，把 HumanMessage 别名为 UserMessage，AIMessage 别名为
AssistantMessage。
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# 项目内习惯命名 -> LangChain 标准类。
UserMessage = HumanMessage
AssistantMessage = AIMessage

__all__ = [
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "HumanMessage",
    "AssistantMessage",
    "AIMessage",
    "ToolMessage",
]
