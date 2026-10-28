"""Session 序列化/反序列化，用于持久化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from .session import Session

# 消息角色 -> 消息类，用于反序列化时还原正确类型。
_ROLE_TO_CLASS: dict[str, type[Message]] = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool": ToolMessage,
}


def serialize_session(session: Session) -> dict[str, Any]:
    """把会话转为可 JSON 序列化的字典。"""
    return {
        "id": session.id,
        "metadata": session.metadata,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
                "metadata": m.metadata,
            }
            for m in session.messages
        ],
        "log": session.log.entries(),
    }


def deserialize_session(data: dict[str, Any]) -> Session:
    """从字典还原会话（消息历史 + 事件日志）。"""
    session = Session(session_id=data.get("id"), metadata=data.get("metadata") or {})

    for msg_data in data.get("messages", []):
        role = msg_data.get("role", "user")
        cls = _ROLE_TO_CLASS.get(role, Message)
        message = cls(
            msg_data.get("content", ""),
            tool_calls=msg_data.get("tool_calls") or [],
            tool_call_id=msg_data.get("tool_call_id"),
            name=msg_data.get("name"),
            metadata=msg_data.get("metadata") or {},
        )
        session.messages.append(message)

    # 还原事件日志（重新 append，保留原始时间戳）。
    for entry in data.get("log", []):
        session.log._entries.append(entry)

    return session


def save_session(session: Session, path: str | Path) -> None:
    """把会话保存为 JSON 文件。"""
    path = Path(path)
    path.write_text(
        json.dumps(serialize_session(session), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_session(path: str | Path) -> Session:
    """从 JSON 文件加载会话。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return deserialize_session(data)
