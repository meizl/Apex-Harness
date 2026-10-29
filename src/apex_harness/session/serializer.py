"""Session 序列化/反序列化，用于持久化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

from .session import Session

# LangChain 消息 type -> 消息类，用于反序列化兜底。
_TYPE_TO_CLASS: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
}


def serialize_session(session: Session) -> dict[str, Any]:
    """把会话转为可 JSON 序列化的字典。"""
    return {
        "id": session.id,
        "metadata": session.metadata,
        "messages": [message_to_dict(m) for m in session.messages],
        "log": session.log.entries(),
    }


def deserialize_session(data: dict[str, Any]) -> Session:
    """从字典还原会话（消息历史 + 事件日志）。"""
    session = Session(session_id=data.get("id"), metadata=data.get("metadata") or {})

    raw_messages = data.get("messages", [])
    if raw_messages:
        # 优先用 LangChain 自带的还原（兼容 message_to_dict 格式）。
        try:
            session.messages = list(messages_from_dict(raw_messages))
        except Exception:  # noqa: BLE001 - 兜底：手动按 type 还原
            for msg_data in raw_messages:
                msg_type = msg_data.get("type", "human")
                cls = _TYPE_TO_CLASS.get(msg_type, HumanMessage)
                session.messages.append(cls(**msg_data))

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
