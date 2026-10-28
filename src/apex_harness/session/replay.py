"""基于事件日志的会话回放、分叉与搜索。"""

from __future__ import annotations

from typing import Any

from .message import Message
from .session import Session


def replay_messages(session: Session) -> list[Message]:
    """从事件日志重建消息序列。

    只提取日志中 ``type == "message"`` 的条目，按写入顺序还原为
    :class:`Message` 列表。
    """
    messages: list[Message] = []
    for entry in session.log:
        if entry.get("type") != "message":
            continue
        data = entry.get("data", {})
        msg_data = data.get("message", {})
        message = Message(
            msg_data.get("content", ""),
            tool_calls=msg_data.get("tool_calls") or [],
            tool_call_id=msg_data.get("tool_call_id"),
            name=msg_data.get("name"),
            metadata=msg_data.get("metadata") or {},
        )
        # 尽量还原角色。
        role = msg_data.get("role")
        if role:
            from .message import (
                AssistantMessage,
                SystemMessage,
                ToolMessage,
                UserMessage,
            )

            role_map = {
                "system": SystemMessage,
                "user": UserMessage,
                "assistant": AssistantMessage,
                "tool": ToolMessage,
            }
            cls = role_map.get(role, Message)
            message = cls(
                msg_data.get("content", ""),
                tool_calls=msg_data.get("tool_calls") or [],
                tool_call_id=msg_data.get("tool_call_id"),
                name=msg_data.get("name"),
                metadata=msg_data.get("metadata") or {},
            )
        messages.append(message)
    return messages


def fork_session(session: Session, session_id: str | None = None) -> Session:
    """从现有会话分叉出一个新会话。

    新会话复制当前的消息历史，事件日志只记录一条 ``"fork"`` 标记，
    之后独立发展。
    """
    fork = Session(session_id=session_id, metadata=dict(session.metadata))
    fork.messages = list(session.messages)
    fork.log.append("fork", parent=session.id)
    return fork


def search_log(session: Session, event_type: str | None = None) -> list[dict[str, Any]]:
    """在事件日志中搜索。

    指定 ``event_type`` 则只返回该类型条目，否则返回全部。
    """
    if event_type is None:
        return session.log.entries()
    return [e for e in session.log if e.get("type") == event_type]
