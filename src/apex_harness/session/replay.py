"""基于事件日志的会话回放、分叉与搜索。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
)

from .session import Session

# 事件日志里存的是 message.model_dump() 结果，type 字段标识角色。
_TYPE_TO_CLASS: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
}


def replay_messages(session: Session) -> list[BaseMessage]:
    """从事件日志重建消息序列。

    只提取日志中 ``type == "message"`` 的条目，按写入顺序还原为
    LangChain :class:`BaseMessage` 列表。
    """
    messages: list[BaseMessage] = []
    for entry in session.log:
        if entry.get("type") != "message":
            continue
        msg_dump = entry.get("data", {}).get("message", {})
        msg_type = msg_dump.get("type", "human")
        cls = _TYPE_TO_CLASS.get(msg_type, HumanMessage)
        # 过滤掉 None 值，避免给不需要的字段传 None。
        kwargs = {k: v for k, v in msg_dump.items() if v is not None}
        messages.append(cls(**kwargs))
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
