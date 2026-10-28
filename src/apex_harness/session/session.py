"""Session 对象，管理一次 Agent 运行的生命周期。"""

from __future__ import annotations

import uuid
from typing import Any

from .event_log import EventLog
from .message import Message


class Session:
    """一次 Agent 运行对应的会话。

    一个会话维护一段对话的消息历史，并把所有发生的事写入事件日志，
    用于追溯和回放。
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        #: 会话唯一 ID，未指定时自动生成 UUID。
        self.id = session_id or uuid.uuid4().hex
        #: 会话元信息（标题、创建时间等）。
        self.metadata = metadata or {}
        #: 消息历史。
        self.messages: list[Message] = []
        #: 事件日志（append-only）。
        self.log = EventLog()

    # -- 消息 --------------------------------------------------------------

    def add_message(self, message: Message) -> Message:
        """追加一条消息到历史，同时写入事件日志。"""
        self.messages.append(message)
        self.log.record_message(message)
        return message

    def add_user(self, content: str, **kwargs: Any) -> Message:
        """便捷方法：追加一条用户消息。"""
        from .message import UserMessage

        return self.add_message(UserMessage(content, **kwargs))

    def add_assistant(self, content: str, **kwargs: Any) -> Message:
        """便捷方法：追加一条助手消息。"""
        from .message import AssistantMessage

        return self.add_message(AssistantMessage(content, **kwargs))

    def add_system(self, content: str, **kwargs: Any) -> Message:
        """便捷方法：追加一条系统消息。"""
        from .message import SystemMessage

        return self.add_message(SystemMessage(content, **kwargs))

    def add_tool(self, content: str, tool_call_id: str, **kwargs: Any) -> Message:
        """便捷方法：追加一条工具结果消息。"""
        from .message import ToolMessage

        return self.add_message(ToolMessage(content, tool_call_id=tool_call_id, **kwargs))

    # -- 查询 --------------------------------------------------------------

    def history(self) -> list[Message]:
        """返回消息历史（浅拷贝）。"""
        return list(self.messages)

    def to_llm_format(self) -> list[dict[str, Any]]:
        """把历史消息转成 LLM 可接受的字典列表。"""
        return [m.to_dict() for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Session id={self.id!r} messages={len(self.messages)}>"
