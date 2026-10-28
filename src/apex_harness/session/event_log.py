"""Append-only 事件日志，用于可追溯性。"""

from __future__ import annotations

import time
from typing import Any, Iterator

from .message import Message


class EventLog:
    """只追加（append-only）的事件日志。

    日志按时间顺序记录会话中发生的一切（消息、工具调用、错误等）。
    条目一旦写入就不能修改或删除，保证历史可追溯、可回放。
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def append(self, event_type: str, **data: Any) -> dict[str, Any]:
        """追加一条日志条目。

        ``event_type`` 是事件类型（如 ``"message"``、``"tool_call"``），
        ``data`` 是随事件携带的内容。自动记录写入时间戳，返回该条目。
        """
        entry: dict[str, Any] = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        self._entries.append(entry)
        return entry

    def record_message(self, message: Message) -> dict[str, Any]:
        """便捷方法：记录一条消息。"""
        return self.append("message", message=message.to_dict())

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._entries[index]

    def entries(self) -> list[dict[str, Any]]:
        """返回所有条目（浅拷贝，避免外部误改内部状态）。"""
        return list(self._entries)

    def clear(self) -> None:
        """清空日志。"""
        self._entries.clear()
