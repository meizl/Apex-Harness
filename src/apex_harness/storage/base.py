"""事件存储抽象：持久化 + 按 event_id 回放。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EventStore(ABC):
    """按 task 存储流式事件，支持 write-ahead 追加与按 id 回放。

    实现方需保证：``append`` 先持久化、再返回，并且分配的 ``event_id``
    在该 task 内单调递增（从 1 开始）。
    """

    @abstractmethod
    def append(self, task_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """追加一个事件（write-ahead），分配并返回带 ``id`` 的事件。"""

    @abstractmethod
    def read_since(self, task_id: str, after_id: int) -> list[dict[str, Any]]:
        """返回 ``event_id > after_id`` 的事件，按 id 升序。"""

    @abstractmethod
    def latest_id(self, task_id: str) -> int:
        """返回该 task 当前最大 event_id；无事件时返回 0。"""
