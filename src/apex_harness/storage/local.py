"""本地 JSONL 事件存储。

每个 task 对应一个 ``<task_id>.jsonl`` 文件，事件逐行追加。append 采用
write-ahead：先落盘、再更新内存索引，保证断线重连 / 进程重启后可按
``event_id`` 回放。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .base import EventStore


class LocalEventStore(EventStore):
    """基于 JSONL 文件的本地事件存储。"""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        # 内存索引：task_id -> 已加载的事件列表（与文件保持一致）。
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _file(self, task_id: str) -> Path:
        return self._dir / f"{task_id}.jsonl"

    def _load(self, task_id: str) -> list[dict[str, Any]]:
        """惰性加载某个 task 的事件到内存索引。"""
        if task_id in self._events:
            return self._events[task_id]

        events: list[dict[str, Any]] = []
        path = self._file(task_id)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._events[task_id] = events
        return events

    def append(self, task_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """write-ahead 追加：先落盘，再更新内存索引。"""
        with self._lock:
            events = self._load(task_id)
            event_id = (events[-1]["id"] + 1) if events else 1
            entry: dict[str, Any] = {"id": event_id, **event}
            # 先持久化
            with self._file(task_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # 再更新内存索引
            events.append(entry)
            return entry

    def read_since(self, task_id: str, after_id: int) -> list[dict[str, Any]]:
        with self._lock:
            events = self._load(task_id)
        return [e for e in events if e["id"] > after_id]

    def latest_id(self, task_id: str) -> int:
        with self._lock:
            events = self._load(task_id)
        return events[-1]["id"] if events else 0
