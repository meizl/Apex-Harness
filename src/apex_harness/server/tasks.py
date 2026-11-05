"""后台任务管理：把生成与 HTTP 连接解耦，支持事件补发与断线重连。

核心思路：
- ``start`` 在后台运行生成器，把每个事件先持久化（write-ahead）再推送给
  订阅者；
- ``subscribe`` 返回一个异步迭代器，先补发 ``event_id > after_id`` 的历史
  事件，再持续 yield 实时事件，直到遇到 ``done`` / ``error`` 终态。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from apex_harness.agent.events import StreamEvent
from apex_harness.storage.base import EventStore

logger = logging.getLogger(__name__)


class TaskManager:
    """管理后台生成任务与其订阅者。"""

    def __init__(self, store: EventStore) -> None:
        self.store = store
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def _subs(self, task_id: str) -> set[asyncio.Queue]:
        return self._subscribers.setdefault(task_id, set())

    def append_event(self, task_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """先持久化事件，再推送给所有订阅者（write-ahead）。"""
        entry = self.store.append(task_id, event)
        for q in list(self._subs(task_id)):
            q.put_nowait(entry)
        return entry

    def start(self, task_id: str, gen: AsyncIterator[StreamEvent]) -> asyncio.Task:
        """后台运行事件生成器，生成过程与 HTTP 连接解耦。"""

        async def runner() -> None:
            try:
                async for ev in gen:
                    self.append_event(task_id, ev.to_dict())
            except asyncio.CancelledError:
                # 取消时显式关闭底层生成器：否则生成器会停在 yield 处泄漏，
                # 正在进行的 LLM 流式连接 / 工具子进程等资源不会被及时释放。
                with suppress(Exception):
                    await gen.aclose()
                raise
            except Exception as exc:  # noqa: BLE001 - 兜底转成 error 事件
                logger.exception("任务 %s 生成失败", task_id)
                self.append_event(
                    task_id,
                    StreamEvent(type="error", text=f"[生成出错] {exc}").to_dict(),
                )

        task = asyncio.create_task(runner())
        self._tasks[task_id] = task
        return task

    async def cancel(self, task_id: str) -> bool:
        """取消后台生成任务，并追加 ``cancelled`` 终态事件。

        已结束或不存在则返回 ``False``。``task.cancel()`` 会在生成器下一个
        ``await`` 挂起点注入 ``CancelledError``，从而打断模型流式输出。
        """
        task = self._tasks.get(task_id)
        if task is None or task.done():
            return False
        task.cancel()
        self.append_event(
            task_id,
            StreamEvent(type="cancelled", text="已停止生成").to_dict(),
        )
        return True

    def read_since(self, task_id: str, after_id: int) -> list[dict[str, Any]]:
        """补发历史事件。"""
        return self.store.read_since(task_id, after_id)

    def subscribe(self, task_id: str, after_id: int) -> AsyncIterator[dict[str, Any]]:
        """订阅某 task 的事件流：先补发，再实时推送。"""
        return self._stream(task_id, after_id)

    async def _stream(self, task_id: str, after_id: int) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue()
        subs = self._subs(task_id)
        # 先注册订阅，再补发历史，用 event_id 去重避免漏发/重发。
        subs.add(q)
        last = after_id
        try:
            for e in self.store.read_since(task_id, after_id):
                if e["id"] > last:
                    yield e
                    last = e["id"]

            while True:
                e = await q.get()
                if e["id"] <= last:
                    continue
                yield e
                last = e["id"]
                if e.get("type") in ("done", "error", "cancelled"):
                    break
        finally:
            subs.discard(q)
