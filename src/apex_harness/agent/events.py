"""Agent 流式事件模型。

把流式执行过程结构化地表达为一系列事件，每个事件携带 ``generation_id``
标识生成轮次。上层（服务层 / 存储层）据此做事件持久化、断线补发与回放。

事件类型：
- ``generation_start``  一轮模型生成开始
- ``text``              文本增量
- ``tool_call``         模型发起工具调用
- ``tool_result``       工具执行结果
- ``generation_end``    一轮模型生成结束
- ``generation_failed`` 一轮模型生成失败（本轮未完成，前端应回退该轮文本）
- ``done``              整个任务完成（终态）
- ``error``             出错（终态）
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_generation_id() -> str:
    """生成一个新的 generation_id，标识一次模型生成。"""
    return uuid.uuid4().hex


@dataclass
class StreamEvent:
    """流式执行中的一个结构化事件。"""

    type: str
    #: 所属生成轮次；终态事件（done / error）可能为空。
    generation_id: str = ""
    #: 文本事件携带的增量文本。
    text: str = ""
    #: 工具调用 / 结果等结构化数据。
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典，用于持久化与推送。"""
        return {
            "type": self.type,
            "generation_id": self.generation_id,
            "text": self.text,
            "data": self.data,
        }
