"""Skill → Tool 适配器。

把一个 :class:`SkillMetadata` 包装成 :class:`~apex_harness.tools.base.Tool`，
注册到 ``ToolRegistry`` 后，LLM 就能像调用普通工具一样通过 function
calling 调用该技能。

适配器不执行任何代码：``run`` 被调用时直接返回技能的 ``instructions``
正文作为工具结果，注入 LLM 上下文，引导 LLM 按指令完成任务。
"""

from __future__ import annotations

from typing import Any

from apex_harness.tools.base import Tool

from .base import SkillMetadata


class SkillToolAdapter(Tool):
    """把 SkillMetadata 包装成 Tool，run 返回 instructions 正文。"""

    def __init__(self, metadata: SkillMetadata) -> None:
        self._metadata = metadata
        self.name = metadata.name
        self.description = metadata.description
        # 技能不需要入参（LLM 直接调用即可激活）
        self.parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> str:
        """返回技能的完整指令正文。"""
        return self._metadata.instructions

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SkillToolAdapter {self.name}>"
