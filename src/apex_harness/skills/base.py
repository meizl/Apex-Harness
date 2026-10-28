"""技能接口。"""

from __future__ import annotations

from typing import Any, ClassVar

from apex_harness.core.plugin import Plugin


class Skill(Plugin):
    """技能插件基类。

    技能是可复用的多步骤能力（相比单个工具更复杂，可能组合多个工具或
    内置一段流程）。具体技能继承本类，实现 :meth:`run`，并通过
    ``instructions`` 描述该技能做什么、如何使用。
    """

    #: 技能说明，写给调用方（Agent 或加载器）看。
    instructions: ClassVar[str] = ""

    async def run(self, **kwargs: Any) -> Any:
        """执行技能，返回执行结果。"""
        raise NotImplementedError
