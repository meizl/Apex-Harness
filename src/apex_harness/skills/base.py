"""技能元数据。

技能以 ``SKILL.md`` 文件定义（见 :mod:`apex_harness.skills.parser`），
解析后生成 :class:`SkillMetadata`。技能本身不包含可执行代码，其
``instructions`` 正文在被 LLM 调用时作为工具结果注入上下文，引导 LLM
按指令完成任务。

渐进式披露：
- 启动时只把 ``name`` + ``description`` 作为 function schema 暴露给 LLM；
- LLM 调用该技能时，才把完整 ``instructions`` 正文返回给 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillMetadata:
    """技能元数据（由 SKILL.md 解析而来）。"""

    name: str
    description: str = ""
    #: 技能的完整指令正文，调用时注入 LLM 上下文。
    instructions: str = ""
    #: 是否作为 function 暴露给 LLM。
    expose_as_tool: bool = True
    #: frontmatter 中的额外字段。
    extra: dict[str, Any] = field(default_factory=dict)
