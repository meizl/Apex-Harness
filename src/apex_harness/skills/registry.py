"""技能注册表：保存技能元数据，供工具适配器查询 instructions。

技能以 :class:`SkillMetadata` 形式注册（启动时从 SKILL.md 解析）。
:class:`SkillToolAdapter` 在 LLM 调用技能时，从注册表取出
``instructions`` 正文返回给 LLM。
"""

from __future__ import annotations

from typing import Any

from apex_harness.core.plugin import Plugin
from apex_harness.tools.registry import ToolRegistry

from .adapter import SkillToolAdapter
from .base import SkillMetadata


class SkillRegistry:
    """技能注册表，按名字索引所有技能元数据。"""

    def __init__(self) -> None:
        self._metadatas: dict[str, SkillMetadata] = {}

    def register(self, metadata: SkillMetadata) -> None:
        """注册一个技能的元数据。"""
        if not metadata.name:
            raise ValueError("SkillMetadata 缺少 name")
        self._metadatas[metadata.name] = metadata

    def get(self, name: str) -> SkillMetadata | None:
        """按名字获取元数据，不存在返回 None。"""
        return self._metadatas.get(name)

    def require(self, name: str) -> SkillMetadata:
        """按名字获取元数据，不存在抛 KeyError。"""
        meta = self._metadatas.get(name)
        if meta is None:
            raise KeyError(f"技能 '{name}' 未注册")
        return meta

    @property
    def all(self) -> dict[str, SkillMetadata]:
        """返回所有已注册元数据的浅拷贝。"""
        return dict(self._metadatas)

    @property
    def names(self) -> list[str]:
        """返回所有已注册技能的名字列表。"""
        return list(self._metadatas.keys())


class SkillRegistryPlugin(Plugin):
    """技能注册表插件：提供 ``skills`` 服务。

    接收初始元数据列表并在 ``setup`` 时注册。``start`` 阶段把
    ``expose_as_tool=True`` 的技能通过 :class:`SkillToolAdapter`
    注册到 ``tools`` 服务，使 LLM 可通过 function calling 调用。
    """

    name = "skill_registry"
    description = "技能注册表，保存 SKILL.md 解析后的元数据"
    provides = ["skills"]
    requires = ["tools"]

    def __init__(self, skills_metadata: list[SkillMetadata] | None = None) -> None:
        super().__init__()
        self._initial_metadata = skills_metadata or []

    async def setup(self) -> None:
        self.registry = SkillRegistry()
        for meta in self._initial_metadata:
            self.registry.register(meta)
        self.ctx.register_service("skills", self.registry)

    async def start(self) -> None:
        # 把暴露为工具的技能注册到 ToolRegistry。
        tool_registry: ToolRegistry = self.ctx.require_service("tools")
        for metadata in self.registry.all.values():
            if metadata.expose_as_tool:
                tool_registry.register(SkillToolAdapter(metadata))
