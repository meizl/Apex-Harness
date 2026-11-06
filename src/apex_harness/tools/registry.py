"""工具注册表：收集所有已挂载的工具插件，供 Agent 查询和调用。

使用方式::

    kernel = Kernel()
    await kernel.mount_all([
        ToolRegistryPlugin(),   # 必须先挂载
        ReadFileTool(),         # 工具在 setup 时自动注册到 registry
        WriteFileTool(),
    ])
    await kernel.start()

    registry = kernel.services.require("tools")
    tools = registry.all  # dict[str, Tool]
"""

from __future__ import annotations

from typing import Any

from apex_harness.core.plugin import Plugin

from .base import Tool


class ToolRegistry:
    """工具注册表，按名字索引所有已注册的工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具，用 ``tool.name`` 作为 key。"""
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} 缺少 name 属性")
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        """按名字移除工具，不存在则静默忽略。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """按名字获取工具，不存在返回 None。"""
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        """按名字获取工具，不存在抛 KeyError。"""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"工具 '{name}' 未注册")
        return tool

    @property
    def all(self) -> dict[str, Tool]:
        """返回所有工具的浅拷贝。"""
        return dict(self._tools)

    @property
    def names(self) -> list[str]:
        """返回所有已注册工具的名字列表。"""
        return list(self._tools.keys())


class ToolRegistryPlugin(Plugin):
    """工具注册表插件：提供 ``tools`` 服务。

    其他工具插件通过 ``requires = ["tools"]`` 依赖本插件，并在
    ``setup`` 中调用 ``registry.register(self)`` 完成注册。
    """

    name = "tool_registry"
    description = "工具注册表，收集所有已挂载的工具插件"
    provides = ["tools"]

    async def setup(self) -> None:
        self.registry = ToolRegistry()
        self.ctx.register_service("tools", self.registry)
