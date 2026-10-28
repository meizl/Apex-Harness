"""工具接口。"""

from __future__ import annotations

from typing import Any, ClassVar

from apex_harness.core.plugin import Plugin


class Tool(Plugin):
    """工具插件基类。

    工具是 Agent 可以调用的单个操作（如读文件、执行命令）。具体工具继承
    本类，实现 :meth:`run`，并填写 ``parameters`` 参数 schema，供 Agent
    理解入参并正确调用。
    """

    #: 参数 schema（JSON Schema 格式），描述 :meth:`run` 的入参。
    parameters: ClassVar[dict[str, Any]] = {}

    async def run(self, **kwargs: Any) -> Any:
        """执行工具，入参由 ``parameters`` schema 定义，返回执行结果。"""
        raise NotImplementedError
