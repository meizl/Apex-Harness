"""LLM 提供者接口。"""

from __future__ import annotations

from typing import Any, ClassVar

from apex_harness.core.plugin import Plugin


class LLMProvider(Plugin):
    """LLM 提供者插件基类。

    具体实现（如 OpenAI、本地模型）继承本类，实现 :meth:`chat`（和可选的
    :meth:`embed`），并在 ``setup`` 中注册服务（通常注册成 ``"llm"``）。
    Agent 等使用方通过 ``ctx.require_service("llm")`` 拿到它。
    """

    #: 模型标识，如 ``"gpt-4"``。由具体实现覆盖。
    model: ClassVar[str] = ""

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """对话补全：传入消息列表，返回助手的文本回复。

        ``messages`` 格式为角色-内容字典列表::

            [
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "你好"},
            ]

        ``kwargs`` 用于透传温度、最大 token 等生成参数。
        """
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入：传入文本列表，返回等长的向量列表。"""
        raise NotImplementedError
