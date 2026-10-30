"""LLM 提供者接口。"""

from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar

from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.core.plugin import Plugin


class LLMProvider(Plugin):
    """LLM 提供者插件基类。

    具体实现（如 OpenAI、本地模型）继承本类，实现 :meth:`chat`（和可选的
    :meth:`embed`），并在 ``setup`` 中注册服务（通常注册成 ``"llm"``）。
    Agent 等使用方通过 ``ctx.require_service("llm")`` 拿到它。
    """

    #: 模型标识，如 ``"gpt-4"``。由具体实现覆盖。
    model: ClassVar[str] = ""

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """对话补全：传入 LangChain 消息列表，返回 :class:`AIMessage`。

        ``messages`` 是 LangChain :class:`BaseMessage` 列表::

            [
                SystemMessage(content="你是一个助手"),
                HumanMessage(content="你好"),
            ]

        ``tools`` 是可选的工具定义列表（JSON Schema 格式），传入后模型可以
        选择发起工具调用，返回的 :class:`AIMessage` 的 ``tool_calls`` 字段
        会携带这些调用。

        ``kwargs`` 用于透传温度、最大 token 等生成参数。
        """
        raise NotImplementedError

    async def stream(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式对话：逐个产出文本片段，适合实时展示。

        参数与 :meth:`chat` 一致，但返回一个异步迭代器，每次 ``yield``
        一段增量文本。默认实现回退为一次性 ``chat`` 后整体产出。
        """
        reply = await self.chat(messages, tools=tools, **kwargs)
        if reply.content:
            yield reply.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入：传入文本列表，返回等长的向量列表。"""
        raise NotImplementedError
