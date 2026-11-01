"""LLM 提供者接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.core.plugin import Plugin


@dataclass
class StreamChunk:
    """流式输出的一个增量片段。

    与 OpenAI 的 ``delta`` 对应：``content`` 是增量文本，``tool_calls``
    是增量工具调用（可能只包含 ``index``、``id``、``function.name``、
    ``function.arguments`` 的一部分，需要调用方按 ``index`` 合并）。
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = field(default=None)


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
    ) -> AsyncIterator[StreamChunk]:
        """流式对话：逐个产出 :class:`StreamChunk`。

        参数与 :meth:`chat` 一致，但返回异步迭代器。每个 ``StreamChunk``
        携带增量 ``content`` 和（可选的）增量 ``tool_calls``。

        默认实现回退为一次性 ``chat`` 后整体产出，方便未实现流式的子类。
        """
        reply = await self.chat(messages, tools=tools, **kwargs)
        tool_calls = None
        if reply.tool_calls:
            tool_calls = [
                {
                    "index": i,
                    "id": tc.get("id", ""),
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                    },
                }
                for i, tc in enumerate(reply.tool_calls)
            ]
        yield StreamChunk(content=reply.content or "", tool_calls=tool_calls)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入：传入文本列表，返回等长的向量列表。"""
        raise NotImplementedError
