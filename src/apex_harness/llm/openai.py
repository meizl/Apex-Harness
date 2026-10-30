"""OpenAI 兼容接口的 LLM 插件（DeepSeek、GPT 等）。

通过 ``httpx`` 直接调用 ``/chat/completions`` 接口，不依赖 openai SDK，
因此兼容所有 OpenAI 格式的服务（DeepSeek、OpenAI、本地 vLLM 等）。

配置来源（优先级从高到低）：
1. ``ctx.config``（Kernel 按插件名分发的配置切片）
2. 环境变量 ``APEX_LLM_API_KEY`` / ``APEX_LLM_BASE_URL`` / ``APEX_LLM_MODEL``

示例环境变量::

    APEX_LLM_API_KEY=sk-xxx
    APEX_LLM_BASE_URL=https://api.deepseek.com/v1
    APEX_LLM_MODEL=deepseek-chat
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def _to_openai_format(message: BaseMessage) -> dict[str, Any]:
    """把 LangChain 消息转成 OpenAI 的 ``{role, content}`` 格式。"""
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }
    role = role_map.get(message.type, "user")
    result: dict[str, Any] = {"role": role, "content": message.content}

    # assistant 消息可能带 tool_calls，需要透传；去掉 LangChain 内部字段
    if message.type == "ai" and getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {k: v for k, v in tc.items() if k != "type"}
            for tc in message.tool_calls
        ]

    # tool 消息需要 tool_call_id
    if message.type == "tool":
        result["tool_call_id"] = getattr(message, "tool_call_id", "")

    return result


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """把 OpenAI 返回的 tool_calls 转成 LangChain 期望的格式。

    OpenAI 返回::

        {"id": ..., "function": {"name": ..., "arguments": "{\"x\": 1}"}}

    LangChain ``AIMessage.tool_calls`` 期望::

        {"name": ..., "args": {...}, "id": ...}
    """
    result: list[dict[str, Any]] = []
    for tc in raw or []:
        fn = tc.get("function", {}) or {}
        args_str = fn.get("arguments", "{}") or "{}"
        try:
            args = json.loads(args_str)
        except (TypeError, json.JSONDecodeError):
            args = {}
        result.append({
            "name": fn.get("name", ""),
            "args": args,
            "id": tc.get("id", ""),
        })
    return result


class OpenAIProvider(LLMProvider):
    """OpenAI 兼容接口的 LLM 提供者插件。"""

    name = "openai_llm"
    description = "OpenAI 兼容接口的 LLM 提供者（DeepSeek/GPT/vLLM）"
    provides = ["llm"]

    # 默认模型标识，可通过配置或环境变量覆盖。
    model = "deepseek-chat"

    async def setup(self) -> None:
        """挂载时注册服务。此时不读配置、不建客户端。"""
        self.ctx.register_service("llm", self)

    async def start(self) -> None:
        """依赖就绪后读配置并建立 HTTP 客户端。"""
        cfg = self.ctx.config or {}

        self.api_key = (
            cfg.get("api_key")
            or os.environ.get("APEX_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.base_url = (
            cfg.get("base_url")
            or os.environ.get("APEX_LLM_BASE_URL", "https://api.deepseek.com/v1")
        ).rstrip("/")
        self.model = cfg.get("model") or os.environ.get("APEX_LLM_MODEL", self.model)

        if not self.api_key:
            logger.warning(
                "未配置 LLM API Key，调用 chat 时会失败。"
                "请设置环境变量 APEX_LLM_API_KEY 或 OPENAI_API_KEY。"
            )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    async def stop(self) -> None:
        """关停时关闭 HTTP 连接。"""
        await self._client.aclose()

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """LangChain 消息列表 → OpenAI dict → 调接口 → AIMessage。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM API Key，请设置环境变量 APEX_LLM_API_KEY 或 OPENAI_API_KEY"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_format(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = _parse_tool_calls(choice.get("tool_calls"))
        return AIMessage(content=content, tool_calls=tool_calls)

    async def stream(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式对话：解析 SSE 响应，逐个产出增量文本。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM API Key，请设置环境变量 APEX_LLM_API_KEY 或 OPENAI_API_KEY"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_format(m) for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {}) or {}
                content = delta.get("content")
                if content:
                    yield content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入，本插件不实现。"""
        raise NotImplementedError("OpenAIProvider 不支持 embed")
