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

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.core.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
    raise_for_llm_status,
    retry,
)
from apex_harness.llm.base import LLMProvider, StreamChunk

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
    # LangChain 的 content 可能是 str 或 list，统一转成 str
    content = message.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    result: dict[str, Any] = {"role": role, "content": content}

    # assistant 消息带 tool_calls 时，需要从 LangChain 格式转回 OpenAI 格式
    if message.type == "ai" and getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                },
            }
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
            raise LLMAuthenticationError(
                "未配置 LLM API Key，请设置环境变量 APEX_LLM_API_KEY 或 OPENAI_API_KEY"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_format(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        data = await self._post_json(payload)

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = _parse_tool_calls(choice.get("tool_calls"))
        return AIMessage(content=content, tool_calls=tool_calls)

    @retry(retries=3)
    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 JSON 请求并做错误映射 + 重试。

        限流（429）、服务端错误（5xx）、超时会指数退避重试；认证错误等
        不可重试的异常直接抛出。
        """
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM 请求超时") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 网络请求失败：{exc}") from exc

        if response.status_code >= 400:
            raise_for_llm_status(response.status_code, response.text[:200])
        return response.json()

    async def stream(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """流式对话：解析 SSE 响应，逐个产出 :class:`StreamChunk`。

        保留 delta 中的 ``tool_calls`` 字段，供调用方检测工具调用并拼装。

        重试策略：仅在尚未产出任何 chunk 时重试可重试异常
        （限流/服务端错误/超时）；一旦已经 yield 过内容，后续错误直接抛出，
        避免重复输出。
        """
        if not self.api_key:
            raise LLMAuthenticationError(
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

        # 流式重试：最多重试 3 次，指数退避；仅在未产出 chunk 时重试。
        max_retries = 3
        base_delay = 1.0
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            yielded_any = False
            try:
                async for chunk in self._stream_once(payload):
                    yielded_any = True
                    yield chunk
                return  # 正常结束
            except (LLMRateLimitError, LLMServiceError, LLMTimeoutError) as exc:
                last_exc = exc
                # 已经产出过内容 → 不能重试，直接抛出
                if yielded_any:
                    raise
                # 未产出内容且还有重试次数 → 退避后重试
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 30.0)
                    logger.warning(
                        "LLM 流式调用失败（第 %d/%d 次）：%s，%.1fs 后重试",
                        attempt + 1, max_retries + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # 重试次数用完
                raise
            except (LLMAuthenticationError, LLMError) as exc:
                # 认证错误、其他网络错误不重试
                raise

        if last_exc:  # pragma: no cover
            raise last_exc

    async def _stream_once(self, payload: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """单次流式请求：建立连接、解析 SSE、逐个 yield chunk。"""
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                # 非 2xx：读错误体后映射成统一异常（限流/认证/服务错误）。
                if response.status_code >= 400:
                    body = await response.aread()
                    raise_for_llm_status(
                        response.status_code, body.decode("utf-8", "ignore")[:200]
                    )
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
                    content = delta.get("content") or ""
                    tool_calls = delta.get("tool_calls")
                    if not content and not tool_calls:
                        continue
                    yield StreamChunk(content=content, tool_calls=tool_calls)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM 流式请求超时") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 网络请求失败：{exc}") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入，本插件不实现。"""
        raise NotImplementedError("OpenAIProvider 不支持 embed")
