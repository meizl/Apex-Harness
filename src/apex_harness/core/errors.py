"""统一的异常类型与重试工具。

框架内所有可预期的错误都继承 :class:`HarnessError`，方便上层统一捕获、
分类并给出友好提示，而不是散落各种 ``RuntimeError`` / ``KeyError``。

同时提供 :func:`retry`（指数退避重试）和 :func:`raise_for_llm_status`
（把 HTTP 状态码映射成具体异常），用于 LLM 等外部服务的容错调用。
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


class HarnessError(Exception):
    """框架所有异常的基类。捕获本类可兜住所有框架内可预期错误。"""


class ConfigurationError(HarnessError):
    """配置缺失或非法。"""


class PluginError(HarnessError):
    """插件生命周期相关错误（挂载、启动、卸载等）。"""


# -- LLM 相关异常 ---------------------------------------------------------


class LLMError(HarnessError):
    """LLM 调用失败的基类。"""


class LLMAuthenticationError(LLMError):
    """API Key 无效或无权限（401/403）。"""


class LLMRateLimitError(LLMError):
    """触发限流（429）。"""


class LLMServiceError(LLMError):
    """LLM 服务端错误（5xx）。"""


class LLMTimeoutError(LLMError):
    """LLM 请求超时。"""


# -- 工具相关异常 ---------------------------------------------------------


class ToolError(HarnessError):
    """工具执行失败。"""


# 默认可重试的异常：限流、服务端错误、超时。认证错误不重试（重试也没用）。
_RETRYABLE: tuple[type[Exception], ...] = (
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)


def raise_for_llm_status(status_code: int, detail: str = "") -> None:
    """把 HTTP 状态码映射成对应的 LLM 异常。

    2xx 直接返回；其余抛出带上下文的 :class:`LLMError` 子类。
    """
    if 200 <= status_code < 300:
        return
    suffix = f"：{detail}" if detail else ""
    if status_code in (401, 403):
        raise LLMAuthenticationError(f"LLM 认证失败（{status_code}）{suffix}")
    if status_code == 429:
        raise LLMRateLimitError(f"LLM 触发限流（429）{suffix}")
    if status_code >= 500:
        raise LLMServiceError(f"LLM 服务端错误（{status_code}）{suffix}")
    raise LLMError(f"LLM 请求失败（{status_code}）{suffix}")


def retry(
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: tuple[type[Exception], ...] = _RETRYABLE,
) -> Callable[[F], F]:
    """异步重试装饰器，指数退避。

    仅对 ``retry_on`` 中的异常重试；其他异常直接抛出。退避间隔为
    ``base_delay * 2 ** attempt``，封顶到 ``max_delay``。
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:
                    if attempt >= retries:
                        raise
                    logger.warning(
                        "调用 %s 失败（第 %d/%d 次）：%s，%.1fs 后重试",
                        getattr(func, "__qualname__", func.__name__),
                        attempt + 1,
                        retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
            raise RuntimeError("unreachable")  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator
