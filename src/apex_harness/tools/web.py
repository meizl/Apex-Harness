"""Web 工具：抓取网页内容。

使用 httpx 发起 HTTP 请求，支持超时和输出截断。
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from .base import Tool


def _strip_html(html: str) -> str:
    """极简 HTML 标签去除，把页面转成纯文本。"""
    # 去掉 script / style 块
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # 去掉标签
    text = re.sub(r"<[^>]+>", "", html)
    # 压缩空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class WebFetchTool(Tool):
    """抓取网页内容并返回纯文本。"""

    name = "web_fetch"
    description = "抓取指定 URL 的网页内容，返回纯文本"
    requires = ["tools"]
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的网页 URL（需包含 http:// 或 https://）",
            },
            "max_chars": {
                "type": "integer",
                "description": "返回的最大字符数，默认 8000",
                "default": 8000,
            },
        },
        "required": ["url"],
    }

    async def setup(self) -> None:
        cfg = self.ctx.config or {}
        self._timeout = cfg.get("timeout", 15)
        self._max_chars = cfg.get("max_chars", 8000)
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, url: str, max_chars: int | None = None, **kwargs: Any) -> str:
        if not url.startswith(("http://", "https://")):
            return f"Error: URL 必须以 http:// 或 https:// 开头: '{url}'"

        max_chars = max_chars or self._max_chars

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; Apex-Harness/0.1; "
                "+https://github.com/meizl/Apex-Harness)"
            ),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException:
            return f"Error: 请求超时（{self._timeout}秒）: {url}"
        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code}: {url}"
        except httpx.RequestError as exc:
            return f"Error: 请求失败: {exc}"

        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type or "text/xml" in content_type:
            text = _strip_html(response.text)
        else:
            text = response.text

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [已截断，超出 {max_chars} 字符]"
        return text
