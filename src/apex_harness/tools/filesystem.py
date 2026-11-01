"""文件系统工具：读取、写入、列出目录、删除文件。

所有文件操作默认限制在 ``base_dir`` 目录下，防止越权访问。
可通过插件配置 ``base_dir`` 指定工作目录，默认是当前工作目录。

配置示例（Kernel config）::

    {
        "read_file": {"base_dir": "/path/to/workspace"},
        "write_file": {"base_dir": "/path/to/workspace"},
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Tool


def _resolve_path(base_dir: Path | None, user_path: str) -> Path:
    """把用户传入的路径解析到 base_dir 下，防止目录穿越。

    如果解析后的路径不在 base_dir 内，抛出 PermissionError。
    """
    base = (base_dir or Path.cwd()).resolve()
    target = (base / user_path).resolve()
    # 检查是否在 base_dir 内（允许等于 base_dir 本身）
    try:
        target.relative_to(base)
    except ValueError:
        raise PermissionError(
            f"路径 '{user_path}' 越出工作目录 '{base}'，禁止访问"
        )
    return target


class ReadFileTool(Tool):
    """读取文件内容。"""

    name = "read_file"
    description = "读取指定文件的文本内容"
    requires = ["tools"]
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于工作目录）",
            },
            "max_bytes": {
                "type": "integer",
                "description": "最大读取字节数，默认 100KB",
                "default": 102400,
            },
        },
        "required": ["path"],
    }

    async def setup(self) -> None:
        cfg = self.ctx.config or {}
        self._base_dir = Path(cfg["base_dir"]).resolve() if "base_dir" in cfg else None
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, path: str, max_bytes: int = 102400, **kwargs: Any) -> str:
        target = _resolve_path(self._base_dir, path)
        if not target.exists():
            return f"Error: 文件不存在 '{path}'"
        if not target.is_file():
            return f"Error: '{path}' 不是文件"
        try:
            data = target.read_bytes()
        except PermissionError:
            return f"Error: 无权限读取 '{path}'"
        if len(data) > max_bytes:
            data = data[:max_bytes]
            return data.decode("utf-8", errors="replace") + f"\n... [已截断，超出 {max_bytes} 字节]"
        return data.decode("utf-8", errors="replace")


class WriteFileTool(Tool):
    """写入文件内容。"""

    name = "write_file"
    description = "把文本内容写入指定文件，文件不存在则创建"
    requires = ["tools"]
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于工作目录）",
            },
            "content": {
                "type": "string",
                "description": "要写入的文本内容",
            },
            "append": {
                "type": "boolean",
                "description": "是否追加模式，默认覆盖",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    async def setup(self) -> None:
        cfg = self.ctx.config or {}
        self._base_dir = Path(cfg["base_dir"]).resolve() if "base_dir" in cfg else None
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, path: str, content: str, append: bool = False, **kwargs: Any) -> str:
        target = _resolve_path(self._base_dir, path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(target, mode, encoding="utf-8") as f:
                f.write(content)
        except PermissionError:
            return f"Error: 无权限写入 '{path}'"
        return f"已写入 {len(content)} 字符到 '{path}'"


class ListDirTool(Tool):
    """列出目录内容。"""

    name = "list_dir"
    description = "列出指定目录下的文件和子目录"
    requires = ["tools"]
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径（相对于工作目录），默认为当前目录",
                "default": ".",
            },
        },
    }

    async def setup(self) -> None:
        cfg = self.ctx.config or {}
        self._base_dir = Path(cfg["base_dir"]).resolve() if "base_dir" in cfg else None
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, path: str = ".", **kwargs: Any) -> str:
        target = _resolve_path(self._base_dir, path)
        if not target.exists():
            return f"Error: 目录不存在 '{path}'"
        if not target.is_dir():
            return f"Error: '{path}' 不是目录"
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"[DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"[FILE] {entry.name}  ({size} bytes)")
        return "\n".join(lines) if lines else "(空目录)"


class DeleteFileTool(Tool):
    """删除文件或空目录。"""

    name = "delete_file"
    description = "删除指定文件或空目录"
    requires = ["tools"]
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要删除的文件或空目录路径（相对于工作目录）",
            },
        },
        "required": ["path"],
    }

    async def setup(self) -> None:
        cfg = self.ctx.config or {}
        self._base_dir = Path(cfg["base_dir"]).resolve() if "base_dir" in cfg else None
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, path: str, **kwargs: Any) -> str:
        target = _resolve_path(self._base_dir, path)
        if not target.exists():
            return f"Error: 路径不存在 '{path}'"
        try:
            if target.is_dir():
                target.rmdir()  # 只删空目录，防止误删
            else:
                target.unlink()
        except OSError as exc:
            return f"Error: 删除失败: {exc}"
        return f"已删除 '{path}'"
