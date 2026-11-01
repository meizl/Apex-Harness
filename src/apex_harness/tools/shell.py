"""Shell 命令执行工具。

支持超时控制和输出截断，防止命令卡死或输出过大。
可通过配置设置允许的命令白名单（``allow_list``），不配置则不限制。

配置示例::

    {
        "shell_exec": {
            "timeout": 30,
            "max_output": 65536,
            "allow_list": ["ls", "cat", "grep", "python"]
        }
    }
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from .base import Tool


class ShellExecTool(Tool):
    """执行 Shell 命令并返回输出。"""

    name = "shell_exec"
    description = "在工作目录下执行一条 shell 命令，返回 stdout 和 stderr"
    requires = ["tools"]
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令",
            },
            "timeout": {
                "type": "integer",
                "description": "超时秒数，默认 30 秒",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    async def setup(self) -> None:
        cfg = self.ctx.config or {}
        self._default_timeout = cfg.get("timeout", 30)
        self._max_output = cfg.get("max_output", 65536)
        self._allow_list: list[str] | None = cfg.get("allow_list")
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, command: str, timeout: int | None = None, **kwargs: Any) -> str:
        # 白名单检查
        if self._allow_list is not None:
            try:
                parts = shlex.split(command)
            except ValueError as exc:
                return f"Error: 命令解析失败: {exc}"
            if not parts:
                return "Error: 空命令"
            cmd_name = parts[0]
            if cmd_name not in self._allow_list:
                return f"Error: 命令 '{cmd_name}' 不在白名单中，禁止执行"

        timeout = timeout or self._default_timeout

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return f"Error: 启动命令失败: {exc}"

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: 命令执行超时（{timeout}秒），已终止"

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        # 输出截断
        if len(stdout_text) > self._max_output:
            stdout_text = stdout_text[: self._max_output] + f"\n... [已截断，超出 {self._max_output} 字符]"
        if len(stderr_text) > self._max_output:
            stderr_text = stderr_text[: self._max_output] + f"\n... [已截断，超出 {self._max_output} 字符]"

        exit_code = proc.returncode
        result = f"[exit code: {exit_code}]\n"
        if stdout_text:
            result += f"--- stdout ---\n{stdout_text}"
        if stderr_text:
            result += f"--- stderr ---\n{stderr_text}"
        return result
