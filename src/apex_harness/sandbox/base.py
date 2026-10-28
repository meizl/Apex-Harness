"""沙箱接口（隔离执行环境）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from apex_harness.core.plugin import Plugin


@dataclass
class SandboxResult:
    """一次代码执行的结果。"""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    #: 额外元信息（如耗时、隔离级别等），由具体沙箱填写。
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """执行是否成功（退出码为 0）。"""
        return self.exit_code == 0


class Sandbox(Plugin):
    """沙箱插件基类。

    沙箱提供一个隔离的执行环境，用于运行代码。具体实现（本地进程、
    Docker 等）继承本类，实现 :meth:`run`，并在 ``setup`` 中注册服务
    （通常注册成 ``"sandbox"``）。
    """

    async def run(self, code: str, language: str = "python") -> SandboxResult:
        """在隔离环境中执行 *code*，返回执行结果。

        *language* 指明代码语言（如 ``"python"``、``"bash"``）。
        """
        raise NotImplementedError
