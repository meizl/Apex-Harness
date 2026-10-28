"""消息类型定义（system / user / assistant / tool）。"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

# 消息角色：系统提示、用户输入、助手回复、工具结果。
MessageRole = Literal["system", "user", "assistant", "tool"]


class Message:
    """一条对话消息。

    一条消息由 ``role``（谁发的）和 ``content``（内容）组成，可附带可选的
    工具调用（assistant 发起）和工具调用结果（tool 返回）。
    """

    #: 消息角色，子类可覆盖为固定值。
    role: ClassVar[MessageRole] = "user"

    def __init__(
        self,
        content: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        #: assistant 消息发起的工具调用列表，如 ``[{"name": "read_file", "args": {...}}]``。
        self.tool_calls = tool_calls or []
        #: tool 消息对应的工具调用 ID，用于把结果关联回某次调用。
        self.tool_call_id = tool_call_id
        #: 可选的名字（如函数名、用户名）。
        self.name = name
        #: 额外元信息（时间戳、token 数等）。
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典（LLM 调用格式）。"""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<{type(self).__name__} role={self.role!r} content={self.content!r}>"


class SystemMessage(Message):
    """系统提示消息。"""

    role: ClassVar[MessageRole] = "system"


class UserMessage(Message):
    """用户输入消息。"""

    role: ClassVar[MessageRole] = "user"


class AssistantMessage(Message):
    """助手回复消息。"""

    role: ClassVar[MessageRole] = "assistant"


class ToolMessage(Message):
    """工具返回结果消息。"""

    role: ClassVar[MessageRole] = "tool"
