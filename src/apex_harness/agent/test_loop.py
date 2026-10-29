"""AgentLoop 主循环的单元测试。

覆盖：
1. 模型直接回复（无工具调用）；
2. 模型调用一次工具后给出最终回复；
3. 调用不存在的工具时返回错误信息，循环继续；
4. 超过最大步数时兜底返回。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.agent.loop import AgentLoop
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session
from apex_harness.tools.base import Tool


class FakeLLM(LLMProvider):
    """按预设回复序列依次返回的 LLM 替身。"""

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.calls.append({"messages": messages, "tools": tools})
        # 按顺序返回，用完了就重复最后一个
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]


class EchoTool(Tool):
    """把收到的参数原样返回的假工具。"""

    name = "echo"
    description = "回显输入的文本"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, **kwargs: Any) -> str:
        return f"echo: {kwargs.get('text', '')}"


@pytest.fixture
def session() -> Session:
    return Session()


async def test_direct_reply_without_tools(session: Session) -> None:
    """模型不调工具，直接返回文本。"""
    llm = FakeLLM([AIMessage(content="你好，有什么可以帮你？")])
    loop = AgentLoop(session, llm)

    reply = await loop.run("你好")

    assert reply == "你好，有什么可以帮你？"
    # 历史应该是 human + ai
    roles = [m.type for m in session.history()]
    assert roles == ["human", "ai"]


async def test_single_tool_call_then_reply(session: Session) -> None:
    """模型先调工具，拿到结果后再给最终回复。"""
    # LangChain tool_calls 格式：name / args(dict) / id
    llm = FakeLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "echo",
                "args": {"text": "hello"},
                "id": "call_1",
            }],
        ),
        AIMessage(content="工具返回了 hello，这是最终回复。"),
    ])
    loop = AgentLoop(session, llm, tools={"echo": EchoTool()})

    reply = await loop.run("帮我 echo 一下 hello")

    assert reply == "工具返回了 hello，这是最终回复。"
    # 历史：human → ai(tool_calls) → tool → ai(final)
    roles = [m.type for m in session.history()]
    assert roles == ["human", "ai", "tool", "ai"]
    # 工具消息的内容是执行结果
    tool_msg = session.history()[2]
    assert tool_msg.content == "echo: hello"
    assert tool_msg.tool_call_id == "call_1"


async def test_unknown_tool_returns_error(session: Session) -> None:
    """模型调了不存在的工具，返回错误信息，不崩溃。"""
    llm = FakeLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "nonexistent",
                "args": {},
                "id": "call_x",
            }],
        ),
        AIMessage(content="看起来没有这个工具，最终回复。"),
    ])
    loop = AgentLoop(session, llm, tools={})

    reply = await loop.run("调一个不存在的工具")

    assert reply == "看起来没有这个工具，最终回复。"
    tool_msg = session.history()[2]
    assert "未知工具" in tool_msg.content


async def test_max_steps_exceeded(session: Session) -> None:
    """模型一直调工具，超过 max_steps 后兜底返回。"""
    # 每次都返回工具调用，永远不给最终回复
    llm = FakeLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "echo",
                "args": {"text": "x"},
                "id": "call_loop",
            }],
        ),
    ])
    loop = AgentLoop(session, llm, tools={"echo": EchoTool()}, max_steps=3)

    reply = await loop.run("无限循环")

    assert "最大步数" in reply
    # 每个 step 产生 ai(tool_calls) + tool，最后还有一个 ai(fallback)
    # 即 human + 3*(ai, tool) + ai(fallback) = 8 条
    assert len(session.history()) == 1 + 3 * 2 + 1
