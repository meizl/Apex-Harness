"""Turn 单轮执行的单元测试。

覆盖点：
1. run() 把用户输入写进 session 历史；
2. run() 调用 llm.chat 时传入的是 session 转出的消息列表；
3. run() 把助手回复写回 session 历史；
4. run() 返回值就是 llm.chat 的回复文本；
5. **kwargs 透传给 llm.chat。
"""

from __future__ import annotations

from typing import Any

import pytest

from apex_harness.agent.turn import Turn
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session
from langchain_core.messages import AIMessage, BaseMessage


class FakeLLM(LLMProvider):
    """不真正发请求的 LLM 替身，记录调用参数并返回固定回复。"""

    def __init__(self, reply: str = "hello") -> None:
        super().__init__()
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return AIMessage(content=self.reply)


@pytest.fixture
def session() -> Session:
    return Session()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM(reply="你好，我是助手")


async def test_run_records_user_message(session: Session, llm: FakeLLM) -> None:
    turn = Turn(session, llm)
    # 先放一条 system，验证历史被保留
    session.add_system("你是一个助手")

    await turn.run("今天天气怎么样？")

    roles = [m.type for m in session.history()]
    assert roles == ["system", "human", "ai"]
    assert session.history()[1].content == "今天天气怎么样？"


async def test_run_passes_session_messages_to_llm(session: Session, llm: FakeLLM) -> None:
    turn = Turn(session, llm)
    session.add_system("system prompt")
    # run 之前的历史就是 LLM 应该收到的内容（assistant 还没写回）
    expected = session.to_llm_format()

    await turn.run("你好")

    sent = llm.calls[0]["messages"]
    assert [m.type for m in sent] == ["system", "human"]
    # 发出去的消息在 run 之前就已确定（不含之后追加的 assistant）
    assert sent[:-1] == expected
    assert sent[-1].content == "你好"


async def test_run_records_assistant_reply(session: Session, llm: FakeLLM) -> None:
    turn = Turn(session, llm)

    await turn.run("hi")

    assistant_msg = session.history()[-1]
    assert assistant_msg.type == "ai"
    assert assistant_msg.content == "你好，我是助手"


async def test_run_returns_llm_reply(session: Session, llm: FakeLLM) -> None:
    turn = Turn(session, llm)

    result = await turn.run("hi")

    assert result == "你好，我是助手"


async def test_run_passes_kwargs_to_llm(session: Session, llm: FakeLLM) -> None:
    turn = Turn(session, llm)

    await turn.run("hi", temperature=0.7, max_tokens=100)

    assert llm.calls[0]["kwargs"] == {"temperature": 0.7, "max_tokens": 100}
