"""Orchestrator 任务路由的单元测试。

覆盖：
1. 启发式分类：短输入 → simple
2. 启发式分类：复杂关键词 → complex
3. LLM 分类兜底
4. run_stream 简单任务走 ReAct
5. run_stream 复杂任务带模式标识
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.agent.orchestrator import Orchestrator
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session


class FakeLLM(LLMProvider):
    """可预设回复的 LLM 替身。"""

    def __init__(self, classify_reply: str = "simple", chat_reply: str = "你好") -> None:
        super().__init__()
        self.classify_reply = classify_reply
        self.chat_reply = chat_reply
        self.chat_calls = 0

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.chat_calls += 1
        # 判断是分类请求还是正常对话（分类请求第一条是 SystemMessage）
        from langchain_core.messages import SystemMessage

        if messages and isinstance(messages[0], SystemMessage) and "复杂度分类器" in (messages[0].content or ""):
            return AIMessage(content=self.classify_reply)
        return AIMessage(content=self.chat_reply)


@pytest.fixture
def make_orch():
    def _make(classify_reply: str = "simple", chat_reply: str = "你好"):
        llm = FakeLLM(classify_reply=classify_reply, chat_reply=chat_reply)
        session = Session()
        return Orchestrator(session, llm), llm

    return _make


class TestClassify:
    """复杂度分类测试。"""

    @pytest.mark.asyncio
    async def test_short_input_is_simple(self, make_orch):
        orch, _ = make_orch()
        assert await orch._classify("你好") == "simple"

    @pytest.mark.asyncio
    async def test_complex_keyword_triggers_complex(self, make_orch):
        orch, _ = make_orch()
        assert await orch._classify("帮我分析这个项目的架构") == "complex"

    @pytest.mark.asyncio
    async def test_multiple_actions_trigger_complex(self, make_orch):
        orch, _ = make_orch()
        assert await orch._classify("读取文件然后执行脚本") == "complex"

    @pytest.mark.asyncio
    async def test_llm_classify_simple(self, make_orch):
        orch, _ = make_orch(classify_reply="simple")
        # 中等长度、无复杂关键词 → 走 LLM 分类
        assert await orch._classify("今天天气怎么样适合出门吗") == "simple"

    @pytest.mark.asyncio
    async def test_llm_classify_complex(self, make_orch):
        orch, _ = make_orch(classify_reply="complex")
        assert await orch._classify("今天天气怎么样适合出门吗") == "complex"


class TestRunStream:
    """run_stream 路由测试。"""

    @pytest.mark.asyncio
    async def test_simple_goes_to_react(self, make_orch):
        orch, _ = make_orch(classify_reply="simple", chat_reply="你好世界")
        chunks = []
        async for chunk in orch.run_stream("你好"):
            chunks.append(chunk)
        text = "".join(chunks)
        assert "你好世界" in text
        assert "复杂任务模式" not in text

    @pytest.mark.asyncio
    async def test_complex_has_mode_marker(self, make_orch):
        orch, _ = make_orch(classify_reply="complex", chat_reply="分析完成")
        chunks = []
        async for chunk in orch.run_stream("分析项目架构"):
            chunks.append(chunk)
        text = "".join(chunks)
        assert "复杂任务模式" in text
        assert "分析完成" in text
