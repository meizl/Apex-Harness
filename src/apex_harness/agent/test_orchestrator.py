"""Orchestrator 任务路由的单元测试。

覆盖：
1. LLM 分类：simple
2. LLM 分类：complex
3. LLM 分类失败兜底为 simple
4. run_stream 简单任务走 ReAct
5. run_stream 复杂任务走 Plan-and-Execute
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from apex_harness.agent.orchestrator import Orchestrator
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session


async def _raise_async(*args, **kwargs):
    """模拟 LLM 调用失败。"""
    raise RuntimeError("LLM 调用失败")


class FakeLLM(LLMProvider):
    """可预设回复的 LLM 替身。"""

    def __init__(self, classify_reply: str = "simple", chat_reply: str = "你好", plan_reply: str = '["步骤一", "步骤二"]') -> None:
        super().__init__()
        self.classify_reply = classify_reply
        self.chat_reply = chat_reply
        self.plan_reply = plan_reply
        self.chat_calls = 0

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.chat_calls += 1
        from langchain_core.messages import SystemMessage

        first_content = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""
        last_content = messages[-1].content if messages else ""

        # 复杂度分类
        if "复杂度分类器" in first_content:
            return AIMessage(content=self.classify_reply)

        # Planner 内部调用 create_plan 工具时，工具内部再调 LLM 生成计划
        if "任务规划助手" in first_content:
            return AIMessage(content=self.plan_reply)

        # Planner 的 AgentLoop 调用（带 tools）→ 返回 create_plan 工具调用
        if tools and any(t.get("function", {}).get("name") == "create_plan" for t in tools):
            # 检查是否已有工具结果（ToolMessage），有则说明已执行过工具，返回最终回复
            from langchain_core.messages import ToolMessage

            has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
            if not has_tool_result:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "create_plan",
                        "args": {"task": str(last_content)},
                        "id": "call_plan_1",
                    }],
                )
            return AIMessage(content="计划已生成。")

        return AIMessage(content=self.chat_reply)


@pytest.fixture
def make_orch():
    def _make(classify_reply: str = "simple", chat_reply: str = "你好", plan_reply: str = '["步骤一", "步骤二"]'):
        llm = FakeLLM(classify_reply=classify_reply, chat_reply=chat_reply, plan_reply=plan_reply)
        session = Session()
        return Orchestrator(session, llm), llm

    return _make


class TestClassify:
    """复杂度分类测试（全部走 LLM）。"""

    @pytest.mark.asyncio
    async def test_llm_classify_simple(self, make_orch):
        orch, _ = make_orch(classify_reply="simple")
        assert await orch._classify("你好") == "simple"

    @pytest.mark.asyncio
    async def test_llm_classify_complex(self, make_orch):
        orch, _ = make_orch(classify_reply="complex")
        assert await orch._classify("分析这个项目的架构") == "complex"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_simple(self, make_orch):
        orch, _ = make_orch()
        # 让 LLM 抛异常
        orch.llm.chat = _raise_async  # type: ignore
        assert await orch._classify("随便什么") == "simple"


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
    async def test_complex_uses_plan_and_execute(self, make_orch):
        orch, _ = make_orch(
            classify_reply="complex",
            chat_reply="执行完成",
            plan_reply='["读 README", "总结架构"]',
        )
        chunks = []
        async for chunk in orch.run_stream("分析项目架构"):
            chunks.append(chunk)
        text = "".join(chunks)
        # 规划阶段
        assert "执行计划" in text
        assert "读 README" in text
        assert "总结架构" in text
        # 执行阶段
        assert "步骤 1/2" in text
        assert "执行完成" in text
        assert "所有步骤执行完成" in text
