"""Orchestrator：任务路由 + 模式调度。

作为 Agent 系统的总入口，负责：
1. 判断用户任务的复杂度（简单 / 复杂）
2. 选择执行模式：
   - 简单任务 → ReAct（AgentLoop，边想边做）
   - 复杂任务 → Plan-and-Execute（Planner 拆步 + Executor 逐步执行）
3. 调度执行器并流式返回结果
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from apex_harness.agent.loop import AgentLoop
from apex_harness.agent.planner import Planner
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session
from apex_harness.tools.base import Tool

# 复杂度分类提示词
_CLASSIFY_PROMPT = """你是一个任务复杂度分类器。判断用户的请求属于「简单」还是「复杂」任务。

简单任务：单步或少量步骤即可完成，不需要规划。例如：
- 回答一个事实性问题
- 读取单个文件
- 执行一个 shell 命令
- 简单的对话

复杂任务：需要多个步骤、依赖关系或全局规划。例如：
- "帮我分析这个项目的架构并生成文档"
- "读取所有配置文件然后合并它们"
- "写一个脚本测试并运行"
- 涉及多个独立操作的任务

只回答一个词：simple 或 complex。不要解释。"""


class Orchestrator:
    """Agent 系统的总入口，负责任务路由和模式调度。

    用法::

        orch = Orchestrator(session, llm, tools={"read_file": ReadFileTool()})
        async for chunk in orch.run_stream("帮我读一下 config.yaml"):
            print(chunk)
    """

    def __init__(
        self,
        session: Session,
        llm: LLMProvider,
        *,
        tools: dict[str, Tool] | None = None,
        max_steps: int = 10,
    ) -> None:
        self.session = session
        self.llm = llm
        #: ReAct 执行器（简单任务直接用，复杂任务中作为 Executor）
        self.react = AgentLoop(session, llm, tools=tools, max_steps=max_steps)
        #: 规划 Agent（复杂任务用，独立会话）
        self.planner = Planner(llm)

    async def _classify(self, user_input: str) -> str:
        """用 LLM 判断任务复杂度，返回 ``"simple"`` 或 ``"complex"``。"""
        try:
            reply = await self.llm.chat([
                SystemMessage(content=_CLASSIFY_PROMPT),
                HumanMessage(content=user_input),
            ])
            text = (reply.content or "").strip().lower()
            if "complex" in text:
                return "complex"
            return "simple"
        except Exception:
            # LLM 分类失败时保守回退为 simple，走 ReAct
            return "simple"

    async def run_stream(self, user_input: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式执行：先路由，再调度执行器。"""
        mode = await self._classify(user_input)

        if mode == "complex":
            async for chunk in self._run_plan_and_execute(user_input, **kwargs):
                yield chunk
        else:
            # 简单任务：直接 ReAct
            async for chunk in self.react.run_stream(user_input, **kwargs):
                yield chunk

    async def _run_plan_and_execute(
        self, user_input: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Plan-and-Execute：Planner 拆步 → Executor 逐步执行。"""
        # 1. 规划
        yield "🔍 正在分析任务并生成执行计划...\n\n"
        steps = await self.planner.plan(user_input)

        if not steps:
            yield "计划生成失败，改为直接执行。\n\n"
            async for chunk in self.react.run_stream(user_input, **kwargs):
                yield chunk
            return

        # 2. 展示计划
        yield "📋 执行计划：\n"
        for i, step in enumerate(steps, 1):
            yield f"  {i}. {step}\n"
        yield "\n"

        # 3. 逐步执行
        for i, step in enumerate(steps, 1):
            yield f"\n▶ 步骤 {i}/{len(steps)}：{step}\n"
            async for chunk in self.react.run_stream(step, **kwargs):
                yield chunk
            yield "\n"

        # 4. 完成
        yield "\n✅ 所有步骤执行完成。"
