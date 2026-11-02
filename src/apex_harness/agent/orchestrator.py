"""Orchestrator：任务路由 + 模式调度。

作为 Agent 系统的总入口，负责：
1. 判断用户任务的复杂度（简单 / 复杂）
2. 选择执行模式：
   - 简单任务 → ReAct（AgentLoop，边想边做）
   - 复杂任务 → Plan-and-Execute（Planner 拆步 + Executor 逐步执行）
3. 调度执行器并流式返回结果
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from apex_harness.agent.events import StreamEvent, new_generation_id
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


class StepStatus(str, Enum):
    """计划步骤的执行状态。"""

    PENDING = "pending"    # 未开始
    RUNNING = "running"    # 执行中
    DONE = "done"          # 执行成功
    FAILED = "failed"      # 执行失败


@dataclass
class PlanStep:
    """计划中的一个步骤及其执行状态。"""

    description: str
    status: StepStatus = StepStatus.PENDING
    error: str = ""


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
        #: 当前计划及每个步骤的执行状态（复杂任务时填充，供进度查询）。
        self.plan: list[PlanStep] = []

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

    async def run_stream_events(self, user_input: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """流式执行并产出结构化事件（先路由，再调度执行器）。

        简单任务直接把 ReAct 的事件透传；复杂任务把 Plan-and-Execute 的
        文本输出包装成事件（用同一 generation_id 标识该次执行）。
        """
        mode = await self._classify(user_input)

        if mode == "complex":
            async for ev in self._run_plan_and_execute_events(user_input, **kwargs):
                yield ev
        else:
            async for ev in self.react.run_stream_events(user_input, **kwargs):
                yield ev

    async def _run_plan_and_execute_events(
        self, user_input: str, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """把 Plan-and-Execute 的文本流包装为事件流。"""
        generation_id = new_generation_id()
        yield StreamEvent(type="generation_start", generation_id=generation_id)
        async for text in self._run_plan_and_execute(user_input, **kwargs):
            if text:
                yield StreamEvent(type="text", generation_id=generation_id, text=text)
        yield StreamEvent(type="done", generation_id=generation_id)

    async def _run_plan_and_execute(
        self, user_input: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Plan-and-Execute：Planner 拆步 → Executor 逐步执行。

        执行过程中实时更新每个步骤的状态（pending → running → done/failed），
        单步失败不中断，继续执行后续步骤。
        """
        # 1. 规划
        yield "🔍 正在分析任务并生成执行计划...\n\n"
        steps = await self.planner.plan(user_input)

        if not steps:
            yield "计划生成失败，改为直接执行。\n\n"
            async for chunk in self.react.run_stream(user_input, **kwargs):
                yield chunk
            return

        # 初始化步骤状态（全部 pending）
        self.plan = [PlanStep(description=s) for s in steps]

        # 2. 展示计划
        yield "📋 执行计划：\n"
        for i, step in enumerate(steps, 1):
            yield f"  {i}. {step}\n"
        yield "\n"

        # 3. 逐步执行，实时更新状态
        for i, step in enumerate(steps, 1):
            plan_step = self.plan[i - 1]
            plan_step.status = StepStatus.RUNNING
            yield f"\n⏳ 步骤 {i}/{len(steps)}：{step}\n"
            try:
                async for chunk in self.react.run_stream(step, **kwargs):
                    yield chunk
                plan_step.status = StepStatus.DONE
                yield f"\n✅ 步骤 {i}/{len(steps)} 完成\n"
            except Exception as exc:  # noqa: BLE001 - 单步失败不中断整体
                plan_step.status = StepStatus.FAILED
                plan_step.error = str(exc)
                yield f"\n❌ 步骤 {i}/{len(steps)} 失败：{exc}\n"

        # 4. 汇总结果
        done = sum(1 for s in self.plan if s.status is StepStatus.DONE)
        failed = sum(1 for s in self.plan if s.status is StepStatus.FAILED)
        yield f"\n✅ 完成 {done} 步，失败 {failed} 步。"

    def plan_status(self) -> list[dict[str, Any]]:
        """返回当前计划的步骤状态（供前端/API 查询进度）。"""
        return [
            {
                "description": s.description,
                "status": s.status.value,
                "error": s.error,
            }
            for s in self.plan
        ]
