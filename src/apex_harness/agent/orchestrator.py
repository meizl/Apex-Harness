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

# 用户中断重规划的触发关键词
_INTERRUPT_KEYWORDS = (
    "不对", "错了", "错", "不对不对", "重新来", "重新规划", "重新计划",
    "停", "停止", "等等", "不对啊", "有问题", "不对吧",
)

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
        max_retries: int = 2,
        enable_observation_replan: bool = True,
    ) -> None:
        self.session = session
        self.llm = llm
        #: ReAct 执行器（简单任务直接用，复杂任务中作为 Executor）
        self.react = AgentLoop(session, llm, tools=tools, max_steps=max_steps)
        #: 规划 Agent（复杂任务用，独立会话）
        self.planner = Planner(llm)
        #: 当前计划及每个步骤的执行状态（复杂任务时填充，供进度查询）。
        self.plan: list[PlanStep] = []
        #: 单步最大重试次数（失败重规划触发阈值）
        self.max_retries = max_retries
        #: 是否开启观察驱动重规划（每步完成后评估是否调整）
        self.enable_observation_replan = enable_observation_replan
        #: 当前 Plan-and-Execute 的原始任务（供用户中断重规划使用）
        self._original_task: str = ""
        #: 已完成步骤及其结果摘要（供重规划使用）
        self._step_results: list[dict[str, str]] = []

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

    def _is_interrupt_signal(self, text: str) -> bool:
        """判断用户输入是否为中断/重规划信号。"""
        t = text.strip()
        if not t:
            return False
        return any(kw in t for kw in _INTERRUPT_KEYWORDS)

    def _has_active_plan(self) -> bool:
        """是否有未完成的 Plan-and-Execute 计划。"""
        return bool(self.plan) and any(
            s.status in (StepStatus.PENDING, StepStatus.RUNNING, StepStatus.FAILED)
            for s in self.plan
        )

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

        用户中断重规划：若用户输入是中断信号（如"不对"）且存在未完成计划，
        则带着已执行结果重新规划剩余步骤并继续执行。
        """
        # 用户中断重规划：检测到中断信号 + 有未完成计划
        if self._is_interrupt_signal(user_input) and self._has_active_plan():
            async for ev in self._run_interrupt_replan(user_input, **kwargs):
                yield ev
            return

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
        """把 Plan-and-Execute 的执行过程产出为事件流（含重规划）。

        重规划机制：
        - 失败重规划：单步连续失败 ``max_retries`` 次后，带着已执行结果重新规划剩余步骤。
        - 观察驱动重规划：每步完成后调用 Planner 评估是否需要调整剩余步骤。
        """
        generation_id = new_generation_id()
        yield StreamEvent(type="generation_start", generation_id=generation_id)

        yield StreamEvent(type="text", generation_id=generation_id, text="🔍 正在分析任务并生成执行计划...\n\n")
        steps = await self.planner.plan(user_input)

        if not steps:
            yield StreamEvent(type="text", generation_id=generation_id, text="计划生成失败，改为直接执行。\n\n")
            async for ev in self.react.run_stream_events(user_input, **kwargs):
                if ev.type == "done":
                    continue
                yield ev
            yield StreamEvent(type="done", generation_id=generation_id)
            return

        self._original_task = user_input
        self._step_results = []
        self.plan = [PlanStep(description=s) for s in steps]

        yield StreamEvent(type="text", generation_id=generation_id, text="📋 执行计划：\n")
        for i, step in enumerate(steps, 1):
            yield StreamEvent(type="text", generation_id=generation_id, text=f"  {i}. {step}\n")
        yield StreamEvent(type="text", generation_id=generation_id, text="\n")

        # 用索引遍历，支持重规划后步骤列表变化
        idx = 0
        while idx < len(self.plan):
            plan_step = self.plan[idx]
            # 跳过已完成的步骤（重规划后可能混入已执行步骤）
            if plan_step.status == StepStatus.DONE:
                idx += 1
                continue

            step_text = plan_step.description
            plan_step.status = StepStatus.RUNNING
            yield StreamEvent(type="text", generation_id=generation_id, text=f"\n⏳ 步骤 {idx + 1}/{len(self.plan)}：{step_text}\n")

            step_output = ""
            success = False
            error_msg = ""

            # 失败重规划：重试 max_retries 次
            for attempt in range(self.max_retries + 1):
                if attempt > 0:
                    yield StreamEvent(type="text", generation_id=generation_id, text=f"   🔄 第 {attempt + 1} 次重试...\n")
                try:
                    step_output = ""
                    async for ev in self.react.run_stream_events(step_text, **kwargs):
                        if ev.type == "done":
                            continue
                        if ev.type == "error":
                            if ev.text:
                                step_output += ev.text
                                yield StreamEvent(type="text", generation_id=ev.generation_id, text=ev.text)
                            continue
                        if ev.type == "text" and ev.text:
                            step_output += ev.text
                        yield ev
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    error_msg = str(exc)
                    yield StreamEvent(type="text", generation_id=generation_id, text=f"   ❌ 尝试 {attempt + 1} 失败：{exc}\n")

            if success:
                plan_step.status = StepStatus.DONE
                self._step_results.append({"step": step_text, "result": step_output[:500]})
                yield StreamEvent(type="text", generation_id=generation_id, text=f"\n✅ 步骤 {idx + 1}/{len(self.plan)} 完成\n")

                # 观察驱动重规划：每步完成后评估是否调整剩余步骤
                if self.enable_observation_replan and idx < len(self.plan) - 1:
                    async for ev in self._maybe_observation_replan(generation_id, **kwargs):
                        yield ev
            else:
                plan_step.status = StepStatus.FAILED
                plan_step.error = error_msg
                self._step_results.append({"step": step_text, "result": f"[FAILED] {error_msg}"})
                yield StreamEvent(type="text", generation_id=generation_id, text=f"\n❌ 步骤 {idx + 1}/{len(self.plan)} 连续失败 {self.max_retries + 1} 次\n")

                # 失败重规划：带着已执行结果重新规划剩余步骤
                async for ev in self._maybe_failure_replan(generation_id, error_msg, **kwargs):
                    yield ev

            idx += 1

        done = sum(1 for s in self.plan if s.status is StepStatus.DONE)
        failed = sum(1 for s in self.plan if s.status is StepStatus.FAILED)
        yield StreamEvent(type="text", generation_id=generation_id, text=f"\n✅ 完成 {done} 步，失败 {failed} 步。")
        yield StreamEvent(type="done", generation_id=generation_id)

    async def _maybe_failure_replan(
        self, generation_id: str, error_msg: str, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """失败重规划：基于已执行结果和错误信息重新规划剩余步骤。"""
        completed = list(self._step_results)
        remaining = [s.description for s in self.plan if s.status in (StepStatus.PENDING, StepStatus.FAILED)]
        if not remaining:
            return

        yield StreamEvent(type="text", generation_id=generation_id, text="   🔄 触发失败重规划，正在调整剩余步骤...\n")
        new_remaining = await self.planner.replan(
            original_task=self._original_task,
            completed=completed,
            remaining=remaining,
            feedback=f"步骤连续失败：{error_msg}",
        )

        # 用新的剩余步骤替换 plan 中未完成的部分
        done_steps = [s for s in self.plan if s.status == StepStatus.DONE]
        self.plan = done_steps + [PlanStep(description=s) for s in new_remaining]
        yield StreamEvent(type="text", generation_id=generation_id, text=f"   📋 调整后的剩余步骤：\n")
        for i, s in enumerate(new_remaining, 1):
            yield StreamEvent(type="text", generation_id=generation_id, text=f"      {i}. {s}\n")

    async def _maybe_observation_replan(
        self, generation_id: str, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """观察驱动重规划：每步完成后评估是否需要调整剩余步骤。"""
        completed = list(self._step_results)
        remaining = [s.description for s in self.plan if s.status == StepStatus.PENDING]
        if not remaining:
            return

        new_remaining = await self.planner.replan(
            original_task=self._original_task,
            completed=completed,
            remaining=remaining,
            feedback="评估已执行结果，判断剩余步骤是否需要调整。如无需调整请原样返回。",
        )

        # 仅当新计划与原剩余步骤不同时才替换
        if new_remaining != remaining:
            yield StreamEvent(type="text", generation_id=generation_id, text="   🔄 观察到计划需要调整，更新剩余步骤...\n")
            done_steps = [s for s in self.plan if s.status == StepStatus.DONE]
            self.plan = done_steps + [PlanStep(description=s) for s in new_remaining]
            for i, s in enumerate(new_remaining, 1):
                yield StreamEvent(type="text", generation_id=generation_id, text=f"      {i}. {s}\n")

    async def _run_interrupt_replan(
        self, user_feedback: str, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """用户中断重规划：用户说"不对"等信号时，基于反馈重新规划并执行。"""
        generation_id = new_generation_id()
        yield StreamEvent(type="generation_start", generation_id=generation_id)

        completed = list(self._step_results)
        remaining = [s.description for s in self.plan if s.status in (StepStatus.PENDING, StepStatus.FAILED, StepStatus.RUNNING)]

        yield StreamEvent(type="text", generation_id=generation_id, text=f"⚠️ 检测到中断信号：{user_feedback}\n")
        yield StreamEvent(type="text", generation_id=generation_id, text="🔄 基于已执行结果和你的反馈重新规划...\n\n")

        new_remaining = await self.planner.replan(
            original_task=self._original_task,
            completed=completed,
            remaining=remaining,
            feedback=f"用户反馈：{user_feedback}",
        )

        # 替换 plan 中未完成的部分
        done_steps = [s for s in self.plan if s.status == StepStatus.DONE]
        self.plan = done_steps + [PlanStep(description=s) for s in new_remaining]

        yield StreamEvent(type="text", generation_id=generation_id, text="📋 调整后的执行计划：\n")
        for i, s in enumerate(new_remaining, 1):
            yield StreamEvent(type="text", generation_id=generation_id, text=f"  {i}. {s}\n")
        yield StreamEvent(type="text", generation_id=generation_id, text="\n")

        # 执行新的剩余步骤（复用 plan-and-execute 的核心循环）
        idx = len(done_steps)
        while idx < len(self.plan):
            plan_step = self.plan[idx]
            step_text = plan_step.description
            plan_step.status = StepStatus.RUNNING
            yield StreamEvent(type="text", generation_id=generation_id, text=f"\n⏳ 步骤 {idx + 1}/{len(self.plan)}：{step_text}\n")

            step_output = ""
            try:
                async for ev in self.react.run_stream_events(step_text, **kwargs):
                    if ev.type == "done":
                        continue
                    if ev.type == "error":
                        if ev.text:
                            yield StreamEvent(type="text", generation_id=ev.generation_id, text=ev.text)
                        continue
                    if ev.type == "text" and ev.text:
                        step_output += ev.text
                    yield ev
                plan_step.status = StepStatus.DONE
                self._step_results.append({"step": step_text, "result": step_output[:500]})
                yield StreamEvent(type="text", generation_id=generation_id, text=f"\n✅ 步骤 {idx + 1}/{len(self.plan)} 完成\n")
            except Exception as exc:  # noqa: BLE001
                plan_step.status = StepStatus.FAILED
                plan_step.error = str(exc)
                yield StreamEvent(type="text", generation_id=generation_id, text=f"\n❌ 步骤 {idx + 1}/{len(self.plan)} 失败：{exc}\n")
            idx += 1

        done = sum(1 for s in self.plan if s.status is StepStatus.DONE)
        failed = sum(1 for s in self.plan if s.status is StepStatus.FAILED)
        yield StreamEvent(type="text", generation_id=generation_id, text=f"\n✅ 完成 {done} 步，失败 {failed} 步。")
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
