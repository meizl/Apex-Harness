"""Planner：把复杂任务拆成可执行的步骤列表。

Planner 是一个只含 ``create_plan`` 工具的 Agent。它收到任务后调用
``create_plan`` 工具，该工具内部再调一次 LLM 生成步骤列表。

步骤列表由 :class:`CreatePlanTool` 暂存，Planner 执行完后直接读取。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from apex_harness.agent.loop import AgentLoop
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session
from apex_harness.tools.base import Tool

# 生成计划的提示词
_PLAN_PROMPT = """你是一个任务规划助手。把用户的任务拆成 3-8 个可执行的步骤。

要求：
- 每个步骤是一个完整的、可独立执行的任务描述
- 步骤之间有逻辑顺序
- 只输出 JSON 数组，不要解释，不要 markdown

示例：
["读取 README.md 了解项目结构", "查看 src 目录下的主要模块", "总结项目架构并生成文档"]"""


# 重规划提示词：基于已执行结果和反馈调整剩余步骤
_REPLAN_PROMPT = """你是一个任务重规划助手。根据已执行步骤的结果和反馈，调整剩余的执行计划。

输入信息：
- 原始任务：{task}
- 已完成步骤及结果：
{completed}
- 剩余步骤：
{remaining}
- 反馈/问题：{feedback}

要求：
- 如果已执行结果表明原计划需要调整，重新规划剩余步骤
- 如果剩余步骤仍然合理，直接返回原剩余步骤（可微调）
- 只输出 JSON 数组（步骤列表），不要解释，不要 markdown

示例：
["修正配置文件路径", "重新运行测试"]"""


class CreatePlanTool(Tool):
    """生成任务计划的工具。调用 LLM 把任务拆成步骤列表。"""

    name = "create_plan"
    description = "把一个复杂任务拆成多个可执行的步骤，返回步骤列表"
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "需要拆解的复杂任务描述",
            },
        },
        "required": ["task"],
    }

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__()
        self._llm = llm
        #: 最近一次生成的计划（步骤列表），Planner 执行后从这里取。
        self.last_plan: list[str] = []

    async def run(self, task: str, **kwargs: Any) -> str:
        """调用 LLM 生成计划，暂存到 ``last_plan``，返回提示文本。"""
        reply = await self._llm.chat([
            SystemMessage(content=_PLAN_PROMPT),
            HumanMessage(content=task),
        ])
        text = (reply.content or "").strip()

        # 尝试解析 JSON 数组
        steps: list[str] = []
        try:
            steps = json.loads(text)
            if not isinstance(steps, list):
                steps = []
        except json.JSONDecodeError:
            # 不是合法 JSON，尝试按行切分
            for line in text.splitlines():
                line = line.strip().lstrip("-*0123456789. ")
                if line:
                    steps.append(line)

        # 过滤掉空步骤
        steps = [s for s in steps if isinstance(s, str) and s.strip()]
        self.last_plan = steps

        if not steps:
            return "计划生成失败，请换个方式描述任务。"
        return f"已生成计划，共 {len(steps)} 步。"


class Planner:
    """规划 Agent：只负责把任务拆成步骤列表。

    内部用一个只含 ``create_plan`` 工具的 AgentLoop，让 LLM 调用该工具
    生成计划。执行完后从工具实例读取步骤列表。
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._plan_tool = CreatePlanTool(llm)
        self._session = Session()
        self._loop = AgentLoop(
            self._session,
            llm,
            tools={"create_plan": self._plan_tool},
            max_steps=3,
        )

    async def plan(self, task: str) -> list[str]:
        """对任务进行规划，返回步骤列表。"""
        # 运行 AgentLoop，LLM 会调用 create_plan 工具
        await self._loop.run(f"请为以下任务生成执行计划：{task}")
        return self._plan_tool.last_plan

    async def replan(
        self,
        original_task: str,
        completed: list[dict[str, str]],
        remaining: list[str],
        feedback: str = "",
    ) -> list[str]:
        """基于已执行结果和反馈，重新规划剩余步骤。

        Args:
            original_task: 原始用户任务
            completed: 已完成步骤列表，每项 ``{"step": ..., "result": ...}``
            remaining: 原计划中剩余未执行的步骤
            feedback: 失败信息或用户反馈（如某步连续失败的错误、用户说"不对"等）

        Returns:
            新的剩余步骤列表。若 LLM 未返回有效列表，则原样返回 ``remaining``。
        """
        completed_text = "\n".join(
            f"  - {c['step']} → {c['result'][:200]}" for c in completed
        ) or "  （无）"
        remaining_text = "\n".join(f"  - {r}" for r in remaining) or "  （无）"

        prompt = _REPLAN_PROMPT.format(
            task=original_task,
            completed=completed_text,
            remaining=remaining_text,
            feedback=feedback or "（无额外反馈）",
        )

        try:
            reply = await self._llm.chat([
                SystemMessage(content=prompt),
                HumanMessage(content="请根据以上信息输出调整后的剩余步骤 JSON 数组。"),
            ])
            text = (reply.content or "").strip()
            steps = self._parse_steps(text)
            return steps if steps else list(remaining)
        except Exception:
            # LLM 调用失败时保守返回原剩余步骤，不中断执行
            return list(remaining)

    @staticmethod
    def _parse_steps(text: str) -> list[str]:
        """从 LLM 输出中解析步骤列表。"""
        steps: list[str] = []
        try:
            steps = json.loads(text)
            if not isinstance(steps, list):
                steps = []
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip().lstrip("-*0123456789. ")
                if line:
                    steps.append(line)
        return [s for s in steps if isinstance(s, str) and s.strip()]
