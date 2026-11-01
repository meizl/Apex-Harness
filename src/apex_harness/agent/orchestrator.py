"""Orchestrator：任务路由 + 模式调度。

作为 Agent 系统的总入口，负责：
1. 判断用户任务的复杂度（简单 / 复杂）
2. 选择执行模式：
   - 简单任务 → ReAct（AgentLoop，边想边做）
   - 复杂任务 → Plan-and-Execute（Planner 拆步 + SubAgent 逐步执行，待实现）
3. 调度执行器并流式返回结果

当前复杂任务暂回退到 ReAct（带模式标识），后续接入 Planner/SubAgent。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from apex_harness.agent.loop import AgentLoop
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
        #: ReAct 执行器（简单任务直接用，复杂任务暂时回退到它）
        self.react = AgentLoop(session, llm, tools=tools, max_steps=max_steps)

    async def _classify(self, user_input: str) -> str:
        """判断任务复杂度，返回 ``"simple"`` 或 ``"complex"``。

        先用启发式规则快速判断，不确定时再调用 LLM 分类。
        """
        # 启发式 1：包含多步骤关键词 → 复杂
        complex_keywords = [
            "分析", "总结", "生成", "创建", "实现", "开发",
            "所有", "全部", "每个", "批量", "依次", "然后",
            "项目", "架构", "文档", "报告", "脚本", "测试",
        ]
        if any(kw in user_input for kw in complex_keywords):
            return "complex"

        # 启发式 2：包含多个动词（可能是多步）
        action_words = ["读", "写", "查", "找", "执行", "运行", "分析", "生成", "创建"]
        action_count = sum(1 for w in action_words if w in user_input)
        if action_count >= 2:
            return "complex"

        # 启发式 3：极短输入大概率是简单任务（问候、单步指令）
        if len(user_input) <= 4:
            return "simple"

        # 启发式无法判断 → 用 LLM 分类
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
            # 复杂任务：TODO 接入 Planner + SubAgent
            # 当前先回退到 ReAct，并告知用户模式
            yield "[复杂任务模式] "
            async for chunk in self.react.run_stream(user_input, **kwargs):
                yield chunk
        else:
            # 简单任务：直接 ReAct
            async for chunk in self.react.run_stream(user_input, **kwargs):
                yield chunk
