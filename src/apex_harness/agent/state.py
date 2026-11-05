"""Agent 执行状态（类 LangGraph StateGraph 的 state）。

``AgentState`` 是一次 Agent 运行过程中**唯一的可变上下文**，每轮与 LLM 的
交互信息都存放在这里。主循环（:class:`~apex_harness.agent.loop.AgentLoop`）
围绕 ``state`` 运转：读取 ``state.messages`` 调 LLM，把工具结果写回
``state.tool_results``，通过 ``state.next_step`` 决定下一步做什么。

字段说明：
- ``messages``        对话历史（LangChain BaseMessage 列表），LLM 调用的输入
- ``user_input``      当前轮次的用户输入
- ``current_plan``    Plan-and-Execute 模式下的执行计划步骤
- ``iteration_count`` 当前迭代次数（每完成一轮 LLM+工具 自增）
- ``approved``        当前动作是否已获批（预留的 human-in-the-loop 开关）
- ``errors``          累积的错误信息
- ``tool_results``    工具调用结果缓存，key 为 ``tool_call_id``
- ``next_step``       下一步动作：``"llm"`` / ``"tools"`` / ``"done"`` / ``"error"``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

# next_step 的合法取值
NEXT_STEP_LLM = "llm"
NEXT_STEP_TOOLS = "tools"
NEXT_STEP_DONE = "done"
NEXT_STEP_ERROR = "error"


@dataclass
class AgentState:
    """一次 Agent 运行的可变状态。

    所有字段都有默认值，便于在循环中逐步填充。``messages`` 默认引用外部
    会话的消息列表（同一个 list 对象），这样写入 state 等于写入会话。
    """

    #: 对话历史，LLM 调用的输入；通常直接引用 session.messages。
    messages: list[BaseMessage] = field(default_factory=list)
    #: 当前轮次用户输入。
    user_input: str = ""
    #: Plan-and-Execute 模式下的计划步骤列表。
    current_plan: list[str] = field(default_factory=list)
    #: 当前迭代次数（每完成一轮 LLM 决策自增）。
    iteration_count: int = 0
    #: 当前动作是否获批（human-in-the-loop 预留）。
    approved: bool = True
    #: 累积的错误信息。
    errors: list[str] = field(default_factory=list)
    #: 工具调用结果缓存，key 为 tool_call_id。
    tool_results: dict[str, str] = field(default_factory=dict)
    #: 下一步动作：llm / tools / done / error。
    next_step: str = NEXT_STEP_LLM
    #: 本轮 LLM 返回的待执行工具调用（llm → tools 之间传递）。
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: 本轮 LLM 生成的 generation_id（llm → tools 之间传递，供事件关联）。
    last_generation_id: str = ""

    # -- 便捷方法 --------------------------------------------------------

    def record_error(self, msg: str) -> None:
        """记录一条错误信息。"""
        self.errors.append(msg)

    def last_error(self) -> str:
        """返回最近一条错误信息，无错误时返回空串。"""
        return self.errors[-1] if self.errors else ""

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（messages 仅保留角色和内容摘要）。"""
        return {
            "user_input": self.user_input,
            "current_plan": list(self.current_plan),
            "iteration_count": self.iteration_count,
            "approved": self.approved,
            "errors": list(self.errors),
            "tool_results": dict(self.tool_results),
            "next_step": self.next_step,
            "messages_count": len(self.messages),
        }
