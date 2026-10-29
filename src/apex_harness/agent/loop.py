"""Agent 主执行循环：prompt → model → tool → next turn。

主循环负责把单轮 ``Turn`` 串成多步推理：模型可以反复调用工具，直到给出
最终回复。这是 Agent 能力的核心（ReAct 模式）。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session
from apex_harness.tools.base import Tool

# 工具定义的标准格式（OpenAI 兼容）。
# 把项目内的 Tool 插件转成 LLM 能理解的 function calling schema。


def _tool_to_schema(tool: Tool) -> dict[str, Any]:
    """把一个 Tool 插件转成 LLM function calling 的 schema。"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class AgentLoop:
    """Agent 主循环。

    一次 ``run`` 会反复调用 LLM：如果模型返回工具调用，就执行工具并把结果
    塞回会话历史，再次调用模型；直到模型不再调工具，返回最终文本。

    用法::

        loop = AgentLoop(session, llm, tools={"read_file": ReadFileTool()})
        reply = await loop.run("帮我读一下 config.yaml")
    """

    def __init__(
        self,
        session: Session,
        llm: LLMProvider,
        *,
        tools: dict[str, Tool] | None = None,
        max_steps: int = 10,
    ) -> None:
        #: 会话，所有消息都写在这里。
        self.session = session
        #: LLM 提供者。
        self.llm = llm
        #: 可用工具，name -> Tool 实例。
        self.tools: dict[str, Tool] = tools or {}
        #: 最大循环步数，防止模型无限调工具。
        self.max_steps = max_steps

    # -- 工具 schema ------------------------------------------------------

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """把所有工具转成 LLM 需要的 schema 列表。"""
        return [_tool_to_schema(t) for t in self.tools.values()]

    # -- 工具执行 ---------------------------------------------------------

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> str:
        """执行一次工具调用，返回结果字符串。

        LangChain ``AIMessage.tool_calls`` 的格式是
        ``{"name": ..., "args": {...}, "id": ...}``，其中 ``args`` 已经是
        解析好的 dict。

        任何异常都被捕获并转成字符串返回，保证循环不中断——模型看到错误信息
        后可以自行纠正。
        """
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}

        tool = self.tools.get(name)
        if tool is None:
            return f"Error: 未知工具 '{name}'"

        try:
            result = await tool.run(**args)
            # 结果可能不是字符串，统一转成文本。
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - 故意捕获所有异常
            return f"Error: 执行工具 '{name}' 时出错: {exc}"

    # -- 主循环 -----------------------------------------------------------

    async def run(self, user_input: str, **kwargs: Any) -> str:
        """执行一次完整的 Agent 交互，返回最终回复文本。"""
        # 1. 写入用户消息
        self.session.add_user(user_input)

        tools_schema = self._tool_schemas() or None

        # 2. 多步循环
        for step in range(self.max_steps):
            # 2a. 调 LLM
            messages = self.session.to_llm_format()
            response: AIMessage = await self.llm.chat(
                messages, tools=tools_schema, **kwargs
            )

            tool_calls = response.tool_calls

            # 2b. 没有工具调用 → 最终回复
            if not tool_calls:
                self.session.add_assistant(response.content)
                return response.content

            # 2c. 有工具调用 → 先把 assistant 消息（带 tool_calls）写进历史
            self.session.add_assistant(response.content, tool_calls=tool_calls)

            # 2d. 逐个执行工具，把结果写回历史
            for tool_call in tool_calls:
                result = await self._execute_tool_call(tool_call)
                call_id = tool_call.get("id", "")
                self.session.add_tool(result, tool_call_id=call_id)

            # 2e. 继续循环：LLM 下一轮会看到工具结果

        # 3. 超过最大步数，兜底返回
        fallback = f"已达到最大步数 {self.max_steps}，未能得到最终回复。"
        self.session.add_assistant(fallback)
        return fallback
