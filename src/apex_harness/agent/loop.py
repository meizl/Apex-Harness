"""Agent 主执行循环：prompt → model → tool → next turn。

主循环负责把单轮 ``Turn`` 串成多步推理：模型可以反复调用工具，直到给出
最终回复。这是 Agent 能力的核心（ReAct 模式）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage

from apex_harness.agent.events import StreamEvent, new_generation_id
from apex_harness.core.errors import LLMError, friendly_llm_error
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
        stream_max_retries: int = 2,
    ) -> None:
        #: 会话，所有消息都写在这里。
        self.session = session
        #: LLM 提供者。
        self.llm = llm
        #: 可用工具，统一用 tool.name 作为 key，避免 dict key 与 name 不一致。
        self.tools: dict[str, Tool] = {
            t.name: t for t in (tools or {}).values()
        }
        #: 最大循环步数，防止模型无限调工具。
        self.max_steps = max_steps
        #: 流式输出中途中断时的最大重跑次数（已产出部分内容才会触发）。
        self.stream_max_retries = stream_max_retries
        #: 已执行工具结果的缓存，key 为 tool_call_id，避免重试/重入时重复执行。
        self._tool_results: dict[str, str] = {}

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

        以 ``tool_call_id`` 为幂等键缓存结果：同一调用重复进入时直接返回
        缓存，避免工具被重复执行。任何异常都被捕获并转成字符串返回，保证
        循环不中断——模型看到错误信息后可以自行纠正。
        """
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}
        call_id = tool_call.get("id", "")

        # 幂等：已执行过的调用直接复用结果，不重复执行。
        if call_id and call_id in self._tool_results:
            return self._tool_results[call_id]

        tool = self.tools.get(name)
        if tool is None:
            result = f"Error: 未知工具 '{name}'"
        else:
            try:
                result = await tool.run(**args)
                # 结果可能不是字符串，统一转成文本。
                result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001 - 故意捕获所有异常
                result = f"Error: 执行工具 '{name}' 时出错: {exc}"

        if call_id:
            self._tool_results[call_id] = result
        return result

    # -- 主循环 -----------------------------------------------------------

    async def run(self, user_input: str, **kwargs: Any) -> str:
        """执行一次完整的 Agent 交互，返回最终回复文本。

        LLM 调用失败时返回友好的错误提示并写入会话，不抛出异常。
        """
        # 1. 写入用户消息
        self.session.add_user(user_input)

        tools_schema = self._tool_schemas() or None

        # 2. 多步循环
        for step in range(self.max_steps):
            # 2a. 调 LLM
            messages = self.session.to_llm_format()
            try:
                response: AIMessage = await self.llm.chat(
                    messages, tools=tools_schema, **kwargs
                )
            except LLMError as exc:
                error_msg = friendly_llm_error(exc)
                self.session.add_assistant(error_msg)
                return error_msg

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

    async def run_stream(self, user_input: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式执行一次 Agent 交互，逐个产出回复文本。

        这是 :meth:`run_stream_events` 的文本封装。它按 ``generation_id``
        缓存每轮文本，遇到 ``generation_failed`` 时丢弃该轮半截文本（回退），
        只在 ``generation_end`` / ``done`` / ``error`` 时落定并产出，避免
        中断重跑时新旧内容混合。
        """
        pending: list[str] = []

        async for ev in self.run_stream_events(user_input, **kwargs):
            if ev.type == "text":
                pending.append(ev.text)
            elif ev.type == "generation_failed":
                # 回退：丢弃本轮未完成的半截文本。
                pending.clear()
            elif ev.type == "generation_end":
                for t in pending:
                    yield t
                pending.clear()
            elif ev.type == "error":
                for t in pending:
                    yield t
                pending.clear()
                yield ev.text
            elif ev.type == "done":
                for t in pending:
                    yield t
                pending.clear()

    async def run_stream_events(self, user_input: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """流式执行一次 Agent 交互，产出结构化 :class:`StreamEvent`。

        与 :meth:`run_stream` 的流程一致，但每轮模型生成都用独立的
        ``generation_id`` 标识。若某轮流式输出中途中断且已产出部分内容，
        则标记该轮 ``generation_failed``、回退到本轮开始前的会话状态，并以
        新的 ``generation_id`` 重新请求，避免新旧内容混合。
        """
        # 1. 写入用户消息
        self.session.add_user(user_input)

        tools_schema = self._tool_schemas() or None

        # 2. 多步循环
        for step in range(self.max_steps):
            # 本轮开始前的会话状态（重跑时复用，即"回退点"）。
            messages = self.session.to_llm_format()
            generation_id = ""
            content_parts: list[str] = []
            # index -> {id, name, args_str}
            tc_map: dict[int, dict[str, Any]] = {}

            # 2a. 流式调用 LLM；中途中断则回退重跑。
            attempt = 0
            while True:
                generation_id = new_generation_id()
                content_parts = []
                tc_map = {}

                yield StreamEvent(type="generation_start", generation_id=generation_id)

                try:
                    async for chunk in self.llm.stream(messages, tools=tools_schema, **kwargs):
                        if chunk.content:
                            content_parts.append(chunk.content)
                            yield StreamEvent(type="text", generation_id=generation_id, text=chunk.content)

                        # 累积工具调用 delta（按 index 合并）
                        for tc in chunk.tool_calls or []:
                            idx = tc.get("index", 0)
                            entry = tc_map.setdefault(idx, {"id": "", "name": "", "args_str": ""})
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                entry["name"] = fn["name"]
                            if fn.get("arguments"):
                                entry["args_str"] += fn["arguments"]
                    # 正常结束本轮
                    break
                except LLMError as exc:
                    produced = bool(content_parts) or bool(tc_map)
                    # 已产出部分内容且还有重跑次数 → 标记 failed，回退重跑。
                    if produced and attempt < self.stream_max_retries:
                        yield StreamEvent(
                            type="generation_failed",
                            generation_id=generation_id,
                            data={"reason": friendly_llm_error(exc)},
                        )
                        attempt += 1
                        await asyncio.sleep(min(1.0 * (2 ** attempt), 30.0))
                        continue
                    # 未产出内容 / 重跑耗尽：丢弃半截内容，报错结束。
                    error_msg = f"\n\n{friendly_llm_error(exc)}"
                    self.session.add_assistant(error_msg)
                    yield StreamEvent(type="error", generation_id=generation_id, text=error_msg)
                    return

            # 2b. 没有工具调用 → 最终回复
            if not tc_map:
                self.session.add_assistant("".join(content_parts))
                yield StreamEvent(type="generation_end", generation_id=generation_id)
                yield StreamEvent(type="done", generation_id=generation_id)
                return

            # 2c. 有工具调用 → 拼装成 LangChain 格式
            tool_calls = []
            for idx in sorted(tc_map.keys()):
                entry = tc_map[idx]
                try:
                    args = json.loads(entry["args_str"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "name": entry["name"],
                    "args": args,
                    "id": entry["id"],
                })

            # 2d. 把 assistant 消息（带 tool_calls）写进历史
            self.session.add_assistant("".join(content_parts), tool_calls=tool_calls)
            yield StreamEvent(type="generation_end", generation_id=generation_id)

            # 2e. 逐个执行工具，把结果写回历史
            for tool_call in tool_calls:
                yield StreamEvent(
                    type="tool_call",
                    generation_id=generation_id,
                    data={
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call["id"],
                    },
                )
                result = await self._execute_tool_call(tool_call)
                call_id = tool_call.get("id", "")
                self.session.add_tool(result, tool_call_id=call_id)
                yield StreamEvent(
                    type="tool_result",
                    generation_id=generation_id,
                    data={"id": call_id, "result": result},
                )

            # 2f. 继续循环：LLM 下一轮会看到工具结果

        # 3. 超过最大步数，兜底返回
        fallback = f"已达到最大步数 {self.max_steps}，未能得到最终回复。"
        self.session.add_assistant(fallback)
        yield StreamEvent(type="text", text=fallback)
        yield StreamEvent(type="done")
