"""Sub-agent：可被主 Agent 异步委托的专职 Agent。

设计模型（方案 B：异步委托 + 状态查询）::

    主 Agent                          SubAgentInstance（后台任务）
    ─────────                         ─────────────────────────────
    start_subagent(name, task)  ────► 创建实例，后台运行 AgentLoop
    返回 agent_id                     │
                                      ├─ 产生的 StreamEvent 放入 event_queue
                                      ├─ 主 Agent 可随时 send_subagent_message
                                      │   → 消息进入 input_queue，下一轮 LLM 读取
                                      └─ 完成后 result = 最终文本，event_queue 放 None
    get_subagent_result(agent_id) ───► 阻塞等待，转发 event_queue 中的事件，返回 result

核心组件：
- :class:`SubAgent`：子 Agent 定义（name / description / system_prompt / 工具权限 / 超时）
- :class:`SubAgentInstance`：一次运行中的子 Agent 实例（后台任务 + 输入队列 + 事件队列）
- :class:`SubAgentRegistry`：子 Agent 定义表 + 运行中实例表
- 四个工具：``start_subagent`` / ``send_subagent_message`` / ``get_subagent_result`` / ``cancel_subagent``
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from apex_harness.agent.events import StreamEvent
from apex_harness.agent.loop import AgentLoop
from apex_harness.core.plugin import Plugin
from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session
from apex_harness.tools.base import Tool

logger = logging.getLogger(__name__)

# 事件队列的结束标记：子 Agent 完成后放入 None 表示事件流结束。
_DONE_SENTINEL = None


@dataclass
class SubAgent:
    """一个可被委托的专职 Agent 定义。

    Attributes:
        name: 唯一标识。
        description: 子 Agent 能做什么，展示给主 Agent 决策。
        system_prompt: 子 Agent 专属系统提示词。
        allowed_tools: 可用工具名列表；``None`` 表示继承主 Agent 全部工具
            （自动排除委托工具自身，防止递归）。
        max_steps: 子 Agent ReAct 循环最大步数。
        timeout: 墙钟超时（秒）；``None`` 表示不限制。超时后子 Agent 被取消，
            返回超时提示。
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None = None
    max_steps: int = 10
    timeout: float | None = None


class SubAgentInstance:
    """一次运行中的子 Agent 实例。

    生命周期::

        instance = SubAgentInstance(...)
        instance.start(initial_task)   # 后台运行
        ...
        instance.send_message("补充信息")  # 随时可发
        result = await instance.wait_result()  # 阻塞等待
        # 或 instance.cancel()
    """

    def __init__(
        self,
        agent: SubAgent,
        llm: LLMProvider,
        tools: dict[str, Tool],
    ) -> None:
        self.agent_id: str = uuid.uuid4().hex[:12]
        self.agent = agent
        self.session = Session()
        self.session.add_system(agent.system_prompt)
        self.loop = AgentLoop(
            self.session, llm, tools=tools, max_steps=agent.max_steps
        )
        #: 主 Agent 发来的消息队列，子 Agent 每轮 LLM 后检查。
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        #: 子 Agent 产生的事件队列，供转发给主事件流。
        self.event_queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self.result: str | None = None
        self._task: asyncio.Task | None = None
        self._finished: asyncio.Event = asyncio.Event()

    # -- 生命周期 --------------------------------------------------------

    def start(self, initial_task: str) -> None:
        """在后台启动子 Agent，立即返回。"""
        self._task = asyncio.create_task(self._run(initial_task))

    async def send_message(self, message: str) -> None:
        """向运行中的子 Agent 发送一条消息。

        消息会在下一轮 LLM 调用时作为用户消息注入会话。若子 Agent 已完成，
        消息被静默丢弃。
        """
        if self._finished.is_set():
            logger.warning("Sub-agent %s 已完成，消息被丢弃", self.agent_id)
            return
        await self.input_queue.put(message)

    async def wait_result(self) -> str:
        """阻塞等待子 Agent 完成，返回最终结果文本。"""
        await self._finished.wait()
        return self.result or ""

    async def cancel(self) -> None:
        """取消运行中的子 Agent。"""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if not self._finished.is_set():
            self.result = f"子 Agent '{self.agent.name}' 被取消"
            await self.event_queue.put(_DONE_SENTINEL)
            self._finished.set()

    # -- 内部主循环 ------------------------------------------------------

    async def _run(self, initial_task: str) -> None:
        """子 Agent 主循环：执行任务 → 检查输入队列 → 有消息则继续，无则结束。"""
        try:
            if self.agent.timeout:
                await asyncio.wait_for(
                    self._run_loop(initial_task), timeout=self.agent.timeout
                )
            else:
                await self._run_loop(initial_task)
        except asyncio.TimeoutError:
            self.result = (
                f"子 Agent '{self.agent.name}' 执行超时（{self.agent.timeout}s）"
            )
            logger.warning("Sub-agent %s 超时", self.agent_id)
        except asyncio.CancelledError:
            self.result = f"子 Agent '{self.agent.name}' 被取消"
            raise
        except Exception as exc:  # noqa: BLE001
            self.result = f"子 Agent '{self.agent.name}' 执行出错: {exc}"
            logger.exception("Sub-agent %s 异常", self.agent_id)
        finally:
            await self.event_queue.put(_DONE_SENTINEL)
            self._finished.set()

    async def _run_loop(self, initial_task: str) -> None:
        """实际的执行循环（不含超时包装，便于 wait_for 包裹）。"""
        current_input = initial_task
        while True:
            # 运行一轮 ReAct 循环（可能多步工具调用），产出流式事件。
            text_parts: list[str] = []
            async for ev in self.loop.run_stream_events(current_input):
                await self.event_queue.put(ev)
                if ev.type == "text" and ev.text:
                    text_parts.append(ev.text)

            # 本轮结束。检查是否有主 Agent 发来的新消息。
            try:
                current_input = self.input_queue.get_nowait()
                logger.info("Sub-agent %s 收到新消息，继续执行", self.agent_id)
            except asyncio.QueueEmpty:
                # 没有新消息，子 Agent 完成
                self.result = "".join(text_parts).strip() or "（子 Agent 未产出文本）"
                break


class SubAgentRegistry:
    """子 Agent 定义表 + 运行中实例表。"""

    def __init__(self) -> None:
        self._agents: dict[str, SubAgent] = {}
        self._instances: dict[str, SubAgentInstance] = {}

    # -- 定义管理 --------------------------------------------------------

    def register(self, agent: SubAgent) -> None:
        if not agent.name:
            raise ValueError("SubAgent 缺少 name")
        self._agents[agent.name] = agent

    def get(self, name: str) -> SubAgent | None:
        return self._agents.get(name)

    @property
    def all(self) -> dict[str, SubAgent]:
        return dict(self._agents)

    @property
    def names(self) -> list[str]:
        return list(self._agents.keys())

    def catalog_text(self) -> str:
        if not self._agents:
            return "（暂无可用子 Agent）"
        lines = []
        for name, agent in self._agents.items():
            tools = agent.allowed_tools or "全部工具"
            lines.append(f"- `{name}`：{agent.description}（可用工具：{tools}）")
        return "\n".join(lines)

    # -- 实例管理 --------------------------------------------------------

    def start_instance(
        self, name: str, task: str, llm: LLMProvider, all_tools: dict[str, Tool]
    ) -> str:
        """启动一个子 Agent 实例，返回 agent_id。"""
        agent = self._agents.get(name)
        if agent is None:
            available = ", ".join(self._agents) or "无"
            raise KeyError(f"未知子 Agent '{name}'，可用：{available}")

        tools = self._build_tools(agent, all_tools)
        instance = SubAgentInstance(agent, llm, tools)
        self._instances[instance.agent_id] = instance
        instance.start(task)
        logger.info("Sub-agent '%s' 实例 %s 已启动", name, instance.agent_id)
        return instance.agent_id

    async def send_message(self, agent_id: str, message: str) -> str:
        instance = self._instances.get(agent_id)
        if instance is None:
            return f"Error: 子 Agent 实例 '{agent_id}' 不存在"
        await instance.send_message(message)
        return f"消息已发送给子 Agent '{instance.agent.name}'"

    async def get_result(self, agent_id: str) -> tuple[str, SubAgentInstance]:
        """等待实例完成并返回 (result, instance)。"""
        instance = self._instances.get(agent_id)
        if instance is None:
            raise KeyError(f"子 Agent 实例 '{agent_id}' 不存在")
        result = await instance.wait_result()
        # 结果取出后清理实例
        self._instances.pop(agent_id, None)
        return result, instance

    async def cancel_instance(self, agent_id: str) -> str:
        instance = self._instances.get(agent_id)
        if instance is None:
            return f"Error: 子 Agent 实例 '{agent_id}' 不存在"
        await instance.cancel()
        self._instances.pop(agent_id, None)
        return f"子 Agent '{instance.agent.name}' 已取消"

    def get_instance(self, agent_id: str) -> SubAgentInstance | None:
        return self._instances.get(agent_id)

    @staticmethod
    def _build_tools(
        agent: SubAgent, all_tools: dict[str, Tool]
    ) -> dict[str, Tool]:
        """构建子 Agent 的工具集，排除委托工具自身防递归。"""
        delegate_names = {
            "start_subagent",
            "send_subagent_message",
            "get_subagent_result",
            "cancel_subagent",
        }
        if agent.allowed_tools is None:
            return {
                n: t for n, t in all_tools.items() if n not in delegate_names
            }
        return {
            n: all_tools[n]
            for n in agent.allowed_tools
            if n in all_tools and n not in delegate_names
        }


# ---------------------------------------------------------------------------
# 四个委托工具
# ---------------------------------------------------------------------------


class StartSubAgentTool(Tool):
    """启动一个子 Agent，返回 agent_id。立即返回，不阻塞。"""

    name = "start_subagent"
    description = "启动一个专职子 Agent 执行任务，立即返回 agent_id（不阻塞）"
    requires = ["tools", "llm", "subagents"]
    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "要启动的子 Agent 名称",
            },
            "task": {
                "type": "string",
                "description": "交给子 Agent 的初始任务",
            },
        },
        "required": ["agent_name", "task"],
    }

    def __init__(
        self,
        registry: SubAgentRegistry,
        llm: LLMProvider,
        all_tools: dict[str, Tool],
    ) -> None:
        super().__init__()
        self._registry = registry
        self._llm = llm
        self._all_tools = all_tools

    async def setup(self) -> None:
        catalog = self._registry.catalog_text()
        self.description = (
            "启动一个专职子 Agent 执行任务，立即返回 agent_id（不阻塞）。\n\n"
            f"可用子 Agent：\n{catalog}\n\n"
            "用法：先调用本工具拿到 agent_id，之后可用 send_subagent_message "
            "补充信息，最后用 get_subagent_result 获取结果。"
        )
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, agent_name: str, task: str, **kwargs: Any) -> str:
        try:
            agent_id = self._registry.start_instance(
                agent_name, task, self._llm, self._all_tools
            )
        except KeyError as exc:
            return str(exc)
        return f"子 Agent 已启动，agent_id={agent_id}。可用 send_subagent_message 补充信息，或用 get_subagent_result 获取结果。"


class SendSubAgentMessageTool(Tool):
    """向运行中的子 Agent 发送补充消息。"""

    name = "send_subagent_message"
    description = "向运行中的子 Agent 发送一条补充消息"
    requires = ["tools", "subagents"]
    parameters = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "start_subagent 返回的 agent_id",
            },
            "message": {
                "type": "string",
                "description": "要发送给子 Agent 的补充消息",
            },
        },
        "required": ["agent_id", "message"],
    }

    def __init__(self, registry: SubAgentRegistry) -> None:
        super().__init__()
        self._registry = registry

    async def setup(self) -> None:
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, agent_id: str, message: str, **kwargs: Any) -> str:
        return await self._registry.send_message(agent_id, message)


class GetSubAgentResultTool(Tool):
    """等待子 Agent 完成并获取结果，同时转发子 Agent 的流式事件。"""

    name = "get_subagent_result"
    description = "等待子 Agent 执行完成并获取最终结果（会阻塞直到完成）"
    requires = ["tools", "subagents"]
    parameters = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "start_subagent 返回的 agent_id",
            },
        },
        "required": ["agent_id"],
    }

    def __init__(self, registry: SubAgentRegistry) -> None:
        super().__init__()
        self._registry = registry

    async def setup(self) -> None:
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(
        self,
        agent_id: str,
        *,
        event_queue: asyncio.Queue[StreamEvent] | None = None,
        **kwargs: Any,
    ) -> str:
        instance = self._registry.get_instance(agent_id)
        if instance is None:
            return f"Error: 子 Agent 实例 '{agent_id}' 不存在或已被清理"

        # 转发子 Agent 的事件到主事件流（直到收到结束标记）
        if event_queue is not None:
            while True:
                ev = await instance.event_queue.get()
                if ev is _DONE_SENTINEL:
                    break
                await event_queue.put(ev)

        result, _ = await self._registry.get_result(agent_id)
        return result


class CancelSubAgentTool(Tool):
    """取消运行中的子 Agent。"""

    name = "cancel_subagent"
    description = "取消一个正在运行的子 Agent"
    requires = ["tools", "subagents"]
    parameters = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "start_subagent 返回的 agent_id",
            },
        },
        "required": ["agent_id"],
    }

    def __init__(self, registry: SubAgentRegistry) -> None:
        super().__init__()
        self._registry = registry

    async def setup(self) -> None:
        registry = self.ctx.require_service("tools")
        registry.register(self)

    async def run(self, agent_id: str, **kwargs: Any) -> str:
        return await self._registry.cancel_instance(agent_id)


class SubAgentRegistryPlugin(Plugin):
    """子 Agent 注册表插件：提供 ``subagents`` 服务，并注册四个委托工具。"""

    name = "subagent_registry"
    description = "子 Agent 注册表与异步委托工具"
    provides = ["subagents"]
    requires = ["tools", "llm"]

    def __init__(self, subagents: list[SubAgent] | None = None) -> None:
        super().__init__()
        self._initial_subagents = subagents or []

    async def setup(self) -> None:
        self.registry = SubAgentRegistry()
        for agent in self._initial_subagents:
            self.registry.register(agent)

        llm = self.ctx.require_service("llm")
        tool_registry = self.ctx.require_service("tools")
        all_tools = tool_registry.all

        # 注册四个委托工具
        self._tools = [
            StartSubAgentTool(self.registry, llm, all_tools),
            SendSubAgentMessageTool(self.registry),
            GetSubAgentResultTool(self.registry),
            CancelSubAgentTool(self.registry),
        ]
        for tool in self._tools:
            tool._bind(self.ctx)
            await tool.setup()

        self.ctx.register_service("subagents", self.registry)


# ---------------------------------------------------------------------------
# 内置子 Agent
# ---------------------------------------------------------------------------


def builtin_subagents() -> list[SubAgent]:
    """框架自带的子 Agent 列表。"""
    return [
        SubAgent(
            name="code_reviewer",
            description="对代码进行安全、性能、可维护性审查，输出结构化审查报告",
            system_prompt=(
                "你是一位资深代码审查专家。当用户要求审查代码时，请按以下维度分析：\n"
                "1. 安全性：是否存在注入、越权、硬编码密钥、不安全的反序列化等风险\n"
                "2. 性能：是否有明显的低效写法（循环内重复查询、不必要的深拷贝等）\n"
                "3. 可维护性：命名是否清晰、函数是否过长、是否有重复代码\n"
                "4. 正确性：边界条件、错误处理是否完善\n\n"
                "输出格式：\n"
                "## 审查结果\n\n"
                "### 问题 N：<标题>\n"
                "- 位置：<文件:行号>\n"
                "- 严重程度：高/中/低\n"
                "- 说明：<问题描述>\n"
                "- 建议：<改进方案>\n\n"
                "如果代码整体质量良好，先给出肯定，再列出可优化的小问题。"
            ),
            allowed_tools=["read_file", "list_dir"],
            max_steps=8,
            timeout=60.0,
        ),
        SubAgent(
            name="researcher",
            description="联网搜索并综合信息，回答需要调研的问题",
            system_prompt=(
                "你是一位信息调研专家。用户会给你一个需要调研的问题，你的任务是：\n"
                "1. 使用 web_fetch 工具抓取相关网页内容\n"
                "2. 综合多个来源的信息\n"
                "3. 给出有据可查的回答，并注明信息来源\n\n"
                "注意：不要编造信息，如果某个说法没有来源支撑，明确说明。"
            ),
            allowed_tools=["web_fetch"],
            max_steps=6,
            timeout=90.0,
        ),
        SubAgent(
            name="file_summarizer",
            description="读取并总结一个或多个文件的内容，输出结构化摘要",
            system_prompt=(
                "你是一位文档总结专家。用户会给你文件路径，你的任务是：\n"
                "1. 使用 read_file 读取文件内容\n"
                "2. 如果需要了解目录结构，使用 list_dir\n"
                "3. 输出文件的结构化摘要，包含：核心目的、关键模块/函数、重要结论\n\n"
                "摘要要简洁、有结构，不要逐行复述。"
            ),
            allowed_tools=["read_file", "list_dir"],
            max_steps=6,
            timeout=45.0,
        ),
    ]
