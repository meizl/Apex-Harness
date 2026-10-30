"""LLM 插件写法示例 + 测试。

这个文件演示如何写一个完整的 LLM 插件，并通过 Kernel 挂载、启动、调用。
为了不依赖真实网络，LLM 的 HTTP 调用层用一个 fake 函数模拟。

阅读顺序：
1. FakeOpenAIClient   —— 模拟的 HTTP 客户端（生产环境换成真实 requests/openai SDK）
2. OpenAILLMPlugin    —— 插件本体，重点看 setup / start / chat
3. 测试用例            —— 看 Kernel 怎么把插件装起来用
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)

from apex_harness.core.config import Config
from apex_harness.core.kernel import Kernel
from apex_harness.llm.base import LLMProvider


# ============================================================================
# 1. 模拟的 HTTP 客户端
# ============================================================================
# 真实环境里这里会用 openai SDK 或 httpx 发请求。
# 这里用一个假的，根据输入返回固定回复，方便测试。

class FakeOpenAIClient:
    """假的 OpenAI 客户端，不发网络请求。"""

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        # 记录最后一次调用的参数，测试里可以断言
        self.last_request: dict[str, Any] | None = None

    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """模拟调用 /chat/completions 接口。"""
        self.last_request = {
            "messages": messages,
            "model": model,
            "tools": tools,
            **kwargs,
        }
        # 假装模型回了一句话
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"[fake-{model}] 收到 {len(messages)} 条消息",
                    }
                }
            ]
        }


# ============================================================================
# 2. LLM 插件本体
# ============================================================================

class OpenAILLMPlugin(LLMProvider):
    """OpenAI 兼容接口的 LLM 插件。

    演示重点：
    - name / provides 声明身份
    - setup() 里注册服务
    - start() 里读配置、建客户端（此时不能在 setup 里建，因为配置在 ctx 里）
    - chat() 实现业务逻辑：LangChain 消息 → dict → 调 API → AIMessage
    """

    # --- 插件声明 ---
    name = "openai_llm"
    description = "OpenAI 兼容接口的 LLM 提供者"
    provides = ["llm"]          # 我提供名叫 "llm" 的服务
    # requires 留空：这个插件不依赖其他插件的服务

    # --- 模型标识（LLMProvider 要求的类属性） ---
    model = "gpt-4o-mini"

    # --- 生命周期 ---
    
    async def setup(self) -> None:
        """挂载时调用。只做一件事：把自己注册成 "llm" 服务。"""
        # 此时不能读配置（ctx 有了，但约定上 setup 只注册不消费）
        self.ctx.register_service("llm", self)

    async def start(self) -> None:
        """所有插件都 setup 完后调用。在这里读配置、建客户端。"""
        # ctx.config 是 Kernel 按插件名分好的配置切片
        cfg = self.ctx.config
        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url", "https://api.openai.com/v1")

        # 用 fake 客户端；生产环境换成真实客户端
        self._client = FakeOpenAIClient(api_key=api_key, base_url=base_url)

    async def stop(self) -> None:
        """关停时调用。这里可以关闭 HTTP 连接池。"""
        # fake 客户端不需要关，留空
        pass

    # --- 业务方法 ---

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """LangChain 消息列表 → dict → 调 API → 包成 AIMessage 返回。"""
        # 1. LangChain 消息转成 OpenAI 要的 dict 格式
        #    messages_to_dict 会输出 {"type": "human", "data": {...}}，
        #    需要再拍平成 {"role": "user", "content": "..."}
        openai_messages = [_to_openai_format(m) for m in messages]

        # 2. 调底层客户端
        response = await self._client.chat_completions(
            openai_messages,
            model=self.model,
            tools=tools,
            **kwargs,
        )

        # 3. 解析响应，包成 AIMessage
        choice = response["choices"][0]["message"]
        content = choice.get("content", "")
        tool_calls = choice.get("tool_calls")  # 真实接口会有，fake 没有
        return AIMessage(content=content, tool_calls=tool_calls or [])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """嵌入接口，这里不实现。"""
        raise NotImplementedError("这个示例不支持 embed")


def _to_openai_format(message: BaseMessage) -> dict[str, Any]:
    """把 LangChain 消息转成 OpenAI 的 {role, content} 格式。"""
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }
    role = role_map.get(message.type, "user")
    result: dict[str, Any] = {"role": role, "content": message.content}

    # assistant 消息可能带 tool_calls，要透传
    # LangChain 的 tool_calls 带 "type": "tool_call" 内部字段，OpenAI 不需要，去掉
    if message.type == "ai" and getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {k: v for k, v in tc.items() if k != "type"}
            for tc in message.tool_calls
        ]

    # tool 消息需要 tool_call_id
    if message.type == "tool":
        result["tool_call_id"] = getattr(message, "tool_call_id", "")

    return result


# ============================================================================
# 3. 测试用例
# ============================================================================

@pytest.fixture
def kernel_with_llm() -> Kernel:
    """启动一个挂了 OpenAILLMPlugin 的 Kernel。"""
    config = Config({
        "openai_llm": {
            "api_key": "sk-test-123",
            "base_url": "https://api.openai.com/v1",
        },
    })
    kernel = Kernel(config=config.to_dict())
    return kernel


async def test_plugin_registers_llm_service(kernel_with_llm: Kernel) -> None:
    """setup 之后，服务注册表里应该有 "llm"。"""
    kernel = kernel_with_llm
    await kernel.mount_all([OpenAILLMPlugin()])

    # 直接查服务注册表
    assert kernel.services.has("llm")

    llm = kernel.services.require("llm")
    assert isinstance(llm, OpenAILLMPlugin)


async def test_plugin_start_reads_config(kernel_with_llm: Kernel) -> None:
    """start 之后，插件应该读到配置并建好客户端。"""
    kernel = kernel_with_llm
    plugin = OpenAILLMPlugin()
    await kernel.mount_all([plugin])
    await kernel.start()

    # start 里建的 fake 客户端应该拿到了配置里的 api_key
    assert plugin._client.api_key == "sk-test-123"
    assert plugin._client.base_url == "https://api.openai.com/v1"


async def test_chat_via_kernel(kernel_with_llm: Kernel) -> None:
    """通过 Kernel 拿到 "llm" 服务，调用 chat，验证完整链路。"""
    kernel = kernel_with_llm
    plugin = OpenAILLMPlugin()
    await kernel.mount_all([plugin])
    await kernel.start()

    # 模拟一个使用方：从服务注册表拿 "llm"
    llm = kernel.services.require("llm")

    messages: list[BaseMessage] = [
        SystemMessage(content="你是一个助手"),
        HumanMessage(content="你好"),
    ]
    response = await llm.chat(messages)

    # 返回的是 AIMessage
    assert isinstance(response, AIMessage)
    assert response.content == "[fake-gpt-4o-mini] 收到 2 条消息"

    # 底层客户端收到的是 OpenAI 格式的 dict
    last = plugin._client.last_request
    assert last is not None
    assert last["model"] == "gpt-4o-mini"
    assert last["messages"] == [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"},
    ]


async def test_chat_with_tool_calls(kernel_with_llm: Kernel) -> None:
    """验证带 tool_calls 的 assistant 消息能正确转换。"""
    kernel = kernel_with_llm
    plugin = OpenAILLMPlugin()
    await kernel.mount_all([plugin])
    await kernel.start()

    llm = kernel.services.require("llm")

    messages: list[BaseMessage] = [
        HumanMessage(content="读一下文件"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}],
        ),
        ToolMessage(content="文件内容", tool_call_id="c1"),
    ]
    await llm.chat(messages)

    sent = plugin._client.last_request["messages"]
    # assistant 消息的 tool_calls 应该被透传
    assert sent[1]["role"] == "assistant"
    assert sent[1]["tool_calls"] == [
        {"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}
    ]
    # tool 消息的 tool_call_id 应该被透传
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "c1"


async def test_lifecycle_cleanup(kernel_with_llm: Kernel) -> None:
    """shutdown 之后，服务应该被自动注销。"""
    kernel = kernel_with_llm
    await kernel.mount_all([OpenAILLMPlugin()])
    await kernel.start()

    assert kernel.services.has("llm")

    await kernel.shutdown()

    # Context.dispose() 自动清理了注册的服务
    assert not kernel.services.has("llm")
