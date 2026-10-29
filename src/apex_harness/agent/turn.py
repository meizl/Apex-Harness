"""单轮执行抽象：一次模型调用（不含工具调用）。"""

from __future__ import annotations

from typing import Any

from apex_harness.llm.base import LLMProvider
from apex_harness.session.session import Session


class Turn:
    """一次对话轮次。

    一个 Turn 完成"用户输入 → 调 LLM → 助手回复"的最小流程。
    工具调用等更复杂的逻辑由上层（Orchestrator）处理。
    """

    def __init__(self, session: Session, llm: LLMProvider) -> None:
        #: 当前会话，记录消息历史。
        self.session = session
        #: LLM 提供者，用于生成回复。
        self.llm = llm

    async def run(self, user_input: str, **kwargs: Any) -> str:
        """执行一轮对话，返回助手回复文本。

        流程：
        1. 把用户输入写入会话历史；
        2. 把历史交给 LLM 生成回复；
        3. 把回复写回会话历史；
        4. 返回回复文本。
        """
        # 1. 记录用户消息
        self.session.add_user(user_input)

        # 2. 调 LLM 生成回复
        messages = self.session.to_llm_format()
        response = await self.llm.chat(messages, **kwargs)
        reply = response.content

        # 3. 记录助手消息并返回
        self.session.add_assistant(reply)
        return reply
