"""SkillExtractor：从对话轨迹中提取候选 Skill。

使用 LLM 分析轨迹，判断是否存在可复用的偏好、SOP、工具链或约束。
每次最多提取 1 个高质量候选 Skill，避免噪声入库；一次性请求、临时问题
直接返回 ``None``。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .models import CandidateSkill, ConversationTrace

logger = logging.getLogger(__name__)


# 要求 LLM 输出的 JSON schema 描述
_EXTRACT_PROMPT = """你是一个技能挖掘专家。分析下面的对话轨迹，判断是否存在一个可复用的
「技能（Skill）」——即用户在多轮对话中反复表现出的偏好、标准操作流程（SOP）、
固定工具链、或强约束。

判定规则：
- 必须是「可复用」的：能迁移到未来相似任务，而非仅针对本次的一次性需求。
- 必须有明确的触发条件（什么样的请求应该使用这个技能）。
- 必须有可执行的指令（instruction），告诉 Agent 遇到这类请求该怎么做。
- 一次性问题、闲聊、临时调试 → 不要提取，返回 null。
- 每次最多提取 1 个最有价值的技能。

请严格输出 JSON，格式如下（不要输出其他文字）：
{
  "name": "skill_name（英文小写，下划线分隔）",
  "description": "一句话描述技能用途",
  "trigger_conditions": ["触发条件1", "触发条件2"],
  "instruction": "技能的完整执行指令，告诉 Agent 该怎么做",
  "tools": ["需要用到的工具名，如 read_file"],
  "examples": ["示例输入1", "示例输入2"],
  "confidence_score": 0.0 到 1.0 之间的浮点数
}

如果没有值得提取的技能，输出：null
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class SkillExtractor:
    """从对话轨迹提取候选 Skill。"""

    def __init__(self, llm: Any, *, confidence_threshold: float = 0.6) -> None:
        #: LLM 提供者，需实现 ``async chat(messages) -> AIMessage``。
        self._llm = llm
        #: 置信度低于该阈值的候选直接丢弃。
        self._confidence_threshold = confidence_threshold

    async def extract(self, trace: ConversationTrace) -> CandidateSkill | None:
        """分析轨迹，返回候选 Skill 或 None。"""
        prompt = self._build_prompt(trace)
        try:
            reply = await self._llm.chat([
                SystemMessage(content=_EXTRACT_PROMPT),
                HumanMessage(content=prompt),
            ])
        except Exception:  # noqa: BLE001
            logger.exception("SkillExtractor 调用 LLM 失败，trace_id=%s", trace.trace_id)
            return None

        text = (reply.content or "").strip()
        if not text or text.lower() == "null":
            return None

        data = self._parse_json(text)
        if data is None:
            logger.warning("SkillExtractor 无法解析 LLM 输出，trace_id=%s", trace.trace_id)
            return None

        try:
            candidate = CandidateSkill(
                name=data.get("name", ""),
                description=data.get("description", ""),
                trigger_conditions=data.get("trigger_conditions", []),
                instruction=data.get("instruction", ""),
                tools=data.get("tools", []),
                examples=data.get("examples", []),
                source_trace_id=trace.trace_id,
                confidence_score=float(data.get("confidence_score", 0.0)),
            )
        except Exception:  # noqa: BLE001
            logger.exception("SkillExtractor 候选 Skill 校验失败，trace_id=%s", trace.trace_id)
            return None

        if not candidate.name or not candidate.instruction:
            return None
        if candidate.confidence_score < self._confidence_threshold:
            logger.info(
                "SkillExtractor 候选置信度 %.2f 低于阈值 %.2f，丢弃：%s",
                candidate.confidence_score, self._confidence_threshold, candidate.name,
            )
            return None

        return candidate

    # -- 内部 --------------------------------------------------------------

    def _build_prompt(self, trace: ConversationTrace) -> str:
        parts = [f"用户原始请求：{trace.user_query}"]
        if trace.messages:
            msgs = "\n".join(
                f"[{m.get('role', '?')}] {str(m.get('content', ''))[:500]}"
                for m in trace.messages
            )
            parts.append(f"对话消息：\n{msgs}")
        if trace.tool_calls:
            parts.append(f"工具调用：{json.dumps(trace.tool_calls, ensure_ascii=False)[:1000]}")
        if trace.user_feedback:
            parts.append(f"用户反馈：{trace.user_feedback}")
        if trace.final_outcome:
            parts.append(f"最终结果：{trace.final_outcome}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """从 LLM 输出中提取 JSON 对象。"""
        text = text.strip()
        # 优先尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 提取 ```json ... ``` 代码块
        if "```" in text:
            block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if block:
                try:
                    return json.loads(block.group(1).strip())
                except json.JSONDecodeError:
                    pass
        # 兜底：取第一个 { ... } 块
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None
