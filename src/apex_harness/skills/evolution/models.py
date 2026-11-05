"""Skill 进化 Worker 的数据模型。

使用 Pydantic v2 定义，所有模型可 JSON 序列化，便于持久化与跨进程传递。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 对话轨迹
# --------------------------------------------------------------------------- #


class ConversationTrace(BaseModel):
    """一段完整对话的轨迹，作为 Skill 进化的输入。"""

    trace_id: str
    user_query: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    user_feedback: str = ""
    final_outcome: str = ""


# --------------------------------------------------------------------------- #
# 候选 Skill
# --------------------------------------------------------------------------- #


class CandidateSkill(BaseModel):
    """从轨迹中提取的候选 Skill（尚未入库）。"""

    name: str
    description: str
    trigger_conditions: list[str] = Field(default_factory=list)
    instruction: str
    tools: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    source_trace_id: str
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)


# --------------------------------------------------------------------------- #
# SkillBank 中的条目
# --------------------------------------------------------------------------- #


class SkillEntry(BaseModel):
    """SkillBank 中已持久化的 Skill 条目，带版本与统计信息。"""

    skill_id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    instruction: str = ""
    trigger_conditions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    changelog: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    success_rate: float = 0.0


# --------------------------------------------------------------------------- #
# 进化决策
# --------------------------------------------------------------------------- #


class EvolutionAction(str, Enum):
    """进化动作。"""

    ADD = "add"
    MERGE = "merge"
    DISCARD = "discard"


class EvolutionDecision(BaseModel):
    """对候选 Skill 的进化决策。"""

    action: EvolutionAction
    reason: str = ""
    target_skill_id: Optional[str] = None
    candidate_skill: CandidateSkill


class EvolutionResult(BaseModel):
    """单次进化流程的最终结果。"""

    trace_id: str
    decision: EvolutionDecision
    skill_entry: Optional[SkillEntry] = None
    similarity_scores: list[float] = Field(default_factory=list)
