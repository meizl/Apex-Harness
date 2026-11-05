"""SkillBankWriter：根据进化决策写入 SkillBank。

- Add：创建新 SkillEntry，初始版本 v0.1.0，分配 skill_id。
- Merge：用 SkillMerger 合并后更新已有条目，版本递增。
- Discard：不写入，仅记录日志。

所有写入通过 SkillBank 的 append-only JSONL 保证幂等与可回溯。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from .bank import SkillBank
from .merger import SkillMerger
from .models import CandidateSkill, EvolutionAction, EvolutionDecision, SkillEntry

logger = logging.getLogger(__name__)


def _new_skill_id() -> str:
    return f"skill_{uuid.uuid4().hex[:12]}"


class SkillBankWriter:
    """技能库写入器。"""

    def __init__(self, bank: SkillBank, *, merger: SkillMerger | None = None) -> None:
        self._bank = bank
        self._merger = merger or SkillMerger()

    async def write(self, decision: EvolutionDecision) -> SkillEntry | None:
        """根据决策执行写入，返回最终的 SkillEntry（Discard 返回 None）。"""
        if decision.action == EvolutionAction.ADD:
            return self._add(decision.candidate_skill, decision.reason)
        if decision.action == EvolutionAction.MERGE:
            return self._merge(decision)
        # DISCARD
        logger.info(
            "Skill 进化 Discard：%s，原因：%s",
            decision.candidate_skill.name, decision.reason,
        )
        return None

    # -- Add ---------------------------------------------------------------

    def _add(self, candidate: CandidateSkill, reason: str) -> SkillEntry:
        # 幂等：同名技能已存在则不重复创建
        existing = self._bank.get_by_name(candidate.name)
        if existing is not None:
            logger.info("技能「%s」已存在，跳过 Add", candidate.name)
            return existing

        entry = SkillEntry(
            skill_id=_new_skill_id(),
            name=candidate.name,
            version="0.1.0",
            description=candidate.description,
            instruction=candidate.instruction,
            trigger_conditions=list(candidate.trigger_conditions),
            tools=list(candidate.tools),
            examples=list(candidate.examples),
            changelog=[
                f"v0.1.0: 新增技能（来源 trace_id={candidate.source_trace_id}）"
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self._bank.put(entry)
        logger.info("技能 Add 成功：%s (%s)", entry.name, entry.skill_id)
        return entry

    # -- Merge -------------------------------------------------------------

    def _merge(self, decision: EvolutionDecision) -> SkillEntry | None:
        target_id = decision.target_skill_id
        if not target_id:
            logger.warning("Merge 决策缺少 target_skill_id，降级为 Discard")
            return None
        existing = self._bank.get(target_id)
        if existing is None:
            logger.warning("Merge 目标技能不存在：%s，降级为 Discard", target_id)
            return None

        merged = self._merger.merge(existing, decision.candidate_skill, decision.reason)
        self._bank.put(merged)
        logger.info(
            "技能 Merge 成功：%s %s → %s",
            merged.name, existing.version, merged.version,
        )
        return merged
