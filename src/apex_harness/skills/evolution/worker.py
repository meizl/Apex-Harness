"""SkillEvolutionWorker：右循环的异步进化主流程。

编排链路::

    ConversationTrace
        → SkillExtractor.extract()        提取候选 Skill
        → SkillBankRetriever.retrieve()   检索相似技能
        → EvolutionDecider.decide()       决策 Add/Merge/Discard
        → [Add/Merge] 进入待审批队列，等待用户审批
        → [Discard]   直接丢弃

整个流程异步执行，不阻塞主对话（左循环）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .bank import SkillBank
from .decider import EvolutionDecider
from .extractor import SkillExtractor
from .models import (
    ConversationTrace,
    EvolutionAction,
    EvolutionDecision,
    EvolutionResult,
    PendingSkill,
)
from .pending_store import PendingSkillStore
from .retriever import Embedder, SkillBankRetriever
from .writer import SkillBankWriter

logger = logging.getLogger(__name__)


class SkillEvolutionWorker:
    """技能进化 Worker（右循环）。"""

    def __init__(
        self,
        llm: Any,
        bank: SkillBank,
        pending_store: PendingSkillStore,
        *,
        embedder: Embedder | None = None,
        top_k: int = 3,
        similarity_threshold: float = 0.4,
        confidence_threshold: float = 0.6,
    ) -> None:
        self._bank = bank
        self._pending_store = pending_store
        self._extractor = SkillExtractor(
            llm, confidence_threshold=confidence_threshold
        )
        self._retriever = SkillBankRetriever(
            bank, embedder=embedder, top_k=top_k, threshold=similarity_threshold
        )
        self._decider = EvolutionDecider()
        self._writer = SkillBankWriter(bank)

    async def run_evolution(self, trace: ConversationTrace) -> EvolutionResult:
        """对一段对话轨迹执行一次技能进化，返回结果。

        Add/Merge 决策不直接写入 SkillBank，而是放入待审批队列。
        """
        # 1. 提取候选 Skill
        candidate = await self._extractor.extract(trace)
        if candidate is None:
            logger.info("trace %s 未提取到候选 Skill", trace.trace_id)
            return EvolutionResult(
                trace_id=trace.trace_id,
                decision=EvolutionDecision(
                    action=EvolutionAction.DISCARD,
                    reason="未从轨迹中提取到候选 Skill",
                    candidate_skill=_placeholder_candidate(trace),
                ),
                similarity_scores=[],
            )

        # 2. 检索相似技能
        similar = await self._retriever.retrieve(candidate)
        scores = [s for _, s in similar]

        # 3. 决策
        decision = self._decider.decide(candidate, similar)

        # 4. Discard 直接丢弃；Add/Merge 进入待审批队列
        if decision.action == EvolutionAction.DISCARD:
            logger.info(
                "trace %s 候选 %s 被丢弃：%s",
                trace.trace_id,
                candidate.name,
                decision.reason,
            )
            return EvolutionResult(
                trace_id=trace.trace_id,
                decision=decision,
                skill_entry=None,
                similarity_scores=scores,
            )

        pending = self._pending_store.add(candidate, decision)
        logger.info(
            "trace %s 候选 %s 进入待审批队列（%s）",
            trace.trace_id,
            candidate.name,
            decision.action.value,
        )

        return EvolutionResult(
            trace_id=trace.trace_id,
            decision=decision,
            skill_entry=None,
            similarity_scores=scores,
            pending_skill=pending,
        )

    async def approve(self, pending_id: str) -> Optional[EvolutionResult]:
        """用户审批通过：将候选写入 SkillBank。"""
        pending = self._pending_store.approve(pending_id)
        if pending is None:
            return None

        # 用（可能被用户修改过的）candidate 重建 decision
        decision = pending.decision.model_copy(
            update={"candidate_skill": pending.candidate}
        )
        skill_entry = await self._writer.write(decision)

        return EvolutionResult(
            trace_id=pending.candidate.source_trace_id,
            decision=decision,
            skill_entry=skill_entry,
        )

    async def reject(self, pending_id: str) -> bool:
        """用户审批拒绝。"""
        return self._pending_store.reject(pending_id) is not None


def _placeholder_candidate(trace: ConversationTrace):
    """未提取到候选时，构造一个占位 CandidateSkill 用于结果记录。"""
    from .models import CandidateSkill

    return CandidateSkill(
        name="__none__",
        description="",
        instruction="",
        source_trace_id=trace.trace_id,
        confidence_score=0.0,
    )
