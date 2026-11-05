"""SkillEvolutionWorker：右循环的异步进化主流程。

编排链路::

    ConversationTrace
        → SkillExtractor.extract()        提取候选 Skill
        → SkillBankRetriever.retrieve()   检索相似技能
        → EvolutionDecider.decide()       决策 Add/Merge/Discard
        → SkillBankWriter.write()         写入 SkillBank
        → EvolutionResult

整个流程异步执行，不阻塞主对话（左循环）。
"""

from __future__ import annotations

import logging
from typing import Any

from .bank import SkillBank
from .decider import EvolutionDecider
from .extractor import SkillExtractor
from .models import ConversationTrace, EvolutionResult
from .retriever import Embedder, SkillBankRetriever
from .writer import SkillBankWriter

logger = logging.getLogger(__name__)


class SkillEvolutionWorker:
    """技能进化 Worker（右循环）。"""

    def __init__(
        self,
        llm: Any,
        bank: SkillBank,
        *,
        embedder: Embedder | None = None,
        top_k: int = 3,
        similarity_threshold: float = 0.4,
        confidence_threshold: float = 0.6,
    ) -> None:
        self._bank = bank
        self._extractor = SkillExtractor(
            llm, confidence_threshold=confidence_threshold
        )
        self._retriever = SkillBankRetriever(
            bank, embedder=embedder, top_k=top_k, threshold=similarity_threshold
        )
        self._decider = EvolutionDecider()
        self._writer = SkillBankWriter(bank)

    async def run_evolution(self, trace: ConversationTrace) -> EvolutionResult:
        """对一段对话轨迹执行一次技能进化，返回结果。"""
        # 1. 提取候选 Skill
        candidate = await self._extractor.extract(trace)
        if candidate is None:
            logger.info("trace %s 未提取到候选 Skill", trace.trace_id)
            from .models import EvolutionAction, EvolutionDecision

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

        # 4. 写入
        skill_entry = await self._writer.write(decision)

        return EvolutionResult(
            trace_id=trace.trace_id,
            decision=decision,
            skill_entry=skill_entry,
            similarity_scores=scores,
        )


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
