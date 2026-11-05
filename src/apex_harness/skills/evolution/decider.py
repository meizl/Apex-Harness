"""EvolutionDecider：对候选 Skill 做出 Add / Merge / Discard 决策。

决策规则：
- 无高相似 Skill + 候选复用价值高 → Add
- 存在高相似 Skill + 候选内容属于补充/修正 → Merge
- 一次性需求、质量低、冲突明显 → Discard
"""

from __future__ import annotations

from .models import (
    CandidateSkill,
    EvolutionAction,
    EvolutionDecision,
    SkillEntry,
)

# 相似度阈值
HIGH_SIMILARITY = 0.75  # ≥ 此值视为高相似，倾向 Merge
# 置信度阈值
HIGH_CONFIDENCE = 0.7  # 候选置信度 ≥ 此值才考虑入库
MIN_CONFIDENCE = 0.5   # 低于此值直接 Discard


class EvolutionDecider:
    """进化决策器。"""

    def decide(
        self,
        candidate: CandidateSkill,
        similar_skills: list[tuple[SkillEntry, float]],
    ) -> EvolutionDecision:
        """根据候选与相似技能列表做出决策。"""
        confidence = candidate.confidence_score

        # 1. 置信度过低 → Discard
        if confidence < MIN_CONFIDENCE:
            return EvolutionDecision(
                action=EvolutionAction.DISCARD,
                reason=f"候选置信度 {confidence:.2f} 低于最低阈值 {MIN_CONFIDENCE}",
                candidate_skill=candidate,
            )

        if not similar_skills:
            # 2. 无相似技能 + 置信度够高 → Add
            if confidence >= HIGH_CONFIDENCE:
                return EvolutionDecision(
                    action=EvolutionAction.ADD,
                    reason="无高相似技能且候选置信度高，作为新技能入库",
                    candidate_skill=candidate,
                )
            return EvolutionDecision(
                action=EvolutionAction.DISCARD,
                reason=f"无相似技能但置信度 {confidence:.2f} 不足，暂不入库",
                candidate_skill=candidate,
            )

        # 3. 存在相似技能
        top_entry, top_score = similar_skills[0]
        if top_score >= HIGH_SIMILARITY:
            # 高相似 → Merge（补充/修正已有技能）
            return EvolutionDecision(
                action=EvolutionAction.MERGE,
                reason=f"与已有技能「{top_entry.name}」相似度 {top_score:.2f}，合并补充",
                target_skill_id=top_entry.skill_id,
                candidate_skill=candidate,
            )

        # 中等相似：若候选置信度高则仍 Add（视为不同但相关的技能）
        if confidence >= HIGH_CONFIDENCE:
            return EvolutionDecision(
                action=EvolutionAction.ADD,
                reason=(
                    f"与已有技能最高相似度 {top_score:.2f} 未达合并阈值，"
                    f"且候选置信度 {confidence:.2f} 高，作为新技能入库"
                ),
                candidate_skill=candidate,
            )

        return EvolutionDecision(
            action=EvolutionAction.DISCARD,
            reason=f"候选置信度 {confidence:.2f} 不足，且无高相似技能可合并",
            candidate_skill=candidate,
        )
