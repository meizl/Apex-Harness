"""SkillMerger：把候选 Skill 合并进已有 SkillEntry。

合并 instruction、trigger_conditions、tools、examples，去重冲突内容，
版本号递增（patch 位 +1），并写入 changelog 记录变更原因与来源 trace_id。
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import CandidateSkill, SkillEntry


def _bump_patch(version: str) -> str:
    """递增 patch 版本号：0.1.3 → 0.1.4。非法格式回退为 0.1.0。"""
    parts = version.split(".")
    if len(parts) != 3:
        return "0.1.0"
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return "0.1.0"
    return f"{major}.{minor}.{patch + 1}"


def _merge_list(existing: list[str], candidate: list[str]) -> list[str]:
    """合并两个字符串列表，保留顺序去重。"""
    merged: list[str] = list(existing)
    for item in candidate:
        if item and item not in merged:
            merged.append(item)
    return merged


class SkillMerger:
    """合并候选 Skill 到已有 SkillEntry。"""

    def merge(
        self, existing: SkillEntry, candidate: CandidateSkill, reason: str = ""
    ) -> SkillEntry:
        """返回合并后的新 SkillEntry（不修改原对象）。"""
        # instruction：候选不为空则追加到已有之后
        new_instruction = existing.instruction
        if candidate.instruction and candidate.instruction not in existing.instruction:
            sep = "\n\n" if existing.instruction else ""
            new_instruction = existing.instruction + sep + candidate.instruction

        merged = SkillEntry(
            skill_id=existing.skill_id,
            name=existing.name,
            version=_bump_patch(existing.version),
            description=existing.description or candidate.description,
            instruction=new_instruction,
            trigger_conditions=_merge_list(
                existing.trigger_conditions, candidate.trigger_conditions
            ),
            tools=_merge_list(existing.tools, candidate.tools),
            examples=_merge_list(existing.examples, candidate.examples),
            changelog=list(existing.changelog) + [
                f"v{_bump_patch(existing.version)}: {reason or '合并候选技能'}"
                f"（来源 trace_id={candidate.source_trace_id}）"
            ],
            created_at=existing.created_at,
            updated_at=datetime.now(timezone.utc),
            usage_count=existing.usage_count,
            success_rate=existing.success_rate,
        )
        return merged
