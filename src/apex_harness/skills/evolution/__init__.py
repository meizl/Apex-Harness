"""Skill 进化模块（右循环）。

从对话轨迹中提取候选 Skill，与 SkillBank 比对后执行 Add / Merge / Discard，
最终版本化写入 SkillBank。整个流程异步执行，不阻塞主对话。
"""

from .bank import SkillBank
from .decider import EvolutionDecider
from .extractor import SkillExtractor
from .merger import SkillMerger
from .models import (
    CandidateSkill,
    ConversationTrace,
    EvolutionAction,
    EvolutionDecision,
    EvolutionResult,
    PendingSkill,
    SkillEntry,
)
from .pending_store import PendingSkillStore
from .retriever import SkillBankRetriever
from .worker import SkillEvolutionWorker
from .writer import SkillBankWriter

__all__ = [
    "SkillBank",
    "SkillBankWriter",
    "SkillBankRetriever",
    "SkillEvolutionWorker",
    "SkillExtractor",
    "SkillMerger",
    "EvolutionDecider",
    "CandidateSkill",
    "ConversationTrace",
    "EvolutionAction",
    "EvolutionDecision",
    "EvolutionResult",
    "PendingSkill",
    "PendingSkillStore",
    "SkillEntry",
]
