"""SkillBankRetriever：在 SkillBank 中向量检索最相似的已有 Skill。

支持两种相似度计算：
1. 传入 ``embedder``（异步函数 ``(texts) -> vectors``），用真实向量 + 余弦相似度。
2. 未传入时使用默认的「字符 n-gram TF-IDF 余弦相似度」，无需外部依赖。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Awaitable, Callable

from .bank import SkillBank
from .models import CandidateSkill, SkillEntry

# 默认 embedder 签名：异步接收文本列表，返回等长向量列表
Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


# --------------------------------------------------------------------------- #
# 默认相似度：字符 n-gram + 余弦
# --------------------------------------------------------------------------- #


def _char_ngrams(text: str, n: int = 2) -> Counter[str]:
    """提取字符 n-gram 并计数。"""
    text = re.sub(r"\s+", " ", text.lower())
    if len(text) < n:
        return Counter([text]) if text else Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _cosine(c1: Counter[str], c2: Counter[str]) -> float:
    """两个 Counter 的余弦相似度。"""
    if not c1 or not c2:
        return 0.0
    common = set(c1) & set(c2)
    dot = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _skill_text(entry: SkillEntry | CandidateSkill) -> str:
    """把 Skill 的关键字段拼成一段文本用于相似度计算。"""
    parts = [entry.name, entry.description]
    if isinstance(entry, SkillEntry):
        parts.append(entry.instruction)
    else:
        parts.append(entry.instruction)
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #


class SkillBankRetriever:
    """在 SkillBank 中检索与候选 Skill 最相似的已有 Skill。"""

    def __init__(
        self,
        bank: SkillBank,
        *,
        embedder: Embedder | None = None,
        top_k: int = 3,
        threshold: float = 0.4,
    ) -> None:
        self._bank = bank
        self._embedder = embedder
        self._top_k = top_k
        self._threshold = threshold

    async def retrieve(
        self, candidate: CandidateSkill
    ) -> list[tuple[SkillEntry, float]]:
        """返回与候选相似的 Top-K SkillEntry 及相似度（降序，过滤低于阈值）。"""
        entries = self._bank.all()
        if not entries:
            return []

        if self._embedder is not None:
            scores = await self._embed_scores(candidate, entries)
        else:
            scores = self._ngram_scores(candidate, entries)

        # 降序排列，过滤阈值
        ranked = sorted(
            zip(entries, scores), key=lambda x: x[1], reverse=True
        )
        return [(e, s) for e, s in ranked if s >= self._threshold][: self._top_k]

    # -- 向量相似度 --------------------------------------------------------

    async def _embed_scores(
        self, candidate: CandidateSkill, entries: list[SkillEntry]
    ) -> list[float]:
        cand_text = _skill_text(candidate)
        entry_texts = [_skill_text(e) for e in entries]
        vectors = await self._embedder([cand_text, *entry_texts])  # type: ignore[misc]
        cand_vec = vectors[0]
        return [_cosine_vec(cand_vec, v) for v in vectors[1:]]

    # -- n-gram 相似度 -----------------------------------------------------

    def _ngram_scores(
        self, candidate: CandidateSkill, entries: list[SkillEntry]
    ) -> list[float]:
        cand_grams = _char_ngrams(_skill_text(candidate))
        return [_cosine(cand_grams, _char_ngrams(_skill_text(e))) for e in entries]


def _cosine_vec(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
