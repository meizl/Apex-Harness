"""待审批 Skill 存储：内存索引 + JSONL append-only 持久化。

右循环提取的候选 Skill（决策为 Add/Merge）不直接写入 SkillBank，而是
进入待审批队列，等待用户查看、修改后决定通过或拒绝。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from .models import CandidateSkill, EvolutionDecision, PendingSkill


class PendingSkillStore:
    """待审批 Skill 的存储（内存 + JSONL）。"""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path
        self._pending: dict[str, PendingSkill] = {}
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---- 内部持久化 ---- #

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                p = PendingSkill.model_validate(data)
                if p.status == "pending":
                    self._pending[p.pending_id] = p
            except Exception:
                continue

    def _append(self, pending: PendingSkill) -> None:
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as f:
            f.write(pending.model_dump_json() + "\n")

    def _rewrite_all(self) -> None:
        """全量重写 JSONL（审批后状态变更时调用）。"""
        if self._path is None:
            return
        with self._path.open("w", encoding="utf-8") as f:
            for p in self._pending.values():
                f.write(p.model_dump_json() + "\n")

    # ---- 公共 API ---- #

    def add(
        self,
        candidate: CandidateSkill,
        decision: EvolutionDecision,
    ) -> PendingSkill:
        """添加一个待审批候选，返回 PendingSkill。"""
        pending_id = f"pending_{uuid.uuid4().hex[:12]}"
        pending = PendingSkill(
            pending_id=pending_id,
            candidate=candidate,
            decision=decision,
        )
        self._pending[pending_id] = pending
        self._append(pending)
        return pending

    def get(self, pending_id: str) -> Optional[PendingSkill]:
        return self._pending.get(pending_id)

    def list_pending(self) -> list[PendingSkill]:
        """返回所有待审批候选，按创建时间倒序。"""
        return sorted(
            self._pending.values(),
            key=lambda p: p.created_at,
            reverse=True,
        )

    def update_candidate(
        self, pending_id: str, candidate: CandidateSkill
    ) -> Optional[PendingSkill]:
        """用户修改候选内容后更新。"""
        pending = self._pending.get(pending_id)
        if pending is None:
            return None
        pending.candidate = candidate
        self._rewrite_all()
        return pending

    def approve(self, pending_id: str) -> Optional[PendingSkill]:
        """审批通过：状态改为 approved，从待审批队列移除。"""
        pending = self._pending.pop(pending_id, None)
        if pending is None:
            return None
        pending.status = "approved"
        self._rewrite_all()
        return pending

    def reject(self, pending_id: str) -> Optional[PendingSkill]:
        """审批拒绝：状态改为 rejected，从待审批队列移除。"""
        pending = self._pending.pop(pending_id, None)
        if pending is None:
            return None
        pending.status = "rejected"
        self._rewrite_all()
        return pending
