"""SkillBank：技能库的内存索引 + JSONL 持久化。

每个 :class:`SkillEntry` 以 ``skill_id`` 为主键存储。持久化采用 JSONL
追加写（append-only），同一 ``skill_id`` 的多个版本按写入顺序回放，最后
一条生效，从而保证可回溯与幂等。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

from .models import SkillEntry


class SkillBank:
    """技能库：内存索引 + JSONL 文件持久化。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._entries: dict[str, SkillEntry] = {}
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- 持久化 ------------------------------------------------------------

    def _load(self) -> None:
        """从 JSONL 回放所有条目（同 skill_id 取最新）。"""
        if self._path is None or not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                entry = SkillEntry.model_validate(data)
            except Exception:  # noqa: BLE001
                continue
            self._entries[entry.skill_id] = entry

    def _persist(self, entry: SkillEntry) -> None:
        """追加写入一条记录（write-ahead）。"""
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    # -- 读写接口 ----------------------------------------------------------

    def get(self, skill_id: str) -> SkillEntry | None:
        with self._lock:
            return self._entries.get(skill_id)

    def get_by_name(self, name: str) -> SkillEntry | None:
        with self._lock:
            for e in self._entries.values():
                if e.name == name:
                    return e
        return None

    def all(self) -> list[SkillEntry]:
        with self._lock:
            return list(self._entries.values())

    def put(self, entry: SkillEntry) -> SkillEntry:
        """写入或覆盖一个条目，返回写入后的条目。"""
        with self._lock:
            self._entries[entry.skill_id] = entry
            self._persist(entry)
            return entry

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[SkillEntry]:
        return iter(self._entries.values())
