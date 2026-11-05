"""技能加载器：扫描目录下的 SKILL.md 文件。

每个技能是一个子目录，目录下包含 ``SKILL.md``。也支持直接把 ``SKILL.md``
放在扫描目录的第一层。

扫描规则::

    skills_dir/
        summarize_file/
            SKILL.md          # 子目录形式（推荐）
        code_review/
            SKILL.md
        translate.md          # 直接以 <name>.md 放在根目录也支持

加载失败的技能会被跳过并记录日志，不影响其他技能。
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import SkillMetadata
from .parser import load_skill_md, parse_skill_md

logger = logging.getLogger(__name__)


def load_skills_from_dir(skills_dir: str | Path) -> list[SkillMetadata]:
    """从目录加载所有技能，返回 :class:`SkillMetadata` 列表。"""
    skills_dir = Path(skills_dir)
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.info("技能目录不存在: %s", skills_dir)
        return []

    metadatas: list[SkillMetadata] = []
    seen_names: set[str] = set()

    # 1. 子目录形式：skills_dir/<name>/SKILL.md
    for sub in sorted(skills_dir.iterdir()):
        if not sub.is_dir():
            continue
        skill_md = sub / "SKILL.md"
        if not skill_md.exists():
            continue
        _try_load(skill_md, metadatas, seen_names)

    # 2. 根目录直接放 .md 文件（SKILL.md 或 <name>.md）
    for md_file in sorted(skills_dir.glob("*.md")):
        # SKILL.md 在根目录也支持，但更推荐子目录形式
        _try_load(md_file, metadatas, seen_names)

    logger.info("从 %s 加载了 %d 个技能", skills_dir, len(metadatas))
    return metadatas


def _try_load(
    md_path: Path,
    metadatas: list[SkillMetadata],
    seen_names: set[str],
) -> None:
    """解析单个 SKILL.md，去重后加入列表。"""
    try:
        parsed = load_skill_md(md_path)
    except Exception:
        logger.exception("解析技能文件 %s 失败", md_path)
        return

    if parsed.name in seen_names:
        logger.warning("技能名重复，跳过: %s (%s)", parsed.name, md_path)
        return

    seen_names.add(parsed.name)
    metadatas.append(
        SkillMetadata(
            name=parsed.name,
            description=parsed.description,
            instructions=parsed.instructions,
            extra=parsed.extra,
        )
    )
