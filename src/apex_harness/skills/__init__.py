"""技能模块。

技能以 ``SKILL.md`` 文件定义（frontmatter + 指令正文），通过
:func:`load_skills_from_dir` 从目录扫描加载。
"""

from pathlib import Path

from .adapter import SkillToolAdapter
from .base import SkillMetadata
from .loader import load_skills_from_dir
from .parser import ParsedSkill, load_skill_md, parse_skill_md
from .registry import SkillRegistry, SkillRegistryPlugin

# 内置技能目录（随框架发布的 SKILL.md）
BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtins"


def load_builtin_skills() -> list[SkillMetadata]:
    """加载内置技能。"""
    return load_skills_from_dir(BUILTIN_SKILLS_DIR)


__all__ = [
    "SkillMetadata",
    "SkillRegistry",
    "SkillRegistryPlugin",
    "SkillToolAdapter",
    "ParsedSkill",
    "load_skills_from_dir",
    "load_skill_md",
    "parse_skill_md",
    "load_builtin_skills",
    "BUILTIN_SKILLS_DIR",
]
