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
# 用户自定义技能目录（用户通过前端新增的技能保存在这里）
USER_SKILLS_DIR = Path(__file__).parent / "user"


def load_builtin_skills() -> list[SkillMetadata]:
    """加载内置技能 + 用户自定义技能（用户技能优先级更高，同名覆盖内置）。"""
    builtins = load_skills_from_dir(BUILTIN_SKILLS_DIR)
    users = load_skills_from_dir(USER_SKILLS_DIR)
    # 用户技能覆盖同名内置技能
    user_names = {m.name for m in users}
    result = [m for m in builtins if m.name not in user_names]
    result.extend(users)
    return result


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
    "USER_SKILLS_DIR",
]
