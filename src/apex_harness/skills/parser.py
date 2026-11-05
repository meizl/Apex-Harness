"""SKILL.md 解析器。

每个技能是一个目录，目录下有一个 ``SKILL.md`` 文件，格式::

    ---
    name: summarize_file
    description: 读取文件内容并生成简洁的中文摘要
    ---

    你是一个文件摘要助手。请用简洁的中文总结给定文件的内容，
    不超过 200 字。只输出摘要，不要解释。

- ``---`` 包裹的 YAML frontmatter 存元数据（name / description 等）
- frontmatter 之后的 Markdown 正文是技能的 instructions，调用时注入 LLM 上下文

依赖 ``pyyaml``（已在 pyproject.toml 中声明）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# 匹配 --- ... --- 的 frontmatter 块
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


@dataclass
class ParsedSkill:
    """解析后的技能内容。"""

    name: str
    description: str
    instructions: str
    #: frontmatter 中除 name/description 外的额外字段
    extra: dict[str, Any]


def parse_skill_md(content: str) -> ParsedSkill:
    """解析 SKILL.md 文本内容。"""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError("SKILL.md 缺少 YAML frontmatter（需以 --- 包裹）")

    frontmatter_text, body = match.group(1), match.group(2)
    meta = yaml.safe_load(frontmatter_text) or {}

    name = meta.get("name", "").strip()
    if not name:
        raise ValueError("SKILL.md frontmatter 缺少必填字段 name")
    description = meta.get("description", "").strip()

    extra = {k: v for k, v in meta.items() if k not in ("name", "description")}

    return ParsedSkill(
        name=name,
        description=description,
        instructions=body.strip(),
        extra=extra,
    )


def load_skill_md(path: str | Path) -> ParsedSkill:
    """从文件路径加载并解析 SKILL.md。"""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return parse_skill_md(content)
