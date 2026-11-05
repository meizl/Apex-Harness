"""会话 checkpoint：把当前会话的消息历史持久化到磁盘，支持崩溃恢复。

设计（单用户场景）：
- 单活跃会话，按 workspace 目录 hash 隔离存储（不同项目的对话不串味）
- 全量覆盖写 + 原子 rename（先写 .tmp 再 rename），崩溃不损坏旧文件
- 只存 messages（system / user / ai / tool），事件日志由 .events/ 负责
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .serializer import deserialize_session, serialize_session
from .session import Session


def workspace_key(workspace_dir: str) -> str:
    """用 workspace 路径的 md5 作为 checkpoint 文件名，避免特殊字符。"""
    return hashlib.md5(workspace_dir.encode("utf-8")).hexdigest()


def checkpoint_path(sessions_dir: Path, workspace_dir: str) -> Path:
    """返回指定 workspace 的 checkpoint 文件路径。"""
    return sessions_dir / f"{workspace_key(workspace_dir)}.json"


def save_checkpoint(
    session: Session, sessions_dir: Path, workspace_dir: str
) -> None:
    """全量覆盖写会话到磁盘（原子写：先写 .tmp 再 rename）。"""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(sessions_dir, workspace_dir)

    data = serialize_session(session)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    fd, tmp_path = tempfile.mkstemp(dir=str(sessions_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(
    sessions_dir: Path, workspace_dir: str
) -> Session | None:
    """从磁盘恢复会话；文件不存在或损坏返回 None。"""
    path = checkpoint_path(sessions_dir, workspace_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_session(data)
    except Exception:
        return None
