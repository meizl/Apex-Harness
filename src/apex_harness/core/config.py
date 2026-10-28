"""配置加载与分发。

配置层负责从 YAML/JSON 文件和环境变量加载配置，并按插件名分发。
Kernel 会把每个插件的配置切片通过 ``ctx.config`` 交给对应插件。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml 是核心依赖
    yaml = None  # type: ignore[assignment]


class ConfigError(Exception):
    """配置加载或解析出错时抛出。"""


class Config:
    """按插件名组织的配置容器。

    配置是一个嵌套的字典，顶层 key 是插件名::

        {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "memory": {"backend": "local"},
        }

    每个顶层 key 对应一个插件，:meth:`get` 返回该插件的配置切片，
    Kernel 会把它交给插件的 ``ctx.config``。
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data) if data else {}

    # -- 加载 --------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """从 YAML 或 JSON 文件加载配置。

        根据文件扩展名判断格式（``.yaml`` / ``.yml`` / ``.json``）。
        不支持的扩展名抛出 :class:`ConfigError`。
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {path}")

        suffix = path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                if yaml is None:
                    raise ConfigError("加载 YAML 需要安装 PyYAML")
                data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            elif suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise ConfigError(
                    f"不支持的配置格式 '{suffix}'（只支持 .yaml / .yml / .json）"
                )
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一包装解析错误
            raise ConfigError(f"解析配置文件失败 {path}: {exc}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件 {path} 顶层必须是键值对（dict）")

        return cls(data)

    def load_env(self, prefix: str = "APEX_") -> "Config":
        """把环境变量合并进配置。

        读取以 ``prefix`` 开头的环境变量，用 ``__`` 作为层级分隔符映射到
        嵌套 key。例如 ``APEX_LLM__MODEL=gpt-4`` 会设置
        ``config["llm"]["model"] = "gpt-4"``。

        返回 ``self``，方便链式调用。
        """
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            path = key[len(prefix):].lower().split("__")
            if not path or not all(path):
                continue
            _set_nested(self._data, path, value)
        return self

    # -- 访问 --------------------------------------------------------------

    def get(self, name: str, default: Any = None) -> Any:
        """返回插件 *name* 的配置切片。"""
        return self._data.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """返回原始配置的浅拷贝。"""
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Config keys={sorted(self._data)!r}>"


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    """在嵌套 key 路径上设置 ``value``，中间层级不存在时自动创建。"""
    current = data
    for part in path[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[path[-1]] = value
