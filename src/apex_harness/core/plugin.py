"""Plugin base class.

A plugin is the unit of composition in Apex Harness. Each plugin declares:

- a unique ``name``
- a list of dependencies (``requires``) — other plugins that must be mounted
  before this one
- a list of services it provides (``provides``)
- lifecycle hooks: :meth:`setup`, :meth:`start`, :meth:`stop`, :meth:`dispose`
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .context import Context

logger = logging.getLogger(__name__)


class PluginMeta(type):
    """Metaclass that collects ``requires`` / ``provides`` declarations."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> "PluginMeta":
        cls = super().__new__(mcs, name, bases, namespace)
        # Gather requires / provides from the class and its bases.
        requires: list[str] = []
        provides: list[str] = []
        for base in reversed(bases):
            requires.extend(getattr(base, "requires", []) or [])
            provides.extend(getattr(base, "provides", []) or [])
        requires.extend(namespace.get("requires", []) or [])
        provides.extend(namespace.get("provides", []) or [])
        cls.requires = list(dict.fromkeys(requires))  # type: ignore[attr-defined]
        cls.provides = list(dict.fromkeys(provides))  # type: ignore[attr-defined]
        return cls


class Plugin(metaclass=PluginMeta):
    """Base class for all plugins.

    Subclass and override the lifecycle hooks to add behavior. The plugin's
    :class:`Context` is available as ``self.ctx`` after :meth:`setup` begins.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.ctx: Context | None = None
        self._started: bool = False

    # -- lifecycle hooks (override these) ----------------------------------

    async def setup(self) -> None:
        """Called once when the plugin is mounted.

        Use this to register services, subscribe to events, and read config.
        """

    async def start(self) -> None:
        """Called after all dependencies have started.

        Use this for work that depends on other plugins' services being
        available (e.g. connecting to an LLM provider).
        """

    async def stop(self) -> None:
        """Called when the harness is shutting down, before :meth:`dispose`."""

    async def dispose(self) -> None:
        """Final cleanup. Most cleanup is handled automatically by the context."""

    # -- internal methods --------------------------------------------------

    def _bind(self, ctx: Context) -> None:
        self.ctx = ctx

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Plugin {self.name or type(self).__name__}>"
