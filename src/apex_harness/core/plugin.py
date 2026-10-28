"""Plugin base class and lifecycle hooks.

A plugin is the unit of composition in Apex Harness. Each plugin declares:

- ``name``: a unique identifier
- ``requires``: names of services this plugin needs (provided by other plugins)
- ``provides``: names of services this plugin registers during setup
- lifecycle hooks: :meth:`setup`, :meth:`start`, :meth:`stop`, :meth:`dispose`
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .context import Context

logger = logging.getLogger(__name__)

#元类，自动合并父类和子类的`requires` /`provides` 
class PluginMeta(type):
    """Metaclass that collects ``requires`` / ``provides`` from the MRO.

    Values declared on a subclass extend (not replace) those from base
    classes. Duplicates are removed while preserving declaration order.
    """

    def __new__(
        mcs,
        name: str,                                                      #唯一标识
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> "PluginMeta":
        cls = super().__new__(mcs, name, bases, namespace)

        requires: list[str] = []                                        #需要的服务名列表
        provides: list[str] = []                                        #提供的服务名列表
        for base in reversed(bases):
            requires.extend(getattr(base, "requires", []) or [])
            provides.extend(getattr(base, "provides", []) or [])
        requires.extend(namespace.get("requires", []) or [])
        provides.extend(namespace.get("provides", []) or [])

        # Deduplicate while keeping order.
        cls.requires = list(dict.fromkeys(requires))  # type: ignore[attr-defined]
        cls.provides = list(dict.fromkeys(provides))  # type: ignore[attr-defined]
        return cls




#所有插件的父类，任何插件都要继承它
class Plugin(metaclass=PluginMeta):
    """Base class for all plugins.

    Subclass and override the lifecycle hooks to add behavior. The plugin's
    :class:`Context` is available as ``self.ctx`` after :meth:`setup` begins.
    """
    #name # 插件名字（唯一） requires # 需要什么服务 provides # 提供什么服务 description # 描述
    #: Unique plugin name.
    name: ClassVar[str] = ""

    #: Human-readable description.
    description: ClassVar[str] = ""

    #: Service names this plugin requires before it can start.
    requires: ClassVar[list[str]] = []

    #: Service names this plugin registers during setup.
    provides: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.ctx: Context | None = None
        self._started: bool = False

    # -- lifecycle hooks (override these) ----------------------------------
    
    async def setup(self) -> None:
        """Called once when the plugin is mounted.

        Use this to register services, subscribe to events, and read config.
        At this point dependencies may not have started yet, so only use
        services you registered yourself.
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

    # -- internal helpers --------------------------------------------------

    def _bind(self, ctx: Context) -> None:
        """Bind the runtime context. Called by the kernel before setup."""
        self.ctx = ctx

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Plugin {self.name or type(self).__name__}>"
