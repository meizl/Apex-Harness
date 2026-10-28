"""The plugin kernel.

The :class:`Kernel` is the top-level manager. It owns the event bus, the
service registry, and the set of mounted plugins. It is responsible for:

- mounting / unmounting plugins
- resolving dependency order (topological sort)
- driving the plugin lifecycle (setup → start → stop → dispose)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from .context import Context
from .events import EventBus
from .plugin import Plugin
from .services import ServiceRegistry

logger = logging.getLogger(__name__)


class KernelError(Exception):
    """Raised for kernel-level errors (missing dependency, cycle, etc.)."""


class Kernel:
    """The plugin manager.

    Typical usage::

        kernel = Kernel()
        await kernel.mount(MyPlugin())
        await kernel.start()
        # ... use the harness ...
        await kernel.stop()
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.event_bus = EventBus()
        self.services = ServiceRegistry()
        self._plugins: dict[str, Plugin] = {}
        self._contexts: dict[str, Context] = {}
        self._started = False

    # -- registration ------------------------------------------------------

    async def mount(self, plugin: Plugin) -> None:
        """Mount a single plugin.

        This calls the plugin's ``setup`` hook and registers any services the
        plugin provides. Dependencies are *not* mounted automatically; call
        :meth:`mount_all` with a full list to get dependency resolution.
        """
        name = plugin.name or type(plugin).__name__
        if name in self._plugins:
            raise KernelError(f"Plugin '{name}' is already mounted")

        ctx = Context(self, self.event_bus, self.services, self.config.get(name))
        plugin._bind(ctx)
        self._plugins[name] = plugin
        self._contexts[name] = ctx

        try:
            await plugin.setup()
            logger.info("Plugin '%s' setup complete", name)
        except Exception:
            logger.exception("Plugin '%s' setup failed", name)
            await self._unmount(name)
            raise

    async def unmount(self, name: str) -> None:
        """Unmount a plugin by name."""
        if name not in self._plugins:
            return
        await self._unmount(name)

    async def _unmount(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        ctx = self._contexts.pop(name, None)
        if plugin is None or ctx is None:
            return
        try:
            await plugin.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Plugin '%s' stop failed", name)
        try:
            await plugin.dispose()
        except Exception:  # noqa: BLE001
            logger.exception("Plugin '%s' dispose failed", name)
        await ctx.dispose()
        logger.info("Plugin '%s' unmounted", name)

    # -- bulk mounting with dependency resolution --------------------------

    async def mount_all(self, plugins: list[Plugin]) -> None:
        """Mount a list of plugins in dependency order.

        Uses a topological sort to ensure dependencies are mounted first.
        Raises :class:`KernelError` if a dependency is missing or a cycle is
        detected.
        """
        order = self._resolve_order(plugins)
        for plugin in order:
            await self.mount(plugin)

    def _resolve_order(self, plugins: list[Plugin]) -> list[Plugin]:
        by_name = {p.name or type(p).__name__: p for p in plugins}
        # Verify all dependencies are present.
        for name, plugin in by_name.items():
            for dep in plugin.requires:
                if dep not in by_name:
                    raise KernelError(
                        f"Plugin '{name}' requires missing plugin '{dep}'"
                    )

        # Kahn's algorithm for topological sort.
        in_degree: dict[str, int] = {name: 0 for name in by_name}
        dependents: dict[str, list[str]] = {name: [] for name in by_name}
        for name, plugin in by_name.items():
            for dep in plugin.requires:
                in_degree[name] += 1
                dependents[dep].append(name)

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for dep in dependents[current]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(ordered) != len(by_name):
            raise KernelError("Cyclic dependency detected among plugins")

        return [by_name[name] for name in ordered]

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start all mounted plugins in dependency order."""
        if self._started:
            return
        for name in self._plugins:
            try:
                await self._plugins[name].start()
                self._plugins[name]._started = True
            except Exception:
                logger.exception("Plugin '%s' start failed", name)
                raise
        self._started = True
        logger.info("Kernel started with %d plugins", len(self._plugins))

    async def stop(self) -> None:
        """Stop all plugins in reverse order and unmount them."""
        if not self._started:
            return
        for name in reversed(list(self._plugins)):
            try:
                await self._plugins[name].stop()
            except Exception:  # noqa: BLE001
                logger.exception("Plugin '%s' stop failed", name)
        self._started = False
        logger.info("Kernel stopped")

    async def shutdown(self) -> None:
        """Stop and unmount all plugins."""
        await self.stop()
        for name in list(self._plugins):
            await self._unmount(name)

    # -- inspection --------------------------------------------------------

    @property
    def plugins(self) -> dict[str, Plugin]:
        return dict(self._plugins)

    def get_plugin(self, name: str) -> Plugin | None:
        return self._plugins.get(name)
