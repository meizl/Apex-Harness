"""Plugin kernel: mount/unmount, dependency resolution, lifecycle orchestration.

The :class:`Kernel` owns the event bus, the service registry, and the set of
mounted plugins. It mounts plugins in dependency order, drives their lifecycle
(``setup`` → ``start`` → ``stop`` → ``dispose``), and guarantees clean teardown.
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
        await kernel.mount_all([LLMPlugin(), ToolPlugin(), AgentPlugin()])
        await kernel.start()
        # ... use the harness ...
        await kernel.shutdown()
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        #: Top-level configuration, keyed by plugin name.
        self.config: dict[str, Any] = config or {}
        self.event_bus = EventBus()
        self.services = ServiceRegistry()

        self._plugins: dict[str, Plugin] = {}
        self._contexts: dict[str, Context] = {}
        self._started: bool = False

    # -- mounting ----------------------------------------------------------

    async def mount(self, plugin: Plugin) -> None:
        """Mount a single plugin.

        Binds a fresh :class:`Context` and calls ``setup``. Does **not**
        resolve dependencies — use :meth:`mount_all` for that.
        """
        # 1. 确定插件名字（用 name 属性，没有就用类名）
        name = plugin.name or type(plugin).__name__
        # 2. 检查是否已经装过同名插件，防止重复
        if name in self._plugins:
            raise KernelError(f"Plugin '{name}' is already mounted")

        # 3. 给插件创建一个专属 Context（绑定共享的 EventBus 和 ServiceRegistry）
        ctx = Context(self.event_bus, self.services, self.config.get(name))
        # 4. 把 ctx 塞给插件，插件通过 self.ctx 使用
        plugin._bind(ctx)
        # 5. 记录到 Kernel 的插件表和 context 表
        self._plugins[name] = plugin
        self._contexts[name] = ctx

        # 6. 调用插件的 setup（注册服务、监听事件）
        try:
            await plugin.setup()
            logger.info("Plugin '%s' setup complete", name)
        except Exception:
            # 7. setup 出错 → 回滚：把刚装的插件拆掉，再抛出错误
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
        """Mount plugins in dependency order.

        Dependencies are resolved through each plugin's ``requires`` and
        ``provides`` declarations. A plugin that *requires* a service must be
        mounted after the plugin that *provides* it.
        """
        order = self._resolve_order(plugins)
        for plugin in order:
            await self.mount(plugin)


    #自动解析
    def _resolve_order(self, plugins: list[Plugin]) -> list[Plugin]:
        by_name: dict[str, Plugin] = {
            p.name or type(p).__name__: p for p in plugins
        }

        # Map service name -> plugin that provides it.
        providers: dict[str, str] = {}
        for name, plugin in by_name.items():
            for service in plugin.provides:
                providers[service] = name

        # Build dependency edges: plugin -> set of plugins it depends on.
        dependents: dict[str, list[str]] = {name: [] for name in by_name}
        in_degree: dict[str, int] = {name: 0 for name in by_name}

        for name, plugin in by_name.items():
            for service in plugin.requires:
                provider = providers.get(service)
                if provider is None:
                    raise KernelError(
                        f"Plugin '{name}' requires service '{service}', "
                        f"but no mounted plugin provides it"
                    )
                if provider == name:
                    continue  # self-dependency is a no-op
                # name depends on provider
                in_degree[name] += 1
                dependents[provider].append(name)

        # Kahn's algorithm.
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
    #挨个喊每个插件"开始干活"
    async def start(self) -> None:
        """Start all mounted plugins (in mount order)."""
        if self._started:
            return
        for name, plugin in self._plugins.items():
            try:
                await plugin.start()
                plugin._started = True
            except Exception:
                logger.exception("Plugin '%s' start failed", name)
                raise
        self._started = True
        logger.info("Kernel started with %d plugin(s)", len(self._plugins))

    async def stop(self) -> None:
        """Stop all plugins in reverse mount order."""
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
