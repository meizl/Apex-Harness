"""Plugin runtime context (event bus, service registry access).

Every plugin receives a :class:`Context` when it is mounted. The context is
the plugin's only window onto the rest of the harness: it wraps the shared
event bus and service registry, and keeps a ledger of everything the plugin
registered so that :meth:`dispose` can clean it up automatically.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from .events import Event, EventBus, EventHandler
from .services import ServiceRegistry

logger = logging.getLogger(__name__)

DisposeCallback = Callable[[], Awaitable[None] | None]


class Context:
    """The runtime context provided to a plugin during its lifecycle.

    A context is scoped to a single plugin. Services registered and event
    handlers attached via the context are automatically removed when the
    owning plugin is unmounted.
    """

    def __init__(
        self,
        event_bus: EventBus,
        services: ServiceRegistry,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._services = services
        #: Plugin-specific configuration dict (read-only from the plugin's view).
        self.config: dict[str, Any] = config or {}

        # Ledger: everything this context registered, for auto-cleanup.
        self._registered_services: list[str] = []
        self._event_handlers: list[tuple[str, EventHandler]] = []
        self._dispose_callbacks: list[DisposeCallback] = []

    # -- event bus helpers -------------------------------------------------

    def on(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event. Auto-removed on :meth:`dispose`."""
        self._event_bus.on(event_name, handler)
        self._event_handlers.append((event_name, handler))

    def subscribe(self, event_name: str) -> Callable[[EventHandler], EventHandler]:
        """Decorator form of :meth:`on`."""

        def decorator(handler: EventHandler) -> EventHandler:
            self.on(event_name, handler)
            return handler

        return decorator

    async def emit(self, event: Event) -> None:
        """Emit an event on the shared bus."""
        await self._event_bus.emit(event)

    # -- service helpers ---------------------------------------------------

    def register_service(self, name: str, service: Any) -> None:
        """Register a service. Auto-unregistered on :meth:`dispose`."""
        self._services.register(name, service)
        self._registered_services.append(name)

    def get_service(self, name: str, default: Any = None) -> Any:
        """Look up a service, returning *default* if not found."""
        return self._services.get(name, default)

    def require_service(self, name: str) -> Any:
        """Look up a required service, raising :class:`KeyError` if absent."""
        return self._services.require(name)

    def has_service(self, name: str) -> bool:
        """Return whether a service is registered under *name*."""
        return self._services.has(name)

    # -- disposal ----------------------------------------------------------

    def on_dispose(self, callback: DisposeCallback) -> None:
        """Register a callback to run when the plugin is unmounted."""
        self._dispose_callbacks.append(callback)

    async def dispose(self) -> None:
        """Clean up everything this context registered.

        Services are unregistered and event handlers detached in reverse
        registration order. Dispose callbacks run last. Failures in any
        single step are logged but do not stop the rest.
        """
        # Services (reverse order).
        for name in reversed(self._registered_services):
            try:
                self._services.unregister(name)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to unregister service %r", name)
        self._registered_services.clear()

        # Event handlers (reverse order).
        for event_name, handler in reversed(self._event_handlers):
            self._event_bus.off(event_name, handler)
        self._event_handlers.clear()

        # User-registered dispose callbacks (reverse order).
        for callback in reversed(self._dispose_callbacks):
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("Error in dispose callback")
        self._dispose_callbacks.clear()
