"""Plugin context.

Every plugin receives a :class:`Context` when it is mounted. The context is the
plugin's window onto the rest of the harness: it exposes the event bus, the
service registry, configuration, and helper methods for registering services
and event handlers that are automatically cleaned up when the plugin is
unmounted.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from .events import Event, EventBus, EventHandler
from .services import ServiceRegistry

logger = logging.getLogger(__name__)


class Context:
    """The runtime context provided to a plugin during its lifecycle.

    A context is *scoped* to a single plugin. Services registered and event
    handlers attached via the context are automatically removed when the
    owning plugin is unmounted.
    """

    def __init__(
        self,
        kernel: Any,
        event_bus: EventBus,
        services: ServiceRegistry,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._kernel = kernel
        self._event_bus = event_bus
        self._services = services
        self.config: dict[str, Any] = config or {}
        # Track registrations so we can clean up on unmount.
        self._registered_services: list[str] = []
        self._event_handlers: list[tuple[str, EventHandler]] = []
        self._dispose_callbacks: list[Callable[[], Awaitable[None] | None]] = []

    # -- Event bus helpers -------------------------------------------------

    def on(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event. The handler is auto-removed on unmount."""
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

    # -- Service helpers ---------------------------------------------------

    def register_service(self, name: str, service: Any) -> None:
        """Register a service. It is auto-unregistered on unmount."""
        self._services.register(name, service)
        self._registered_services.append(name)

    def get_service(self, name: str, default: Any = None) -> Any:
        """Look up a service."""
        return self._services.get(name, default)

    def require_service(self, name: str) -> Any:
        """Look up a required service, raising if absent."""
        return self._services.require(name)

    # -- Lifecycle ---------------------------------------------------------

    def on_dispose(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        """Register a callback to run when the plugin is unmounted."""
        self._dispose_callbacks.append(callback)

    async def dispose(self) -> None:
        """Clean up everything this context registered."""
        for name in reversed(self._registered_services):
            try:
                self._services.unregister(name)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to unregister service %r", name)
        self._registered_services.clear()

        for event_name, handler in reversed(self._event_handlers):
            self._event_bus.off(event_name, handler)
        self._event_handlers.clear()

        for callback in reversed(self._dispose_callbacks):
            try:
                result = callback()
                if asyncio_iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("Error in dispose callback")
        self._dispose_callbacks.clear()


def asyncio_iscoroutine(obj: Any) -> bool:
    """Inline helper to avoid import-time circulars."""
    import asyncio

    return asyncio.iscoroutine(obj)
