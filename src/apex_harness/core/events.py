"""Event system for the plugin kernel.

Events are the primary communication mechanism between plugins. Plugins can
emit events and subscribe to events from other plugins. This decouples plugins
from each other while still allowing rich interaction.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

EventHandler = Callable[["Event"], Awaitable[None] | None]


class Event:
    """Base event class.

    Events carry a name and an arbitrary payload. Subclass to create typed
    events with structured data.
    """

    name: str = "event"

    def __init__(self, **payload: Any) -> None:
        self.payload = payload
        self.propagate: bool = True

    def stop_propagation(self) -> None:
        """Prevent further handlers from receiving this event."""
        self.propagate = False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} {self.payload!r}>"


class EventBus:
    """A simple publish/subscribe event bus.

    Handlers can be sync or async. Events are dispatched in registration order.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def on(self, event_name: str, handler: EventHandler) -> None:
        """Register a handler for an event by name."""
        self._handlers[event_name].append(handler)

    def off(self, event_name: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

    def subscribe(self, event_name: str) -> Callable[[EventHandler], EventHandler]:
        """Decorator to register a handler for an event."""

        def decorator(handler: EventHandler) -> EventHandler:
            self.on(event_name, handler)
            return handler

        return decorator

    async def emit(self, event: Event) -> None:
        """Dispatch an event to all registered handlers.

        Handlers are run sequentially in registration order. If a handler calls
        ``event.stop_propagation()``, remaining handlers are skipped.
        """
        handlers = list(self._handlers.get(event.name, []))
        for handler in handlers:
            if not event.propagate:
                break
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("Error in event handler for %r", event.name)
