"""Event bus for inter-plugin communication.

Plugins communicate by emitting and subscribing to events rather than calling
each other directly. This keeps plugins decoupled: the emitter does not know
(or care) who is listening.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# A handler receives an Event and returns nothing (sync or async).
EventHandler = Callable[["Event"], Awaitable[None] | None]


class Event:
    """Base event class.

    Subclass to create typed events with structured data. The base class
    carries a free-form ``payload`` dict for convenience.
    """

    #: 事件名字
    #： payload   事件携带的数据
    #： propagate   是否继续往下传
    name: str = "event"                                                             

    def __init__(self, **payload: Any) -> None:
        self.payload: dict[str, Any] = payload
        #: Set to False by :meth:`stop_propagation` to skip later handlers.
        self.propagate: bool = True

    def stop_propagation(self) -> None:
        """Prevent remaining handlers from receiving this event."""
        self.propagate = False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name!r} payload={self.payload!r}>"


class EventBus:
    """A simple publish/subscribe event bus.

    Handlers may be sync or async. They run sequentially in registration
    order. Exceptions in one handler are logged and do not stop the others.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    # -- registration ------------------------------------------------------
    #"这个事件发生了叫我"
    def on(self, event_name: str, handler: EventHandler) -> None:
        """Register *handler* for events named *event_name*."""
        self._handlers[event_name].append(handler)

    #"我不听了"
    def off(self, event_name: str, handler: EventHandler) -> None:
        """Remove a previously registered handler. Unknown names are ignored."""
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass
    #"把这封信按地址送出去"
    def subscribe(self, event_name: str) -> Callable[[EventHandler], EventHandler]:
        """Decorator to register a handler for *event_name*.

        Usage::

            @bus.subscribe("message.created")
            async def on_message(event):
                ...
        """

        def decorator(handler: EventHandler) -> EventHandler:
            self.on(event_name, handler)
            return handler

        return decorator

    # -- dispatch ----------------------------------------------------------

    async def emit(self, event: Event) -> None:
        """Dispatch *event* to all handlers registered for its name.

        Handlers run sequentially. If a handler calls
        ``event.stop_propagation()``, remaining handlers are skipped.
        Exceptions in a handler are logged but do not interrupt dispatch.
        """
        # Snapshot the list so handlers can safely register/unregister.
        handlers = list(self._handlers.get(event.name, []))
        for handler in handlers:
            if not event.propagate:
                break
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - isolate handler failures
                logger.exception("Error in event handler for %r", event.name)
