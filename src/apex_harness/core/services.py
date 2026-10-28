"""Service registry for shared plugin services.

Plugins publish capabilities by registering them under a name. Other plugins
look them up by name — never by importing the implementing class. This is the
second pillar of plugin decoupling (the first is the event bus).
"""

from __future__ import annotations

from typing import Any


class ServiceRegistry:
    """A name-based registry for shared services.

    Services are registered by name. Lookup is always by name so that
    consumers do not depend on the concrete implementation.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    # -- mutation ----------------------------------------------------------

    def register(self, name: str, service: Any) -> None:
        """Register *service* under *name*.

        Raises :class:`KeyError` if *name* is already registered, to prevent
        one plugin from silently clobbering another's service.
        """
        if name in self._services:
            raise KeyError(f"Service '{name}' is already registered")
        self._services[name] = service

    def unregister(self, name: str) -> None:
        """Remove *name* from the registry. Unknown names are ignored."""
        self._services.pop(name, None)

    # -- lookup ------------------------------------------------------------

    def get(self, name: str, default: Any = None) -> Any:
        """Return the service registered under *name*, or *default*."""
        return self._services.get(name, default)

    def require(self, name: str) -> Any:
        """Return the service under *name*.

        Raises :class:`KeyError` if no service is registered under *name*.
        Use this when the service is required for the plugin to function.
        """
        if name not in self._services:
            raise KeyError(f"Required service '{name}' is not available")
        return self._services[name]

    def has(self, name: str) -> bool:
        """Return whether a service is registered under *name*."""
        return name in self._services

    def names(self) -> list[str]:
        """Return a sorted list of all registered service names."""
        return sorted(self._services)
