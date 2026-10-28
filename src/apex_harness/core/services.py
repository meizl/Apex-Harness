"""Service registry for the plugin kernel.

Services are shared objects that plugins provide and consume. A plugin can
register a service under a name, and other plugins can look it up. This is the
primary mechanism for inter-plugin data sharing (alongside events).
"""

from __future__ import annotations

from typing import Any


class ServiceRegistry:
    """A name-based registry for shared services."""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, name: str, service: Any) -> None:
        """Register a service under *name*.

        Raises :class:`KeyError` if a service with the same name is already
        registered (to prevent accidental overwrites).
        """
        if name in self._services:
            raise KeyError(f"Service '{name}' is already registered")
        self._services[name] = service

    def unregister(self, name: str) -> None:
        """Remove a service. Silently ignores unknown names."""
        self._services.pop(name, None)

    def get(self, name: str, default: Any = None) -> Any:
        """Look up a service by name, returning *default* if not found."""
        return self._services.get(name, default)

    def require(self, name: str) -> Any:
        """Look up a service by name, raising :class:`KeyError` if absent."""
        if name not in self._services:
            raise KeyError(f"Required service '{name}' is not available")
        return self._services[name]

    def has(self, name: str) -> bool:
        """Return whether a service is registered."""
        return name in self._services

    def names(self) -> list[str]:
        """Return a sorted list of registered service names."""
        return sorted(self._services)
