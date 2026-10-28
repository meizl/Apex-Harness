"""Core plugin kernel for Apex Harness.

This module implements the "everything is a plugin" architecture inspired by
Cordis. It provides:
- :class:`Context`: the runtime context passed to every plugin
- :class:`Plugin`: base class for all plugins
- :class:`Kernel`: the plugin manager that handles mounting, unmounting,
  dependencies and lifecycle
"""

from .context import Context
from .events import EventBus, Event
from .plugin import Plugin, PluginMeta
from .kernel import Kernel
from .services import ServiceRegistry

__all__ = ["Context", "EventBus", "Event", "Plugin", "PluginMeta", "Kernel", "ServiceRegistry"]
