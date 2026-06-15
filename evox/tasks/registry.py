"""Task registry: name -> Task factory.

Lets configs and the suite runner refer to tasks by string name and construct
them with parameters, so multi-task evaluation is a matter of registering more
tasks — not editing the engine.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import Task

_REGISTRY: Dict[str, Callable[..., Task]] = {}


def register_task(name: str, factory: Callable[..., Task]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"task {name!r} already registered")
    _REGISTRY[name] = factory


def create_task(name: str, params: Dict[str, Any] = None) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))


def available_tasks() -> list:
    return sorted(_REGISTRY)


# Register built-in tasks. Import here so that importing the registry makes the
# V0 task available without callers needing to import the module explicitly.
from . import blackbox as _blackbox  # noqa: E402

register_task("toy_blackbox", _blackbox.ToyBlackBoxTask)
