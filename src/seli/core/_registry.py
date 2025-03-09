import logging
from collections.abc import Hashable
from typing import Any

import jax.tree_util as jtu

from seli.core._typecheck import typecheck

logger = logging.getLogger(__name__)


__all__ = [
    "registry_add",
    "registry_str",
    "registry_obj",
    "is_registry_str",
]


REGISTRY: dict[str, Hashable] = {}
REGISTRY_INVERSE: dict[Hashable, str] = {}


@typecheck
class ModuleBase:
    def __init_subclass__(
        cls,
        name: str | None = None,
        overwrite: bool = False,
    ):
        if hasattr(cls, "tree_flatten") and hasattr(cls, "tree_unflatten"):
            cls = jtu.register_pytree_node_class(cls)

        if name is not None:
            registry_add(name, cls, overwrite=overwrite)


@typecheck
def registry_add(
    name: str,
    module: type,
    overwrite: bool = False,
) -> None:
    if not overwrite and name in REGISTRY:
        if REGISTRY[name] is module:
            return

        msg = f"Module {name} already registered, skipping {module} ({type(module)})"
        msg += f" already registered as {REGISTRY[name]} ({type(REGISTRY[name])})"
        logger.warning(msg)
        return

    REGISTRY[name] = module
    REGISTRY_INVERSE[module] = name


@typecheck
def registry_str(obj: Any) -> str:
    return f"__registry__:{REGISTRY_INVERSE[obj]}"


@typecheck
def registry_obj(name: str) -> Hashable:
    if not is_registry_str(name):
        raise ValueError(f"Invalid registry string: {name}")

    name = name[len("__registry__:") :]

    if name not in REGISTRY:
        raise ValueError(f"Module {name} not registered")

    return REGISTRY[name]


@typecheck
def is_registry_str(obj: Any) -> bool:
    return isinstance(obj, str) and obj.startswith("__registry__:")
