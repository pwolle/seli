import logging
from collections.abc import Hashable
from typing import Any

import jax.tree_util as jtu

logger = logging.getLogger(__name__)


REGISTRY: dict[str, Hashable] = {}
REGISTRY_INVERSE: dict[Hashable, str] = {}


class ModuleBase:
    def __init_subclass__(
        cls,
        name: str | None = None,
        overwrite: bool | None = None,
    ):
        if hasattr(cls, "tree_flatten") and hasattr(cls, "tree_unflatten"):
            cls = jtu.register_pytree_node_class(cls)

        if name is not None:
            registry_add(name, cls, overwrite=overwrite)


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


def registry_str(obj: Any) -> str:
    return REGISTRY_INVERSE[obj]


def registry_obj(name: str) -> Hashable:
    return REGISTRY[name]
