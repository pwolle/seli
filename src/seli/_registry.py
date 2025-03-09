import logging

import jax.tree_util as jtu

logger = logging.getLogger(__name__)


REGISTRY = {}


class ModuleBase:
    def __init_subclass__(
        cls,
        name: str | None = None,
        overwrite: bool | None = None,
    ):
        if hasattr(cls, "tree_flatten") and hasattr(cls, "tree_unflatten"):
            cls = jtu.register_pytree_node_class(cls)

        if name is not None:
            register_module(name, cls, overwrite=overwrite)


def register_module(
    name: str,
    module: type,
    overwrite: bool = False,
) -> None:
    if not overwrite and name in REGISTRY:
        logger.warning(f"Module {name} already registered, skipping")
        return

    REGISTRY[name] = module
