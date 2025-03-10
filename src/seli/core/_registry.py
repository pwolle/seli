"""
This module provides a system for registering modules. This is useful for
serializing and deserializing modules without having to know the module
structure beforehand.

See the `_serialize` module for more information on how this is used. This is
a separate module, because the registry is needed in the module system, but
the module system is needed in the serialize module.
"""

import logging
from collections.abc import Hashable
from typing import Any

import jax.nn as jnn
import jax.numpy as jnp
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


# make common activation functions serializable
registry_add("jax.nn.celu", jnn.celu)
registry_add("jax.nn.elu", jnn.elu)
registry_add("jax.nn.gelu", jnn.gelu)
registry_add("jax.nn.glu", jnn.glu)
registry_add("jax.nn.hard_sigmoid", jnn.hard_sigmoid)
registry_add("jax.nn.hard_silu", jnn.hard_silu)
registry_add("jax.nn.hard_swish", jnn.hard_swish)
registry_add("jax.nn.hard_tanh", jnn.hard_tanh)
registry_add("jax.nn.leaky_relu", jnn.leaky_relu)
registry_add("jax.nn.log_sigmoid", jnn.log_sigmoid)
registry_add("jax.nn.log_softmax", jnn.log_softmax)
registry_add("jax.nn.logsumexp", jnn.logsumexp)
registry_add("jax.nn.standardize", jnn.standardize)
registry_add("jax.nn.relu", jnn.relu)
registry_add("jax.nn.relu6", jnn.relu6)
registry_add("jax.nn.selu", jnn.selu)
registry_add("jax.nn.sigmoid", jnn.sigmoid)
registry_add("jax.nn.soft_sign", jnn.soft_sign)
registry_add("jax.nn.softmax", jnn.softmax)
registry_add("jax.nn.softplus", jnn.softplus)
registry_add("jax.nn.sparse_plus", jnn.sparse_plus)
registry_add("jax.nn.sparse_sigmoid", jnn.sparse_sigmoid)
registry_add("jax.nn.silu", jnn.silu)
registry_add("jax.nn.swish", jnn.swish)
registry_add("jax.nn.squareplus", jnn.squareplus)
registry_add("jax.nn.mish", jnn.mish)

# make the jax data types serializable
registry_add("jax.numpy.bool_", jnp.bool_)
registry_add("jax.numpy.complex64", jnp.complex64)
registry_add("jax.numpy.complex128", jnp.complex128)
registry_add("jax.numpy.float16", jnp.float16)
registry_add("jax.numpy.float32", jnp.float32)
registry_add("jax.numpy.float64", jnp.float64)
registry_add("jax.numpy.bfloat16", jnp.bfloat16)
registry_add("jax.numpy.int8", jnp.int8)
registry_add("jax.numpy.int16", jnp.int16)
registry_add("jax.numpy.int32", jnp.int32)
registry_add("jax.numpy.int64", jnp.int64)
registry_add("jax.numpy.uint8", jnp.uint8)
registry_add("jax.numpy.uint16", jnp.uint16)
registry_add("jax.numpy.uint32", jnp.uint32)
registry_add("jax.numpy.uint64", jnp.uint64)
