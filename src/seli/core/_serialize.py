import json
from collections.abc import Hashable
from pathlib import Path

import jax
import jax.numpy as jnp

from seli.core._module import (
    Module,
    NodeType,
    PathKey,
    dfs_map,
    to_tree,
    to_tree_inverse,
)
from seli.core._registry import (
    REGISTRY_INVERSE,
    is_registry_str,
    registry_obj,
    registry_str,
)
from seli.core._typecheck import typecheck

__all__ = [
    "ArrayPlaceholder",
    "save",
    "load",
    "to_arrays_and_json",
    "from_arrays_and_json",
]


@typecheck
class ArrayPlaceholder(Module, name="builtin.ArrayPlaceholder"):
    index: int

    def __init__(self, index: int):
        self.index = index


@typecheck
def to_arrays_and_json(obj: NodeType) -> tuple[list[jax.Array], str]:
    arrays = []

    def fun_arrays(_: PathKey, x: NodeType):
        if isinstance(x, jax.Array):
            arrays.append(x)
            return ArrayPlaceholder(len(arrays) - 1)

        return x

    def fun_modules(_: PathKey, x: NodeType):
        if isinstance(x, dict):
            assert "__class__" not in x, "dicts cannot have __class__"
            return x

        if not isinstance(x, Module):
            return x

        keys = []
        if hasattr(x, "__dict__"):
            keys.extend(x.__dict__.keys())

        if hasattr(x, "__slots__"):
            keys.extend(x.__slots__)

        as_dict = {key: getattr(x, key) for key in keys}
        as_dict["__class__"] = x.__class__
        return as_dict

    def fun_registry(_: PathKey, x: NodeType):
        if isinstance(x, (type(None), bool, int, float, str, list, dict)):
            return x

        assert isinstance(x, Hashable)
        assert x in REGISTRY_INVERSE, f"{x} not in {REGISTRY_INVERSE}"
        return registry_str(x)

    obj = to_tree(obj)
    obj = dfs_map(obj, fun_arrays)
    obj = dfs_map(obj, fun_modules)
    obj = dfs_map(obj, fun_registry)

    return arrays, json.dumps(obj)


@typecheck
def from_arrays_and_json(arrays: list[jax.Array], obj_json: str) -> NodeType:
    def fun_registry(_: PathKey, x: NodeType):
        if is_registry_str(x):
            return registry_obj(x)

        return x

    def fun_modules(_: PathKey, x: NodeType):
        if not isinstance(x, dict):
            return x

        if "__class__" not in x:
            return x

        cls = x.pop("__class__")
        assert issubclass(cls, Module)

        module = object.__new__(cls)
        for key, value in x.items():
            object.__setattr__(module, key, value)

        return module

    def fun_arrays(_: PathKey, x: NodeType):
        if isinstance(x, ArrayPlaceholder):
            return arrays[x.index]

        return x

    obj = json.loads(obj_json)
    obj = dfs_map(obj, fun_registry)
    obj = dfs_map(obj, fun_modules)
    obj = dfs_map(obj, fun_arrays)

    return to_tree_inverse(obj)


@typecheck
def save(path: str | Path, obj: NodeType) -> None:
    arrays, obj_json = to_arrays_and_json(obj)
    arrays.append(obj_json)
    jnp.savez(path, *arrays)


@typecheck
def load(path: str | Path) -> NodeType:
    arrays = list(jnp.load(path).values())

    # pop the last element, which is the json string
    # this one cannot be converted to a jax.Array
    obj_json = str(arrays.pop(-1))

    # convert all arrays to jax, as those are expected by dfs_map used in
    # from_arrays_and_json, since they are immutable
    arrays = [jnp.array(array) for array in arrays]

    obj = from_arrays_and_json(arrays, obj_json)
    return obj
