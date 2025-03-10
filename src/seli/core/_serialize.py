"""
This module provides functionality for serializing and deserializing modules.
The core idea is to take out all arrays from the module and then serialize the
module structure as a JSON string with the help of the registry.
"""

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
    """
    Placeholder for an array that will be serialized and deserialized later.
    """

    index: int

    def __init__(self, index: int):
        self.index = index


@typecheck
def to_arrays_and_json(obj: NodeType) -> tuple[list[jax.Array], str]:
    """
    Serialize a nested structure of modules and arrays and other containers
    to a JSON string and a list of arrays.

    This works by traversing the nested structure and replacing arrays with
    ArrayPlaceholder objects. The ArrayPlaceholder objects are then replaced
    with the actual arrays during deserialization.

    Then the Modules are replaced with dictionaries containing the module
    class and all its attributes. The class is stored as "__class__" in the
    dictionary and turned into a string using the registry.

    Parameters
    ---
    obj: NodeType
        The nested structure to serialize.

    Returns
    ---
    tuple[list[jax.Array], str]
        A tuple containing a list of arrays and a JSON string.
    """
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
    """
    Deserialize a nested structure of modules and arrays and other containers
    from a JSON string and a list of arrays. Inverse of `to_arrays_and_json`.

    Parameters
    ---
    arrays: list[jax.Array]
        The list of arrays to deserialize.

    obj_json: str
        The JSON string to deserialize.

    Returns
    ---
    NodeType
        The deserialized nested structure.
    """

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
    """
    Save a nested structure of modules and arrays and other containers to a
    file. All leaves of the nested structure must be serializable, i.e.
    standard json serializable objects, or part of the registry. All modules
    must be registered, i.e. their type is in the registry. Modules can be
    registered by specifying the `name` parameter when inheriting from
    `Module`.

    Parameters
    ---
    path: str | Path
        The path to save the serialized object to.

    obj: NodeType
        The nested structure to serialize.
    """
    arrays, obj_json = to_arrays_and_json(obj)
    arrays.append(obj_json)
    jnp.savez(path, *arrays)


@typecheck
def load(path: str | Path) -> NodeType:
    """
    Load a nested structure of modules and arrays and other containers from a
    file. Inverse of `save`.

    Parameters
    ---
    path: str | Path
        The path to load the serialized object from.

    Returns
    ---
    NodeType
        The deserialized nested structure.
    """
    arrays = list(jnp.load(path).values())

    # pop the last element, which is the json string
    # this one cannot be converted to a jax.Array
    obj_json = str(arrays.pop(-1))

    # convert all arrays to jax, as those are expected by dfs_map used in
    # from_arrays_and_json, since they are immutable
    arrays = [jnp.array(array) for array in arrays]

    obj = from_arrays_and_json(arrays, obj_json)
    return obj
