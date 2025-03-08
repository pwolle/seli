from dataclasses import dataclass
from typing import Callable, TypeAlias

import jax


class Module:
    pass


LeafType: TypeAlias = None | bool | int | float | str | jax.Array
NodeType: TypeAlias = list | dict | Module


@dataclass
class ItemKey:
    key: str | int

    def get(self, obj):
        return obj[self.key]

    def set(self, obj, value):
        obj[self.key] = value

    def __repr__(self):
        return f"[{self.key}]"


@dataclass
class AttrKey:
    attr: str

    def get(self, obj):
        return getattr(obj, self.attr)

    def set(self, obj, value):
        setattr(obj, self.attr, value)

    def __repr__(self):
        return f".{self.attr}"


@dataclass
class PathKey:
    path: list[ItemKey]

    def __add__(self, item: ItemKey) -> "PathKey":
        return PathKey(self.path + [item])

    def get(self, obj):
        for item in self.path:
            obj = item.get(obj)

        return obj

    def set(self, obj, value):
        # Handle empty path
        if not self.path:
            return

        # Navigate to the parent object, stopping before the last item
        parent = obj
        for item in self.path[:-1]:
            parent = item.get(parent)

        # Set the value using the last item on the parent object
        last_item = self.path[-1]
        last_item.set(parent, value)

    def __repr__(self):
        return "".join(str(item) for item in self.path)


def dfs_map(
    obj: NodeType | LeafType,
    fun: Callable[[NodeType | LeafType], NodeType | LeafType],
    path: PathKey | None = None,
    ids: dict[int, NodeType | LeafType] | None = None,
) -> NodeType | LeafType:
    path = path or PathKey([])
    ids = ids or {}

    if id(obj) in ids:
        return ids[id(obj)]

    obj_fun = fun(obj)
    ids[id(obj)] = obj_fun

    if isinstance(obj_fun, LeafType):
        return obj_fun

    if isinstance(obj_fun, dict):
        assert all(isinstance(key, str) for key in obj_fun.keys())

        obj_new = {}

        for key, value in sorted(obj.items(), key=lambda x: x[0]):
            obj_new[key] = dfs_map(value, fun, path + ItemKey(key), ids)

        return obj_new

    if isinstance(obj_fun, list):
        obj_new = []

        for i, value in enumerate(obj_fun):
            obj_new.append(dfs_map(value, fun, path + ItemKey(i), ids))

        return obj_new

    if isinstance(obj_fun, Module):
        keys = []
        if hasattr(obj_fun, "__dict__"):
            keys.extend(obj_fun.__dict__.keys())

        if hasattr(obj_fun, "__slots__"):
            keys.extend(obj_fun.__slots__)

        obj_new = object.__new__(Module)

        for key in sorted(keys):
            value = getattr(obj_fun, key)
            setattr(obj_new, key, dfs_map(value, fun, path + AttrKey(key), ids))

        return obj_new

    raise ValueError(f"Unknown object type: {type(obj)}")
