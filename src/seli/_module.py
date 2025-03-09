from dataclasses import dataclass
from typing import Callable, TypeAlias

import jax


class Module:
    pass


LeafType: TypeAlias = None | bool | int | float | str | jax.Array
DeepType: TypeAlias = list | dict | Module
NodeType: TypeAlias = LeafType | DeepType


@dataclass
class ItemKey(Module):
    key: str | int

    def get(self, obj):
        return obj[self.key]

    def set(self, obj, value):
        obj[self.key] = value

    def __repr__(self):
        return f"[{self.key}]"


@dataclass
class AttrKey(Module):
    attr: str

    def get(self, obj):
        return getattr(obj, self.attr)

    def set(self, obj, value):
        setattr(obj, self.attr, value)

    def __repr__(self):
        return f".{self.attr}"


@dataclass
class PathKey(Module):
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
    obj: NodeType,
    fun: Callable[[PathKey, NodeType], NodeType],
    *,
    refs: dict[int, NodeType] | None = None,
    path: PathKey | None = None,
    refs_fun: Callable[[PathKey, NodeType], NodeType] | None = None,
) -> DeepType | LeafType:
    """
    Performs a depth-first traversal of a nested data structure, applying a
    transformation function to each element.

    This function traverses dictionaries, lists, and Module objects recursively
    in a depth-first manner, applying the provided transformation function to
    each element. It builds a new structure with the same shape as the original,
    but with transformed values. During traversal, it tracks the path to each
    element and handles circular references to prevent infinite recursion.

    Args:
        obj: The object to traverse, which can be a dictionary, list, Module,
            or a leaf value.
            - Dictionaries must have string keys.
            - Lists are traversed in order.
            - Module objects have their attributes traversed alphabetically.
            - All other types are treated as leaf values.

        fun: A transformation function to apply to each element in the structure.
            The function should accept two arguments:
                - path: A PathKey object representing the current path
                - x: The current element being processed
            And return a transformed version of the element.

        refs: Optional. A dictionary mapping object IDs to their transformed
            versions. Used internally to track already-processed objects and
            handle circular references. Default is None (an empty dict will be
            created).

        path: Optional. A PathKey object representing the current path in the
            structure. Used for tracking position during recursive calls.
            Default is None (an empty PathKey will be created).

        refs_fun: Optional. A function to handle repeated references.
            When an object is encountered multiple times during traversal:
                - If refs_fun is None, the already-processed version is
                  returned directly.
                - If refs_fun is provided, it's called with (path, processed_obj)
                  to determine what to return for the repeated reference.
            Default is None.

    Returns:
        A new structure with the same shape as the input, but with all elements
        transformed according to the provided function.

    Raises:
        ValueError: If an object of an unsupported type is encountered.
            Supported types are: dictionaries, lists, Module objects, and leaf
            values.
        TypeError: If a dictionary with non-string keys is encountered.

    Notes:
        - The function preserves the structure of the original object while
          creating a new transformed copy.
        - Dictionary keys and Module attributes are processed in sorted order for
          deterministic traversal.
        - For circular references, the function uses the refs_fun parameter to
          determine how to handle them.
        - Module objects are created using object.__new__ without calling
          __init__, which may bypass important initialization logic.
        - The path parameter tracks the exact location of each element in the
          nested structure using:
            - ItemKey for dictionary keys and list indices
            - AttrKey for Module attributes
    """
    path = path or PathKey([])
    refs = refs or {}

    if id(obj) in refs:
        if refs_fun is None:
            return refs[id(obj)]

        return refs_fun(path, refs[id(obj)])

    obj_fun = fun(path, obj)
    refs[id(obj)] = obj_fun

    if isinstance(obj_fun, LeafType):
        return obj_fun

    if isinstance(obj_fun, dict):
        if not all(isinstance(key, str) for key in obj_fun.keys()):
            error = f"Dictionary keys must be strings got {obj_fun.keys()}"
            raise TypeError(error)

        obj_new = {}

        for key, value in sorted(obj_fun.items(), key=lambda x: x[0]):
            obj_new[key] = dfs_map(
                value,
                fun,
                path=path + ItemKey(key),
                refs=refs,
                refs_fun=refs_fun,
            )

        return obj_new

    if isinstance(obj_fun, list):
        obj_new = []

        for i, value in enumerate(obj_fun):
            obj_new.append(
                dfs_map(
                    value,
                    fun,
                    path=path + ItemKey(i),
                    refs=refs,
                    refs_fun=refs_fun,
                ),
            )
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
            setattr(
                obj_new,
                key,
                dfs_map(
                    value,
                    fun,
                    path=path + AttrKey(key),
                    refs=refs,
                    refs_fun=refs_fun,
                ),
            )

        return obj_new

    raise ValueError(f"Unknown object type: {type(obj)}")
