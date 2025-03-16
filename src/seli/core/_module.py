"""
Modules are the core data structure for organising state. This module provides
the core functionality for creating, traversing, and modifying modules.
"""

import re
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Self, TypeAlias

import jax

from seli.core._registry import REGISTRY_INVERSE, ModuleBase
from seli.core._typecheck import typecheck
from seli.core._utils import array_summary

__all__ = [
    "Module",
    "ItemKey",
    "AttrKey",
    "PathKey",
    "NodeType",
    "LeafType",
    "to_tree",
    "to_tree_inverse",
    "flat_path_dict",
    "dfs_map",
]


@typecheck
class Module(ModuleBase, name="builtin.Module"):
    """
    Base class for all modules. Modules can be used to implement parameterized
    functions like neural networks.

    Modules are PyTrees, which means they can be flattened and unflattened
    using JAX's tree_util functions.

    The flattening will automatically go through all attributes including slots.
    Submodules as well as dictionaries, lists, and arrays will also be
    recursively flattened.

    If the module and its children do not contain arrays, the module supports
    hashing and equality checking. These checks even respect the structure of
    shared references.
    """

    def __hash__(self):
        flat = flat_path_dict(self)
        return hash(tuple(flat.items()))

    def __eq__(self, other):
        return flat_path_dict(self) == flat_path_dict(other)

    def tree_flatten(
        self,
    ) -> tuple[list[jax.Array], tuple[list["PathKey"], "NodeType"]]:
        tree = to_tree(self)
        arrs: dict[PathKey, jax.Array] = {}

        def get_arrs(path: PathKey, obj: NodeType):
            if isinstance(obj, jax.Array):
                arrs[path] = obj
                return None

            return obj

        tree = dfs_map(tree, get_arrs)

        arrs_keys = list(arrs.keys())
        arrs_vals = [arrs[key] for key in arrs_keys]

        return arrs_vals, (arrs_keys, tree)

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        aux_data: tuple[list["PathKey"], "NodeType"],
        arrs_vals: Sequence[jax.Array | jax.ShapeDtypeStruct],
    ) -> Self:
        arrs_keys, tree = aux_data
        obj = to_tree_inverse(tree)

        for path, child in zip(arrs_keys, arrs_vals):
            path.set(obj, child)

        return obj

    def __repr__(self) -> str:
        return node_repr(self)


LeafType: TypeAlias = (
    None | bool | int | float | str | type | jax.Array | jax.ShapeDtypeStruct
)
DeepType: TypeAlias = list | dict | Module
NodeType: TypeAlias = LeafType | DeepType | Any


@typecheck
class ItemKey(Module, name="builtin.ItemKey"):
    """
    Key for accessing items using the [] operator.
    Used to access dictionary items by key or sequence items by index.
    """

    key: str | int

    def __init__(self, key: str | int) -> None:
        self.key = key

    def get(self, obj: dict | list) -> Any:
        return obj[self.key]

    def set(self, obj: dict | list, value: Any) -> None:
        obj[self.key] = value

    def __repr__(self):
        return f"[{self.key!r}]"

    # add sorting to allow deterministic traversal
    def __lt__(self, other: "ItemKey | AttrKey") -> bool:
        return _keys_lt(self, other)

    def __hash__(self) -> int:
        return hash((type(self), self.key))

    def __eq__(self, other: "ItemKey | AttrKey") -> bool:
        return isinstance(other, ItemKey) and self.key == other.key

    @classmethod
    def from_str(cls, s: str) -> "ItemKey":
        if not s.startswith("[") and not s.endswith("]"):
            raise ValueError(f"Invalid item key string: {s}")

        key = s[1:-1]
        if key.startswith("'") and key.endswith("'"):
            key = key[1:-1]
            return cls(key)

        if not key.isdigit():
            raise ValueError(f"Invalid item key string: `{key}`")

        return cls(int(key))


@typecheck
class AttrKey(ItemKey, name="builtin.AttrKey"):
    """
    Key for accessing object attributes using the dot operator.
    Used to access attributes of an object using the dot notation (obj.attr).
    """

    key: str

    def __init__(self, key: str) -> None:
        self.key = key

    def get(self, obj: Any) -> Any:
        return getattr(obj, self.key)

    def set(self, obj: Any, value: Any) -> None:
        object.__setattr__(obj, self.key, value)

    def __repr__(self):
        return f".{self.key}"

    @classmethod
    def from_str(cls, s: str) -> "AttrKey":
        if not s.startswith(".") or len(s) < 2:
            raise ValueError(f"Invalid attribute key string: {s}")

        if not s[1:].isidentifier():
            raise ValueError(f"Invalid attribute key string: {s}")

        return cls(s[1:])


@typecheck
def _keys_lt(a: ItemKey | AttrKey, b: ItemKey | AttrKey) -> bool:
    if type(a) is not type(b):
        return type(a) is ItemKey

    if type(a.key) is not type(b.key):
        return isinstance(a.key, int)

    return a.key < b.key


@typecheck
class PathKey(Module, name="builtin.PathKey"):
    """
    Sequence of keys that enables access to nested data structures.
    Combines multiple ItemKey and AttrKey objects to navigate through nested
    objects, dictionaries, and sequences.
    """

    path: list[ItemKey | AttrKey]

    def __init__(self, path: list[ItemKey | AttrKey]) -> None:
        self.path = path

    def __add__(self, item: ItemKey | AttrKey) -> "PathKey":
        return PathKey(self.path + [item])

    def get(self, obj):
        for item in self.path:
            obj = item.get(obj)

        return obj

    def set(self, obj: DeepType, value: NodeType):
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
        return "".join(repr(item) for item in self.path)

    # add sorting to allow deterministic traversal
    def __lt__(self, other):
        return tuple(self.path) < tuple(other.path)

    def __hash__(self):
        return hash((type(self), tuple(self.path)))

    def __eq__(self, other):
        return isinstance(other, PathKey) and self.path == other.path

    @classmethod
    def from_str(cls, s: str) -> "PathKey":
        key_parts = re.split(r"(?=[.\[])", s)
        keys = []

        for part in key_parts:
            # regular expression might produce emtpy string at the start or end
            if not part:
                continue

            if part.startswith("."):
                keys.append(AttrKey.from_str(part))
                continue

            if part.startswith("["):
                keys.append(ItemKey.from_str(part))
                continue

            raise ValueError(f"Invalid path key string: {s}")

        return cls(keys)

    def __getitem__(self, item: int | slice) -> "ItemKey | AttrKey | PathKey":
        if isinstance(item, slice):
            return type(self)(self.path[item])

        return self.path[item]


def dfs_map(
    obj: NodeType,
    fun: Callable[[PathKey, NodeType], NodeType] = lambda _, x: x,
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
    each element. It builds a new structure with the same shape as the
    original, but with transformed values. During traversal, it tracks the path
    to each element and handles circular references to prevent infinite
    recursion.

    Parameters
    ---
    obj: NodeType
        The object to traverse, which can be a dictionary, list, Module, or a
        leaf value.
        - Dictionaries must have string keys.
        - Lists are traversed in order.
        - Module objects have their attributes traversed alphabetically.
        - All other types are treated as leaf values.

    fun: Callable[[PathKey, NodeType], NodeType]
        A transformation function to apply to each element in the structure.
        The function should accept two arguments:
            - path: A PathKey object representing the current path
            - x: The current element being processed
        And return a transformed version of the element.

    refs: dict[int, NodeType] | None
        A dictionary mapping object IDs to their transformed
        versions. Used internally to track already-processed objects and
        handle circular references. Default is None (an empty dict will be
        created).

    path: PathKey | None
        A PathKey object representing the current path in the structure. Used
        for tracking position during recursive calls. Default is None (an empty
        PathKey will be created).

    refs_fun: Callable[[PathKey, NodeType], NodeType] | None
        A function to handle repeated references.
        When an object is encountered multiple times during traversal:
            - If refs_fun is None, the already-processed version is
                returned directly.
            - If refs_fun is provided, it's called with (path, processed_obj)
                to determine what to return for the repeated reference.
        Default is None.

    Returns
    ---
    A new structure with the same shape as the input, but with all elements
    transformed according to the provided function.

    Raises
    ---
    ValueError: If an object of an unsupported type is encountered.
        Supported types are: dictionaries, lists, Module objects, and leaf
            values.
        TypeError: If a dictionary with non-string keys is encountered.

    Notes
    ---
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

    if id(obj) in refs and not isinstance(obj, LeafType):
        if refs_fun is None:
            return refs[id(obj)]

        return refs_fun(path, refs[id(obj)])

    obj_fun = fun(path, obj)
    refs[id(obj)] = obj_fun

    if isinstance(obj_fun, LeafType):
        return obj_fun

    # if object is registered it is also a valid type, since we can covert it
    # to a string and back, we need to test for hashability and non-module
    # otherwise we cannot perform the isin check, the obj_fun may not be a
    # Module, since Module.__hash__ would be a RecursionError.
    if isinstance(obj_fun, Hashable) and not isinstance(obj_fun, Module):
        if obj_fun in REGISTRY_INVERSE:
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

        obj_new = object.__new__(type(obj_fun))

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

    raise ValueError(f"Unknown object type: {type(obj_fun)}, {obj_fun}")


def to_tree(obj: NodeType):
    """
    Convert shared/cyclic references into a PathKeys, the result is a tree.

    This function transforms complex nested data structures that may contain
    shared references (the same object referenced multiple times) or cyclic
    references (loops in the reference graph) into a tree structure. Instead
    of maintaining the actual shared or cyclic references, it replaces them
    with path references.

    Parameters
    ---
    obj : NodeType
        The input object to convert to a tree. Can be any supported type:
        dictionaries, lists, Module objects, or leaf values (None, bool,
        int, float, str, or jax.Array).

    Returns
    ---
    NodeType
        A tree-structured version of the input, with all shared and
        cyclic references replaced by path references.

    Notes
    ---
    - This function is useful for serializing complex object graphs or
      visualizing structures with cycles.
    - Path references can be used to reconstruct the original structure
      if needed.
    - The function uses dfs_map internally to traverse the structure.
    """
    id_to_path: dict[int, PathKey] = {}

    def fun(path: PathKey, obj: NodeType):
        id_to_path[id(obj)] = path
        return obj

    def refs_fun(_: PathKey, obj: NodeType):
        return id_to_path[id(obj)]

    return dfs_map(obj, fun, refs_fun=refs_fun)


def to_tree_inverse(obj: NodeType):
    """
    Reconstructs the original object structure from a tree produced by to_tree.

    This function is the inverse operation of to_tree. It takes a tree structure
    where shared or cyclic references have been replaced with PathKey objects,
    and reconstructs the original structure by resolving those path references
    back into actual object references.

    Parameters
    ---
    obj : NodeType
        A tree structure, typically produced by to_tree, where shared or cyclic
        references have been replaced with PathKey objects pointing to their
        location in the tree.

    Returns
    ---
    NodeType
        The reconstructed object structure with all path references resolved
        back into actual object references, restoring the original shared
        references and cycles.

    Notes
    ---
    - This function reverses the transformation performed by to_tree
    - When a PathKey is encountered during traversal, it gets resolved by
      accessing the object at that path in the tree
    - The function uses dfs_map internally for traversal, similar to to_tree
    - While to_tree eliminates cycles by replacing them with path references,
      this function reintroduces those cycles
    """

    refs: dict[PathKey, PathKey] = {}

    def fun(path: PathKey, obj: NodeType):
        if isinstance(obj, PathKey):
            refs[path] = obj

        return obj

    obj = dfs_map(obj, fun, refs_fun=fun)

    for path, ref in refs.items():
        path.set(obj, ref.get(obj))

    return obj


def flat_path_dict(obj: NodeType):
    """
    Convert a nested object structure into a flat dictionary representation.

    This function transforms a potentially nested object into a flat dictionary
    where:
    - Each entry is keyed by a PathKey representing its location in the original
      structure
    - Leaf values and PathKey references are preserved directly
    - For non-leaf nodes, their class name is stored under a __class__ attribute
      key

    The resulting dictionary provides a serializable, deterministic
    representation of the object's structure that preserves paths and type
    information.

    Args:
        obj: The object to convert to a flat path dictionary

    Returns:
        A dictionary mapping PathKey objects to values, sorted by path for
        deterministic output
    """
    tree = to_tree(obj)
    nodes: dict[PathKey, NodeType] = {}

    def add_node(path, node: NodeType):
        if isinstance(node, (LeafType, PathKey)):
            nodes[path] = node
            return node

        nodes[path + AttrKey("__class__")] = type(node)
        return node

    dfs_map(tree, add_node)

    # sort dict by keys for deterministic output
    return dict(sorted(nodes.items(), key=lambda x: x[0]))


def node_repr(obj: NodeType, /, indent: str = " " * 4) -> str:
    """
    Generate a structured, readable string representation of nested objects.

    Creates a hierarchical string representation of Module objects and other
    complex nested structures with appropriate indentation. The function handles
    various types differently:

    - JAX arrays: Summarized using `array_summary`
    - PathKey objects: Displayed with their string representation
    - Lists: Formatted with each item on a new indented line
    - Dictionaries: Formatted with key-value pairs on indented lines
    - Module objects: Displayed with class name and attribute values
    - Other types: Using their native repr() representation

    Parameters
    ---
    obj : NodeType
        The object to represent as a string
    indent : str, default="    "
        The indentation string used for nested levels

    Returns
    ---
    str
        A formatted string representation of the object
    """
    obj = to_tree(obj)

    if isinstance(obj, jax.Array):
        return array_summary(obj)

    if isinstance(obj, PathKey):
        return f"<obj{obj!r}>"

    if isinstance(obj, list):
        head = "[\n"
        body = ""

        for item in obj:
            item_repr = node_repr(item, indent=indent)
            item_repr = item_repr.replace("\n", "\n" + indent)
            body += f"{indent}{item_repr},\n"

        tail = "]"
        return head + body + tail

    if isinstance(obj, dict):
        head = "{\n"
        body = ""

        for key, value in sorted(obj.items(), key=lambda x: x[0]):
            value_repr = node_repr(value, indent=indent)
            value_repr = value_repr.replace("\n", "\n" + indent)
            body += f"{indent}{key!r}: {value_repr},\n"

        tail = "}"
        return head + body + tail

    if isinstance(obj, Module):
        head = f"{obj.__class__.__name__}(\n"
        body = ""

        keys = []
        if hasattr(obj, "__dict__"):
            keys.extend(obj.__dict__.keys())

        if hasattr(obj, "__slots__"):
            keys.extend(obj.__slots__)

        for key in sorted(keys):
            value = getattr(obj, key)
            value_repr = node_repr(value, indent=indent)
            value_repr = value_repr.replace("\n", "\n" + indent)
            body += f"{indent}{key}={value_repr},\n"

        tail = ")"
        return head + body + tail

    return repr(obj)
