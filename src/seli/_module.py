from dataclasses import dataclass
from typing import Callable, TypeAlias

import jax
import jax.tree_util as jtu


@jtu.register_pytree_node_class
class Module:
    def __hash__(self):
        flat = flat_path_dict(self)
        return hash(tuple(flat.items()))

    def __eq__(self, other):
        return flat_path_dict(self) == flat_path_dict(other)

    def tree_flatten(obj: "Module"):
        tree = to_tree(obj)
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
        cls,
        aux_data: tuple[list["PathKey"], "NodeType"],
        arrs_vals: list[jax.Array],
    ):
        arrs_keys, tree = aux_data
        obj = to_tree_inverse(tree)

        for path, child in zip(arrs_keys, arrs_vals):
            path.set(obj, child)

        return obj

    def __repr__(self):
        head = f"{self.__class__.__name__}(\n"
        for key, value in self.__dict__.items():
            head += f"  {key}: {str(value).replace('\n', '\n  ')}\n"

        head += ")"
        return head


LeafType: TypeAlias = None | bool | int | float | str | jax.Array
DeepType: TypeAlias = list | dict | Module
NodeType: TypeAlias = LeafType | DeepType


@dataclass(frozen=True)
class ItemKey(Module):
    """
    Key for accessing items using the [] operator.
    Used to access dictionary items by key or sequence items by index.
    """

    key: str | int

    def get(self, obj):
        return obj[self.key]

    def set(self, obj, value):
        obj[self.key] = value

    def __repr__(self):
        return f"[{self.key}]"

    # add sorting to allow deterministic traversal
    def __lt__(self, other):
        return keys_lt(self, other)


@dataclass(frozen=True)
class AttrKey(Module):
    """
    Key for accessing object attributes using the dot operator.
    Used to access attributes of an object using the dot notation (obj.attr).
    """

    key: str

    def get(self, obj):
        return getattr(obj, self.key)

    def set(self, obj, value):
        setattr(obj, self.key, value)

    def __repr__(self):
        return f".{self.key}"

    # add sorting to allow deterministic traversal
    def __lt__(self, other):
        return keys_lt(self, other)


def keys_lt(a: ItemKey | AttrKey, b: ItemKey | AttrKey) -> bool:
    if type(a) is not type(b):
        return isinstance(a, ItemKey)

    if type(a.key) is not type(b.key):
        return isinstance(a.key, int)

    return a.key < b.key


@dataclass(frozen=True)
class PathKey(Module):
    """
    Sequence of keys that enables access to nested data structures.
    Combines multiple ItemKey and AttrKey objects to navigate through nested
    objects, dictionaries, and sequences.
    """

    path: list[ItemKey | AttrKey]

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

    # add sorting to allow deterministic traversal
    def __lt__(self, other):
        return tuple(self.path) < tuple(other.path)

    def __hash__(self):
        return hash((type(self), tuple(self.path)))


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

    if id(obj) in refs:
        if refs_fun is None:
            return refs[id(obj)]

        return refs_fun(path, refs[id(obj)])

    obj_fun = fun(path, obj)
    refs[id(obj)] = obj_fun

    if isinstance(obj_fun, (LeafType, PathKey)):
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

    raise ValueError(f"Unknown object type: {type(obj)}")


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
