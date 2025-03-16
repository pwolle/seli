from collections.abc import Callable
from functools import partial, wraps
from typing import Any, ParamSpec, TypeVar

import jax
import jax.numpy as jnp

from seli.core._module import AttrKey, NodeType, PathKey, dfs_map
from seli.core._typecheck import typecheck
from seli.net._param import Param

P = ParamSpec("P")
T = TypeVar("T")


FLOAT_TYPES = (
    jnp.float16,
    jnp.float32,
    jnp.float64,
    jnp.bfloat16,
)


@typecheck
def get_arrays(
    module: NodeType,
    collection: str | None = None,
) -> dict[str, jax.Array]:
    """
    Extract arrays from parameters in a module.

    This function traverses the module and extracts all arrays from Param
    objects, optionally filtering by collection. It returns a copy of the
    module with the array values set to None, and a dictionary mapping path
    strings to arrays.

    Parameters
    ---
    module : NodeType
        The module to extract arrays from.

    collection : str | None, default=None
        If provided, only extract arrays from Param objects with this
        collection.
        If None, extract arrays from all Param objects.

    Returns
    ---
    dict[str, jax.Array]
        A dictionary mapping path strings to arrays
    """
    arrays_paths: dict[PathKey, jax.Array] = {}

    def fun(path: PathKey, obj: NodeType):
        if not isinstance(obj, jax.Array):
            return obj

        # if no collection is provided, return all arrays
        if collection is None:
            arrays_paths[path] = obj
            return obj

        assert collection is not None

        # if a collection is provided, the base object cannot be the array
        # in a Param object
        if not path.path:
            return obj

        # if the last item is not the value attribute, return the object
        if path[-1] != AttrKey("value"):
            return obj

        # get the parent Param object
        parent_path = path[:-1]
        parent = parent_path.get(module)

        # if the parent is not a Param object, return the object
        if not isinstance(parent, Param):
            return obj

        # if the collection does not match, return the object
        if parent.collection != collection:
            return obj

        arrays_paths[path] = obj
        return obj

    # does not create any side effects
    module = dfs_map(module, fun)

    arrays = {repr(path): arr for path, arr in arrays_paths.items()}
    return arrays


@typecheck
def set_arrays(
    module: NodeType,
    arrays: dict[str, jax.Array],
) -> NodeType:
    """
    Set arrays back into parameters in a module.

    This function takes a module and a dictionary of arrays, and sets the
    arrays back into the corresponding Param objects in the module. The paths
    in the dictionary should match those returned by get_arrays.

    Parameters
    ---
    module : NodeType
        The module to set arrays into.

    arrays : dict[str, jax.Array]
        A dictionary mapping path strings to arrays.

    Returns
    ---
    NodeType
        A new module with the arrays set into the parameters.

    Raises
    ---
    ValueError
        If a path in the arrays dictionary doesn't point to a Param object.
    """
    array_paths = {PathKey.from_str(path): arr for path, arr in arrays.items()}

    if PathKey([]) in array_paths:
        if len(arrays) != 1:
            error = f"Base object is set to an array, but got path {arrays}"
            raise ValueError(error)

        return array_paths[PathKey([])]

    # perform memory efficient copy
    module = dfs_map(module)

    for path, arr in array_paths.items():
        path.set(module, arr)

    return module


def grad(
    func: Callable[P, T],
    *,
    collection: str | None = None,
    has_aux: bool = False,
) -> Callable[P, Any]:
    """
    Create a function that computes gradients with respect to module
    parameters.

    This function wraps a loss function that takes a module as its first
    argument and returns a new function that computes the gradients of the loss
    with respect to the module's parameters.

    The gradient function extracts arrays from the module, computes gradients,
    and returns them in a dictionary mapping path strings to gradient arrays.

    Parameters
    ---
    func : Callable
        The function to compute gradients for. It should take a module as its
        first argument and return a scalar loss value.

    collection : str | None, default=None
        If provided, only extract arrays from Param objects with this
        collection.
        If None, extract arrays from all Param objects.

    has_aux : bool, default=False
        Whether the function returns auxiliary data. If True, the function
        should return a tuple (loss, aux_data), where loss is a scalar and
        aux_data can be any type.

    Returns
    ---
    Callable
        A new function that takes the same arguments as func but returns
        gradients with respect to the module's parameters. If has_aux is True,
        it returns a tuple (gradients, aux_data).

    Examples
    ---
    >>> def loss_fn(module, x, y):
    ...     pred = module(x)
    ...     return ((pred - y) ** 2).mean()
    >>> grad_fn = grad(loss_fn)
    >>> gradients = grad_fn(module, x, y)
    """

    @wraps(func)
    def wrap_fn(module: NodeType, *args: P.args, **kwargs: P.kwargs) -> Any:
        arrays = get_arrays(module, collection)
        arrays = {k: v for k, v in arrays.items() if v.dtype in FLOAT_TYPES}

        if not arrays:
            error = f"No float arrays in module with collection {collection}"
            raise ValueError(error)

        @partial(jax.grad, has_aux=has_aux)
        def grad_fn(
            arrays: dict[str, jax.Array],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Any:
            module_ = set_arrays(module, arrays)
            return func(module_, *args, **kwargs)

        return grad_fn(arrays, *args, **kwargs)

    return wrap_fn


def value_and_grad(
    func: Callable[P, T],
    *,
    collection: str | None = None,
    has_aux: bool = False,
) -> Callable[P, Any]:
    """
    Create a function that computes gradients with respect to module
    parameters.

    This function wraps a loss function that takes a module as its first
    argument and returns a new function that computes the gradients of the loss
    with respect to the module's parameters.

    The gradient function extracts arrays from the module, computes gradients,
    and returns them in a dictionary mapping path strings to gradient arrays.

    Parameters
    ---
    func : Callable
        The function to compute gradients for. It should take a module as its
        first argument and return a scalar loss value.

    collection : str | None, default=None
        If provided, only extract arrays from Param objects with this
        collection.
        If None, extract arrays from all Param objects.

    has_aux : bool, default=False
        Whether the function returns auxiliary data. If True, the function
        should return a tuple (loss, aux_data), where loss is a scalar and
        aux_data can be any type.

    Returns
    ---
    Callable
        A new function that takes the same arguments as func but returns
        values and gradients with respect to the module's parameters.
        If has_aux is True, it returns a tuple (gradients, aux_data).

    Examples
    ---
    >>> def loss_fn(module, x, y):
    ...     pred = module(x)
    ...     return ((pred - y) ** 2).mean()
    >>> grad_fn = grad(loss_fn)
    >>> value, gradients = grad_fn(module, x, y)
    """

    @wraps(func)
    def wrap_fn(module: NodeType, *args: P.args, **kwargs: P.kwargs) -> Any:
        arrays = get_arrays(module, collection)
        arrays = {k: v for k, v in arrays.items() if v.dtype in FLOAT_TYPES}

        if not arrays:
            error = f"No float arrays in module with collection {collection}"
            raise ValueError(error)

        @partial(jax.value_and_grad, has_aux=has_aux)
        def grad_fn(
            arrays: dict[str, jax.Array],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Any:
            module_ = set_arrays(module, arrays)
            return func(module_, *args, **kwargs)

        return grad_fn(arrays, *args, **kwargs)

    return wrap_fn
