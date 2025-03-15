from collections.abc import Callable
from functools import partial, wraps
from typing import Any, ParamSpec, TypeVar

import jax

from seli.core._module import NodeType, PathKey, dfs_map
from seli.core._typecheck import typecheck
from seli.net._param import Param

P = ParamSpec("P")
T = TypeVar("T")


@typecheck
def get_arrays(
    module: NodeType,
    collection: str | None = None,
) -> tuple[NodeType, dict[str, jax.Array]]:
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
    tuple[NodeType, dict[str, jax.Array]]
        A tuple containing:
        - A copy of the module with array values set to None
        - A dictionary mapping path strings to arrays
    """
    arrays_paths: dict[PathKey, jax.Array] = {}

    def fun(path: PathKey, obj: NodeType):
        if isinstance(obj, Param):
            if collection is None or collection == obj.collection:
                arrays_paths[path] = obj.value
                obj.value = None

        return obj

    module = dfs_map(module, fun)

    arrays = {repr(path): arr for path, arr in arrays_paths.items()}
    return module, arrays


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
    module = dfs_map(module)  # perform memory efficient copy

    for path, array in arrays.items():
        path = PathKey.from_str(path)
        param = path.get(module)

        if not isinstance(param, Param):
            raise ValueError(f"Expected Param, got {type(param)}")

        param.value = array

    return module


def grad(func: Callable[P, T], has_aux: bool = False) -> Callable[P, Any]:
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
        module, arrays = get_arrays(module)

        @partial(jax.grad, has_aux=has_aux)
        def grad_fn(
            arrays: dict[str, jax.Array],
            module: NodeType,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Any:
            module = set_arrays(module, arrays)
            return func(module, *args, **kwargs)

        return grad_fn(arrays, module, *args, **kwargs)

    return wrap_fn
