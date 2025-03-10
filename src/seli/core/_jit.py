from functools import partial, wraps
from typing import Any, Callable, ParamSpec, TypeVar

import jax

from seli.core._module import Module

T = TypeVar("T")
P = ParamSpec("P")


class Arguments(Module, name="builtin.Arguments"):
    """
    Wrapper for the arguments used to call a function.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Result(Module, name="builtin.Result"):
    """
    Wrapper for the result of a function.
    """

    value: Any

    def __init__(self, value: Any) -> None:
        self.value = value


@partial(jax.jit, static_argnames=("function",))
def _apply_filter_jit(module: Arguments, function: Any) -> Any:
    result = function(*module.args, **module.kwargs)
    return Result(result)


def filter_jit(function: Callable[P, T]) -> Callable[P, T]:
    """
    Just-in-time compiling functions.

    This is a drop-in replacement for jax.jit, that traces shared references
    between the different arguments. Using jax.jit with references shared
    between arguments will untie the references in the body of the function
    and the output.

    The function will not recompile if only the values inside of the jax.Arrays
    change.

    Parameters
    ---
    function: Callable
        Function to apply the jax.jit to.

    Returns
    ---
    compiled: Callable
        The compiled function. This function takes the same arguments as the
        original function.
    """

    @wraps(function)
    def compiled(*args: P.args, **kwargs: P.kwargs) -> T:
        module = Arguments(args, kwargs)
        return _apply_filter_jit(module, function).value

    return compiled
