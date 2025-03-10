from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

from seli.core._jit import jit
from seli.core._module import Module

T = TypeVar("T")
P = ParamSpec("P")


# a function which applies a module to its arguments
# this is needed to have the jax.jit compilation cache all in one place
@jit
def _filter_jit_apply(
    module: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return module(*args, **kwargs)


class Jit(Module, Generic[P, T], name="net.Jit"):
    """
    Wrapper for just-in-time compiling the __call__ method of the given module.
    """

    module: Callable[P, T]

    def __init__(self, module: Callable[P, T]):
        self.module = module

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return _filter_jit_apply(self.module, *args, **kwargs)


class Partial(Module, Generic[T], name="net.Partial"):
    """
    Fix parts of the arguments of a callable module. Can be used in a similar
    manner to `functools.partial`.
    """

    def __init__(
        self,
        module: Callable[..., T],
        *args,
        **kwargs,
    ):
        self.func = module
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs) -> T:
        return self.func(*self.args, *args, **{**self.kwargs, **kwargs})
