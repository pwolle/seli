"""
Parametrized linear and affine transformations layers.
"""

from jaxtyping import Array, Float, jaxtyped

from seli.core._module import Module
from seli.core._typecheck import typecheck
from seli.net._init import Normal, Zeros
from seli.net._param import Param

__all__ = [
    "Linear",
    "Bias",
    "Scale",
]


class Linear(Module, name="net.Linear"):
    """
    Apply a learnable linear transformation to last axis of the input.

    Parameters
    ---
    key: PRNGKeyArray
        Key to use for random initialization.

    dim: int
        Dimensionality of the output. The input dimension is inferred from
        the last axis of the first input.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.weight = Param(init=Normal(init="He"))

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim_in"],
    ) -> Float[Array, "*batch {self.dim}"]:
        w = self.weight((x.shape[-1], self.dim), x.dtype)
        return x @ w

    @property
    def dim_in(self) -> int | None:
        """
        Return the input dimension of the module. If the module does not have
        a fixed input dimension yet, return None.
        """
        if not self.weight.initialized:
            return None

        return self.weight.value.shape[0]


class Bias(Module, name="net.Bias"):
    """
    Add a learnable bias to the last axis of the input.
    """

    def __init__(self) -> None:
        self.bias = Param(init=Zeros())

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        b = self.bias((x.shape[-1],), x.dtype)
        return x + b

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the bias. If the bias has not been initialized
        yet, return None.
        """
        if not self.bias.initialized:
            return None

        return self.bias.value.shape[0]


class Affine(Module, name="net.Affine"):
    """
    Apply a learnable linear transformation followed by a learnable bias.

    Parameters
    ---
    dim: int
        The output dimension of the linear transformation. The input dimension
        is inferred from the last axis of the first input.
    """

    def __init__(self, dim: int) -> None:
        self.linear = Linear(dim)
        self.bias = Bias()

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim_in"],
    ) -> Float[Array, "*batch dim"]:
        return self.bias(self.linear(x))

    @property
    def dim_in(self) -> int | None:
        return self.linear.dim_in


class Scale(Module, name="net.Scale"):
    """
    Scale the last axis of the input by a learnable vector.

    Parameters
    ---
    offset: bool
        If True the input will be scaled by `1 + scale` instead of `scale`.
        The scale is initialized to 0.
    """

    def __init__(self, offset: float = 1) -> None:
        self.offset = offset
        self.scale = Param(init=Zeros())

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        s = self.scale((x.shape[-1],), x.dtype)
        return x * (s + self.offset)

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the scale. If the scale has not been initialized
        yet, return None.
        """
        if not self.scale.initialized:
            return None

        return self.scale.value.shape[0]
