"""
Normalization layers.
"""

import jax.lax as lax
from jaxtyping import Array, Float, jaxtyped

from seli.core._module import Module
from seli.core._typecheck import typecheck
from seli.net._init import Ones, Zeros
from seli.net._param import Param

__all__ = [
    "LayerNorm",
    "RMSNorm",
]


class LayerNorm(Module, name="net.LayerNorm"):
    """
    Normalize the input along the last axis. Then add a learnable offset and
    scale by a learnable weight along the last axis.

    Parameters
    ---
    eps: float
        Epsilon value for numerical stability.

    offset: bool
        Whether to add 1 to the scale weight before multiplying. If true, the
        model is initialized to the identity function. If false, the model is
        initialized to the constant zero function.
    """

    @typecheck
    def __init__(
        self,
        eps: float = 1e-6,
        offset: float | int = 1,
    ) -> None:
        self.eps = eps
        self.offset = offset

        self.weight = Param(init=Zeros())
        self.bias = Param(init=Zeros())

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        w = self.weight((x.shape[-1],), x.dtype)
        b = self.bias((x.shape[-1],), x.dtype)

        m = x.mean(axis=-1, keepdims=True)
        x = x - m

        v = x.var(axis=-1, keepdims=True)
        r = lax.rsqrt(v + self.eps)
        x = x * r

        x = x * (w + self.offset)
        x = x + b
        return x


class RMSNorm(Module, name="net.RMSNorm"):
    """
    Scale the input by the reciprocal of the root mean square along the last
    axis. Then add a learnable offset and scale by a learnable weight along the
    last axis.

    Parameters
    ---
    eps: float
        Epsilon value for numerical stability.

    axis: int
        The axis to calculate the root mean square.
    """

    @typecheck
    def __init__(
        self,
        eps: float = 1e-6,
        offset: float | int = 1,
    ) -> None:
        self.eps = eps
        self.offset = offset

        self.weight = Param(init=Ones())
        self.bias = Param(init=Zeros())

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        w = self.weight((x.shape[-1],), x.dtype)
        b = self.bias((x.shape[-1],), x.dtype)

        v = (x * x).mean(axis=-1, keepdims=True)
        r = lax.rsqrt(v + self.eps)
        x = x * r

        x = x * (w + self.offset)
        x = x + b
        return x
