"""
Parametrized linear and affine transformations layers.
"""

import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from seli import Module, typecheck

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

    weight: Float[Array, "dim_in dim"] | None

    def __init__(self, key: PRNGKeyArray, dim: int) -> None:
        self.key = key
        self.dim = dim

        self.weight = None

    def _build(self, x) -> None:
        if self.weight is not None:
            return

        dim_in = x.shape[-1]
        glorot = dim_in**-0.5

        self.weight = jrn.uniform(
            self.key,
            (dim_in, self.dim),
            dtype=x.dtype,
            minval=-glorot,
            maxval=+glorot,
        )

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim_in"],
    ) -> Float[Array, "*batch {self.dim}"]:
        self._build(x)

        assert self.weight is not None
        return x @ self.weight

    @property
    def dim_in(self) -> int | None:
        """
        Return the input dimension of the module. If the module does not have
        a fixed input dimension yet, return None.
        """
        if self.weight is None:
            return None

        return self.weight.shape[0]


class Bias(Module, name="net.Bias"):
    """
    Add a learnable bias to the last axis of the input.

    Parameters
    ---
    key: PRNGKeyArray
        Key to use for the initialization.
    """

    bias: Float[Array, "dim"] | None

    def __init__(self, key: PRNGKeyArray) -> None:
        self.key = key
        self.bias = None

    def _build(self, x) -> None:
        if self.bias is not None:
            return

        dim_in = x.shape[-1]
        glorot = dim_in**-0.5

        self.bias = jrn.uniform(
            self.key,
            (dim_in,),
            dtype=x.dtype,
            minval=-glorot,
            maxval=glorot,
        )

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        self._build(x)
        assert self.bias is not None

        return x + self.bias

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the bias. If the bias has not been initialized
        yet, return None.
        """
        if self.bias is None:
            return None

        return self.bias.shape[0]


class Affine(Module, name="net.Affine"):
    """
    Apply a learnable linear transformation followed by a learnable bias.

    Parameters
    ---
    key: PRNGKeyArray
        Key to use for random initialization.

    dim: int
        The output dimension of the linear transformation. The input dimension
        is inferred from the last axis of the first input.
    """

    def __init__(self, key: PRNGKeyArray, dim: int) -> None:
        key_linear, key_bias = jrn.split(key)
        self.linear = Linear(key_linear, dim)
        self.bias = Bias(key_bias)

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

    scale: Float[Array, "dim"] | None

    def __init__(self, offset: float = 1) -> None:
        self.offset = offset
        self.scale = None

    def _build(self, x) -> None:
        if self.scale is not None:
            return

        self.scale = jnp.zeros((x.shape[-1],), x.dtype)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        self._build(x)
        assert self.scale is not None
        return x * (self.scale + self.offset)

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the scale. If the scale has not been initialized
        yet, return None.
        """
        if self.scale is None:
            return None

        return self.scale.shape[0]
