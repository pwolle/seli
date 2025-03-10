from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jrn
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from seli import Module, typecheck


class Initializer(Module, "initializers.Initializer"):
    def __call__(
        self,
        key: PRNGKeyArray,
        shape: Sequence[int],
    ) -> jax.Array:
        raise NotImplementedError("Initializers must be implemented")


@typecheck
class Zeros(Initializer, "initializers.Zeros"):
    def __call__(
        self,
        key: PRNGKeyArray,
        shape: Sequence[int],
        dtype: DTypeLike = jnp.float32,
    ) -> jax.Array:
        return jnp.zeros(shape, dtype)


@typecheck
class Ones(Initializer, "initializers.Ones"):
    def __call__(
        self,
        key: PRNGKeyArray,
        shape: Sequence[int],
        dtype: DTypeLike = jnp.float32,
    ) -> jax.Array:
        return jnp.ones(shape, dtype)


@typecheck
class Normal(Initializer, "initializers.Normal"):
    """
    Initialize weights from a normal distribution. Weights can be scaled by
    the number of input or output units, or by the sum of input and output,
    or by a custom factor. The methods provided are based on
    - He et al. (2015): https://arxiv.org/abs/1502.01852
    - Glorot & Bengio (2010): https://proceedings.mlr.press/v9/glorot10a.html
    """

    def __init__(
        self,
        scale: float | Literal["He", "Glorot"] | None = None,
    ):
        self.scale = scale

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: Sequence[int],
        dtype: DTypeLike = jnp.float32,
    ) -> jax.Array:
        if self.scale is None:
            if len(shape) == 2:
                scale = "He"
            else:
                scale = 1.0
        else:
            scale = self.scale

        w = jrn.normal(key, shape, dtype)

        if scale == "He":
            assert len(shape) == 2
            return w * jnp.sqrt(2.0 / shape[0])

        if scale == "Glorot" or scale == "Xavier":
            assert len(shape) == 2
            return w * jnp.sqrt(2.0 / (shape[0] + shape[1]))

        assert isinstance(scale, float)
        return w * scale


@typecheck
class Uniform(Initializer, "initializers.Uniform"):
    """
    Initialize weights from a uniform distribution. Weights can be scaled by
    the number of input or output units, or by the sum of input and output,
    or by a custom factor. The methods provided are based on
    - He et al. (2015): https://arxiv.org/abs/1502.01852
    - Glorot & Bengio (2010): https://proceedings.mlr.press/v9/glorot10a.html
    """

    def __init__(
        self,
        scale: float | Literal["He", "Glorot", "Xavier"] | None = None,
    ):
        self.scale = scale

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: Sequence[int],
        dtype: DTypeLike = jnp.float32,
    ) -> jax.Array:
        if self.scale is None:
            if len(shape) == 2:
                scale = "He"
            else:
                scale = 1.0
        else:
            scale = self.scale

        w = jrn.uniform(key, shape, dtype, minval=-1, maxval=1)

        if scale == "He":
            assert len(shape) == 2
            return w * jnp.sqrt(2.0 / shape[0])

        if scale == "Glorot" or scale == "Xavier":
            assert len(shape) == 2
            return w * jnp.sqrt(2.0 / (shape[0] + shape[1]))

        assert isinstance(scale, float)
        return w * scale
