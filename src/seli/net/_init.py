from typing import Literal

import jax
import jax.nn.initializers as jni
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from seli.core._module import Module

__all__ = [
    "Init",
    "Zeros",
    "Ones",
    "Constant",
    "TruncatedNormal",
    "Normal",
    "Uniform",
    "Orthogonal",
]


class Init(Module, name="net.init.Initializer"):
    """
    Base class for all initializers. Initializers are typically used to
    initialize the weights of neural networks.

    Hyperparameters are specified in the constructor and the initializer is
    called with a key, a shape and a dtype.
    """

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        raise NotImplementedError


class Zeros(Init, name="net.init.Zeros"):
    """
    Initializes all values to zero.
    """

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        return jnp.zeros(shape, dtype)


class Ones(Init, name="net.init.Ones"):
    """
    Initializes all values to one.
    """

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        return jnp.ones(shape, dtype)


class Constant(Init, name="net.init.Constant"):
    """
    Initializes all values to a constant value.
    """

    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        return jnp.full(shape, self.value, dtype)


class TruncatedNormal(Init, name="net.init.TruncatedNormal"):
    """
    Initializes values from a truncated normal distribution.
    """

    def __init__(
        self,
        minv: float = -1.0,
        maxv: float = 1.0,
        shift: float = 0.0,
        scale: float = 1.0,
    ):
        self.minv = minv
        self.maxv = maxv
        self.shift = shift
        self.scale = scale

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        return (
            jax.random.truncated_normal(
                key,
                self.minv,
                self.maxv,
                shape,
                dtype,
            )
            * self.scale
            + self.shift
        )


class Normal(Init, name="net.init.Normal"):
    """
    Initializes values from a normal distribution. The following automatic
    scaling methods are supported:

    - Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of
      training deep feedforward neural networks."
      In *Proceedings of the Thirteenth International Conference on Artificial
      Intelligence and Statistics* (pp. 249-256). JMLR.org.

    - He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving deep into
      rectifiers: Surpassing human-level performance on ImageNet
      classification."

    - LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). "Efficient
      backprop."

    The default is He, the same as in PyTorch.
    """

    def __init__(
        self,
        init: Literal[
            "Unit",
            "He",
            "Xavier",
            "Glorot",
            "Kaiming",
            "LeCun",
        ] = "He",
        shift: float = 0.0,
        scale: float = 1.0,
    ):
        self.init = init
        self.shift = shift
        self.scale = scale

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        if self.init == "Unit":
            w = jax.random.uniform(key, shape, dtype, -1, 1)

        elif self.init == "He" or self.init == "Xavier":
            w = jni.he_normal()(key, shape, dtype)

        elif self.init == "Glorot" or self.init == "Kaiming":
            w = jni.glorot_normal()(key, shape, dtype)

        elif self.init == "LeCun":
            w = jni.lecun_normal()(key, shape, dtype)

        else:
            raise ValueError(f"Invalid initializer: {self.init}")

        return w * self.scale + self.shift


class Uniform(Init, name="net.init.Uniform"):
    """
    Initializes values from a uniform distribution. The following automatic
    scaling methods are supported:

    - Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of
      training deep feedforward neural networks."
      In *Proceedings of the Thirteenth International Conference on Artificial
      Intelligence and Statistics* (pp. 249-256). JMLR.org.

    - He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving deep into
      rectifiers: Surpassing human-level performance on ImageNet
      classification."

    - LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). "Efficient
      backprop."

    The default is He, the same as in PyTorch.
    """

    def __init__(
        self,
        init: Literal[
            "Unit",
            "He",
            "Xavier",
            "Glorot",
            "Kaiming",
            "LeCun",
        ] = "He",
        shift: float = 0.0,
        scale: float = 1.0,
    ):
        self.init = init
        self.shift = shift
        self.scale = scale

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        if self.init == "Unit":
            w = jax.random.uniform(key, shape, dtype, -1, 1)

        elif self.init == "He" or self.init == "Xavier":
            w = jni.he_uniform()(key, shape, dtype)

        elif self.init == "Glorot" or self.init == "Kaiming":
            w = jni.glorot_uniform()(key, shape, dtype)

        elif self.init == "LeCun":
            w = jni.lecun_uniform()(key, shape, dtype)

        else:
            raise ValueError(f"Invalid initializer: {self.init}")

        return w * self.scale + self.shift


class Orthogonal(Init, name="net.init.Orthogonal"):
    """
    Initialize weights as an orthogonal matrices.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> jax.Array:
        return jni.orthogonal()(key, shape, dtype) * self.scale
