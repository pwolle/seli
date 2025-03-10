from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from seli import Module, typecheck
from seli.net._initializers import Initializer


@typecheck
class Param(Module, name="net.Param"):
    def __init__(self, initializer: Initializer):
        self.initializer = initializer

        self._val = None
        self._key = None

    def __call__(
        self,
        shape: Sequence[int],
        dtype: DTypeLike = jnp.float32,
    ) -> jax.Array:
        if self._val is not None:
            if self._val.shape != shape:
                error = f"Shape mismatch {self._val.shape} != {shape}"
                raise ValueError(error)

            if self._val.dtype != dtype:
                error = f"Dtype mismatch {self._val.dtype} != {dtype}"
                raise ValueError(error)

            return self._val

        if self._key is None:
            error = "Parameter needs a key for initialization"
            raise ValueError(error)

        assert isinstance(self._key, PRNGKeyArray)
        self._val = self.initializer(self._key, shape, dtype)
        return self._val
