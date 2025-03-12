from typing import Generic, TypeVar

import jax
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from seli._env import DEFAULT_FLOAT
from seli.core._module import Module
from seli.net._init import Initializer
from seli.net._key import Key

__all__ = [
    "Param",
]

# make generic to differentiate between initialized and uninitialized
V = TypeVar("V", bound=jax.Array | None)


class Param(Module, Generic[V], name="net.Param"):
    value: V

    def __init__(
        self,
        init: Initializer,
        *,
        key: PRNGKeyArray | None = None,
        collection: str | None = None,
    ):
        self.init = init
        self.key = Key(key, collection)

        self.value = None

    @property
    def collection(self) -> str | None:
        return self.key.collection

    @property
    def initialized(self) -> bool:
        return self.value is not None

    def __call__(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = DEFAULT_FLOAT,
    ) -> jax.Array:
        if not self.initialized:
            if not self.key.initialized:
                error = "Key has not been set"
                raise ValueError(error)

            self.value = self.init(self.key.key, shape, dtype)

        if self.value.shape != shape:
            error = f"Expected shape {shape}, got {self.value.shape}"
            raise ValueError(error)

        if self.value.dtype != dtype:
            error = f"Expected dtype {dtype}, got {self.value.dtype}"
            raise ValueError(error)

        return self.value
