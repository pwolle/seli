from typing import Generic, TypeVar

from jax import Array
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from seli._env import DEFAULT_FLOAT
from seli.core._module import Module
from seli.net._init import Init
from seli.net._key import RNGs

__all__ = [
    "Param",
]

# make generic to differentiate between initialized and uninitialized
# at type inference time
V = TypeVar("V", bound=Array | None)


class Param(Module, Generic[V], name="net.Param"):
    """
    Organizes a parameter
    """

    value: V

    init: Init | None
    rngs: RNGs | None

    def __init__(
        self,
        *,
        init: Init,
        rngs: PRNGKeyArray | None = None,
        value: V | None = None,
        collection: str | None = "param",
    ) -> None:
        self.init = init
        self.value = value

        self.collection = collection
        self.rngs = RNGs(rngs, "init")

    @classmethod
    def from_value(
        cls,
        value: Array,
        *,
        collection: str | None = "param",
    ) -> "Param[Array]":
        return cls(init=None, data=value, collection=collection)

    @property
    def initialized(self) -> bool:
        return self.value is not None

    def __call__(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = DEFAULT_FLOAT,
    ) -> Array:
        if not self.initialized:
            if not self.rngs.initialized:
                error = "Key has not been set"
                raise ValueError(error)

            assert self.init is not None, "Init or value was changed to None?"
            self.value = self.init(self.rngs.key, shape, dtype)

        if self.value.shape != shape:
            error = f"Expected shape {shape}, got {self.value.shape}"
            raise ValueError(error)

        if self.value.dtype != dtype:
            error = f"Expected dtype {dtype}, got {self.value.dtype}"
            raise ValueError(error)

        return self.value
