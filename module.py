# %%
# TODO
# - parameters/collections -> wrap in Param class
# - iterate/modify in pace -> use jtu
# - container modules
# - Results type and unwrap
# - Sugar for replacing with unwrap

import builtins
import dataclasses
import logging
from typing import Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrn
import jax.tree_util as jtu
import treescope
from jaxtyping import Array, Float, PRNGKeyArray

# treescope.basic_interactive_setup(
#     autovisualize_arrays=True,
#     abbreviation_threshold=1,
# )
logger = logging.getLogger(__name__)

REGISTRY = {}

A = TypeVar("A", bound=Array)


class Registered:
    def __init_subclass__(
        cls,
        name: str | Literal[False] | None = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if name is False:
            return

        if name is None:
            raise ValueError(
                f"No name was provided for the subclass '{cls.__name__}' of "
                "'PyTreeDataclass'.\n\n"
                "When subclassing 'PyTreeDataclass' or 'Module', the 'name' "
                "keyword argument is required:\n"
                f">>> class Subclass(PyTreeDataclass, name='subclass'):\n"
                f">>>     ...                         ^^^^^^^^^^^^^^^"
            )

        if name in REGISTRY:
            logger.warning(
                f"PyTreeDataclass already registered at '{name}', "
                f"overwriting '{cls}' to '{REGISTRY[name]}'."
            )

        REGISTRY[name] = cls


class FrozenDataclass:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(cls, frozen=True)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class PyTreeDataclass(FrozenDataclass):
    def __init_subclass__(cls, static: list[str] | None = None, **kwargs):
        if "__init__" in cls.__dict__:
            raise ValueError(
                f"Cannot create subclass '{cls}' of 'PyTreeDataclass' with an "
                "explicit __init__ method. The PyTree system requires that the "
                "arguments to the constructor are exactly the fields of the "
                "dataclass, as this allows for using optimized C++ code for "
                "flattening and unflattening the tree."
            )

        super().__init_subclass__(**kwargs)
        jtu.register_dataclass(cls, meta_fields=static)


class Module(Registered, PyTreeDataclass, name="builtins.Module"):
    pass


class ExampleModule(Module, name="ExampleModule"):
    x: int
    y: float


@jax.jit
def identity(module: Module) -> Module:
    return module


module = ExampleModule(x=1, y=2.0)


# %%
class Param(Module, Generic[A], name="builtins.Param"):
    value: A
    collections: tuple[str, ...] = ("train",)


class LazyLinear(Module, name="LazyLinear"):
    key: PRNGKeyArray
    dim: int

    def __call__(self, x: Float[Array, "*b dim_in"]) -> Float[Array, "*b dim"]:
        linear = Linear.new(self.key, x.shape[-1], self.dim)
        return linear(x)


class Linear(Module, name="Linear"):
    w: Param[Float[Array, "dim_in dim"]]
    b: Param[Float[Array, "dim"]]

    @classmethod
    def new(cls, key: PRNGKeyArray, dim_in: int, dim: int) -> "Linear":
        w = jrn.normal(key, (dim_in, dim)) / jnp.sqrt(dim_in)
        b = jnp.zeros((dim,))
        return cls(Param(w), Param(b))

    @classmethod
    def forward(
        cls,
        x: Float[Array, "*b dim_in"],
        w: Float[Array, "dim_in dim"],
        b: Float[Array, "dim"],
    ):
        return x @ w + b

    def __call__(self, x: Float[Array, "*b dim_in"]):
        return self, self.forward(x, self.w, self.b)

    @property
    def dim_in(self) -> int:
        return self.w.shape[0]

    @property
    def dim(self) -> int:
        return self.w.shape[1]


linear = Linear.new(jrn.PRNGKey(0), 10, 10)
