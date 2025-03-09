import jax

from seli import Module


class Param(Module):
    def __init__(self, value: jax.Array):
        self.value = value

    def __call__(self) -> jax.Array:
        return self.value

    def __repr__(self):
        return f"Param({self.value})"
