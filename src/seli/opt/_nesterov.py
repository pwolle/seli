import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer


class Nesterov(Optimizer, name="opt.Nesterov"):
    """
    Nesterov Accelerated Gradient optimizer.

    Improves on standard momentum by computing gradients at a "lookahead"
    position, providing better convergence rates.
    """

    def __init__(self, lr: float = 1e-3, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.v: dict[str, Float[Array, "*_"]] = {}

    def call_param(
        self,
        key: str,
        grad: Float[Array, "*s"],
        **_,
    ) -> Float[Array, "*s"]:
        if key not in self.v:
            self.v[key] = jnp.zeros_like(grad)

        # Calculate the update using Nesterov momentum
        velocity_prev = self.v[key]
        self.v[key] = velocity_prev * self.beta + grad

        # This effectively computes the gradient at a "lookahead" position
        return self.lr * (self.beta * self.v[key] + (1 - self.beta) * grad)
