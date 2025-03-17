import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer


class Adagrad(Optimizer, name="opt.Adagrad"):
    """
    Adaptive Gradient optimizer.

    Adapts learning rates per-parameter by scaling with the inverse square root
    of accumulated squared gradients.
    """

    def __init__(self, lr: float = 1e-2, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.G2: dict[str, Float[Array, "*_"]] = {}

    def call_param(
        self,
        key: str,
        grad: Float[Array, "*s"],
        **_,
    ) -> Float[Array, "*s"]:
        if key not in self.G2:
            self.G2[key] = jnp.zeros_like(grad)

        # Accumulate squared gradients
        self.G2[key] = self.G2[key] + jnp.square(grad)

        # Compute the adaptive learning rate update
        return self.lr * grad / (jnp.sqrt(self.G2[key]) + self.eps)
