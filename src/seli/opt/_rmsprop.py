import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer
from seli.opt._utils import lerp


class RMSProp(Optimizer, name="opt.RMSProp"):
    """
    Root Mean Square Propagation optimizer.

    Addresses Adagrad's diminishing learning rates by using exponential moving
    average of squared gradients.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.g2: dict[str, Float[Array, "*_"]] = {}

    def call_param(
        self,
        key: str,
        grad: Float[Array, "*s"],
        **_,
    ) -> Float[Array, "*s"]:
        if key not in self.g2:
            self.g2[key] = jnp.zeros_like(grad)

        # compute the EMA of the squared gradients
        self.g2[key] = lerp(self.g2[key], jnp.square(grad), self.beta)

        # Normalize the gradient by the EMA of the squared gradients
        return self.lr * grad / (jnp.sqrt(self.g2[key]) + self.eps)
