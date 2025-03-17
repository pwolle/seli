import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer


class Momentum(Optimizer, name="opt.Momentum"):
    """
    Momentum optimizer.

    Accelerates optimization by accumulating a velocity vector in the direction
    of persistent gradient directions. This is analogous to the momentum of a
    ball rolling down a hill.

    The velocity is updated with the gradient and a decay factor. The decay
    factor is a hyperparameter that controls the influence of previous
    gradients on the current update.

    For well-behaved functions, momentum often leads to faster convergence,
    when compared to SGD.
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

        # update the velocity
        self.v[key] = self.v[key] * self.beta + grad

        # scale the velocity by the learning rate
        return self.v[key] * self.lr
