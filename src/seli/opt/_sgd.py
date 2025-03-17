from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer


def lerp(a, b, t):
    """
    Linear interpolation between a and b with factor t.
    """
    return a * t + b * (1 - t)


class SGD(Optimizer, name="opt.SGD"):
    """
    Stochastic Gradient Descent optimizer.

    The gradient is the direction of steepest descent. The SGD update simply
    scaled the gradient by the learning rate and takes a step in that
    direction. It does not account for information from previous gradients.

    There has been some evidence that SGD has a regularization effect,
    which leads to better generalization performance, at the cost of slower
    convergence.
    """

    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def call_param(self, grad: Float[Array, "*s"], **_) -> Float[Array, "*s"]:
        # scale the gradient by the learning rate
        return grad * self.lr
