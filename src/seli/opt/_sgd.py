from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def call_param(self, grad: Float[Array, "*s"], **_) -> Float[Array, "*s"]:
        return grad * self.lr
