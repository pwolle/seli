import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer


def lerp(a, b, t):
    """
    Linear interpolation between a and b with factor t.
    """
    return a * t + b * (1 - t)


class SGD(Optimizer):
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


class Momentum(Optimizer):
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


class Nesterov(Optimizer):
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


class Adagrad(Optimizer):
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


class RMSProp(Optimizer):
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


class Adam(Optimizer):
    """
    Adaptive Moment Estimation optimizer.

    Combines momentum and RMSProp, maintaining both first moment (mean) and
    second moment (variance) of gradients with bias correction.

    Adam has become the de facto standard optimizer for deep learning.
    """

    def __init__(
        self,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # First moment (momentum)
        self.m: dict[str, Float[Array, "*_"]] = {}
        # Second moment (RMSProp)
        self.v: dict[str, Float[Array, "*_"]] = {}
        # Timestep counter for bias correction
        self.t = jnp.zeros(())

    def call_param(
        self,
        key: str,
        grad: Float[Array, "*s"],
        **_,
    ) -> Float[Array, "*s"]:
        # Initialize moments if not already done
        if key not in self.m:
            self.m[key] = jnp.zeros_like(grad)
            self.v[key] = jnp.zeros_like(grad)

        # Update biased first moment estimate (momentum) using lerp
        self.m[key] = lerp(self.m[key], grad, self.beta1)

        # Update biased second moment estimate (RMSProp) using lerp
        self.v[key] = lerp(self.v[key], jnp.square(grad), self.beta2)

        # Compute bias-corrected first moment estimate
        m_corrected = self.m[key] / (1 - self.beta1**self.t)

        # Compute bias-corrected second moment estimate
        v_corrected = self.v[key] / (1 - self.beta2**self.t)

        # Compute the Adam update
        return self.lr * m_corrected / (jnp.sqrt(v_corrected) + self.eps)

    def call_model(self, grads, **_):
        self.t = self.t = 1
        return grads
