import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.opt._opt import Optimizer
from seli.opt._utils import lerp


class Adam(Optimizer, name="opt.Adam"):
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
        self.t = self.t + 1
        return grads
