"""
Implementation of a normalizing flow in 1D.

Normalizing flows are a class of generative models, that use invertible
architectures and the density transformation theorem to maximize the log
likelihood of the data under a simple base distribution.

Sampling is done by drawing samples from the base distribution and applying
the inverse of the flow to them.

This example implements the invertible architecture proposed in
"Unconstrained Monotonic Neural Networks" by Wehenkel & Louppe [2019].
The code idea is that it is easy to constrain neural networks to output
positive values, by applying a positive function to the output. If the output
is positive, the integral is (strictly) monotonic and therefore invertible
in one dimension. This leaves us with a universal approximator of monotonic
functions.
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
from tqdm import trange
from utils import (
    get_plot_path,
    plot_1d_generative_model,
    two_gaussians,
    two_gaussians_likelihood,
)

import seli


class NormalizingFlow(seli.Module):
    def __init__(
        self,
        dim: int = 128,
        std: float = 1.0,
        integration_steps: int = 16,
    ):
        self.dim = dim
        self.std = std
        self.integration_steps = integration_steps

        # the neural network that will be integrated
        self.net = seli.net.Sequential(
            seli.net.Affine(dim),
            jnn.gelu,
            seli.net.Affine(dim),
            jnn.gelu,
            seli.net.Affine(1),
        )
        self.bias = seli.net.Bias()

    def inverse(self, x_input):
        """
        Monotonic function by integratig the output of a posivie function.
        """
        # get interelugration grid
        xs_lin = jnp.linspace(0, x_input, self.integration_steps, axis=-1)

        # calculate derivatives
        ys_lin = self.net(xs_lin[..., None])[..., 0]

        # make derivatives positive, also add a small constant to avoid log(0)
        # later in the log likelihood calculation
        ys_lin = jnn.softplus(ys_lin) + 1e-6

        # integrate to get a monotonic function
        integrated = jnp.sum(
            ys_lin[..., 1:] * (xs_lin[..., 1:] - xs_lin[..., :-1]),
            axis=-1,
        )
        # add bias
        return self.bias(integrated[..., None])[..., 0]

    def prior_log_likelihood(self, x):
        return -0.5 * jnp.log(2 * jnp.pi) - 0.5 * x**2

    def log_likelihood(self, x_input):
        # ensure that we only need a single vectorization over the gradient
        # since calculating the gradient requires scalar outputs
        assert x_input.ndim == 1

        # get the value and gradient of the inverse at the input
        @jax.vmap
        @jax.value_and_grad
        def model_value_and_grad(x):
            return self.inverse(x)

        value, grad = model_value_and_grad(x_input)

        # calculate the log likelihood
        log_grad = jnp.log(jnp.abs(grad))

        # get the likelihood of the input under the base distribution
        # using the density transformation theorem
        return self.prior_log_likelihood(value) + log_grad

    def __call__(
        self,
        x,
        lower=-1e2,
        upper=1e2,
        tol=1e-6,
    ):
        """
        Invert the flow to get data samples from base distribution samples.
        """
        # calculate the number of iterations needed to reach the desired
        # tolerance
        iterations = int(jnp.log2((upper - lower) / tol))

        # perform bisection to find the value that has the inverse value of x
        # we can perform bisection since the inverse is monotonic
        def bisect(x, lower, upper):
            middle = (lower + upper) / 2
            output = self.inverse(middle)
            direct = output < x

            # update the lower or upper bound in a parallel manner
            lower = jnp.where(direct, middle, lower)
            upper = jnp.where(direct, upper, middle)
            return lower, upper

        # run bisection to find the value that has the inverse value of x
        lower = jnp.zeros_like(x) + lower
        upper = jnp.zeros_like(x) + upper

        for _ in range(iterations):
            lower, upper = bisect(x, lower, upper)

        return lower


# Maximizing the log likelihood is equivalent to minimizing the negative
# log likelihood.
class LogLikelihoodLoss(seli.opt.Loss):
    def __call__(self, model, x):
        return -model.log_likelihood(x).mean()


# setup the model, loss and optimizer
model = NormalizingFlow().set_rngs(42)
loss = LogLikelihoodLoss()
opt = seli.opt.Adam(3e-4)

key = jrn.PRNGKey(42)
losses = []


# run the training and keep track of the loss
for _ in trange(10000, desc="Training"):
    # we want to generate a new independent batch for each iteration
    key, subkey = jrn.split(key)
    opt, model, loss_value = opt.minimize(
        loss,
        model,
        two_gaussians(subkey, 128),
    )
    losses.append(loss_value)


x = jnp.linspace(-3, 3, 128)
likelihood = two_gaussians_likelihood(x)
likelihood_model = jnp.exp(model.log_likelihood(x))

samples = two_gaussians(subkey, 2048)
samples_model = model(jrn.normal(subkey, (2048,)))

fig = plot_1d_generative_model(
    losses,
    samples,
    samples_model,
    x,
    likelihood,
    likelihood_model,
)
fig.suptitle("Normalizing Flow")
fig.savefig(
    get_plot_path("normalizing_flow_1d.png"),
    dpi=256,
    bbox_inches="tight",
)
