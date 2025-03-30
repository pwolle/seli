import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
from tqdm import trange
from utils import (
    get_plot_path,
    plot_1d_generative_model,
    two_gaussians,
)

import seli


class Model(seli.Module):
    def __init__(self, dim: int = 32):
        self.layers = [
            seli.net.Affine(dim),
            jnn.relu,
            seli.net.Affine(dim),
            jnn.relu,
            seli.net.Affine(1),
        ]

    def __call__(self, x):
        x = x[..., None]

        for layer in self.layers:
            x = layer(x)

        return x[..., 0]

    def prior(self, key, batch_size: int):
        return jrn.normal(key, (batch_size,))

    def sample(self, key, batch_size: int):
        return self(self.prior(key, batch_size))


def rbf_kernel(x, y, sigma: float = 1.0):
    dist = jnp.sum(jnp.square(x - y))
    return jnp.exp(-dist / (2 * sigma**2))


def maximum_mean_discrepancy(x, y, kernel):
    k_fn = jax.vmap(jax.vmap(kernel, in_axes=(0, None)), in_axes=(None, 0))
    k_xx = k_fn(x, x)
    k_xy = k_fn(x, y)
    k_yy = k_fn(y, y)
    return jnp.mean(k_xx) + jnp.mean(k_yy) - 2 * jnp.mean(k_xy)


class MaximumMeanDiscrepancy(seli.opt.Loss):
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, model, key, x):
        batch_size = x.shape[0]
        y = model.sample(key, batch_size)
        return maximum_mean_discrepancy(
            x,
            y,
            lambda x, y: rbf_kernel(x, y, self.sigma),
        )


model = Model().set_rngs(42)
loss = MaximumMeanDiscrepancy()
opt = seli.opt.Adam(1e-3)

batch_size = 32
key = jrn.PRNGKey(0)
losses = []

for _ in trange(1000, desc="Training"):
    key, key_generator, key_data = jrn.split(key, 3)
    opt, model, loss_val = opt.minimize(
        loss,
        model,
        key_generator,
        two_gaussians(key_data, batch_size),
    )
    losses.append(loss_val)


key, key_generator, key_data = jrn.split(jrn.PRNGKey(0), 3)
samples = two_gaussians(key_data, 1024 * 32)
samples_model = model.sample(key_generator, 1024 * 32)

fig = plot_1d_generative_model(
    losses,
    samples,
    samples_model,
)
fig.suptitle("Maximum Mean Discrepancy")
fig.savefig(
    get_plot_path("maximum_mean_discrepancy.png"),
    dpi=256,
    bbox_inches="tight",
)
