import os

import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import seaborn as sns
from jax import Array

import seli


def get_plot_path(filename: str):
    """
    Get the path to the folder where the plots of examples are saved.
    """
    folder = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)
    return os.path.join(folder, "plots", filename)


@seli.jit
def moons_dataset(key, n: int):
    """
    Generate a dataset of points on a moon shape.
    """
    key_angle, key_noise = jrn.split(key)

    angle = jrn.uniform(key_angle, (n,), minval=0, maxval=2 * jnp.pi)
    y = (angle > jnp.pi).astype(jnp.float32)

    x = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    x = x.at[:, 0].add(y - 0.5)
    x = x.at[:, 1].add((y - 0.5) * 0.2)

    x = x + jrn.normal(key_noise, (n, 2)) * 0.2
    return x, y


@seli.jit
def two_gaussians(key, n: int, scale: float = 0.5):
    """
    Generate a dataset of points on two gaussians.
    """
    key_noise, key_mode = jrn.split(key)
    z = jrn.normal(key_noise, (n,))
    r = jrn.rademacher(key_mode, (n,))
    return z * scale + r


def two_gaussians_likelihood(x, scale: float = 0.5):
    """
    Compute the likelihood of the two gaussians.
    """
    log_Z = -0.5 * jnp.log(2 * jnp.pi * scale**2)
    log_likelihood_1 = -0.5 * ((x + 1) / scale) ** 2 + log_Z
    log_likelihood_2 = -0.5 * ((x - 1) / scale) ** 2 + log_Z

    prob_1 = jnp.exp(log_likelihood_1)
    prob_2 = jnp.exp(log_likelihood_2)

    return 0.5 * (prob_1 + prob_2)


def plot_1d_generative_model(
    losses: list[float],
    samples: Array,
    samples_model: Array,
    x: Array | None = None,
    likelihood: Array | None = None,
    likelihood_model: Array | None = None,
):
    """
    Plot the results of a 1D generative model.
    """
    fig, (ax_loss, ax_samples) = plt.subplots(1, 2, figsize=(10, 5))

    ax_loss.plot(losses)
    ax_loss.set_yscale("log")

    ax_loss.set_xlim(0, len(losses))
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("log likelihood")

    ax_samples.hist(
        [samples, samples_model],
        bins=32,
        density=True,
        label=["Data samples", "Model samples"],
        histtype="step",
        color=["tab:blue", "tab:red"],
    )
    if not any(i is None for i in [x, likelihood, likelihood_model]):
        ax_samples.plot(
            x,
            likelihood,
            label="True density",
            color="tab:blue",
        )
        ax_samples.plot(
            x,
            likelihood_model,
            label="Model density",
            color="tab:red",
        )

    ax_samples.set_xlim(x.min(), x.max())
    ax_samples.set_xlabel("x")
    ax_samples.set_ylabel("density")
    ax_samples.legend(frameon=False, ncol=2)

    sns.despine(ax=ax_samples)
    return fig
