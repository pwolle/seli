import os

import jax.numpy as jnp
import jax.random as jrn

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
