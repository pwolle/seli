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

    x = x + jrn.normal(key_noise, (n, 2)) * 0.1
    return x, y
