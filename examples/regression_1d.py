"""
A simple 1D regression example. Fit a non linear MLP model using gradient
based optimization.
"""

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import seaborn as sns
from jaxtyping import PRNGKeyArray
from tqdm import trange
from utils import get_plot_path

import seli

x_min, x_max = -5.0, 5.0


# the true underlying function is a non linear function of x
def true_fn(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(1.5 * x) + 0.1 * x**2


# the standard deviation of the observation noise is a function of x
def stddev_fn(x: jnp.ndarray) -> jnp.ndarray:
    return 0.2 + 0.05 * jnp.abs(x)


# generate data from the true function with observation noise
def generate_data(key: PRNGKeyArray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    key_x, key_y = jrn.split(key)
    x = jrn.uniform(key_x, (n,), minval=x_min, maxval=x_max)
    y = true_fn(x)
    eps = jrn.normal(key_y, (n,)) * stddev_fn(x)
    return x, y + eps


# generate a finite dataset of points
x_data, y_data = generate_data(jrn.PRNGKey(0), 64)


# define a non linear MLP model
class MLP(seli.Module):
    def __init__(self, dim: int = 16):
        self.layers = [
            seli.net.Affine(dim),
            jnn.elu,
            seli.net.Affine(dim),
            jnn.elu,
            seli.net.Affine(1),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x[:, None]

        for layer in self.layers:
            x = layer(x)

        return x[:, 0]


model = MLP().set_rngs(42)
opt = seli.opt.Adam(1e-2)

# train with mean squared error loss, which fits the Gaussian noise model
loss_fn = seli.opt.MeanSquaredError()

for _ in trange(1000, desc="Training"):
    opt, model, loss = opt.minimize(loss_fn, model, y_data, x_data)

# make predictions on a fine grid of points
x_lin = jnp.linspace(x_min, x_max, 128)
x_lin_err = jnp.linspace(x_min, x_max, 128)

y_pred = model(x_lin)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# plot the data
ax.scatter(x_data, y_data, label="data", s=10, color="tab:blue")
ax.plot(x_lin, true_fn(x_lin), label="true", color="tab:blue", alpha=0.8)
ax.fill_between(
    x_lin_err,
    true_fn(x_lin_err) + stddev_fn(x_lin_err) * 1.645,
    true_fn(x_lin_err) - stddev_fn(x_lin_err) * 1.645,
    alpha=0.1,
    color="tab:blue",
    label="90% CI",
)

# plot the model predictions
ax.plot(x_lin, y_pred, label="model", color="tab:orange")

ax.set_xlim(x_min, x_max)
ax.legend(frameon=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("1D regression")
sns.despine()

# save the final plot
fig.savefig(
    get_plot_path("regression_1d.png"),
    dpi=256,
    bbox_inches="tight",
)
