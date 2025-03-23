"""
Example of supervised learning on a 2D dataset.

The dataset is generated using the `moons_dataset` function, which generates
a dataset of two interleaving half circles.
"""

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from utils import get_plot_path, moons_dataset

import seli

x_data, y_data = moons_dataset(jrn.PRNGKey(0), 128)


class MLP(seli.Module):
    def __init__(self):
        self.layers2 = [
            seli.net.Affine(32),
            jnn.relu,
            seli.net.Affine(32),
            jnn.relu,
            seli.net.Affine(1),
        ]

    def __call__(self, x):
        for layer in self.layers2:
            x = layer(x)

        return x


model = MLP().set_rngs(42)
loss = seli.opt.BinaryCrossEntropy()
optimizer = seli.opt.Adam(1e-3)

x = jnp.linspace(-2, 2, 1024)
y = jnp.linspace(-2, 2, 1024)
xi, yi = jnp.meshgrid(x, y)
xy = jnp.stack([xi, yi], axis=-1)

loss_values = []
titles = {
    10: "10 iterations",
    100: "100 iterations",
    1000: "1k iterations",
    10000: "10k iterations",
}
predictions = {}

for i in trange(max(titles.keys()) + 1, desc="Training"):
    if i in titles:
        predictions[i] = model(xy)

    optimizer, model, loss_value = optimizer.minimize(loss, model, y_data, x_data)
    loss_values.append(loss_value)

mosaic = [titles, ["loss"] * len(titles)]

fig, axs = plt.subplot_mosaic(
    mosaic,
    height_ratios=[3, 2],
    figsize=(len(titles) * 3, 5),
)

for i, iteration in enumerate(titles):
    ax, pred = axs[iteration], predictions[iteration]

    # normalize predictions to be between -1 and 1
    pred = jnn.tanh(pred * 0.05)

    norm = mcolors.TwoSlopeNorm(
        vmin=pred.min(),
        vcenter=0,
        vmax=pred.max(),
    )
    r = ax.contourf(
        xi,
        yi,
        pred.reshape(1024, 1024),
        cmap="bwr",
        levels=100,
        alpha=0.5,
        norm=norm,
    )
    ax.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap="bwr")
    sns.despine(ax=ax)
    ax.set_xlabel("$x_1$")
    ax.set_title(titles[iteration])

    # share y axis
    if i == 0:
        ax.set_ylabel("$x_2$")
    else:
        ax.set_yticks([])

# plot loss
ax = axs["loss"]
ax.plot(loss_values)
ax.set_xlim(5, max(titles.keys()) * 1.5)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Loss")
ax.set_xlabel("Iteration")
sns.despine(ax=ax)

# add space between rows
fig.subplots_adjust(hspace=0.3)

# add title
fig.suptitle("2D Supervised Learning", fontsize=16)

# save
fig.savefig(
    get_plot_path("supervised_2d.png"),
    dpi=256,
    bbox_inches="tight",
)
