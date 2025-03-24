"""
Example of classification on a 2D dataset using a simple 3 layer MLP.
"""

# %%
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from utils import get_plot_path, moons_dataset

import seli

# generate the 2D dataset
# this two moons dataset is a classic toy dataset for binary classification
# it contains two interleved moon shaped clusters of points, which are
# easily separable by a genarl model, but not linearly separable by a linear
# model
x_data, y_data = moons_dataset(jrn.PRNGKey(0), 128)


# define the MLP
# this MLP has 3 layers, the first two have 32 neurons and the last one has 1
# neuron (for the sigmoid output), the sigmoid activation function is applied
# in the loss for efficency. The popular and simlle activation function ReLU
# is used for the intermediate layers. This increases the expressiveness of
# the model and makes it possible to fit the non-linear data.
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

        return x[..., 0]


# create the model and set the random number generator seed for reproducibility
model = MLP().set_rngs(42)

# create the loss and optimizer
# the loss function is the binary cross entropy loss, this essentially maximizes
# the log likelihood of the labels under the model predicted probabilities
loss = seli.opt.BinaryCrossEntropy()

# the optimizer is the Adam optimizer, which is a popular optimizer for
# training neural networks
optimizer = seli.opt.Adam(1e-3)

# create a grid of points to evaluate the model on
# this is used to create a contour plot of the model predictions
x = jnp.linspace(-2, 2, 1024)
y = jnp.linspace(-2, 2, 1024)
xi, yi = jnp.meshgrid(x, y)
xy = jnp.stack([xi, yi], axis=-1)

# We will plot the loss value and the model predictions for different numbers
# of iterations, so we store the loss values and predictions in the following
# dictionaries
loss_values = []
titles = {
    10: "10 iterations",
    100: "100 iterations",
    1000: "1k iterations",
    10000: "10k iterations",
}
predictions = {}

# perform the training loop
for i in trange(max(titles.keys()) + 1, desc="Training"):
    if i in titles:
        predictions[i] = model(xy)

    optimizer, model, loss_value = optimizer.minimize(loss, model, y_data, x_data)
    loss_values.append(loss_value)


# create the final plot for visualizing the model training
mosaic = [titles, ["loss"] * len(titles)]
fig, axs = plt.subplot_mosaic(
    mosaic,
    height_ratios=[3, 2],
    figsize=(len(titles) * 3, 5),
)

# plot the model predictions for each iteration
for i, iteration in enumerate(titles):
    ax, pred = axs[iteration], predictions[iteration]

    # normalize predictions to make the contour plot more informative
    pred = jnn.tanh(pred * 0.1)

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

# plot how the loss value evolves over the iterations
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
fig.suptitle("2D Classification", fontsize=16)

# save
fig.savefig(
    get_plot_path("classification_2d.png"),
    dpi=256,
    bbox_inches="tight",
)
