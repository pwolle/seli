"""
Plot the optimization trajectory for different optimizers on a 2D example.

The Rosenbrock function is a classic optimization problem that is a good
test case for optimizers. It is a non-convex function with a narrow valley
and a global minimum.
"""

import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import get_plot_path

import seli


# Define the rosenbrock function, which is to be optimized
def rosenbrock_elementwise(x):
    return (1 - x[..., 0]) ** 2 + 100 * (x[..., 1] - x[..., 0] ** 2) ** 2


# Wrap it in a seli.opt.Loss, so it can be used with the optimizers
class RosenbrockLoss(seli.opt.Loss):
    def __call__(self, x):
        # the loss function should return a scalar value, so we sum over all
        # elements of the array
        return jnp.sum(rosenbrock_elementwise(x))


# Create the loss object, the collection is set to be None instead of the
# defaul value "param", since we want to optimize with respect to all input
# arrays (in this case we only input a simple array x)
loss = RosenbrockLoss()
loss.collection = None

# The starting point of the optimization, lets put this far from the optimum
# to see how well the optimizers cope with the curvature of the loss surface
start = jnp.array([-0.5, 1])


# Compute the trajectory of the optimizer, this is a helper function that
# runs the optimizer for a given number of steps and returns the trajectory
# of the optimizer.
# We could also run this in a for loop, but this directly compiles the whole
# optimization loop as well, so it does not need to get back to the python
# process in each iterationm, which would be slower.
def compute_trajectory(optimizer, start):
    model = start
    remaining_steps = 2**12

    trajectory = [model]
    current_optimizer = optimizer
    current_model = model

    # run for a few steps manually to make sure the optimizer is initialized
    # this is necessary, since lax.scan expects the computation graph to be
    # the same for all iterations
    for _ in range(2):
        current_optimizer, current_model, _ = current_optimizer.minimize(
            loss, current_model
        )
        trajectory.append(current_model)

    def scan_body(carry, _):
        optimizer_state, current_model = carry
        optimizer_updated, model_updated, _ = optimizer_state.minimize(
            loss, current_model
        )
        return (optimizer_updated, model_updated), model_updated

    # Initialize carry with the current optimizer and model after manual steps
    init_carry = (current_optimizer, current_model)

    # Run lax.scan for remaining iterations
    _, trajectory_models = lax.scan(
        scan_body,
        init_carry,
        None,  # xs=None means we don't have inputs for each iteration
        length=remaining_steps,
    )

    # Combine manual trajectory with lax.scan trajectory
    full_trajectory = trajectory + [
        trajectory_models[i] for i in range(trajectory_models.shape[0])
    ]
    return full_trajectory


# These are the optimizers wel will test, the parameters are chosen to maximize
# the performance for the rosenbrock function
optimizers = {
    "SGD": seli.opt.SGD(1e-3),
    "Momentum": seli.opt.Momentum(1e-3, 0.9),
    "RMSProp": seli.opt.RMSProp(1e-2, 0.99),
    "Adagrad": seli.opt.Adagrad(1e-1),
    "Nesterov": seli.opt.Nesterov(1e-3, 0.9),
    "Adam": seli.opt.Adam(1e-1, 0.9, 0.999, 1e-12),
}

results = {}

# Compute the trajectory for each optimizer
for name, optimizer in tqdm(optimizers.items(), desc="Computing trajectories"):
    results[name] = compute_trajectory(optimizer, start)


# Create a grid of points to evaluate the loss function on we will later use
# this to create a contour plot of the loss function
x = jnp.linspace(-1.1, 1.1, 1024)
y = jnp.linspace(-0.25, 1.17, 1024)
xi, yi = jnp.meshgrid(x, y)

# Evaluate the loss function on the grid
xy = jnp.stack([xi, yi], axis=-1)
loss_values = rosenbrock_elementwise(xy)

# Applu a transformation to the loss values to spread the values out more,
# otherwise the contour plot will be dominated by the narrow valley
loss_values = jnp.arcsinh(loss_values**0.5)

fig, ax = plt.subplots(figsize=(4, 4))

ax.contourf(
    xi,
    yi,
    loss_values,
    levels=16,
    cmap="turbo_r",
    alpha=0.5,
    zorder=-101,
)

# plot the trajectory of the optimizers
for name, trajectory in results.items():
    ax.plot(
        [t[0] for t in trajectory],
        [t[1] for t in trajectory],
        label=name,
    )

# plot the starting point
ax.scatter(start[0], start[1], color="k", marker="o", s=20, zorder=10)
ax.text(start[0] - 0.05, start[1], "Start", ha="right", va="center")

# plot the optimum
optimum = jnp.array([1, 1])
ax.scatter(optimum[0], optimum[1], color="k", marker="X", s=30, zorder=10)
ax.text(optimum[0] - 0.05, optimum[1], "Optimum", ha="right", va="center")

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
sns.despine(ax=ax)

# put legend outside of the plot
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)

ax.set_title("Optimizing the Rosenbrock function", pad=10)

fig.savefig(
    get_plot_path("rosenbrock.png"),
    dpi=256,
    bbox_inches="tight",
)
