"""
Implement optimal transport flow matching in 1D.

As shown in the previous example (normalizing_flow_1d.py), we can use invertible
functions to train a density model. Here we parametrize this invertible function
by an ODE solver, which gets solved along an auxilliary time variable.

The break-through by Lipman et al. (2022) is that we can regress regress the
conditional flow of the forward process, which transforms the source
distribution into the target distribution. Regressing this conditional flow
will make the model learn the marginal flow, which is still gives a valid
transport map.

Tong et al. [2023] extended this idea by using the fact that arbitrary couplings
in the source and target distributions yield different, but all valid transport
maps. Using minibatch optimal transport couplings the model effectively learns
the derivative of the Wasserstein geodesic between the source and target
distributions.

In this example, we will implement this idea by using the fact that optimal
transport couplings are very cheap to calculate in 1D. We can still colculate
log-likelihoods after training using the instanteneous change of density
formula (Chen et al. [2018]).
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from utils import get_plot_path, two_gaussians, two_gaussians_likelihood

import seli


class Flow(seli.Module):
    def __init__(self, dim: int = 128):
        # parametrize the time derivative of the distributions
        # i.e. the flow field
        self.layers = [
            seli.net.Affine(dim),
            jnn.gelu,
            seli.net.Affine(dim),
            jnn.gelu,
            seli.net.Affine(1),
        ]

    def __call__(self, x, t):
        x = jnp.stack([x, t], axis=-1)

        for layer in self.layers:
            x = layer(x)

        return x[..., 0]

    @seli.jit
    def forward(self, source, num_steps: int = 1024):
        # solve the ODE using the Euler-Maruyama method
        # we start at t = 0 and integrate for num_steps from the source
        # distribution to the target distribution, the neural network
        # gives us the time derivative of the distribution at the current
        # time step
        t = jnp.zeros((source.shape[0],))
        dt = 1 / num_steps
        x_t = source

        def body_fun(_, carry):
            t, x_t = carry
            t = t + dt
            v_t = self(x_t, t)
            x_t = x_t + v_t * dt
            return (t, x_t)

        t, x_t = jax.lax.fori_loop(0, num_steps, body_fun, (t, x_t))
        return x_t

    @seli.jit
    def forward_with_log_likelihood(self, source, num_steps: int = 1024):
        # we can compute the log-likelihood of the target distribution by
        # integrating the negative gradient of the jacobi trace, this is
        # shown in the paper by Chen et al. [2018]
        assert source.ndim == 1

        t = jnp.zeros((source.shape[0],))
        dt = 1 / num_steps
        x_t = source
        ll = self.prior_log_likelihood(source)

        model_value_and_grad = jax.vmap(jax.value_and_grad(self))

        def body_fun(_, carry):
            t, x_t, ll = carry
            t = t + dt
            v_t, grad_v_t = model_value_and_grad(x_t, t)
            x_t = x_t + v_t * dt
            ll = ll - grad_v_t * dt
            return (t, x_t, ll)

        t, x_t, ll = jax.lax.fori_loop(0, num_steps, body_fun, (t, x_t, ll))
        return x_t, ll

    def prior_log_likelihood(self, x):
        return -0.5 * jnp.log(2 * jnp.pi) - 0.5 * x**2


class FlowMatchingLoss(seli.opt.Loss):
    def __init__(self, optimal_transport: bool = False):
        self.optimal_transport = optimal_transport

    def __call__(self, model, key, source, target):
        # 1D optimal transport is equivalent to sorting the source and target
        # distributions and matching the sorted values
        if self.optimal_transport:
            source = jnp.sort(source)
            target = jnp.sort(target)

        # sample a random time variable
        t = jrn.uniform(key, (source.shape[0],))

        # interpolate between the source and target distributions, in full
        # optimal transport this would be equivalent to the Wasserstein2
        # geodesic
        x_t = (1 - t) * source + t * target

        # compute the difference between the target and source distributions
        # this is the conditional flow of the forward process
        v = target - source
        v_model = model(x_t, t)

        # use the L2 loss to optimize the model, we need this to be the l2
        # loss to ensure that the model converges to the marginal flow, i.e.
        # the conditional expectation of the conditional flows
        # in the full optimal transport limit any pre-metric would suffice,
        # since geodesics do not cross and therefore the conditional expectation
        # becomes deterministic (only one possible output per condition)
        return jnp.mean(jnp.square(v_model - v))


# setup the training
model = Flow().set_rngs(42)
loss = FlowMatchingLoss(optimal_transport=True)
opt = seli.opt.Adam(3e-4)

batch_size = 256
key = jrn.PRNGKey(42)
losses = []

for _ in trange(10000, desc="Training"):
    key, key_t, key_source, key_target = jrn.split(key, 4)
    opt, model, loss_value = opt.minimize(
        loss,
        model,
        key_t,
        jrn.normal(key_source, (batch_size,)),
        two_gaussians(key_target, batch_size),
    )
    losses.append(loss_value)


num_steps = 1024 * 8
fig, (ax_loss, ax_samples) = plt.subplots(1, 2, figsize=(10, 5))

ax_loss.plot(losses)
ax_loss.set_yscale("log")

ax_loss.set_xlim(0, len(losses))
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("log loss")

ax_loss.set_title("Loss")
sns.despine(ax=ax_loss)

key, subkey = jrn.split(key)

x = jnp.linspace(-3, 3, 128)
y = jnp.exp(model.forward_with_log_likelihood(x, num_steps)[1])

y_true = two_gaussians_likelihood(x)
samples = two_gaussians(subkey, 1024 * 4)
samples_model = model.forward(jrn.normal(subkey, (1024 * 4,)), num_steps)

ax_samples.hist(
    [
        samples,
        samples_model,
    ],
    bins=32,
    density=True,
    label=["Data samples", "Flow samples"],
    histtype="step",
    color=["tab:blue", "tab:red"],
)

ax_samples.plot(x, y, label="Flow density", color="tab:red")
ax_samples.plot(x, y_true, label="True density", color="tab:blue")

ax_samples.set_xlim(x.min(), x.max())
ax_samples.set_xlabel("x")
ax_samples.set_ylabel("density")
ax_samples.legend(frameon=False, ncol=2)
ax_samples.set_title("Samples")

sns.despine(ax=ax_samples)
fig.suptitle("Optimal Transport Flow Matching")

plt.savefig(
    get_plot_path("flow_matching_1d.png"),
    dpi=256,
    bbox_inches="tight",
)
