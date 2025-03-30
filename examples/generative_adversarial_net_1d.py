"""
Implementation of a Generative Adversarial Network (GAN).

GANs consist of a generator network and a discriminator network.
The generator network maps random noise to samples that look like the data,
while the discriminator network tries to distinguish between the generated
samples and the true data.

The training process alternates between training the generator to generate
more realistic samples and training the discriminator to become better at
distinguishing between real and generated samples.
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from utils import get_plot_path, two_gaussians

import seli


# Simple multi-layer perceptron with ReLU activation function for the generator
# and the discriminator.
class Network(seli.Module):
    def __init__(self, dim: int = 128):
        self.layers = [
            seli.net.Affine(dim),
            jnn.leaky_relu,
            seli.net.Affine(dim),
            jnn.leaky_relu,
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


class LossGenerator(seli.opt.Loss):
    def __call__(self, generator, discriminator, key, batch_size: int):
        samples_fake = generator.sample(key, batch_size)
        pred = discriminator(samples_fake)

        # minimize the prediction of the discriminator on the generated samples,
        # i.e. the generated sample should look like the true data, this is
        # also called the wasserstein loss
        return jnp.mean(pred)


class LossDiscriminator(seli.opt.Loss):
    def __call__(self, discriminator, generator, key, samples_real):
        @jax.vmap
        @jax.value_and_grad
        def discriminator_with_grad(x):
            return discriminator(x)

        batch_size = samples_real.shape[0]
        samples_fake = generator.sample(key, batch_size)

        preds_real, grads_real = discriminator_with_grad(samples_real)
        preds_fake, grads_fake = discriminator_with_grad(samples_fake)

        # wasserstein loss, minimize the prediction on the real data and
        # maximize the prediction on the generated samples
        pred_loss = preds_real.mean() - preds_fake.mean()

        # R1 and R2 gradient penalty, this ensures that the discriminator
        # is more Lipschitz continuous, which is a property that is desirable
        # for the stability of the training and has some theoretical
        # justifications in the computation of the Wasserstein-1-distance
        grad_loss_real = jnp.square(grads_real).mean()
        grad_loss_fake = jnp.square(grads_fake).mean()
        grad_loss = grad_loss_real + grad_loss_fake

        return pred_loss + grad_loss * 5


# the training step consists in alternating the optimization of the generator
# and the discriminator networks
@seli.jit
def train_step(
    generator,
    discriminator,
    key,
    batch_size: int,
    opt_generator,
    opt_discriminator,
):
    # unroll multiple optimization steps for better performance, since
    # the neural networks are small, so the pre and post jit calls are
    # a significant overhead
    for _ in range(5):
        key, key_generator1, key_generator2, key_data = jrn.split(key, 4)

        opt_generator, generator, _ = opt_generator.minimize(
            LossGenerator(),
            generator,
            discriminator,
            key_generator1,
            batch_size,
        )

        opt_discriminator, discriminator, _ = opt_discriminator.minimize(
            LossDiscriminator(),
            discriminator,
            generator,
            key_generator2,
            two_gaussians(key_data, batch_size),
        )

    return generator, discriminator, opt_generator, opt_discriminator


# initialize the generator and the discriminator and their optimizers
generator = Network().set_rngs(0)
discriminator = Network().set_rngs(1)

opt_generator = seli.opt.Adam(1e-3)
opt_discriminator = seli.opt.Adam(2e-3)

batch_size = 128
key = jrn.PRNGKey(42)

# train the generator and the discriminator for 10000 steps
for _ in trange(10000 // 5, desc="Training"):
    key, subkey = jrn.split(key)
    generator, discriminator, opt_generator, opt_discriminator = train_step(
        generator,
        discriminator,
        subkey,
        batch_size,
        opt_generator,
        opt_discriminator,
    )


# plot the generated samples and the true data
fig, ax = plt.subplots(figsize=(5, 5))

ax.hist(
    [
        generator.sample(key, 1024 * 32),
        two_gaussians(key, 1024 * 32),
    ],
    bins=64,
    density=True,
    label=["Generator", "True"],
    histtype="step",
    color=["tab:red", "tab:blue"],
)
ax.set_xlim(-3, 3)
ax.legend(frameon=False, ncol=2)

ax.set_title("Generative Adversarial Network")
ax.set_xlabel("x")
ax.set_ylabel("density")

sns.despine(ax=ax)
fig.savefig(get_plot_path("generative_adversarial_net_1d.png"))
