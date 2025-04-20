<h1 align="center">
    Seli - Fast Neural Networks Research in Jax
</h1>

<p align="center">
    <a href="https://github.com/pwolle/seli/actions/workflows/pytest.yml"><img src="https://github.com/pwolle/seli/actions/workflows/pytest.yml/badge.svg" alt="Python Tests"></a>
    <a href="https://pypi.org/project/seli/"><img src="https://img.shields.io/pypi/v/seli.svg" alt="PyPI version"></a>
    <a href="https://seli.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/seli/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/seli"><img src="https://pepy.tech/badge/seli/month" alt="Downloads"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>


Minimizing the time from idea to implementation with flexible neural networks in seli. Less boilerplate, more utilities for quick iteration speed.


## Features
- Mutable modules for quick and dirty modifications via Module
- Serialization of modules via `@saveable`, `save`, and `load`
- Systematically modifying modules by traversing nested modules
- Safely handling shared/cyclical references and static arguments through `seli.jit`
- Commonly used NN layers and optimizers are included
- As a small codebase, it is relatively easy to understand and extend


## Quick Example

Define new layers by subclassing `seli.Module`. All modules are PyTrees.
Check out the [examples](examples) for more sophisticated usage.

``` python
# add a name to make the module saveable
class Linear(seli.Module, name="example:Linear");
    def __init__(self, dim: int)
        self.dim = dim

        # parameters can be directly initialized
        # or an initialization method can be passed
        self.weight = seli.netParam(init=seli.net.InitNormal("He"))

    def __call__(self, x):
        # the weight gets initialized on the first call
        # by providing the shape, the value is stored
        return x @ self.weight((x.shape[-1], self.dim))

# set the rngs for all submodules at once
# no code for passing rngs around is needed
model = Linear(10).set_rngs(42)
y = model(jnp.ones(8))
```

A training step can be written as follows, it requires python 3.11 or newer.

``` python
optimizer = seli.opt.Adam(1e-3)
loss = seli.opt.MeanSquaredError()

x = jax.numpy.ones(32, 8)
y = jax.numpy.ones(32, 10)

optimizer, model, loss_value = optimizer.minimize(loss, model, y, x)
```

Models can be serialized and loaded. This process is safe and does not use pickling.

``` python
seli.save(model, "model.npz")

# load the model
model = seli.load("model.npz")
assert isinstance(model, Linear)
```

## Installation

You can install from PyPI using pip

```bash
pip install seli
```

## See Also
- [Jax Docs](https://jax.readthedocs.io/en/latest/): Jax is a library for numerical computing that is designed to be composable and fast.
- [Equinox library](https://github.com/patrick-kidger/equinox): FlareJax is heavily inspired by this awesome library.
- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html): Many of the principles of mutability are inspired by PyTorch's `torch.nn.Module`.
- [NNX Docs](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html/): NNX is a library for neural networks in flax that also supports mutability.
- Always feel free to reach out to me via [email](mailto:paul.wollenhaupt@gmail.com).
