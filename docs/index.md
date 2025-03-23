# Seli Documentation

Welcome to the documentation for seli, a library for building flexible neural networks in JAX.

Seli minimizes the time from idea to implementation with flexible neural networks by combining the elegance of PyTorch-style modules with the power of JAX.


## Key Features

- **Mutable modules**: Quickly modify modules during development via the `Module` class
- **Serialization**: Save and load models easily with `@saveable`, `save`, and `load`
- **Systematic modifications**: Traverse nested modules to make structural changes
- **Reference handling**: Safe handling of shared/cyclical references and static arguments through `seli.jit`
- **Built-in components**: Common neural network layers and optimizers are included
- **Simple codebase**: The library is relatively easy to understand and extend

## Installation

You can install seli from PyPI using pip:

```bash
pip install seli
```


## Getting Started

```python
import seli as sl
import jax.numpy as jnp

# Define a model by subclassing seli.Module
class Linear(sl.Module, name="example:Linear"):
    def __init__(self, dim: int):
        self.dim = dim
        # Parameters can be directly initialized or use initialization methods
        self.weight = sl.net.Param(init=sl.net.Normal("Kaiming"))

    def __call__(self, x):
        # The weight gets initialized on the first call
        # by providing the shape, the value is stored
        return x @ self.weight((x.shape[-1], self.dim))

# Set the random number generators for all submodules at once
model = Linear(10).set_rngs(42)
y = model(jnp.ones(8))

# Train the model using a built-in optimizer
optimizer = sl.opt.Adam(1e-3)
loss = sl.opt.MeanSquaredError()

x = jnp.ones((32, 8))
y = jnp.ones((32, 10))

optimizer, model, loss_value = optimizer.minimize(loss, model, y, x)

# Save and load the model
sl.save(model, "model.npz")
loaded_model = sl.load("model.npz")
```


```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

usage
```

```{toctree}
:maxdepth: 3
:hidden:
:caption: API Reference

seli
net
opt
```
