import jax.numpy as jnp

import seli


class Loss(seli.opt.Loss):
    def __call__(self, array):
        print("compiling")
        return jnp.sum((array - 1) ** 2)


model = jnp.zeros(())

loss = Loss()
loss.collection = None

optimizer = seli.opt.SGD(lr=0.1)


loss_values = []

for _ in range(10):
    optimizer, model, loss_value = optimizer.minimize(loss, model)
    print(model, loss_value)
