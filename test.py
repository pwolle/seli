import jax.nn.initializers as jni
import jax.random as jrn

key = jrn.PRNGKey(0)

w = jni.orthogonal()(key, (10, 10, 100))

print(w.shape)
