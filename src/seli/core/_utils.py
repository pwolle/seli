"""
General utility functions.
"""

import jax


def dtype_summary(dtype: jax.numpy.dtype, /) -> str:
    return dtype.str[1:]


def array_summary(x: jax.Array, /) -> str:
    shape = "×".join(str(d) for d in x.shape)
    dtype = dtype_summary(x.dtype)
    return f"{dtype}[{shape}]"
