"""
General utility functions.
"""

import jax


def dtype_summary(dtype: jax.numpy.dtype, /) -> str:
    """
    Compress the dtype to a short string string, float32 becomes f32, and
    int64 becomes i64.
    """
    return dtype.str[1:]


def array_summary(x: jax.Array, /) -> str:
    """
    Compress the array to a short string string.
    """
    shape = "×".join(str(d) for d in x.shape)
    dtype = dtype_summary(x.dtype)
    return f"{dtype}[{shape}]"
