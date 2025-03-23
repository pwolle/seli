"""
General utility functions.
"""

import jax


def dtype_summary(dtype: jax.numpy.dtype, /) -> str:
    """
    Compress the dtype to a short string string, float32 becomes f32, and
    int64 becomes i64.

    Parameters
    ----------
    dtype : jax.numpy.dtype
        The dtype to compress.

    Returns
    -------
    str
        A short alias for the dtype.

    Examples
    --------
    >>> dtype_summary(jnp.float32)
    'f32'
    >>> dtype_summary(jnp.int64)
    'i64'
    """
    return dtype.str[1:]


def array_summary(x: jax.Array, /) -> str:
    """
    Compress the array to a short string string. The resulting string can be
    used to identify the shape and dtype of the array in a human readable way.

    Parameters
    ----------
    x : jax.Array
        The array to compress.

    Returns
    -------
    str
        A short string string that identifies the shape and dtype of the array.

    Examples
    --------
    >>> array_summary(jnp.array([1, 2, 3]))
    'f32[3]'
    >>> array_summary(jnp.array([[1, 2], [3, 4]]))
    'f32[2×2]'
    """
    shape = "×".join(str(d) for d in x.shape)
    dtype = dtype_summary(x.dtype)
    return f"{dtype}[{shape}]"
