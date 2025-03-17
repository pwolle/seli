"""
This module contains environment variables.
"""

import jax.numpy as jnp
from jax.typing import DTypeLike

__all__ = [
    "DEFAULT_FLOAT",
]

DEFAULT_FLOAT: DTypeLike = jnp.float32
DEFAULT_FLOAT.__doc__ = """The default floating point type"""
