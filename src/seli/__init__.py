"""Seli package."""

__version__ = "0.1.0"

from .core._module import AttrKey, ItemKey, Module, PathKey, dfs_map

__all__ = ["Module", "ItemKey", "AttrKey", "PathKey", "dfs_map"]
