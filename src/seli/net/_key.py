from jax import random as jrn
from jaxtyping import PRNGKeyArray

from seli.core._module import Module, NodeType, dfs_map

__all__ = [
    "Key",
    "set_keys",
]


class Key(Module, name="net.Key"):
    """
    Placeholder for a Jax PRNG key. The class does not need to be initialized
    with a key directly, but can be used to indicate a spot in a module which
    needs to be filled with a key later.

    All key submodules can be initialized with a key later by calling
    `set_keys` with a PRNG key or an integer seed.

    The collections attribute can be used to group keys, for example when
    initializing the same module on different devices the keys for the
    parameters should probably be the same, while the keys for dropout can be
    different.
    """

    collection: str | None

    def __init__(
        self,
        key: PRNGKeyArray | None = None,
        collection: str | None = None,
    ):
        self._key = key
        self.collection = collection

    @property
    def initialized(self) -> bool:
        return self._key is not None

    @property
    def key(self) -> PRNGKeyArray:
        if self._key is None:
            error = "Key has not been set"
            raise ValueError(error)

        self._key, key = jrn.split(self._key)
        return key


def set_keys(
    module: NodeType,
    key_or_seed: PRNGKeyArray | int,
    collection: list[str] | None = None,
):
    """
    Initialize all Key submodules in the given module. If a collection is
    provided only keys in the given collection will be initialized.

    Parameters
    ---
    module: NodeType
        The module to initialize the keys for.

    key_or_seed: PRNGKeyArray | int
        The PRNG key or seed to use for initialization.

    collection: list[str] | None
        The collection to initialize the keys for.

    Returns
    ---
    module: NodeType
        The module with the keys initialized.
    """
    if isinstance(key_or_seed, int):
        key_or_seed = jrn.PRNGKey(key_or_seed)

    assert isinstance(key_or_seed, PRNGKeyArray)

    keys_in_module = []

    def fun(_, node):
        if isinstance(node, Key) and node.initialized:
            if collection is None or node.collection in collection:
                keys_in_module.append(node)

        return

    module = dfs_map(module, fun)

    for key, key_module in zip(
        jrn.split(key_or_seed, len(keys_in_module)),
        keys_in_module,
    ):
        key_module._key = key

    return module
