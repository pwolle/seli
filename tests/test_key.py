import jax.numpy as jnp
import jax.random as jrn
import pytest

from seli.core._module import Module
from seli.net._key import RNGs, set_rngs


def test_key_init():
    # Test initialization without a key
    key_module = RNGs()
    assert key_module.collection is None
    assert not key_module.initialized

    # Test initialization with a key
    test_key = jrn.PRNGKey(0)
    key_module = RNGs(key=test_key, collection="test_collection")
    assert key_module.collection == "test_collection"
    assert key_module.initialized


def test_key_property():
    # Test that the key property splits the key correctly
    test_key = jrn.PRNGKey(0)
    key_module = RNGs(key=test_key)

    # Get key, which should split the internal key
    first_key = key_module.key

    # Get another key, which should be different
    second_key = key_module.key
    assert not jnp.array_equal(first_key, second_key)

    # Ensure we don't use up the original key
    assert key_module.initialized


def test_key_property_error():
    # Test that accessing the key property raises an error if not initialized
    key_module = RNGs()
    with pytest.raises(ValueError, match="Key has not been set"):
        _ = key_module.key


class DeepModule(Module, name="test_key.DeepModule"):
    def __init__(self):
        self.key1 = RNGs(collection="collection1")
        self.key2 = RNGs(collection="collection2")
        self.nested = NestedModule()


class NestedModule(Module, name="test_key.NestedModule"):
    def __init__(self):
        self.key3 = RNGs(collection="collection1")
        self.key4 = RNGs(collection="collection2")


def test_set_keys_with_key():
    # Test set_keys with a PRNGKey
    module = DeepModule()
    test_key = jrn.PRNGKey(0)

    # Verify keys are not initialized
    assert not module.key1.initialized
    assert not module.key2.initialized
    assert not module.nested.key3.initialized
    assert not module.nested.key4.initialized

    # Set keys
    result = set_rngs(module, test_key)

    # Verify all keys are initialized
    assert result.key1.initialized
    assert result.key2.initialized
    assert result.nested.key3.initialized
    assert result.nested.key4.initialized


def test_set_keys_with_seed():
    # Test set_keys with an integer seed
    module = DeepModule()
    seed = 42

    # Set keys using the seed
    result = set_rngs(module, seed)

    # Verify all keys are initialized
    assert result.key1.initialized
    assert result.key2.initialized
    assert result.nested.key3.initialized
    assert result.nested.key4.initialized


def test_set_keys_with_collection():
    # Test set_keys with a specific collection
    module = DeepModule()
    test_key = jrn.PRNGKey(0)

    # Set keys only for collection1
    result = set_rngs(module, test_key, collection=["collection1"])

    # Verify only keys in collection1 are initialized
    assert result.key1.initialized
    assert not result.key2.initialized
    assert result.nested.key3.initialized
    assert not result.nested.key4.initialized


def test_set_keys_with_already_initialized():
    # Test that already initialized keys are not re-initialized
    module = DeepModule()

    # Pre-initialize one key
    initial_key = jrn.PRNGKey(1)
    module.key1._key = initial_key

    # Set keys for all
    test_key = jrn.PRNGKey(0)
    result = set_rngs(module, test_key)

    # Verify the pre-initialized key produced the same output
    # as if it had been initialized with test_key
    assert result.key1._key is initial_key
    assert result.key2.initialized

    assert result.nested.key3.initialized
    assert result.nested.key4.initialized


def test_set_keys_determinism():
    # Test that set_keys is deterministic with the same seed
    module1 = DeepModule()
    module2 = DeepModule()
    seed = 42

    # Set keys for both modules
    result1 = set_rngs(module1, seed)
    result2 = set_rngs(module2, seed)

    # Get keys from both modules
    key1_1 = result1.key1._key
    key1_2 = result2.key1._key
    key2_1 = result1.key2._key
    key2_2 = result2.key2._key

    # Verify keys are the same for the same position in different modules
    assert jnp.array_equal(key1_1, key1_2)
    assert jnp.array_equal(key2_1, key2_2)


class ParameterizedModule(Module, name="test_key.ParameterizedModule"):
    def __init__(self):
        self.dropout_key = RNGs(collection="dropout")
        self.params_key = RNGs(collection="params")
        self.nested = {
            "layer1": NestedParameterizedModule(),
            "layer2": NestedParameterizedModule(),
        }


class NestedParameterizedModule(Module, name="test_key.NestedParameterizedModule"):
    def __init__(self):
        self.dropout_key = RNGs(collection="dropout")
        self.params_key = RNGs(collection="params")


def test_set_keys_complex_structure():
    # Test set_keys with a more complex module structure
    module = ParameterizedModule()
    test_key = jrn.PRNGKey(0)

    # Set only dropout keys
    result = set_rngs(module, test_key, collection=["dropout"])

    # Verify only dropout keys are initialized
    assert result.dropout_key.initialized
    assert not result.params_key.initialized
    assert result.nested["layer1"].dropout_key.initialized
    assert not result.nested["layer1"].params_key.initialized
    assert result.nested["layer2"].dropout_key.initialized
    assert not result.nested["layer2"].params_key.initialized

    # Now set params keys
    new_key = jrn.PRNGKey(1)
    result = set_rngs(result, new_key, collection=["params"])

    # Verify all keys are initialized
    assert result.dropout_key.initialized
    assert result.params_key.initialized
    assert result.nested["layer1"].dropout_key.initialized
    assert result.nested["layer1"].params_key.initialized
    assert result.nested["layer2"].dropout_key.initialized
    assert result.nested["layer2"].params_key.initialized
