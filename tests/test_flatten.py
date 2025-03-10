import jax
import jax.numpy as jnp
import numpy as np

from seli.core._module import (
    AttrKey,
    ItemKey,
    Module,
    PathKey,
    flat_path_dict,
)


class _TestModuleClass(Module, name="test_flatten.TestModuleClass"):
    def __init__(self):
        self.value = 1
        self.array = jnp.array([1, 2, 3])
        self.nested = {"key": jnp.array([4, 5, 6])}
        self.nested_module = _NestedModuleClass()


class _NestedModuleClass(Module, name="test_flatten.NestedModuleClass"):
    def __init__(self):
        self.value = 2
        self.array = jnp.array([7, 8, 9])


def test_flat_path_dict_simple():
    data = {"a": 1, "b": 2, "c": 3}
    flat = flat_path_dict(data)

    # Check that the flat dictionary contains the expected keys and values
    assert flat[PathKey([ItemKey("a")])] == 1
    assert flat[PathKey([ItemKey("b")])] == 2
    assert flat[PathKey([ItemKey("c")])] == 3
    assert (
        flat[PathKey([AttrKey("__class__")])] is dict
    )  # Check for actual class object


def test_flat_path_dict_nested_dict():
    data = {"a": 1, "b": {"c": 2, "d": 3}}
    flat = flat_path_dict(data)

    # Check that the flat dictionary contains the expected keys and values
    assert flat[PathKey([ItemKey("a")])] == 1
    assert (
        flat[PathKey([ItemKey("b"), AttrKey("__class__")])] is dict
    )  # Check for actual class object
    assert flat[PathKey([ItemKey("b"), ItemKey("c")])] == 2
    assert flat[PathKey([ItemKey("b"), ItemKey("d")])] == 3
    assert (
        flat[PathKey([AttrKey("__class__")])] is dict
    )  # Check for actual class object


def test_flat_path_dict_module():
    module = _TestModuleClass()
    flat = flat_path_dict(module)

    # Check that the flat dictionary contains the expected keys and values
    assert flat[PathKey([AttrKey("value")])] == 1
    assert isinstance(flat[PathKey([AttrKey("array")])], jax.Array)
    np.testing.assert_array_equal(
        flat[PathKey([AttrKey("array")])], jnp.array([1, 2, 3])
    )
    assert flat[PathKey([AttrKey("nested"), AttrKey("__class__")])] is dict
    np.testing.assert_array_equal(
        flat[PathKey([AttrKey("nested"), ItemKey("key")])], jnp.array([4, 5, 6])
    )
    # Check that the class is preserved
    assert (
        flat[PathKey([AttrKey("nested_module"), AttrKey("__class__")])]
        is _NestedModuleClass
    )
    assert flat[PathKey([AttrKey("nested_module"), AttrKey("value")])] == 2
    np.testing.assert_array_equal(
        flat[PathKey([AttrKey("nested_module"), AttrKey("array")])],
        jnp.array([7, 8, 9]),
    )


def test_flat_path_dict_circular_reference():
    # Create a circular reference
    a = {"key": "value"}
    b = {"a": a}
    a["b"] = b  # Create circular reference

    flat = flat_path_dict(a)

    # The circular reference should be represented as a PathKey
    circular_ref_key = PathKey([ItemKey("b"), ItemKey("a")])
    assert isinstance(flat[circular_ref_key], PathKey)
    # The path key should point to the root
    assert flat[circular_ref_key] == PathKey([])


def test_flat_path_dict_shared_reference():
    # Create a shared reference
    shared = {"key": "value"}
    data = {
        "a": shared,
        "b": shared,  # Same object referenced twice
    }

    flat = flat_path_dict(data)

    # The second shared reference should be represented as a PathKey
    shared_ref_key = PathKey([ItemKey("b")])
    assert isinstance(flat[shared_ref_key], PathKey)
    # The path key should point to the first occurrence
    assert flat[shared_ref_key] == PathKey([ItemKey("a")])


def test_tree_flatten_unflatten_simple():
    module = _TestModuleClass()

    # Flatten the module
    arrays, aux_data = Module.tree_flatten(module)

    # Check that arrays contains all jax.Array objects
    assert len(arrays) == 3  # There are 3 arrays in the module
    assert all(isinstance(arr, jax.Array) for arr in arrays)

    # Unflatten the module
    reconstructed = Module.tree_unflatten(aux_data, arrays)

    # Check that the reconstructed module has the expected attributes
    assert isinstance(
        reconstructed, _TestModuleClass
    )  # Result has the same class as the original
    assert reconstructed.value == 1
    np.testing.assert_array_equal(reconstructed.array, jnp.array([1, 2, 3]))
    np.testing.assert_array_equal(reconstructed.nested["key"], jnp.array([4, 5, 6]))
    assert reconstructed.nested_module.value == 2
    np.testing.assert_array_equal(
        reconstructed.nested_module.array, jnp.array([7, 8, 9])
    )


def test_tree_flatten_unflatten_jax_jit():
    class SimpleModule(Module, name="test_flatten.SimpleModule"):
        def __init__(self, x):
            self.weight = x
            self.bias = x * 2

    module = SimpleModule(jnp.array([1.0, 2.0, 3.0]))
    module_new = jax.jit(lambda x: x)(module)

    assert module.weight.shape == module_new.weight.shape
    assert module.bias.shape == module_new.bias.shape

    np.testing.assert_array_equal(module.weight, module_new.weight)
    np.testing.assert_array_equal(module.bias, module_new.bias)


def test_tree_flatten_unflatten_complex():
    # Create a complex module with nested arrays
    class ComplexModule(Module, name="test_flatten.ComplexModule"):
        def __init__(self):
            self.arrays = {
                "a": jnp.array([1, 2, 3]),
                "b": [jnp.array([4, 5]), jnp.array([6, 7])],
                "c": {"d": jnp.array([8, 9, 10]), "e": jnp.array([11, 12])},
            }
            self.nested = NestedModuleForComplex()
            self.value = "not an array"

    class NestedModuleForComplex(Module, name="test_flatten.NestedModuleForComplex"):
        def __init__(self):
            self.value = 2
            self.array = jnp.array([7, 8, 9])

    module = ComplexModule()

    # Flatten the module
    arrays, aux_data = Module.tree_flatten(module)

    # Check that arrays contains all jax.Array objects
    assert len(arrays) == 6  # There are 6 arrays in the complex structure
    assert all(isinstance(arr, jax.Array) for arr in arrays)

    # Apply a transformation to the arrays
    transformed_arrays = [arr + 1 for arr in arrays]

    # Unflatten the module
    reconstructed = Module.tree_unflatten(aux_data, transformed_arrays)

    # Check that the arrays have been transformed
    np.testing.assert_array_equal(reconstructed.arrays["a"], jnp.array([2, 3, 4]))
    np.testing.assert_array_equal(reconstructed.arrays["b"][0], jnp.array([5, 6]))
    np.testing.assert_array_equal(reconstructed.arrays["b"][1], jnp.array([7, 8]))
    np.testing.assert_array_equal(
        reconstructed.arrays["c"]["d"], jnp.array([9, 10, 11])
    )
    np.testing.assert_array_equal(reconstructed.arrays["c"]["e"], jnp.array([12, 13]))
    np.testing.assert_array_equal(reconstructed.nested.array, jnp.array([8, 9, 10]))

    # Non-array values should remain unchanged
    assert reconstructed.value == "not an array"
    assert reconstructed.nested.value == 2


def test_tree_jit_shared_references():
    class SharedModule(Module, name="test_flatten.SharedModule"):
        def __init__(self, x):
            self.shared1 = {"key": x}
            self.shared2 = self.shared1
            self.self = self

    module = SharedModule(jnp.array([1, 2, 3]))
    module_new = jax.jit(lambda x: x)(module)

    np.testing.assert_array_equal(module.shared1["key"], module_new.shared1["key"])

    assert module_new is module_new.self
    assert module_new.shared1 is module_new.shared2
    assert module_new.shared1 is not module.shared1
