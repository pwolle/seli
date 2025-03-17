import json

import jax
import jax.numpy as jnp
import numpy as np

from seli.core._module import Module
from seli.core._serialize import (
    ArrayPlaceholder,
    from_arrays_and_json,
    to_arrays_and_json,
)


class SimpleModule(Module, name="test_serialize.SimpleModule"):
    def __init__(self, value, array):
        self.value = value
        self.array = array


class NestedModule(Module, name="test_serialize.NestedModule"):
    def __init__(self, simple_module, extra_value):
        self.simple_module = simple_module
        self.extra_value = extra_value


class ComplexModule(Module, name="test_serialize.ComplexModule"):
    def __init__(self):
        self.value = 42
        self.array = jnp.array([1, 2, 3])
        self.nested = {"key": jnp.array([4, 5, 6])}
        self.nested_module = NestedModule(
            SimpleModule(100, jnp.array([7, 8, 9])), "extra"
        )
        self.list_of_arrays = [jnp.array([10, 11]), jnp.array([12, 13, 14])]


def test_array_placeholder():
    placeholder = ArrayPlaceholder(42)
    assert placeholder.index == 42


def test_serialize_simple_scalar():
    value = 42
    arrays, json_str = to_arrays_and_json(value)

    assert len(arrays) == 0  # No arrays in a scalar value

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    assert result == value


def test_serialize_simple_array():
    array = jnp.array([1, 2, 3])
    arrays, json_str = to_arrays_and_json(array)

    assert len(arrays) == 1
    np.testing.assert_array_equal(arrays[0], array)

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    np.testing.assert_array_equal(result, array)


def test_serialize_simple_module():
    module = SimpleModule(42, jnp.array([1, 2, 3]))
    arrays, json_str = to_arrays_and_json(module)

    assert len(arrays) == 1
    np.testing.assert_array_equal(arrays[0], module.array)

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    assert isinstance(result, SimpleModule)
    assert result.value == module.value
    np.testing.assert_array_equal(result.array, module.array)


def test_serialize_nested_module():
    simple = SimpleModule(42, jnp.array([1, 2, 3]))
    module = NestedModule(simple, "extra_value")

    arrays, json_str = to_arrays_and_json(module)

    assert len(arrays) == 1
    np.testing.assert_array_equal(arrays[0], simple.array)

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    assert isinstance(result, NestedModule)
    assert isinstance(result.simple_module, SimpleModule)
    assert result.simple_module.value == simple.value
    np.testing.assert_array_equal(result.simple_module.array, simple.array)
    assert result.extra_value == module.extra_value


def test_serialize_complex_module():
    module = ComplexModule()

    arrays, json_str = to_arrays_and_json(module)

    # The module has 5 arrays
    assert len(arrays) == 5

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    assert isinstance(result, ComplexModule)
    assert result.value == module.value
    np.testing.assert_array_equal(result.array, module.array)
    np.testing.assert_array_equal(result.nested["key"], module.nested["key"])

    assert isinstance(result.nested_module, NestedModule)
    assert isinstance(result.nested_module.simple_module, SimpleModule)
    assert (
        result.nested_module.simple_module.value
        == module.nested_module.simple_module.value
    )
    np.testing.assert_array_equal(
        result.nested_module.simple_module.array,
        module.nested_module.simple_module.array,
    )

    assert len(result.list_of_arrays) == len(module.list_of_arrays)
    for i in range(len(result.list_of_arrays)):
        np.testing.assert_array_equal(
            result.list_of_arrays[i], module.list_of_arrays[i]
        )


def test_serialize_with_empty_structures():
    # Create a module with empty structures
    class EmptyStructuresModule(Module, name="test.EmptyStructuresModule"):
        def __init__(self):
            self.empty_list = []
            self.empty_dict = {}
            self.value = 42

    module = EmptyStructuresModule()

    arrays, json_str = to_arrays_and_json(module)

    # No arrays
    assert len(arrays) == 0

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    assert isinstance(result, EmptyStructuresModule)
    assert result.empty_list == []
    assert result.empty_dict == {}
    assert result.value == 42


def test_serialize_with_none_values():
    # Create a module with None values
    class NoneValuesModule(Module, name="test.NoneValuesModule"):
        def __init__(self):
            self.none_value = None
            self.array = jnp.array([1, 2, 3])
            self.list_with_none = [1, None, jnp.array([4, 5])]

    module = NoneValuesModule()

    arrays, json_str = to_arrays_and_json(module)

    # There should be 2 arrays
    assert len(arrays) == 2

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    assert isinstance(result, NoneValuesModule)
    assert result.none_value is None
    np.testing.assert_array_equal(result.array, module.array)
    assert result.list_with_none[0] == 1
    assert result.list_with_none[1] is None
    np.testing.assert_array_equal(result.list_with_none[2], module.list_with_none[2])


def test_jit_with_serialization():
    class SimpleJitModule(Module, name="test.SimpleJitModule"):
        def __init__(self, x):
            self.value = x
            self.array = jnp.array([x, x + 1, x + 2])

        def compute(self, y):
            return self.value * y + self.array.sum()

    # Create a function that will serialize, then deserialize, then use the module
    def workflow(module, y):
        arrays, json_str = to_arrays_and_json(module)
        restored = from_arrays_and_json(arrays, json_str)
        return restored.compute(y)

    # JIT the function
    jit_workflow = jax.jit(workflow)

    # Create a module and test
    module = SimpleJitModule(5)
    y = 3

    # Get expected result without JIT
    expected = workflow(module, y)

    # Get result with JIT
    result = jit_workflow(module, y)

    # Results should be equal
    np.testing.assert_allclose(result, expected)


def test_serialize_nested_dict_with_arrays():
    # Create a nested dictionary with arrays
    nested_dict = {
        "level1": {
            "array1": jnp.array([1, 2, 3]),
            "level2": {"array2": jnp.array([4, 5, 6]), "value": 42},
        }
    }

    arrays, json_str = to_arrays_and_json(nested_dict)

    # There should be 2 arrays
    assert len(arrays) == 2

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    # Check structure
    assert "level1" in result
    assert "array1" in result["level1"]
    assert "level2" in result["level1"]
    assert "array2" in result["level1"]["level2"]
    assert "value" in result["level1"]["level2"]

    # Check values
    np.testing.assert_array_equal(
        result["level1"]["array1"], nested_dict["level1"]["array1"]
    )
    np.testing.assert_array_equal(
        result["level1"]["level2"]["array2"], nested_dict["level1"]["level2"]["array2"]
    )
    assert (
        result["level1"]["level2"]["value"] == nested_dict["level1"]["level2"]["value"]
    )


def test_serialize_list_of_modules():
    # Create a list of modules
    modules = [SimpleModule(i, jnp.array([i, i + 1, i + 2])) for i in range(3)]

    arrays, json_str = to_arrays_and_json(modules)

    # There should be 3 arrays, one for each module
    assert len(arrays) == 3

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    # Check length
    assert len(result) == len(modules)

    # Check each module
    for i, module in enumerate(result):
        assert isinstance(module, SimpleModule)
        assert module.value == modules[i].value
        np.testing.assert_array_equal(module.array, modules[i].array)


def test_serialize_class_with_slots():
    class SlotsModule(Module, name="test.SlotsModule"):
        __slots__ = ["value", "array"]

        def __init__(self, value, array):
            self.value = value
            self.array = array

    module = SlotsModule(42, jnp.array([1, 2, 3]))

    arrays, json_str = to_arrays_and_json(module)

    # There should be 1 array
    assert len(arrays) == 1

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    # Check values
    assert isinstance(result, SlotsModule)
    assert result.value == module.value
    np.testing.assert_array_equal(result.array, module.array)


def test_serialize_different_array_dtypes():
    class DtypesModule(Module, name="test.DtypesModule"):
        def __init__(self):
            self.int_array = jnp.array([1, 2, 3], dtype=jnp.int32)
            self.float_array = jnp.array([1.1, 2.2, 3.3], dtype=jnp.float32)
            self.bool_array = jnp.array([True, False, True], dtype=jnp.bool_)

    module = DtypesModule()

    arrays, json_str = to_arrays_and_json(module)

    # There should be 3 arrays
    assert len(arrays) == 3

    # Deserialize
    result = from_arrays_and_json(arrays, json_str)

    # Check values and dtypes
    assert isinstance(result, DtypesModule)
    np.testing.assert_array_equal(result.int_array, module.int_array)
    assert result.int_array.dtype == module.int_array.dtype

    np.testing.assert_array_equal(result.float_array, module.float_array)
    assert result.float_array.dtype == module.float_array.dtype

    np.testing.assert_array_equal(result.bool_array, module.bool_array)
    assert result.bool_array.dtype == module.bool_array.dtype


def test_round_trip_equivalence():
    # Create a complex module
    module = ComplexModule()

    # First serialization
    arrays1, json_str1 = to_arrays_and_json(module)

    # Deserialize
    result1 = from_arrays_and_json(arrays1, json_str1)

    # Second serialization of the deserialized result
    arrays2, json_str2 = to_arrays_and_json(result1)

    # Deserialize again
    result2 = from_arrays_and_json(arrays2, json_str2)

    # The JSON strings should be equivalent (though not necessarily identical)
    # because they represent the same structure
    loaded1 = json.loads(json_str1)
    loaded2 = json.loads(json_str2)
    assert loaded1 == loaded2

    # Check arrays are the same shape and type
    assert len(arrays1) == len(arrays2)
    for arr1, arr2 in zip(arrays1, arrays2):
        assert arr1.shape == arr2.shape
        assert arr1.dtype == arr2.dtype
        np.testing.assert_allclose(arr1, arr2)

    # Check results are equivalent
    assert isinstance(result1, ComplexModule)
    assert isinstance(result2, ComplexModule)
    assert result1.value == result2.value
    np.testing.assert_array_equal(result1.array, result2.array)

    # Check nested structures
    np.testing.assert_array_equal(result1.nested["key"], result2.nested["key"])

    # Check nested modules
    assert result1.nested_module.extra_value == result2.nested_module.extra_value
    assert (
        result1.nested_module.simple_module.value
        == result2.nested_module.simple_module.value
    )
    np.testing.assert_array_equal(
        result1.nested_module.simple_module.array,
        result2.nested_module.simple_module.array,
    )

    # Check lists of arrays
    for i in range(len(result1.list_of_arrays)):
        np.testing.assert_array_equal(
            result1.list_of_arrays[i], result2.list_of_arrays[i]
        )
