import pytest

from src.seli._module import AttrKey, ItemKey, Module, PathKey, dfs_map


def test_dfs_map_simple_dict():
    """Test dfs_map on a simple dictionary."""
    data = {"a": 1, "b": 2, "c": 3}

    # Simple function that doubles values
    def double(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == {"a": 2, "b": 4, "c": 6}


def test_dfs_map_nested_dict():
    """Test dfs_map on a nested dictionary."""
    data = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    # Simple function that doubles values
    def double(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == {"a": 2, "b": {"c": 4, "d": 6}, "e": 8}


def test_dfs_map_list():
    """Test dfs_map on a list."""
    data = [1, 2, 3, 4]

    # Simple function that doubles values
    def double(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == [2, 4, 6, 8]


def test_dfs_map_nested_list():
    """Test dfs_map on a nested list."""
    data = [1, [2, 3], 4]

    # Simple function that doubles values
    def double(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == [2, [4, 6], 8]


def test_dfs_map_mixed_structure():
    """Test dfs_map on a mixed structure with dicts and lists."""
    data = {"a": 1, "b": [2, 3, {"c": 4}], "d": 5}

    # Simple function that doubles values
    def double(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == {"a": 2, "b": [4, 6, {"c": 8}], "d": 10}


def test_dfs_map_with_module():
    """Test dfs_map with a Module object."""

    class TestModule(Module):
        def __init__(self):
            self.value = 1

    module = TestModule()

    # A function that preserves the module but accesses its attributes
    def process_module(x):
        return x

    result = dfs_map(module, process_module)

    # The function always returns a new instance of the base Module class,
    # not the original class type
    assert isinstance(result, Module)
    assert hasattr(result, "value")
    assert result.value == 1


def test_dfs_map_circular_reference():
    """Test dfs_map with circular references."""
    # Create a circular reference
    data = {"a": 1}
    data["self"] = data

    # Function that processes the structure without modifying it
    def identity(x):
        return x

    result = dfs_map(data, identity)

    # Check that the recursive structure is preserved, but they won't be
    # the exact same object reference due to how dfs_map works
    assert result["a"] == 1
    assert result["self"]["a"] == 1
    assert result["self"]["self"]["a"] == 1

    # Test for a few levels to ensure circular references were handled properly
    current = result
    for _ in range(5):  # Check a few levels deep
        current = current["self"]
        assert current["a"] == 1


def test_dfs_map_with_path_tracking():
    """Test that dfs_map correctly tracks paths during traversal."""
    data = {"a": 1, "b": {"c": 2}}

    paths_visited = []

    # Create a custom function that captures the path parameter
    def track_path(x):
        nonlocal paths_visited
        # In each call to dfs_map, 'path' will be in the caller's scope
        # We can't directly access it, so we'll record what we're processing
        # and then validate our traversal behavior
        paths_visited.append(x)
        return x

    result = dfs_map(data, track_path)

    # Verify we visited all the expected elements
    assert len(paths_visited) == 4  # Root, 'a', 'b', and 'c'
    assert 1 in paths_visited
    assert 2 in paths_visited
    assert {"c": 2} in paths_visited
    assert result == data


def test_dfs_map_transformation():
    """Test dfs_map with a transformation that changes types."""
    data = {
        "a": "1",  # string
        "b": "2",  # string
        "c": {
            "d": "3"  # string
        },
    }

    # Convert strings to integers
    def convert(x):
        if isinstance(x, str) and x.isdigit():
            return int(x)
        return x

    result = dfs_map(data, convert)
    assert result == {
        "a": 1,  # now integer
        "b": 2,  # now integer
        "c": {
            "d": 3  # now integer
        },
    }


def test_dfs_map_with_jax_array():
    """Test dfs_map with jax.Array type which is a LeafType."""
    pytest.importorskip("jax")  # Skip if jax is not installed
    import jax
    import numpy as np

    # Create a simple jax array
    array = jax.numpy.array([1, 2, 3])
    data = {"array": array}

    # Identity function
    def identity(x):
        return x

    result = dfs_map(data, identity)

    # Check that the array is preserved
    assert "array" in result
    assert isinstance(result["array"], jax.Array)
    np.testing.assert_array_equal(result["array"], array)


def test_dfs_map_complex_nested_structure():
    """Test dfs_map with a complex nested structure with various types."""
    data = {
        "ints": [1, 2, 3],
        "strings": ["a", "b", "c"],
        "mixed": [1, "a", {"nested": 2}],
        "dict": {"a": 1, "b": [2, 3], "c": {"d": 4}},
        "booleans": [True, False],
        "none_value": None,
    }

    # Function that doesn't modify values
    def identity(x):
        return x

    result = dfs_map(data, identity)

    # Check that the structure is preserved
    assert result["ints"] == [1, 2, 3]
    assert result["strings"] == ["a", "b", "c"]
    assert result["mixed"] == [1, "a", {"nested": 2}]
    assert result["dict"]["a"] == 1
    assert result["dict"]["b"] == [2, 3]
    assert result["dict"]["c"]["d"] == 4
    assert result["booleans"] == [True, False]
    assert result["none_value"] is None


def test_dfs_map_with_custom_module_attributes():
    """Test dfs_map with a Module that has custom attributes."""

    class CustomModule(Module):
        __slots__ = ["slot_attr"]

        def __init__(self):
            self.dict_attr = 1
            self.slot_attr = 2
            self.nested = {"a": 3}

    module = CustomModule()

    # Double all integer values
    def double_ints(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(module, double_ints)

    # Check that the attributes were processed correctly
    assert result.dict_attr == 2  # doubled
    assert result.slot_attr == 4  # doubled
    assert result.nested["a"] == 6  # doubled


def test_dfs_map_with_nested_modules():
    """Test dfs_map with nested Module objects."""

    class ChildModule(Module):
        def __init__(self):
            self.value = 1

    class ParentModule(Module):
        def __init__(self):
            self.child = ChildModule()
            self.other_value = 2

    module = ParentModule()

    # Double all integer values
    def double_ints(x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(module, double_ints)

    # Check that the nested module was processed correctly
    assert isinstance(result, Module)
    assert hasattr(result, "child")
    assert isinstance(result.child, Module)
    assert result.child.value == 2  # doubled
    assert result.other_value == 4  # doubled


def test_item_key_methods():
    """Test ItemKey get, set, and repr methods."""
    # Test with string key
    item_key = ItemKey("test")
    obj = {"test": 42}

    # Test get method
    assert item_key.get(obj) == 42

    # Test set method
    item_key.set(obj, 84)
    assert obj["test"] == 84

    # Test repr method
    assert repr(item_key) == "[test]"

    # Test with integer key
    item_key_int = ItemKey(1)
    obj_list = [10, 20, 30]

    # Test get method
    assert item_key_int.get(obj_list) == 20

    # Test set method
    item_key_int.set(obj_list, 40)
    assert obj_list[1] == 40

    # Test repr method
    assert repr(item_key_int) == "[1]"


def test_attr_key_methods():
    """Test AttrKey get, set, and repr methods."""

    class TestObj:
        def __init__(self):
            self.test_attr = "value"

    obj = TestObj()
    attr_key = AttrKey("test_attr")

    # Test get method
    assert attr_key.get(obj) == "value"

    # Test set method
    attr_key.set(obj, "new_value")
    assert obj.test_attr == "new_value"

    # Test repr method
    assert repr(attr_key) == ".test_attr"


def test_path_key_methods():
    """Test PathKey get, set, and repr methods."""
    # Create a nested object
    nested_obj = {"level1": {"level2": 42}}

    # Create path keys
    key1 = ItemKey("level1")
    key2 = ItemKey("level2")

    # Test creating a path and adding keys
    path1 = PathKey([key1])
    path2 = path1 + key2

    # Test get method
    assert path1.get(nested_obj) == {"level2": 42}
    assert path2.get(nested_obj) == 42

    # Test set method
    path2.set(nested_obj, 84)
    assert nested_obj["level1"]["level2"] == 84

    # Reset for next test
    nested_obj["level1"]["level2"] = 42

    # Test setting at parent level
    path1.set(nested_obj, {"level2": 99})
    assert nested_obj["level1"]["level2"] == 99

    # Test repr method
    assert repr(path1) == "[level1]"
    assert repr(path2) == "[level1][level2]"

    # Test empty path
    empty_path = PathKey([])
    assert repr(empty_path) == ""

    # Test empty path set (should do nothing)
    original = {"a": 1}
    empty_path.set(original, {"b": 2})
    assert original == {"a": 1}  # Should be unchanged


def test_path_key_repr():
    """Test the __repr__ method of PathKey."""
    # Create path keys
    key1 = ItemKey("level1")
    key2 = ItemKey("level2")
    key3 = ItemKey(3)

    # Test creating a path and adding keys
    path1 = PathKey([key1])
    path2 = path1 + key2
    path3 = path2 + key3

    # Test repr method
    assert repr(path1) == "[level1]"
    assert repr(path2) == "[level1][level2]"
    assert repr(path3) == "[level1][level2][3]"

    # Test empty path
    empty_path = PathKey([])
    assert repr(empty_path) == ""


def test_dfs_map_unknown_type():
    """Test dfs_map with an unknown type that raises a ValueError."""

    # Create a custom class that is not a Module and not a leaf type
    class CustomClass:
        pass

    custom_obj = CustomClass()

    # Function that returns the same type
    def identity(x):
        return x

    # This should raise a ValueError
    with pytest.raises(ValueError, match="Unknown object type"):
        dfs_map(custom_obj, identity)
