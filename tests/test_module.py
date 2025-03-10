import pytest

from seli.core._module import (
    AttrKey,
    ItemKey,
    Module,
    PathKey,
    _keys_lt,
    dfs_map,
    to_tree,
    to_tree_inverse,
)


def test_dfs_map_simple_dict():
    data = {"a": 1, "b": 2, "c": 3}

    # Simple function that doubles values
    def double(_, x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == {"a": 2, "b": 4, "c": 6}


def test_dfs_map_nested_dict():
    data = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    # Simple function that doubles values
    def double(_, x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == {"a": 2, "b": {"c": 4, "d": 6}, "e": 8}


def test_dfs_map_list():
    data = [1, 2, 3, 4]

    # Simple function that doubles values
    def double(_, x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == [2, 4, 6, 8]


def test_dfs_map_nested_list():
    data = [1, [2, 3], 4]

    # Simple function that doubles values
    def double(_, x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == [2, [4, 6], 8]


def test_dfs_map_mixed_structure():
    data = {"a": 1, "b": [2, 3, {"c": 4}], "d": 5}

    # Simple function that doubles values
    def double(_, x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(data, double)
    assert result == {"a": 2, "b": [4, 6, {"c": 8}], "d": 10}


def test_dfs_map_with_module():
    class TestModule(Module, name="test_module.TestModule"):
        def __init__(self):
            self.value = 1

    module = TestModule()

    # A function that preserves the module but accesses its attributes
    def process_module(_, x):
        return x

    result = dfs_map(module, process_module)

    # The function always returns a new instance of the base Module class,
    # not the original class type
    assert isinstance(result, Module)
    assert hasattr(result, "value")
    assert result.value == 1


def test_dfs_map_circular_reference():
    # Create a circular reference
    data = {"a": 1}
    data["self"] = data

    # Function that processes the structure without modifying it
    def identity(_, x):
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
    data = {"a": 1, "b": {"c": 2}}

    paths_visited = []

    # Create a custom function that captures the path parameter
    def track_path(_, x):
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
    data = {
        "a": "1",  # string
        "b": "2",  # string
        "c": {
            "d": "3"  # string
        },
    }

    # Convert strings to integers
    def convert(_, x):
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
    pytest.importorskip("jax")  # Skip if jax is not installed
    import jax
    import numpy as np

    # Create a simple jax array
    array = jax.numpy.array([1, 2, 3])
    data = {"array": array}

    # Identity function
    def identity(_, x):
        return x

    result = dfs_map(data, identity)

    # Check that the array is preserved
    assert "array" in result
    assert isinstance(result["array"], jax.Array)
    np.testing.assert_array_equal(result["array"], array)


def test_dfs_map_complex_nested_structure():
    data = {
        "ints": [1, 2, 3],
        "strings": ["a", "b", "c"],
        "mixed": [1, "a", {"nested": 2}],
        "dict": {"a": 1, "b": [2, 3], "c": {"d": 4}},
        "booleans": [True, False],
        "none_value": None,
    }

    # Function that doesn't modify values
    def identity(_, x):
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
    class CustomModule(Module, name="test_module.CustomModule"):
        __slots__ = ["slot_attr"]

        def __init__(self):
            self.dict_attr = 1
            self.slot_attr = 2
            self.nested = {"a": 3}

    module = CustomModule()

    # Double all integer values
    def double_ints(_, x):
        if isinstance(x, int):
            return x * 2
        return x

    result = dfs_map(module, double_ints)

    # Check that the attributes were processed correctly
    assert result.dict_attr == 2  # doubled
    assert result.slot_attr == 4  # doubled
    assert result.nested["a"] == 6  # doubled


def test_dfs_map_with_nested_modules():
    class ChildModule(Module, name="test_module.ChildModule"):
        def __init__(self):
            self.value = 1

    class ParentModule(Module, name="test_module.ParentModule"):
        def __init__(self):
            self.child = ChildModule()
            self.other_value = 2

    module = ParentModule()

    # Double all integer values
    def double_ints(_, x):
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
    # Test with string key
    item_key = ItemKey("test")
    obj = {"test": 42}

    # Test get method
    assert item_key.get(obj) == 42

    # Test set method
    item_key.set(obj, 84)
    assert obj["test"] == 84

    # Test repr method
    assert repr(item_key) == "['test']"

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
    assert repr(path1) == "['level1']"
    assert repr(path2) == "['level1']['level2']"

    # Test empty path
    empty_path = PathKey([])
    assert repr(empty_path) == ""

    # Test empty path set (should do nothing)
    original = {"a": 1}
    empty_path.set(original, {"b": 2})
    assert original == {"a": 1}  # Should be unchanged


def test_path_key_repr():
    # Create path keys
    key1 = ItemKey("level1")
    key2 = ItemKey("level2")
    key3 = ItemKey(3)

    # Test creating a path and adding keys
    path1 = PathKey([key1])
    path2 = path1 + key2
    path3 = path2 + key3

    # Test repr method
    assert repr(path1) == "['level1']"
    assert repr(path2) == "['level1']['level2']"
    assert repr(path3) == "['level1']['level2'][3]"

    # Test empty path
    empty_path = PathKey([])
    assert repr(empty_path) == ""


def test_dfs_map_unknown_type():
    # Create a custom class that is not a Module and not a leaf type
    class CustomClass:
        pass

    custom_obj = CustomClass()

    # Function that returns the same type
    def identity(_, x):
        return x

    # This should raise a ValueError
    with pytest.raises(ValueError, match="Unknown object type"):
        dfs_map(custom_obj, identity)


def test_dfs_map_path_tracking_dict():
    data = {"a": 1, "b": {"c": 2, "d": 3}}

    paths_with_values = []

    def collect_paths(path, x):
        paths_with_values.append((str(path), x))
        return x

    dfs_map(data, collect_paths)

    # Check that paths are correctly built and passed
    assert ("", data) in paths_with_values
    assert ("['a']", 1) in paths_with_values
    assert ("['b']", {"c": 2, "d": 3}) in paths_with_values
    assert ("['b']['c']", 2) in paths_with_values
    assert ("['b']['d']", 3) in paths_with_values


def test_dfs_map_path_tracking_list():
    data = [1, [2, 3], 4]

    paths_with_values = []

    def collect_paths(path, x):
        paths_with_values.append((str(path), x))
        return x

    dfs_map(data, collect_paths)

    # Check that paths are correctly built and passed
    assert ("", data) in paths_with_values
    assert ("[0]", 1) in paths_with_values
    assert ("[1]", [2, 3]) in paths_with_values
    assert ("[1][0]", 2) in paths_with_values
    assert ("[1][1]", 3) in paths_with_values
    assert ("[2]", 4) in paths_with_values


def test_dfs_map_path_tracking_mixed():
    data = {"a": 1, "b": [2, {"c": 3}], "d": {"e": [4, 5]}}

    paths_with_values = []

    def collect_paths(path, x):
        paths_with_values.append((str(path), x))
        return x

    dfs_map(data, collect_paths)

    # Check that paths are correctly built and passed
    assert ("", data) in paths_with_values
    assert ("['a']", 1) in paths_with_values
    assert ("['b']", [2, {"c": 3}]) in paths_with_values
    assert ("['b'][0]", 2) in paths_with_values
    assert ("['b'][1]", {"c": 3}) in paths_with_values
    assert ("['b'][1]['c']", 3) in paths_with_values
    assert ("['d']", {"e": [4, 5]}) in paths_with_values
    assert ("['d']['e']", [4, 5]) in paths_with_values
    assert ("['d']['e'][0]", 4) in paths_with_values
    assert ("['d']['e'][1]", 5) in paths_with_values


def test_dfs_map_using_path_for_transformation():
    data = {"a": 1, "b": {"c": 2, "d": 3}}

    def transform_based_on_path(path, x):
        # Double values at path [b][c]
        if str(path) == "['b']['c']" and isinstance(x, int):
            return x * 2
        # Triple values at path [a]
        elif str(path) == "['a']" and isinstance(x, int):
            return x * 3
        return x

    result = dfs_map(data, transform_based_on_path)

    assert result["a"] == 3  # Tripled
    assert result["b"]["c"] == 4  # Doubled
    assert result["b"]["d"] == 3  # Unchanged


def test_dfs_map_initial_path():
    data = {"a": 1, "b": 2}

    paths_with_values = []
    initial_path = PathKey([ItemKey("root"), ItemKey("data")])

    def collect_paths(path, x):
        paths_with_values.append((str(path), x))
        return x

    dfs_map(data, collect_paths, path=initial_path)

    # Paths should include the initial path
    assert ("['root']['data']", data) in paths_with_values
    assert ("['root']['data']['a']", 1) in paths_with_values
    assert ("['root']['data']['b']", 2) in paths_with_values


def test_dfs_map_path_consistency():
    data = {"a": {"b": {"c": 1}}}

    path_sequence = []

    def collect_path_sequence(path, x):
        path_sequence.append(str(path))
        return x

    dfs_map(data, collect_path_sequence)

    # Check the paths are built up correctly in sequence
    assert path_sequence[0] == ""  # Root
    assert path_sequence[1] == "['a']"
    assert path_sequence[2] == "['a']['b']"
    assert path_sequence[3] == "['a']['b']['c']"


def test_dfs_map_module_path():
    class TestModule(Module, name="test_module.TestModule"):
        def __init__(self):
            self.value = 1
            self.nested = {"a": 2}

    module = TestModule()

    paths_with_values = []

    def collect_paths(path, x):
        if not isinstance(
            x, Module
        ):  # We're not interested in collecting the module itself
            paths_with_values.append((str(path), x))
        return x

    dfs_map(module, collect_paths)

    # Check paths for module attributes
    assert (".value", 1) in paths_with_values
    assert (".nested", {"a": 2}) in paths_with_values
    assert (".nested['a']", 2) in paths_with_values


def test_dfs_map_refs_fun_circular_reference():
    # Create a circular reference structure
    a = {"name": "a"}
    b = {"name": "b", "ref": a}
    a["ref"] = b  # Create circular reference

    # Keep track of how many times each object is processed
    process_count = {}

    def count_refs(path, x):
        obj_id = id(x)
        process_count[obj_id] = process_count.get(obj_id, 0) + 1
        return x

    # Custom function for handling reference objects
    def refs_handler(path, x):
        # Replace circular references with a marker
        return {"circular_ref": True, "name": x.get("name", "unknown")}

    # Process the structure with refs_fun
    result = dfs_map(a, count_refs, refs_fun=refs_handler)

    # When we find b in a["ref"], it should be replaced with the marker
    # And when we find a in b["ref"], it should also be replaced with the marker
    assert result["ref"]["ref"]["circular_ref"] is True

    # Original objects should only be processed once
    assert process_count[id(a)] == 1
    assert process_count[id(b)] == 1


def test_dfs_map_refs_fun_transformation():
    # Create a structure with duplicate references
    inner = {"value": 10}
    data = {
        "a": inner,
        "b": inner,  # Same reference as a
        "c": {"value": 10},  # Same value but different reference
    }

    seen_objects = set()
    paths_seen = []

    def track_objects(path, x):
        paths_seen.append(str(path))
        seen_objects.add(id(x))
        return x

    # Refs function that transforms repeated references
    def transform_refs(path, x):
        # Only transform dictionary objects
        if isinstance(x, dict) and "value" in x:
            return {"transformed": True, "original_value": x["value"]}
        return x

    result = dfs_map(data, track_objects, refs_fun=transform_refs)

    # Check for the transformed value in b
    assert "value" in result["a"]
    assert "transformed" in result["b"]

    # The object with same value but different reference should be unchanged
    assert "transformed" not in result["c"]


def test_dfs_map_refs_fun_with_modules():
    # Create a module with a reference to itself
    class SelfReferencingModule(Module, name="test_module.SelfReferencingModule"):
        def __init__(self):
            self.value = 5
            self.self_ref = None

    module = SelfReferencingModule()
    module.self_ref = module  # Create self-reference

    paths_processed = []

    def collect_path(path, x):
        if not isinstance(x, Module):
            paths_processed.append(str(path))
        return x

    # Custom function to handle references to avoid infinite recursion
    def handle_module_refs(path, x):
        if isinstance(x, Module):
            return {"module_ref": True, "path": str(path)}
        return x

    result = dfs_map(module, collect_path, refs_fun=handle_module_refs)

    # Check that we have the original module with properly handled self-reference
    assert isinstance(result, Module)
    assert isinstance(result.self_ref, dict)
    assert result.self_ref["module_ref"] is True
    assert "path" in result.self_ref

    # We should have only processed .value once
    assert paths_processed.count(".value") == 1


def test_dfs_map_refs_fun_none():
    # Test behavior when refs_fun is None

    # Create structure with duplicate references
    shared = {"data": 123}
    obj = {"a": shared, "b": shared}

    def identity(_, x):
        return x

    # Let's examine the implementation logic of dfs_map
    # Even with refs_fun=None, new objects are created during traversal
    result = dfs_map(obj, identity)  # refs_fun defaults to None

    # Test that the values are the same even if the objects are different
    assert result["a"]["data"] == 123
    assert result["b"]["data"] == 123

    # Modify one reference and check that it doesn't affect the other
    # since they are separate objects in the result
    result["a"]["data"] = 456
    assert result["b"]["data"] == 123


def test_keys_lt_different_types():
    item_key = ItemKey(key="test")
    attr_key = AttrKey(key="test")

    # According to the implementation, an ItemKey should be less than an AttrKey
    # Note: Current implementation has an error in this case
    result = _keys_lt(item_key, attr_key)
    assert result is True
    # The reverse should be False
    result = _keys_lt(attr_key, item_key)
    assert result is False


def test_keys_lt_same_type_different_key_types():
    int_item_key = ItemKey(key=1)
    str_item_key = ItemKey(key="test")

    # According to implementation, int key should be less than str key
    # Note: Current implementation has an error in this case
    result = _keys_lt(int_item_key, str_item_key)
    assert result is True

    # The reverse should be False
    result = _keys_lt(str_item_key, int_item_key)
    assert result is False


def test_keys_lt_same_type_same_key_type():
    smaller_item_key = ItemKey(key=1)
    larger_item_key = ItemKey(key=5)

    result = _keys_lt(smaller_item_key, larger_item_key)
    assert result is True

    result = _keys_lt(larger_item_key, smaller_item_key)
    assert result is False

    # Test with AttrKey instances
    smaller_attr_key = AttrKey(key="apple")
    larger_attr_key = AttrKey(key="banana")

    result = _keys_lt(smaller_attr_key, larger_attr_key)
    assert result is True

    result = _keys_lt(larger_attr_key, smaller_attr_key)
    assert result is False


def test_keys_lt_equal_keys():
    item_key1 = ItemKey(key="same")
    item_key2 = ItemKey(key="same")

    result = _keys_lt(item_key1, item_key2)
    assert result is False

    # Test with AttrKey instances
    attr_key1 = AttrKey(key="same")
    attr_key2 = AttrKey(key="same")

    result = _keys_lt(attr_key1, attr_key2)
    assert result is False


def test_to_tree_simple_dict():
    """Test to_tree with a simple dictionary."""
    data = {"a": 1, "b": 2, "c": 3}
    result = to_tree(data)
    # For simple structure without shared references or cycles,
    # the result should be structurally equivalent to the input
    assert result == data


def test_to_tree_shared_reference():
    """Test to_tree with a structure containing shared references."""
    shared_dict = {"name": "shared"}
    data = {
        "a": shared_dict,
        "b": shared_dict,  # Same object referenced twice
    }

    # Before transformation, changing the shared object affects both references
    shared_dict["name"] = "modified"
    assert data["a"]["name"] == "modified"
    assert data["b"]["name"] == "modified"

    result = to_tree(data)

    # The result should be a PathKey reference instead of a circular reference
    result["a"]["name"] = "changed again"
    assert result["a"]["name"] == "changed again"
    assert result["b"] == PathKey([ItemKey("a")])


def test_to_tree_with_nested_shared_reference():
    shared_dict = {"value": 42}
    nested = {
        "x": shared_dict,
        "y": shared_dict,
    }
    data = {
        "outer": {
            "inner1": nested,
            "inner2": nested,  # Same nested dict referenced multiple times
        }
    }

    # Before transformation, changing the shared object affects all references
    shared_dict["value"] = 100
    assert data["outer"]["inner1"]["x"]["value"] == 100
    assert data["outer"]["inner1"]["y"]["value"] == 100
    assert data["outer"]["inner2"]["x"]["value"] == 100
    assert data["outer"]["inner2"]["y"]["value"] == 100

    result = to_tree(data)

    # After transformation, the shared references should be replaced with PathKeys
    assert result["outer"]["inner1"]["x"]["value"] == 100
    assert isinstance(result["outer"]["inner1"]["y"], PathKey)
    assert isinstance(result["outer"]["inner2"], PathKey)

    # Check that path keys point to the correct location
    assert result["outer"]["inner1"]["y"] == PathKey(
        [ItemKey("outer"), ItemKey("inner1"), ItemKey("x")]
    )
    assert result["outer"]["inner2"] == PathKey([ItemKey("outer"), ItemKey("inner1")])


def test_to_tree_with_cyclic_reference():
    # Create a structure with a cycle
    cycle_dict = {"name": "cycle"}
    cycle_dict["self_ref"] = cycle_dict  # Self-reference creates a cycle

    # Before transformation, we have an actual cycle
    assert cycle_dict["self_ref"] is cycle_dict

    result = to_tree(cycle_dict)

    # After transformation, the cycle should be replaced with a PathKey
    assert result["name"] == "cycle"
    assert isinstance(result["self_ref"], PathKey)
    assert result["self_ref"] == PathKey([])  # Root path

    # Changing a value in the result should not cause infinite recursion
    result["name"] = "transformed"
    assert result["name"] == "transformed"


def test_to_tree_with_complex_structure():
    # Create a structure with multiple shared and cyclic references
    shared_list = [1, 2, 3]
    a = {"name": "a", "list": shared_list}
    b = {"name": "b", "list": shared_list, "ref_to_a": a}
    a["ref_to_b"] = b  # Create a cycle between a and b

    data = {"a": a, "b": b, "shared_list": shared_list}

    result = to_tree(data)

    # Check that the basic values are preserved
    assert result["a"]["name"] == "a"
    # b is likely replaced by a PathKey in this scenario
    assert isinstance(result["b"], PathKey)
    assert result["a"]["list"] == [1, 2, 3]

    # Check that shared list is properly represented
    assert isinstance(result["shared_list"], PathKey)


def test_to_tree_inverse_simple_dict():
    """Test to_tree_inverse with a simple dictionary."""
    data = {"a": 1, "b": 2, "c": 3}
    # For simple structures, to_tree should not change anything
    tree = to_tree(data)
    assert tree == data

    # And to_tree_inverse should return the same structure
    result = to_tree_inverse(tree)
    assert result == data


def test_to_tree_inverse_shared_reference():
    """Test to_tree_inverse with a structure containing shared references."""
    # Create a structure with shared references
    shared_dict = {"name": "shared"}
    data = {
        "a": shared_dict,
        "b": shared_dict,  # Same object referenced twice
    }

    # Transform using to_tree
    tree = to_tree(data)

    # Verify that to_tree has replaced the shared reference with a PathKey
    assert isinstance(tree["b"], PathKey)
    assert tree["b"] == PathKey([ItemKey("a")])

    # Now inverse the transformation
    result = to_tree_inverse(tree)

    # Verify the structure is correct
    assert result["a"]["name"] == "shared"
    assert result["b"]["name"] == "shared"

    # Verify that shared references are restored by modifying one reference
    # and checking that the other is also modified
    result["a"]["name"] = "modified"
    assert result["b"]["name"] == "modified"


def test_to_tree_inverse_nested_shared_reference():
    # Create a structure with nested shared references
    shared_dict = {"value": 42}
    nested = {
        "x": shared_dict,
        "y": shared_dict,
    }
    data = {
        "outer": {
            "inner1": nested,
            "inner2": nested,  # Same nested dict referenced multiple times
        }
    }

    # Transform using to_tree
    tree = to_tree(data)

    # Now inverse the transformation
    result = to_tree_inverse(tree)

    # Verify that the structure is correct
    assert result["outer"]["inner1"]["x"]["value"] == 42
    assert result["outer"]["inner1"]["y"]["value"] == 42
    assert result["outer"]["inner2"]["x"]["value"] == 42
    assert result["outer"]["inner2"]["y"]["value"] == 42

    # Verify shared references are restored by modifying and checking
    result["outer"]["inner1"]["x"]["value"] = 100
    assert result["outer"]["inner1"]["y"]["value"] == 100
    assert result["outer"]["inner2"]["x"]["value"] == 100
    assert result["outer"]["inner2"]["y"]["value"] == 100


def test_to_tree_inverse_cyclic_reference():
    # Create a structure with a cycle
    cycle_dict = {"name": "cycle"}
    cycle_dict["self_ref"] = cycle_dict  # Self-reference creates a cycle

    # Transform using to_tree
    tree = to_tree(cycle_dict)

    # Verify that to_tree has replaced the cycle with a PathKey
    assert isinstance(tree["self_ref"], PathKey)

    # Now inverse the transformation
    result = to_tree_inverse(tree)

    # Verify the structure is correct
    assert result["name"] == "cycle"
    assert "self_ref" in result

    # Modifying through the cycle
    result["name"] = "modified"
    assert result["self_ref"]["name"] == "modified"


def test_to_tree_inverse_complex_structure():
    # Create a structure with multiple shared and cyclic references
    shared_list = [1, 2, 3]
    a = {"name": "a", "list": shared_list}
    b = {"name": "b", "list": shared_list, "ref_to_a": a}
    a["ref_to_b"] = b  # Create a cycle between a and b

    data = {"a": a, "b": b, "shared_list": shared_list}

    # Transform using to_tree
    tree = to_tree(data)

    # Now inverse the transformation
    result = to_tree_inverse(tree)

    # Verify that basic structure is preserved
    assert result["a"]["name"] == "a"
    assert result["b"]["name"] == "b"

    # Verify that shared references work correctly
    result["shared_list"][0] = 99
    assert result["a"]["list"][0] == 99
    assert result["b"]["list"][0] == 99

    # Verify that cyclic references work correctly
    result["a"]["name"] = "modified a"
    assert result["b"]["ref_to_a"]["name"] == "modified a"
