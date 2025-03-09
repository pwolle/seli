import pytest

from seli.core._module import AttrKey, ItemKey, PathKey


def test_item_key_from_str_with_string_key():
    # Test with string key in single quotes
    item_key = ItemKey.from_str("['test']")
    assert isinstance(item_key, ItemKey)
    assert item_key.key == "test"
    assert repr(item_key) == "['test']"


def test_item_key_from_str_with_integer_key():
    # Test with integer key
    item_key = ItemKey.from_str("[42]")
    assert isinstance(item_key, ItemKey)
    assert item_key.key == 42
    assert repr(item_key) == "[42]"


def test_item_key_from_str_error_cases():
    # Test with invalid format
    with pytest.raises(ValueError, match="Invalid item key string"):
        ItemKey.from_str("invalid")

    with pytest.raises(ValueError, match="Invalid item key string"):
        ItemKey.from_str(".attribute")

    with pytest.raises(ValueError, match="Invalid item key string"):
        ItemKey.from_str("[]")  # Empty brackets

    # Test with non-digit and non-string
    with pytest.raises(ValueError, match="Invalid item key string"):
        ItemKey.from_str("[abc]")  # Not quoted and not a digit


def test_attr_key_from_str_valid_cases():
    # Test with valid attribute name
    attr_key = AttrKey.from_str(".attribute")
    assert isinstance(attr_key, AttrKey)
    assert attr_key.key == "attribute"
    assert repr(attr_key) == ".attribute"

    # Test with underscore in name
    attr_key = AttrKey.from_str(".attr_name0")
    assert isinstance(attr_key, AttrKey)
    assert attr_key.key == "attr_name0"
    assert repr(attr_key) == ".attr_name0"


def test_attr_key_from_str_error_cases():
    # Test with invalid format - no dot
    with pytest.raises(ValueError, match="Invalid attribute key string"):
        AttrKey.from_str("attribute")

    # Test with invalid format - just a dot
    with pytest.raises(ValueError, match="Invalid attribute key string"):
        AttrKey.from_str(".")

    # Test with invalid attribute name
    with pytest.raises(ValueError, match="Invalid attribute key string"):
        AttrKey.from_str(".123")  # Starts with a number

    # Test with invalid characters
    with pytest.raises(ValueError, match="Invalid attribute key string"):
        AttrKey.from_str(".attr-name")  # Contains a hyphen


def test_path_key_from_str_simple_cases():
    # Test with a single attribute key
    path_key = PathKey.from_str(".attribute")
    assert isinstance(path_key, PathKey)
    assert len(path_key.path) == 1
    assert isinstance(path_key.path[0], AttrKey)
    assert path_key.path[0].key == "attribute"
    assert repr(path_key) == ".attribute"

    # Test with a single item key (string)
    path_key = PathKey.from_str("['item']")
    assert isinstance(path_key, PathKey)
    assert len(path_key.path) == 1
    assert isinstance(path_key.path[0], ItemKey)
    assert path_key.path[0].key == "item"
    assert repr(path_key) == "['item']"

    # Test with a single item key (integer)
    path_key = PathKey.from_str("[0]")
    assert isinstance(path_key, PathKey)
    assert len(path_key.path) == 1
    assert isinstance(path_key.path[0], ItemKey)
    assert path_key.path[0].key == 0
    assert repr(path_key) == "[0]"


def test_path_key_from_str_complex_cases():
    # Test with attribute followed by item
    path_key = PathKey.from_str(".attribute['item']")
    assert isinstance(path_key, PathKey)
    assert len(path_key.path) == 2
    assert isinstance(path_key.path[0], AttrKey)
    assert path_key.path[0].key == "attribute"
    assert isinstance(path_key.path[1], ItemKey)
    assert path_key.path[1].key == "item"
    assert repr(path_key) == ".attribute['item']"

    # Test with item followed by attribute
    path_key = PathKey.from_str("['item'].attribute")
    assert isinstance(path_key, PathKey)
    assert len(path_key.path) == 2
    assert isinstance(path_key.path[0], ItemKey)
    assert path_key.path[0].key == "item"
    assert isinstance(path_key.path[1], AttrKey)
    assert path_key.path[1].key == "attribute"
    assert repr(path_key) == "['item'].attribute"

    # Test with a complex path with multiple keys of different types
    path_key = PathKey.from_str(".attribute1['item1'].attribute2")
    assert isinstance(path_key, PathKey)
    assert len(path_key.path) == 3
    assert isinstance(path_key.path[0], AttrKey)
    assert path_key.path[0].key == "attribute1"
    assert isinstance(path_key.path[1], ItemKey)
    assert path_key.path[1].key == "item1"
    assert isinstance(path_key.path[2], AttrKey)
    assert path_key.path[2].key == "attribute2"
    assert repr(path_key) == ".attribute1['item1'].attribute2"


def test_path_key_from_str_error_cases():
    # Test with invalid format
    with pytest.raises(ValueError, match="Invalid path key string"):
        PathKey.from_str("invalid")

    # Test with invalid part in path
    with pytest.raises(ValueError):
        # This will ultimately fail because of the invalid attribute name
        # The exact error message depends on internal implementation details
        PathKey.from_str(".123['item']")

    # Test with invalid item key format
    with pytest.raises(ValueError):
        # This will fail because [abc] is not a valid ItemKey format
        PathKey.from_str(".valid[abc]")
