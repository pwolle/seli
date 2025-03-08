"""Tests for the seli.core module."""

from seli.core import example_function


def test_example_function_default():
    """Test example_function with default argument."""
    result = example_function()
    assert result == "Hello, World!"
    assert isinstance(result, str)


def test_example_function_custom():
    """Test example_function with custom argument."""
    result = example_function("Hello, Seli!")
    assert result == "Hello, Seli!"
    assert isinstance(result, str)
