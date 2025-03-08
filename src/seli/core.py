"""Core functionality for the seli package."""


def example_function(value: str = "Hello, World!") -> str:
    """Return a greeting message.

    This is an example function that demonstrates how to use docstrings
    and type hints in Python functions.

    Args:
        value: The greeting message to return. Defaults to "Hello, World!".

    Returns:
        The greeting message.

    Examples:
        >>> example_function()
        'Hello, World!'
        >>> example_function("Hello, Seli!")
        'Hello, Seli!'
    """
    return value 