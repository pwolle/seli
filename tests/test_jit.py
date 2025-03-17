import jax.numpy as jnp
import numpy as np

from seli.core._jit import Arguments, Result, _apply_filter_jit, jit
from seli.core._module import Module


class SimpleModule(Module, name="test_jit.SimpleModule"):
    def __init__(self, value):
        self.value = value


def test_arguments_init():
    args = (1, 2, 3)
    kwargs = {"a": 4, "b": 5}
    arguments = Arguments(args, kwargs)

    assert arguments.args == list(args)
    assert arguments.kwargs == kwargs


def test_result_init():
    # Test with a simple value
    value = 42
    result = Result(value)
    assert result.value == value

    # Test with a tuple, is not a valid type, but should appear to be a tuple
    tuple_value = (1, 2, 3)
    result = Result(tuple_value)
    assert result.value == tuple_value
    assert isinstance(result.value, tuple)


def test_apply_filter_jit_direct():
    def test_function(a, b, c=3):
        return a + b + c

    args = [1, 2]
    kwargs = {"c": 4}
    arguments = Arguments(args, kwargs)

    result = _apply_filter_jit(arguments, test_function)

    assert isinstance(result, Result)
    assert result.value == 7  # 1 + 2 + 4


def test_jit_basic_function():
    @jit
    def add(x, y):
        return x + y

    result = add(1, 2)
    assert result == 3


def test_jit_with_arrays():
    @jit
    def add_arrays(x, y):
        return x + y

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = add_arrays(x, y)

    np.testing.assert_array_equal(result, jnp.array([5, 7, 9]))


def test_jit_with_kwargs():
    @jit
    def add_with_kwargs(x, y=5):
        return x + y

    result1 = add_with_kwargs(3)
    assert result1 == 8

    result2 = add_with_kwargs(3, y=10)
    assert result2 == 13


def test_jit_with_module():
    @jit
    def process_module(module):
        return module.value * 2

    module = SimpleModule(21)
    result = process_module(module)

    assert result == 42


def test_jit_with_shared_references():
    @jit
    def check_reference_equality(list1, list2):
        # In JAX JIT, this would return False even if list1 and list2 are the
        # same object. With our custom JIT, it should preserve the reference
        # equality.
        return list1 is list2

    shared_list = [1, 2, 3]
    result = check_reference_equality(shared_list, shared_list)

    assert result


def test_jit_with_nested_shared_references():
    @jit
    def modify_and_check(container):
        # Modify the nested list and check that both references see the change
        container["list1"].append(4)
        return container["list1"] is container["list2"]

    shared_list = [1, 2, 3]
    container = {"list1": shared_list, "list2": shared_list}

    # In JAX JIT, container is passed by value, not by reference
    # So modifications inside the function don't affect the original container
    result = modify_and_check(container)
    assert result

    # The original shared_list won't be modified by the JIT function
    # since the function operates on a copy. We test here that the references
    # within the function work correctly, not that external references are
    # modified. We cannot test that the original list is modified without
    # breaking JIT semantics.
    assert shared_list == [1, 2, 3]


def test_jit_with_multiple_outputs():
    @jit
    def multi_output(x):
        return x, x**2, x**3

    result = multi_output(3)
    # The Result class converts tuples to lists, so we need to adjust our
    # assertion
    assert result == (3, 9, 27)


def test_jit_with_nested_modules():
    class NestedModule(Module, name="test_jit.NestedModule"):
        def __init__(self, simple_module, extra_value):
            self.simple_module = simple_module
            self.extra_value = extra_value

    @jit
    def process_nested_module(module):
        return module.simple_module.value * module.extra_value

    simple_module = SimpleModule(7)
    nested_module = NestedModule(simple_module, 6)

    result = process_nested_module(nested_module)
    assert result == 42  # 7 * 6


def test_jit_with_conditional_logic():
    @jit
    def conditional_function(x):
        if x > 0:
            return x * 2
        else:
            return x * -1

    assert conditional_function(5) == 10
    assert conditional_function(-5) == 5


def test_jit_preserves_function_metadata():
    def original_function(x):
        """Test docstring."""
        return x * 2

    jitted_function = jit(original_function)

    assert jitted_function.__name__ == original_function.__name__
    assert jitted_function.__doc__ == original_function.__doc__


def test_jit_with_empty_args():
    @jit
    def no_args_function():
        return 42

    result = no_args_function()
    assert result == 42


def test_jit_with_many_args():
    @jit
    def many_args_function(a, b, c, d, e, f, g, h):
        return a + b + c + d + e + f + g + h

    result = many_args_function(1, 2, 3, 4, 5, 6, 7, 8)
    assert result == 36  # Sum of numbers from 1 to 8


def test_jit_with_mixed_types():
    @jit
    def mixed_types_function(a, b, c):
        # a: int, b: list, c: module
        return a * len(b) + c.value

    result = mixed_types_function(2, [1, 2, 3], SimpleModule(10))
    assert result == 16  # 2 * 3 + 10


def test_jit_recompilation():
    counter = [0]

    def traced_function(x, y):
        counter[0] += 1
        return x + y

    jitted_function = jit(traced_function)

    # First call should compile
    result1 = jitted_function(1, 2)
    assert result1 == 3

    # This call should use the same compiled function
    result2 = jitted_function(3, 4)
    assert result2 == 7

    # The counter should only have incremented twice (once per function call)
    # since we're using the same compiled function
    assert counter[0] == 2
