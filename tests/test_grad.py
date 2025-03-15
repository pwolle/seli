import jax
import jax.numpy as jnp

from seli.core._module import Module
from seli.net._init import Zeros
from seli.net._key import set_rngs
from seli.net._param import Param
from seli.opt._grad import get_arrays, grad, set_arrays


class SimpleModule(Module, name="test_grad.SimpleModule"):
    def __init__(self):
        self.param1 = Param(init=Zeros(), collection="param")
        self.param2 = Param(init=Zeros(), collection="param")
        self.param3 = Param(init=Zeros(), collection="other")

        # Initialize params using set_rngs and capture the returned module
        key = jax.random.PRNGKey(0)
        module = set_rngs(self, key)

        # Use the returned module's parameters
        self.param1 = module.param1
        self.param2 = module.param2
        self.param3 = module.param3

        # Call to initialize with shapes
        self.param1((2, 3))
        self.param2((3, 1))
        self.param3((2, 2))

        # Replace with specific values for testing
        self.param1.value = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.param2.value = jnp.array([[0.1], [0.2], [0.3]])
        self.param3.value = jnp.array([[0.5, 0.5], [0.5, 0.5]])


def test_get_arrays():
    module = SimpleModule()

    # Get all arrays
    module_copy, arrays = get_arrays(module)

    # Check that arrays were extracted
    assert len(arrays) == 3

    # Check that values are None in the module_copy
    assert module_copy.param1.value is None
    assert module_copy.param2.value is None
    assert module_copy.param3.value is None

    # Check arrays dictionary contains the correct values
    for path_str, array in arrays.items():
        if "param1" in path_str:
            assert array.shape == (2, 3)
            assert jnp.array_equal(array, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        elif "param2" in path_str:
            assert array.shape == (3, 1)
            assert jnp.array_equal(array, jnp.array([[0.1], [0.2], [0.3]]))
        elif "param3" in path_str:
            assert array.shape == (2, 2)
            assert jnp.array_equal(array, jnp.array([[0.5, 0.5], [0.5, 0.5]]))

    # Test with specific collection
    module = SimpleModule()
    module_copy, arrays = get_arrays(module, collection="param")

    # Should only have param1 and param2
    assert len(arrays) == 2

    # param3 should not be extracted as it's in a different collection
    for path_str in arrays.keys():
        assert "param3" not in path_str


def test_set_arrays():
    module = SimpleModule()

    # Get arrays
    _, arrays = get_arrays(module)

    # Create a new module with null values
    new_module = SimpleModule()
    new_module.param1.value = None
    new_module.param2.value = None
    new_module.param3.value = None

    # Set arrays back
    updated_module = set_arrays(new_module, arrays)

    # Check that arrays were set correctly
    assert jnp.array_equal(
        updated_module.param1.value, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    assert jnp.array_equal(
        updated_module.param2.value, jnp.array([[0.1], [0.2], [0.3]])
    )
    assert jnp.array_equal(
        updated_module.param3.value, jnp.array([[0.5, 0.5], [0.5, 0.5]])
    )


def test_grad_simple_function():
    module = SimpleModule()

    def simple_loss(module):
        # Simple loss function: sum of all parameter values
        loss = (
            jnp.sum(module.param1.value)
            + jnp.sum(module.param2.value)
            + jnp.sum(module.param3.value)
        )
        return loss

    # Use the grad function from _grad.py
    grad_fn = grad(simple_loss)
    gradients = grad_fn(module)

    # Check that we got gradients for all parameters
    assert len(gradients) == 3

    # For this loss function, all gradients should be 1.0
    for path_str, gradient in gradients.items():
        if "param1" in path_str:
            assert jnp.allclose(gradient, jnp.ones((2, 3)))
        elif "param2" in path_str:
            assert jnp.allclose(gradient, jnp.ones((3, 1)))
        elif "param3" in path_str:
            assert jnp.allclose(gradient, jnp.ones((2, 2)))


def test_grad_with_aux():
    module = SimpleModule()

    def loss_with_aux(module):
        # Loss function with auxiliary outputs
        loss = jnp.sum(module.param1.value) + jnp.sum(module.param2.value)
        aux_data = {"param1_shape": module.param1.value.shape}
        return loss, aux_data

    # Use the grad function with has_aux=True
    grad_fn = grad(loss_with_aux, has_aux=True)
    gradients, aux = grad_fn(module)

    # Check gradients
    assert len(gradients) == 3

    # Check that aux data is returned correctly
    assert "param1_shape" in aux
    assert aux["param1_shape"] == (2, 3)

    # Check gradients values
    for path_str, gradient in gradients.items():
        if "param1" in path_str:
            assert jnp.allclose(gradient, jnp.ones((2, 3)))
        elif "param2" in path_str:
            assert jnp.allclose(gradient, jnp.ones((3, 1)))
        elif "param3" in path_str:
            # param3 is not used in the loss, so gradients should be zeros
            assert jnp.allclose(gradient, jnp.zeros((2, 2)))


def test_grad_complex_function():
    module = SimpleModule()

    # Calculate expected gradients manually for verification before calling grad_fn
    # For this operation:
    # d(loss)/d(param1) = 2 * (param1 @ param2) @ param2.T
    # d(loss)/d(param2) = 2 * param1.T @ (param1 @ param2)
    result = module.param1.value @ module.param2.value
    expected_grad_param1 = 2 * result @ module.param2.value.T
    expected_grad_param2 = 2 * module.param1.value.T @ result

    def complex_loss(module):
        # Matrix multiplication of param1 @ param2 and square the result
        result = module.param1.value @ module.param2.value
        loss = jnp.sum(result**2)
        return loss

    grad_fn = grad(complex_loss)
    gradients = grad_fn(module)

    for path_str, gradient in gradients.items():
        if "param1" in path_str:
            assert jnp.allclose(gradient, expected_grad_param1)
        elif "param2" in path_str:
            assert jnp.allclose(gradient, expected_grad_param2)
        elif "param3" in path_str:
            # param3 is not used in the loss, so gradients should be zeros
            assert jnp.allclose(gradient, jnp.zeros((2, 2)))
