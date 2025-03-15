import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

from seli.net import Bias, Linear, Scale
from seli.net._key import set_rngs
from seli.net._linear import Affine


class TestLinear:
    def test_linear_layer_initialization(self):
        # Test initialization parameters
        dim = 16
        linear = Linear(dim)

        assert linear.dim == dim
        assert not linear.weight.initialized  # Weight should be lazily initialized
        assert linear.dim_in is None  # Not initialized yet

    def test_linear_layer_forward(self):
        # Test the forward pass with a controlled input
        key = jrn.PRNGKey(0)
        dim = 3
        linear = Linear(dim)
        linear = set_rngs(linear, key)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Get output and ensure the weight is initialized
        output = linear(x)
        assert linear.weight.initialized
        assert linear.weight.value.shape == (2, 3)
        assert output.shape == (2, 3)

        # Test for deterministic behavior with fixed seed
        linear_dup = Linear(dim)
        linear_dup = set_rngs(linear_dup, key)
        output_dup = linear_dup(x)
        np.testing.assert_array_equal(output, output_dup)


class TestBias:
    def test_bias_layer_initialization(self):
        # Test initialization parameters
        bias = Bias()

        assert not bias.bias.initialized  # Bias should be lazily initialized
        assert bias.dim is None  # Not initialized yet

    def test_bias_layer_forward(self):
        # Test the forward pass to verify initialization
        key = jrn.PRNGKey(0)
        bias = Bias()
        bias = set_rngs(bias, key)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Get output and ensure the bias is initialized
        output = bias(x)
        assert bias.bias.initialized
        assert bias.bias.value.shape == (3,)
        assert output.shape == (2, 3)

        # Test for deterministic behavior with fixed seed
        bias_dup = Bias()
        bias_dup = set_rngs(bias_dup, key)
        output_dup = bias_dup(x)
        np.testing.assert_array_equal(output, output_dup)


class TestAffine:
    def test_affine_layer_initialization(self):
        # Test initialization parameters
        dim = 16
        affine = Affine(dim)

        assert isinstance(affine.linear, Linear)
        assert isinstance(affine.bias, Bias)
        assert affine.linear.dim == dim
        assert affine.dim_in is None  # Not initialized yet

    def test_affine_layer_forward(self):
        # Test the forward pass
        key = jrn.PRNGKey(0)
        dim = 3
        affine = Affine(dim)
        affine = set_rngs(affine, key)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Forward pass
        output = affine(x)

        # Check that shapes are correct
        assert output.shape == (2, 3)
        assert affine.linear.weight.initialized
        assert affine.bias.bias.initialized

        # Test for deterministic behavior with fixed seed
        affine_dup = Affine(dim)
        affine_dup = set_rngs(affine_dup, key)
        output_dup = affine_dup(x)
        np.testing.assert_array_equal(output, output_dup)


class TestScale:
    def test_scale_layer_initialization(self):
        # Test initialization parameters
        scale = Scale(offset=1.0)

        assert scale.offset == 1.0
        assert not scale.scale.initialized  # Scale should be lazily initialized
        assert scale.dim is None  # Not initialized yet

        # Test with different offset
        scale2 = Scale(offset=0.5)
        assert scale2.offset == 0.5

    def test_scale_layer_forward(self):
        # Test the forward pass
        key = jrn.PRNGKey(0)
        scale = Scale(offset=1.0)
        scale = set_rngs(scale, key)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Get output and ensure scale is initialized
        output = scale(x)
        assert scale.scale.initialized
        assert scale.scale.value.shape == (3,)
        assert output.shape == (2, 3)

        # Test for deterministic behavior with fixed seed
        scale_dup = Scale(offset=1.0)
        scale_dup = set_rngs(scale_dup, key)
        output_dup = scale_dup(x)
        np.testing.assert_array_equal(output, output_dup)

    def test_scale_layer_with_offset_zero(self):
        # Test with offset=0
        key = jrn.PRNGKey(0)
        scale = Scale(offset=0.0)
        scale = set_rngs(scale, key)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Forward pass
        output = scale(x)

        # Verify that the scale is initialized and output shape is correct
        assert scale.scale.initialized
        assert output.shape == (2, 3)


def test_linear_jit_compatibility():
    # Test JIT compatibility for Linear layer
    key = jrn.PRNGKey(0)
    dim = 16
    dim_in = 8

    @jax.jit
    def apply_linear(x, linear):
        return linear(x)

    linear = Linear(dim)
    linear = set_rngs(linear, key)
    x = jnp.ones((4, dim_in))

    # This should compile and run without errors
    result = apply_linear(x, linear)

    assert result.shape == (4, dim)


def test_bias_jit_compatibility():
    # Test JIT compatibility for Bias layer
    key = jrn.PRNGKey(0)

    @jax.jit
    def apply_bias(x, bias):
        return bias(x)

    bias = Bias()
    bias = set_rngs(bias, key)
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_bias(x, bias)

    assert result.shape == (4, 8)


def test_affine_jit_compatibility():
    # Test JIT compatibility for Affine layer
    key = jrn.PRNGKey(0)
    dim = 16

    @jax.jit
    def apply_affine(x, affine):
        return affine(x)

    affine = Affine(dim)
    affine = set_rngs(affine, key)
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_affine(x, affine)

    assert result.shape == (4, dim)


def test_scale_jit_compatibility():
    # Test JIT compatibility for Scale layer
    key = jrn.PRNGKey(0)

    @jax.jit
    def apply_scale(x, scale):
        return scale(x)

    scale = Scale()
    scale = set_rngs(scale, key)
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_scale(x, scale)

    assert result.shape == (4, 8)


def test_multiple_calls_consistency():
    # Test that multiple calls with the same input produce the same output
    key = jrn.PRNGKey(0)
    dim = 16
    dim_in = 8

    linear = Linear(dim)
    # Set the RNG key for parameter initialization
    linear = set_rngs(linear, key)

    x = jnp.ones((4, dim_in))

    # First call should initialize weights
    out1 = linear(x)

    # Second call should use the same weights
    out2 = linear(x)

    # Outputs should be identical
    np.testing.assert_array_equal(out1, out2)
