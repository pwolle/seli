import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

from seli.net import Bias, Linear, Scale
from seli.net._linear import Affine


class TestLinear:
    def test_linear_layer_initialization(self):
        # Test initialization parameters
        key = jrn.PRNGKey(0)
        dim = 16
        linear = Linear(key, dim)

        assert linear.dim == dim
        assert linear.weight is None  # Weight should be lazily initialized
        assert linear.dim_in is None  # Not initialized yet

    def test_linear_layer_build(self):
        # Test the _build method with a sample input
        key = jrn.PRNGKey(0)
        dim = 16
        dim_in = 8
        linear = Linear(key, dim)

        # Create a sample input tensor
        batch_size = 4
        x = jnp.ones((batch_size, dim_in))

        # Manually trigger build
        linear._build(x)

        # Check weight shape and type
        assert linear.weight is not None
        assert linear.weight.shape == (dim_in, dim)
        assert linear.dim_in == dim_in

    def test_linear_layer_forward(self):
        # Test the forward pass with a controlled input
        key = jrn.PRNGKey(0)
        dim = 3
        linear = Linear(key, dim)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Override the random weight initialization for deterministic testing
        known_weights = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        linear.weight = known_weights

        # Compute expected output manually
        expected_output = x @ known_weights

        # Get the actual output
        actual_output = linear(x)

        # Check that the outputs match
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5)


class TestBias:
    def test_bias_layer_initialization(self):
        # Test initialization parameters
        key = jrn.PRNGKey(0)
        bias = Bias(key)

        assert bias.bias is None  # Bias should be lazily initialized
        assert bias.dim is None  # Not initialized yet

    def test_bias_layer_build(self):
        # Test the _build method with a sample input
        key = jrn.PRNGKey(0)
        bias = Bias(key)

        # Create a sample input tensor
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Manually trigger build
        bias._build(x)

        # Check bias shape and type
        assert bias.bias is not None
        assert bias.bias.shape == (dim,)
        assert bias.dim == dim

    def test_bias_layer_forward(self):
        # Test the forward pass with a controlled input
        key = jrn.PRNGKey(0)
        bias = Bias(key)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Override the random bias initialization for deterministic testing
        known_bias = jnp.array([0.1, 0.2, 0.3])
        bias.bias = known_bias

        # Compute expected output manually
        expected_output = x + known_bias

        # Get the actual output
        actual_output = bias(x)

        # Check that the outputs match
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5)


class TestAffine:
    def test_affine_layer_initialization(self):
        # Test initialization parameters
        key = jrn.PRNGKey(0)
        dim = 16
        affine = Affine(key, dim)

        assert isinstance(affine.linear, Linear)
        assert isinstance(affine.bias, Bias)
        assert affine.linear.dim == dim
        assert affine.dim_in is None  # Not initialized yet

    def test_affine_layer_forward(self):
        # Test the forward pass with a controlled input
        key = jrn.PRNGKey(0)
        dim = 3
        affine = Affine(key, dim)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Override the random weight and bias initialization for deterministic testing
        known_weights = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        known_bias = jnp.array([0.1, 0.2, 0.3])

        affine.linear.weight = known_weights
        affine.bias.bias = known_bias

        # Compute expected output manually
        expected_output = (x @ known_weights) + known_bias

        # Get the actual output
        actual_output = affine(x)

        # Check that the outputs match
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5)


class TestScale:
    def test_scale_layer_initialization(self):
        # Test initialization parameters
        scale = Scale(offset=1.0)

        assert scale.offset == 1.0
        assert scale.scale is None  # Scale should be lazily initialized
        assert scale.dim is None  # Not initialized yet

        # Test with different offset
        scale2 = Scale(offset=0.5)
        assert scale2.offset == 0.5

    def test_scale_layer_build(self):
        # Test the _build method with a sample input
        scale = Scale()

        # Create a sample input tensor
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Manually trigger build
        scale._build(x)

        # Check scale shape and type
        assert scale.scale is not None
        assert scale.scale.shape == (dim,)
        assert jnp.all(scale.scale == 0)  # Scale should be initialized to zeros
        assert scale.dim == dim

    def test_scale_layer_forward(self):
        # Test the forward pass with a controlled input
        scale = Scale(offset=1.0)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Override the random scale initialization for deterministic testing
        known_scale = jnp.array([0.1, 0.2, 0.3])
        scale.scale = known_scale

        # Compute expected output manually (scale * (scale + offset))
        expected_output = x * (known_scale + scale.offset)

        # Get the actual output
        actual_output = scale(x)

        # Check that the outputs match
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5)

    def test_scale_layer_with_offset_zero(self):
        # Test with offset=0
        scale = Scale(offset=0.0)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Override the scale for deterministic testing
        known_scale = jnp.array([0.1, 0.2, 0.3])
        scale.scale = known_scale

        # With offset=0, output should be x * scale
        expected_output = x * known_scale

        # Get the actual output
        actual_output = scale(x)

        # Check that the outputs match
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5)


def test_linear_jit_compatibility():
    # Test JIT compatibility for Linear layer
    key = jrn.PRNGKey(0)
    dim = 16
    dim_in = 8

    @jax.jit
    def apply_linear(x, linear):
        return linear(x)

    linear = Linear(key, dim)
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

    bias = Bias(key)
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_bias(x, bias)

    assert result.shape == (4, 8)


def test_affine_jit_compatibility():
    # Test JIT compatibility for Affine layer
    key = jrn.PRNGKey(0)
    dim = 16
    dim_in = 8

    @jax.jit
    def apply_affine(x, affine):
        return affine(x)

    affine = Affine(key, dim)
    x = jnp.ones((4, dim_in))

    # This should compile and run without errors
    result = apply_affine(x, affine)

    assert result.shape == (4, dim)


def test_scale_jit_compatibility():
    # Test JIT compatibility for Scale layer

    @jax.jit
    def apply_scale(x, scale):
        return scale(x)

    scale = Scale()
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_scale(x, scale)

    assert result.shape == (4, 8)


def test_multiple_calls_consistency():
    # Test that multiple calls with the same input produce the same output
    key = jrn.PRNGKey(0)
    dim = 16
    dim_in = 8

    linear = Linear(key, dim)
    x = jnp.ones((4, dim_in))

    # First call should initialize weights
    out1 = linear(x)

    # Second call should use the same weights
    out2 = linear(x)

    # Outputs should be identical
    np.testing.assert_array_equal(out1, out2)
