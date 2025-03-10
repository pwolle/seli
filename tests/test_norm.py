import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from seli.net import LayerNorm, RMSNorm


class TestLayerNorm:
    def test_layernorm_initialization(self):
        # Test initialization with default parameters
        layernorm = LayerNorm()

        assert layernorm.eps == 1e-6
        assert layernorm.offset == 1
        assert layernorm.weight is None  # Weight should be lazily initialized
        assert layernorm.bias is None  # Bias should be lazily initialized

        # Test initialization with custom parameters
        custom_eps = 1e-5
        custom_offset = 0.5
        layernorm_custom = LayerNorm(eps=custom_eps, offset=custom_offset)

        assert layernorm_custom.eps == custom_eps
        assert layernorm_custom.offset == custom_offset

    def test_layernorm_build(self):
        # Test the _build method with a sample input
        layernorm = LayerNorm()

        # Create a sample input tensor
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Manually trigger build
        layernorm._build(x)

        # Check weight and bias shape and initialization values
        assert layernorm.weight is not None
        assert layernorm.bias is not None
        assert layernorm.weight.shape == (dim,)
        assert layernorm.bias.shape == (dim,)
        assert jnp.all(layernorm.weight == 0)  # Weight should be initialized to zeros
        assert jnp.all(layernorm.bias == 0)  # Bias should be initialized to zeros

    def test_layernorm_forward(self):
        # Test the forward pass
        layernorm = LayerNorm()

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Initialize weights and bias manually for deterministic testing
        layernorm.weight = jnp.array([0.1, 0.2, 0.3])
        layernorm.bias = jnp.array([0.4, 0.5, 0.6])

        # Compute output
        output = layernorm(x)

        # Manually compute the expected output
        # First normalize
        mean = x.mean(axis=-1, keepdims=True)  # [1, 2, 0]
        x_centered = x - mean
        var = x_centered.var(axis=-1, keepdims=True)
        x_norm = x_centered * lax.rsqrt(var + layernorm.eps)

        # Then apply scale and shift
        expected = x_norm * (layernorm.weight + layernorm.offset) + layernorm.bias

        # Check that the outputs match
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_layernorm_different_batch_shapes(self):
        # Test with different batch shapes

        # 1D input (just features)
        layernorm1 = LayerNorm()
        x1 = jnp.array([1.0, 2.0, 3.0])
        out1 = layernorm1(x1)
        assert out1.shape == x1.shape

        # 2D input (batch x features)
        layernorm2 = LayerNorm()
        x2 = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out2 = layernorm2(x2)
        assert out2.shape == x2.shape

        # 3D input (batch x seq_len x features)
        layernorm3 = LayerNorm()
        x3 = jnp.ones((2, 3, 4))
        out3 = layernorm3(x3)
        assert out3.shape == x3.shape

    def test_layernorm_zero_offset(self):
        # Test with offset=0
        layernorm = LayerNorm(offset=0)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Initialize weights and bias manually
        layernorm.weight = jnp.array([0.1, 0.2, 0.3])
        layernorm.bias = jnp.array([0.4, 0.5, 0.6])

        # Compute output
        output = layernorm(x)

        # Manually compute the expected output with offset=0
        mean = x.mean(axis=-1, keepdims=True)
        x_centered = x - mean
        var = x_centered.var(axis=-1, keepdims=True)
        x_norm = x_centered * lax.rsqrt(var + layernorm.eps)

        # Scale and shift (with offset=0)
        expected = x_norm * layernorm.weight + layernorm.bias

        # Check that the outputs match
        np.testing.assert_allclose(output, expected, rtol=1e-5)


class TestRMSNorm:
    def test_rmsnorm_initialization(self):
        # Test initialization with default parameters
        rmsnorm = RMSNorm()

        assert rmsnorm.eps == 1e-6
        assert rmsnorm.offset == 1
        assert rmsnorm.weight is None  # Weight should be lazily initialized
        assert rmsnorm.bias is None  # Bias should be lazily initialized

        # Test initialization with custom parameters
        custom_eps = 1e-5
        custom_offset = 0.5
        rmsnorm_custom = RMSNorm(eps=custom_eps, offset=custom_offset)

        assert rmsnorm_custom.eps == custom_eps
        assert rmsnorm_custom.offset == custom_offset

    def test_rmsnorm_build(self):
        # Test the _build method with a sample input
        rmsnorm = RMSNorm()

        # Create a sample input tensor
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Manually trigger build
        rmsnorm._build(x)

        # Check weight and bias shape and initialization values
        assert rmsnorm.weight is not None
        assert rmsnorm.bias is not None
        assert rmsnorm.weight.shape == (dim,)
        assert rmsnorm.bias.shape == (dim,)
        assert jnp.all(rmsnorm.weight == 1)  # Weight should be initialized to ones
        assert jnp.all(rmsnorm.bias == 0)  # Bias should be initialized to zeros

    def test_rmsnorm_forward(self):
        # Test the forward pass
        rmsnorm = RMSNorm()

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Initialize weights and bias manually for deterministic testing
        rmsnorm.weight = jnp.array([0.1, 0.2, 0.3])
        rmsnorm.bias = jnp.array([0.4, 0.5, 0.6])

        # Compute output
        output = rmsnorm(x)

        # Manually compute the expected output
        # First compute RMS normalization
        rms = jnp.sqrt((x**2).mean(axis=-1, keepdims=True) + rmsnorm.eps)
        x_norm = x / rms

        # Then apply scale and shift
        expected = x_norm * (rmsnorm.weight + rmsnorm.offset) + rmsnorm.bias

        # Check that the outputs match
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_rmsnorm_different_batch_shapes(self):
        # Test with different batch shapes

        # 1D input (just features)
        rmsnorm1 = RMSNorm()
        x1 = jnp.array([1.0, 2.0, 3.0])
        out1 = rmsnorm1(x1)
        assert out1.shape == x1.shape

        # 2D input (batch x features)
        rmsnorm2 = RMSNorm()
        x2 = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out2 = rmsnorm2(x2)
        assert out2.shape == x2.shape

        # 3D input (batch x seq_len x features)
        rmsnorm3 = RMSNorm()
        x3 = jnp.ones((2, 3, 4))
        out3 = rmsnorm3(x3)
        assert out3.shape == x3.shape

    def test_rmsnorm_zero_offset(self):
        # Test with offset=0
        rmsnorm = RMSNorm(offset=0)

        # Create a sample input tensor
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Initialize weights and bias manually
        rmsnorm.weight = jnp.array([0.1, 0.2, 0.3])
        rmsnorm.bias = jnp.array([0.4, 0.5, 0.6])

        # Compute output
        output = rmsnorm(x)

        # Manually compute the expected output with offset=0
        rms = jnp.sqrt((x**2).mean(axis=-1, keepdims=True) + rmsnorm.eps)
        x_norm = x / rms

        # Scale and shift (with offset=0)
        expected = x_norm * rmsnorm.weight + rmsnorm.bias

        # Check that the outputs match
        np.testing.assert_allclose(output, expected, rtol=1e-5)


def test_layernorm_jit_compatibility():
    # Test JIT compatibility for LayerNorm
    @jax.jit
    def apply_layernorm(x, norm):
        return norm(x)

    layernorm = LayerNorm()
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_layernorm(x, layernorm)

    assert result.shape == (4, 8)


def test_rmsnorm_jit_compatibility():
    # Test JIT compatibility for RMSNorm
    @jax.jit
    def apply_rmsnorm(x, norm):
        return norm(x)

    rmsnorm = RMSNorm()
    x = jnp.ones((4, 8))

    # This should compile and run without errors
    result = apply_rmsnorm(x, rmsnorm)

    assert result.shape == (4, 8)


def test_layernorm_multiple_calls_consistency():
    # Test that multiple calls with the same input produce the same output
    layernorm = LayerNorm()
    x = jnp.ones((4, 8))

    # First call should initialize weights and bias
    out1 = layernorm(x)

    # Second call should use the same weights and bias
    out2 = layernorm(x)

    # Outputs should be identical
    np.testing.assert_array_equal(out1, out2)


def test_rmsnorm_multiple_calls_consistency():
    # Test that multiple calls with the same input produce the same output
    rmsnorm = RMSNorm()
    x = jnp.ones((4, 8))

    # First call should initialize weights and bias
    out1 = rmsnorm(x)

    # Second call should use the same weights and bias
    out2 = rmsnorm(x)

    # Outputs should be identical
    np.testing.assert_array_equal(out1, out2)


def test_layernorm_and_rmsnorm_numerical_stability():
    # Test normalization with near-zero variance
    x = jnp.ones((4, 8)) * 1e-8  # Very small values

    # Should not have NaN values in output
    layernorm = LayerNorm()
    rmsnorm = RMSNorm()

    layernorm_output = layernorm(x)
    rmsnorm_output = rmsnorm(x)

    assert not jnp.any(jnp.isnan(layernorm_output))
    assert not jnp.any(jnp.isnan(rmsnorm_output))
