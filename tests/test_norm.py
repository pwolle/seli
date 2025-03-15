import jax
import jax.numpy as jnp
import numpy as np

from seli.net import LayerNorm, RMSNorm
from seli.net._key import set_rngs


class TestLayerNorm:
    def test_layernorm_initialization(self):
        # Test initialization with default parameters
        layernorm = LayerNorm()

        assert layernorm.eps == 1e-6
        assert layernorm.offset == 1
        assert not layernorm.weight.initialized  # Weight should be lazily initialized
        assert not layernorm.bias.initialized  # Bias should be lazily initialized

        # Test initialization with custom parameters
        custom_eps = 1e-5
        custom_offset = 0.5
        layernorm_custom = LayerNorm(eps=custom_eps, offset=custom_offset)

        assert layernorm_custom.eps == custom_eps
        assert layernorm_custom.offset == custom_offset

    def test_layernorm_forward(self):
        # Test the forward pass
        key = jax.random.PRNGKey(0)
        layernorm = LayerNorm()
        layernorm = set_rngs(layernorm, key)

        # Create a sample input tensor
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Get output and ensure parameters are initialized
        output = layernorm(x)
        assert layernorm.weight.initialized
        assert layernorm.bias.initialized
        assert layernorm.weight.value.shape == (dim,)
        assert layernorm.bias.value.shape == (dim,)
        assert output.shape == x.shape

        # Test for deterministic behavior with fixed seed
        layernorm_dup = LayerNorm()
        layernorm_dup = set_rngs(layernorm_dup, key)
        output_dup = layernorm_dup(x)
        np.testing.assert_array_equal(output, output_dup)

    def test_layernorm_different_batch_shapes(self):
        # Test with different batch shapes
        key = jax.random.PRNGKey(0)
        layernorm = LayerNorm()
        layernorm = set_rngs(layernorm, key)

        # 1D input (just features)
        dim = 8
        x_1d = jnp.ones((dim,))
        output_1d = layernorm(x_1d)
        assert output_1d.shape == (dim,)

        # 2D input (batch, features)
        batch_size = 4
        x_2d = jnp.ones((batch_size, dim))
        output_2d = layernorm(x_2d)
        assert output_2d.shape == (batch_size, dim)

        # 3D input (batch, seq, features)
        seq_len = 10
        x_3d = jnp.ones((batch_size, seq_len, dim))
        output_3d = layernorm(x_3d)
        assert output_3d.shape == (batch_size, seq_len, dim)

    def test_layernorm_zero_offset(self):
        # Test with offset=0
        key = jax.random.PRNGKey(0)
        layernorm = LayerNorm(offset=0)
        layernorm = set_rngs(layernorm, key)

        # Create inputs
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Initial forward pass
        output = layernorm(x)

        # Outputs should be influenced by the zero offset
        assert output.shape == x.shape


class TestRMSNorm:
    def test_rmsnorm_initialization(self):
        # Test initialization with default parameters
        rmsnorm = RMSNorm()

        assert rmsnorm.eps == 1e-6
        assert rmsnorm.offset == 1
        assert not rmsnorm.weight.initialized  # Weight should be lazily initialized
        assert not rmsnorm.bias.initialized  # Bias should be lazily initialized

        # Test initialization with custom parameters
        custom_eps = 1e-5
        custom_offset = 0.5
        rmsnorm_custom = RMSNorm(eps=custom_eps, offset=custom_offset)

        assert rmsnorm_custom.eps == custom_eps
        assert rmsnorm_custom.offset == custom_offset

    def test_rmsnorm_forward(self):
        # Test the forward pass
        key = jax.random.PRNGKey(0)
        rmsnorm = RMSNorm()
        rmsnorm = set_rngs(rmsnorm, key)

        # Create a sample input tensor
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Get output and ensure parameters are initialized
        output = rmsnorm(x)
        assert rmsnorm.weight.initialized
        assert rmsnorm.bias.initialized
        assert rmsnorm.weight.value.shape == (dim,)
        assert rmsnorm.bias.value.shape == (dim,)
        assert output.shape == x.shape

        # Test for deterministic behavior with fixed seed
        rmsnorm_dup = RMSNorm()
        rmsnorm_dup = set_rngs(rmsnorm_dup, key)
        output_dup = rmsnorm_dup(x)
        np.testing.assert_array_equal(output, output_dup)

    def test_rmsnorm_different_batch_shapes(self):
        # Test with different batch shapes
        key = jax.random.PRNGKey(0)
        rmsnorm = RMSNorm()
        rmsnorm = set_rngs(rmsnorm, key)

        # 1D input (just features)
        dim = 8
        x_1d = jnp.ones((dim,))
        output_1d = rmsnorm(x_1d)
        assert output_1d.shape == (dim,)

        # 2D input (batch, features)
        batch_size = 4
        x_2d = jnp.ones((batch_size, dim))
        output_2d = rmsnorm(x_2d)
        assert output_2d.shape == (batch_size, dim)

        # 3D input (batch, seq, features)
        seq_len = 10
        x_3d = jnp.ones((batch_size, seq_len, dim))
        output_3d = rmsnorm(x_3d)
        assert output_3d.shape == (batch_size, seq_len, dim)

    def test_rmsnorm_zero_offset(self):
        # Test with offset=0
        key = jax.random.PRNGKey(0)
        rmsnorm = RMSNorm(offset=0)
        rmsnorm = set_rngs(rmsnorm, key)

        # Create inputs
        dim = 8
        batch_size = 4
        x = jnp.ones((batch_size, dim))

        # Initial forward pass
        output = rmsnorm(x)

        # Outputs should be influenced by the zero offset
        assert output.shape == x.shape


def test_layernorm_jit_compatibility():
    # Test JIT compatibility for LayerNorm
    key = jax.random.PRNGKey(0)

    @jax.jit
    def apply_layernorm(x, norm):
        return norm(x)

    layernorm = LayerNorm()
    layernorm = set_rngs(layernorm, key)

    # Create a sample input tensor
    dim = 8
    batch_size = 4
    x = jnp.ones((batch_size, dim))

    # This should compile and run without errors
    result = apply_layernorm(x, layernorm)

    assert result.shape == x.shape


def test_rmsnorm_jit_compatibility():
    # Test JIT compatibility for RMSNorm
    key = jax.random.PRNGKey(0)

    @jax.jit
    def apply_rmsnorm(x, norm):
        return norm(x)

    rmsnorm = RMSNorm()
    rmsnorm = set_rngs(rmsnorm, key)

    # Create a sample input tensor
    dim = 8
    batch_size = 4
    x = jnp.ones((batch_size, dim))

    # This should compile and run without errors
    result = apply_rmsnorm(x, rmsnorm)

    assert result.shape == x.shape


def test_layernorm_multiple_calls_consistency():
    # Test that multiple calls with the same input produce the same output
    key = jax.random.PRNGKey(0)
    layernorm = LayerNorm()
    layernorm = set_rngs(layernorm, key)

    dim = 8
    batch_size = 4
    x = jnp.ones((batch_size, dim))

    # First call should initialize parameters
    out1 = layernorm(x)

    # Second call should use the same parameters
    out2 = layernorm(x)

    # Outputs should be identical
    np.testing.assert_array_equal(out1, out2)


def test_rmsnorm_multiple_calls_consistency():
    # Test that multiple calls with the same input produce the same output
    key = jax.random.PRNGKey(0)
    rmsnorm = RMSNorm()
    rmsnorm = set_rngs(rmsnorm, key)

    dim = 8
    batch_size = 4
    x = jnp.ones((batch_size, dim))

    # First call should initialize parameters
    out1 = rmsnorm(x)

    # Second call should use the same parameters
    out2 = rmsnorm(x)

    # Outputs should be identical
    np.testing.assert_array_equal(out1, out2)


def test_layernorm_and_rmsnorm_numerical_stability():
    # Test normalization with near-zero variance
    key = jax.random.PRNGKey(0)

    layernorm = LayerNorm()
    layernorm = set_rngs(layernorm, key)

    rmsnorm = RMSNorm()
    rmsnorm = set_rngs(rmsnorm, key)

    # Create an input tensor with almost identical values
    x = jnp.ones((4, 8)) + jnp.ones((4, 8)) * 1e-8

    # Test LayerNorm (should handle near-zero variance gracefully)
    ln_output = layernorm(x)
    assert not jnp.any(jnp.isnan(ln_output))
    assert not jnp.any(jnp.isinf(ln_output))

    # Test RMSNorm (should handle near-zero variance gracefully)
    rms_output = rmsnorm(x)
    assert not jnp.any(jnp.isnan(rms_output))
    assert not jnp.any(jnp.isinf(rms_output))
