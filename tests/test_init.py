import jax
import jax.numpy as jnp
import pytest

from seli.net._init import (
    Constant,
    Init,
    Normal,
    Orthogonal,
    TruncatedNormal,
    Uniform,
    Zeros,
)


class TestZeros:
    def test_zeros_initializer(self):
        # Test Zeros initializer produces all zeros
        key = jax.random.PRNGKey(0)
        shape = (3, 4)
        dtype = jnp.float32

        initializer = Zeros()
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        assert jnp.all(result == 0)


class TestConstant:
    def test_constant_initializer_default(self):
        # Test Constant initializer with default value (1.0)
        key = jax.random.PRNGKey(0)
        shape = (2, 5)
        dtype = jnp.float32

        initializer = Constant()  # default value=1.0
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        assert jnp.all(result == 1.0)

    def test_constant_initializer_custom(self):
        # Test Constant initializer with custom value
        key = jax.random.PRNGKey(0)
        shape = (3, 3)
        dtype = jnp.float32
        value = 3.14

        initializer = Constant(value=value)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        assert jnp.all(result == value)


class TestTruncatedNormal:
    def test_truncated_normal_initializer_default(self):
        # Test TruncatedNormal initializer with default parameters
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)  # Large shape for statistical testing
        dtype = jnp.float32

        initializer = TruncatedNormal()  # default minv=-1.0, maxv=1.0
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # All values should be within the truncation bounds
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)
        # Mean should be approximately 0
        assert -0.1 < jnp.mean(result) < 0.1

    def test_truncated_normal_custom_bounds(self):
        # Test TruncatedNormal initializer with custom bounds
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)
        dtype = jnp.float32
        minv = 0.0
        maxv = 2.0

        initializer = TruncatedNormal(minv=minv, maxv=maxv)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # All values should be within the truncation bounds
        assert jnp.all(result >= minv)
        assert jnp.all(result <= maxv)

    def test_truncated_normal_shift_scale(self):
        # Test TruncatedNormal initializer with shift and scale
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)
        dtype = jnp.float32
        shift = 5.0
        scale = 2.0

        initializer = TruncatedNormal(shift=shift, scale=scale)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # Values should be shifted and scaled
        assert jnp.all(result >= (-1.0 * scale + shift))
        assert jnp.all(result <= (1.0 * scale + shift))
        # Mean should be approximately equal to shift
        assert (shift - 0.5) < jnp.mean(result) < (shift + 0.5)


class TestNormal:
    def test_normal_unit_initializer(self):
        # Test Normal initializer with "Unit" mode
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)
        dtype = jnp.float32

        initializer = Normal(init="Unit")
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # Values should be within [-1, 1] for Unit initialization
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)

    @pytest.mark.parametrize(
        "init_method", ["He", "Xavier", "Glorot", "Kaiming", "LeCun"]
    )
    def test_normal_various_methods(self, init_method):
        # Test Normal initializer with various initialization methods
        key = jax.random.PRNGKey(0)
        shape = (100, 100)
        dtype = jnp.float32

        initializer = Normal(init=init_method)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype

    def test_normal_shift_scale(self):
        # Test Normal initializer with shift and scale
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)
        dtype = jnp.float32
        shift = 3.0
        scale = 2.0

        initializer = Normal(shift=shift, scale=scale)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # Mean should be approximately equal to shift
        assert (shift - 0.5) < jnp.mean(result) < (shift + 0.5)
        # Standard deviation should be approximately scale times the original std
        # (considering He initialization, it won't be exactly scale)


class TestUniform:
    def test_uniform_unit_initializer(self):
        # Test Uniform initializer with "Unit" mode
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)
        dtype = jnp.float32

        initializer = Uniform(init="Unit")
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # Values should be within [-1, 1] for Unit initialization
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)

    @pytest.mark.parametrize(
        "init_method", ["He", "Xavier", "Glorot", "Kaiming", "LeCun"]
    )
    def test_uniform_various_methods(self, init_method):
        # Test Uniform initializer with various initialization methods
        key = jax.random.PRNGKey(0)
        shape = (100, 100)
        dtype = jnp.float32

        initializer = Uniform(init=init_method)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype

    def test_uniform_shift_scale(self):
        # Test Uniform initializer with shift and scale
        key = jax.random.PRNGKey(0)
        shape = (1000, 1000)
        dtype = jnp.float32
        shift = 3.0
        scale = 2.0

        initializer = Uniform(shift=shift, scale=scale)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype
        # Mean should be approximately equal to shift
        assert (shift - 0.5) < jnp.mean(result) < (shift + 0.5)


class TestOrthogonal:
    def test_orthogonal_initializer(self):
        # Test Orthogonal initializer produces orthogonal matrices
        key = jax.random.PRNGKey(0)
        shape = (10, 10)  # Square matrix for simplicity
        dtype = jnp.float32

        initializer = Orthogonal()
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype

        # Check orthogonality: W^T W should be approximately identity
        product = jnp.matmul(result.T, result)
        assert jnp.allclose(product, jnp.eye(shape[1]), atol=1e-5)

    def test_orthogonal_rectangular(self):
        # Test Orthogonal initializer with rectangular matrices
        key = jax.random.PRNGKey(0)
        shape = (20, 10)  # Rectangular matrix
        dtype = jnp.float32

        initializer = Orthogonal()
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype

        # For rectangular matrix, check W^T W (if rows > cols)
        product = jnp.matmul(result.T, result)
        assert jnp.allclose(product, jnp.eye(shape[1]), atol=1e-5)

    def test_orthogonal_scale(self):
        # Test Orthogonal initializer with scale
        key = jax.random.PRNGKey(0)
        shape = (10, 10)
        dtype = jnp.float32
        scale = 2.0

        initializer = Orthogonal(scale=scale)
        result = initializer(key, shape, dtype)

        assert result.shape == shape
        assert result.dtype == dtype

        # Check scaled orthogonality: W^T W should be approximately scale^2 * identity
        product = jnp.matmul(result.T, result)
        assert jnp.allclose(product, scale**2 * jnp.eye(shape[1]), atol=1e-4)


def test_init_error():
    # Test that the base Init class raises NotImplementedError when called
    key = jax.random.PRNGKey(0)
    shape = (3, 4)
    dtype = jnp.float32

    initializer = Init()
    with pytest.raises(NotImplementedError):
        initializer(key, shape, dtype)


def test_invalid_normal_init():
    # Test that Normal raises ValueError with invalid init method
    key = jax.random.PRNGKey(0)
    shape = (3, 4)
    dtype = jnp.float32

    initializer = Normal(init="InvalidMethod")  # Invalid method
    with pytest.raises(ValueError):
        initializer(key, shape, dtype)


def test_invalid_uniform_init():
    # Test that Uniform raises ValueError with invalid init method
    key = jax.random.PRNGKey(0)
    shape = (3, 4)
    dtype = jnp.float32

    initializer = Uniform(init="InvalidMethod")  # Invalid method
    with pytest.raises(ValueError):
        initializer(key, shape, dtype)
