import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

from seli.net import CrossAttention, DotProductAttention
from seli.net._attention import normalize, softcap
from seli.net._key import set_rngs


def test_normalize_function():
    # Test the normalize function
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Normalize along axis 1
    normalized = normalize(x, axis=1)

    # Manually compute the expected output
    norms = jnp.sqrt(jnp.sum(x**2, axis=1, keepdims=True))
    expected = x / jnp.maximum(norms, 1e-6)

    # Check that the outputs match
    np.testing.assert_allclose(normalized, expected, rtol=1e-5)


def test_softcap_function():
    # Test the softcap function
    x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    cap = 5.0

    # Apply softcap
    capped = softcap(x, cap)

    # Manually compute the expected output
    expected = jnp.tanh(x / cap) * cap

    # Check that the outputs match
    np.testing.assert_allclose(capped, expected, rtol=1e-5)

    # Check that values are properly capped
    assert jnp.all(jnp.abs(capped) <= cap)


class TestDotProductAttention:
    def test_initialization(self):
        # Test initialization with required parameters
        dim = 16
        heads_q = 4

        attn = DotProductAttention(dim=dim, heads_q=heads_q)

        assert attn.dim == dim
        assert attn.heads_q == heads_q
        assert attn.heads_k == heads_q  # Default is heads_q
        assert attn.dim_head == dim // heads_q
        assert attn.norm is False
        assert attn.tanh_cap is None
        assert attn.scale is None
        assert attn.is_causal is False
        assert attn.key_value_seq_lengths is None
        assert attn.implementation is None

        # Test with custom parameters (heads_q must be multiple of heads_k)
        heads_q_custom = 8
        heads_k_custom = 2  # 8 % 2 == 0, satisfying the JAX constraint

        attn_custom = DotProductAttention(
            dim=dim,
            heads_q=heads_q_custom,
            heads_k=heads_k_custom,
        )

        assert attn_custom.heads_q == heads_q_custom
        assert attn_custom.heads_k == heads_k_custom

    def test_forward_shapes(self):
        # Test that the forward pass produces outputs with expected shapes
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        attn = DotProductAttention(dim=dim, heads_q=heads_q)
        attn = set_rngs(attn, key)

        # Create inputs with batch and sequence dimensions
        batch_size = 2
        seq_len = 10
        x = jnp.ones((batch_size, seq_len, dim))

        # Call the attention module
        y = attn(x)

        # Check output shape
        assert y.shape == (batch_size, seq_len, dim)

    def test_forward_with_bias_and_mask(self):
        # Test the forward pass with bias and mask
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        attn = DotProductAttention(dim=dim, heads_q=heads_q)
        attn = set_rngs(attn, key)

        # Create inputs and attention bias/mask
        batch_size = 2
        seq_len = 8
        x = jnp.ones((batch_size, seq_len, dim))

        # Create a bias tensor
        bias = jnp.zeros((1, heads_q, seq_len, seq_len))

        # Create a mask tensor (boolean)
        mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool)

        # Call with bias and mask
        y = attn(x, bias=bias, mask=mask)

        # Check output shape
        assert y.shape == (batch_size, seq_len, dim)

    def test_causal_attention(self):
        # Test causal attention mask
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        # Create causal attention
        causal_attn = DotProductAttention(dim=dim, heads_q=heads_q, is_causal=True)
        causal_attn = set_rngs(causal_attn, key)

        # Create inputs
        batch_size = 2
        seq_len = 6
        x = jnp.ones((batch_size, seq_len, dim))

        # Call the attention module
        y = causal_attn(x)

        # Check output shape
        assert y.shape == (batch_size, seq_len, dim)

    def test_dim_in_property(self):
        # Test the dim_in property
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        attn = DotProductAttention(dim=dim, heads_q=heads_q)
        attn = set_rngs(attn, key)

        # Initially, dim_in should be None
        assert attn.dim_in is None

        # After a forward pass, dim_in should be set
        batch_size = 2
        seq_len = 10
        x = jnp.ones((batch_size, seq_len, dim))

        attn(x)

        assert attn.dim_in == dim

    def test_jit_compatibility(self):
        # Test JIT compatibility
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        attn = DotProductAttention(dim=dim, heads_q=heads_q)
        attn = set_rngs(attn, key)

        batch_size = 2
        seq_len = 10
        x = jnp.ones((batch_size, seq_len, dim))

        # JIT compile the forward function
        @jax.jit
        def forward(module, inputs):
            return module(inputs)

        # This should compile and run without errors
        output = forward(attn, x)

        assert output.shape == x.shape


class TestCrossAttention:
    def test_initialization(self):
        # Test initialization with required parameters
        dim = 16
        heads_q = 4

        cross_attn = CrossAttention(dim=dim, heads_q=heads_q)

        assert cross_attn.dim == dim
        assert cross_attn.heads_q == heads_q
        assert cross_attn.heads_k == heads_q  # Default is heads_q
        assert cross_attn.dim_head == dim // heads_q
        assert cross_attn.bias is None
        assert cross_attn.mask is None
        assert cross_attn.scale is None
        assert cross_attn.is_causal is False
        assert cross_attn.key_value_seq_lengths is None
        assert cross_attn.implementation is None

        # Test with custom parameters (heads_q must be multiple of heads_k)
        heads_q_custom = 8
        heads_k_custom = 2  # 8 % 2 == 0, satisfying the JAX constraint

        cross_attn_custom = CrossAttention(
            dim=dim,
            heads_q=heads_q_custom,
            heads_k=heads_k_custom,
        )

        assert cross_attn_custom.heads_q == heads_q_custom
        assert cross_attn_custom.heads_k == heads_k_custom

    def test_forward_shapes(self):
        # Test that the forward pass produces outputs with expected shapes
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        cross_attn = CrossAttention(dim=dim, heads_q=heads_q)
        cross_attn = set_rngs(cross_attn, key)

        # Create inputs with different sequence lengths but same embedding dimension
        batch_size = 2
        seq_len_q = 10
        seq_len_kv = 15
        x = jnp.ones((batch_size, seq_len_q, dim))
        y = jnp.ones((batch_size, seq_len_kv, dim))

        # Call the cross-attention module
        output = cross_attn(x, y)

        # Check output shape - should match query shape
        assert output.shape == (batch_size, seq_len_q, dim)

    def test_causal_cross_attention(self):
        # Test causal cross attention
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        # Create causal cross attention
        causal_cross_attn = CrossAttention(dim=dim, heads_q=heads_q, is_causal=True)
        causal_cross_attn = set_rngs(causal_cross_attn, key)

        # Create inputs with different sequence lengths
        batch_size = 2
        seq_len_q = 8
        seq_len_kv = 8  # For causal, usually seq_q == seq_kv
        x = jnp.ones((batch_size, seq_len_q, dim))
        y = jnp.ones((batch_size, seq_len_kv, dim))

        # Call with causal mask
        output = causal_cross_attn(x, y)

        # Check output shape
        assert output.shape == (batch_size, seq_len_q, dim)

    def test_dim_in_properties(self):
        # Test the dim_in_x and dim_in_y properties
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        cross_attn = CrossAttention(dim=dim, heads_q=heads_q)
        cross_attn = set_rngs(cross_attn, key)

        # Initially, dim_in_x and dim_in_y should be None
        assert cross_attn.dim_in_x is None
        assert cross_attn.dim_in_y is None

        # After a forward pass, dim_in_x and dim_in_y should be set
        batch_size = 2
        seq_len_q = 10
        seq_len_kv = 15
        x = jnp.ones((batch_size, seq_len_q, dim))
        y = jnp.ones((batch_size, seq_len_kv, dim))

        cross_attn(x, y)

        assert cross_attn.dim_in_x == dim
        assert cross_attn.dim_in_y == dim

    def test_jit_compatibility(self):
        # Test JIT compatibility
        key = jrn.PRNGKey(0)
        dim = 16
        heads_q = 4

        cross_attn = CrossAttention(dim=dim, heads_q=heads_q)
        cross_attn = set_rngs(cross_attn, key)

        batch_size = 2
        seq_len_q = 10
        seq_len_kv = 15
        x = jnp.ones((batch_size, seq_len_q, dim))
        y = jnp.ones((batch_size, seq_len_kv, dim))

        # JIT compile the forward function
        @jax.jit
        def forward(module, inputs_x, inputs_y):
            return module(inputs_x, inputs_y)

        # This should compile and run without errors
        output = forward(cross_attn, x, y)

        assert output.shape == (batch_size, seq_len_q, dim)


def test_valid_attention_head_configurations():
    # Test valid attention configurations that work with the JAX constraint
    key = jrn.PRNGKey(0)
    dim = 16

    # For standard self-attention (equal heads)
    std_heads = 4
    std_attn = DotProductAttention(dim=dim, heads_q=std_heads)
    std_attn = set_rngs(std_attn, key)

    # For MHA where query heads are a multiple of key/value heads (GQA)
    heads_q = 8
    heads_k = 2  # 8 % 2 == 0
    gqa_attn = DotProductAttention(dim=dim, heads_q=heads_q, heads_k=heads_k)
    gqa_attn = set_rngs(gqa_attn, key)

    # Check configurations
    assert std_attn.heads_q == std_heads
    assert std_attn.heads_k == std_heads

    assert gqa_attn.heads_q == heads_q
    assert gqa_attn.heads_k == heads_k
