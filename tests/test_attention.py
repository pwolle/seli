import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

from seli.net import CrossAttention, DotProductAttention
from seli.net._attention import normalize, softcap


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
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        attn = DotProductAttention(key, dim=dim, heads_q=heads_q)

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
            key,
            dim=dim,
            heads_q=heads_q_custom,
            heads_k=heads_k_custom,
            norm=True,
            tanh_cap=5.0,
            scale=0.125,
            is_causal=True,
        )

        assert attn_custom.heads_q == heads_q_custom
        assert attn_custom.heads_k == heads_k_custom
        assert attn_custom.norm is True
        assert attn_custom.tanh_cap == 5.0
        assert attn_custom.scale == 0.125
        assert attn_custom.is_causal is True

    def test_forward_shapes(self):
        # Test that the forward pass produces outputs with expected shapes
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        attn = DotProductAttention(key, dim=dim, heads_q=heads_q)

        # Test with batch shape and sequence length
        batch_size = 2
        seq_len = 10
        x = jnp.ones((batch_size, seq_len, dim))

        output = attn(x)

        # Output should have the same shape as input
        assert output.shape == x.shape

    def test_forward_with_bias_and_mask(self):
        # Test the forward pass with bias and mask
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        attn = DotProductAttention(key, dim=dim, heads_q=heads_q)

        batch_size = 2
        seq_len = 10
        x = jnp.ones((batch_size, seq_len, dim))

        # Create a bias tensor (attention_heads, seq_len, seq_len)
        # JAX expects (1, heads, seq_len, seq_len) for broadcasting to batch
        bias = jnp.zeros((1, heads_q, seq_len, seq_len))

        # Create a mask tensor (1, 1, seq_len, seq_len)
        mask = jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_)

        # Forward pass with bias and mask
        output = attn(x, bias=bias, mask=mask)

        # Output should have the same shape as input
        assert output.shape == x.shape

    def test_causal_attention(self):
        # Test causal attention mask
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        # Create causal attention
        causal_attn = DotProductAttention(key, dim=dim, heads_q=heads_q, is_causal=True)

        batch_size = 2
        seq_len = 10
        x = jnp.ones((batch_size, seq_len, dim))

        # Forward pass with causal mask
        output_causal = causal_attn(x)

        # Output should have the same shape as input
        assert output_causal.shape == x.shape

    def test_dim_in_property(self):
        # Test the dim_in property
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        attn = DotProductAttention(key, dim=dim, heads_q=heads_q)

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
        dim = 64
        heads_q = 4

        attn = DotProductAttention(key, dim=dim, heads_q=heads_q)

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
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        cross_attn = CrossAttention(key, dim=dim, heads_q=heads_q)

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

        bias = jnp.zeros((1, heads_q_custom, 5, 7))  # (1, heads, seq_q, seq_kv)
        mask = jnp.ones((1, 1, 5, 7), dtype=jnp.bool_)  # (1, 1, seq_q, seq_kv)

        cross_attn_custom = CrossAttention(
            key,
            dim=dim,
            heads_q=heads_q_custom,
            heads_k=heads_k_custom,
            bias=bias,
            mask=mask,
            scale=0.125,
            is_causal=True,
        )

        assert cross_attn_custom.heads_q == heads_q_custom
        assert cross_attn_custom.heads_k == heads_k_custom
        assert cross_attn_custom.bias is bias
        assert cross_attn_custom.mask is mask
        assert cross_attn_custom.scale == 0.125
        assert cross_attn_custom.is_causal is True

    def test_forward_shapes(self):
        # Test that the forward pass produces outputs with expected shapes
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        cross_attn = CrossAttention(key, dim=dim, heads_q=heads_q)

        # Test with batch shape and different sequence lengths for x and y
        batch_size = 2
        seq_len_x = 10
        seq_len_y = 15

        x = jnp.ones((batch_size, seq_len_x, dim))
        y = jnp.ones((batch_size, seq_len_y, dim))

        # Forward pass
        output = cross_attn(x, y)

        # Output should have the same shape as x
        assert output.shape == x.shape

    def test_causal_cross_attention(self):
        # Test causal cross attention
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        # Create causal cross attention
        causal_cross_attn = CrossAttention(
            key, dim=dim, heads_q=heads_q, is_causal=True
        )

        batch_size = 2
        seq_len_x = 10
        seq_len_y = 10  # Same length for causal attention

        x = jnp.ones((batch_size, seq_len_x, dim))
        y = jnp.ones((batch_size, seq_len_y, dim))

        # Forward pass with causal mask
        output_causal = causal_cross_attn(x, y)

        # Output should have the same shape as x
        assert output_causal.shape == x.shape

    def test_dim_in_properties(self):
        # Test the dim_in_x and dim_in_y properties
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        cross_attn = CrossAttention(key, dim=dim, heads_q=heads_q)

        # Initially, properties should be None
        assert cross_attn.dim_in_x is None
        assert cross_attn.dim_in_y is None

        # After a forward pass, properties should be set
        batch_size = 2
        seq_len_x = 10
        seq_len_y = 15

        x = jnp.ones((batch_size, seq_len_x, dim))
        y = jnp.ones((batch_size, seq_len_y, dim))

        cross_attn(x, y)

        assert cross_attn.dim_in_x == dim
        assert cross_attn.dim_in_y == dim

    def test_jit_compatibility(self):
        # Test JIT compatibility
        key = jrn.PRNGKey(0)
        dim = 64
        heads_q = 4

        cross_attn = CrossAttention(key, dim=dim, heads_q=heads_q)

        batch_size = 2
        seq_len_x = 10
        seq_len_y = 15

        x = jnp.ones((batch_size, seq_len_x, dim))
        y = jnp.ones((batch_size, seq_len_y, dim))

        # Create fresh keys for the q and kv linear layers
        # This ensures the keys aren't consumed during tracing
        cross_attn.q.key = jrn.PRNGKey(1)
        cross_attn.kv.key = jrn.PRNGKey(2)

        # JIT compile the forward function
        @jax.jit
        def forward(module, inputs_x, inputs_y):
            return module(inputs_x, inputs_y)

        # This should compile and run without errors
        output = forward(cross_attn, x, y)

        assert output.shape == x.shape


def test_valid_attention_head_configurations():
    # Test valid attention configurations that work with the JAX constraint
    key = jrn.PRNGKey(0)
    dim = 64

    # For standard self-attention (equal heads)
    std_heads = 4
    std_attn = DotProductAttention(key, dim=dim, heads_q=std_heads)

    # For grouped query attention (GQA) where multiple query heads share a
    # key/value head
    heads_q_gqa1 = 8
    heads_k_gqa1 = 4  # 8 % 4 == 0
    gqa_attn1 = DotProductAttention(
        key, dim=dim, heads_q=heads_q_gqa1, heads_k=heads_k_gqa1
    )

    # Another GQA configuration
    heads_q_gqa2 = 16
    heads_k_gqa2 = 2  # 16 % 2 == 0
    gqa_attn2 = DotProductAttention(
        key, dim=dim, heads_q=heads_q_gqa2, heads_k=heads_k_gqa2
    )

    # For true multi-query attention (MQA) where all query heads share a single
    #  key/value head
    heads_q_mqa = 4
    heads_k_mqa = 1  # The defining characteristic of MQA is heads_k = 1
    mqa_attn = DotProductAttention(
        key, dim=dim, heads_q=heads_q_mqa, heads_k=heads_k_mqa
    )

    batch_size = 2
    seq_len = 10
    x = jnp.ones((batch_size, seq_len, dim))

    # Run all configurations
    output_std = std_attn(x)
    output_gqa1 = gqa_attn1(x)
    output_gqa2 = gqa_attn2(x)
    output_mqa = mqa_attn(x)

    # Check shapes
    assert output_std.shape == x.shape
    assert output_gqa1.shape == x.shape
    assert output_gqa2.shape == x.shape
    assert output_mqa.shape == x.shape

    # Same for CrossAttention
    cross_std = CrossAttention(key, dim=dim, heads_q=std_heads)
    cross_gqa1 = CrossAttention(
        key, dim=dim, heads_q=heads_q_gqa1, heads_k=heads_k_gqa1
    )
    cross_gqa2 = CrossAttention(
        key, dim=dim, heads_q=heads_q_gqa2, heads_k=heads_k_gqa2
    )
    cross_mqa = CrossAttention(key, dim=dim, heads_q=heads_q_mqa, heads_k=heads_k_mqa)

    seq_len_y = 15
    y = jnp.ones((batch_size, seq_len_y, dim))

    output_cross_std = cross_std(x, y)
    output_cross_gqa1 = cross_gqa1(x, y)
    output_cross_gqa2 = cross_gqa2(x, y)
    output_cross_mqa = cross_mqa(x, y)

    assert output_cross_std.shape == x.shape
    assert output_cross_gqa1.shape == x.shape
    assert output_cross_gqa2.shape == x.shape
    assert output_cross_mqa.shape == x.shape
