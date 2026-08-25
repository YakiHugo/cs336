"""Neural network building blocks (Assignment 1 §3).

Handout invariants:
- Linear stores W with shape (d_out, d_in); forward computes y = xWᵀ, the
  row-major/batch-outer convention (§3.2.1, §3.3.2).
- Embedding is a lookup table with shape (vocab_size, d_model); indexing appends
  d_model to the token-id shape (§3.3.3).
- RMSNorm normalizes only the final d_model axis, preserves all leading dims,
  and computes the squared mean in float32 (§3.4.1).
- No bias terms anywhere in this assignment (§3.3.2, following modern LLMs).
- Linear init: trunc_normal_(mean=0, std=sqrt(2/(d_in+d_out))), clamped to ±3σ (§3.3.1).
- Leading ("batch-like") dims are free lanes: every op accepts (..., d) inputs (§3.2).
"""

import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn


class Linear(nn.Module):
    """y = xWᵀ, no bias (§3.3.2). Weight shape (d_out, d_in), like nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Map token IDs to rows of a learned (vocab_size, d_model) table (§3.3.3)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize each final-axis vector by its RMS, then apply learned gain (§3.4.1)."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inverse_rms * self.weight).to(in_dtype)


def silu(x: Tensor) -> Tensor:
    """SiLU(x) = x * sigmoid(x), applied elementwise (§3.4.2)."""

    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along one axis (§3.4.4)."""

    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


class SwiGLU(nn.Module):
    """Position-wise gated feed-forward network (§3.4.2)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """Ungated SiLU feed-forward network for the parameter-matched §7.3 ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """Apply position-dependent pairwise rotations to query or key vectors (§3.4.3)."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.d_k = d_k
        inverse_frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, inverse_frequencies, "position, pair -> position pair")
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        positions = token_positions.to(device=self.cos.device, dtype=torch.long)
        cos = self.cos[positions].to(dtype=x.dtype)
        sin = self.sin[positions].to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(start_dim=-2)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention over arbitrary leading batch-like dimensions."""

    scores = einsum(query, key, "... query d_k, ... key d_k -> ... query key") / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = softmax(scores, dim=-1)
    return einsum(probabilities, value, "... query key, ... key d_v -> ... query d_v")


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE on Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        query = rearrange(self.q_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        key = rearrange(self.k_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        value = rearrange(self.v_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)

        sequence_length = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            # The head axis is batch-like; inserting a singleton lets positions broadcast over it.
            query = self.rope(query, token_positions.unsqueeze(-2))
            key = self.rope(key, token_positions.unsqueeze(-2))

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            device=x.device,
            dtype=torch.bool,
        ).tril()
        attended = scaled_dot_product_attention(query, key, value, causal_mask)
        attended = rearrange(attended, "... head seq d -> ... seq (head d)")
        return self.output_proj(attended)


class TransformerBlock(nn.Module):
    """Decoder block supporting the §7.3 pre/post/no-normalization variants."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        normalization: str = "pre",
        positional_encoding: str = "rope",
        feed_forward: str = "swiglu",
    ) -> None:
        super().__init__()
        if normalization not in {"pre", "post", "none"}:
            raise ValueError("normalization must be 'pre', 'post', or 'none'")
        if positional_encoding not in {"rope", "none"}:
            raise ValueError("positional_encoding must be 'rope' or 'none'")
        if feed_forward not in {"swiglu", "silu"}:
            raise ValueError("feed_forward must be 'swiglu' or 'silu'")
        self.normalization = normalization
        rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
            if positional_encoding == "rope"
            else None
        )
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if normalization != "none" else None
        ffn_class = SwiGLU if feed_forward == "swiglu" else SiLUFFN
        self.ffn = ffn_class(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if normalization != "none" else None

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.normalization == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))
        if self.normalization == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(x, token_positions)
        return x + self.ffn(x)


class TransformerLM(nn.Module):
    """Decoder-only Transformer LM with §7.3 normalization variants."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        normalization: str = "pre",
        positional_encoding: str = "rope",
        feed_forward: str = "swiglu",
        tie_embeddings: bool = False,
        tied_embedding_std: float = 0.02,
        embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if normalization not in {"pre", "post", "none"}:
            raise ValueError("normalization must be 'pre', 'post', or 'none'")
        if positional_encoding not in {"rope", "none"}:
            raise ValueError("positional_encoding must be 'rope' or 'none'")
        if feed_forward not in {"swiglu", "silu"}:
            raise ValueError("feed_forward must be 'swiglu' or 'silu'")
        if tied_embedding_std <= 0:
            raise ValueError("tied_embedding_std must be positive")
        if embedding_scale <= 0:
            raise ValueError("embedding_scale must be positive")
        self.context_length = context_length
        self.normalization = normalization
        self.positional_encoding = positional_encoding
        self.feed_forward = feed_forward
        self.tie_embeddings = tie_embeddings
        self.embedding_scale = embedding_scale
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                    normalization=normalization,
                    positional_encoding=positional_encoding,
                    feed_forward=feed_forward,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if normalization == "pre" else None
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        if tie_embeddings:
            # The assignment's ordinary embedding init has std=1, which would
            # produce enormous vocabulary logits if reused as the LM head.
            # Reinitialize the one shared table at the small modern-LM scale.
            nn.init.trunc_normal_(
                self.token_embeddings.weight,
                mean=0.0,
                std=tied_embedding_std,
                a=-3 * tied_embedding_std,
                b=3 * tied_embedding_std,
            )
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.lm_head(self.forward_hidden(token_ids))

    def forward_hidden(self, token_ids: Tensor) -> Tensor:
        """Return final normalized states before the vocabulary projection."""

        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.context_length}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embeddings(token_ids) * self.embedding_scale
        for layer in self.layers:
            x = layer(x, positions)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return x
