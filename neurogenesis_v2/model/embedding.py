import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding:
    """Rotary Position Embeddings (RoPE) — computed, not stored."""

    def __init__(self, d_head: int, max_seq_len: int = 2048):
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self._cache = {}

    def _get_freqs(self, seq_len: int, device: torch.device):
        key = (seq_len, device)
        if key not in self._cache:
            position = torch.arange(seq_len, device=device).unsqueeze(1).float()
            dim_idx = torch.arange(0, self.d_head, 2, device=device).float()
            freq = 1.0 / (10000.0 ** (dim_idx / self.d_head))
            angles = position * freq
            cos_vals = angles.cos()
            sin_vals = angles.sin()
            self._cache[key] = (cos_vals, sin_vals)
        return self._cache[key]

    def __call__(self, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE to query and key tensors.
        Args:
            q: (batch, n_heads, seq_len, d_head)
            k: (batch, n_heads, seq_len, d_head)
        """
        seq_len = q.shape[2]
        cos_vals, sin_vals = self._get_freqs(seq_len, q.device)

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            rotated = torch.stack([
                x1 * cos_vals - x2 * sin_vals,
                x1 * sin_vals + x2 * cos_vals,
            ], dim=-1)
            return rotated.flatten(-2)

        return rotate(q), rotate(k)


class TokenEmbedding(nn.Module):
    """Token embedding with weight tying support."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * math.sqrt(self.d_model)

    @property
    def weight(self):
        return self.embedding.weight
