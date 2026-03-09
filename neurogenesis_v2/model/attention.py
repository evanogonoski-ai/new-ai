import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EntropyGatedAttention(nn.Module):
    """
    Multi-head attention where each head's contribution is gated by its
    attention entropy. Focused heads (low entropy) dominate; diffuse heads
    (high entropy) are suppressed. Zero additional parameters.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 gate_floor: float = 0.0, gate_min: float = 0.0, gate_max: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.gate_floor = gate_floor
        self.gate_min = gate_min
        self.gate_max = gate_max

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_fn=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            rope_fn: optional RoPE function
        Returns:
            output: (batch, seq_len, d_model)
            gates: (batch, n_heads) entropy-based gates for logging/loss
        """
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if rope_fn is not None:
            Q, K = rope_fn(Q, K)

        # Scaled dot-product attention with causal mask
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Entropy gating
        eps = 1e-8
        entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # (B, H, S)
        entropy = entropy.mean(dim=-1)  # (B, H) mean over query positions

        max_entropy = math.log(S + eps)
        normalized_entropy = entropy / max_entropy  # (B, H) in ~[0, 1]

        # Gate: focused (low entropy) → high gate, diffuse → low gate
        raw_gates = 1.0 - normalized_entropy  # (B, H)

        # V3: Apply range compression (lateral inhibition)
        if self.gate_min > 0.0 or self.gate_max < 1.0:
            gates = self.gate_min + (self.gate_max - self.gate_min) * raw_gates
        else:
            gates = raw_gates

        # V3: Apply gate floor (biological baseline firing)
        if self.gate_floor > 0.0:
            gates = torch.clamp(gates, min=self.gate_floor)

        # Apply attention
        head_output = torch.matmul(attn_weights, V)  # (B, H, S, d_head)

        # Gate each head
        gates_expanded = gates.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        gated_output = head_output * gates_expanded

        # Concat and project
        gated_output = gated_output.transpose(1, 2).contiguous().view(B, S, -1)
        output = self.W_o(gated_output)

        return output, gates
