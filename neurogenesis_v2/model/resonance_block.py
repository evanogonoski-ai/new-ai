import torch
import torch.nn as nn

from neurogenesis_v2.model.attention import EntropyGatedAttention
from neurogenesis_v2.model.feedforward import ExpandableFFN


class MomentumResidual(nn.Module):
    """Residual connection with momentum for oscillatory dynamics."""

    def __init__(self, beta_init: float = 0.1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, state: torch.Tensor, update: torch.Tensor,
                prev_state: torch.Tensor) -> torch.Tensor:
        momentum = torch.sigmoid(self.beta) * (state - prev_state)
        return state + update + momentum


class ResonanceBlock(nn.Module):
    """
    Single shared computation block: attention + FFN + norms + momentum.
    Applied repeatedly in the resonance loop.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 d_ffn_max: int, d_ffn_start: int,
                 momentum_beta_init: float = 0.1,
                 gate_floor: float = 0.0, gate_min: float = 0.0,
                 gate_max: float = 1.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = EntropyGatedAttention(
            d_model, n_heads, d_head,
            gate_floor=gate_floor, gate_min=gate_min, gate_max=gate_max
        )
        self.residual1 = MomentumResidual(momentum_beta_init)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ExpandableFFN(d_model, d_ffn_max, d_ffn_start)
        self.residual2 = MomentumResidual(momentum_beta_init)

    def forward(self, state: torch.Tensor, prev_state: torch.Tensor,
                rope_fn=None, use_momentum: bool = True):
        """
        Args:
            state: (batch, seq_len, d_model) current state
            prev_state: (batch, seq_len, d_model) state from previous iteration
            rope_fn: optional RoPE function
            use_momentum: whether to apply momentum (disabled in Phase 1)
        Returns:
            new_state: (batch, seq_len, d_model)
            gates: (batch, n_heads) attention entropy gates
        """
        # Attention sublayer
        normed = self.norm1(state)
        attn_out, gates = self.attention(normed, rope_fn=rope_fn)
        if use_momentum:
            state = self.residual1(state, attn_out, prev_state)
        else:
            state = state + attn_out

        # FFN sublayer
        normed = self.norm2(state)
        ffn_out = self.ffn(normed)
        if use_momentum:
            state = self.residual2(state, ffn_out, prev_state)
        else:
            state = state + ffn_out

        return state, gates
