import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableFFN(nn.Module):
    """
    Feedforward network with an expandable neuron pool.
    Pre-allocated at max size but masked — only alive neurons participate.
    Supports neuronal mitosis (splitting) and pruning (death).
    """

    def __init__(self, d_model: int, d_ffn_max: int, d_ffn_start: int):
        super().__init__()
        self.d_model = d_model
        self.d_ffn_max = d_ffn_max

        self.linear1 = nn.Linear(d_model, d_ffn_max)
        self.linear2 = nn.Linear(d_ffn_max, d_model)

        # Alive mask — only first d_ffn_start neurons active
        self.register_buffer('alive_mask', torch.zeros(d_ffn_max))
        self.alive_mask[:d_ffn_start] = 1.0

        # Zero out dormant neuron weights
        with torch.no_grad():
            self.linear1.weight[d_ffn_start:] = 0
            self.linear1.bias[d_ffn_start:] = 0
            self.linear2.weight[:, d_ffn_start:] = 0

        # Track per-neuron activation magnitudes (for pruning decisions)
        self.register_buffer('activation_ema', torch.zeros(d_ffn_max))
        self.register_buffer('gradient_var_ema', torch.zeros(d_ffn_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)  # (B, S, d_ffn_max)
        h = h * self.alive_mask  # mask dormant neurons
        h = F.gelu(h)

        # Track activation magnitudes
        if self.training:
            with torch.no_grad():
                act_mag = h.abs().mean(dim=(0, 1))  # (d_ffn_max,)
                self.activation_ema = 0.99 * self.activation_ema + 0.01 * act_mag

        return self.linear2(h)

    @property
    def num_alive(self) -> int:
        return int(self.alive_mask.sum().item())

    def get_alive_indices(self) -> torch.Tensor:
        return torch.where(self.alive_mask > 0)[0]

    def get_dormant_indices(self) -> torch.Tensor:
        return torch.where(self.alive_mask == 0)[0]
