import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Primitive(nn.Module):
    """A single computational primitive: 2-layer MLP with residual + gate."""

    def __init__(self, d_model: int = 128, d_inner: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.gate_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, slots, d_model) or (batch, d_model)
        Returns:
            output: same shape as x (residual applied)
            gate: (batch, slots, 1) or (batch, 1) sigmoid gate values
        """
        residual = x
        h = F.gelu(self.linear1(x))
        out = self.linear2(h) + residual
        gate = torch.sigmoid(self.gate_proj(x))
        return out, gate


class PrimitiveLibrary(nn.Module):
    """Collection of composable computational primitives."""

    def __init__(self, num_primitives: int = 64, num_structured: int = 16,
                 d_model: int = 128, d_inner: int = 256):
        super().__init__()
        self.num_primitives = num_primitives
        self.d_model = d_model
        self.primitives = nn.ModuleList([
            Primitive(d_model, d_inner) for _ in range(num_primitives)
        ])
        # Usage tracking (not a parameter, but persisted in state dict)
        self.register_buffer(
            'usage_frequency',
            torch.zeros(num_primitives)
        )
        # Initialize structured primitives
        self._init_structured(num_structured)

    def _init_structured(self, num_structured: int):
        """Initialize first num_structured primitives with useful functions.

        linear1 shape: (d_inner, d_model) e.g. (256, 128)
        linear2 shape: (d_model, d_inner) e.g. (128, 256)
        The MLP output is: linear2(GELU(linear1(x))) + x (residual)
        So the MLP branch computes the *delta* added to x.
        """
        if num_structured > self.num_primitives:
            num_structured = self.num_primitives

        d = self.d_model

        with torch.no_grad():
            for i in range(num_structured):
                prim = self.primitives[i]
                d_inner = prim.linear1.out_features
                # Reset to near-zero first
                nn.init.zeros_(prim.linear1.weight)
                nn.init.zeros_(prim.linear1.bias)
                nn.init.zeros_(prim.linear2.weight)
                nn.init.zeros_(prim.linear2.bias)

                if i == 0:
                    # Identity: MLP branch is zero, residual passes through
                    pass
                elif i == 1:
                    # Negation: output = -x, so delta = -2x
                    # linear1[:d, :d] = I (pass-through to first d dims of inner)
                    # linear2[:d, :d] = -2*I (scale and negate)
                    eye_d = torch.eye(d)
                    prim.linear1.weight[:d, :d].copy_(eye_d * 0.5)
                    prim.linear2.weight[:d, :d].copy_(-2.0 * eye_d * 0.5)
                elif i == 2:
                    # Shift (rotate dimensions by 1): delta = roll(x,1) - x
                    eye_d = torch.eye(d)
                    prim.linear1.weight[:d, :d].copy_(eye_d * 0.5)
                    shift_mat = torch.zeros(d, d)
                    for j in range(d):
                        shift_mat[j, (j + 1) % d] = 1.0
                    prim.linear2.weight[:d, :d].copy_((shift_mat - eye_d) * 0.5)
                elif i == 3:
                    # Scale by 0.5: output = 0.5*x, so delta = -0.5*x
                    eye_d = torch.eye(d)
                    prim.linear1.weight[:d, :d].copy_(eye_d * 0.5)
                    prim.linear2.weight[:d, :d].copy_(-0.5 * eye_d * 0.5)
                elif i == 4:
                    # Normalize-like (subtract mean)
                    prim.linear1.weight[:1, :].fill_(1.0 / d)
                elif i == 5:
                    # Amplify: output ≈ 2*x, so delta ≈ x
                    eye_d = torch.eye(d)
                    prim.linear1.weight[:d, :d].copy_(eye_d * 0.5)
                    prim.linear2.weight[:d, :d].copy_(eye_d * 0.5)
                else:
                    # Primitives 6-15: small random for organic specialization
                    nn.init.xavier_uniform_(prim.linear1.weight, gain=0.1)
                    nn.init.xavier_uniform_(prim.linear2.weight, gain=0.1)

    def forward(self, idx: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply primitive at index idx to input x."""
        return self.primitives[idx](x)

    def update_usage(self, selected_indices: torch.Tensor, decay: float = 0.999):
        """Update usage frequency with EMA."""
        with torch.no_grad():
            self.usage_frequency *= decay
            for idx in selected_indices.flatten():
                if 0 <= idx < self.num_primitives:
                    self.usage_frequency[idx] += (1 - decay)
