import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.embedding import TokenEmbedding, RotaryPositionalEmbedding
from neurogenesis_v2.model.resonance_block import ResonanceBlock


class ResonanceModel(nn.Module):
    """
    Neurogenesis V2: Resonance Architecture.
    One shared block looped with adaptive halting via state convergence.
    """

    def __init__(self, config: ResonanceConfig = None):
        super().__init__()
        if config is None:
            config = ResonanceConfig()
        self.config = config

        # Embedding
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEmbedding(config.d_head, config.max_seq_len)

        # V3: noise injection scale
        self.noise_scale = config.noise_scale

        # The single shared resonance block
        self.block = ResonanceBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            d_ffn_max=config.d_ffn_max,
            d_ffn_start=config.d_ffn_start,
            momentum_beta_init=config.momentum_beta_init,
            gate_floor=config.gate_floor,
            gate_min=config.gate_min,
            gate_max=config.gate_max,
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output projection — weight-tied with embedding
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Multi-timescale prediction heads (training only)
        self.surface_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.semantic_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        max_iterations: int = None,
        use_momentum: bool = True,
        return_all_states: bool = False,
        force_iterations: bool = False,
    ) -> dict:
        """
        Forward pass with resonance loop.

        Args:
            input_ids: (batch, seq_len)
            max_iterations: override max loop iterations
            use_momentum: enable momentum (disabled in Phase 1)
            return_all_states: whether to return all intermediate states
            force_iterations: if True, always run exactly max_iterations (skip convergence check)
        Returns:
            dict with logits, ponder_cost, all_gates, all_states, convergence_deltas
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations

        B, S = input_ids.shape

        # Embed
        state = self.embedding(input_ids)  # (B, S, d_model)

        # V3: Noise injection during training (analogous to varied replay)
        if self.training and self.noise_scale > 0.0:
            noise = torch.randn_like(state) * self.noise_scale
            state = state + noise

        prev_state = torch.zeros_like(state)

        all_states = [state]
        all_gates = []
        convergence_deltas = []
        num_iterations = max_iterations  # default if no early halt

        for t in range(max_iterations):
            new_state, gates = self.block(
                state, prev_state, rope_fn=self.rope, use_momentum=use_momentum
            )
            all_states.append(new_state)
            all_gates.append(gates)

            # Convergence check
            delta = (new_state - state).norm() / (state.norm() + 1e-8)
            convergence_deltas.append(delta.item())

            if (not force_iterations
                    and t >= self.config.min_iterations - 1
                    and delta.item() < self.config.convergence_threshold):
                num_iterations = t + 1
                prev_state = state
                state = new_state
                break

            prev_state = state
            state = new_state

        # Final norm and projection
        normed = self.final_norm(state)
        # Predict from last token position
        last_hidden = normed[:, -1, :]  # (B, d_model)
        logits = F.linear(last_hidden, self.embedding.weight, self.output_bias)

        ponder_cost = num_iterations / max_iterations

        result = {
            'logits': logits,
            'ponder_cost': torch.tensor(ponder_cost),
            'num_iterations': num_iterations,
            'all_gates': all_gates,
            'convergence_deltas': convergence_deltas,
        }

        if return_all_states:
            result['all_states'] = all_states

        # Multi-timescale predictions
        if self.training and len(all_states) > self.config.surface_state_index:
            surface_hidden = all_states[self.config.surface_state_index][:, -1, :]
            result['surface_logits'] = self.surface_head(surface_hidden)

        if (self.training
                and len(all_states) > self.config.semantic_state_index):
            semantic_hidden = all_states[self.config.semantic_state_index][:, -1, :]
            result['semantic_logits'] = self.semantic_head(semantic_hidden)

        return result

    def count_parameters(self, include_training_only: bool = False) -> int:
        """Count parameters, optionally excluding training-only heads."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not include_training_only:
            # Subtract multi-timescale heads
            total -= sum(p.numel() for p in self.surface_head.parameters())
            total -= sum(p.numel() for p in self.semantic_head.parameters())
        return total

    def count_active_parameters(self) -> int:
        """Count only parameters for alive neurons."""
        total = self.count_parameters()
        # Subtract dormant FFN parameters
        dormant = self.config.d_ffn_max - self.block.ffn.num_alive
        dormant_params = dormant * (self.config.d_model + 1 + self.config.d_model)
        return total - dormant_params
