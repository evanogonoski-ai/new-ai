import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.primitives import PrimitiveLibrary
from neurogenesis.model.router import Router
from neurogenesis.model.working_memory import WorkingMemory
from neurogenesis.model.halting import AdaptiveHalting


def apply_rotary_embeddings(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to input."""
    d = x.shape[-1]
    position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
    dim_idx = torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
    freq = 1.0 / (10000.0 ** (dim_idx / d))
    angles = position * freq  # (seq_len, d//2)

    cos_vals = angles.cos()
    sin_vals = angles.sin()

    # Apply rotation to pairs of dimensions
    x_reshape = x.view(*x.shape[:-1], -1, 2)
    x_rot = torch.stack([
        x_reshape[..., 0] * cos_vals - x_reshape[..., 1] * sin_vals,
        x_reshape[..., 0] * sin_vals + x_reshape[..., 1] * cos_vals,
    ], dim=-1)
    return x_rot.flatten(-2)


class NeurogenesisModel(nn.Module):
    """Full Neurogenesis model assembly."""

    def __init__(self, config: NeurogenesisConfig = None):
        super().__init__()
        if config is None:
            config = NeurogenesisConfig()
        self.config = config

        # Embedding (shared with output projection via weight tying)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Core components
        self.primitive_library = PrimitiveLibrary(
            num_primitives=config.num_primitives,
            num_structured=config.num_structured_primitives,
            d_model=config.d_model,
            d_inner=config.d_inner,
        )
        self.router = Router(
            d_model=config.d_model,
            num_primitives=config.num_primitives,
            num_selected=config.num_selected_primitives,
            num_slots=config.num_memory_slots,
        )
        self.working_memory = WorkingMemory(
            num_slots=config.num_memory_slots,
            d_model=config.d_model,
        )
        self.halting = AdaptiveHalting(
            d_model=config.d_model,
            max_iterations=config.max_iterations,
            min_iterations=config.min_iterations,
            epsilon=config.halt_epsilon,
        )

        # Attention pooling: learned query vector attends over memory slots
        self.pool_query = nn.Parameter(torch.randn(config.d_model))

        # Output projection (weight-tied with embedding)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Multi-timescale prediction heads (training only)
        self.surface_head = nn.Linear(config.d_model, config.vocab_size)
        self.syntactic_head = nn.Linear(config.d_model, config.vocab_size)
        self.semantic_head = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def _attention_pool(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Attention-pool working memory into a single vector.
        Args:
            memory: (batch, num_slots, d_model)
        Returns:
            pooled: (batch, d_model)
        """
        # Query attends over all slots
        query = self.pool_query.unsqueeze(0).unsqueeze(1)  # (1, 1, d_model)
        scores = (query * memory).sum(dim=-1) / math.sqrt(self.config.d_model)
        weights = F.softmax(scores, dim=-1)  # (batch, num_slots)
        pooled = (weights.unsqueeze(-1) * memory).sum(dim=1)  # (batch, d_model)
        return pooled

    def _project_to_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """Weight-tied output projection."""
        return F.linear(pooled, self.embedding.weight, self.output_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        training: bool = True,
        max_iterations: int = None,
    ) -> dict:
        """
        Full forward pass — vectorized for performance.

        During training: uses soft-weighted combination of top-K primitives
        (no per-batch-item loop). During inference: applies top-1 primitive
        per position for efficiency.
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations

        batch_size, seq_len = input_ids.shape

        # Embed tokens with RoPE
        token_emb = self.embedding(input_ids)
        token_emb = apply_rotary_embeddings(token_emb, seq_len)

        # Initialize working memory
        memory = self.working_memory.initialize(token_emb)

        # Recurrent processing loop
        cumulative_prob = torch.zeros(batch_size, 1, device=input_ids.device)
        halt_probs = []
        iteration_outputs = []
        selected_primitives_list = []
        total_iterations = torch.zeros(batch_size, device=input_ids.device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        weighted_output = torch.zeros(batch_size, self.config.d_model, device=input_ids.device)

        for t in range(max_iterations):
            if not active_mask.any():
                break

            # Router selects primitives
            compressed = self.working_memory.read_compressed(memory)
            indices, weights, slot_masks = self.router(compressed, training=training)
            selected_primitives_list.append(indices)

            # Apply primitives — vectorized using soft weighting
            # Apply all unique selected primitives to entire batch with soft weights
            # Same logic for both training and inference to avoid mismatch
            for k in range(self.config.num_selected_primitives):
                mask_k = slot_masks[:, k, :]  # (batch, num_slots)

                unique_prims = indices[:, k].unique()
                blended_out = torch.zeros_like(memory)
                blended_gate = torch.zeros(batch_size, self.config.num_memory_slots, 1,
                                           device=memory.device)

                for prim_idx in unique_prims:
                    prim_idx_val = prim_idx.item()
                    out, gate = self.primitive_library(prim_idx_val, memory)
                    prim_mask = (indices[:, k] == prim_idx).float()
                    w = (prim_mask * weights[:, k]).unsqueeze(-1).unsqueeze(-1)
                    blended_out = blended_out + w * out
                    blended_gate = blended_gate + w * gate

                total_w = weights[:, k].unsqueeze(-1).unsqueeze(-1) + 1e-8
                blended_out = blended_out / total_w
                blended_gate = (blended_gate / total_w).clamp(0, 1)

                memory = self.working_memory.apply_gated_write(
                    memory, mask_k, blended_out, blended_gate
                )

            # Store per-iteration output
            pooled = self._attention_pool(memory)
            iteration_outputs.append(pooled)

            # Check halting
            compressed = self.working_memory.read_compressed(memory)
            halt_prob, cumulative_prob, should_halt = self.halting(
                compressed, t, cumulative_prob
            )
            halt_probs.append(halt_prob)

            # Accumulate weighted output (ACT-style)
            weighted_output += halt_prob.squeeze(-1).unsqueeze(-1) * pooled * active_mask.float().unsqueeze(-1)
            total_iterations += active_mask.float()

            # Update active mask
            active_mask = active_mask & ~should_halt

        # Remainder for ACT
        remainder = 1.0 - cumulative_prob.squeeze(-1)
        final_pooled = self._attention_pool(memory)
        weighted_output += remainder.unsqueeze(-1) * final_pooled

        # Project to logits
        logits = self._project_to_logits(weighted_output)

        # Compute ponder cost
        ponder_cost = total_iterations.mean()

        result = {
            'logits': logits,
            'ponder_cost': ponder_cost,
            'halt_probs': halt_probs,
            'iteration_outputs': iteration_outputs,
            'selected_primitives': selected_primitives_list,
        }

        # Multi-timescale predictions during training
        if training and len(iteration_outputs) >= 3:
            result['surface_logits'] = self.surface_head(iteration_outputs[0])
            mid_idx = len(iteration_outputs) // 2
            result['syntactic_logits'] = self.syntactic_head(iteration_outputs[mid_idx])
            result['semantic_logits'] = self.semantic_head(iteration_outputs[-1])

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
