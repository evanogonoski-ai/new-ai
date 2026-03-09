import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemory(nn.Module):
    """Fixed-size differentiable buffer for maintaining computation state."""

    def __init__(self, num_slots: int = 64, d_model: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        # Compression layer for sequences longer than num_slots
        self.compression = nn.Linear(d_model, d_model)
        # EMA decay for slot 0 during generation
        self.ema_decay = 0.9

    def initialize(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Initialize working memory from token embeddings.

        Args:
            token_embeddings: (batch, seq_len, d_model)
        Returns:
            memory: (batch, num_slots, d_model)
        """
        batch_size, seq_len, d_model = token_embeddings.shape

        if seq_len <= self.num_slots:
            # Pad with zeros
            memory = torch.zeros(
                batch_size, self.num_slots, d_model,
                device=token_embeddings.device, dtype=token_embeddings.dtype
            )
            memory[:, :seq_len, :] = token_embeddings
        else:
            # Compress: take evenly spaced tokens and compress via learned layer
            # Use strided selection + compression
            indices = torch.linspace(0, seq_len - 1, self.num_slots).long()
            selected = token_embeddings[:, indices, :]
            memory = self.compression(selected)

        return memory

    def apply_gated_write(
        self,
        memory: torch.Tensor,
        slot_mask: torch.Tensor,
        new_values: torch.Tensor,
        gates: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gated write to memory slots.

        Args:
            memory: (batch, num_slots, d_model)
            slot_mask: (batch, num_slots) soft attention mask for which slots to write
            new_values: (batch, num_slots, d_model) proposed new values
            gates: (batch, num_slots, 1) write gate values
        Returns:
            updated memory: (batch, num_slots, d_model)
        """
        # Combine gate with slot mask
        effective_gate = gates * slot_mask.unsqueeze(-1)
        updated = effective_gate * new_values + (1 - effective_gate) * memory
        return updated

    def read_compressed(self, memory: torch.Tensor) -> torch.Tensor:
        """Mean-pool across slots for router input."""
        return memory.mean(dim=1)  # (batch, d_model)

    def update_for_generation(
        self, memory: torch.Tensor, new_token_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding compression for autoregressive generation.
        Fold slot 1 into slot 0 via EMA, shift left, new token at last slot.

        Args:
            memory: (batch, num_slots, d_model)
            new_token_embedding: (batch, d_model)
        Returns:
            updated memory: (batch, num_slots, d_model)
        """
        # Fold slot 1 into slot 0 via EMA
        slot0 = self.ema_decay * memory[:, 0, :] + (1 - self.ema_decay) * memory[:, 1, :]

        # Shift slots left
        new_memory = torch.zeros_like(memory)
        new_memory[:, 0, :] = slot0
        new_memory[:, 1:-1, :] = memory[:, 2:, :]
        new_memory[:, -1, :] = new_token_embedding

        return new_memory
