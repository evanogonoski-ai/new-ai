import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Selects and sequences primitives based on working memory state."""

    def __init__(self, d_model: int = 128, num_primitives: int = 64,
                 num_selected: int = 4, num_slots: int = 64):
        super().__init__()
        self.num_primitives = num_primitives
        self.num_selected = num_selected
        self.num_slots = num_slots

        # Primitive selection network
        self.select_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_primitives),
        )

        # Per-slot attention mask: for each selected primitive, which slots to operate on
        self.slot_attention = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_selected * num_slots),
        )

        self.temperature = 2.0  # Gumbel-Softmax temperature, annealed externally

    def forward(
        self, memory_state: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            memory_state: (batch, d_model) compressed working memory
            training: whether to use soft Gumbel-Softmax or hard argmax
        Returns:
            selected_indices: (batch, K) indices of selected primitives
            selection_weights: (batch, K) soft weights for selected primitives
            slot_masks: (batch, K, num_slots) per-primitive slot attention masks
        """
        batch_size = memory_state.shape[0]

        # Compute selection logits
        logits = self.select_net(memory_state)  # (batch, num_primitives)

        if training:
            # Gumbel-Softmax for differentiable selection
            soft_selection = F.gumbel_softmax(
                logits, tau=self.temperature, hard=False, dim=-1
            )  # (batch, num_primitives)
            # Get top-K indices and their soft weights
            topk_weights, topk_indices = soft_selection.topk(self.num_selected, dim=-1)
        else:
            # Hard selection at inference
            topk_weights, topk_indices = logits.topk(self.num_selected, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        # Compute per-slot attention masks
        slot_logits = self.slot_attention(memory_state)  # (batch, K * num_slots)
        slot_logits = slot_logits.view(batch_size, self.num_selected, self.num_slots)
        slot_masks = torch.sigmoid(slot_logits)  # (batch, K, num_slots)

        return topk_indices, topk_weights, slot_masks

    def get_entropy(self, memory_state: torch.Tensor) -> torch.Tensor:
        """Compute entropy of selection distribution for regularization."""
        logits = self.select_net(memory_state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy
