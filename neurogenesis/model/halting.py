import torch
import torch.nn as nn


class AdaptiveHalting(nn.Module):
    """Adaptive computation time: decides when to stop iterating."""

    def __init__(self, d_model: int = 128, max_iterations: int = 16,
                 min_iterations: int = 2, epsilon: float = 0.01):
        super().__init__()
        self.confidence = nn.Linear(d_model, 1)
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.epsilon = epsilon

    def forward(self, memory_state: torch.Tensor, iteration: int,
                cumulative_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            memory_state: (batch, d_model) compressed working memory
            iteration: current iteration number (0-indexed)
            cumulative_prob: (batch, 1) accumulated halt probability so far
        Returns:
            halt_prob: (batch, 1) halt probability for this step
            new_cumulative: (batch, 1) updated cumulative probability
            should_halt: (batch,) boolean mask
        """
        halt_prob = torch.sigmoid(self.confidence(memory_state))  # (batch, 1)

        # Force continuation if below minimum iterations
        if iteration < self.min_iterations:
            should_halt = torch.zeros(memory_state.shape[0], dtype=torch.bool,
                                      device=memory_state.device)
            # Still accumulate probability for gradient flow
            new_cumulative = cumulative_prob + halt_prob
            return halt_prob, new_cumulative, should_halt

        new_cumulative = cumulative_prob + halt_prob

        # Halt when cumulative probability exceeds threshold
        should_halt = (new_cumulative.squeeze(-1) >= (1.0 - self.epsilon))

        # Force halt at max iterations
        if iteration >= self.max_iterations - 1:
            should_halt = torch.ones_like(should_halt)

        return halt_prob, new_cumulative, should_halt
