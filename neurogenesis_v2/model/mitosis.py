import torch
from neurogenesis_v2.model.feedforward import ExpandableFFN


class MitosisEngine:
    """Handles neuronal mitosis (splitting) and pruning (death)."""

    def __init__(self, ffn: ExpandableFFN, config):
        self.ffn = ffn
        self.config = config
        self.step_count = 0
        # Track gradient variance per neuron
        self.grad_var_accumulator = torch.zeros(ffn.d_ffn_max)
        self.grad_mean_accumulator = torch.zeros(ffn.d_ffn_max)
        self.grad_count = 0

    def record_gradients(self):
        """Call after backward pass to record gradient statistics."""
        if self.ffn.linear1.weight.grad is None:
            return

        with torch.no_grad():
            # Per-neuron gradient norm (along input dimension)
            grad_norms = self.ffn.linear1.weight.grad.norm(dim=1)  # (d_ffn_max,)
            self.grad_count += 1
            # Online variance tracking
            delta = grad_norms - self.grad_mean_accumulator
            self.grad_mean_accumulator += delta / self.grad_count
            delta2 = grad_norms - self.grad_mean_accumulator
            self.grad_var_accumulator += delta * delta2

    def check_and_apply(self, step: int) -> dict:
        """Check mitosis/pruning conditions and apply if needed."""
        self.step_count = step

        if step % self.config.mitosis_check_interval != 0:
            return {'mitosis': 0, 'pruned': 0, 'alive': self.ffn.num_alive}

        if self.grad_count < 10:
            return {'mitosis': 0, 'pruned': 0, 'alive': self.ffn.num_alive}

        results = {'mitosis': 0, 'pruned': 0}

        # Compute variance
        grad_variance = self.grad_var_accumulator / max(self.grad_count - 1, 1)

        # Mitosis: high gradient variance neurons should split
        alive_indices = self.ffn.get_alive_indices()
        dormant_indices = self.ffn.get_dormant_indices()

        for idx in alive_indices:
            i = idx.item()
            if (grad_variance[i] > self.config.mitosis_gradient_var_threshold
                    and len(dormant_indices) > 0
                    and self.ffn.num_alive < self.config.max_alive_neurons):
                # Split neuron
                j = dormant_indices[0].item()
                dormant_indices = dormant_indices[1:]

                with torch.no_grad():
                    noise = torch.randn_like(self.ffn.linear1.weight[i]) * 0.01
                    self.ffn.linear1.weight[j] = self.ffn.linear1.weight[i] + noise
                    self.ffn.linear1.bias[j] = self.ffn.linear1.bias[i]
                    self.ffn.linear2.weight[:, j] = self.ffn.linear2.weight[:, i] * 0.5
                    self.ffn.linear2.weight[:, i] = self.ffn.linear2.weight[:, i] * 0.5
                    self.ffn.alive_mask[j] = 1.0

                results['mitosis'] += 1

        # Pruning: low activation + low gradient neurons should die
        alive_indices = self.ffn.get_alive_indices()
        for idx in alive_indices:
            i = idx.item()
            if (self.ffn.activation_ema[i] < self.config.prune_activation_threshold
                    and self.grad_mean_accumulator[i].abs() < self.config.prune_gradient_threshold
                    and self.ffn.num_alive > self.config.min_alive_neurons):
                with torch.no_grad():
                    self.ffn.alive_mask[i] = 0.0
                    self.ffn.linear1.weight[i] = 0
                    self.ffn.linear1.bias[i] = 0
                    self.ffn.linear2.weight[:, i] = 0
                results['pruned'] += 1

        results['alive'] = self.ffn.num_alive

        # Reset accumulators
        self.grad_var_accumulator.zero_()
        self.grad_mean_accumulator.zero_()
        self.grad_count = 0

        return results
