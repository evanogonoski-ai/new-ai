import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from neurogenesis.model.primitives import Primitive, PrimitiveLibrary


class ConsolidationEngine:
    """Compresses learned patterns into more efficient primitive representations."""

    def __init__(self, primitive_library: PrimitiveLibrary, d_model: int = 128,
                 d_inner: int = 256, max_primitives: int = 128):
        self.library = primitive_library
        self.d_model = d_model
        self.d_inner = d_inner
        self.max_primitives = max_primitives
        # Track co-occurrence counts: (i, j) -> count
        self.cooccurrence = defaultdict(int)

    def record_cooccurrence(self, selected_indices: torch.Tensor):
        """
        Record which primitives were selected together.
        Args:
            selected_indices: (batch, K) primitive indices from a single iteration
        """
        with torch.no_grad():
            for b in range(selected_indices.shape[0]):
                indices = selected_indices[b].tolist()
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pair = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                        self.cooccurrence[pair] += 1

    def find_top_pairs(self, min_count: int = 100, top_k: int = 4) -> list:
        """Find most frequently co-occurring primitive pairs."""
        sorted_pairs = sorted(self.cooccurrence.items(), key=lambda x: x[1], reverse=True)
        return [(pair, count) for pair, count in sorted_pairs[:top_k] if count >= min_count]

    def create_compound_primitive(
        self, prim_a_idx: int, prim_b_idx: int,
        sample_data: torch.Tensor, num_steps: int = 500, lr: float = 1e-3
    ) -> Primitive:
        """
        Train a new primitive to approximate prim_b(prim_a(x)) in a single step.

        Args:
            prim_a_idx: index of first primitive
            prim_b_idx: index of second primitive
            sample_data: (num_samples, d_model) training data
            num_steps: distillation training steps
            lr: learning rate
        Returns:
            compound: new Primitive module
        """
        prim_a = self.library.primitives[prim_a_idx]
        prim_b = self.library.primitives[prim_b_idx]
        compound = Primitive(self.d_model, self.d_inner)

        optimizer = torch.optim.Adam(compound.parameters(), lr=lr)

        prim_a.eval()
        prim_b.eval()
        compound.train()

        for step in range(num_steps):
            # Teacher: sequential application
            with torch.no_grad():
                intermediate, _ = prim_a(sample_data)
                target, _ = prim_b(intermediate)

            # Student: single compound primitive
            prediction, _ = compound(sample_data)
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return compound

    def run_consolidation(
        self, sample_data: torch.Tensor, min_count: int = 100,
        num_compounds: int = 2, distill_steps: int = 500
    ) -> list:
        """
        Run a full consolidation cycle.

        Returns:
            list of (pair, compound_primitive) that were created
        """
        top_pairs = self.find_top_pairs(min_count=min_count, top_k=num_compounds)
        if not top_pairs:
            return []

        created = []
        for (idx_a, idx_b), count in top_pairs:
            if self.library.num_primitives >= self.max_primitives:
                # Prune lowest-usage primitive first
                self._prune_lowest_usage()

            compound = self.create_compound_primitive(
                idx_a, idx_b, sample_data,
                num_steps=distill_steps
            )
            # Add to library
            self.library.primitives.append(compound)
            self.library.num_primitives += 1
            # Extend usage buffer
            new_usage = torch.zeros(self.library.num_primitives,
                                    device=self.library.usage_frequency.device)
            new_usage[:len(self.library.usage_frequency)] = self.library.usage_frequency
            self.library.usage_frequency = new_usage
            self.library.register_buffer('usage_frequency', new_usage)

            created.append(((idx_a, idx_b), compound))

        # Reset co-occurrence counts
        self.cooccurrence.clear()

        return created

    def _prune_lowest_usage(self):
        """Remove the primitive with lowest usage frequency."""
        if self.library.num_primitives <= 16:
            return  # Don't prune structured primitives

        # Find lowest usage among non-structured primitives
        usage = self.library.usage_frequency[16:]  # Skip structured
        min_idx = usage.argmin().item() + 16

        # Remove from module list
        del self.library.primitives[min_idx]
        self.library.num_primitives -= 1

        # Update usage buffer
        new_usage = torch.cat([
            self.library.usage_frequency[:min_idx],
            self.library.usage_frequency[min_idx + 1:]
        ])
        self.library.usage_frequency = new_usage
        self.library.register_buffer('usage_frequency', new_usage)
