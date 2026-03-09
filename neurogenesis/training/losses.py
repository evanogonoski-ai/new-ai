import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogenesis.model.primitives import PrimitiveLibrary


def homeostatic_loss(
    primitive_library: PrimitiveLibrary,
    lambda_h: float = 0.001
) -> torch.Tensor:
    """
    Penalize weight norms of underused primitives.
    Frequently used primitives keep large weights; unused ones decay.
    """
    loss = torch.tensor(0.0, device=primitive_library.usage_frequency.device)
    usage = primitive_library.usage_frequency
    # Normalize usage to [0, 1]
    if usage.max() > 0:
        norm_usage = usage / (usage.max() + 1e-8)
    else:
        norm_usage = usage

    for i, prim in enumerate(primitive_library.primitives):
        if i >= len(norm_usage):
            break
        weight_norm = sum(p.norm() for p in prim.parameters())
        # Higher penalty for less-used primitives
        penalty = (1.0 - norm_usage[i]) * weight_norm
        loss = loss + penalty

    return lambda_h * loss


def ponder_cost_loss(ponder_cost: torch.Tensor, lambda_p: float = 0.01) -> torch.Tensor:
    """Penalize excessive computation (too many loop iterations)."""
    return lambda_p * ponder_cost


def entropy_bonus(router, memory_state: torch.Tensor, lambda_e: float = 0.1) -> torch.Tensor:
    """Encourage diverse primitive selection (prevent mode collapse)."""
    return -lambda_e * router.get_entropy(memory_state)


def diversity_loss(memory: torch.Tensor, lambda_d: float = 0.01) -> torch.Tensor:
    """
    Penalize working memory slots collapsing to same representation.
    Args:
        memory: (batch, num_slots, d_model)
    """
    # Normalize slots
    normed = F.normalize(memory, dim=-1)
    # Pairwise cosine similarity
    sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # (batch, slots, slots)
    # Mask diagonal
    eye = torch.eye(sim_matrix.shape[1], device=sim_matrix.device).unsqueeze(0)
    sim_matrix = sim_matrix * (1 - eye)
    # Mean similarity (we want this to be low)
    mean_sim = sim_matrix.abs().mean()
    return lambda_d * mean_sim


def multitimescale_loss(
    model_output: dict,
    targets: torch.Tensor,
    surface_weight: float = 0.3,
    syntactic_weight: float = 0.1,
    semantic_weight: float = 0.05,
) -> torch.Tensor:
    """
    Multi-timescale prediction losses from different loop iterations.

    Args:
        model_output: dict from model forward pass
        targets: (batch,) target token ids
    """
    loss = torch.tensor(0.0, device=targets.device)

    if 'surface_logits' in model_output:
        loss = loss + surface_weight * F.cross_entropy(
            model_output['surface_logits'], targets
        )
    if 'syntactic_logits' in model_output:
        loss = loss + syntactic_weight * F.cross_entropy(
            model_output['syntactic_logits'], targets
        )
    if 'semantic_logits' in model_output:
        loss = loss + semantic_weight * F.cross_entropy(
            model_output['semantic_logits'], targets
        )

    return loss
