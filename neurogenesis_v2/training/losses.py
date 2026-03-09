import torch
import torch.nn.functional as F


def gate_sparsity_loss(gates: torch.Tensor, lambda_gs: float = 0.01) -> torch.Tensor:
    """
    Encourage gates to be bimodal (near 0 or near 1).
    Minimizing binary entropy of gates pushes them toward extremes.
    """
    eps = 1e-8
    # Binary entropy of each gate value
    h = -(gates * torch.log(gates + eps) + (1 - gates) * torch.log(1 - gates + eps))
    return lambda_gs * h.mean()


def ponder_cost_loss(ponder_cost: torch.Tensor, lambda_p: float = 0.01) -> torch.Tensor:
    return lambda_p * ponder_cost


def multitimescale_loss(output: dict, targets: torch.Tensor,
                        surface_weight: float = 0.3,
                        semantic_weight: float = 0.1) -> torch.Tensor:
    loss = torch.tensor(0.0, device=targets.device)
    if 'surface_logits' in output:
        loss = loss + surface_weight * F.cross_entropy(output['surface_logits'], targets)
    if 'semantic_logits' in output:
        loss = loss + semantic_weight * F.cross_entropy(output['semantic_logits'], targets)
    return loss
