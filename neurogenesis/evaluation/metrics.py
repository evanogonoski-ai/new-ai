"""Evaluation metrics for the Neurogenesis model."""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurogenesis.model.neurogenesis import NeurogenesisModel


def compute_perplexity(
    model: NeurogenesisModel,
    dataloader: DataLoader,
    device: str = 'cpu',
    max_batches: int = None,
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(tqdm(dataloader, desc="Perplexity", leave=False)):
            if max_batches and i >= max_batches:
                break
            input_ids = input_ids.to(device).clamp(0, model.config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, model.config.vocab_size - 1)

            output = model(input_ids, training=False, max_iterations=8)
            loss = F.cross_entropy(output['logits'], target_ids[:, -1])
            total_loss += loss.item() * input_ids.shape[0]
            total_tokens += input_ids.shape[0]

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 20))


def primitive_utilization_entropy(model: NeurogenesisModel) -> float:
    """
    Compute entropy of primitive usage distribution.
    High entropy = diverse usage = good.
    Low entropy = few primitives dominate = bad.
    """
    usage = model.primitive_library.usage_frequency
    if usage.sum() == 0:
        return 0.0
    probs = usage / (usage.sum() + 1e-8)
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()
    return entropy


def adaptive_halting_variance(
    model: NeurogenesisModel,
    dataloader: DataLoader,
    device: str = 'cpu',
    max_batches: int = 50,
) -> dict:
    """
    Measure how much the halting iteration count varies across inputs.
    High variance = model allocates compute based on difficulty = good.
    """
    model.eval()
    model = model.to(device)
    all_iterations = []

    with torch.no_grad():
        for i, (input_ids, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device).clamp(0, model.config.vocab_size - 1)
            output = model(input_ids, training=False)
            num_iters = output['ponder_cost'].item()
            all_iterations.append(num_iters)

    if not all_iterations:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

    iters_tensor = torch.tensor(all_iterations)
    return {
        'mean': iters_tensor.mean().item(),
        'std': iters_tensor.std().item(),
        'min': iters_tensor.min().item(),
        'max': iters_tensor.max().item(),
    }


def primitive_ablation(
    model: NeurogenesisModel,
    dataloader: DataLoader,
    device: str = 'cpu',
    max_batches: int = 20,
) -> dict:
    """
    Remove each primitive one at a time and measure impact on loss.
    Returns dict mapping primitive index -> loss increase.
    """
    model.eval()
    model = model.to(device)

    # Baseline loss
    baseline_loss = _compute_avg_loss(model, dataloader, device, max_batches)

    ablation_results = {}
    num_prims = model.primitive_library.num_primitives

    for prim_idx in range(min(num_prims, 20)):  # Only test first 20 to save time
        # Temporarily zero out primitive weights
        prim = model.primitive_library.primitives[prim_idx]
        saved_state = {name: param.data.clone() for name, param in prim.named_parameters()}

        with torch.no_grad():
            for param in prim.parameters():
                param.zero_()

        ablated_loss = _compute_avg_loss(model, dataloader, device, max_batches)

        # Restore
        with torch.no_grad():
            for name, param in prim.named_parameters():
                param.data.copy_(saved_state[name])

        impact = ablated_loss - baseline_loss
        ablation_results[prim_idx] = {
            'loss_increase': impact,
            'baseline': baseline_loss,
            'ablated': ablated_loss,
        }

    return ablation_results


def _compute_avg_loss(model, dataloader, device, max_batches):
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device).clamp(0, model.config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, model.config.vocab_size - 1)
            output = model(input_ids, training=False, max_iterations=8)
            loss = F.cross_entropy(output['logits'], target_ids[:, -1])
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)
