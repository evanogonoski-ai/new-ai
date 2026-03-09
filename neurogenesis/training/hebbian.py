import torch

from neurogenesis.model.primitives import PrimitiveLibrary


def hebbian_update(
    primitive_library: PrimitiveLibrary,
    selected_indices: torch.Tensor,
    input_activations: torch.Tensor,
    output_activations: torch.Tensor,
    global_reward: float,
    hebbian_lr: float = 1e-5,
):
    """
    Apply Hebbian local learning rule to recently-used primitives.

    global_reward: scalar in [-1, 1], derived from whether this forward pass
    had lower loss than the running average (positive = better than expected).
    This modulates local correlation-based learning with a global signal,
    analogous to dopaminergic modulation in the brain.

    Args:
        primitive_library: the primitive library
        selected_indices: (batch, K) indices of selected primitives
        input_activations: (batch, d_model) mean input to primitives
        output_activations: (batch, d_model) mean output from primitives
        global_reward: scalar reward signal
        hebbian_lr: learning rate for Hebbian updates
    """
    if abs(global_reward) < 1e-8:
        return

    with torch.no_grad():
        # Compute correlation between mean output and mean input
        inp = input_activations.mean(dim=0)  # (d_model,)
        out = output_activations.mean(dim=0)  # (d_model,)

        correlation = torch.outer(out, inp)  # (d_model, d_model)
        update_magnitude = global_reward * hebbian_lr

        # Apply to each selected primitive's first linear layer
        unique_indices = selected_indices.unique()
        for idx in unique_indices:
            idx_val = idx.item()
            if 0 <= idx_val < primitive_library.num_primitives:
                prim = primitive_library.primitives[idx_val]
                w = prim.linear1.weight  # (d_inner, d_model)
                d_inner, d_model = w.shape
                # Correlation is (d_model, d_model), need (d_inner, d_model)
                # Project correlation into the weight space by repeating/tiling
                update = update_magnitude * correlation[:min(d_inner, correlation.shape[0]), :d_model]
                # Pad if d_inner > d_model
                if d_inner > correlation.shape[0]:
                    pad = torch.zeros(d_inner - correlation.shape[0], d_model,
                                     device=w.device)
                    update = torch.cat([update, pad], dim=0)
                update = update.clamp(-0.01, 0.01)
                w.data += update
