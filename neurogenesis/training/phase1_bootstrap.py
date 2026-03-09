"""Phase 1: Bootstrap primitives on synthetic transformation tasks."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.primitives import PrimitiveLibrary
from neurogenesis.training.data import SyntheticTransformDataset


def train_phase1(
    config: NeurogenesisConfig,
    num_steps: int = None,
    device: str = 'cpu',
    checkpoint_path: str = None,
) -> PrimitiveLibrary:
    """
    Train primitives independently on synthetic transformation tasks.
    Each primitive is trained to perform a specific transformation.
    """
    if num_steps is None:
        num_steps = config.phase1_steps

    library = PrimitiveLibrary(
        num_primitives=config.num_primitives,
        num_structured=config.num_structured_primitives,
        d_model=config.d_model,
        d_inner=config.d_inner,
    ).to(device)

    dataset = SyntheticTransformDataset(d_model=config.d_model, num_samples=max(num_steps * 4, 10000))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Each structured primitive gets trained on its matching task
    # Map primitive index -> task index
    task_map = {i: i for i in range(min(config.num_structured_primitives, len(dataset.TASK_TYPES)))}

    optimizer = torch.optim.Adam(library.parameters(), lr=config.phase1_lr)

    step = 0
    epoch = 0
    losses_history = []

    print(f"Phase 1: Training {config.num_primitives} primitives on synthetic tasks")
    print(f"  Steps: {num_steps}, LR: {config.phase1_lr}")

    while step < num_steps:
        epoch += 1
        pbar = tqdm(loader, desc=f"Phase 1 Epoch {epoch}", leave=False)
        for inputs, targets, task_indices in pbar:
            if step >= num_steps:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            task_indices = task_indices.to(device)

            total_loss = torch.tensor(0.0, device=device)
            count = 0

            # Train each primitive on examples matching its task
            for prim_idx in range(min(config.num_structured_primitives, len(dataset.TASK_TYPES))):
                mask = task_indices == prim_idx
                if not mask.any():
                    continue

                prim_inputs = inputs[mask]
                prim_targets = targets[mask]

                output, _ = library(prim_idx, prim_inputs)
                loss = F.mse_loss(output, prim_targets)
                total_loss = total_loss + loss
                count += 1

            # Also train random primitives on random tasks (general capability)
            for prim_idx in range(config.num_structured_primitives, config.num_primitives):
                # Pick random subset
                sample_idx = torch.randint(0, len(inputs), (min(4, len(inputs)),))
                prim_inputs = inputs[sample_idx]
                prim_targets = targets[sample_idx]

                output, _ = library(prim_idx, prim_inputs)
                loss = F.mse_loss(output, prim_targets) * 0.1  # Lower weight for random prims
                total_loss = total_loss + loss
                count += 1

            if count > 0:
                avg_loss = total_loss / count
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(library.parameters(), 1.0)
                optimizer.step()

                losses_history.append(avg_loss.item())
                pbar.set_postfix(loss=f"{avg_loss.item():.4f}", step=step)

            step += 1

    # Report results
    print(f"\nPhase 1 Complete!")
    print(f"  Final loss: {losses_history[-1]:.4f}")
    print(f"  Min loss: {min(losses_history):.4f}")

    # Test structured primitives
    print("\nStructured primitive MSE on target tasks:")
    test_data = SyntheticTransformDataset(d_model=config.d_model, num_samples=1000, seed=999)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    library.eval()
    with torch.no_grad():
        for inputs, targets, task_indices in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            for prim_idx in range(min(config.num_structured_primitives, len(dataset.TASK_TYPES))):
                mask = task_indices == prim_idx
                if not mask.any():
                    continue
                output, _ = library(prim_idx, inputs[mask])
                mse = F.mse_loss(output, targets[mask]).item()
                task_name = dataset.TASK_TYPES[prim_idx]
                status = "OK" if mse < 0.1 else "NEEDS WORK"
                print(f"  [{status}] Primitive {prim_idx} ({task_name}): MSE={mse:.4f}")
            break  # One batch is enough

    if checkpoint_path:
        torch.save(library.state_dict(), checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")

    return library
