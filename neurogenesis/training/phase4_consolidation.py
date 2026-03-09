"""Phase 4: Consolidation — compress learned patterns into compound primitives."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.model.consolidation import ConsolidationEngine
from neurogenesis.training.data import load_tinystories
from neurogenesis.tokenizer.bpe import load_tokenizer


def train_phase4(
    model: NeurogenesisModel,
    config: NeurogenesisConfig,
    device: str = 'cpu',
    checkpoint_path: str = None,
    tokenizer_path: str = None,
    num_record_steps: int = 1000,
    router_realign_steps: int = 500,
) -> NeurogenesisModel:
    """
    Run consolidation: record co-occurrences, create compounds, retrain router.
    """
    model = model.to(device)

    engine = ConsolidationEngine(
        model.primitive_library,
        d_model=config.d_model,
        d_inner=config.d_inner,
        max_primitives=config.max_primitives,
    )

    tokenizer = load_tokenizer(tokenizer_path)
    dataset = load_tinystories(tokenizer, seq_len=64, max_samples=20000)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Step 1: Record co-occurrences
    print("Phase 4: Recording primitive co-occurrences...")
    model.eval()
    step = 0
    with torch.no_grad():
        for input_ids, _ in tqdm(loader, desc="Recording", leave=False):
            if step >= num_record_steps:
                break
            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            output = model(input_ids, training=False, max_iterations=8)
            for sel in output['selected_primitives']:
                engine.record_cooccurrence(sel)
            step += 1

    # Step 2: Find top pairs and create compounds
    print("Finding frequently co-occurring primitive pairs...")
    top_pairs = engine.find_top_pairs(min_count=50, top_k=4)
    if not top_pairs:
        print("  No frequently co-occurring pairs found. Skipping consolidation.")
        return model

    print(f"  Found {len(top_pairs)} candidate pairs:")
    for (a, b), count in top_pairs:
        print(f"    Primitives ({a}, {b}): co-occurred {count} times")

    # Generate sample data for distillation
    sample_data = torch.randn(500, config.d_model, device=device)

    print("Creating compound primitives...")
    created = engine.run_consolidation(
        sample_data, min_count=50, num_compounds=min(2, len(top_pairs)),
        distill_steps=300
    )
    print(f"  Created {len(created)} compound primitives")
    print(f"  Library size: {model.primitive_library.num_primitives}")

    # Step 3: Router re-alignment
    if created:
        print("Re-aligning router to new primitives...")
        # Freeze everything except router
        for name, param in model.named_parameters():
            if 'router' not in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(model.router.parameters(), lr=config.phase2_lr)
        model.train()

        step = 0
        for input_ids, target_ids in tqdm(loader, desc="Re-alignment", leave=False):
            if step >= router_realign_steps:
                break
            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

            output = model(input_ids, training=True, max_iterations=4)
            loss = torch.nn.functional.cross_entropy(output['logits'], target_ids[:, -1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True

    print("Phase 4 Complete!")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

    return model
