"""Unified training script that handles all phases in a practical way."""
import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.model.primitives import PrimitiveLibrary
from neurogenesis.training.data import (
    SyntheticTransformDataset, TextDataset, generate_synthetic_stories
)
from neurogenesis.training.losses import homeostatic_loss, ponder_cost_loss
from neurogenesis.training.hebbian import hebbian_update
from neurogenesis.tokenizer.bpe import train_tokenizer, load_tokenizer


def run_phase1(config, device='cpu', num_steps=1000):
    """Phase 1: Bootstrap primitives on synthetic tasks."""
    print("=" * 60)
    print("PHASE 1: Primitive Bootstrapping")
    print("=" * 60)

    library = PrimitiveLibrary(
        num_primitives=config.num_primitives,
        num_structured=config.num_structured_primitives,
        d_model=config.d_model,
        d_inner=config.d_inner,
    ).to(device)

    dataset = SyntheticTransformDataset(d_model=config.d_model, num_samples=max(num_steps * 4, 10000))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(library.parameters(), lr=config.phase1_lr)

    step = 0
    losses = []
    for inputs, targets, task_indices in loader:
        if step >= num_steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for prim_idx in range(min(config.num_structured_primitives, len(dataset.TASK_TYPES))):
            mask = task_indices == prim_idx
            if not mask.any():
                continue
            output, _ = library(prim_idx, inputs[mask])
            loss = F.mse_loss(output, targets[mask])
            total_loss = total_loss + loss
            count += 1

        if count > 0:
            avg_loss = total_loss / count
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(library.parameters(), 1.0)
            optimizer.step()
            losses.append(avg_loss.item())

        step += 1
        if step % 200 == 0:
            avg = sum(losses[-200:]) / min(200, len(losses))
            print(f"  Step {step}/{num_steps}: loss={avg:.4f}")

    print(f"Phase 1 done: final loss={losses[-1]:.4f}")
    return library


def ensure_tokenizer(path='data/tokenizer.json'):
    """Create or load tokenizer."""
    if os.path.exists(path):
        return load_tokenizer(path)

    print("Training tokenizer on synthetic stories...")
    stories = generate_synthetic_stories(50000)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok = train_tokenizer(stories, vocab_size=8192, save_path=path)
    print(f"Tokenizer vocab: {tok.get_vocab_size()}")
    return tok


def prepare_text_data(tokenizer, seq_len=32, num_stories=50000):
    """Prepare text dataset."""
    stories = generate_synthetic_stories(num_stories)
    all_ids = []
    for story in stories:
        encoded = tokenizer.encode(story)
        all_ids.extend(encoded.ids)
    print(f"Text data: {len(all_ids)} tokens, {len(all_ids) // seq_len} sequences")
    return TextDataset(all_ids, seq_len=seq_len)


def run_phase2_and_3(
    model, config, tokenizer, device='cpu',
    phase2_steps=3000, phase3_steps=5000, seq_len=32
):
    """
    Combined Phase 2+3: Train entire model end-to-end.

    Phase 2 was originally router-only, but this doesn't work because
    the frozen embedding can't produce useful output. Instead, we do:
    - Phase 2: Train embedding + router + output projection (primitives frozen)
    - Phase 3: Unfreeze all, add auxiliary losses
    """
    vocab_size = config.vocab_size

    # Prepare data
    dataset = prepare_text_data(tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # ===== Phase 2: Train embedding + router (primitives frozen) =====
    print("\n" + "=" * 60)
    print("PHASE 2: Embedding + Router Training")
    print("=" * 60)

    # Freeze only primitives
    for name, param in model.named_parameters():
        if 'primitive_library' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.router.temperature = config.gumbel_temperature_start
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.phase2_lr)

    model.train()
    step = 0
    losses = []
    start = time.time()

    for input_ids, target_ids in loader:
        if step >= phase2_steps:
            break
        input_ids = input_ids.clamp(0, vocab_size - 1)
        target_ids = target_ids.clamp(0, vocab_size - 1)

        output = model(input_ids, training=True, max_iterations=4)
        loss = F.cross_entropy(output['logits'], target_ids[:, -1])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        losses.append(loss.item())
        step += 1
        if step % 500 == 0:
            avg = sum(losses[-500:]) / min(500, len(losses))
            ppl = math.exp(min(avg, 20))
            speed = step / (time.time() - start)
            print(f"  Step {step}/{phase2_steps}: loss={avg:.3f}, ppl={ppl:.1f}, {speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    print(f"Phase 2 done: loss={avg:.3f}, ppl={ppl:.1f}")

    # ===== Phase 3: Joint training =====
    print("\n" + "=" * 60)
    print("PHASE 3: Joint Training (all components)")
    print("=" * 60)

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config.phase3_lr)
    running_loss = None
    step = 0
    losses = []
    start = time.time()

    for input_ids, target_ids in loader:
        if step >= phase3_steps:
            break
        input_ids = input_ids.clamp(0, vocab_size - 1)
        target_ids = target_ids.clamp(0, vocab_size - 1)

        # Anneal temperature
        progress = step / phase3_steps
        temp = config.gumbel_temperature_end + 0.5 * (
            config.gumbel_temperature_start - config.gumbel_temperature_end
        ) * (1 + math.cos(math.pi * progress))
        model.router.temperature = temp

        # Forward
        output = model(input_ids, training=True, max_iterations=6)

        # Losses
        ce_loss = F.cross_entropy(output['logits'], target_ids[:, -1])
        ponder = ponder_cost_loss(output['ponder_cost'], config.ponder_cost_lambda)
        homeo = homeostatic_loss(model.primitive_library, config.homeostatic_lambda_start)
        total_loss = ce_loss + ponder + homeo

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Hebbian
        current = ce_loss.item()
        if running_loss is None:
            running_loss = current
        else:
            running_loss = 0.99 * running_loss + 0.01 * current
        reward = max(-1, min(1, (running_loss - current) / (running_loss + 1e-8)))

        if output['selected_primitives'] and abs(reward) > 0.01:
            compressed = model.working_memory.read_compressed(
                model.working_memory.initialize(model.embedding(input_ids))
            ).detach()
            hebbian_update(
                model.primitive_library,
                output['selected_primitives'][0],
                compressed,
                output['iteration_outputs'][0].detach() if output['iteration_outputs'] else compressed,
                reward, config.hebbian_lr,
            )

        # Usage tracking
        for sel in output['selected_primitives']:
            model.primitive_library.update_usage(sel, config.usage_ema_decay)

        losses.append(current)
        step += 1
        if step % 500 == 0:
            avg = sum(losses[-500:]) / min(500, len(losses))
            ppl = math.exp(min(avg, 20))
            speed = step / (time.time() - start)
            usage_ent = _usage_entropy(model.primitive_library)
            print(f"  Step {step}/{phase3_steps}: loss={avg:.3f}, ppl={ppl:.1f}, "
                  f"temp={temp:.2f}, ent={usage_ent:.2f}, {speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    print(f"Phase 3 done: loss={avg:.3f}, ppl={ppl:.1f}")

    return model


def _usage_entropy(library):
    usage = library.usage_frequency
    if usage.sum() == 0:
        return 0.0
    probs = usage / (usage.sum() + 1e-8)
    probs = probs[probs > 0]
    return -(probs * probs.log()).sum().item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1-steps', type=int, default=1000)
    parser.add_argument('--phase2-steps', type=int, default=3000)
    parser.add_argument('--phase3-steps', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    config = NeurogenesisConfig()

    # Tokenizer
    tokenizer = ensure_tokenizer()
    actual_vocab = tokenizer.get_vocab_size()
    config.vocab_size = actual_vocab
    print(f"Vocab size: {actual_vocab}")

    # Phase 1
    library = run_phase1(config, device=args.device, num_steps=args.phase1_steps)
    torch.save(library.state_dict(), os.path.join(args.checkpoint_dir, 'phase1_library.pt'))

    # Build full model
    model = NeurogenesisModel(config)
    model.primitive_library.load_state_dict(library.state_dict())
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Phase 2 + 3
    model = run_phase2_and_3(
        model, config, tokenizer, device=args.device,
        phase2_steps=args.phase2_steps,
        phase3_steps=args.phase3_steps,
    )

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'model_final.pt')
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'd_inner': config.d_inner,
            'num_primitives': config.num_primitives,
        },
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    # Quick generation test
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    from neurogenesis.inference.generate import generate
    model.eval()
    for prompt in ["Once upon a time", "The little", "A happy"]:
        text = generate(model, tokenizer, prompt, max_new_tokens=50,
                       temperature=0.8, top_k=20, device=args.device)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text[:200]}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
