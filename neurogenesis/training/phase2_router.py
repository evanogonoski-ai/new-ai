"""Phase 2: Train router on TinyStories for next-token prediction."""
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.training.data import load_tinystories, load_tinystories_validation
from neurogenesis.tokenizer.bpe import load_tokenizer, train_tokenizer


def train_phase2(
    model: NeurogenesisModel,
    config: NeurogenesisConfig,
    num_steps: int = None,
    seq_len: int = 32,
    device: str = 'cpu',
    checkpoint_path: str = None,
    tokenizer_path: str = None,
) -> NeurogenesisModel:
    """
    Train router only on TinyStories next-token prediction.
    Primitives are frozen; only the router learns to compose them.
    """
    if num_steps is None:
        num_steps = config.phase2_steps

    model = model.to(device)

    # Freeze everything except router
    for name, param in model.named_parameters():
        if 'router' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Set fixed loop iterations (halting not yet trained)
    fixed_iterations = 4

    # Set high Gumbel temperature (explore freely)
    model.router.temperature = config.gumbel_temperature_start

    # Load tokenizer
    tokenizer = _ensure_tokenizer(tokenizer_path)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {actual_vocab}")

    # Load data
    train_dataset = load_tinystories(tokenizer, seq_len=seq_len, max_samples=50000)
    print(f"Training sequences: {len(train_dataset)}")

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Optimizer: only router params
    router_params = [p for p in model.router.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(router_params, lr=config.phase2_lr)

    step = 0
    epoch = 0
    losses = []
    last_checkpoint_time = time.time()

    print(f"\nPhase 2: Training router on TinyStories")
    print(f"  Steps: {num_steps}, LR: {config.phase2_lr}")
    print(f"  Fixed iterations: {fixed_iterations}, Gumbel temp: {model.router.temperature}")

    while step < num_steps:
        epoch += 1
        pbar = tqdm(loader, desc=f"Phase 2 Epoch {epoch}", leave=False)
        for input_ids, target_ids in pbar:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Clamp to vocab size
            input_ids = input_ids.clamp(0, config.vocab_size - 1)
            target_ids = target_ids.clamp(0, config.vocab_size - 1)

            # Forward pass with fixed iterations
            output = model(input_ids, training=True, max_iterations=fixed_iterations)

            # Next-token prediction loss (predict last token)
            logits = output['logits']  # (batch, vocab_size)
            targets = target_ids[:, -1]  # (batch,) last target token
            loss = F.cross_entropy(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                pbar.set_postfix(loss=f"{avg_loss:.3f}", ppl=f"{perplexity:.1f}", step=step)

            # Checkpoint
            if checkpoint_path and (time.time() - last_checkpoint_time) > config.checkpoint_interval_minutes * 60:
                torch.save(model.state_dict(), checkpoint_path)
                last_checkpoint_time = time.time()

    # Final stats
    avg_loss = sum(losses[-100:]) / min(100, len(losses))
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\nPhase 2 Complete!")
    print(f"  Final avg loss: {avg_loss:.3f}")
    print(f"  Final perplexity: {perplexity:.1f}")

    # Unfreeze all parameters for subsequent phases
    for param in model.parameters():
        param.requires_grad = True

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

    return model


def _ensure_tokenizer(path=None):
    """Load or train tokenizer."""
    from neurogenesis.tokenizer.bpe import TOKENIZER_PATH
    if path is None:
        path = TOKENIZER_PATH

    if os.path.exists(path):
        print(f"Loading existing tokenizer from {path}")
        return load_tokenizer(path)

    print("Training new tokenizer...")
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds['text'][:20000]
    except Exception:
        print("HuggingFace unavailable, using synthetic stories for tokenizer training")
        from neurogenesis.training.data import generate_synthetic_stories
        texts = generate_synthetic_stories(20000)

    tokenizer = train_tokenizer(texts, vocab_size=8192, save_path=path)
    print(f"Tokenizer trained and saved to {path}")
    return tokenizer
