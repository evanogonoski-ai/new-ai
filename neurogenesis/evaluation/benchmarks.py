"""Baseline transformer for comparison benchmarking."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaselineTransformer(nn.Module):
    """
    Minimal transformer with ~5M parameters for fair comparison.
    4 layers, 4 heads, d_model=128.
    """

    def __init__(self, vocab_size: int = 8192, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Weight tying
        self.output_proj.weight = self.embedding.weight

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)

        logits = self.output_proj(x[:, -1, :])  # predict from last position
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_baseline(
    dataloader: DataLoader,
    vocab_size: int = 8192,
    num_steps: int = 5000,
    lr: float = 3e-4,
    device: str = 'cpu',
) -> BaselineTransformer:
    """Train the baseline transformer."""
    model = BaselineTransformer(vocab_size=vocab_size).to(device)
    print(f"Baseline Transformer: {model.count_parameters()} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    step = 0
    losses = []
    for epoch in range(100):
        for input_ids, target_ids in tqdm(dataloader, desc=f"Baseline Epoch {epoch+1}", leave=False):
            if step >= num_steps:
                break
            input_ids = input_ids.to(device).clamp(0, vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, vocab_size - 1)

            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids[:, -1])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg = sum(losses[-100:]) / min(100, len(losses))
                ppl = math.exp(min(avg, 20))
                print(f"  Step {step}: loss={avg:.3f}, ppl={ppl:.1f}")
        if step >= num_steps:
            break

    return model


def run_benchmark(
    neurogenesis_model,
    dataloader: DataLoader,
    vocab_size: int = 8192,
    num_train_steps: int = 5000,
    device: str = 'cpu',
) -> dict:
    """
    Train baseline and compare against Neurogenesis model.
    """
    print("=" * 60)
    print("BENCHMARK: Neurogenesis vs Baseline Transformer")
    print("=" * 60)

    # Train baseline on same data
    print("\nTraining baseline transformer...")
    baseline = train_baseline(dataloader, vocab_size=vocab_size,
                               num_steps=num_train_steps, device=device)

    # Evaluate both
    print("\nEvaluating both models...")
    neuro_loss = _eval_loss(neurogenesis_model, dataloader, device, is_neurogenesis=True,
                            vocab_size=vocab_size)
    base_loss = _eval_loss(baseline, dataloader, device, is_neurogenesis=False,
                           vocab_size=vocab_size)

    neuro_ppl = math.exp(min(neuro_loss, 20))
    base_ppl = math.exp(min(base_loss, 20))

    neuro_params = neurogenesis_model.count_parameters()
    base_params = baseline.count_parameters()

    results = {
        'neurogenesis': {
            'loss': neuro_loss,
            'perplexity': neuro_ppl,
            'parameters': neuro_params,
        },
        'baseline': {
            'loss': base_loss,
            'perplexity': base_ppl,
            'parameters': base_params,
        },
    }

    print(f"\nResults:")
    print(f"  Neurogenesis: {neuro_params:,} params, loss={neuro_loss:.3f}, ppl={neuro_ppl:.1f}")
    print(f"  Baseline:     {base_params:,} params, loss={base_loss:.3f}, ppl={base_ppl:.1f}")

    if neuro_ppl < base_ppl:
        improvement = (base_ppl - neuro_ppl) / base_ppl * 100
        print(f"\n  Neurogenesis WINS by {improvement:.1f}% lower perplexity!")
    else:
        gap = (neuro_ppl - base_ppl) / base_ppl * 100
        print(f"\n  Baseline wins by {gap:.1f}% lower perplexity.")

    return results


def _eval_loss(model, dataloader, device, is_neurogenesis=True, vocab_size=8192, max_batches=50):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device).clamp(0, vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, vocab_size - 1)

            if is_neurogenesis:
                output = model(input_ids, training=False, max_iterations=8)
                logits = output['logits']
            else:
                logits = model(input_ids)

            loss = F.cross_entropy(logits, target_ids[:, -1])
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)
