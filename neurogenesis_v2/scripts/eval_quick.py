"""Quick evaluation script for slow VPS — evaluates model on holdout + generation."""
import sys
import os
import json
import math
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis.tokenizer.bpe import load_tokenizer
from neurogenesis_v2.training.category_dataset import load_pretokenized
from torch.utils.data import DataLoader
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
TOKENIZER_PATH = os.path.join(ROOT, 'tokenizer', 'large_corpus_tokenizer.json')
HOLDOUT_JSONL = os.path.join(ROOT, 'data', 'holdout_corpus.jsonl')

HOLDOUT_AUTHORS = {
    'plato_apology': 'Plato (Apology)',
    'plato_republic_full': 'Plato (Republic)',
    'plato_symposium': 'Plato (Symposium)',
    'austen_pride_prejudice': 'Jane Austen',
    'darwin_origin_species': 'Charles Darwin',
    'dostoevsky_crime_punishment': 'Dostoevsky (Crime & Punishment)',
    'dostoevsky_brothers_karamazov': 'Dostoevsky (Brothers Karamazov)',
}

BASELINE_CONFIG = {
    'vocab_size': 8192, 'd_model': 512, 'n_heads': 16,
    'n_layers': 4, 'd_ffn': 512, 'max_seq_len': 256,
}


def eval_holdout_dataset(model, holdout_pt, vocab_size, max_batches=500):
    """Evaluate on holdout dataset (limited batches for speed)."""
    ds = load_pretokenized(holdout_pt)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids, cat_idx) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            input_ids = input_ids.clamp(0, vocab_size - 1)
            target_ids = target_ids.clamp(0, vocab_size - 1)
            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids[:, -1])
            total_loss += loss.item()
            total_count += 1

    avg_loss = total_loss / max(1, total_count)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl, total_count


def eval_train_sample(model, train_pt, vocab_size, max_batches=200):
    """Evaluate on a small sample of training data."""
    ds = load_pretokenized(train_pt)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids, cat_idx) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            input_ids = input_ids.clamp(0, vocab_size - 1)
            target_ids = target_ids.clamp(0, vocab_size - 1)
            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids[:, -1])
            total_loss += loss.item()
            total_count += 1

    avg_loss = total_loss / max(1, total_count)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl, total_count


def eval_per_author(model, tokenizer, vocab_size, max_seq_len=256, max_per_author=50):
    """Evaluate per-author on holdout data."""
    sources = defaultdict(list)
    with open(HOLDOUT_JSONL, 'r') as f:
        for line in f:
            entry = json.loads(line)
            sources[entry['source']].append(entry['text'])

    model.eval()
    results = {}

    with torch.no_grad():
        for source, texts in sorted(sources.items()):
            author_losses = []
            for text in texts[:max_per_author]:
                enc = tokenizer.encode(text)
                ids = [min(i, vocab_size - 1) for i in enc.ids]
                if len(ids) < max_seq_len + 1:
                    ids = ids + [0] * (max_seq_len + 1 - len(ids))
                ids = ids[:max_seq_len + 1]

                input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
                target_id = torch.tensor([ids[-1]], dtype=torch.long)
                logits = model(input_ids)
                loss = F.cross_entropy(logits, target_id)
                author_losses.append(loss.item())

            if not author_losses:
                continue
            avg_loss = sum(author_losses) / len(author_losses)
            avg_ppl = math.exp(min(avg_loss, 20))
            display = HOLDOUT_AUTHORS.get(source, source[:40])
            results[source] = {'ppl': avg_ppl, 'loss': avg_loss, 'n': len(author_losses), 'display': display}

    return results


def generate_text(model, tokenizer, prompt, vocab_size, max_new_tokens=80,
                  temperature=0.8, top_k=50, max_seq_len=256):
    """Generate text from prompt."""
    model.eval()
    enc = tokenizer.encode(prompt)
    token_ids = [min(t, vocab_size - 1) for t in enc.ids]
    generated_ids = list(token_ids)
    eos_id = tokenizer.token_to_id("<eos>")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated_ids[-max_seq_len:]
            inp = torch.tensor([ctx], dtype=torch.long)
            logits = model(inp).squeeze(0)

            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[-1]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == eos_id:
                break
            generated_ids.append(next_token)

    return tokenizer.decode(generated_ids[len(token_ids):])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='baseline', choices=['baseline', 'resonance'])
    parser.add_argument('--train-batches', type=int, default=200)
    parser.add_argument('--holdout-batches', type=int, default=0, help='0 = all')
    parser.add_argument('--authors-per', type=int, default=50)
    args = parser.parse_args()

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    cfg = BASELINE_CONFIG.copy()
    cfg['vocab_size'] = vocab_size

    print(f"Loading {args.model_type} from {args.checkpoint}...")

    if args.model_type == 'baseline':
        model = BaselineTransformer(**cfg)
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        from neurogenesis_v2.model.resonance_model import ResonanceModel
        from neurogenesis_v2.model.config import ResonanceConfig
        # Will be extended for resonance
        raise NotImplementedError("Use train_v3_scale.py --eval-only for resonance")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # 1. Train sample eval
    print(f"\n{'='*60}")
    print(f"TRAIN EVAL (sample, {args.train_batches} batches)")
    print(f"{'='*60}")
    t0 = time.time()
    train_loss, train_ppl, train_n = eval_train_sample(
        model, os.path.join(ROOT, 'data', 'large_train_tokenized.pt'),
        vocab_size, max_batches=args.train_batches
    )
    print(f"  Train PPL: {train_ppl:.4f} (loss={train_loss:.4f}, n={train_n}, {time.time()-t0:.1f}s)")

    # 2. Holdout eval
    print(f"\n{'='*60}")
    print(f"HOLDOUT EVAL")
    print(f"{'='*60}")
    t0 = time.time()
    holdout_loss, holdout_ppl, holdout_n = eval_holdout_dataset(
        model, os.path.join(ROOT, 'data', 'large_holdout_tokenized.pt'),
        vocab_size, max_batches=args.holdout_batches
    )
    print(f"  Holdout PPL: {holdout_ppl:.4f} (loss={holdout_loss:.4f}, n={holdout_n}, {time.time()-t0:.1f}s)")
    print(f"  Generalization Gap: +{holdout_ppl - train_ppl:.4f}")

    # 3. Per-author eval
    print(f"\n{'='*60}")
    print(f"PER-AUTHOR HOLDOUT (max {args.authors_per} per author)")
    print(f"{'='*60}")
    t0 = time.time()
    author_results = eval_per_author(model, tokenizer, vocab_size,
                                      max_seq_len=cfg['max_seq_len'],
                                      max_per_author=args.authors_per)
    for source, data in sorted(author_results.items(), key=lambda x: x[1]['ppl']):
        print(f"  {data['display']:<40}: ppl={data['ppl']:.4f} (n={data['n']})")
    print(f"  ({time.time()-t0:.1f}s)")

    # 4. Generation samples
    print(f"\n{'='*60}")
    print(f"GENERATION SAMPLES")
    print(f"{'='*60}")
    prompts = [
        "Once upon a time",
        "The philosopher argued that",
        "In the beginning",
        "The experiment showed that",
        "She walked into the room and",
    ]
    for prompt in prompts:
        t0 = time.time()
        text = generate_text(model, tokenizer, prompt, vocab_size,
                            max_new_tokens=80, max_seq_len=cfg['max_seq_len'])
        elapsed = time.time() - t0
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: {text[:300]}")
        print(f"  ({elapsed:.1f}s)")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model_type}")
    print(f"{'='*60}")
    print(f"  Train PPL:   {train_ppl:.4f}")
    print(f"  Holdout PPL: {holdout_ppl:.4f}")
    print(f"  Gen Gap:     +{holdout_ppl - train_ppl:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
