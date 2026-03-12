#!/usr/bin/env python3
"""
V3-A Scale Experiment: 10M Parameter Model
===========================================
Resonance (1 shared block × 8 iterations) vs Baseline (4 unique layers)
Both ~10M parameters, trained on 48M-token diverse corpus.
"""

import os
import sys
import math
import time
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.resonance_model import ResonanceModel
from neurogenesis_v2.model.mitosis import MitosisEngine
from neurogenesis_v2.training.category_dataset import (
    CategoryTextDataset, CATEGORY_NAMES, CATEGORY_MAP,
    pretokenize_corpus, load_pretokenized,
)
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis.tokenizer.bpe import load_tokenizer

# ─── 10M Scale Config ──────────────────────────────────────────────────────

RESONANCE_CONFIG = dict(
    d_model=384,
    n_heads=12,
    d_head=32,
    d_ffn_start=256,
    d_ffn_max=512,
    vocab_size=8192,
    max_seq_len=256,
    # V3-A winner
    gate_min=0.20,
    gate_max=0.80,
    gate_floor=0.0,
    gate_diversity_lambda=0.0,
    convergence_threshold=0.18,
    max_iterations=8,
)

BASELINE_CONFIG = dict(
    vocab_size=8192,
    d_model=512,
    n_heads=16,
    n_layers=4,
    d_ffn=512,
    max_seq_len=256,
)

BATCH_SIZE = 8
LEARNING_RATE = 3e-4

# Paths
DATA_DIR = os.path.join(ROOT, 'data')
TRAIN_JSONL = os.path.join(DATA_DIR, 'large_train_corpus.jsonl')
HOLDOUT_JSONL = os.path.join(DATA_DIR, 'large_holdout_corpus.jsonl')
TRAIN_PT = os.path.join(DATA_DIR, 'large_train_tokenized.pt')
HOLDOUT_PT = os.path.join(DATA_DIR, 'large_holdout_tokenized.pt')
TOKENIZER_PATH = os.path.join(ROOT, 'tokenizer', 'large_corpus_tokenizer.json')
CHECKPOINT_DIR = os.path.join(ROOT, 'checkpoints_10M')
LOG_DIR = os.path.join(ROOT, 'logs')

# Holdout authors for per-author breakdown
HOLDOUT_AUTHORS = {
    'plato_apology': 'Plato (Apology)',
    'plato_republic': 'Plato (Republic)',
    'plato_symposium': 'Plato (Symposium)',
    'jane_austen_pride_and_prejudice': 'Jane Austen',
    'charles_darwin_origin_of_species': 'Charles Darwin',
    'fyodor_dostoevsky_crime_and_punishment': 'Dostoevsky (Crime & Punishment)',
    'fyodor_dostoevsky_brothers_karamazov': 'Dostoevsky (Brothers Karamazov)',
}


# ─── Logging utility ──────────────────────────────────────────────────────

class Logger:
    def __init__(self, path=None):
        self.path = path
        self.f = open(path, 'w') if path else None

    def log(self, msg):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + '\n')
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# ─── Data loading ──────────────────────────────────────────────────────────

def load_datasets(tokenizer, seq_len=256):
    # Ensure category map includes our categories
    if 'wikipedia' not in CATEGORY_MAP:
        CATEGORY_MAP['wikipedia'] = 5
        CATEGORY_NAMES[5] = 'wikipedia'

    datasets = []
    for jsonl, pt, label in [
        (TRAIN_JSONL, TRAIN_PT, 'train'),
        (HOLDOUT_JSONL, HOLDOUT_PT, 'holdout'),
    ]:
        if os.path.exists(pt):
            print(f"Loading {label} from {pt}...", flush=True)
            ds = load_pretokenized(pt)
        else:
            print(f"Pre-tokenizing {label} from {jsonl}...", flush=True)
            ds = pretokenize_corpus(jsonl, tokenizer, seq_len=seq_len, save_path=pt)
        print(f"  {label}: {len(ds):,} samples", flush=True)
        datasets.append(ds)

    return datasets[0], datasets[1]


# ─── LR Schedule ──────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ─── Training: Resonance ──────────────────────────────────────────────────

def train_resonance(config, dataset, num_steps, device, checkpoint_interval,
                    logger, gradient_accumulation=1, resume_checkpoint=None):
    model = ResonanceModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Resonance 10M model: {total_params:,} total, {active_params:,} active params")
    logger.log(f"V3-A config: gate_min={config.gate_min}, gate_max={config.gate_max}")
    logger.log(f"d_model={config.d_model}, n_heads={config.n_heads}, d_head={config.d_head}")
    logger.log(f"d_ffn_start={config.d_ffn_start}, d_ffn_max={config.d_ffn_max}")
    logger.log(f"max_seq_len={config.max_seq_len}, max_iterations={config.max_iterations}")
    logger.log(f"batch_size={BATCH_SIZE}, gradient_accumulation={gradient_accumulation}")
    logger.log(f"effective_batch_size={BATCH_SIZE * gradient_accumulation}")

    # Resume from checkpoint if provided
    start_step = 0
    if resume_checkpoint:
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device, weights_only=True))
        import re
        m = re.search(r'step(\d+)', resume_checkpoint)
        if m:
            start_step = int(m.group(1))
        logger.log(f"Resumed from {resume_checkpoint} at step {start_step}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=num_steps,
                                       base_lr=LEARNING_RATE)
    # Advance scheduler to resume point
    for _ in range(start_step):
        scheduler.step()

    mitosis = MitosisEngine(model.block.ffn, config)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    logger.log(f"\n{'='*70}")
    logger.log(f"RESONANCE 10M TRAINING: {num_steps} steps (from step {start_step})")
    logger.log(f"Training samples: {len(dataset):,}")
    logger.log(f"{'='*70}")

    model.train()
    losses = []
    step = start_step
    accum_count = 0
    start = time.time()

    while step < num_steps:
        for input_ids, target_ids, cat_idx in loader:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

            output = model(input_ids, max_iterations=config.max_iterations, use_momentum=True)
            logits = output['logits']

            ce_loss = F.cross_entropy(logits, target_ids[:, -1])
            ponder_loss = output['ponder_cost'] * 0.01 if output['ponder_cost'] is not None else 0
            total_loss = ce_loss + ponder_loss

            scaled_loss = total_loss / gradient_accumulation
            scaled_loss.backward()
            accum_count += 1

            if accum_count >= gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Mitosis check
                mitosis.record_gradients()
                mresult = mitosis.check_and_apply(step)
                if mresult and (mresult['mitosis'] > 0 or mresult['pruned'] > 0):
                    logger.log(f"  [Mitosis] step {step}: +{mresult['mitosis']} split, "
                              f"-{mresult['pruned']} pruned, alive={mresult['alive']}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1

                losses.append(ce_loss.item())

                if step % 100 == 0:
                    avg = sum(losses[-100:]) / min(100, len(losses))
                    ppl = math.exp(min(avg, 20))
                    iters = output['num_iterations']
                    gate_mean = output['all_gates'][-1].mean().item() if output['all_gates'] else 0
                    gate_std = output['all_gates'][-1].std().item() if output['all_gates'] else 0
                    alive = model.block.ffn.num_alive
                    speed = step / (time.time() - start)
                    lr = scheduler.get_last_lr()[0]
                    logger.log(f"  Step {step}/{num_steps}: loss={avg:.4f}, ppl={ppl:.2f}, "
                              f"iters={iters}, gates={gate_mean:.3f}\u00b1{gate_std:.3f}, "
                              f"alive={alive}, lr={lr:.2e}, {speed:.2f} it/s")

                # Checkpoint
                if step % checkpoint_interval == 0 and step < num_steps:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"resonance_step{step}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.log(f"  Checkpoint saved: {ckpt_path}")

    avg = sum(losses[-100:]) / min(100, len(losses))
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - start
    logger.log(f"\nResonance done: loss={avg:.4f}, ppl={ppl:.2f} ({elapsed:.0f}s, {elapsed/60:.1f}m)")

    final_path = os.path.join(CHECKPOINT_DIR, "resonance_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.log(f"Final checkpoint: {final_path}")

    return model


# ─── Training: Baseline ──────────────────────────────────────────────────

def train_baseline(num_steps, dataset, device, checkpoint_interval, logger,
                   gradient_accumulation=1, resume_checkpoint=None):
    cfg = BASELINE_CONFIG
    model = BaselineTransformer(
        vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
        n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
        d_ffn=cfg['d_ffn'], max_seq_len=cfg['max_seq_len'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Baseline 10M model: {total_params:,} params")
    logger.log(f"d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, n_layers={cfg['n_layers']}, d_ffn={cfg['d_ffn']}")
    logger.log(f"batch_size={BATCH_SIZE}, gradient_accumulation={gradient_accumulation}")
    logger.log(f"effective_batch_size={BATCH_SIZE * gradient_accumulation}")

    # Resume from checkpoint if provided
    start_step = 0
    if resume_checkpoint:
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device, weights_only=True))
        # Extract step number from checkpoint filename
        import re
        m = re.search(r'step(\d+)', resume_checkpoint)
        if m:
            start_step = int(m.group(1))
        logger.log(f"Resumed from {resume_checkpoint} at step {start_step}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=num_steps,
                                       base_lr=LEARNING_RATE)
    # Advance scheduler to resume point
    for _ in range(start_step):
        scheduler.step()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    logger.log(f"\n{'='*70}")
    logger.log(f"BASELINE 10M TRAINING: {num_steps} steps (from step {start_step})")
    logger.log(f"Training samples: {len(dataset):,}")
    logger.log(f"{'='*70}")

    model.train()
    losses = []
    step = start_step
    accum_count = 0
    start = time.time()

    while step < num_steps:
        for input_ids, target_ids, cat_idx in loader:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device).clamp(0, cfg['vocab_size'] - 1)
            target_ids = target_ids.to(device).clamp(0, cfg['vocab_size'] - 1)

            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids[:, -1])

            scaled_loss = loss / gradient_accumulation
            scaled_loss.backward()
            accum_count += 1

            if accum_count >= gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1

                losses.append(loss.item())

                if step % 100 == 0:
                    avg = sum(losses[-100:]) / min(100, len(losses))
                    ppl = math.exp(min(avg, 20))
                    speed = step / (time.time() - start)
                    lr = scheduler.get_last_lr()[0]
                    logger.log(f"  Step {step}/{num_steps}: loss={avg:.4f}, ppl={ppl:.2f}, "
                              f"lr={lr:.2e}, {speed:.2f} it/s")

                if step % checkpoint_interval == 0 and step < num_steps:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"baseline_step{step}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.log(f"  Checkpoint saved: {ckpt_path}")

    avg = sum(losses[-100:]) / min(100, len(losses))
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - start
    logger.log(f"\nBaseline done: loss={avg:.4f}, ppl={ppl:.2f} ({elapsed:.0f}s, {elapsed/60:.1f}m)")

    final_path = os.path.join(CHECKPOINT_DIR, "baseline_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.log(f"Final checkpoint: {final_path}")

    return model


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_model(model, dataset, config_or_dict, label, device='cpu',
               is_resonance=True, logger=None, max_batches=0):
    """Evaluate model, returning per-category metrics.
    max_batches: if >0, stop after this many batches (for speed on large datasets).
    """
    log = logger.log if logger else print
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()

    vocab_size = config_or_dict.vocab_size if hasattr(config_or_dict, 'vocab_size') else config_or_dict['vocab_size']

    per_cat = defaultdict(lambda: {'loss': [], 'count': 0, 'gates': [], 'iters': []})
    total_loss = 0.0
    total_count = 0
    all_gate_values = []

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids, cat_idx) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            input_ids = input_ids.to(device).clamp(0, vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, vocab_size - 1)

            if is_resonance:
                output = model(input_ids, max_iterations=8, use_momentum=True)
                logits = output['logits']
                last_gates = output['all_gates'][-1] if output['all_gates'] else None
                gate_mean = last_gates.mean().item() if last_gates is not None else 0
                iters = output['num_iterations']
                if last_gates is not None:
                    all_gate_values.append(last_gates.detach())
            else:
                logits = model(input_ids)
                gate_mean = iters = None

            loss = F.cross_entropy(logits, target_ids[:, -1])
            total_loss += loss.item()
            total_count += 1

            for i in range(len(cat_idx)):
                c = cat_idx[i].item()
                per_cat[c]['loss'].append(loss.item())
                per_cat[c]['count'] += 1
                if is_resonance:
                    per_cat[c]['gates'].append(gate_mean)
                    per_cat[c]['iters'].append(iters)

    avg_loss = total_loss / total_count if total_count > 0 else 0
    avg_ppl = math.exp(min(avg_loss, 20))

    gate_variance = 0.0
    if all_gate_values:
        all_gates_tensor = torch.cat(all_gate_values, dim=0)
        gate_variance = all_gates_tensor.var(dim=0).mean().item()

    log(f"\n  {label}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f} ({total_count} batches)")
    if is_resonance:
        log(f"    Gate variance: {gate_variance:.6f}")

    for c in sorted(per_cat.keys()):
        cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')
        m = per_cat[c]
        cat_loss = sum(m['loss']) / len(m['loss']) if m['loss'] else 0
        cat_ppl = math.exp(min(cat_loss, 20))
        extra = ''
        if is_resonance and m['gates']:
            g = sum(m['gates']) / len(m['gates'])
            it = sum(m['iters']) / len(m['iters'])
            extra = f', gates={g:.3f}, iters={it:.1f}'
        log(f"    {cat_name:<12}: loss={cat_loss:.4f}, ppl={cat_ppl:.2f}{extra}")

    return avg_loss, avg_ppl, dict(per_cat), gate_variance


def eval_holdout_by_author(model, holdout_jsonl, tokenizer, config_or_dict, label,
                            device='cpu', is_resonance=True, logger=None):
    """Evaluate on holdout data, broken down by author."""
    log = logger.log if logger else print
    log(f"\n  {label} \u2014 Per-Author Holdout Breakdown:")

    vocab_size = config_or_dict.vocab_size if hasattr(config_or_dict, 'vocab_size') else config_or_dict['vocab_size']
    max_seq_len = config_or_dict.max_seq_len if hasattr(config_or_dict, 'max_seq_len') else config_or_dict['max_seq_len']

    sources = defaultdict(list)
    with open(holdout_jsonl, 'r') as f:
        for line in f:
            entry = json.loads(line)
            sources[entry['source']].append(entry['text'])

    model.eval()
    results = {}

    with torch.no_grad():
        for source, texts in sorted(sources.items()):
            author_losses = []
            author_gates = []
            author_iters = []

            # Limit to first 200 texts per source for speed
            for text in texts[:200]:
                enc = tokenizer.encode(text)
                ids = [min(i, vocab_size - 1) for i in enc.ids]

                if len(ids) < max_seq_len + 1:
                    ids = ids + [0] * (max_seq_len + 1 - len(ids))
                ids = ids[:max_seq_len + 1]

                input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                target_ids = torch.tensor([ids[-1]], dtype=torch.long, device=device)

                if is_resonance:
                    output = model(input_ids, max_iterations=8, use_momentum=True)
                    logits = output['logits']
                    if output['all_gates']:
                        author_gates.append(output['all_gates'][-1].mean().item())
                    author_iters.append(output['num_iterations'])
                else:
                    logits = model(input_ids)

                loss = F.cross_entropy(logits, target_ids)
                author_losses.append(loss.item())

            if not author_losses:
                continue

            avg_loss = sum(author_losses) / len(author_losses)
            avg_ppl = math.exp(min(avg_loss, 20))
            display_name = HOLDOUT_AUTHORS.get(source, source[:40])

            extra = ''
            if is_resonance and author_gates:
                g = sum(author_gates) / len(author_gates)
                it = sum(author_iters) / len(author_iters)
                extra = f', gates={g:.3f}, iters={it:.1f}'

            log(f"    {display_name:<40}: ppl={avg_ppl:.2f}, n={len(author_losses)}{extra}")

            results[source] = {
                'loss': avg_loss, 'ppl': avg_ppl, 'n': len(author_losses),
                'display_name': display_name,
            }
            if is_resonance and author_gates:
                results[source]['gates'] = sum(author_gates) / len(author_gates)
                results[source]['iters'] = sum(author_iters) / len(author_iters)

    return results


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8,
                  top_k=50, device='cpu', is_resonance=True, config=None):
    """Generate text from a prompt."""
    model.eval()

    encoded = tokenizer.encode(prompt)
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else config['vocab_size']
    max_seq_len = config.max_seq_len if hasattr(config, 'max_seq_len') else config['max_seq_len']
    token_ids = [min(t, vocab_size - 1) for t in encoded.ids]
    generated_ids = list(token_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated_ids[-max_seq_len:]
            input_ids = torch.tensor([ctx], dtype=torch.long, device=device)

            if is_resonance:
                output = model(input_ids, max_iterations=8)
                logits = output['logits'].squeeze(0)
            else:
                logits = model(input_ids).squeeze(0)

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[-1]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            eos_id = tokenizer.token_to_id("<eos>")
            if next_token == eos_id:
                break

            generated_ids.append(next_token)

    generated_text = tokenizer.decode(generated_ids[len(token_ids):])
    return generated_text


def run_generation_samples(model, tokenizer, device, is_resonance, config, logger):
    """Generate samples from 5 prompts."""
    prompts = [
        "Once upon a time",
        "The philosopher argued that",
        "In the beginning",
        "The experiment showed that",
        "She walked into the room and",
    ]
    model_name = "Resonance" if is_resonance else "Baseline"

    logger.log(f"\n{'='*70}")
    logger.log(f"GENERATION SAMPLES \u2014 {model_name}")
    logger.log(f"{'='*70}")

    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, max_new_tokens=100,
                           temperature=0.8, top_k=50, device=device,
                           is_resonance=is_resonance, config=config)
        logger.log(f"\n  Prompt: \"{prompt}\"")
        logger.log(f"  Output: {text[:500]}")


def run_speed_test(model, tokenizer, device, is_resonance, config, logger):
    """Measure inference speed: generate 100 tokens, time it."""
    model_name = "Resonance" if is_resonance else "Baseline"
    prompt = "The history of civilization shows that"

    logger.log(f"\n{'='*70}")
    logger.log(f"INFERENCE SPEED TEST \u2014 {model_name}")
    logger.log(f"{'='*70}")

    # Warmup
    _ = generate_text(model, tokenizer, prompt, max_new_tokens=10,
                     device=device, is_resonance=is_resonance, config=config)

    # Timed run
    start = time.time()
    text = generate_text(model, tokenizer, prompt, max_new_tokens=100,
                        temperature=0.8, top_k=50, device=device,
                        is_resonance=is_resonance, config=config)
    elapsed = time.time() - start

    tokens_generated = len(tokenizer.encode(text).ids)
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    logger.log(f"  Generated {tokens_generated} tokens in {elapsed:.2f}s "
              f"({tokens_per_sec:.1f} tokens/sec)")

    return elapsed, tokens_per_sec


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V3-A Scale Experiment (10M params)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resonance', 'baseline'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--speed-test', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    is_resonance = args.model == 'resonance'

    # Logger
    log_name = f'v3_scale_{args.model}'
    if args.eval_only:
        log_name += '_eval'
    train_logger = Logger(os.path.join(LOG_DIR, f'{log_name}_train.txt'))

    # Tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    train_logger.log(f"Tokenizer: vocab={vocab_size}")

    # Update vocab size in configs
    RESONANCE_CONFIG['vocab_size'] = vocab_size
    BASELINE_CONFIG['vocab_size'] = vocab_size

    # Datasets
    train_dataset, holdout_dataset = load_datasets(tokenizer, seq_len=256)

    if is_resonance:
        config = ResonanceConfig()
        for key, val in RESONANCE_CONFIG.items():
            setattr(config, key, val)
    else:
        config = BASELINE_CONFIG

    # ─── TRAINING or LOADING ─────────────────────────────────────────────
    if args.eval_only:
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = os.path.join(CHECKPOINT_DIR, f"{args.model}_final.pt")
        train_logger.log(f"\n--- Loading from checkpoint: {ckpt} ---")

        if is_resonance:
            model = ResonanceModel(config).to(args.device)
        else:
            cfg = BASELINE_CONFIG
            model = BaselineTransformer(
                vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
                n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
                d_ffn=cfg['d_ffn'], max_seq_len=cfg['max_seq_len'],
            ).to(args.device)

        model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
        train_logger.log(f"Loaded {args.model} from {ckpt}")
    else:
        if is_resonance:
            model = train_resonance(
                config, train_dataset, args.steps, args.device,
                args.checkpoint_every, train_logger, args.gradient_accumulation,
                resume_checkpoint=args.resume
            )
        else:
            model = train_baseline(
                args.steps, train_dataset, args.device,
                args.checkpoint_every, train_logger, args.gradient_accumulation,
                resume_checkpoint=args.resume
            )

    train_logger.close()

    # ─── EVALUATION ──────────────────────────────────────────────────────
    eval_logger = Logger(os.path.join(LOG_DIR, f'v3_scale_{args.model}_eval.txt'))

    eval_logger.log(f"\n{'='*70}")
    eval_logger.log(f"EVALUATION: {args.model} (10M scale)")
    eval_logger.log(f"{'='*70}")

    # Eval on train data (sample 5000 batches = 40K examples for speed)
    res_train = eval_model(
        model, train_dataset, config, f"{args.model} on TRAIN (sampled)",
        args.device, is_resonance=is_resonance, logger=eval_logger,
        max_batches=5000
    )

    # Eval on holdout data (full)
    res_holdout = eval_model(
        model, holdout_dataset, config, f"{args.model} on HOLDOUT",
        args.device, is_resonance=is_resonance, logger=eval_logger
    )

    # Per-author holdout
    res_authors = eval_holdout_by_author(
        model, HOLDOUT_JSONL, tokenizer, config,
        f"{args.model} per-author", args.device, is_resonance=is_resonance,
        logger=eval_logger
    )

    # Generation samples
    if args.generate or not args.eval_only:
        run_generation_samples(model, tokenizer, args.device, is_resonance, config, eval_logger)

    # Speed test
    if args.speed_test or not args.eval_only:
        run_speed_test(model, tokenizer, args.device, is_resonance, config, eval_logger)

    eval_logger.close()

    # Print summary
    train_loss, train_ppl, _, _ = res_train
    hold_loss, hold_ppl, _, gate_var = res_holdout
    gen_gap = hold_ppl - train_ppl

    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.model} (10M)")
    print(f"{'='*70}")
    print(f"  Train PPL:   {train_ppl:.2f}")
    print(f"  Holdout PPL: {hold_ppl:.2f}")
    print(f"  Gen Gap:     {gen_gap:+.2f}")
    if is_resonance:
        print(f"  Gate Var:    {gate_var:.6f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
