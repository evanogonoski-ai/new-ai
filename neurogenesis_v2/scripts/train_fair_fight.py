#!/usr/bin/env python3
"""
Fair Fight Experiment: 4 Models at Controlled Parameters and FLOPs
==================================================================
Model A: Baseline Transformer (4 unique layers, ~16.8M params)
Model B: Resonance V3-A 4-iter (1 shared block × 4, ~7.3M params)
Model C: Resonance V3-A 8-iter (1 shared block × 8, ~7.3M params)
Model D: Universal Transformer (1 shared block × 4, no gating, ~7.3M params)

All use d_model=512, n_heads=16, d_ffn=2048, vocab=8192, seq_len=256.
All-position causal LM loss.
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
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis_v2.training.category_dataset import (
    CATEGORY_NAMES, CATEGORY_MAP, pretokenize_corpus, load_pretokenized,
)
from neurogenesis.tokenizer.bpe import load_tokenizer

# ─── Shared Config ────────────────────────────────────────────────────────────

SHARED = dict(
    d_model=512,
    n_heads=16,
    d_head=32,
    d_ffn=2048,
    vocab_size=8192,
    max_seq_len=256,
)

BATCH_SIZE = 8
LEARNING_RATE = 3e-4

# Paths
DATA_DIR = os.path.join(ROOT, 'data')
TOKENIZER_PATH = os.path.join(ROOT, 'tokenizer', 'fair_fight_tokenizer.json')
TRAIN_PT = os.path.join(DATA_DIR, 'fair_train_tokenized.pt')
HOLDOUT_PT = os.path.join(DATA_DIR, 'fair_holdout_tokenized.pt')
TRAIN_JSONL = os.path.join(DATA_DIR, 'fair_train_corpus.jsonl')
HOLDOUT_JSONL = os.path.join(DATA_DIR, 'fair_holdout_corpus.jsonl')
CHECKPOINT_DIR = os.path.join(ROOT, 'checkpoints_fair')
LOG_DIR = os.path.join(ROOT, 'logs')

HOLDOUT_AUTHORS = {
    'plato_apology': 'Plato (Apology)',
    'plato_republic': 'Plato (Republic)',
    'plato_symposium': 'Plato (Symposium)',
    'plato_republic_full': 'Plato (Republic)',
    'jane_austen_pride_and_prejudice': 'Jane Austen',
    'charles_darwin_origin_of_species': 'Charles Darwin',
    'fyodor_dostoevsky_crime_and_punishment': 'Dostoevsky (C&P)',
    'fyodor_dostoevsky_brothers_karamazov': 'Dostoevsky (BK)',
}


# ─── Logger ──────────────────────────────────────────────────────────────────

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


# ─── LR Schedule ─────────────────────────────────────────────────────────────

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


# ─── Model Builders ──────────────────────────────────────────────────────────

def build_model(model_type, vocab_size, n_iterations=4):
    """Build one of the 4 models. Returns (model, is_resonance, config_dict)."""

    if model_type == 'baseline':
        model = BaselineTransformer(
            vocab_size=vocab_size, d_model=512, n_heads=16,
            n_layers=4, d_ffn=2048, max_seq_len=256,
        )
        return model, False, {'vocab_size': vocab_size, 'max_seq_len': 256, 'n_layers': 4}

    # Resonance-based models (B, C, D)
    config = ResonanceConfig()
    config.vocab_size = vocab_size
    config.d_model = 512
    config.n_heads = 16
    config.d_head = 32
    config.d_ffn_start = 2048
    config.d_ffn_max = 2048
    config.max_seq_len = 256
    config.max_iterations = n_iterations
    config.convergence_threshold = 0.18
    config.momentum_beta_init = 0.05
    config.noise_scale = 0.0

    if model_type == 'universal':
        # Model D: disable entropy gating (gate = 1.0 always)
        config.gate_min = 1.0
        config.gate_max = 1.0
        config.gate_floor = 0.0
        config.momentum_beta_init = 0.0  # No momentum
        config.convergence_threshold = 999.0  # Never halt early
    elif model_type in ('resonance',):
        # Model B/C: full V3-A
        config.gate_min = 0.20
        config.gate_max = 0.80
        config.gate_floor = 0.0

    # Build model WITHOUT multi-timescale heads for fair param count
    model = ResonanceModel(config)

    # Zero out and freeze the training-only heads so they don't affect param count
    # (They still exist in the model but won't be trained)
    for p in model.surface_head.parameters():
        p.requires_grad = False
    for p in model.semantic_head.parameters():
        p.requires_grad = False

    return model, True, config


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(model, is_resonance, config, dataset, num_steps, device,
                checkpoint_interval, logger, model_label, n_iterations=4,
                force_iterations=False):
    """Train any of the 4 models with identical training loop."""
    model = model.to(device)
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else config['vocab_size']

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.log(f"{model_label}: {trainable:,} trainable params ({total:,} total)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=200, total_steps=num_steps,
                                       base_lr=LEARNING_RATE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    logger.log(f"\n{'='*70}")
    logger.log(f"TRAINING: {model_label} — {num_steps} steps")
    logger.log(f"Training samples: {len(dataset):,}, batch_size={BATCH_SIZE}")
    logger.log(f"{'='*70}")

    model.train()
    losses = []
    step = 0
    start = time.time()

    while step < num_steps:
        for input_ids, target_ids, cat_idx in loader:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device).clamp(0, vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, vocab_size - 1)

            if is_resonance:
                use_mom = not force_iterations  # No momentum for universal
                output = model(input_ids, max_iterations=n_iterations,
                             use_momentum=use_mom, force_iterations=force_iterations)
                logits = output['logits']  # (B, S, V)
            else:
                logits = model(input_ids)  # (B, S, V)

            loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            losses.append(loss.item())

            if step % 100 == 0:
                avg = sum(losses[-100:]) / min(100, len(losses))
                ppl = math.exp(min(avg, 20))
                speed = 100 / (time.time() - start) if step == 100 else step / (time.time() - start)
                lr = scheduler.get_last_lr()[0]
                extra = ''
                if is_resonance and not force_iterations:
                    iters = output['num_iterations']
                    if output['all_gates']:
                        g = output['all_gates'][-1]
                        extra = f', iters={iters}, gates={g.mean().item():.3f}±{g.std().item():.3f}'
                    else:
                        extra = f', iters={iters}'
                logger.log(f"  Step {step}/{num_steps}: loss={avg:.4f}, ppl={ppl:.2f}, "
                          f"lr={lr:.2e}, {speed:.2f} it/s{extra}")

            if step % checkpoint_interval == 0 and step < num_steps:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_label}_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                logger.log(f"  Checkpoint: {ckpt_path}")

    avg = sum(losses[-100:]) / min(100, len(losses))
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - start
    logger.log(f"\n{model_label} done: loss={avg:.4f}, ppl={ppl:.2f} ({elapsed:.0f}s, {elapsed/60:.1f}m)")

    final_path = os.path.join(CHECKPOINT_DIR, f"{model_label}_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.log(f"Final checkpoint: {final_path}")

    return model


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_model(model, dataset, vocab_size, label, device='cpu',
               is_resonance=False, n_iterations=4, force_iterations=False,
               logger=None, max_batches=0):
    """Evaluate model on dataset. Returns (loss, ppl, per_cat, gate_var)."""
    log = logger.log if logger else print
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()

    per_cat = defaultdict(lambda: {'loss': [], 'gates': [], 'iters': []})
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
                use_mom = not force_iterations
                output = model(input_ids, max_iterations=n_iterations,
                             use_momentum=use_mom, force_iterations=force_iterations)
                logits = output['logits']
                if output.get('all_gates') and not force_iterations:
                    last_gates = output['all_gates'][-1]
                    all_gate_values.append(last_gates.detach())
                    gate_mean = last_gates.mean().item()
                    iters = output['num_iterations']
                else:
                    gate_mean = iters = None
            else:
                logits = model(input_ids)
                gate_mean = iters = None

            loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            total_loss += loss.item()
            total_count += 1

            for i in range(len(cat_idx)):
                c = cat_idx[i].item()
                per_cat[c]['loss'].append(loss.item())
                if gate_mean is not None:
                    per_cat[c]['gates'].append(gate_mean)
                    per_cat[c]['iters'].append(iters)

    avg_loss = total_loss / max(total_count, 1)
    avg_ppl = math.exp(min(avg_loss, 20))

    gate_variance = 0.0
    if all_gate_values:
        all_g = torch.cat(all_gate_values, dim=0)
        gate_variance = all_g.var(dim=0).mean().item()

    log(f"\n  {label}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f} ({total_count} batches)")
    if all_gate_values:
        log(f"    Gate variance: {gate_variance:.6f}")

    for c in sorted(per_cat.keys()):
        cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')
        m = per_cat[c]
        cat_loss = sum(m['loss']) / len(m['loss']) if m['loss'] else 0
        cat_ppl = math.exp(min(cat_loss, 20))
        extra = ''
        if m['gates']:
            g = sum(m['gates']) / len(m['gates'])
            it = sum(m['iters']) / len(m['iters'])
            extra = f', gates={g:.3f}, iters={it:.1f}'
        log(f"    {cat_name:<12}: loss={cat_loss:.4f}, ppl={cat_ppl:.2f}{extra}")

    return avg_loss, avg_ppl, dict(per_cat), gate_variance


def eval_holdout_by_author(model, holdout_jsonl, tokenizer, vocab_size, label,
                           device='cpu', is_resonance=False, n_iterations=4,
                           force_iterations=False, logger=None):
    """Per-author holdout eval."""
    log = logger.log if logger else print
    log(f"\n  {label} — Per-Author Holdout:")

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

            for text in texts[:100]:  # Limit per author for speed
                enc = tokenizer.encode(text)
                ids = [min(i, vocab_size - 1) for i in enc.ids]
                if len(ids) < 257:
                    ids = ids + [0] * (257 - len(ids))
                ids = ids[:257]

                input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                target_ids = torch.tensor([ids[1:]], dtype=torch.long, device=device)

                if is_resonance:
                    use_mom = not force_iterations
                    output = model(input_ids, max_iterations=n_iterations,
                                 use_momentum=use_mom, force_iterations=force_iterations)
                    logits = output['logits']
                    if output.get('all_gates') and not force_iterations:
                        author_gates.append(output['all_gates'][-1].mean().item())
                    author_iters.append(output.get('num_iterations', n_iterations))
                else:
                    logits = model(input_ids)

                loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
                author_losses.append(loss.item())

            if not author_losses:
                continue

            avg_loss = sum(author_losses) / len(author_losses)
            avg_ppl = math.exp(min(avg_loss, 20))
            display = HOLDOUT_AUTHORS.get(source, source[:30])

            extra = ''
            if author_gates:
                g = sum(author_gates) / len(author_gates)
                it = sum(author_iters) / len(author_iters)
                extra = f', gates={g:.3f}, iters={it:.1f}'

            log(f"    {display:<30}: ppl={avg_ppl:.2f} (n={len(author_losses)}){extra}")
            results[source] = {'loss': avg_loss, 'ppl': avg_ppl, 'n': len(author_losses)}
            if author_gates:
                results[source]['gates'] = sum(author_gates) / len(author_gates)
                results[source]['iters'] = sum(author_iters) / len(author_iters)

    return results


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt, vocab_size, max_new_tokens=100,
                  temperature=0.8, top_k=50, device='cpu',
                  is_resonance=False, n_iterations=4, force_iterations=False):
    model.eval()
    enc = tokenizer.encode(prompt)
    ids = [min(t, vocab_size - 1) for t in enc.ids]
    generated = list(ids)
    eos = tokenizer.token_to_id("<eos>")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated[-256:]
            inp = torch.tensor([ctx], dtype=torch.long, device=device)

            if is_resonance:
                use_mom = not force_iterations
                output = model(inp, max_iterations=n_iterations,
                             use_momentum=use_mom, force_iterations=force_iterations)
                logits = output['logits'][:, -1, :].squeeze(0)
            else:
                logits = model(inp)[:, -1, :].squeeze(0)

            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[-1]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1).item()
            if tok == eos:
                break
            generated.append(tok)

    return tokenizer.decode(generated[len(ids):])


def run_generation_samples(model, tokenizer, vocab_size, device, is_resonance,
                           n_iterations, force_iterations, model_label, logger):
    prompts = [
        "Once upon a time",
        "The philosopher argued that",
        "In the beginning",
        "The experiment showed that",
        "She walked into the room and",
    ]
    logger.log(f"\n{'='*70}")
    logger.log(f"GENERATION — {model_label}")
    logger.log(f"{'='*70}")

    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, vocab_size,
                           max_new_tokens=100, device=device,
                           is_resonance=is_resonance, n_iterations=n_iterations,
                           force_iterations=force_iterations)
        logger.log(f"\n  \"{prompt}\"")
        logger.log(f"  → {text[:400]}")


def run_speed_test(model, tokenizer, vocab_size, device, is_resonance,
                   n_iterations, force_iterations, model_label, logger):
    prompt = "The history of civilization shows that"
    logger.log(f"\n{'='*70}")
    logger.log(f"SPEED TEST — {model_label}")
    logger.log(f"{'='*70}")

    # Warmup
    _ = generate_text(model, tokenizer, prompt, vocab_size, max_new_tokens=10,
                     device=device, is_resonance=is_resonance,
                     n_iterations=n_iterations, force_iterations=force_iterations)

    # Timed runs
    times = []
    tokens = []
    for _ in range(5):
        t0 = time.time()
        text = generate_text(model, tokenizer, prompt, vocab_size,
                           max_new_tokens=100, device=device,
                           is_resonance=is_resonance, n_iterations=n_iterations,
                           force_iterations=force_iterations)
        elapsed = time.time() - t0
        n_tok = len(tokenizer.encode(text).ids)
        times.append(elapsed)
        tokens.append(n_tok)

    avg_time = sum(times) / len(times)
    avg_tok = sum(tokens) / len(tokens)
    tok_per_sec = avg_tok / avg_time if avg_time > 0 else 0

    logger.log(f"  {avg_tok:.0f} tokens in {avg_time:.2f}s → {tok_per_sec:.1f} tokens/sec")
    return tok_per_sec


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fair Fight Experiment')
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'resonance', 'universal'])
    parser.add_argument('--n-iterations', type=int, default=4)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Determine model label
    if args.model == 'baseline':
        model_label = 'A_baseline'
    elif args.model == 'universal':
        model_label = 'D_universal'
    elif args.n_iterations == 8:
        model_label = 'C_resonance_8i'
    else:
        model_label = 'B_resonance_4i'

    is_resonance = args.model in ('resonance', 'universal')
    force_iterations = args.model == 'universal'

    # Logger
    log_suffix = '_eval' if args.eval_only else ''
    logger = Logger(os.path.join(LOG_DIR, f'fair_{model_label}{log_suffix}.txt'))

    # Tokenizer
    tok_path = TOKENIZER_PATH
    if not os.path.exists(tok_path):
        # Fallback to existing tokenizer
        tok_path = os.path.join(ROOT, 'tokenizer', 'large_corpus_tokenizer.json')
    tokenizer = load_tokenizer(tok_path)
    vocab_size = tokenizer.get_vocab_size()
    logger.log(f"Tokenizer: vocab={vocab_size}")

    # Datasets
    if 'wikipedia' not in CATEGORY_MAP:
        CATEGORY_MAP['wikipedia'] = 5
        CATEGORY_NAMES[5] = 'wikipedia'

    for pt, jsonl, label in [
        (TRAIN_PT, TRAIN_JSONL, 'train'),
        (HOLDOUT_PT, HOLDOUT_JSONL, 'holdout'),
    ]:
        if not os.path.exists(pt):
            if os.path.exists(jsonl):
                print(f"Pre-tokenizing {label}...", flush=True)
                pretokenize_corpus(jsonl, tokenizer, seq_len=256, save_path=pt)
            else:
                # Fallback to existing data
                fallback = os.path.join(DATA_DIR, f'large_{label}_tokenized.pt')
                if os.path.exists(fallback):
                    print(f"Using fallback {label} data: {fallback}", flush=True)
                    # Symlink
                    os.symlink(fallback, pt)

    train_dataset = load_pretokenized(TRAIN_PT)
    holdout_dataset = load_pretokenized(HOLDOUT_PT)
    logger.log(f"Train: {len(train_dataset):,} samples, Holdout: {len(holdout_dataset):,}")

    # Build model
    model, is_res, config = build_model(args.model, vocab_size, args.n_iterations)

    # ─── TRAINING or LOADING ─────────────────────────────────────────────
    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"{model_label}_final.pt")
        logger.log(f"Loading from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
        model = model.to(args.device)
    else:
        model = train_model(
            model, is_res, config, train_dataset, args.steps, args.device,
            args.checkpoint_every, logger, model_label,
            n_iterations=args.n_iterations, force_iterations=force_iterations,
        )

    logger.close()

    # ─── EVALUATION ──────────────────────────────────────────────────────
    eval_logger = Logger(os.path.join(LOG_DIR, f'fair_{model_label}_eval.txt'))

    eval_logger.log(f"\n{'='*70}")
    eval_logger.log(f"EVALUATION: {model_label}")
    eval_logger.log(f"{'='*70}")

    # Train sample eval
    res_train = eval_model(
        model, train_dataset, vocab_size, f"{model_label} on TRAIN (500 batches)",
        args.device, is_resonance=is_res, n_iterations=args.n_iterations,
        force_iterations=force_iterations, logger=eval_logger, max_batches=500
    )

    # Full holdout eval
    res_holdout = eval_model(
        model, holdout_dataset, vocab_size, f"{model_label} on HOLDOUT",
        args.device, is_resonance=is_res, n_iterations=args.n_iterations,
        force_iterations=force_iterations, logger=eval_logger
    )

    # Per-author holdout
    holdout_jsonl = HOLDOUT_JSONL
    if not os.path.exists(holdout_jsonl):
        holdout_jsonl = os.path.join(DATA_DIR, 'large_holdout_corpus.jsonl')
    if os.path.exists(holdout_jsonl):
        eval_holdout_by_author(
            model, holdout_jsonl, tokenizer, vocab_size,
            model_label, args.device, is_resonance=is_res,
            n_iterations=args.n_iterations, force_iterations=force_iterations,
            logger=eval_logger
        )

    # Generation
    run_generation_samples(
        model, tokenizer, vocab_size, args.device, is_res,
        args.n_iterations, force_iterations, model_label, eval_logger
    )

    # Speed test
    tok_per_sec = run_speed_test(
        model, tokenizer, vocab_size, args.device, is_res,
        args.n_iterations, force_iterations, model_label, eval_logger
    )

    eval_logger.close()

    # Summary
    train_loss, train_ppl, _, _ = res_train
    hold_loss, hold_ppl, _, gate_var = res_holdout
    gen_gap = hold_ppl - train_ppl

    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_label}")
    print(f"{'='*70}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params:      {trainable:,}")
    print(f"  Train PPL:   {train_ppl:.2f}")
    print(f"  Holdout PPL: {hold_ppl:.2f}")
    print(f"  Gen Gap:     {gen_gap:+.2f}")
    if gate_var > 0:
        print(f"  Gate Var:    {gate_var:.6f}")
    print(f"  Speed:       {tok_per_sec:.1f} tok/s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
