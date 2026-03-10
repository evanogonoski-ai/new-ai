#!/usr/bin/env python3
"""V3 Biological Constraints Experiments.

Tests biological constraints on entropy gating to improve generalization:
  V3-B: Gate floor (baseline firing rate)
  V3-A: Gate temperature (lateral inhibition / range compression)
  V3-C: Diversity regularization (homeostatic plasticity)
  V3-D: Noise injection (varied replay)
  V3-Combined: Best of each

Usage:
  # Experiment 1: Gate floor variants
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-B --gate-floor 0.10
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-B --gate-floor 0.05
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-B --gate-floor 0.15

  # Experiment 2: Gate temperature variants
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-A --gate-min 0.15 --gate-max 0.85
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-A --gate-min 0.10 --gate-max 0.90
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-A --gate-min 0.20 --gate-max 0.80

  # Experiment 3: Diversity regularization variants
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-C --diversity-lambda 0.10
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-C --diversity-lambda 0.01
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-C --diversity-lambda 1.00

  # Experiment 4: Noise injection variants
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-D --noise-scale 0.05
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-D --noise-scale 0.01
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-D --noise-scale 0.10

  # Experiment 5: Combined
  python neurogenesis_v2/scripts/train_v3.py --experiment V3-Combined \
    --gate-floor 0.10 --gate-min 0.15 --gate-max 0.85 \
    --diversity-lambda 0.10 --noise-scale 0.05

  # Eval-only (load checkpoint, run evaluation)
  python neurogenesis_v2/scripts/train_v3.py --eval-only --checkpoint checkpoints_v3/V3-B_010_final.pt
"""
import argparse
import os
import sys
import time
import math
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.resonance_model import ResonanceModel
from neurogenesis_v2.model.mitosis import MitosisEngine
from neurogenesis_v2.training.losses import (
    gate_sparsity_loss, ponder_cost_loss, multitimescale_loss,
    gate_diversity_loss
)
from neurogenesis_v2.training.category_dataset import (
    CategoryTextDataset, CATEGORY_NAMES, build_train_holdout_datasets
)
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis.tokenizer.bpe import load_tokenizer


# ─── Locked config (unchanged from V2) ──────────────────────────────────────
LOCKED_CONFIG = dict(
    d_model=128, n_heads=8, d_head=16, vocab_size=8192, max_seq_len=128,
    d_ffn_start=512, d_ffn_max=1024,
    convergence_threshold=0.18, max_iterations=8, momentum_beta_init=0.05,
    mitosis_gradient_var_threshold=0.00005, mitosis_check_interval=500,
    gate_sparsity_lambda=0.01, ponder_cost_lambda=0.01,
    surface_loss_weight=0.3, semantic_loss_weight=0.1,
    gradient_clip=1.0, weight_decay=0.01,
)

TOTAL_STEPS = 2000
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 1000  # save periodic checkpoints

TOKENIZER_PATH = 'tokenizer/expanded_tokenizer.json'

HOLDOUT_AUTHORS = {
    'plato_apology': 'Plato (Apology)',
    'plato_republic': 'Plato (Republic)',
    'plato_symposium': 'Plato (Symposium)',
    'jane_austen_pride_and_prejudice': 'Jane Austen',
    'charles_darwin_origin_of_species': 'Charles Darwin',
}

# V2.2 reference results (from completed experiment)
V2_REFERENCE = {
    'baseline': {'train_ppl': 1.13, 'holdout_ppl': 1.65, 'gen_gap': 0.52},
    'resonance': {'train_ppl': 1.01, 'holdout_ppl': 1.72, 'gen_gap': 0.71,
                  'gate_var': 0.0014},
}


def variant_name(args):
    """Generate a short variant name from args."""
    parts = [args.experiment]
    if args.gate_floor > 0:
        parts.append(f"floor{args.gate_floor:.3f}".rstrip('0').rstrip('.'))
    if args.gate_min > 0 or args.gate_max < 1.0:
        parts.append(f"range{args.gate_min:.2f}-{args.gate_max:.2f}")
    if args.diversity_lambda > 0:
        parts.append(f"div{args.diversity_lambda}")
    if args.noise_scale > 0:
        parts.append(f"noise{args.noise_scale}")
    return '_'.join(parts)


# ─── Training ────────────────────────────────────────────────────────────────

def train_resonance(config, dataset, num_steps, device, checkpoint_dir, name, log_file=None):
    """Train Resonance model with V3 modifications."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ResonanceModel(config).to(device)

    with torch.no_grad():
        model.block.residual1.beta.fill_(-2.94)
        model.block.residual2.beta.fill_(-2.94)

    total_p = model.count_parameters()
    active_p = model.count_active_parameters()

    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log(f"Resonance model: {total_p:,} total, {active_p:,} active params")
    log(f"V3 config: gate_floor={config.gate_floor}, gate_min={config.gate_min}, "
        f"gate_max={config.gate_max}, diversity_lambda={config.gate_diversity_lambda}, "
        f"noise_scale={config.noise_scale}")

    config.mitosis_enabled = True
    engine = MitosisEngine(model.block.ffn, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=config.weight_decay)
    model.train()

    step = 0
    losses = []
    start = time.time()

    log(f"\n{'='*60}")
    log(f"V3 RESONANCE TRAINING [{name}]: {num_steps} steps")
    log(f"Training samples: {len(dataset):,} (holdout authors excluded)")
    log(f"{'='*60}")

    while step < num_steps:
        for input_ids, target_ids, cat_idx in loader:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

            output = model(
                input_ids, max_iterations=8, use_momentum=True,
                return_all_states=True,
            )

            ce_loss = F.cross_entropy(output['logits'], target_ids[:, -1])
            total_loss = ce_loss

            if output['all_gates']:
                all_gates = torch.stack(output['all_gates'])
                gs_loss = gate_sparsity_loss(all_gates, config.gate_sparsity_lambda)
                total_loss = total_loss + gs_loss

                # V3-C: Diversity regularization
                if config.gate_diversity_lambda > 0:
                    last_gates = output['all_gates'][-1]  # (B, H)
                    div_loss = gate_diversity_loss(last_gates, config.gate_diversity_lambda)
                    total_loss = total_loss + div_loss

            p_loss = ponder_cost_loss(output['ponder_cost'], config.ponder_cost_lambda)
            total_loss = total_loss + p_loss

            mt_loss = multitimescale_loss(
                output, target_ids[:, -1],
                config.surface_loss_weight, config.semantic_loss_weight
            )
            total_loss = total_loss + mt_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            # Mitosis
            engine.record_gradients()
            mresult = engine.check_and_apply(step)
            if mresult['mitosis'] > 0 or mresult['pruned'] > 0:
                log(f"  [Mitosis] step {step}: +{mresult['mitosis']} split, "
                    f"-{mresult['pruned']} pruned, alive={mresult['alive']}")

            losses.append(ce_loss.item())
            step += 1

            if step % 500 == 0:
                avg = sum(losses[-500:]) / min(500, len(losses))
                ppl = math.exp(min(avg, 20))
                iters = output['num_iterations']
                gate_mean = output['all_gates'][-1].mean().item() if output['all_gates'] else 0
                gate_std = output['all_gates'][-1].std().item() if output['all_gates'] else 0
                alive = model.block.ffn.num_alive
                speed = step / (time.time() - start)
                log(f"  Step {step}/{num_steps}: loss={avg:.4f}, ppl={ppl:.2f}, "
                    f"iters={iters}, gates={gate_mean:.3f}±{gate_std:.3f}, alive={alive}, "
                    f"{speed:.1f} it/s")

            # Periodic checkpoint
            if step % CHECKPOINT_INTERVAL == 0 and step < num_steps:
                ckpt_path = os.path.join(checkpoint_dir, f"{name}_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                log(f"  Checkpoint saved: {ckpt_path}")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - start
    log(f"\nResonance done: loss={avg:.4f}, ppl={ppl:.2f} ({elapsed:.0f}s)")

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, f"{name}_final.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final checkpoint: {final_path}")

    return model


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_model(model, dataset, config, label, device='cpu', is_resonance=True, log_fn=print):
    """Evaluate model, returning per-category metrics with gate stats."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()

    if is_resonance:
        per_cat = defaultdict(lambda: {'loss': [], 'count': 0, 'gates': [], 'iters': []})
    else:
        per_cat = defaultdict(lambda: {'loss': [], 'count': 0})

    total_loss = 0.0
    total_count = 0
    all_gate_values = []  # for computing cross-category variance

    with torch.no_grad():
        for input_ids, target_ids, cat_idx in loader:
            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

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

    # Compute gate variance across all samples
    gate_variance = 0.0
    if all_gate_values:
        all_gates_tensor = torch.cat(all_gate_values, dim=0)  # (N, H)
        gate_variance = all_gates_tensor.var(dim=0).mean().item()

    log_fn(f"\n  {label}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f} ({total_count} batches)")
    if is_resonance:
        log_fn(f"    Gate variance (across samples): {gate_variance:.6f}")

    for c in sorted(per_cat.keys()):
        cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')
        m = per_cat[c]
        cat_loss = sum(m['loss']) / len(m['loss']) if m['loss'] else 0
        cat_ppl = math.exp(min(cat_loss, 20))
        extra = ''
        if is_resonance and 'gates' in m and m['gates']:
            g = sum(m['gates']) / len(m['gates'])
            it = sum(m['iters']) / len(m['iters'])
            extra = f', gates={g:.3f}, iters={it:.1f}'
        log_fn(f"    {cat_name:<12}: loss={cat_loss:.4f}, ppl={cat_ppl:.2f}{extra}")

    return avg_loss, avg_ppl, dict(per_cat), gate_variance


def eval_holdout_by_author(model, holdout_jsonl, tokenizer, config, label,
                            device='cpu', is_resonance=True, log_fn=print):
    """Evaluate on holdout data, broken down by author."""
    log_fn(f"\n  {label} — Per-Author Holdout Breakdown:")

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

            for text in texts:
                enc = tokenizer.encode(text)
                ids = [min(i, config.vocab_size - 1) for i in enc.ids]

                if len(ids) < config.max_seq_len + 1:
                    ids = ids + [0] * (config.max_seq_len + 1 - len(ids))
                ids = ids[:config.max_seq_len + 1]

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

            avg_loss = sum(author_losses) / len(author_losses)
            avg_ppl = math.exp(min(avg_loss, 20))
            display_name = HOLDOUT_AUTHORS.get(source, source)

            extra = ''
            if is_resonance and author_gates:
                g = sum(author_gates) / len(author_gates)
                it = sum(author_iters) / len(author_iters)
                extra = f', gates={g:.3f}, iters={it:.1f}'

            log_fn(f"    {display_name:<25}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f}, "
                   f"n={len(author_losses)}{extra}")

            results[source] = {
                'loss': avg_loss, 'ppl': avg_ppl, 'n': len(author_losses),
                'display_name': display_name,
            }
            if is_resonance and author_gates:
                results[source]['gates'] = sum(author_gates) / len(author_gates)
                results[source]['iters'] = sum(author_iters) / len(author_iters)

    return results


# ─── Report ──────────────────────────────────────────────────────────────────

def generate_report(name, res_train, res_holdout, res_authors, args):
    """Generate V3 experiment report for a single variant."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"V3 EXPERIMENT REPORT: {name}")
    lines.append(f"Author-Holdout Test: Plato, Austen, Darwin")
    lines.append("=" * 70)

    lines.append(f"\nModifications:")
    if args.gate_floor > 0:
        lines.append(f"  Gate floor: {args.gate_floor}")
    if args.gate_min > 0 or args.gate_max < 1.0:
        lines.append(f"  Gate range: [{args.gate_min}, {args.gate_max}]")
    if args.diversity_lambda > 0:
        lines.append(f"  Diversity lambda: {args.diversity_lambda}")
    if args.noise_scale > 0:
        lines.append(f"  Noise scale: {args.noise_scale}")
    if args.gate_floor == 0 and args.gate_min == 0 and args.gate_max == 1.0 and \
       args.diversity_lambda == 0 and args.noise_scale == 0:
        lines.append(f"  (No modifications — V2 reference run)")

    res_train_loss, res_train_ppl, res_train_cats, train_gate_var = res_train
    res_hold_loss, res_hold_ppl, res_hold_cats, hold_gate_var = res_holdout

    # Section 1: Train PPL
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 1: TRAINING DATA PPL")
    lines.append(f"{'='*70}")
    lines.append(f"  {name} train PPL:    {res_train_ppl:.2f} (loss={res_train_loss:.4f})")
    lines.append(f"  V2 Resonance ref:    {V2_REFERENCE['resonance']['train_ppl']:.2f}")
    lines.append(f"  Baseline ref:        {V2_REFERENCE['baseline']['train_ppl']:.2f}")

    # Section 2: Holdout PPL (THE KEY METRIC)
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 2: HOLDOUT PPL (THE KEY METRIC)")
    lines.append(f"{'='*70}")
    lines.append(f"  {name} holdout PPL:  {res_hold_ppl:.2f} (loss={res_hold_loss:.4f})")
    lines.append(f"  V2 Resonance ref:    {V2_REFERENCE['resonance']['holdout_ppl']:.2f}")
    lines.append(f"  Baseline ref:        {V2_REFERENCE['baseline']['holdout_ppl']:.2f}")

    base_ref = V2_REFERENCE['baseline']['holdout_ppl']
    v2_ref = V2_REFERENCE['resonance']['holdout_ppl']
    if res_hold_ppl < base_ref:
        lines.append(f"\n  VERDICT: BEATS BASELINE ({res_hold_ppl:.2f} < {base_ref:.2f})")
    elif res_hold_ppl < v2_ref:
        lines.append(f"\n  VERDICT: Improves over V2 ({res_hold_ppl:.2f} < {v2_ref:.2f}) "
                      f"but does not beat baseline ({base_ref:.2f})")
    else:
        lines.append(f"\n  VERDICT: No improvement over V2 ({res_hold_ppl:.2f} vs {v2_ref:.2f})")

    # Section 3: Generalization gap
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 3: GENERALIZATION GAP (holdout PPL - train PPL)")
    lines.append(f"{'='*70}")
    gen_gap = res_hold_ppl - res_train_ppl
    lines.append(f"  {name} gap:          {gen_gap:+.2f} ({res_train_ppl:.2f} → {res_hold_ppl:.2f})")
    lines.append(f"  V2 Resonance gap:    {V2_REFERENCE['resonance']['gen_gap']:+.2f}")
    lines.append(f"  Baseline gap:        {V2_REFERENCE['baseline']['gen_gap']:+.2f}")

    # Section 4: Gate variance
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 4: GATE VARIANCE (adaptive behavior)")
    lines.append(f"{'='*70}")
    lines.append(f"  {name} train gate var:   {train_gate_var:.6f}")
    lines.append(f"  {name} holdout gate var: {hold_gate_var:.6f}")
    lines.append(f"  V2 Resonance gate var:   {V2_REFERENCE['resonance']['gate_var']:.6f}")

    if train_gate_var > 0.01:
        lines.append(f"\n  SIGNAL: Gate variance meaningfully increased (>{0.01})")
    elif train_gate_var > V2_REFERENCE['resonance']['gate_var'] * 2:
        lines.append(f"\n  Modest improvement in gate variance")
    else:
        lines.append(f"\n  Gate variance not meaningfully changed")

    # Per-category gate breakdown
    if res_train_cats:
        lines.append(f"\n  Per-category gates (train):")
        for c in sorted(res_train_cats.keys()):
            cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')
            m = res_train_cats[c]
            if 'gates' in m and m['gates']:
                g = sum(m['gates']) / len(m['gates'])
                it = sum(m['iters']) / len(m['iters'])
                lines.append(f"    {cat_name:<12}: gates={g:.4f}, iters={it:.1f}")

    if res_hold_cats:
        lines.append(f"\n  Per-category gates (holdout):")
        for c in sorted(res_hold_cats.keys()):
            cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')
            m = res_hold_cats[c]
            if 'gates' in m and m['gates']:
                g = sum(m['gates']) / len(m['gates'])
                it = sum(m['iters']) / len(m['iters'])
                lines.append(f"    {cat_name:<12}: gates={g:.4f}, iters={it:.1f}")

    # Section 5: Per-author holdout
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 5: PER-AUTHOR HOLDOUT BREAKDOWN")
    lines.append(f"{'='*70}")

    author_groups = {
        'Plato': ['plato_apology', 'plato_republic', 'plato_symposium'],
        'Jane Austen': ['jane_austen_pride_and_prejudice'],
        'Charles Darwin': ['charles_darwin_origin_of_species'],
    }

    header = f"  {'Author':<20} {'V3 PPL':>10} {'Base ref':>10} {'V2 Res ref':>10}"
    lines.append(header)
    lines.append(f"  {'-'*55}")

    for author, sources in author_groups.items():
        res_losses = []
        res_gates_all = []
        res_iters_all = []

        for src in sources:
            if src in res_authors:
                res_losses.append(res_authors[src]['loss'])
                if 'gates' in res_authors[src]:
                    res_gates_all.append(res_authors[src]['gates'])
                    res_iters_all.append(res_authors[src]['iters'])

        if res_losses:
            r_loss = sum(res_losses) / len(res_losses)
            r_ppl = math.exp(min(r_loss, 20))
            lines.append(f"  {author:<20} {r_ppl:>10.2f} {'1.65':>10} {'1.72':>10}")

            if res_gates_all:
                g = sum(res_gates_all) / len(res_gates_all)
                it = sum(res_iters_all) / len(res_iters_all)
                lines.append(f"    (gates={g:.3f}, iters={it:.1f})")

    # Section 6: Adaptive behavior
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 6: ADAPTIVE BEHAVIOR — HOLDOUT vs TRAIN")
    lines.append(f"{'='*70}")

    if res_train_cats and res_hold_cats:
        train_gates = []
        train_iters = []
        for c in res_train_cats:
            m = res_train_cats[c]
            if 'gates' in m and m['gates']:
                train_gates.extend(m['gates'])
                train_iters.extend(m['iters'])

        hold_gates = []
        hold_iters = []
        for c in res_hold_cats:
            m = res_hold_cats[c]
            if 'gates' in m and m['gates']:
                hold_gates.extend(m['gates'])
                hold_iters.extend(m['iters'])

        if train_gates and hold_gates:
            tg = sum(train_gates) / len(train_gates)
            ti = sum(train_iters) / len(train_iters)
            hg = sum(hold_gates) / len(hold_gates)
            hi = sum(hold_iters) / len(hold_iters)

            lines.append(f"  Train:   gates={tg:.3f}, iters={ti:.1f}")
            lines.append(f"  Holdout: gates={hg:.3f}, iters={hi:.1f}")
            lines.append(f"  Delta:   gates={hg-tg:+.3f}, iters={hi-ti:+.1f}")

    lines.append(f"\n{'='*70}")
    lines.append("END OF REPORT")
    lines.append(f"{'='*70}")

    return '\n'.join(lines)


def append_to_summary(name, res_train, res_holdout, res_authors, args, summary_path):
    """Append one row to the running V3 summary table."""
    res_train_loss, res_train_ppl, _, train_gate_var = res_train
    res_hold_loss, res_hold_ppl, _, hold_gate_var = res_holdout
    gen_gap = res_hold_ppl - res_train_ppl

    # Find best/worst author
    best_author = worst_author = None
    best_ppl = float('inf')
    worst_ppl = 0

    author_groups = {
        'Plato': ['plato_apology', 'plato_republic', 'plato_symposium'],
        'Austen': ['jane_austen_pride_and_prejudice'],
        'Darwin': ['charles_darwin_origin_of_species'],
    }

    for author, sources in author_groups.items():
        losses = [res_authors[s]['loss'] for s in sources if s in res_authors]
        if losses:
            avg = sum(losses) / len(losses)
            ppl = math.exp(min(avg, 20))
            if ppl < best_ppl:
                best_ppl = ppl
                best_author = f"{author} {ppl:.2f}"
            if ppl > worst_ppl:
                worst_ppl = ppl
                worst_author = f"{author} {ppl:.2f}"

    # Write header if file doesn't exist
    write_header = not os.path.exists(summary_path)

    with open(summary_path, 'a') as f:
        if write_header:
            f.write("=== V3 EXPERIMENT SUMMARY ===\n\n")
            header = (f"{'Model':<22} | {'Train PPL':>9} | {'Holdout PPL':>11} | "
                      f"{'Gen Gap':>7} | {'Gate Var':>10} | {'Best Author':>15} | {'Worst Author':>15}")
            f.write(header + '\n')
            f.write('-' * len(header) + '\n')

            # Write reference rows
            f.write(f"{'Baseline (ref)':<22} | {'1.13':>9} | {'1.65':>11} | "
                    f"{'+0.52':>7} | {'N/A':>10} | {'Austen 1.26':>15} | {'Plato 1.68':>15}\n")
            f.write(f"{'V2 Resonance (ref)':<22} | {'1.01':>9} | {'1.72':>11} | "
                    f"{'+0.71':>7} | {'0.0014':>10} | {'Austen 1.27':>15} | {'Plato 1.74':>15}\n")

        f.write(f"{name:<22} | {res_train_ppl:>9.2f} | {res_hold_ppl:>11.2f} | "
                f"{gen_gap:>+7.2f} | {train_gate_var:>10.6f} | "
                f"{(best_author or 'N/A'):>15} | {(worst_author or 'N/A'):>15}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V3 Biological Constraints Experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['V3-B', 'V3-A', 'V3-C', 'V3-D', 'V3-Combined'],
                        help='Experiment name')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS)

    # V3-B: Gate floor
    parser.add_argument('--gate-floor', type=float, default=0.0,
                        help='Minimum gate value (biological baseline firing)')
    # V3-A: Gate temperature (range compression)
    parser.add_argument('--gate-min', type=float, default=0.0,
                        help='Gate range lower bound')
    parser.add_argument('--gate-max', type=float, default=1.0,
                        help='Gate range upper bound')
    # V3-C: Diversity regularization
    parser.add_argument('--diversity-lambda', type=float, default=0.0,
                        help='Gate diversity regularization strength')
    # V3-D: Noise injection
    parser.add_argument('--noise-scale', type=float, default=0.0,
                        help='Embedding noise injection scale')

    # Eval-only mode
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, load from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for eval-only mode')

    args = parser.parse_args()

    # Config
    config = ResonanceConfig()
    for key, val in LOCKED_CONFIG.items():
        setattr(config, key, val)

    # Apply V3 modifications to config
    config.gate_floor = args.gate_floor
    config.gate_min = args.gate_min
    config.gate_max = args.gate_max
    config.gate_diversity_lambda = args.diversity_lambda
    config.noise_scale = args.noise_scale

    # Load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: vocab={config.vocab_size}")

    # Load datasets
    train_dataset, holdout_dataset = build_train_holdout_datasets(
        tokenizer, seq_len=config.max_seq_len
    )

    name = variant_name(args)
    checkpoint_dir = 'checkpoints_v3'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ─── TRAINING or LOADING ─────────────────────────────────────────────

    log_path = f'logs/v3_{name}_train.txt'
    log_file = open(log_path, 'w')

    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(checkpoint_dir, f"{name}_final.pt")
        print(f"\n--- Loading from checkpoint: {ckpt} ---")
        model = ResonanceModel(config).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
        print(f"Loaded {name} from {ckpt}")
    else:
        model = train_resonance(
            config, train_dataset, args.steps, args.device,
            checkpoint_dir, name, log_file
        )

    log_file.close()

    # ─── EVALUATION ──────────────────────────────────────────────────────

    eval_log_path = f'logs/v3_{name}_eval.txt'
    eval_log = open(eval_log_path, 'w')

    def log(msg):
        print(msg)
        eval_log.write(msg + '\n')
        eval_log.flush()

    log(f"\n{'='*70}")
    log(f"EVALUATION: {name}")
    log(f"{'='*70}")

    # Eval on train data
    res_train = eval_model(
        model, train_dataset, config, f"{name} on TRAIN",
        args.device, is_resonance=True, log_fn=log
    )

    # Eval on holdout data
    res_holdout = eval_model(
        model, holdout_dataset, config, f"{name} on HOLDOUT",
        args.device, is_resonance=True, log_fn=log
    )

    # Per-author holdout
    res_authors = eval_holdout_by_author(
        model, 'data/holdout_corpus.jsonl', tokenizer, config,
        f"{name} per-author", args.device, is_resonance=True, log_fn=log
    )

    eval_log.close()

    # ─── REPORT ──────────────────────────────────────────────────────────

    report = generate_report(name, res_train, res_holdout, res_authors, args)
    print(f"\n{report}")

    report_path = f'logs/v3_{name}_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Append to summary table
    summary_path = 'logs/v3_summary.txt'
    append_to_summary(name, res_train, res_holdout, res_authors, args, summary_path)
    print(f"Summary updated: {summary_path}")


if __name__ == '__main__':
    main()
