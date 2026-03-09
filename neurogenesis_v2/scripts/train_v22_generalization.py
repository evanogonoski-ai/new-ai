#!/usr/bin/env python3
"""V2.2 Generalization Experiment — Author-Holdout Test.

Trains on corpus with Plato, Austen, and Darwin held out entirely.
Evaluates on held-out authors to test whether entropy gating aids
generalization to unseen writing styles.

Key hypothesis: If entropy gating is adaptive, Resonance should
generalize better to unseen authors than Baseline, even if both
memorize training data equally well (PPL ~1.1 on train).

Usage:
  python neurogenesis_v2/scripts/train_v22_generalization.py --mode both --device cpu
  python neurogenesis_v2/scripts/train_v22_generalization.py --mode resonance --device cpu
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
from neurogenesis_v2.training.losses import gate_sparsity_loss, ponder_cost_loss, multitimescale_loss
from neurogenesis_v2.training.category_dataset import (
    CategoryTextDataset, CATEGORY_NAMES, build_train_holdout_datasets
)
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis.tokenizer.bpe import load_tokenizer


# ─── Locked config (same as V2.1 — no architecture changes) ─────────────────
LOCKED_CONFIG = dict(
    d_model=128, n_heads=8, d_head=16, vocab_size=8192, max_seq_len=128,
    d_ffn_start=512, d_ffn_max=1024,
    convergence_threshold=0.18, max_iterations=8, momentum_beta_init=0.05,
    mitosis_gradient_var_threshold=0.00005, mitosis_check_interval=500,
    gate_sparsity_lambda=0.01, ponder_cost_lambda=0.01,
    surface_loss_weight=0.3, semantic_loss_weight=0.1,
    gradient_clip=1.0, weight_decay=0.01,
)

TOTAL_STEPS = 15000
LEARNING_RATE = 3e-4
BATCH_SIZE = 32

TOKENIZER_PATH = 'tokenizer/expanded_tokenizer.json'

# Holdout author source names (must match source field in JSONL)
HOLDOUT_AUTHORS = {
    'plato_apology': 'Plato (Apology)',
    'plato_republic': 'Plato (Republic)',
    'plato_symposium': 'Plato (Symposium)',
    'jane_austen_pride_and_prejudice': 'Jane Austen',
    'charles_darwin_origin_of_species': 'Charles Darwin',
}


# ─── Training ────────────────────────────────────────────────────────────────

def train_resonance(config, dataset, num_steps, device='cpu'):
    """Train Resonance model on train-only data (no holdout contamination)."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ResonanceModel(config).to(device)

    with torch.no_grad():
        model.block.residual1.beta.fill_(-2.94)
        model.block.residual2.beta.fill_(-2.94)

    total_p = model.count_parameters()
    active_p = model.count_active_parameters()
    print(f"Resonance model: {total_p:,} total, {active_p:,} active params")

    config.mitosis_enabled = True
    engine = MitosisEngine(model.block.ffn, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=config.weight_decay)
    model.train()

    step = 0
    losses = []
    start = time.time()

    print(f"\n{'='*60}")
    print(f"V2.2 RESONANCE TRAINING (holdout-clean): {num_steps} steps")
    print(f"Training samples: {len(dataset):,} (holdout authors excluded)")
    print(f"{'='*60}")

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
                print(f"  [Mitosis] step {step}: +{mresult['mitosis']} split, "
                      f"-{mresult['pruned']} pruned, alive={mresult['alive']}")

            losses.append(ce_loss.item())
            step += 1

            if step % 500 == 0:
                avg = sum(losses[-500:]) / min(500, len(losses))
                ppl = math.exp(min(avg, 20))
                iters = output['num_iterations']
                gate_mean = output['all_gates'][-1].mean().item() if output['all_gates'] else 0
                alive = model.block.ffn.num_alive
                speed = step / (time.time() - start)
                print(f"  Step {step}/{num_steps}: loss={avg:.3f}, ppl={ppl:.2f}, "
                      f"iters={iters}, gates={gate_mean:.3f}, alive={alive}, "
                      f"{speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - start
    print(f"\nResonance done: loss={avg:.3f}, ppl={ppl:.2f} ({elapsed:.0f}s)")

    return model


def train_baseline(config, dataset, num_steps, device='cpu'):
    """Train baseline transformer on train-only data."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = BaselineTransformer(
        vocab_size=config.vocab_size, d_model=128,
        n_heads=8, n_layers=2, d_ffn=512, max_seq_len=128
    ).to(device)
    print(f"Baseline Transformer: {model.count_parameters():,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=config.weight_decay)
    model.train()

    step = 0
    losses = []
    start = time.time()

    print(f"\n{'='*60}")
    print(f"V2.2 BASELINE TRAINING (holdout-clean): {num_steps} steps")
    print(f"Training samples: {len(dataset):,} (holdout authors excluded)")
    print(f"{'='*60}")

    while step < num_steps:
        for input_ids, target_ids, cat_idx in loader:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids[:, -1])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 500 == 0:
                avg = sum(losses[-500:]) / min(500, len(losses))
                ppl = math.exp(min(avg, 20))
                speed = step / (time.time() - start)
                print(f"  Baseline step {step}/{num_steps}: loss={avg:.3f}, "
                      f"ppl={ppl:.2f}, {speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - start
    print(f"Baseline done: loss={avg:.3f}, ppl={ppl:.2f} ({elapsed:.0f}s)")

    return model


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_model(model, dataset, config, label, device='cpu', is_resonance=True):
    """Evaluate a model on a dataset, returning per-category metrics.

    For Resonance models, also tracks gates and iterations.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()

    per_cat = defaultdict(lambda: {'loss': [], 'count': 0})
    if is_resonance:
        for k in per_cat:
            per_cat[k].update({'gates': [], 'iters': []})

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for input_ids, target_ids, cat_idx in loader:
            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

            if is_resonance:
                output = model(input_ids, max_iterations=8, use_momentum=True)
                logits = output['logits']
                gate_mean = output['all_gates'][-1].mean().item() if output['all_gates'] else 0
                iters = output['num_iterations']
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

    print(f"\n  {label}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f} ({total_count} batches)")
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
        print(f"    {cat_name:<12}: loss={cat_loss:.4f}, ppl={cat_ppl:.2f}{extra}")

    return avg_loss, avg_ppl, dict(per_cat)


def eval_holdout_by_author(model, holdout_jsonl, tokenizer, config, label,
                            device='cpu', is_resonance=True):
    """Evaluate on holdout data, broken down by individual author.

    Reads the holdout JSONL directly to get per-source attribution,
    since the .pt file doesn't store source info.
    """
    print(f"\n  {label} — Per-Author Holdout Breakdown:")

    # Group holdout entries by source
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

            print(f"    {display_name:<25}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f}, "
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

def generate_report(res_train, res_holdout, base_train, base_holdout,
                    res_authors, base_authors):
    """Generate the V2.2 generalization analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("V2.2 GENERALIZATION EXPERIMENT REPORT")
    lines.append("Author-Holdout Test: Plato, Austen, Darwin")
    lines.append("=" * 70)

    # Unpack: (loss, ppl, per_cat)
    res_train_loss, res_train_ppl, _ = res_train
    res_hold_loss, res_hold_ppl, _ = res_holdout
    base_train_loss, base_train_ppl, _ = base_train
    base_hold_loss, base_hold_ppl, _ = base_holdout

    # Section 1: Train PPL (sanity check)
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 1: TRAINING DATA PPL (sanity check — expect ~1.1 for both)")
    lines.append(f"{'='*70}")
    lines.append(f"  Resonance train PPL:  {res_train_ppl:.2f} (loss={res_train_loss:.4f})")
    lines.append(f"  Baseline train PPL:   {base_train_ppl:.2f} (loss={base_train_loss:.4f})")
    train_loss_ratio = base_train_loss / res_train_loss if res_train_loss > 0 else 0
    lines.append(f"  Loss ratio (base/res): {train_loss_ratio:.2f}x")

    # Section 2: Holdout PPL (THE KEY METRIC)
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 2: HOLDOUT PPL (THE KEY METRIC)")
    lines.append("  Models evaluated on authors never seen during training")
    lines.append(f"{'='*70}")
    lines.append(f"  Resonance holdout PPL:  {res_hold_ppl:.2f} (loss={res_hold_loss:.4f})")
    lines.append(f"  Baseline holdout PPL:   {base_hold_ppl:.2f} (loss={base_hold_loss:.4f})")
    hold_loss_ratio = base_hold_loss / res_hold_loss if res_hold_loss > 0 else 0
    lines.append(f"  Loss ratio (base/res): {hold_loss_ratio:.2f}x")

    if base_hold_ppl > res_hold_ppl * 1.05:
        lines.append(f"\n  VERDICT: Resonance generalizes better ({base_hold_ppl:.2f} vs "
                      f"{res_hold_ppl:.2f} PPL)")
    elif res_hold_ppl > base_hold_ppl * 1.05:
        lines.append(f"\n  VERDICT: Baseline generalizes better ({base_hold_ppl:.2f} vs "
                      f"{res_hold_ppl:.2f} PPL)")
    else:
        lines.append(f"\n  VERDICT: No significant generalization difference "
                      f"({res_hold_ppl:.2f} vs {base_hold_ppl:.2f})")

    # Section 3: Generalization gap
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 3: GENERALIZATION GAP (holdout PPL - train PPL)")
    lines.append(f"{'='*70}")
    res_gap = res_hold_ppl - res_train_ppl
    base_gap = base_hold_ppl - base_train_ppl
    lines.append(f"  Resonance gap: {res_gap:+.2f} PPL ({res_train_ppl:.2f} → {res_hold_ppl:.2f})")
    lines.append(f"  Baseline gap:  {base_gap:+.2f} PPL ({base_train_ppl:.2f} → {base_hold_ppl:.2f})")
    if abs(res_gap) < abs(base_gap):
        lines.append(f"  Resonance has smaller generalization penalty")
    elif abs(base_gap) < abs(res_gap):
        lines.append(f"  Baseline has smaller generalization penalty")
    else:
        lines.append(f"  Generalization penalty is similar")

    # Section 4: Per-author breakdown
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 4: PER-AUTHOR HOLDOUT BREAKDOWN")
    lines.append(f"{'='*70}")

    # Group by author (collapse Plato sub-works)
    author_groups = {
        'Plato': ['plato_apology', 'plato_republic', 'plato_symposium'],
        'Jane Austen': ['jane_austen_pride_and_prejudice'],
        'Charles Darwin': ['charles_darwin_origin_of_species'],
    }

    header = f"  {'Author':<20} {'Res PPL':>10} {'Base PPL':>10} {'Advantage':>12}"
    lines.append(header)
    lines.append(f"  {'-'*52}")

    for author, sources in author_groups.items():
        # Average over all sources for this author
        res_losses = []
        base_losses = []
        res_gates_all = []
        res_iters_all = []

        for src in sources:
            if src in res_authors:
                res_losses.append(res_authors[src]['loss'])
                if 'gates' in res_authors[src]:
                    res_gates_all.append(res_authors[src]['gates'])
                    res_iters_all.append(res_authors[src]['iters'])
            if src in base_authors:
                base_losses.append(base_authors[src]['loss'])

        if res_losses and base_losses:
            r_loss = sum(res_losses) / len(res_losses)
            b_loss = sum(base_losses) / len(base_losses)
            r_ppl = math.exp(min(r_loss, 20))
            b_ppl = math.exp(min(b_loss, 20))

            if b_ppl > r_ppl:
                adv = f"Res +{((b_ppl/r_ppl)-1)*100:.1f}%"
            elif r_ppl > b_ppl:
                adv = f"Base +{((r_ppl/b_ppl)-1)*100:.1f}%"
            else:
                adv = "tie"

            lines.append(f"  {author:<20} {r_ppl:>10.2f} {b_ppl:>10.2f} {adv:>12}")

            if res_gates_all:
                g = sum(res_gates_all) / len(res_gates_all)
                it = sum(res_iters_all) / len(res_iters_all)
                lines.append(f"    (Resonance gates={g:.3f}, iters={it:.1f})")

    # Section 5: Adaptive behavior on holdout vs train
    lines.append(f"\n{'='*70}")
    lines.append("SECTION 5: ADAPTIVE BEHAVIOR — HOLDOUT vs TRAIN")
    lines.append("  Does the model adjust gates/iters for unseen text?")
    lines.append(f"{'='*70}")

    _, _, res_train_cats = res_train
    _, _, res_hold_cats = res_holdout

    if res_train_cats and res_hold_cats:
        # Average gates/iters across train categories
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

            if abs(hg - tg) > 0.01 or abs(hi - ti) > 0.5:
                lines.append(f"\n  SIGNAL: Model adjusts computation for unseen text")
            else:
                lines.append(f"\n  No significant adaptation detected")

    lines.append(f"\n{'='*70}")
    lines.append("END OF REPORT")
    lines.append(f"{'='*70}")

    return '\n'.join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V2.2 Generalization Experiment')
    parser.add_argument('--mode', type=str, choices=['resonance', 'baseline', 'both'],
                        default='both')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    # Config
    config = ResonanceConfig()
    for key, val in LOCKED_CONFIG.items():
        setattr(config, key, val)

    # Load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: vocab={config.vocab_size}")

    # Load train and holdout datasets
    train_dataset, holdout_dataset = build_train_holdout_datasets(
        tokenizer, seq_len=config.max_seq_len
    )

    os.makedirs('checkpoints_v2.2', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    res_model = base_model = None

    # ─── TRAINING PHASE ──────────────────────────────────────────────────

    if args.mode in ('baseline', 'both'):
        base_model = train_baseline(config, train_dataset, args.steps, args.device)
        torch.save(base_model.state_dict(), 'checkpoints_v2.2/baseline_v22.pt')
        print("Saved checkpoints_v2.2/baseline_v22.pt")

    if args.mode in ('resonance', 'both'):
        res_model = train_resonance(config, train_dataset, args.steps, args.device)
        torch.save(res_model.state_dict(), 'checkpoints_v2.2/resonance_v22.pt')
        print("Saved checkpoints_v2.2/resonance_v22.pt")

    # ─── EVALUATION PHASE ────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("EVALUATION PHASE")
    print(f"{'='*70}")

    res_train_metrics = res_holdout_metrics = (0, 0, {})
    base_train_metrics = base_holdout_metrics = (0, 0, {})
    res_authors = base_authors = {}

    if base_model is not None:
        print("\n--- Baseline Evaluation ---")
        base_train_metrics = eval_model(
            base_model, train_dataset, config, "Baseline on TRAIN",
            args.device, is_resonance=False
        )
        base_holdout_metrics = eval_model(
            base_model, holdout_dataset, config, "Baseline on HOLDOUT",
            args.device, is_resonance=False
        )
        base_authors = eval_holdout_by_author(
            base_model, 'data/holdout_corpus.jsonl', tokenizer, config,
            "Baseline per-author", args.device, is_resonance=False
        )

    if res_model is not None:
        print("\n--- Resonance Evaluation ---")
        res_train_metrics = eval_model(
            res_model, train_dataset, config, "Resonance on TRAIN",
            args.device, is_resonance=True
        )
        res_holdout_metrics = eval_model(
            res_model, holdout_dataset, config, "Resonance on HOLDOUT",
            args.device, is_resonance=True
        )
        res_authors = eval_holdout_by_author(
            res_model, 'data/holdout_corpus.jsonl', tokenizer, config,
            "Resonance per-author", args.device, is_resonance=True
        )

    # ─── REPORT ───────────────────────────────────────────────────────────

    if args.mode == 'both' and res_model is not None and base_model is not None:
        report = generate_report(
            res_train_metrics, res_holdout_metrics,
            base_train_metrics, base_holdout_metrics,
            res_authors, base_authors
        )
        print(f"\n{report}")

        report_path = 'logs/v2.2_generalization_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
