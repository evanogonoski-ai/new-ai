#!/usr/bin/env python3
"""V2.1 training script — expanded corpus with per-category instrumentation.

Runs two experiments:
  B1: Resonance vs Baseline on expanded corpus (scale test)
  B2: Per-category gate/iter analysis (adaptive behavior test)

Usage:
  python neurogenesis_v2/scripts/train_v21.py --mode resonance --device cpu
  python neurogenesis_v2/scripts/train_v21.py --mode baseline --device cpu
  python neurogenesis_v2/scripts/train_v21.py --mode both --device cpu
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
from neurogenesis_v2.training.category_dataset import CategoryTextDataset, CATEGORY_NAMES
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis.tokenizer.bpe import load_tokenizer


# ─── Locked config ──────────────────────────────────────────────────────────
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

CORPUS_PATH = 'data/expanded_corpus.jsonl'
TOKENIZER_PATH = 'tokenizer/expanded_tokenizer.json'


def train_resonance(config, dataset, num_steps, device='cpu', mitosis_enabled=True):
    """Train Resonance model with per-category instrumentation."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ResonanceModel(config).to(device)

    # Apply tuned momentum
    with torch.no_grad():
        model.block.residual1.beta.fill_(-2.94)  # sigmoid(-2.94) ≈ 0.05
        model.block.residual2.beta.fill_(-2.94)

    total_p = model.count_parameters()
    active_p = model.count_active_parameters()
    print(f"Resonance model: {total_p:,} total params, {active_p:,} active params")

    # Mitosis engine
    engine = None
    if mitosis_enabled:
        config.mitosis_enabled = True
        engine = MitosisEngine(model.block.ffn, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=config.weight_decay)
    model.train()

    step = 0
    losses = []
    per_cat_metrics = defaultdict(lambda: {'loss': [], 'gates': [], 'iters': []})
    start = time.time()

    print(f"\n{'='*60}")
    print(f"V2.1 RESONANCE TRAINING: {num_steps} steps, lr={LEARNING_RATE}")
    print(f"Mitosis: {'enabled' if mitosis_enabled else 'disabled'}")
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

            # Main loss
            ce_loss = F.cross_entropy(output['logits'], target_ids[:, -1])
            total_loss = ce_loss

            # Auxiliary losses
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
            if engine is not None:
                engine.record_gradients()
                mresult = engine.check_and_apply(step)
                if mresult['mitosis'] > 0 or mresult['pruned'] > 0:
                    print(f"  [Mitosis] step {step}: +{mresult['mitosis']} split, "
                          f"-{mresult['pruned']} pruned, alive={mresult['alive']}")

            # Per-category tracking
            gate_mean = output['all_gates'][-1].mean().item() if output['all_gates'] else 0
            iters = output['num_iterations']

            for i in range(len(cat_idx)):
                c = cat_idx[i].item()
                per_cat_metrics[c]['loss'].append(ce_loss.item())
                per_cat_metrics[c]['gates'].append(gate_mean)
                per_cat_metrics[c]['iters'].append(iters)

            losses.append(ce_loss.item())
            step += 1

            if step % 500 == 0:
                avg = sum(losses[-500:]) / min(500, len(losses))
                ppl = math.exp(min(avg, 20))
                speed = step / (time.time() - start)
                alive = model.block.ffn.num_alive
                print(f"  Step {step}/{num_steps}: loss={avg:.3f}, ppl={ppl:.1f}, "
                      f"iters={iters}, gates={gate_mean:.2f}, alive={alive}, "
                      f"{speed:.1f} it/s")

                # Per-category report
                for c in sorted(per_cat_metrics.keys()):
                    cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')
                    recent = per_cat_metrics[c]
                    n = len(recent['loss'])
                    if n > 0:
                        cat_loss = sum(recent['loss'][-200:]) / min(200, n)
                        cat_gates = sum(recent['gates'][-200:]) / min(200, n)
                        cat_iters = sum(recent['iters'][-200:]) / min(200, n)
                        print(f"    {cat_name:<12}: loss={cat_loss:.3f}, "
                              f"gates={cat_gates:.3f}, iters={cat_iters:.1f}")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    print(f"\nResonance done: loss={avg:.3f}, ppl={ppl:.1f}")

    return model, avg, ppl, per_cat_metrics


def train_baseline_model(config, dataset, num_steps, device='cpu'):
    """Train baseline transformer on expanded corpus."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = BaselineTransformer(
        vocab_size=config.vocab_size, d_model=128,
        n_heads=8, n_layers=2, d_ffn=512, max_seq_len=128
    ).to(device)
    print(f"\nBaseline Transformer: {model.count_parameters():,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=config.weight_decay)
    model.train()

    step = 0
    losses = []
    per_cat_metrics = defaultdict(lambda: {'loss': []})
    start = time.time()

    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING: {num_steps} steps, lr={LEARNING_RATE}")
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

            for i in range(len(cat_idx)):
                c = cat_idx[i].item()
                per_cat_metrics[c]['loss'].append(loss.item())

            losses.append(loss.item())
            step += 1

            if step % 500 == 0:
                avg = sum(losses[-500:]) / min(500, len(losses))
                ppl = math.exp(min(avg, 20))
                speed = step / (time.time() - start)
                print(f"  Baseline step {step}/{num_steps}: loss={avg:.3f}, ppl={ppl:.1f}, "
                      f"{speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    print(f"Baseline done: loss={avg:.3f}, ppl={ppl:.1f}")

    return model, avg, ppl, per_cat_metrics


def analyze_results(res_ppl, base_ppl, res_cats, base_cats):
    """Produce B1 and B2 analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("V2.1 ANALYSIS REPORT")
    lines.append("=" * 70)

    # B1: Scale test
    lines.append("\n" + "=" * 70)
    lines.append("B1: SCALE TEST — Does 8x advantage hold on diverse data?")
    lines.append("=" * 70)
    ratio = base_ppl / res_ppl if res_ppl > 0 else 0
    lines.append(f"  Resonance PPL: {res_ppl:.2f}")
    lines.append(f"  Baseline PPL:  {base_ppl:.2f}")
    lines.append(f"  Ratio: {ratio:.1f}x advantage for Resonance")
    if ratio >= 5:
        lines.append(f"  VERDICT: Strong advantage maintained ({ratio:.1f}x)")
    elif ratio >= 2:
        lines.append(f"  VERDICT: Moderate advantage ({ratio:.1f}x)")
    else:
        lines.append(f"  VERDICT: Advantage narrowed ({ratio:.1f}x)")

    # V2 comparison
    lines.append(f"\n  V2 original ratio: 8.1x (9.4 vs 76.6 on synthetic+quran)")
    lines.append(f"  V2.1 expanded ratio: {ratio:.1f}x (on diverse 5-category corpus)")

    # B2: Per-category analysis
    lines.append("\n" + "=" * 70)
    lines.append("B2: PER-CATEGORY GATE/ITER ANALYSIS — Is gating adaptive?")
    lines.append("=" * 70)
    lines.append(f"{'Category':<15} {'Res Loss':>10} {'Base Loss':>10} "
                 f"{'Res PPL':>10} {'Base PPL':>10} {'Gates':>8} {'Iters':>8}")
    lines.append(f"{'-'*70}")

    for c in sorted(set(list(res_cats.keys()) + list(base_cats.keys()))):
        cat_name = CATEGORY_NAMES.get(c, f'cat_{c}')

        if c in res_cats and res_cats[c]['loss']:
            r_loss = sum(res_cats[c]['loss'][-500:]) / min(500, len(res_cats[c]['loss']))
            r_ppl = math.exp(min(r_loss, 20))
            r_gates = sum(res_cats[c]['gates'][-500:]) / min(500, len(res_cats[c]['gates']))
            r_iters = sum(res_cats[c]['iters'][-500:]) / min(500, len(res_cats[c]['iters']))
        else:
            r_loss = r_ppl = r_gates = r_iters = 0

        if c in base_cats and base_cats[c]['loss']:
            b_loss = sum(base_cats[c]['loss'][-500:]) / min(500, len(base_cats[c]['loss']))
            b_ppl = math.exp(min(b_loss, 20))
        else:
            b_loss = b_ppl = 0

        lines.append(f"  {cat_name:<13} {r_loss:>10.3f} {b_loss:>10.3f} "
                      f"{r_ppl:>10.2f} {b_ppl:>10.2f} {r_gates:>8.3f} {r_iters:>8.1f}")

    # Adaptive behavior analysis
    if res_cats:
        gate_values = {}
        iter_values = {}
        for c in res_cats:
            if res_cats[c]['gates']:
                gate_values[CATEGORY_NAMES.get(c, f'cat_{c}')] = (
                    sum(res_cats[c]['gates'][-500:]) / min(500, len(res_cats[c]['gates']))
                )
                iter_values[CATEGORY_NAMES.get(c, f'cat_{c}')] = (
                    sum(res_cats[c]['iters'][-500:]) / min(500, len(res_cats[c]['iters']))
                )

        if gate_values:
            max_gate_cat = max(gate_values, key=gate_values.get)
            min_gate_cat = min(gate_values, key=gate_values.get)
            gate_range = max(gate_values.values()) - min(gate_values.values())

            lines.append(f"\n  Gate range across categories: {gate_range:.4f}")
            lines.append(f"  Highest gates: {max_gate_cat} ({gate_values[max_gate_cat]:.3f})")
            lines.append(f"  Lowest gates: {min_gate_cat} ({gate_values[min_gate_cat]:.3f})")

            if gate_range > 0.05:
                lines.append(f"  VERDICT: Gates show adaptive behavior (range={gate_range:.3f})")
            else:
                lines.append(f"  VERDICT: Gates relatively uniform (range={gate_range:.3f})")

    report = '\n'.join(lines)
    return report


def main():
    parser = argparse.ArgumentParser(description='V2.1 Training')
    parser.add_argument('--mode', type=str, choices=['resonance', 'baseline', 'both'],
                        default='both')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS)
    parser.add_argument('--mitosis', action='store_true', default=True,
                        help='Enable mitosis (default: True)')
    parser.add_argument('--no-mitosis', action='store_true', default=False,
                        help='Disable mitosis for B3 comparison')
    args = parser.parse_args()

    mitosis_enabled = not args.no_mitosis

    # Config
    config = ResonanceConfig()
    for key, val in LOCKED_CONFIG.items():
        setattr(config, key, val)

    # Load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: vocab={config.vocab_size}")

    # Load dataset
    print(f"Loading dataset from {CORPUS_PATH}...")
    dataset = CategoryTextDataset(CORPUS_PATH, tokenizer, seq_len=config.max_seq_len)
    print(f"Dataset: {len(dataset):,} samples")

    os.makedirs('checkpoints_v2.1', exist_ok=True)
    os.makedirs('checkpoints_baseline', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    res_ppl = base_ppl = 0
    res_cats = base_cats = {}

    # Train baseline
    if args.mode in ('baseline', 'both'):
        base_model, base_loss, base_ppl, base_cats = train_baseline_model(
            config, dataset, args.steps, args.device
        )
        torch.save(base_model.state_dict(), 'checkpoints_baseline/expanded_corpus_baseline.pt')
        print(f"Baseline saved to checkpoints_baseline/expanded_corpus_baseline.pt")

    # Train resonance
    if args.mode in ('resonance', 'both'):
        res_model, res_loss, res_ppl, res_cats = train_resonance(
            config, dataset, args.steps, args.device, mitosis_enabled=mitosis_enabled
        )
        ckpt_name = 'with_mitosis_final.pt' if mitosis_enabled else 'without_mitosis_final.pt'
        torch.save(res_model.state_dict(), f'checkpoints_v2.1/{ckpt_name}')
        print(f"Resonance saved to checkpoints_v2.1/{ckpt_name}")

    # Analysis
    if args.mode == 'both' and res_ppl > 0 and base_ppl > 0:
        report = analyze_results(res_ppl, base_ppl, res_cats, base_cats)
        print(report)
        with open('logs/v2.1_analysis_report.txt', 'w') as f:
            f.write(report)
        print(f"\nReport saved to logs/v2.1_analysis_report.txt")


if __name__ == '__main__':
    main()
