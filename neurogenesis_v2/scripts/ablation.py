#!/usr/bin/env python3
"""Iteration depth ablation study for Resonance V2.

Evaluates a trained model at different iteration depths to find:
- Optimal iteration count
- Diminishing returns curve
- Whether convergence threshold needs tuning
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.resonance_model import ResonanceModel
from neurogenesis_v2.training.data import (
    TextDataset, prepare_combined_corpus,
    train_tokenizer, load_tokenizer
)


def evaluate_at_depth(model, loader, num_iterations, num_eval_batches=200, device='cpu'):
    """Evaluate model at a fixed iteration depth."""
    model.eval()
    total_loss = 0
    total_gates = 0
    total_deltas = []
    count = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            if count >= num_eval_batches:
                break
            input_ids = input_ids.to(device).clamp(0, model.config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, model.config.vocab_size - 1)

            output = model(
                input_ids,
                max_iterations=num_iterations,
                use_momentum=True,
                force_iterations=True,
                return_all_states=True,
            )

            loss = F.cross_entropy(output['logits'], target_ids[:, -1])
            total_loss += loss.item()

            if output['all_gates']:
                gate_mean = torch.stack(output['all_gates']).mean().item()
                total_gates += gate_mean

            if output['convergence_deltas']:
                total_deltas.append(output['convergence_deltas'][-1])

            count += 1

    avg_loss = total_loss / count
    avg_ppl = math.exp(min(avg_loss, 20))
    avg_gates = total_gates / count
    avg_delta = sum(total_deltas) / len(total_deltas) if total_deltas else 0

    return {
        'iterations': num_iterations,
        'loss': avg_loss,
        'ppl': avg_ppl,
        'gate_mean': avg_gates,
        'final_delta': avg_delta,
    }


def main():
    parser = argparse.ArgumentParser(description='Iteration Depth Ablation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v2/phase2.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--eval-batches', type=int, default=200)
    args = parser.parse_args()

    # Load config and model
    config = ResonanceConfig()

    # Load tokenizer
    tok_path = 'data/tokenizer_v2.json'
    if not os.path.exists(tok_path):
        tok_path = 'data/tokenizer.json'
    tokenizer = load_tokenizer(tok_path)
    config.vocab_size = tokenizer.get_vocab_size()

    # Build model
    model = ResonanceModel(config)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, weights_only=True, map_location=args.device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found at {args.checkpoint}, using random weights")

    model = model.to(args.device)

    # Prepare eval data
    corpus = prepare_combined_corpus(num_synthetic=10000, include_quran=True)
    all_ids = []
    for text in corpus:
        enc = tokenizer.encode(text)
        all_ids.extend(enc.ids)
    dataset = TextDataset(all_ids, seq_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Ablation depths
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]

    print(f"\n{'='*70}")
    print(f"ITERATION DEPTH ABLATION")
    print(f"{'='*70}")
    print(f"Model: {model.count_active_parameters():,} active params")
    print(f"Eval batches: {args.eval_batches}")
    print(f"{'='*70}\n")

    results = []
    for depth in depths:
        r = evaluate_at_depth(model, loader, depth, args.eval_batches, args.device)
        results.append(r)
        print(f"  iters={depth:2d}: loss={r['loss']:.4f}  ppl={r['ppl']:.2f}  "
              f"gates={r['gate_mean']:.3f}  delta={r['final_delta']:.6f}")

    # Analysis
    print(f"\n{'='*70}")
    print(f"MARGINAL IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Depth':>6} {'PPL':>8} {'Δ PPL':>8} {'% Improve':>10} {'Verdict':>12}")
    print(f"{'-'*50}")

    best_depth = 1
    best_ppl = results[0]['ppl']

    for i, r in enumerate(results):
        if i == 0:
            print(f"{r['iterations']:6d} {r['ppl']:8.2f} {'—':>8} {'—':>10} {'baseline':>12}")
        else:
            prev = results[i-1]
            delta_ppl = r['ppl'] - prev['ppl']
            pct = (delta_ppl / prev['ppl']) * 100

            if delta_ppl < -0.1:
                verdict = "HELPFUL"
            elif delta_ppl > 0.5:
                verdict = "HARMFUL"
            else:
                verdict = "MARGINAL"

            print(f"{r['iterations']:6d} {r['ppl']:8.2f} {delta_ppl:8.2f} {pct:9.1f}% {verdict:>12}")

        if r['ppl'] < best_ppl:
            best_ppl = r['ppl']
            best_depth = r['iterations']

    # Recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"  Best iteration depth: {best_depth} (ppl={best_ppl:.2f})")
    print(f"  Single-pass (depth=1) ppl: {results[0]['ppl']:.2f}")
    print(f"  Recurrence benefit: {results[0]['ppl'] - best_ppl:.2f} ppl reduction")

    # Check if PPL keeps dropping at high depths
    ppl_at_8 = next(r['ppl'] for r in results if r['iterations'] == 8)
    ppl_at_16 = next(r['ppl'] for r in results if r['iterations'] == 16)

    if ppl_at_16 < ppl_at_8 - 0.5:
        print(f"\n  ACTION: PPL still improving at 16 iters ({ppl_at_8:.2f}→{ppl_at_16:.2f})")
        print(f"  → Consider increasing max_iterations for Phase 3")
    elif ppl_at_16 > ppl_at_8 + 0.5:
        print(f"\n  ACTION: PPL degrading beyond 8 iters ({ppl_at_8:.2f}→{ppl_at_16:.2f})")
        print(f"  → Consider reducing momentum_beta_init or adding per-iter grad clipping")
    else:
        print(f"\n  ACTION: PPL plateaus around 8 iters ({ppl_at_8:.2f}→{ppl_at_16:.2f})")
        print(f"  → Consider loosening convergence_threshold to 0.05-0.1")

    # Check convergence deltas
    delta_at_best = next(r['final_delta'] for r in results if r['iterations'] == best_depth)
    print(f"\n  Convergence delta at best depth: {delta_at_best:.6f}")
    if delta_at_best < config.convergence_threshold:
        print(f"  → Current threshold ({config.convergence_threshold}) would catch this")
    else:
        print(f"  → Current threshold ({config.convergence_threshold}) too tight, suggest {delta_at_best*2:.4f}")

    return results


if __name__ == '__main__':
    main()
