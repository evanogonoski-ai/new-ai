#!/usr/bin/env python3
"""Evaluate Neurogenesis model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch.utils.data import DataLoader

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.training.data import load_tinystories_validation
from neurogenesis.tokenizer.bpe import load_tokenizer
from neurogenesis.evaluation.metrics import (
    compute_perplexity, primitive_utilization_entropy,
    adaptive_halting_variance, primitive_ablation,
)
from neurogenesis.evaluation.benchmarks import run_benchmark


def main():
    parser = argparse.ArgumentParser(description='Evaluate Neurogenesis model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase3_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run comparison benchmark against baseline transformer')
    parser.add_argument('--ablation', action='store_true',
                        help='Run primitive ablation analysis')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = NeurogenesisConfig()
    model = NeurogenesisModel(config)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint}, using random weights")

    tokenizer = load_tokenizer()
    val_dataset = load_tinystories_validation(tokenizer, seq_len=64, max_samples=5000)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Perplexity
    ppl = compute_perplexity(model, val_loader, device=args.device, max_batches=100)
    print(f"Perplexity: {ppl:.1f}")

    # Primitive utilization
    usage_ent = primitive_utilization_entropy(model)
    print(f"Primitive utilization entropy: {usage_ent:.2f}")

    # Halting variance
    halt_stats = adaptive_halting_variance(model, val_loader, device=args.device)
    print(f"Halting iterations - mean: {halt_stats['mean']:.1f}, std: {halt_stats['std']:.2f}, "
          f"range: [{halt_stats['min']:.1f}, {halt_stats['max']:.1f}]")

    # Ablation
    if args.ablation:
        print("\nPrimitive Ablation Analysis:")
        results = primitive_ablation(model, val_loader, device=args.device)
        for idx, info in sorted(results.items(), key=lambda x: x[1]['loss_increase'], reverse=True):
            impact = info['loss_increase']
            marker = "***" if abs(impact) > 0.1 else ""
            print(f"  Primitive {idx}: loss change = {impact:+.4f} {marker}")

    # Benchmark
    if args.benchmark:
        print()
        train_loader = DataLoader(
            load_tinystories(tokenizer, seq_len=64, max_samples=20000),
            batch_size=16, shuffle=True
        )
        run_benchmark(model, train_loader, vocab_size=config.vocab_size,
                      num_train_steps=5000, device=args.device)


if __name__ == '__main__':
    main()
