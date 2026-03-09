#!/usr/bin/env python3
"""Generation comparison across Phase 1 and Phase 2 checkpoints."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.resonance_model import ResonanceModel
from neurogenesis_v2.training.data import load_tokenizer
from neurogenesis_v2.inference.generate import generate


PROMPTS = [
    "Once upon a time",
    "The little cat",
    "A happy girl",
    "In the name of God",
    "The brave warrior discovered",
]

CHECKPOINTS = [
    ('Phase 1 (Single-Pass)', 'checkpoints_v2/phase1_final.pt', 1),
    ('Phase 2 (Recurrence)', 'checkpoints_v2/phase2_final.pt', 8),
]


def main():
    parser = argparse.ArgumentParser(description='Generation Comparison')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()

    config = ResonanceConfig()
    tokenizer = load_tokenizer('data/tokenizer_v2.json')
    config.vocab_size = tokenizer.get_vocab_size()

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("GENERATION COMPARISON — Phase 1 vs Phase 2")
    output_lines.append(f"Settings: max_tokens={args.max_tokens}, temp={args.temperature}, top_k={args.top_k}")
    output_lines.append("=" * 70)

    for phase_name, ckpt_path, max_iters in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            output_lines.append(f"\n--- {phase_name}: checkpoint not found ({ckpt_path}) ---")
            continue

        model = ResonanceModel(config)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=args.device))

        # Apply Phase 3 tuning for Phase 2
        if max_iters > 1:
            config.convergence_threshold = 0.18
            with torch.no_grad():
                model.block.residual1.beta.fill_(-2.94)
                model.block.residual2.beta.fill_(-2.94)

        model = model.to(args.device)

        output_lines.append(f"\n{'='*70}")
        output_lines.append(f"  {phase_name} (max_iterations={max_iters})")
        output_lines.append(f"{'='*70}")

        for prompt in PROMPTS:
            text = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_iterations=max_iters,
                device=args.device,
            )
            output_lines.append(f"\n  Prompt: \"{prompt}\"")
            output_lines.append(f"  Output: {prompt}{text}")
            output_lines.append(f"  {'—'*50}")

    report = '\n'.join(output_lines)
    print(report)

    os.makedirs('logs', exist_ok=True)
    with open('logs/generation_comparison.txt', 'w') as f:
        f.write(report)
    print(f"\nSaved to logs/generation_comparison.txt")


if __name__ == '__main__':
    main()
