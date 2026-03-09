#!/usr/bin/env python3
"""Interactive text generation with Neurogenesis."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.inference.generate import generate
from neurogenesis.tokenizer.bpe import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description='Generate text with Neurogenesis')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase3_model.pt')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt (non-interactive mode)')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-iterations', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = NeurogenesisConfig()
    model = NeurogenesisModel(config)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        print(f"WARNING: No checkpoint found at {args.checkpoint}")

    tokenizer = load_tokenizer()
    model.eval()

    if args.prompt:
        # Single prompt mode
        text = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_iterations=args.max_iterations,
            device=args.device,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {text}")
    else:
        # Interactive mode
        print("Neurogenesis Text Generator (type 'quit' to exit)")
        print("-" * 50)
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue

            text = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_iterations=args.max_iterations,
                device=args.device,
            )
            print(f"Generated: {text}")


if __name__ == '__main__':
    main()
