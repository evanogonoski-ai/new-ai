#!/usr/bin/env python3
"""Run consolidation manually."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.training.phase4_consolidation import train_phase4
from neurogenesis.tokenizer.bpe import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description='Run consolidation on Neurogenesis model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase3_model.pt')
    parser.add_argument('--output', type=str, default='checkpoints/consolidated_model.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = NeurogenesisConfig()
    model = NeurogenesisModel(config)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return

    model = train_phase4(
        model, config,
        device=args.device,
        checkpoint_path=args.output,
    )

    print(f"Consolidated model saved to {args.output}")


if __name__ == '__main__':
    main()
