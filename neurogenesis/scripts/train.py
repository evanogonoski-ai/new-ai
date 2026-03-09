#!/usr/bin/env python3
"""Main training script for Neurogenesis."""
import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.training.phase1_bootstrap import train_phase1
from neurogenesis.training.phase2_router import train_phase2
from neurogenesis.training.phase3_joint import train_phase3
from neurogenesis.training.phase4_consolidation import train_phase4


def main():
    parser = argparse.ArgumentParser(description='Train Neurogenesis model')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                        help='Training phase to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all phases sequentially')
    parser.add_argument('--abbreviated', action='store_true', default=True,
                        help='Run abbreviated training (10%% of steps)')
    parser.add_argument('--full', action='store_true',
                        help='Run full training')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to train on')
    args = parser.parse_args()

    config = NeurogenesisConfig()

    # Abbreviated mode: 10% of steps
    if args.full:
        args.abbreviated = False

    if args.abbreviated:
        config.phase1_steps = 1000
        config.phase2_steps = 5000
        config.phase3_steps = 20000

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    tokenizer_path = os.path.join('data', 'tokenizer.json')

    if args.all or args.phase == 1:
        print("=" * 60)
        print("PHASE 1: Primitive Bootstrapping")
        print("=" * 60)
        library = train_phase1(
            config,
            num_steps=config.phase1_steps,
            device=args.device,
            checkpoint_path=os.path.join(args.checkpoint_dir, 'phase1_library.pt'),
        )

    if args.all or args.phase in [2, 3, 4]:
        # Build full model
        model = NeurogenesisModel(config)
        print(f"\nModel parameters: {model.count_parameters():,}")

        # Load Phase 1 checkpoint if available
        phase1_path = os.path.join(args.checkpoint_dir, 'phase1_library.pt')
        if os.path.exists(phase1_path):
            print(f"Loading Phase 1 primitives from {phase1_path}")
            state = torch.load(phase1_path, weights_only=True)
            model.primitive_library.load_state_dict(state)

    if args.all or args.phase == 2:
        print("\n" + "=" * 60)
        print("PHASE 2: Router Training")
        print("=" * 60)
        model = train_phase2(
            model, config,
            num_steps=config.phase2_steps,
            seq_len=32,
            device=args.device,
            checkpoint_path=os.path.join(args.checkpoint_dir, 'phase2_model.pt'),
            tokenizer_path=tokenizer_path,
        )

    if args.all or args.phase == 3:
        # Load Phase 2 checkpoint if running Phase 3 standalone
        if args.phase == 3:
            phase2_path = os.path.join(args.checkpoint_dir, 'phase2_model.pt')
            if os.path.exists(phase2_path):
                print(f"Loading Phase 2 model from {phase2_path}")
                model.load_state_dict(torch.load(phase2_path, weights_only=True))

        print("\n" + "=" * 60)
        print("PHASE 3: Joint Training")
        print("=" * 60)
        model = train_phase3(
            model, config,
            num_steps=config.phase3_steps,
            seq_len=64,
            device=args.device,
            checkpoint_path=os.path.join(args.checkpoint_dir, 'phase3_model.pt'),
            tokenizer_path=tokenizer_path,
        )

    if args.all or args.phase == 4:
        # Load Phase 3 checkpoint if running Phase 4 standalone
        if args.phase == 4:
            phase3_path = os.path.join(args.checkpoint_dir, 'phase3_model.pt')
            if os.path.exists(phase3_path):
                print(f"Loading Phase 3 model from {phase3_path}")
                model.load_state_dict(torch.load(phase3_path, weights_only=True))

        print("\n" + "=" * 60)
        print("PHASE 4: Consolidation")
        print("=" * 60)
        model = train_phase4(
            model, config,
            device=args.device,
            checkpoint_path=os.path.join(args.checkpoint_dir, 'phase4_model.pt'),
            tokenizer_path=tokenizer_path,
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
