#!/usr/bin/env python3
"""Main training script for Neurogenesis V2 Resonance Architecture."""
import argparse
import os
import sys
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.resonance_model import ResonanceModel
from neurogenesis_v2.model.mitosis import MitosisEngine
from neurogenesis_v2.training.losses import gate_sparsity_loss, ponder_cost_loss, multitimescale_loss
from neurogenesis_v2.training.data import (
    TextDataset, generate_enhanced_stories, prepare_combined_corpus,
    train_tokenizer, load_tokenizer
)
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis_v2.inference.generate import generate


def ensure_tokenizer(path='data/tokenizer_v2.json', include_quran=True):
    if os.path.exists(path):
        return load_tokenizer(path)
    print("Training tokenizer on combined corpus...")
    corpus = prepare_combined_corpus(num_synthetic=50000, include_quran=include_quran)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok = train_tokenizer(corpus, vocab_size=8192, save_path=path)
    print(f"Tokenizer vocab: {tok.get_vocab_size()}")
    return tok


def prepare_data(tokenizer, seq_len=128, num_synthetic=50000, include_quran=True):
    corpus = prepare_combined_corpus(num_synthetic=num_synthetic, include_quran=include_quran)
    all_ids = []
    for text in corpus:
        enc = tokenizer.encode(text)
        all_ids.extend(enc.ids)
    print(f"Data: {len(all_ids):,} tokens from {len(corpus):,} texts")
    return TextDataset(all_ids, seq_len=seq_len)


def train_phase(model, config, loader, num_steps, lr, phase_name,
                use_momentum=True, max_iterations=1, mitosis_engine=None,
                device='cpu'):
    """Generic training loop for any phase."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    step = 0
    losses = []
    start = time.time()
    vocab = config.vocab_size

    print(f"\n{'='*60}")
    print(f"{phase_name}: {num_steps} steps, lr={lr}, max_iter={max_iterations}")
    print(f"{'='*60}")

    for input_ids, target_ids in loader:
        if step >= num_steps:
            break
        input_ids = input_ids.to(device).clamp(0, vocab - 1)
        target_ids = target_ids.to(device).clamp(0, vocab - 1)

        output = model(
            input_ids,
            max_iterations=max_iterations,
            use_momentum=use_momentum,
            return_all_states=True,
        )

        # Main loss
        ce_loss = F.cross_entropy(output['logits'], target_ids[:, -1])

        # Auxiliary losses
        total_loss = ce_loss

        if output['all_gates']:
            all_gates = torch.stack(output['all_gates'])  # (iters, B, H)
            gs_loss = gate_sparsity_loss(all_gates, config.gate_sparsity_lambda)
            total_loss = total_loss + gs_loss

        if max_iterations > 1:
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
        if mitosis_engine is not None:
            mitosis_engine.record_gradients()
            mresult = mitosis_engine.check_and_apply(step)
            if mresult['mitosis'] > 0 or mresult['pruned'] > 0:
                print(f"  [Mitosis] step {step}: +{mresult['mitosis']} split, "
                      f"-{mresult['pruned']} pruned, alive={mresult['alive']}")

        losses.append(ce_loss.item())
        step += 1

        if step % 500 == 0:
            avg = sum(losses[-500:]) / min(500, len(losses))
            ppl = math.exp(min(avg, 20))
            speed = step / (time.time() - start)
            iters = output['num_iterations']
            gates_mean = output['all_gates'][-1].mean().item() if output['all_gates'] else 0
            alive = model.block.ffn.num_alive
            print(f"  Step {step}/{num_steps}: loss={avg:.3f}, ppl={ppl:.1f}, "
                  f"iters={iters}, gates={gates_mean:.2f}, alive={alive}, "
                  f"{speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    print(f"{phase_name} done: loss={avg:.3f}, ppl={ppl:.1f}")
    return avg, ppl


def train_baseline(config, loader, num_steps, lr, device='cpu'):
    """Train baseline transformer with matched params."""
    model = BaselineTransformer(
        vocab_size=config.vocab_size, d_model=128,
        n_heads=8, n_layers=2, d_ffn=512, max_seq_len=128
    ).to(device)
    print(f"\nBaseline Transformer: {model.count_parameters():,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    model.train()

    step = 0
    losses = []
    start = time.time()

    for input_ids, target_ids in loader:
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
            print(f"  Baseline step {step}: loss={avg:.3f}, ppl={ppl:.1f}, {speed:.1f} it/s")

    avg = sum(losses[-500:]) / min(500, len(losses))
    ppl = math.exp(min(avg, 20))
    print(f"Baseline done: loss={avg:.3f}, ppl={ppl:.1f}")
    return model, avg, ppl


def main():
    parser = argparse.ArgumentParser(description='Train Resonance V2')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--abbreviated', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_v2')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    config = ResonanceConfig()

    # Tokenizer
    tokenizer = ensure_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab: {config.vocab_size}")

    # Data
    dataset = prepare_data(tokenizer, seq_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Abbreviated mode
    if args.full:
        args.abbreviated = False
    if args.abbreviated:
        config.phase1_steps = 5000
        config.phase2_steps = 10000
        config.phase3_steps = 15000

    # Build model
    model = ResonanceModel(config)
    total_p = model.count_parameters()
    active_p = model.count_active_parameters()
    print(f"Resonance model: {total_p:,} total params, {active_p:,} active params")
    print(f"FFN alive neurons: {model.block.ffn.num_alive}/{config.d_ffn_max}")

    results = {}

    # Phase 1: Single-pass
    if args.all or args.phase == 1:
        loss1, ppl1 = train_phase(
            model, config, loader,
            num_steps=config.phase1_steps,
            lr=config.phase1_lr,
            phase_name="PHASE 1 (Single-Pass)",
            use_momentum=False,
            max_iterations=1,
            device=args.device,
        )
        results['phase1'] = {'loss': loss1, 'ppl': ppl1}
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'phase1.pt'))

    # Baseline comparison
    if args.all or args.baseline:
        baseline_model, base_loss, base_ppl = train_baseline(
            config, loader,
            num_steps=config.phase1_steps,
            lr=config.phase1_lr,
            device=args.device,
        )
        results['baseline'] = {'loss': base_loss, 'ppl': base_ppl}
        torch.save(baseline_model.state_dict(), os.path.join(args.checkpoint_dir, 'baseline.pt'))

        if 'phase1' in results:
            print(f"\n{'='*60}")
            print("Phase 1 vs Baseline:")
            print(f"  Resonance: ppl={results['phase1']['ppl']:.1f}")
            print(f"  Baseline:  ppl={results['baseline']['ppl']:.1f}")
            ratio = results['phase1']['ppl'] / results['baseline']['ppl']
            if ratio <= 2.0:
                print(f"  Ratio: {ratio:.2f}x — PASS (within 2x)")
            else:
                print(f"  Ratio: {ratio:.2f}x — NEEDS INVESTIGATION")
            print(f"{'='*60}")

    # Phase 2: Unlock recurrence
    if args.all or args.phase == 2:
        if args.phase == 2:
            p1_path = os.path.join(args.checkpoint_dir, 'phase1.pt')
            if os.path.exists(p1_path):
                model.load_state_dict(torch.load(p1_path, weights_only=True))

        loss2, ppl2 = train_phase(
            model, config, loader,
            num_steps=config.phase2_steps,
            lr=config.phase2_lr,
            phase_name="PHASE 2 (Recurrence)",
            use_momentum=True,
            max_iterations=8,
            device=args.device,
        )
        results['phase2'] = {'loss': loss2, 'ppl': ppl2}
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'phase2.pt'))

    # Phase 3: Mitosis
    if args.all or args.phase == 3:
        if args.phase == 3:
            p2_path = os.path.join(args.checkpoint_dir, 'phase2.pt')
            if os.path.exists(p2_path):
                model.load_state_dict(torch.load(p2_path, weights_only=True))

        config.mitosis_enabled = True
        # Ablation-informed tuning: optimal depth is 6-7, PPL degrades beyond 8
        # Loosen convergence threshold so model can halt early at natural convergence
        config.convergence_threshold = 0.18
        # Reduce momentum to prevent instability at higher iterations
        config.momentum_beta_init = 0.05
        with torch.no_grad():
            model.block.residual1.beta.fill_(-2.94)  # sigmoid(-2.94) ≈ 0.05
            model.block.residual2.beta.fill_(-2.94)

        # Diagnostic findings: at lr=1e-4 with grad_clip=1.0, gradient variance is:
        #   mean=0.00003, max=0.001, P90=0.00005, P95=0.00008
        # Previous threshold of 0.008 was 100x too high. Setting to P90.
        config.mitosis_gradient_var_threshold = 0.00005
        config.mitosis_check_interval = 500  # check more frequently
        engine = MitosisEngine(model.block.ffn, config)

        print(f"  [Ablation tuning] convergence_threshold=0.18, momentum_beta=0.05")
        print(f"  [Ablation tuning] max_iterations=8 (was 16, PPL degrades beyond 8)")
        print(f"  [Diagnostic] mitosis_gradient_var_threshold=0.00005 (P90 of measured grad var)")
        print(f"  [Diagnostic] check_interval=500 (was 1000)")

        loss3, ppl3 = train_phase(
            model, config, loader,
            num_steps=config.phase3_steps,
            lr=config.phase3_lr,
            phase_name="PHASE 3 (Mitosis)",
            use_momentum=True,
            max_iterations=8,
            mitosis_engine=engine,
            device=args.device,
        )
        results['phase3'] = {'loss': loss3, 'ppl': ppl3}
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'phase3.pt'))

    # Generation test
    print(f"\n{'='*60}")
    print("GENERATION TEST")
    print(f"{'='*60}")
    model.eval()
    for prompt in ["Once upon a time", "The little cat", "A happy girl"]:
        text = generate(model, tokenizer, prompt, max_new_tokens=50,
                       temperature=0.8, top_k=20, device=args.device)
        print(f"  [{prompt}] → {text[:150]}")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for phase, r in results.items():
        print(f"  {phase}: loss={r['loss']:.3f}, ppl={r['ppl']:.1f}")
    print(f"  Active FFN neurons: {model.block.ffn.num_alive}/{config.d_ffn_max}")

    # Save final
    torch.save({
        'model_state': model.state_dict(),
        'config': vars(config),
        'results': results,
    }, os.path.join(args.checkpoint_dir, 'final.pt'))
    print(f"\nSaved to {args.checkpoint_dir}/final.pt")


if __name__ == '__main__':
    main()
