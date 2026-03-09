#!/usr/bin/env python3
"""Unit tests for Neurogenesis model components."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.primitives import Primitive, PrimitiveLibrary
from neurogenesis.model.working_memory import WorkingMemory
from neurogenesis.model.router import Router
from neurogenesis.model.halting import AdaptiveHalting
from neurogenesis.model.neurogenesis import NeurogenesisModel


def test_primitive():
    print("Testing Primitive...")
    prim = Primitive(d_model=128, d_inner=256)
    x = torch.randn(4, 64, 128)  # batch=4, slots=64, d_model=128
    out, gate = prim(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert gate.shape == (4, 64, 1), f"Expected (4, 64, 1), got {gate.shape}"
    assert (gate >= 0).all() and (gate <= 1).all(), "Gate should be in [0, 1]"
    print("  OK: output shape correct, gate in valid range")

    # Test gradient flow
    loss = out.sum() + gate.sum()
    loss.backward()
    for name, param in prim.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("  OK: gradients flow through primitive")


def test_primitive_library():
    print("Testing PrimitiveLibrary...")
    lib = PrimitiveLibrary(num_primitives=64, num_structured=16, d_model=128, d_inner=256)
    assert len(lib.primitives) == 64
    assert lib.usage_frequency.shape == (64,)

    x = torch.randn(2, 128)
    out, gate = lib(0, x)
    assert out.shape == x.shape
    print("  OK: library forward works")

    # Test usage tracking
    indices = torch.tensor([[0, 1, 2, 3]])
    lib.update_usage(indices)
    assert lib.usage_frequency[0] > 0, "Usage should be tracked"
    print("  OK: usage tracking works")


def test_working_memory():
    print("Testing WorkingMemory...")
    wm = WorkingMemory(num_slots=64, d_model=128)

    # Short sequence
    emb = torch.randn(2, 10, 128)
    mem = wm.initialize(emb)
    assert mem.shape == (2, 64, 128)
    print("  OK: initialization (short seq)")

    # Long sequence
    emb_long = torch.randn(2, 100, 128)
    mem_long = wm.initialize(emb_long)
    assert mem_long.shape == (2, 64, 128)
    print("  OK: initialization (long seq, compression)")

    # Gated write
    new_vals = torch.randn(2, 64, 128)
    gates = torch.rand(2, 64, 1)
    slot_mask = torch.ones(2, 64)
    updated = wm.apply_gated_write(mem, slot_mask, new_vals, gates)
    assert updated.shape == mem.shape
    print("  OK: gated write")

    # Read compressed
    compressed = wm.read_compressed(mem)
    assert compressed.shape == (2, 128)
    print("  OK: read compressed")

    # Generation update
    new_token = torch.randn(2, 128)
    updated = wm.update_for_generation(mem, new_token)
    assert updated.shape == mem.shape
    print("  OK: generation update")


def test_router():
    print("Testing Router...")
    router = Router(d_model=128, num_primitives=64, num_selected=4, num_slots=64)

    state = torch.randn(2, 128)

    # Training mode (Gumbel-Softmax)
    indices, weights, masks = router(state, training=True)
    assert indices.shape == (2, 4), f"Expected (2, 4), got {indices.shape}"
    assert weights.shape == (2, 4)
    assert masks.shape == (2, 4, 64)
    assert (masks >= 0).all() and (masks <= 1).all()
    print("  OK: training mode selection")

    # Inference mode
    indices_inf, weights_inf, masks_inf = router(state, training=False)
    assert indices_inf.shape == (2, 4)
    print("  OK: inference mode selection")

    # Entropy
    ent = router.get_entropy(state)
    assert ent.shape == ()
    assert ent.item() > 0
    print(f"  OK: entropy = {ent.item():.2f}")

    # Gradient flow
    loss = weights.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in router.parameters())
    assert has_grad, "No gradients in router"
    print("  OK: gradients flow through router")


def test_halting():
    print("Testing AdaptiveHalting...")
    halt = AdaptiveHalting(d_model=128, max_iterations=16, min_iterations=2)
    state = torch.randn(2, 128)
    cum_prob = torch.zeros(2, 1)

    # Before min_iterations: should not halt
    prob, cum, should = halt(state, iteration=0, cumulative_prob=cum_prob)
    assert not should.any(), "Should not halt before min_iterations"
    print("  OK: no halt before minimum")

    # Run to completion
    halted = False
    for t in range(16):
        prob, cum_prob, should = halt(state, t, cum_prob)
        if should.all():
            halted = True
            break
    assert halted, "Should have halted by max_iterations"
    print(f"  OK: halted at iteration {t}")


def test_full_model():
    print("Testing NeurogenesisModel...")
    config = NeurogenesisConfig()
    model = NeurogenesisModel(config)
    num_params = model.count_parameters()
    print(f"  Total parameters: {num_params:,}")

    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    output = model(input_ids, training=True, max_iterations=4)

    assert 'logits' in output
    assert output['logits'].shape == (2, config.vocab_size)
    print(f"  OK: forward pass, logits shape = {output['logits'].shape}")

    assert 'ponder_cost' in output
    print(f"  OK: ponder cost = {output['ponder_cost'].item():.1f}")

    assert 'selected_primitives' in output
    assert len(output['selected_primitives']) > 0
    print(f"  OK: {len(output['selected_primitives'])} iterations of primitive selection")

    # Backward pass
    loss = torch.nn.functional.cross_entropy(
        output['logits'], torch.randint(0, config.vocab_size, (2,))
    )
    loss.backward()
    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  OK: backward pass, {grad_params}/{total_params} params have gradients")

    # Inference mode
    model.eval()
    with torch.no_grad():
        output_inf = model(input_ids, training=False, max_iterations=4)
    assert output_inf['logits'].shape == (2, config.vocab_size)
    print("  OK: inference mode works")


def test_all():
    print("=" * 50)
    print("Neurogenesis Component Tests")
    print("=" * 50)
    print()

    test_primitive()
    print()
    test_primitive_library()
    print()
    test_working_memory()
    print()
    test_router()
    print()
    test_halting()
    print()
    test_full_model()
    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
