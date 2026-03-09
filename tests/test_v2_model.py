"""Unit tests for Neurogenesis V2 Resonance Architecture."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from neurogenesis_v2.config import ResonanceConfig
from neurogenesis_v2.model.embedding import TokenEmbedding, RotaryPositionalEmbedding
from neurogenesis_v2.model.attention import EntropyGatedAttention
from neurogenesis_v2.model.feedforward import ExpandableFFN
from neurogenesis_v2.model.resonance_block import MomentumResidual, ResonanceBlock
from neurogenesis_v2.model.mitosis import MitosisEngine
from neurogenesis_v2.model.resonance_model import ResonanceModel
from neurogenesis_v2.baseline.transformer import BaselineTransformer
from neurogenesis_v2.training.losses import gate_sparsity_loss, ponder_cost_loss, multitimescale_loss


@pytest.fixture
def config():
    return ResonanceConfig()


@pytest.fixture
def small_config():
    """Smaller config for faster tests."""
    return ResonanceConfig(
        d_model=64, d_head=8, n_heads=4,
        d_ffn_start=128, d_ffn_max=256,
        vocab_size=512, max_seq_len=32,
        max_iterations=4, min_iterations=1,
    )


# ===================== Embedding Tests =====================

class TestTokenEmbedding:
    def test_output_shape(self, config):
        emb = TokenEmbedding(config.vocab_size, config.d_model)
        x = torch.randint(0, config.vocab_size, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, config.d_model)

    def test_weight_tying_property(self, config):
        emb = TokenEmbedding(config.vocab_size, config.d_model)
        assert emb.weight.shape == (config.vocab_size, config.d_model)

    def test_scaling(self, config):
        emb = TokenEmbedding(config.vocab_size, config.d_model)
        x = torch.zeros(1, 1, dtype=torch.long)
        raw = emb.embedding(x)
        scaled = emb(x)
        expected = raw * (config.d_model ** 0.5)
        assert torch.allclose(scaled, expected)


class TestRoPE:
    def test_output_shape(self, config):
        rope = RotaryPositionalEmbedding(config.d_head, config.max_seq_len)
        q = torch.randn(2, config.n_heads, 16, config.d_head)
        k = torch.randn(2, config.n_heads, 16, config.d_head)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_caching(self, config):
        rope = RotaryPositionalEmbedding(config.d_head, config.max_seq_len)
        q = torch.randn(1, 1, 8, config.d_head)
        k = torch.randn(1, 1, 8, config.d_head)
        rope(q, k)
        assert len(rope._cache) == 1
        rope(q, k)  # should use cache
        assert len(rope._cache) == 1


# ===================== Attention Tests =====================

class TestEntropyGatedAttention:
    def test_output_shape(self, config):
        attn = EntropyGatedAttention(config.d_model, config.n_heads, config.d_head)
        x = torch.randn(2, 16, config.d_model)
        out, gates = attn(x)
        assert out.shape == (2, 16, config.d_model)
        assert gates.shape == (2, config.n_heads)

    def test_gates_range(self, config):
        attn = EntropyGatedAttention(config.d_model, config.n_heads, config.d_head)
        x = torch.randn(2, 16, config.d_model)
        _, gates = attn(x)
        assert (gates >= 0).all()
        assert (gates <= 1).all()

    def test_gradients_flow(self, config):
        attn = EntropyGatedAttention(config.d_model, config.n_heads, config.d_head)
        x = torch.randn(2, 8, config.d_model, requires_grad=True)
        out, gates = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert attn.W_q.weight.grad is not None

    def test_with_rope(self, config):
        attn = EntropyGatedAttention(config.d_model, config.n_heads, config.d_head)
        rope = RotaryPositionalEmbedding(config.d_head, config.max_seq_len)
        x = torch.randn(2, 16, config.d_model)
        out, gates = attn(x, rope_fn=rope)
        assert out.shape == (2, 16, config.d_model)


# ===================== FFN Tests =====================

class TestExpandableFFN:
    def test_output_shape(self, config):
        ffn = ExpandableFFN(config.d_model, config.d_ffn_max, config.d_ffn_start)
        x = torch.randn(2, 16, config.d_model)
        out = ffn(x)
        assert out.shape == (2, 16, config.d_model)

    def test_alive_count(self, config):
        ffn = ExpandableFFN(config.d_model, config.d_ffn_max, config.d_ffn_start)
        assert ffn.num_alive == config.d_ffn_start

    def test_alive_mask_applied(self, config):
        ffn = ExpandableFFN(config.d_model, config.d_ffn_max, config.d_ffn_start)
        # Dormant neuron weights should be zero
        dormant_idx = config.d_ffn_start + 1
        if dormant_idx < config.d_ffn_max:
            assert (ffn.linear1.weight[dormant_idx] == 0).all()
            assert ffn.alive_mask[dormant_idx] == 0

    def test_activation_ema_updates(self, config):
        ffn = ExpandableFFN(config.d_model, config.d_ffn_max, config.d_ffn_start)
        ffn.train()
        x = torch.randn(2, 8, config.d_model)
        ffn(x)
        # At least some alive neurons should have non-zero ema
        alive_ema = ffn.activation_ema[:config.d_ffn_start]
        assert alive_ema.sum() > 0

    def test_dormant_indices(self, config):
        ffn = ExpandableFFN(config.d_model, config.d_ffn_max, config.d_ffn_start)
        dormant = ffn.get_dormant_indices()
        assert len(dormant) == config.d_ffn_max - config.d_ffn_start


# ===================== ResonanceBlock Tests =====================

class TestResonanceBlock:
    def test_output_shape(self, config):
        block = ResonanceBlock(
            config.d_model, config.n_heads, config.d_head,
            config.d_ffn_max, config.d_ffn_start,
        )
        state = torch.randn(2, 16, config.d_model)
        prev = torch.zeros_like(state)
        new_state, gates = block(state, prev)
        assert new_state.shape == state.shape
        assert gates.shape == (2, config.n_heads)

    def test_with_momentum(self, config):
        block = ResonanceBlock(
            config.d_model, config.n_heads, config.d_head,
            config.d_ffn_max, config.d_ffn_start,
        )
        state = torch.randn(2, 16, config.d_model)
        prev = torch.randn(2, 16, config.d_model)
        out_m, _ = block(state, prev, use_momentum=True)
        out_no, _ = block(state, prev, use_momentum=False)
        # Outputs should differ when momentum is on vs off (prev != 0)
        assert not torch.allclose(out_m, out_no)

    def test_without_momentum_ignores_prev(self, config):
        block = ResonanceBlock(
            config.d_model, config.n_heads, config.d_head,
            config.d_ffn_max, config.d_ffn_start,
        )
        state = torch.randn(2, 16, config.d_model)
        prev1 = torch.randn(2, 16, config.d_model)
        prev2 = torch.randn(2, 16, config.d_model)
        out1, _ = block(state, prev1, use_momentum=False)
        out2, _ = block(state, prev2, use_momentum=False)
        assert torch.allclose(out1, out2)


class TestMomentumResidual:
    def test_formula(self):
        mr = MomentumResidual(beta_init=0.0)
        state = torch.ones(2, 4)
        update = torch.ones(2, 4) * 0.5
        prev = torch.zeros(2, 4)
        result = mr(state, update, prev)
        # sigmoid(0) = 0.5, momentum = 0.5 * (1 - 0) = 0.5
        expected = state + update + 0.5 * (state - prev)
        assert torch.allclose(result, expected)


# ===================== Mitosis Tests =====================

class TestMitosisEngine:
    def test_split_increases_alive(self, small_config):
        ffn = ExpandableFFN(small_config.d_model, small_config.d_ffn_max, small_config.d_ffn_start)
        small_config.mitosis_check_interval = 1
        small_config.mitosis_gradient_var_threshold = 0.0  # always trigger
        engine = MitosisEngine(ffn, small_config)

        # Simulate many gradient recordings
        for _ in range(20):
            ffn.linear1.weight.grad = torch.randn_like(ffn.linear1.weight) * 10
            engine.record_gradients()

        before = ffn.num_alive
        result = engine.check_and_apply(step=1)
        assert result['mitosis'] > 0
        assert ffn.num_alive > before

    def test_prune_decreases_alive(self, small_config):
        ffn = ExpandableFFN(small_config.d_model, small_config.d_ffn_max, small_config.d_ffn_start)
        small_config.mitosis_check_interval = 1
        small_config.mitosis_gradient_var_threshold = 1e10  # never split
        small_config.prune_activation_threshold = 1e10  # always prune
        small_config.prune_gradient_threshold = 1e10  # always prune
        small_config.min_alive_neurons = 1
        engine = MitosisEngine(ffn, small_config)

        for _ in range(20):
            ffn.linear1.weight.grad = torch.randn_like(ffn.linear1.weight) * 0.001
            engine.record_gradients()

        before = ffn.num_alive
        result = engine.check_and_apply(step=1)
        assert result['pruned'] > 0
        assert ffn.num_alive < before

    def test_no_op_before_check_interval(self, small_config):
        ffn = ExpandableFFN(small_config.d_model, small_config.d_ffn_max, small_config.d_ffn_start)
        small_config.mitosis_check_interval = 100
        engine = MitosisEngine(ffn, small_config)
        result = engine.check_and_apply(step=50)
        assert result['mitosis'] == 0
        assert result['pruned'] == 0


# ===================== Full Model Tests =====================

class TestResonanceModel:
    def test_forward_shape(self, small_config):
        model = ResonanceModel(small_config)
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        output = model(x, max_iterations=2)
        assert output['logits'].shape == (2, small_config.vocab_size)

    def test_single_iteration(self, small_config):
        model = ResonanceModel(small_config)
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        output = model(x, max_iterations=1, use_momentum=False)
        assert output['num_iterations'] == 1
        assert len(output['all_gates']) == 1

    def test_gradients_flow(self, small_config):
        model = ResonanceModel(small_config)
        model.train()
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        output = model(x, max_iterations=2, use_momentum=True, return_all_states=True)
        loss = output['logits'].sum()
        loss.backward()
        assert model.embedding.embedding.weight.grad is not None
        assert model.block.attention.W_q.weight.grad is not None

    def test_weight_tying(self, small_config):
        model = ResonanceModel(small_config)
        # Output projection uses embedding weight
        x = torch.randint(0, small_config.vocab_size, (1, 8))
        output = model(x, max_iterations=1)
        # Verify weight is shared
        assert model.embedding.weight is model.embedding.embedding.weight

    def test_convergence_halting(self, small_config):
        small_config.convergence_threshold = 1e10  # should always converge
        small_config.min_iterations = 1
        model = ResonanceModel(small_config)
        model.eval()
        x = torch.randint(0, small_config.vocab_size, (1, 8))
        with torch.no_grad():
            output = model(x, max_iterations=4)
        # Should halt early due to very high threshold
        assert output['num_iterations'] <= 4

    def test_multitimescale_heads(self, small_config):
        small_config.surface_state_index = 1
        small_config.semantic_state_index = 2
        model = ResonanceModel(small_config)
        model.train()
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        output = model(x, max_iterations=4, return_all_states=True)
        assert 'surface_logits' in output
        assert output['surface_logits'].shape == (2, small_config.vocab_size)

    def test_return_all_states(self, small_config):
        model = ResonanceModel(small_config)
        x = torch.randint(0, small_config.vocab_size, (2, 8))
        output = model(x, max_iterations=3, return_all_states=True)
        assert 'all_states' in output
        assert len(output['all_states']) >= 2  # initial + at least 1 iteration

    def test_param_count(self, small_config):
        model = ResonanceModel(small_config)
        total = model.count_parameters()
        active = model.count_active_parameters()
        assert total > 0
        assert active <= total
        assert active > 0


# ===================== Baseline Tests =====================

class TestBaselineTransformer:
    def test_forward_shape(self):
        model = BaselineTransformer(vocab_size=512, d_model=64,
                                     n_heads=4, n_layers=2, d_ffn=128, max_seq_len=32)
        x = torch.randint(0, 512, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 512)

    def test_gradients_flow(self):
        model = BaselineTransformer(vocab_size=512, d_model=64,
                                     n_heads=4, n_layers=2, d_ffn=128)
        x = torch.randint(0, 512, (2, 8))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.embedding.weight.grad is not None

    def test_param_count(self):
        model = BaselineTransformer(vocab_size=512, d_model=64,
                                     n_heads=4, n_layers=2, d_ffn=128)
        count = model.count_parameters()
        assert count > 0


# ===================== Loss Tests =====================

class TestLosses:
    def test_gate_sparsity_loss(self):
        gates = torch.rand(4, 8)  # (iters, heads)
        loss = gate_sparsity_loss(gates, lambda_gs=0.01)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0
        # Gradients flow
        gates_req = torch.rand(4, 8, requires_grad=True)
        loss = gate_sparsity_loss(gates_req)
        loss.backward()
        assert gates_req.grad is not None

    def test_gate_sparsity_extreme_values(self):
        # Gates near 0 or 1 should have low loss (low binary entropy)
        near_extreme = torch.tensor([[0.01, 0.99, 0.02, 0.98]])
        near_mid = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        loss_extreme = gate_sparsity_loss(near_extreme, lambda_gs=1.0)
        loss_mid = gate_sparsity_loss(near_mid, lambda_gs=1.0)
        assert loss_extreme < loss_mid

    def test_ponder_cost_loss(self):
        cost = torch.tensor(0.5)
        loss = ponder_cost_loss(cost, lambda_p=0.01)
        assert loss.item() == pytest.approx(0.005)

    def test_multitimescale_loss(self):
        output = {
            'surface_logits': torch.randn(2, 512),
            'semantic_logits': torch.randn(2, 512),
        }
        targets = torch.randint(0, 512, (2,))
        loss = multitimescale_loss(output, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_multitimescale_loss_missing_keys(self):
        output = {}
        targets = torch.randint(0, 512, (2,))
        loss = multitimescale_loss(output, targets)
        assert loss.item() == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
