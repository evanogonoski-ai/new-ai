from dataclasses import dataclass


@dataclass
class ResonanceConfig:
    # Model dimensions
    d_model: int = 128
    d_head: int = 16
    n_heads: int = 8
    d_ffn_start: int = 512
    d_ffn_max: int = 1024
    vocab_size: int = 8192
    max_seq_len: int = 128

    # Resonance loop
    max_iterations: int = 16
    min_iterations: int = 2
    convergence_threshold: float = 0.01
    momentum_beta_init: float = 0.1
    ponder_cost_lambda: float = 0.01

    # Entropy gating
    gate_sparsity_lambda: float = 0.01

    # Neuronal mitosis
    mitosis_enabled: bool = False
    mitosis_check_interval: int = 1000
    mitosis_gradient_var_threshold: float = 2.0
    prune_activation_threshold: float = 0.01
    prune_gradient_threshold: float = 0.01
    min_alive_neurons: int = 256
    max_alive_neurons: int = 1024
    gradient_variance_ema_decay: float = 0.99

    # Multi-timescale prediction
    surface_loss_weight: float = 0.3
    semantic_loss_weight: float = 0.1
    surface_state_index: int = 1
    semantic_state_index: int = 3

    # Training
    phase1_lr: float = 1e-3
    phase2_lr: float = 3e-4
    phase3_lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    phase1_steps: int = 20000
    phase2_steps: int = 50000
    phase3_steps: int = 100000
    gradient_clip: float = 1.0

    # Inference
    generation_temperature: float = 0.8
    top_k: int = 50
    max_new_tokens: int = 256
