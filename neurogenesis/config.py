from dataclasses import dataclass


@dataclass
class NeurogenesisConfig:
    # Model dimensions
    d_model: int = 128
    d_inner: int = 256
    vocab_size: int = 8192

    # Primitive library
    num_primitives: int = 64
    num_structured_primitives: int = 16
    max_primitives: int = 128  # cap after consolidation

    # Router
    num_selected_primitives: int = 4  # K primitives per loop iteration
    gumbel_temperature_start: float = 2.0
    gumbel_temperature_end: float = 0.1
    entropy_bonus_start: float = 0.1
    entropy_bonus_end: float = 0.01

    # Working memory
    num_memory_slots: int = 64

    # Recurrent loop
    max_iterations: int = 16
    min_iterations: int = 2
    halt_epsilon: float = 0.01
    ponder_cost_lambda: float = 0.01

    # Homeostatic pressure
    homeostatic_lambda_start: float = 0.001
    homeostatic_lambda_end: float = 0.01
    usage_ema_decay: float = 0.999

    # Multi-timescale prediction
    surface_loss_weight: float = 0.3
    syntactic_loss_weight: float = 0.1
    semantic_loss_weight: float = 0.05

    # Hebbian learning
    hebbian_lr: float = 1e-5

    # Training
    phase1_lr: float = 1e-3
    phase2_lr: float = 3e-4
    phase3_lr: float = 1e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    phase1_steps: int = 10000
    phase2_steps: int = 50000
    phase3_steps: int = 200000
    consolidation_interval: int = 10000
    checkpoint_interval_minutes: int = 30

    # Inference
    default_max_iterations: int = 8
    generation_temperature: float = 0.8
    max_new_tokens: int = 256
