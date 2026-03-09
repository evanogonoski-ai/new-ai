"""Phase 3: Joint training of all components."""
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurogenesis.config import NeurogenesisConfig
from neurogenesis.model.neurogenesis import NeurogenesisModel
from neurogenesis.training.data import load_tinystories
from neurogenesis.training.losses import (
    homeostatic_loss, ponder_cost_loss, entropy_bonus, multitimescale_loss
)
from neurogenesis.training.hebbian import hebbian_update
from neurogenesis.tokenizer.bpe import load_tokenizer


def train_phase3(
    model: NeurogenesisModel,
    config: NeurogenesisConfig,
    num_steps: int = None,
    seq_len: int = 64,
    device: str = 'cpu',
    checkpoint_path: str = None,
    tokenizer_path: str = None,
) -> NeurogenesisModel:
    """Joint training of all components with full loss landscape."""
    if num_steps is None:
        num_steps = config.phase3_steps

    model = model.to(device)
    model.train()

    # Ensure all params are trainable
    for param in model.parameters():
        param.requires_grad = True

    tokenizer = load_tokenizer(tokenizer_path)
    train_dataset = load_tinystories(tokenizer, seq_len=seq_len, max_samples=100000)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.phase3_lr)

    # Running average loss for Hebbian reward signal
    running_loss = None
    running_alpha = 0.99

    step = 0
    epoch = 0
    losses_history = []
    last_checkpoint_time = time.time()

    print(f"\nPhase 3: Joint training of all components")
    print(f"  Steps: {num_steps}, LR: {config.phase3_lr}")

    while step < num_steps:
        epoch += 1
        pbar = tqdm(loader, desc=f"Phase 3 Epoch {epoch}", leave=False)
        for input_ids, target_ids in pbar:
            if step >= num_steps:
                break

            input_ids = input_ids.to(device).clamp(0, config.vocab_size - 1)
            target_ids = target_ids.to(device).clamp(0, config.vocab_size - 1)

            # Anneal Gumbel temperature (cosine schedule)
            progress = step / num_steps
            temp = config.gumbel_temperature_end + 0.5 * (
                config.gumbel_temperature_start - config.gumbel_temperature_end
            ) * (1 + math.cos(math.pi * progress))
            model.router.temperature = temp

            # Anneal entropy bonus
            entropy_lambda = config.entropy_bonus_end + (
                config.entropy_bonus_start - config.entropy_bonus_end
            ) * (1 - progress)

            # Anneal homeostatic pressure
            homeo_lambda = config.homeostatic_lambda_start + (
                config.homeostatic_lambda_end - config.homeostatic_lambda_start
            ) * progress

            # Forward pass
            output = model(input_ids, training=True)

            # Main loss: next-token prediction
            logits = output['logits']
            targets = target_ids[:, -1]
            ce_loss = F.cross_entropy(logits, targets)

            # Auxiliary losses
            ponder_loss = ponder_cost_loss(output['ponder_cost'], config.ponder_cost_lambda)
            homeo_loss = homeostatic_loss(model.primitive_library, homeo_lambda)
            mt_loss = multitimescale_loss(output, targets,
                                          config.surface_loss_weight,
                                          config.syntactic_loss_weight,
                                          config.semantic_loss_weight)

            # Entropy bonus (encourages diverse primitive selection)
            compressed = model.working_memory.read_compressed(
                model.working_memory.initialize(model.embedding(input_ids))
            )
            ent_loss = entropy_bonus(model.router, compressed.detach(), entropy_lambda)

            total_loss = ce_loss + ponder_loss + homeo_loss + mt_loss + ent_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Hebbian update (supplementary)
            current_loss = ce_loss.item()
            if running_loss is None:
                running_loss = current_loss
            else:
                running_loss = running_alpha * running_loss + (1 - running_alpha) * current_loss

            global_reward = (running_loss - current_loss) / (running_loss + 1e-8)
            global_reward = max(-1.0, min(1.0, global_reward))

            if output['selected_primitives'] and abs(global_reward) > 0.01:
                hebbian_update(
                    model.primitive_library,
                    output['selected_primitives'][0],
                    compressed.detach(),
                    output['iteration_outputs'][0].detach() if output['iteration_outputs'] else compressed.detach(),
                    global_reward,
                    config.hebbian_lr,
                )

            # Update usage tracking
            for sel in output['selected_primitives']:
                model.primitive_library.update_usage(sel, config.usage_ema_decay)

            losses_history.append(current_loss)
            step += 1

            if step % 100 == 0:
                avg_loss = sum(losses_history[-100:]) / min(100, len(losses_history))
                ppl = math.exp(min(avg_loss, 20))
                usage_ent = _usage_entropy(model.primitive_library)
                pbar.set_postfix(
                    loss=f"{avg_loss:.3f}", ppl=f"{ppl:.1f}",
                    temp=f"{temp:.2f}", ent=f"{usage_ent:.2f}", step=step
                )

            # Checkpoint
            if checkpoint_path and (time.time() - last_checkpoint_time) > config.checkpoint_interval_minutes * 60:
                torch.save(model.state_dict(), checkpoint_path)
                last_checkpoint_time = time.time()

    # Final report
    avg_loss = sum(losses_history[-100:]) / min(100, len(losses_history))
    ppl = math.exp(min(avg_loss, 20))
    usage_ent = _usage_entropy(model.primitive_library)

    print(f"\nPhase 3 Complete!")
    print(f"  Final avg loss: {avg_loss:.3f}")
    print(f"  Final perplexity: {ppl:.1f}")
    print(f"  Primitive usage entropy: {usage_ent:.2f}")
    print(f"  Gumbel temperature: {model.router.temperature:.3f}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

    return model


def _usage_entropy(library) -> float:
    """Compute entropy of primitive usage distribution."""
    usage = library.usage_frequency
    if usage.sum() == 0:
        return 0.0
    probs = usage / (usage.sum() + 1e-8)
    probs = probs[probs > 0]
    return -(probs * probs.log()).sum().item()
