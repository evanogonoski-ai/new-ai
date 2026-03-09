"""Text generation with the Neurogenesis model."""
import torch
import torch.nn.functional as F

from neurogenesis.model.neurogenesis import NeurogenesisModel, apply_rotary_embeddings


def generate(
    model: NeurogenesisModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_iterations: int = None,
    device: str = 'cpu',
) -> str:
    """
    Generate text autoregressively.

    Args:
        model: trained NeurogenesisModel
        tokenizer: BPE tokenizer
        prompt: input text
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling threshold
        max_iterations: override loop iterations (lower = faster)
        device: device to run on
    Returns:
        generated text string
    """
    model.eval()
    model = model.to(device)

    if max_iterations is None:
        max_iterations = model.config.default_max_iterations

    # Tokenize prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    input_ids = input_ids.clamp(0, model.config.vocab_size - 1)

    # Initialize working memory from prompt
    token_emb = model.embedding(input_ids)
    seq_len = token_emb.shape[1]
    token_emb = apply_rotary_embeddings(token_emb, seq_len)
    memory = model.working_memory.initialize(token_emb)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Run recurrent processing loop
            cumulative_prob = torch.zeros(1, 1, device=device)
            active = True

            for t in range(max_iterations):
                if not active:
                    break

                compressed = model.working_memory.read_compressed(memory)
                indices, weights, slot_masks = model.router(compressed, training=False)

                for k in range(model.config.num_selected_primitives):
                    idx = indices[0, k].item()
                    mask_k = slot_masks[:, k, :]
                    out, gate = model.primitive_library(idx, memory)
                    memory = model.working_memory.apply_gated_write(
                        memory, mask_k, out, gate
                    )

                compressed = model.working_memory.read_compressed(memory)
                halt_prob, cumulative_prob, should_halt = model.halting(
                    compressed, t, cumulative_prob
                )
                if should_halt.all():
                    active = False

            # Project to logits
            pooled = model._attention_pool(memory)
            logits = model._project_to_logits(pooled)  # (1, vocab_size)
            logits = logits.squeeze(0)  # (vocab_size,)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = logits.topk(top_k)
                threshold = top_k_vals[-1]
                logits[logits < threshold] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative probability above threshold
                remove_mask = cumulative_probs > top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                sorted_logits[remove_mask] = float('-inf')
                # Scatter back
                logits.scatter_(0, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            # Check for EOS
            eos_id = tokenizer.token_to_id("<eos>")
            if next_token == eos_id:
                break

            generated_ids.append(next_token)

            # Update working memory with new token
            new_emb = model.embedding(torch.tensor([[next_token]], device=device))
            new_emb = apply_rotary_embeddings(new_emb, 1)
            memory = model.working_memory.update_for_generation(memory, new_emb.squeeze(1))

    # Decode
    generated_text = tokenizer.decode(generated_ids)
    return generated_text
