"""Text generation with the Resonance model."""
import torch
import torch.nn.functional as F

from neurogenesis_v2.model.resonance_model import ResonanceModel


def generate(
    model: ResonanceModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    max_iterations: int = None,
    device: str = 'cpu',
) -> str:
    model.eval()
    model = model.to(device)

    if max_iterations is None:
        max_iterations = model.config.max_iterations

    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    token_ids = [min(t, model.config.vocab_size - 1) for t in token_ids]

    generated_ids = list(token_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Use last max_seq_len tokens
            ctx = generated_ids[-model.config.max_seq_len:]
            input_ids = torch.tensor([ctx], dtype=torch.long, device=device)

            output = model(input_ids, max_iterations=max_iterations)
            logits = output['logits'].squeeze(0)

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = logits.topk(top_k)
                logits[logits < topk_vals[-1]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            eos_id = tokenizer.token_to_id("<eos>")
            if next_token == eos_id:
                break

            generated_ids.append(next_token)

    # Decode only the generated part
    generated_text = tokenizer.decode(generated_ids[len(token_ids):])
    return generated_text
