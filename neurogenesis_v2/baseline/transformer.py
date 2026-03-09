"""Matched-parameter baseline transformer for fair comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineTransformer(nn.Module):
    """
    Standard transformer matched to Resonance's ~1.25M active params.
    2 layers, 8 heads, d_model=128, d_ffn=512.
    """

    def __init__(self, vocab_size: int = 8192, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 2, d_ffn: int = 512,
                 max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=0.1, batch_first=True,
            norm_first=True,  # pre-norm to match Resonance
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Weight tying
        self.output_proj.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(S, S, device=input_ids.device), diagonal=1
        ).bool()
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.final_norm(x)

        logits = self.output_proj(x[:, -1, :])  # predict from last position
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
