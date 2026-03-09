"""Per-category dataset for V2.1 experiments — returns (input_ids, target_ids, category_idx)."""
import json
import os
import torch
from torch.utils.data import Dataset


CATEGORY_MAP = {
    'philosophy': 0,
    'fiction': 1,
    'science': 2,
    'synthetic': 3,
    'quran': 4,
}
CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAP.items()}


class CategoryTextDataset(Dataset):
    """Dataset that returns tokenized chunks with category labels."""

    def __init__(self, jsonl_path, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.samples = []  # list of (token_ids, category_idx)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry['text']
                category = entry['category']
                cat_idx = CATEGORY_MAP.get(category, 3)  # default to synthetic

                enc = tokenizer.encode(text)
                ids = enc.ids

                # Clamp to vocab_size
                ids = [min(i, tokenizer.get_vocab_size() - 1) for i in ids]

                # Create overlapping windows of seq_len+1 tokens
                if len(ids) >= seq_len + 1:
                    for start in range(0, len(ids) - seq_len, seq_len // 2):
                        window = ids[start:start + seq_len + 1]
                        if len(window) == seq_len + 1:
                            self.samples.append((window, cat_idx))
                elif len(ids) > 10:
                    # Pad short sequences
                    padded = ids + [0] * (seq_len + 1 - len(ids))
                    self.samples.append((padded[:seq_len + 1], cat_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, cat_idx = self.samples[idx]
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        input_ids = ids_tensor[:-1]     # (seq_len,)
        target_ids = ids_tensor[1:]     # (seq_len,)
        return input_ids, target_ids, cat_idx
