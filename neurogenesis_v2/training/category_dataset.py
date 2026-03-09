"""Per-category dataset for V2.1 experiments — returns (input_ids, target_ids, category_idx).

Two loading modes:
  1. From pre-tokenized .pt file (fast, preferred)
  2. From JSONL + tokenizer (slow, used to create the .pt file)
"""
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

    def __init__(self, samples_tensor, categories_tensor):
        """Initialize from pre-built tensors.

        Args:
            samples_tensor: (N, seq_len+1) token IDs
            categories_tensor: (N,) category indices
        """
        self.samples = samples_tensor
        self.categories = categories_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        input_ids = ids[:-1]     # (seq_len,)
        target_ids = ids[1:]     # (seq_len,)
        cat_idx = self.categories[idx].item()
        return input_ids, target_ids, cat_idx


def pretokenize_corpus(jsonl_path, tokenizer, seq_len=128, save_path=None):
    """Tokenize JSONL corpus and save as .pt for fast loading.

    This is the slow step — run once, then load from .pt.
    """
    all_windows = []
    all_cats = []
    vocab_size = tokenizer.get_vocab_size()

    print(f"Pre-tokenizing {jsonl_path}...")
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            text = entry['text']
            category = entry['category']
            cat_idx = CATEGORY_MAP.get(category, 3)

            enc = tokenizer.encode(text)
            ids = [min(i, vocab_size - 1) for i in enc.ids]

            # Create overlapping windows
            if len(ids) >= seq_len + 1:
                for start in range(0, len(ids) - seq_len, seq_len // 2):
                    window = ids[start:start + seq_len + 1]
                    if len(window) == seq_len + 1:
                        all_windows.append(window)
                        all_cats.append(cat_idx)
            elif len(ids) > 10:
                padded = ids + [0] * (seq_len + 1 - len(ids))
                all_windows.append(padded[:seq_len + 1])
                all_cats.append(cat_idx)

            count += 1
            if count % 5000 == 0:
                print(f"  Processed {count} entries, {len(all_windows)} windows...")

    samples_tensor = torch.tensor(all_windows, dtype=torch.long)
    cats_tensor = torch.tensor(all_cats, dtype=torch.long)

    print(f"  Total: {count} entries → {len(all_windows)} training windows")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({'samples': samples_tensor, 'categories': cats_tensor}, save_path)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"  Saved to {save_path} ({size_mb:.1f} MB)")

    return CategoryTextDataset(samples_tensor, cats_tensor)


def load_pretokenized(path):
    """Load pre-tokenized dataset from .pt file."""
    data = torch.load(path, weights_only=True)
    return CategoryTextDataset(data['samples'], data['categories'])
