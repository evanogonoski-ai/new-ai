#!/usr/bin/env python3
"""Train BPE tokenizer on the full expanded corpus (all categories)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis.tokenizer.bpe import train_tokenizer

CORPUS_DIR = 'corpus/raw'
SAVE_PATH = 'tokenizer/expanded_tokenizer.json'
VOCAB_SIZE = 8192

# Key terms that should ideally be single tokens
KEY_TERMS = [
    "virtue", "justice", "species", "hypothesis", "wisdom",
    "courage", "nature", "reason", "truth", "knowledge",
]


def collect_all_raw_texts():
    """Read all raw text files from corpus/raw/{category}/."""
    texts = []
    total_words = 0
    categories = ['philosophy', 'fiction', 'science', 'synthetic', 'quran']

    for category in categories:
        cat_dir = os.path.join(CORPUS_DIR, category)
        if not os.path.exists(cat_dir):
            print(f"  Warning: {cat_dir} not found, skipping")
            continue

        cat_words = 0
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith('.txt'):
                continue
            filepath = os.path.join(cat_dir, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into ~500 word chunks for tokenizer training
            words = content.split()
            cat_words += len(words)
            for i in range(0, len(words), 500):
                chunk = ' '.join(words[i:i+500])
                if chunk.strip():
                    texts.append(chunk)

        total_words += cat_words
        print(f"  {category}: {cat_words:,} words")

    print(f"  TOTAL: {total_words:,} words across {len(texts):,} training chunks")
    return texts


def main():
    print("=" * 60)
    print("TRAINING EXPANDED TOKENIZER")
    print("=" * 60)
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Save path: {SAVE_PATH}")
    print()

    print("Collecting raw texts...")
    texts = collect_all_raw_texts()
    print()

    print("Training BPE tokenizer...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    tokenizer = train_tokenizer(texts, vocab_size=VOCAB_SIZE, save_path=SAVE_PATH)
    print(f"Tokenizer saved to {SAVE_PATH}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Verify key terms
    print(f"\nKey term tokenization:")
    for term in KEY_TERMS:
        enc = tokenizer.encode(term)
        tokens = [tokenizer.id_to_token(i) for i in enc.ids if i not in (
            tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
        )]
        is_single = len(tokens) == 1
        status = "OK (single)" if is_single else f"SPLIT ({len(tokens)} tokens)"
        print(f"  '{term}' → {tokens} — {status}")

    # Verify dialogue punctuation
    print(f"\nDialogue punctuation check:")
    test_dialogue = '"Hello," said Tom. "How are you?"'
    enc = tokenizer.encode(test_dialogue)
    tokens = [tokenizer.id_to_token(i) for i in enc.ids]
    print(f"  Input: {test_dialogue}")
    print(f"  Tokens ({len(tokens)}): {tokens[:15]}...")

    print(f"\nTokenizer training complete.")


if __name__ == '__main__':
    main()
