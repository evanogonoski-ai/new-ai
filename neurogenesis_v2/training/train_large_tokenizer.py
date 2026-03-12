#!/usr/bin/env python3
"""Train a BPE tokenizer on the full large corpus for the 10M parameter experiment.

Reads all .txt files from all data/raw_texts/ subdirectories, plus existing
JSONL corpora (data/train_corpus.jsonl, data/expanded_corpus.jsonl), and trains
a BPE tokenizer with vocab_size=8192 using the HuggingFace tokenizers library.

Output: tokenizer/large_corpus_tokenizer.json
"""
import json
import os
import sys

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

# All raw text source directories
RAW_TEXT_DIRS = [
    'data/raw_texts/gutenberg',
    'data/raw_texts/gutenberg_expanded',
    'data/raw_texts/synthetic',
    'data/raw_texts/synthetic_scaled',
    'data/raw_texts/additional_synthetic',
    'data/raw_texts/quran',
    'data/raw_texts/wikipedia_synthetic',
]

# Existing JSONL corpora to also include
JSONL_SOURCES = [
    'data/train_corpus.jsonl',
    'data/expanded_corpus.jsonl',
]

SAVE_PATH = os.path.join(BASE_DIR, 'tokenizer', 'large_corpus_tokenizer.json')
VOCAB_SIZE = 8192
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']

# Sample sentences for verifying the trained tokenizer
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "To be, or not to be, that is the question.",
    "In the beginning God created the heavens and the earth.",
    "The theory of evolution by natural selection was proposed by Darwin.",
    "Machine learning models require large amounts of training data.",
]


def collect_raw_texts():
    """Read all .txt files from all raw_texts subdirectories."""
    texts = []
    total_words = 0
    dir_stats = {}

    for rel_dir in RAW_TEXT_DIRS:
        abs_dir = os.path.join(BASE_DIR, rel_dir)
        if not os.path.exists(abs_dir):
            print(f"  Warning: {rel_dir} not found, skipping")
            continue

        dir_words = 0
        file_count = 0
        for fname in sorted(os.listdir(abs_dir)):
            if not fname.endswith('.txt'):
                continue
            filepath = os.path.join(abs_dir, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                continue

            # Split into ~500 word chunks for tokenizer training efficiency
            words = content.split()
            dir_words += len(words)
            file_count += 1
            for i in range(0, len(words), 500):
                chunk = ' '.join(words[i:i + 500])
                if chunk.strip():
                    texts.append(chunk)

        total_words += dir_words
        dir_label = os.path.basename(rel_dir)
        dir_stats[dir_label] = {'files': file_count, 'words': dir_words}
        print(f"  {dir_label}: {file_count} files, {dir_words:,} words")

    print(f"  Raw texts total: {total_words:,} words across {len(texts):,} chunks")
    return texts, dir_stats


def collect_jsonl_texts():
    """Read texts from existing JSONL corpus files."""
    texts = []
    total_entries = 0

    for rel_path in JSONL_SOURCES:
        abs_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(abs_path):
            print(f"  Warning: {rel_path} not found, skipping")
            continue

        count = 0
        with open(abs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    if text.strip():
                        texts.append(text)
                        count += 1
                except json.JSONDecodeError:
                    continue

        total_entries += count
        print(f"  {os.path.basename(rel_path)}: {count:,} entries")

    print(f"  JSONL total: {total_entries:,} entries")
    return texts


def main():
    print("=" * 60)
    print("TRAINING LARGE CORPUS TOKENIZER")
    print("=" * 60)
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print(f"Save path: {SAVE_PATH}")
    print()

    # ---------------------------------------------------------------
    # Collect training texts
    # ---------------------------------------------------------------
    print("Collecting raw text files...")
    raw_texts, dir_stats = collect_raw_texts()
    print()

    print("Collecting JSONL corpus texts...")
    jsonl_texts = collect_jsonl_texts()
    print()

    all_texts = raw_texts + jsonl_texts
    print(f"Total training chunks: {len(all_texts):,}")
    print()

    if not all_texts:
        print("ERROR: No training texts found. Check that data directories exist.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Train BPE tokenizer
    # ---------------------------------------------------------------
    print("Training BPE tokenizer...")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer: Unicode NFC normalization
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer: byte-level encoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(all_texts, trainer=trainer)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    tokenizer.save(SAVE_PATH)
    print(f"\nTokenizer saved to {SAVE_PATH}")

    # ---------------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------------
    actual_vocab = tokenizer.get_vocab_size()
    print(f"\nVocab size: {actual_vocab}")

    # Verify special tokens
    print(f"\nSpecial tokens:")
    for tok in SPECIAL_TOKENS:
        tok_id = tokenizer.token_to_id(tok)
        print(f"  {tok} -> id {tok_id}")

    # Sample encodings
    print(f"\nSample encodings:")
    for sentence in SAMPLE_SENTENCES:
        encoded = tokenizer.encode(sentence)
        n_tokens = len(encoded.ids)
        # Show first few token strings
        token_strs = [tokenizer.id_to_token(tid) for tid in encoded.ids[:10]]
        suffix = "..." if n_tokens > 10 else ""
        print(f"  \"{sentence[:60]}{'...' if len(sentence) > 60 else ''}\"")
        print(f"    -> {n_tokens} tokens: {token_strs}{suffix}")

    # Compression ratio estimate
    print(f"\nCompression ratio estimate:")
    total_chars = sum(len(t) for t in all_texts[:1000])
    total_toks = sum(len(tokenizer.encode(t).ids) for t in all_texts[:1000])
    if total_toks > 0:
        ratio = total_chars / total_toks
        print(f"  ~{ratio:.1f} chars/token (estimated from first 1000 chunks)")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
