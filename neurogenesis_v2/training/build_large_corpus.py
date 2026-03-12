#!/usr/bin/env python3
"""Build a large JSONL corpus for the 10M parameter experiment.

Reads raw text files from multiple source directories, chunks on sentence
boundaries at ~256 tokens with 30-token overlap, applies category tags,
and produces train/holdout splits.

Outputs:
  - data/large_train_corpus.jsonl   (training data)
  - data/large_holdout_corpus.jsonl (held-out for generalization testing)

Each line: {"text": "...", "source": "filename_without_ext", "category": "literature"}
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis.tokenizer.bpe import load_tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

# Source directories (relative to BASE_DIR) and their categories
SOURCE_DIRS = {
    'data/raw_texts/gutenberg':            'literature',
    'data/raw_texts/gutenberg_expanded':   'literature',
    'data/raw_texts/synthetic':            'synthetic',
    'data/raw_texts/synthetic_scaled':     'synthetic',
    'data/raw_texts/additional_synthetic': 'synthetic',
    'data/raw_texts/quran':                'religious',
    'data/raw_texts/wikipedia_synthetic':  'wikipedia',
    'data/raw_texts/wikipedia':            'wikipedia',
}

CATEGORY_IDS = {
    'literature': 0,
    'synthetic':  1,
    'religious':  4,
    'wikipedia':  5,
}

TRAIN_OUTPUT = os.path.join(BASE_DIR, 'data', 'large_train_corpus.jsonl')
HOLDOUT_OUTPUT = os.path.join(BASE_DIR, 'data', 'large_holdout_corpus.jsonl')

# Tokenizer trained on the large corpus (or fall back to expanded)
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'large_corpus_tokenizer.json')
FALLBACK_TOKENIZER = os.path.join(BASE_DIR, 'tokenizer', 'expanded_tokenizer.json')

TARGET_CHUNK_TOKENS = 256
OVERLAP_TOKENS = 30

# Holdout authors (matched as prefix of the source filename without extension)
HOLDOUT_AUTHORS = ['plato_', 'jane_austen_', 'charles_darwin_', 'fyodor_dostoevsky_']

# Wikipedia holdout: 10% random articles per category (seeded)
WIKI_HOLDOUT_FRACTION = 0.10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sentence_split(text):
    """Split text into sentences on period/question/exclamation boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text, tokenizer, target_tokens=TARGET_CHUNK_TOKENS,
               overlap_tokens=OVERLAP_TOKENS):
    """Chunk text on sentence boundaries at ~target_tokens with overlap."""
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_token_count = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent).ids)

        if current_token_count + sent_tokens > target_tokens and current_sentences:
            # Emit current chunk
            chunk_text_str = ' '.join(current_sentences)
            chunks.append(chunk_text_str)

            # Overlap: keep last few sentences that fit within overlap_tokens
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_toks = len(tokenizer.encode(s).ids)
                if overlap_count + s_toks > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_toks

            current_sentences = overlap_sents
            current_token_count = overlap_count

        current_sentences.append(sent)
        current_token_count += sent_tokens

    # Final chunk
    if current_sentences:
        chunk_text_str = ' '.join(current_sentences)
        if len(tokenizer.encode(chunk_text_str).ids) > 20:  # min chunk size
            chunks.append(chunk_text_str)

    return chunks


def is_holdout_author(source):
    """Check if a source belongs to a holdout author."""
    return any(source.startswith(prefix) for prefix in HOLDOUT_AUTHORS)


def determine_wiki_holdout(sources_by_category, seed=RANDOM_SEED):
    """Determine which wikipedia sources are held out (10% per category).

    Groups wikipedia files by their subdirectory-inferred category (the filename
    prefix before the first underscore, or the full name). Returns a set of
    source names to hold out.
    """
    rng = random.Random(seed)
    holdout_sources = set()

    # Group by first token of filename as a rough category proxy
    categories = {}
    for src in sources_by_category:
        key = src.split('_')[0] if '_' in src else src
        categories.setdefault(key, []).append(src)

    for cat_key, srcs in categories.items():
        srcs_sorted = sorted(srcs)
        n_holdout = max(1, int(len(srcs_sorted) * WIKI_HOLDOUT_FRACTION))
        selected = rng.sample(srcs_sorted, min(n_holdout, len(srcs_sorted)))
        holdout_sources.update(selected)

    return holdout_sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BUILDING LARGE CORPUS (10M parameter experiment)")
    print("=" * 60)

    # Load tokenizer
    tok_path = TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else FALLBACK_TOKENIZER
    if not os.path.exists(tok_path):
        print(f"ERROR: No tokenizer found at {TOKENIZER_PATH} or {FALLBACK_TOKENIZER}")
        print("Run train_large_tokenizer.py or train_expanded_tokenizer.py first.")
        sys.exit(1)

    tokenizer = load_tokenizer(tok_path)
    print(f"Loaded tokenizer from {tok_path} (vocab={tokenizer.get_vocab_size()})")
    print(f"Chunk target: {TARGET_CHUNK_TOKENS} tokens, overlap: {OVERLAP_TOKENS} tokens")
    print(f"Holdout authors: {HOLDOUT_AUTHORS}")
    print()

    all_chunks = []
    category_stats = {}
    wikipedia_sources = []  # track for wiki holdout selection

    # ------------------------------------------------------------------
    # Pass 1: Collect all wikipedia sources for holdout selection
    # ------------------------------------------------------------------
    for rel_dir, category in SOURCE_DIRS.items():
        if category != 'wikipedia':
            continue
        abs_dir = os.path.join(BASE_DIR, rel_dir)
        if not os.path.exists(abs_dir):
            continue
        for fname in sorted(os.listdir(abs_dir)):
            if fname.endswith('.txt'):
                wikipedia_sources.append(fname.replace('.txt', ''))

    wiki_holdout_set = determine_wiki_holdout(wikipedia_sources)
    if wiki_holdout_set:
        print(f"Wikipedia holdout: {len(wiki_holdout_set)} articles selected")

    # ------------------------------------------------------------------
    # Pass 2: Read, chunk, and tag all files
    # ------------------------------------------------------------------
    for rel_dir, category in SOURCE_DIRS.items():
        abs_dir = os.path.join(BASE_DIR, rel_dir)
        if not os.path.exists(abs_dir):
            print(f"  Warning: {rel_dir} not found, skipping")
            continue

        cat_chunks = 0
        cat_tokens = 0
        dir_label = os.path.basename(rel_dir)

        txt_files = sorted(f for f in os.listdir(abs_dir) if f.endswith('.txt'))
        if not txt_files:
            print(f"  Warning: {rel_dir} has no .txt files, skipping")
            continue

        for fname in txt_files:
            filepath = os.path.join(abs_dir, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                continue

            source = fname.replace('.txt', '')

            # Determine holdout status
            if category == 'wikipedia':
                holdout = source in wiki_holdout_set
            else:
                holdout = is_holdout_author(source)

            # For synthetic texts, split on separator first
            if category == 'synthetic':
                parts = content.split('\n\n---\n\n')
            else:
                parts = [content]

            file_chunks = 0
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                chunks = chunk_text(part, tokenizer)
                for i, chunk in enumerate(chunks):
                    token_count = len(tokenizer.encode(chunk).ids)
                    entry = {
                        'text': chunk,
                        'source': source,
                        'category': category,
                    }
                    all_chunks.append((entry, holdout))
                    cat_tokens += token_count
                file_chunks += len(chunks)

            cat_chunks += file_chunks
            if holdout and file_chunks > 0:
                print(f"    [HOLDOUT] {source}: {file_chunks} chunks ({dir_label})")

        # Accumulate stats per category
        if category not in category_stats:
            category_stats[category] = {'chunks': 0, 'tokens': 0}
        category_stats[category]['chunks'] += cat_chunks
        category_stats[category]['tokens'] += cat_tokens
        print(f"  {dir_label}: {cat_chunks:,} chunks, {cat_tokens:,} tokens -> {category}")

    # ------------------------------------------------------------------
    # Shuffle and split
    # ------------------------------------------------------------------
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_chunks)

    train_entries = [entry for entry, holdout in all_chunks if not holdout]
    holdout_entries = [entry for entry, holdout in all_chunks if holdout]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)

    for path, entries, label in [
        (TRAIN_OUTPUT, train_entries, 'train'),
        (HOLDOUT_OUTPUT, holdout_entries, 'holdout'),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"\n  Saved {label}: {path}")
        print(f"    {len(entries):,} chunks, {size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    total_chunks = len(all_chunks)
    total_tokens = sum(s['tokens'] for s in category_stats.values())

    print(f"\n{'=' * 60}")
    print("CORPUS STATISTICS")
    print(f"{'=' * 60}")
    print(f"{'Category':<15} {'ID':>4} {'Chunks':>10} {'Tokens':>12} {'%':>8}")
    print(f"{'-' * 50}")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        cat_id = CATEGORY_IDS.get(cat, '?')
        pct = (s['tokens'] / total_tokens * 100) if total_tokens > 0 else 0
        print(f"  {cat:<13} {cat_id:>4} {s['chunks']:>10,} {s['tokens']:>12,} {pct:>7.1f}%")
    print(f"{'-' * 50}")
    print(f"  {'TOTAL':<13} {'':>4} {total_chunks:>10,} {total_tokens:>12,}")

    print(f"\n{'=' * 60}")
    print("SPLIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Train:   {len(train_entries):>8,} chunks")
    print(f"  Holdout: {len(holdout_entries):>8,} chunks")
    if total_chunks > 0:
        pct = len(holdout_entries) / total_chunks * 100
        print(f"  Holdout is {pct:.1f}% of corpus")

    print(f"\nDone.")
    return category_stats


if __name__ == '__main__':
    main()
