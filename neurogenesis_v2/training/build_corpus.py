#!/usr/bin/env python3
"""Chunk all raw texts with category tags and save as JSONL.

Reads from corpus/raw/{category}/, chunks on sentence boundaries
at ~128 tokens with 10-20 token overlap, tags with metadata,
saves as data/expanded_corpus.jsonl.

V2.2: Also produces author-holdout split for generalization testing.
  - data/train_corpus.jsonl  (everything except holdout authors)
  - data/holdout_corpus.jsonl (holdout authors only)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis.tokenizer.bpe import load_tokenizer

CORPUS_DIR = 'corpus/raw'
OUTPUT_PATH = 'data/expanded_corpus.jsonl'
TRAIN_PATH = 'data/train_corpus.jsonl'
HOLDOUT_PATH = 'data/holdout_corpus.jsonl'
TOKENIZER_PATH = 'tokenizer/expanded_tokenizer.json'
TARGET_CHUNK_TOKENS = 128
OVERLAP_TOKENS = 15

CATEGORIES = ['philosophy', 'fiction', 'science', 'synthetic', 'quran']

# V2.2: Authors held out for generalization testing.
# Model never sees these during training; evaluated on them afterward.
HOLDOUT_PREFIXES = ['plato_', 'jane_austen_', 'charles_darwin_']


def sentence_split(text):
    """Split text into sentences on period/question/exclamation boundaries."""
    # Split on sentence-ending punctuation followed by space or newline
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


def is_holdout(source):
    """Check if a source belongs to a holdout author."""
    return any(source.startswith(prefix) for prefix in HOLDOUT_PREFIXES)


def build_corpus():
    """Read all raw texts, chunk, tag, and save as JSONL.

    Produces three files:
      - expanded_corpus.jsonl  (full corpus, backward compat)
      - train_corpus.jsonl     (everything except holdout authors)
      - holdout_corpus.jsonl   (holdout authors only)
    """
    print("=" * 60)
    print("BUILDING EXPANDED CORPUS (with V2.2 holdout split)")
    print("=" * 60)

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        print("Run train_expanded_tokenizer.py first.")
        sys.exit(1)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"Loaded tokenizer: vocab={tokenizer.get_vocab_size()}")
    print(f"Holdout authors: {HOLDOUT_PREFIXES}")

    all_chunks = []
    category_stats = {}

    for category in CATEGORIES:
        cat_dir = os.path.join(CORPUS_DIR, category)
        if not os.path.exists(cat_dir):
            print(f"  Warning: {cat_dir} not found, skipping")
            continue

        cat_chunks = 0
        cat_tokens = 0

        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith('.txt'):
                continue

            filepath = os.path.join(cat_dir, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            source = fname.replace('.txt', '')
            holdout = is_holdout(source)

            # For synthetic texts, split on separator first
            if category == 'synthetic':
                parts = content.split('\n\n---\n\n')
            else:
                parts = [content]

            file_chunks = 0
            for part in parts:
                chunks = chunk_text(part.strip(), tokenizer)
                for i, chunk in enumerate(chunks):
                    token_count = len(tokenizer.encode(chunk).ids)
                    entry = {
                        'text': chunk,
                        'category': category,
                        'source': source,
                        'chunk_id': file_chunks + i,
                        'token_count': token_count,
                        'holdout': holdout,
                    }
                    all_chunks.append(entry)
                    cat_tokens += token_count
                file_chunks += len(chunks)

            cat_chunks += file_chunks
            if holdout:
                print(f"    [HOLDOUT] {source}: {file_chunks} chunks")

        category_stats[category] = {'chunks': cat_chunks, 'tokens': cat_tokens}
        print(f"  {category}: {cat_chunks:,} chunks, {cat_tokens:,} tokens")

    # Shuffle chunks
    import random
    random.seed(42)
    random.shuffle(all_chunks)

    # Split into train and holdout
    train_chunks = [c for c in all_chunks if not c['holdout']]
    holdout_chunks = [c for c in all_chunks if c['holdout']]

    # Save all three files
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for path, chunks, label in [
        (OUTPUT_PATH, all_chunks, 'full corpus'),
        (TRAIN_PATH, train_chunks, 'train split'),
        (HOLDOUT_PATH, holdout_chunks, 'holdout split'),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in chunks:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Saved {label}: {path} ({len(chunks):,} chunks, {size_mb:.1f} MB)")

    # Summary
    total_chunks = len(all_chunks)
    total_tokens = sum(s['tokens'] for s in category_stats.values())
    train_tokens = sum(c['token_count'] for c in train_chunks)
    holdout_tokens = sum(c['token_count'] for c in holdout_chunks)

    print(f"\n{'='*60}")
    print("CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<15} {'Chunks':>10} {'Tokens':>12} {'%':>8}")
    print(f"{'-'*45}")
    for cat in CATEGORIES:
        if cat in category_stats:
            s = category_stats[cat]
            pct = (s['tokens'] / total_tokens * 100) if total_tokens > 0 else 0
            print(f"  {cat:<13} {s['chunks']:>10,} {s['tokens']:>12,} {pct:>7.1f}%")
    print(f"{'-'*45}")
    print(f"  {'TOTAL':<13} {total_chunks:>10,} {total_tokens:>12,}")

    print(f"\n{'='*60}")
    print("HOLDOUT SPLIT")
    print(f"{'='*60}")
    print(f"  Train:   {len(train_chunks):>8,} chunks, {train_tokens:>10,} tokens")
    print(f"  Holdout: {len(holdout_chunks):>8,} chunks, {holdout_tokens:>10,} tokens")
    pct = holdout_tokens / total_tokens * 100 if total_tokens > 0 else 0
    print(f"  Holdout is {pct:.1f}% of corpus")

    # Per-author holdout breakdown
    holdout_by_source = {}
    for c in holdout_chunks:
        src = c['source']
        if src not in holdout_by_source:
            holdout_by_source[src] = {'chunks': 0, 'tokens': 0, 'category': c['category']}
        holdout_by_source[src]['chunks'] += 1
        holdout_by_source[src]['tokens'] += c['token_count']

    print(f"\n  Holdout authors:")
    for src in sorted(holdout_by_source.keys()):
        info = holdout_by_source[src]
        print(f"    {src:<45} {info['category']:<12} "
              f"{info['chunks']:>6} chunks, {info['tokens']:>8} tokens")

    return category_stats


if __name__ == '__main__':
    build_corpus()
