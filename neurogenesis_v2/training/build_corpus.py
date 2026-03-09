#!/usr/bin/env python3
"""Chunk all raw texts with category tags and save as JSONL.

Reads from corpus/raw/{category}/, chunks on sentence boundaries
at ~128 tokens with 10-20 token overlap, tags with metadata,
saves as data/expanded_corpus.jsonl.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neurogenesis.tokenizer.bpe import load_tokenizer

CORPUS_DIR = 'corpus/raw'
OUTPUT_PATH = 'data/expanded_corpus.jsonl'
TOKENIZER_PATH = 'tokenizer/expanded_tokenizer.json'
TARGET_CHUNK_TOKENS = 128
OVERLAP_TOKENS = 15

CATEGORIES = ['philosophy', 'fiction', 'science', 'synthetic', 'quran']


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


def build_corpus():
    """Read all raw texts, chunk, tag, and save as JSONL."""
    print("=" * 60)
    print("BUILDING EXPANDED CORPUS")
    print("=" * 60)

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        print("Run train_expanded_tokenizer.py first.")
        sys.exit(1)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"Loaded tokenizer: vocab={tokenizer.get_vocab_size()}")

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
                    }
                    all_chunks.append(entry)
                    cat_tokens += token_count
                file_chunks += len(chunks)

            cat_chunks += file_chunks

        category_stats[category] = {'chunks': cat_chunks, 'tokens': cat_tokens}
        print(f"  {category}: {cat_chunks:,} chunks, {cat_tokens:,} tokens")

    # Shuffle chunks
    import random
    random.seed(42)
    random.shuffle(all_chunks)

    # Save as JSONL
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for entry in all_chunks:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Summary
    total_chunks = len(all_chunks)
    total_tokens = sum(s['tokens'] for s in category_stats.values())

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
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")

    return category_stats


if __name__ == '__main__':
    build_corpus()
