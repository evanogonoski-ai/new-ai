#!/usr/bin/env python3
"""
Build the Fair Fight corpus from all available text sources.
Outputs: fair_train_corpus.jsonl, fair_holdout_corpus.jsonl, corpus_stats.txt
Then trains BPE tokenizer and pre-tokenizes both splits.

Sources:
  1. WikiText-103 (from wikitext103/ directory — user provides)
  2. Gutenberg (gutenberg/ + gutenberg_expanded/)
  3. Synthetic (synthetic/ + synthetic_scaled/ + additional_synthetic/ + wikipedia_synthetic/)
  4. Religious (quran/)

Holdout: Plato, Austen, Darwin, Dostoevsky (entire authors) + 10% WikiText articles.
"""

import os
import sys
import json
import random
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, 'data', 'raw_texts')
CORPUS_RAW_DIR = os.path.join(ROOT, 'corpus', 'raw')
WIKITEXT_DIR = os.path.join(ROOT, 'corpus', 'wikitext103')
DATA_DIR = os.path.join(ROOT, 'data')
TOKENIZER_DIR = os.path.join(ROOT, 'tokenizer')

TRAIN_JSONL = os.path.join(DATA_DIR, 'fair_train_corpus.jsonl')
HOLDOUT_JSONL = os.path.join(DATA_DIR, 'fair_holdout_corpus.jsonl')
TRAIN_PT = os.path.join(DATA_DIR, 'fair_train_tokenized.pt')
HOLDOUT_PT = os.path.join(DATA_DIR, 'fair_holdout_tokenized.pt')
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, 'fair_fight_tokenizer.json')
STATS_PATH = os.path.join(DATA_DIR, 'corpus_stats.txt')

# Authors to hold out entirely
HOLDOUT_AUTHOR_KEYWORDS = [
    'plato', 'austen', 'darwin', 'dostoevsky',
]

WIKI_HOLDOUT_FRACTION = 0.10  # Hold out 10% of WikiText articles


def is_holdout_author(filename):
    """Check if filename belongs to a held-out author."""
    fn_lower = filename.lower()
    for kw in HOLDOUT_AUTHOR_KEYWORDS:
        if kw in fn_lower:
            return True
    return False


def source_from_filename(filename):
    """Extract source/author tag from filename for per-author eval."""
    fn = os.path.splitext(os.path.basename(filename))[0].lower()
    # Gutenberg files
    for kw in HOLDOUT_AUTHOR_KEYWORDS:
        if kw in fn:
            return fn  # Full filename as source
    if 'gutenberg' in fn or fn.startswith('philosophy_') or fn.startswith('fiction_') or fn.startswith('science_'):
        return fn
    if 'wiki' in fn:
        return 'wikipedia'
    if 'synthetic' in fn:
        return 'synthetic'
    if 'quran' in fn or 'bible' in fn or 'religious' in fn:
        return 'quran'
    return fn


def category_from_path(filepath):
    """Determine category from file path."""
    fp = filepath.lower()
    if 'quran' in fp or 'bible' in fp or 'religious' in fp:
        return 'quran'
    if 'wiki' in fp:
        return 'wikipedia'
    if 'synthetic' in fp:
        return 'synthetic'
    if 'philosophy' in fp or 'plato' in fp or 'aristotle' in fp or 'nietzsche' in fp:
        return 'philosophy'
    if 'fiction' in fp or 'austen' in fp or 'dostoevsky' in fp or 'dickens' in fp or 'twain' in fp:
        return 'fiction'
    if 'science' in fp or 'darwin' in fp or 'einstein' in fp or 'newton' in fp:
        return 'science'
    # Default
    if 'gutenberg' in fp:
        return 'fiction'
    return 'synthetic'


def load_text_files(directory, min_chars=100):
    """Load all .txt files from a directory."""
    entries = []
    if not os.path.exists(directory):
        return entries
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith('.txt'):
            continue
        fp = os.path.join(directory, fn)
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read().strip()
        except Exception:
            continue
        if len(text) < min_chars:
            continue
        entries.append({
            'text': text,
            'filename': fn,
            'filepath': fp,
            'category': category_from_path(fp),
            'source': source_from_filename(fn),
        })
    return entries


def load_wikitext103_jsonl(directory):
    """Load WikiText-103 from pre-processed JSONL files."""
    entries = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith('.jsonl'):
            continue
        fp = os.path.join(directory, fn)
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry.setdefault('category', 'wikipedia')
                entry.setdefault('source', entry.get('author', 'wikipedia'))
                entry.setdefault('filename', fn)
                entries.append(entry)
    return entries


def load_wikitext103_from_chunks(directory):
    """Load WikiText-103 from monolithic chunk files (train_chunk_*.txt).

    Parses article boundaries using ' = Title = ' headers.
    Returns list of entry dicts, one per article.
    """
    import re

    # First check for concatenated file, else concatenate chunks
    concat_path = os.path.join(directory, 'wikitext103_train.txt')
    if not os.path.exists(concat_path):
        chunk_files = sorted(glob.glob(os.path.join(directory, 'train_chunk_*.txt')))
        if not chunk_files:
            return []
        print(f"  Concatenating {len(chunk_files)} chunk files...")
        with open(concat_path, 'w', encoding='utf-8') as out:
            for cf in chunk_files:
                with open(cf, 'r', encoding='utf-8') as inp:
                    out.write(inp.read())

    print(f"  Reading {concat_path}...")
    with open(concat_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ensure content starts with newline for uniform splitting
    if not content.startswith('\n'):
        content = '\n' + content
    # Split on article headers: lines like "= Title =" (no leading space)
    parts = re.split(r'\n= ([^=\n]+?) =\n', content)

    entries = []
    # parts[0] = preamble, then alternating title/body pairs
    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ''

        # Clean sub-section markup (lines like "= = Section = =" or "= = = Sub = = =")
        body = re.sub(r'^= = = (.+?) = = =\s*$', r'\1.', body, flags=re.MULTILINE)
        body = re.sub(r'^= = (.+?) = =\s*$', r'\1.', body, flags=re.MULTILINE)

        # Clean WikiText tokenization artifacts
        body = body.replace(' @-@ ', '-')
        body = body.replace(' @.@ ', '.')
        body = body.replace(' @,@ ', ',')
        body = re.sub(r" (\.|,|;|:|\?|!|'s|'t|'re|'ve|'ll|'d|'m|n't)", r'\1', body)

        # Normalize whitespace
        lines = [l.strip() for l in body.split('\n') if l.strip()]
        body = ' '.join(lines)

        if len(body.split()) < 100:
            continue

        safe_name = re.sub(r'[^a-z0-9_]', '_', title.lower())[:60]
        entries.append({
            'text': body,
            'filename': f'wt103_{safe_name}.txt',
            'filepath': concat_path,
            'category': 'wikipedia',
            'source': f'wt103_{safe_name}',
        })

    return entries


def load_wikitext103_txt(directory):
    """Load WikiText-103 from raw text files."""
    return load_text_files(directory)


def build_corpus():
    """Build the full fair fight corpus."""
    print("=" * 70)
    print("BUILDING FAIR FIGHT CORPUS")
    print("=" * 70)

    all_entries = []

    # 1. WikiText-103
    # Check multiple possible locations
    wiki_entries = []
    if os.path.exists(WIKITEXT_DIR):
        # Primary: corpus/wikitext103/ with chunk files
        chunk_files = glob.glob(os.path.join(WIKITEXT_DIR, 'train_chunk_*.txt'))
        concat_file = os.path.join(WIKITEXT_DIR, 'wikitext103_train.txt')
        if chunk_files or os.path.exists(concat_file):
            wiki_entries = load_wikitext103_from_chunks(WIKITEXT_DIR)
            print(f"WikiText-103 (chunks): {len(wiki_entries)} articles")
        else:
            wiki_entries = load_wikitext103_txt(WIKITEXT_DIR)
            print(f"WikiText-103 (TXT): {len(wiki_entries)} articles")
    elif os.path.exists(os.path.join(RAW_DIR, 'wikitext103')):
        # Fallback: data/raw_texts/wikitext103/
        wiki_dir = os.path.join(RAW_DIR, 'wikitext103')
        jsonl_files = glob.glob(os.path.join(wiki_dir, '*.jsonl'))
        if jsonl_files:
            wiki_entries = load_wikitext103_jsonl(wiki_dir)
            print(f"WikiText-103 (JSONL): {len(wiki_entries)} articles")
        else:
            wiki_entries = load_wikitext103_txt(wiki_dir)
            print(f"WikiText-103 (TXT): {len(wiki_entries)} articles")
    else:
        print("WARNING: WikiText-103 not found!")
        print("  Expected at: corpus/wikitext103/ or data/raw_texts/wikitext103/")

    # Split wiki: 90% train, 10% holdout
    if wiki_entries:
        random.seed(42)
        random.shuffle(wiki_entries)
        n_holdout = max(1, int(len(wiki_entries) * WIKI_HOLDOUT_FRACTION))
        wiki_holdout = wiki_entries[:n_holdout]
        wiki_train = wiki_entries[n_holdout:]
        print(f"  Wiki train: {len(wiki_train)}, Wiki holdout: {n_holdout}")
    else:
        wiki_train, wiki_holdout = [], []

    # 2. Gutenberg / Literature
    gutenberg_entries = []
    for d in ['gutenberg', 'gutenberg_expanded']:
        gutenberg_entries.extend(load_text_files(os.path.join(RAW_DIR, d)))
    # Also check corpus/raw/ directories (fiction, philosophy, science)
    for subdir in ['fiction', 'philosophy', 'science']:
        corpus_dir = os.path.join(CORPUS_RAW_DIR, subdir)
        if os.path.isdir(corpus_dir):
            gutenberg_entries.extend(load_text_files(corpus_dir))
    print(f"Gutenberg/Literature: {len(gutenberg_entries)} files")

    # 3. Synthetic
    synthetic_entries = []
    for d in ['synthetic', 'synthetic_scaled', 'additional_synthetic', 'wikipedia_synthetic']:
        synthetic_entries.extend(load_text_files(os.path.join(RAW_DIR, d)))
    # Also check corpus/raw/synthetic
    corpus_syn = os.path.join(CORPUS_RAW_DIR, 'synthetic')
    if os.path.isdir(corpus_syn):
        synthetic_entries.extend(load_text_files(corpus_syn))
    print(f"Synthetic: {len(synthetic_entries)} files")

    # 4. Religious
    religious_entries = load_text_files(os.path.join(RAW_DIR, 'quran'))
    # Also check corpus/raw/quran
    corpus_quran = os.path.join(CORPUS_RAW_DIR, 'quran')
    if os.path.isdir(corpus_quran):
        religious_entries.extend(load_text_files(corpus_quran))
    print(f"Religious: {len(religious_entries)} files")

    # Wikipedia hand-written (small set)
    wiki_small = load_text_files(os.path.join(RAW_DIR, 'wikipedia'))
    print(f"Wikipedia (hand-written): {len(wiki_small)} files")

    # Combine non-wiki entries
    non_wiki = gutenberg_entries + synthetic_entries + religious_entries + wiki_small

    # Split non-wiki: holdout by author
    train_entries = []
    holdout_entries = list(wiki_holdout)  # Start with wiki holdout

    for entry in non_wiki:
        if is_holdout_author(entry['filename']):
            holdout_entries.append(entry)
        else:
            train_entries.append(entry)

    # Add wiki train
    train_entries.extend(wiki_train)

    # Shuffle
    random.seed(42)
    random.shuffle(train_entries)
    random.shuffle(holdout_entries)

    # Stats
    train_chars = sum(len(e['text']) for e in train_entries)
    holdout_chars = sum(len(e['text']) for e in holdout_entries)

    print(f"\n{'='*50}")
    print(f"Train:   {len(train_entries):>8,} entries, ~{train_chars//4/1e6:.1f}M tokens")
    print(f"Holdout: {len(holdout_entries):>8,} entries, ~{holdout_chars//4/1e6:.1f}M tokens")
    print(f"Total:   {len(train_entries)+len(holdout_entries):>8,} entries, ~{(train_chars+holdout_chars)//4/1e6:.1f}M tokens")
    print(f"{'='*50}")

    # Count by category
    cat_counts = {}
    for entry in train_entries + holdout_entries:
        cat = entry['category']
        cat_counts[cat] = cat_counts.get(cat, 0) + len(entry['text'])
    print("\nBy category (chars):")
    for cat, chars in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<15}: {chars//4/1e6:.1f}M tokens ({100*chars/(train_chars+holdout_chars):.1f}%)")

    # Write JSONL files
    print(f"\nWriting {TRAIN_JSONL}...")
    with open(TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            json.dump({
                'text': entry['text'],
                'category': entry['category'],
                'source': entry['source'],
            }, f, ensure_ascii=False)
            f.write('\n')

    print(f"Writing {HOLDOUT_JSONL}...")
    with open(HOLDOUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in holdout_entries:
            json.dump({
                'text': entry['text'],
                'category': entry['category'],
                'source': entry['source'],
            }, f, ensure_ascii=False)
            f.write('\n')

    train_mb = os.path.getsize(TRAIN_JSONL) / 1024 / 1024
    holdout_mb = os.path.getsize(HOLDOUT_JSONL) / 1024 / 1024
    print(f"  Train JSONL:   {train_mb:.1f} MB")
    print(f"  Holdout JSONL: {holdout_mb:.1f} MB")

    # Write stats
    with open(STATS_PATH, 'w') as f:
        f.write("Fair Fight Corpus Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Train entries: {len(train_entries):,}\n")
        f.write(f"Holdout entries: {len(holdout_entries):,}\n")
        f.write(f"Train tokens (est): ~{train_chars//4/1e6:.1f}M\n")
        f.write(f"Holdout tokens (est): ~{holdout_chars//4/1e6:.1f}M\n")
        f.write(f"\nBy category:\n")
        for cat, chars in sorted(cat_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {cat:<15}: ~{chars//4/1e6:.1f}M tokens\n")
        f.write(f"\nHeldout authors: {', '.join(HOLDOUT_AUTHOR_KEYWORDS)}\n")
        f.write(f"WikiText holdout: {WIKI_HOLDOUT_FRACTION*100:.0f}% of articles\n")

    return train_entries, holdout_entries


def train_tokenizer(train_jsonl, vocab_size=8192):
    """Train BPE tokenizer on training corpus."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print(f"\nTraining BPE tokenizer (vocab={vocab_size})...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )

    # Feed from JSONL
    def text_iterator():
        with open(train_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)['text']

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"  Saved tokenizer: {TOKENIZER_PATH} (vocab={tokenizer.get_vocab_size()})")
    return tokenizer


def pretokenize(tokenizer, jsonl_path, pt_path, seq_len=256):
    """Pre-tokenize JSONL corpus into .pt file."""
    from neurogenesis_v2.training.category_dataset import CATEGORY_MAP, pretokenize_corpus

    # Ensure wikipedia category exists
    if 'wikipedia' not in CATEGORY_MAP:
        CATEGORY_MAP['wikipedia'] = 5

    print(f"\nPre-tokenizing {jsonl_path} → {pt_path}...")
    pretokenize_corpus(jsonl_path, tokenizer, seq_len=seq_len, save_path=pt_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build Fair Fight corpus')
    parser.add_argument('--skip-corpus', action='store_true', help='Skip corpus assembly (use existing JSONL)')
    parser.add_argument('--skip-tokenizer', action='store_true', help='Skip tokenizer training')
    parser.add_argument('--skip-pretokenize', action='store_true', help='Skip pre-tokenization')
    parser.add_argument('--vocab-size', type=int, default=8192)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    # Step 1: Build corpus JSONL
    if not args.skip_corpus:
        build_corpus()
    else:
        print("Skipping corpus assembly (using existing JSONL files)")

    # Step 2: Train tokenizer
    if not args.skip_tokenizer:
        if not os.path.exists(TRAIN_JSONL):
            print(f"ERROR: {TRAIN_JSONL} not found. Run corpus build first.")
            sys.exit(1)
        tokenizer = train_tokenizer(TRAIN_JSONL, args.vocab_size)
    else:
        print("Skipping tokenizer training")

    # Step 3: Pre-tokenize
    if not args.skip_pretokenize:
        if not os.path.exists(TOKENIZER_PATH):
            print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
            sys.exit(1)

        from neurogenesis.tokenizer.bpe import load_tokenizer
        tokenizer = load_tokenizer(TOKENIZER_PATH)

        for jsonl, pt, label in [
            (TRAIN_JSONL, TRAIN_PT, 'train'),
            (HOLDOUT_JSONL, HOLDOUT_PT, 'holdout'),
        ]:
            if os.path.exists(jsonl):
                pretokenize(tokenizer, jsonl, pt, args.seq_len)
            else:
                print(f"WARNING: {jsonl} not found, skipping {label}")

    print("\nDone! Corpus is ready for training.")
    print(f"  Train:    {TRAIN_PT}")
    print(f"  Holdout:  {HOLDOUT_PT}")
    print(f"  Tokenizer: {TOKENIZER_PATH}")


if __name__ == '__main__':
    main()
