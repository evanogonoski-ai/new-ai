#!/usr/bin/env python3
"""
Download WikiText-103 dataset and split into individual article files.

WikiText-103 is a large-scale language modeling dataset containing ~100M tokens
of clean Wikipedia text. This script downloads the zip archive, extracts it,
splits the monolithic files into individual articles, cleans tokenization
artifacts, and saves to data/raw_texts/wikitext103/.

Source: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
"""

import os
import re
import sys
import time
import zipfile
import urllib.request
import urllib.error

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw_texts', 'wikitext103')
ZIP_PATH = os.path.join(BASE_DIR, 'data', 'wikitext-103-v1.zip')
EXTRACT_DIR = os.path.join(BASE_DIR, 'data', 'wikitext-103-extracted')

URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'

MIN_ARTICLE_WORDS = 200


def download_with_progress(url, dest_path, retries=3):
    """Download a large file with progress reporting and retry logic."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000_000:
        print(f"  [cached] {dest_path} ({os.path.getsize(dest_path) / 1e6:.0f}MB)")
        return True

    for attempt in range(retries):
        try:
            print(f"  Downloading {url}")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Research Project)',
            })
            resp = urllib.request.urlopen(req, timeout=120)
            total = int(resp.headers.get('Content-Length', 0))

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            downloaded = 0
            last_pct = -1

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct >= last_pct + 10:
                            last_pct = pct
                            print(f"    {pct}% ({downloaded / 1e6:.1f}MB / {total / 1e6:.1f}MB)")

            print(f"  Download complete: {downloaded / 1e6:.1f}MB")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry in {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False
    return False


def extract_zip(zip_path, extract_dir):
    """Extract the WikiText-103 zip file."""
    print(f"  Extracting {zip_path}...")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # Find the token files (they're in a subdirectory)
    token_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.tokens'):
                token_files.append(os.path.join(root, f))

    print(f"  Found {len(token_files)} token files")
    return sorted(token_files)


def clean_article(text):
    """Clean WikiText-103 tokenization artifacts from article text.

    WikiText-103 is pre-tokenized with specific artifacts:
    - @-@ for hyphens (e.g., "well @-@ known" → "well-known")
    - @.@ for decimal points (e.g., "3 @.@ 14" → "3.14")
    - @,@ for commas in numbers (e.g., "1 @,@ 000" → "1,000")
    - Extra spaces before punctuation
    - Spaces before possessives/contractions
    """
    # Fix tokenization artifacts
    text = text.replace(' @-@ ', '-')
    text = text.replace(' @.@ ', '.')
    text = text.replace(' @,@ ', ',')

    # Remove sub-section markup but keep section title text
    # = = Section = =  and  = = = Subsection = = =
    text = re.sub(r' = = = (.+?) = = = ', r'\1', text)
    text = re.sub(r' = = (.+?) = = ', r'\1', text)

    # Fix spacing before punctuation
    text = re.sub(r" (\.|,|;|:|\?|!|'s|'t|'re|'ve|'ll|'d|'m|n't)", r'\1', text)

    # Fix spacing around quotes and brackets
    text = re.sub(r'" (.+?) "', r'"\1"', text)
    text = re.sub(r'\( (.+?) \)', r'(\1)', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_into_articles(token_file_path, output_dir):
    """Split a WikiText-103 token file into individual article files.

    Articles are delimited by lines matching: ' = Article Title = \\n'
    (single = with leading/trailing space, at the start of a line).
    """
    with open(token_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on top-level article headers: lines like " = Title = \n"
    # These have exactly one = on each side (not two or three)
    parts = re.split(r'\n = ([^=\n]+?) = \n', content)

    # parts[0] is text before first article (usually empty/whitespace)
    # parts[1] = title1, parts[2] = body1, parts[3] = title2, parts[4] = body2, ...

    articles_saved = 0
    articles_skipped = 0
    total_words = 0

    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ''

        cleaned = clean_article(body)
        word_count = len(cleaned.split())

        if word_count < MIN_ARTICLE_WORDS:
            articles_skipped += 1
            continue

        # Generate safe filename
        safe_name = re.sub(r'[^a-z0-9_]', '_', title.lower().strip())
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        if len(safe_name) > 80:
            safe_name = safe_name[:80]
        filename = f"wt103_{safe_name}.txt"
        filepath = os.path.join(output_dir, filename)

        # Handle duplicate titles by appending a counter
        if os.path.exists(filepath):
            base = filename[:-4]
            counter = 2
            while os.path.exists(os.path.join(output_dir, f"{base}_{counter}.txt")):
                counter += 1
            filepath = os.path.join(output_dir, f"{base}_{counter}.txt")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        articles_saved += 1
        total_words += word_count

    return articles_saved, articles_skipped, total_words


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if already populated
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')] if os.path.exists(OUTPUT_DIR) else []
    if len(existing) > 1000:
        total_words = 0
        for f in existing[:100]:  # Sample 100 files for word estimate
            with open(os.path.join(OUTPUT_DIR, f), 'r') as fh:
                total_words += len(fh.read().split())
        est_total = total_words * len(existing) // 100
        print(f"  [cached] {len(existing)} articles already exist (~{est_total:,} words estimated)")
        print(f"  Delete {OUTPUT_DIR} to re-download")
        return

    # Step 1: Download
    print("Step 1: Downloading WikiText-103...")
    if not download_with_progress(URL, ZIP_PATH):
        print("\nERROR: Download failed.")
        print("Alternative: try downloading manually from Hugging Face:")
        print("  pip install datasets")
        print("  python -c \"from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1')\"")
        sys.exit(1)

    # Step 2: Extract
    print("\nStep 2: Extracting zip archive...")
    token_files = extract_zip(ZIP_PATH, EXTRACT_DIR)

    if not token_files:
        print("ERROR: No .tokens files found in archive")
        sys.exit(1)

    # Step 3: Split into articles
    print(f"\nStep 3: Splitting into individual articles...")
    grand_saved = 0
    grand_skipped = 0
    grand_words = 0

    for tf in token_files:
        basename = os.path.basename(tf)
        print(f"\n  Processing {basename}...")
        saved, skipped, words = split_into_articles(tf, OUTPUT_DIR)
        grand_saved += saved
        grand_skipped += skipped
        grand_words += words
        print(f"    Saved: {saved}, Skipped (too short): {skipped}, Words: {words:,}")

    # Step 4: Cleanup extracted files (keep the zip)
    print(f"\nStep 4: Cleaning up extracted files...")
    import shutil
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"WIKITEXT-103 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Articles saved:   {grand_saved:,}")
    print(f"  Articles skipped: {grand_skipped:,} (< {MIN_ARTICLE_WORDS} words)")
    print(f"  Total words:      {grand_words:,}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  Zip cached at:    {ZIP_PATH}")


if __name__ == '__main__':
    main()
