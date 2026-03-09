#!/usr/bin/env python3
"""Download public domain texts from GITenberg GitHub mirrors for corpus expansion.

Project Gutenberg (gutenberg.org) is firewalled. GITenberg hosts individual
GitHub repos for each book at github.com/GITenberg with raw text access via
raw.githubusercontent.com.
"""
import os
import re
import time
import urllib.request
import urllib.error

# GITenberg repos: (gutenberg_id, repo_name, file_variant, author, title, category)
# file_variant: None = {id}.txt, "-0" = {id}-0.txt, "-8" = {id}-8.txt
GITENBERG_TEXTS = [
    # Philosophy (~25%)
    (3794, "L.-Annaeus-Seneca-on-Benefits_3794", None,
     "seneca", "on_benefits", "philosophy"),
    (56075, "Seneca-s-Morals-of-a-Happy-Life-Benefits-Anger-and-Clemency_56075", "-0",
     "seneca", "morals_happy_life", "philosophy"),
    (2680, "Meditations_2680", None,
     "marcus_aurelius", "meditations", "philosophy"),
    (45109, "The-Enchiridion_45109", None,
     "epictetus", "enchiridion", "philosophy"),
    (1497, "The-Republic_1497", None,
     "plato", "republic", "philosophy"),
    (1600, "Symposium_1600", None,
     "plato", "symposium", "philosophy"),
    (1656, "Apology_1656", None,
     "plato", "apology", "philosophy"),
    (8438, "The-Ethics-of-Aristotle_8438", "-8",
     "aristotle", "nicomachean_ethics", "philosophy"),
    (34901, "On-Liberty_34901", None,
     "john_stuart_mill", "on_liberty", "philosophy"),
    (9662, "An-Enquiry-Concerning-Human-Understanding_9662", None,
     "david_hume", "enquiry_concerning_human_understanding", "philosophy"),
    (216, "The-Tao-Teh-King-or-the-Tao-and-its-Characteristics_216", None,
     "lao_tzu", "tao_te_ching", "philosophy"),

    # Fiction (~25%)
    (1342, "Pride-and-Prejudice_1342", None,
     "jane_austen", "pride_and_prejudice", "fiction"),
    (844, "The-Importance-of-Being-Earnest--A-Trivial-Comedy-for-Serious-People_844", None,
     "oscar_wilde", "importance_of_being_earnest", "fiction"),
    (2701, "Moby-Dick--Or-The-Whale_2701", None,
     "herman_melville", "moby_dick", "fiction"),
    (215, "The-Call-of-the-Wild_215", None,
     "jack_london", "call_of_the_wild", "fiction"),
    (76, "Adventures-of-Huckleberry-Finn_76", None,
     "mark_twain", "adventures_of_huckleberry_finn", "fiction"),
    (2147, "The-Works-of-Edgar-Allan-Poe---Volume-1_2147", "-0",
     "edgar_allan_poe", "collected_tales", "fiction"),
    (2776, "The-Four-Million_2776", None,
     "o_henry", "collected_stories", "fiction"),

    # Science (~15%)
    (1228, "On-the-Origin-of-Species-By-Means-of-Natural-Selection--13-Or-the-Preservation-of-Favoured-Rac__1228", None,
     "charles_darwin", "origin_of_species", "science"),
    (5116, "Pragmatism--A-New-Name-for-Some-Old-Ways-of-Thinking_5116", None,
     "william_james", "pragmatism", "science"),
    (14474, "The-Chemical-History-of-a-Candle_14474", None,
     "michael_faraday", "chemical_history_of_a_candle", "science"),
    (5001, "Relativity-the-Special-and-General-Theory_5001", None,
     "albert_einstein", "relativity", "science"),
]

GITENBERG_URL = "https://raw.githubusercontent.com/GITenberg/{repo}/master/{filename}"


def build_url(repo_name, gutenberg_id, file_variant):
    """Build the raw content URL for a GITenberg text."""
    if file_variant:
        filename = f"{gutenberg_id}{file_variant}.txt"
    else:
        filename = f"{gutenberg_id}.txt"
    return GITENBERG_URL.format(repo=repo_name, filename=filename)


def strip_gutenberg_boilerplate(text):
    """Remove Project Gutenberg headers, footers, and license text."""
    # Find start of actual content
    start_markers = [
        r'\*\*\*\s*START OF (THE |THIS )?PROJECT GUTENBERG',
        r'\*\*\*\s*START OF THIS PROJECT GUTENBERG',
        r'CHAPTER I',
    ]
    start_idx = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            next_newline = text.find('\n', match.end())
            if next_newline != -1:
                start_idx = next_newline + 1
            break

    # Find end of actual content
    end_markers = [
        r'\*\*\*\s*END OF (THE |THIS )?PROJECT GUTENBERG',
        r'End of (the )?Project Gutenberg',
        r'END OF THIS PROJECT GUTENBERG',
    ]
    end_idx = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_idx = match.start()
            break

    text = text[start_idx:end_idx]

    # Remove table of contents patterns
    text = re.sub(r'(?i)^CONTENTS?\s*$.*?(?=\n[A-Z])', '', text, flags=re.MULTILINE | re.DOTALL)

    # Remove footnote markers like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Normalize whitespace: collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    text = text.strip()
    return text


def download_text(url, max_retries=3):
    """Download a text from a GITenberg URL."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Research Project)',
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    text = raw.decode('latin-1')
                return text
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry in {wait}s ({e})")
                time.sleep(wait)
            else:
                return None
    return None


def download_all(output_dir='corpus/raw'):
    """Download all texts, strip boilerplate, save to category directories."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for gid, repo_name, file_variant, author, title, category in GITENBERG_TEXTS:
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        filename = f"{author}_{title}.txt"
        filepath = os.path.join(cat_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            word_count = len(open(filepath).read().split())
            print(f"  [cached] {category}/{filename}: {word_count:,} words")
            results[filename] = {'category': category, 'words': word_count, 'status': 'cached'}
            continue

        url = build_url(repo_name, gid, file_variant)
        print(f"  Downloading: {author} - {title} (ID: {gid})...", end=' ')
        raw = download_text(url)

        if raw is None:
            print("FAILED")
            results[filename] = {'category': category, 'words': 0, 'status': 'failed'}
            continue

        cleaned = strip_gutenberg_boilerplate(raw)
        word_count = len(cleaned.split())

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"{word_count:,} words")
        results[filename] = {'category': category, 'words': word_count, 'status': 'ok'}

        # Be polite to GitHub servers
        time.sleep(0.5)

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for cat in ['philosophy', 'fiction', 'science']:
        cat_results = {k: v for k, v in results.items() if v['category'] == cat}
        total_words = sum(v['words'] for v in cat_results.values())
        ok_count = sum(1 for v in cat_results.values() if v['status'] in ('ok', 'cached'))
        print(f"  {cat}: {ok_count}/{len(cat_results)} texts, {total_words:,} words")

    total = sum(v['words'] for v in results.values())
    print(f"  TOTAL: {total:,} words")
    return results


if __name__ == '__main__':
    download_all()
