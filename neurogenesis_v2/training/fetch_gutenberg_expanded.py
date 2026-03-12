#!/usr/bin/env python3
"""Download additional Project Gutenberg books beyond the original 22 for a larger corpus.

Uses GITenberg GitHub mirrors (raw.githubusercontent.com/GITenberg) as the
primary source, with fallback to direct Gutenberg cache URLs.
"""
import os
import re
import time
import urllib.request
import urllib.error

# (author_slug, title, gitenberg_repo, filename)
# gitenberg_repo is the repo name under github.com/GITenberg
# filename is the text file inside that repo
EXPANDED_BOOKS = [
    # Charles Dickens
    ("charles_dickens", "great_expectations",
     "Great-Expectations_1400", "1400-0.txt"),
    ("charles_dickens", "a_tale_of_two_cities",
     "A-Tale-of-Two-Cities_98", "98-0.txt"),
    ("charles_dickens", "oliver_twist",
     "Oliver-Twist_730", "730-0.txt"),

    # Fyodor Dostoevsky
    ("fyodor_dostoevsky", "crime_and_punishment",
     "Crime-and-Punishment_2554", "2554-0.txt"),
    ("fyodor_dostoevsky", "the_brothers_karamazov",
     "The-Brothers-Karamazov_28054", "28054-0.txt"),

    # Leo Tolstoy
    ("leo_tolstoy", "war_and_peace",
     "War-and-Peace_2600", "2600-0.txt"),
    ("leo_tolstoy", "anna_karenina",
     "Anna-Karenina_1399", "1399-0.txt"),

    # Arthur Conan Doyle
    ("arthur_conan_doyle", "adventures_of_sherlock_holmes",
     "The-Adventures-of-Sherlock-Holmes_1661", "1661-0.txt"),
    ("arthur_conan_doyle", "hound_of_the_baskervilles",
     "The-Hound-of-the-Baskervilles_2852", "2852-0.txt"),
    ("arthur_conan_doyle", "a_study_in_scarlet",
     "A-Study-in-Scarlet_244", "244-0.txt"),
    ("arthur_conan_doyle", "the_sign_of_the_four",
     "The-Sign-of-the-Four_2097", "2097-0.txt"),

    # Bible (King James Version)
    ("bible", "king_james_version",
     "The-King-James-Version-of-the-Bible_10", "10-0.txt"),

    # Homer
    ("homer", "the_odyssey",
     "The-Odyssey_1727", "1727-0.txt"),
    ("homer", "the_iliad",
     "The-Iliad_6130", "6130-0.txt"),

    # Bram Stoker
    ("bram_stoker", "dracula",
     "Dracula_345", "345-0.txt"),

    # H.G. Wells
    ("hg_wells", "the_time_machine",
     "The-Time-Machine_35", "35-0.txt"),
    ("hg_wells", "war_of_the_worlds",
     "The-War-of-the-Worlds_36", "36-0.txt"),

    # Mark Twain
    ("mark_twain", "adventures_of_huckleberry_finn",
     "Adventures-of-Huckleberry-Finn_76", "76-0.txt"),
    ("mark_twain", "the_adventures_of_tom_sawyer",
     "The-Adventures-of-Tom-Sawyer_74", "74-0.txt"),

    # Victor Hugo
    ("victor_hugo", "les_miserables",
     "Les-Mis-rables_135", "135-0.txt"),

    # Charlotte Bronte
    ("charlotte_bronte", "jane_eyre",
     "Jane-Eyre--An-Autobiography_1260", "1260-0.txt"),

    # Emily Bronte
    ("emily_bronte", "wuthering_heights",
     "Wuthering-Heights_768", "768-0.txt"),

    # Louisa May Alcott
    ("louisa_may_alcott", "little_women",
     "Little-Women_514", "514-0.txt"),

    # Robert Louis Stevenson
    ("robert_louis_stevenson", "treasure_island",
     "Treasure-Island_120", "120-0.txt"),

    # F. Scott Fitzgerald
    ("f_scott_fitzgerald", "this_side_of_paradise",
     "This-Side-of-Paradise_805", "805-0.txt"),

    # Joseph Conrad
    ("joseph_conrad", "heart_of_darkness",
     "Heart-of-Darkness_219", "219-0.txt"),

    # James Joyce
    ("james_joyce", "dubliners",
     "Dubliners_2814", "2814-0.txt"),

    # Oscar Wilde
    ("oscar_wilde", "the_picture_of_dorian_gray",
     "The-Picture-of-Dorian-Gray_174", "174-0.txt"),

    # Nathaniel Hawthorne
    ("nathaniel_hawthorne", "the_scarlet_letter",
     "The-Scarlet-Letter_25344", "25344-0.txt"),

    # Voltaire
    ("voltaire", "candide",
     "Candide_19942", "19942-0.txt"),

    # Walt Whitman
    ("walt_whitman", "leaves_of_grass",
     "Leaves-of-Grass_1322", "1322-0.txt"),

    # Henry David Thoreau
    ("henry_david_thoreau", "walden",
     "Walden-and-On-The-Duty-Of-Civil-Disobedience_205", "205-0.txt"),

    # Frederick Douglass
    ("frederick_douglass", "narrative_of_the_life",
     "Narrative-of-the-Life-of-Frederick-Douglass-an-American-Slave_23", "23-0.txt"),

    # Adam Smith
    ("adam_smith", "wealth_of_nations",
     "An-Inquiry-into-the-Nature-and-Causes-of-the-Wealth-of-Nations_3300", "3300-0.txt"),
]

GITENBERG_URL = "https://raw.githubusercontent.com/GITenberg/{repo}/master/{filename}"
GUTENBERG_CACHE_URL = "https://www.gutenberg.org/cache/epub/{book_id}/{book_id}-0.txt"

OUTPUT_DIR = os.path.join("data", "raw_texts", "gutenberg_expanded")


def _extract_book_id(filename):
    """Extract the numeric Gutenberg ID from a filename like '1400-0.txt'."""
    match = re.match(r"(\d+)", filename)
    if match:
        return match.group(1)
    return None


def _build_gitenberg_url(repo, filename):
    """Build the raw GitHub URL for a GITenberg text."""
    return GITENBERG_URL.format(repo=repo, filename=filename)


def _build_gutenberg_fallback_url(filename):
    """Build the direct Gutenberg cache URL as a fallback."""
    book_id = _extract_book_id(filename)
    if book_id:
        return GUTENBERG_CACHE_URL.format(book_id=book_id)
    return None


def download_text(url, max_retries=3):
    """Download text from a URL with retries."""
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


def strip_gutenberg_boilerplate(text):
    """Remove Project Gutenberg header and footer markers and surrounding text."""
    # Find start of actual content (after *** START ... ***)
    start_markers = [
        r'\*\*\*\s*START OF (THE |THIS )?PROJECT GUTENBERG',
        r'\*\*\*\s*START OF THIS PROJECT GUTENBERG',
    ]
    start_idx = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            next_newline = text.find('\n', match.end())
            if next_newline != -1:
                start_idx = next_newline + 1
            break

    # Find end of actual content (before *** END ... ***)
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

    # Normalize whitespace: collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def sanitize_filename(title):
    """Convert a title string into a safe filename component."""
    return re.sub(r'[^a-z0-9_]', '', title.lower().replace(' ', '_'))


def main(output_dir=None):
    """Download all expanded Gutenberg books, stripping boilerplate.

    Returns a dict mapping output filenames to download result info.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    downloaded = 0
    cached = 0
    failed = 0

    for author_slug, title, repo, filename in EXPANDED_BOOKS:
        safe_title = sanitize_filename(title)
        out_filename = f"{author_slug}_{safe_title}.txt"
        out_path = os.path.join(output_dir, out_filename)

        # Skip if already downloaded and non-trivial
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            word_count = len(open(out_path, encoding='utf-8').read().split())
            print(f"  [cached] {out_filename}: {word_count:,} words")
            results[out_filename] = {'words': word_count, 'status': 'cached'}
            cached += 1
            continue

        # Try GITenberg first
        gitenberg_url = _build_gitenberg_url(repo, filename)
        print(f"  Downloading: {author_slug} - {title}...", end=' ')
        raw = download_text(gitenberg_url)

        # Fallback to direct Gutenberg cache URL
        if raw is None:
            fallback_url = _build_gutenberg_fallback_url(filename)
            if fallback_url:
                print(f"(trying fallback)...", end=' ')
                raw = download_text(fallback_url)

        if raw is None:
            print("FAILED")
            results[out_filename] = {'words': 0, 'status': 'failed'}
            failed += 1
            continue

        cleaned = strip_gutenberg_boilerplate(raw)
        word_count = len(cleaned.split())

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"{word_count:,} words")
        results[out_filename] = {'words': word_count, 'status': 'ok'}
        downloaded += 1

        # Be polite to servers
        time.sleep(0.5)

    # Print summary stats
    total_words = sum(v['words'] for v in results.values())
    total_books = len(EXPANDED_BOOKS)

    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total books:      {total_books}")
    print(f"  Downloaded:       {downloaded}")
    print(f"  Cached:           {cached}")
    print(f"  Failed:           {failed}")
    print(f"  Total words:      {total_words:,}")
    print(f"  Output directory:  {os.path.abspath(output_dir)}")
    print(f"{'=' * 60}")

    return results


if __name__ == '__main__':
    main()
