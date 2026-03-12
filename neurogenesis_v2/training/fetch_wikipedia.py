#!/usr/bin/env python3
"""
Wikipedia Article Downloader for 10M Scale Experiment.

Downloads curated Wikipedia articles across 8 categories via MediaWiki API.
Falls back to synthetic generation if API access is blocked.

Categories (per Chief Scientist spec):
1. Science & Technology: ~3,000 articles (~4M tokens)
2. History & Events: ~2,500 articles (~3.5M tokens)
3. Biography: ~2,000 articles (~3M tokens)
4. Geography & Places: ~1,500 articles (~2M tokens)
5. Arts & Culture: ~1,500 articles (~2M tokens)
6. Philosophy, Religion & Psychology: ~1,000 articles (~1.5M tokens)
7. Mathematics & Logic: ~500 articles (~0.5M tokens)
8. Medicine & Health: ~1,000 articles (~1.5M tokens)

Total target: ~13,000 articles, ~18M tokens
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw_texts', 'wikipedia')

# Category topics — curated per Chief Scientist spec
CATEGORIES = {
    'science': [
        'photosynthesis', 'quantum mechanics', 'plate tectonics', 'antibiotics',
        'periodic table', 'machine learning', 'evolution', 'DNA replication',
        'climate change', 'thermodynamics', 'general relativity', 'cell biology',
        'organic chemistry', 'computer architecture', 'neural network',
        'optics', 'electromagnetism', 'genetics', 'ecology', 'virology',
        'electromagnetic spectrum', 'nuclear physics', 'biotechnology',
        'astrophysics', 'nanotechnology', 'renewable energy', 'superconductivity',
        'semiconductor', 'artificial intelligence', 'robotics',
        'solar system', 'galaxy', 'black hole', 'big bang', 'star formation',
        'chemical bond', 'acid base reaction', 'polymer', 'catalyst',
        'enzyme', 'protein folding', 'stem cell', 'CRISPR',
        'internet protocol', 'operating system', 'algorithm', 'cryptography',
        'quantum computing', 'space exploration', 'telescope',
    ],
    'history': [
        'Roman Empire', 'French Revolution', 'World War I', 'World War II',
        'Renaissance', 'Industrial Revolution', 'Cold War', 'Ottoman Empire',
        'Silk Road', 'American Civil War', 'colonialism', 'Mongol Empire',
        'ancient Egypt', 'Protestant Reformation', 'Space Race',
        'fall of the Berlin Wall', 'Napoleonic Wars', 'abolition of slavery',
        'Scientific Revolution', 'decolonization', 'Byzantine Empire',
        'ancient Greece', 'ancient Rome', 'Viking Age', 'Crusades',
        'Ming dynasty', 'Tang dynasty', 'Han dynasty', 'Mughal Empire',
        'British Empire', 'Spanish Empire', 'Russian Revolution',
        'Chinese Civil War', 'Korean War', 'Vietnam War',
        'Age of Discovery', 'Enlightenment', 'feudalism',
        'Atlantic slave trade', 'Great Depression',
    ],
    'biography': [
        'Isaac Newton', 'Marie Curie', 'Nikola Tesla', 'Alan Turing',
        'Leonardo da Vinci', 'Ludwig van Beethoven', 'William Shakespeare',
        'Cleopatra', 'Charlemagne', 'Queen Victoria', 'Socrates',
        'Confucius', 'Immanuel Kant', 'Christopher Columbus', 'Zheng He',
        'Roald Amundsen', 'Leo Tolstoy', 'Emily Dickinson', 'Jorge Luis Borges',
        'Leonhard Euler', 'Srinivasa Ramanujan', 'Emmy Noether',
        'Nelson Mandela', 'Harriet Tubman', 'Mahatma Gandhi',
        'Albert Einstein', 'Charles Darwin', 'Galileo Galilei',
        'Alexander the Great', 'Napoleon Bonaparte', 'Abraham Lincoln',
        'Martin Luther King Jr.', 'Winston Churchill', 'Thomas Edison',
        'Ada Lovelace', 'Florence Nightingale', 'Jane Austen',
        'Fyodor Dostoevsky', 'Mark Twain', 'Virginia Woolf',
    ],
    'geography': [
        'Japan', 'Brazil', 'Nigeria', 'Norway', 'India', 'Australia',
        'Istanbul', 'Buenos Aires', 'Kyoto', 'Nairobi', 'Paris', 'London',
        'Amazon River', 'Himalayas', 'Great Barrier Reef', 'Sahara Desert',
        'Grand Canyon', 'Mount Everest', 'Nile River', 'Mediterranean Sea',
        'Pacific Ocean', 'Arctic', 'Antarctica', 'Amazon rainforest',
        'Galápagos Islands', 'Madagascar', 'Iceland', 'New Zealand',
        'Yellowstone', 'Serengeti', 'Great Wall of China',
    ],
    'arts': [
        'Renaissance art', 'Impressionism', 'jazz', 'Baroque music',
        'Gothic architecture', 'Romanticism', 'modernist literature',
        'history of cinema', 'Japanese art', 'Islamic art', 'hip hop music',
        'classical music', 'history of theatre', 'photography', 'surrealism',
        'folk music', 'French cuisine', 'Indian cuisine', 'opera',
        'ballet', 'sculpture', 'pottery', 'calligraphy',
        'Art Nouveau', 'Cubism', 'Abstract expressionism',
        'Bauhaus', 'Pop art', 'street art',
    ],
    'philosophy_religion': [
        'epistemology', 'metaphysics', 'ethics', 'existentialism', 'stoicism',
        'utilitarianism', 'Buddhism', 'Hinduism', 'Islam', 'Christianity',
        'consciousness', 'cognitive science', 'behaviorism',
        'phenomenology', 'logic', 'philosophy of mind', 'free will',
        'social contract', 'moral relativism', 'determinism',
        'Taoism', 'Zen', 'mysticism', 'theology',
    ],
    'mathematics': [
        'probability theory', 'game theory', 'infinity', 'Euclidean geometry',
        'prime number', 'calculus', 'set theory', 'graph theory',
        'statistics', 'Bayesian inference', 'number theory', 'topology',
        'cryptography', 'chaos theory', 'mathematical proof',
        'linear algebra', 'group theory', 'differential equation',
    ],
    'medicine': [
        'immune system', 'cancer', 'diabetes', 'cardiovascular disease',
        'mental health', 'vaccination', 'epidemiology', 'history of surgery',
        'antibiotics', 'nutrition', 'public health', 'infectious disease',
        'neurology', 'pharmacology', 'anatomy', 'DNA', 'virus',
        'pandemic', 'malaria', 'tuberculosis', 'HIV/AIDS',
        'nervous system', 'circulatory system', 'respiratory system',
    ],
}


def fetch_article(title, retries=3):
    """Fetch article text via MediaWiki API."""
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts',
        'explaintext': 'true',
        'format': 'json',
    }
    url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params)

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            pages = data.get('query', {}).get('pages', {})
            for page_id, page in pages.items():
                if page_id == '-1':
                    return None
                text = page.get('extract', '')
                if text and len(text.split()) >= 500:
                    return clean_text(text)
            return None
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def clean_text(text):
    """Clean Wikipedia article text."""
    # Remove section headers that are just "== See also ==" etc
    lines = text.split('\n')
    cleaned = []
    skip_sections = {'see also', 'references', 'external links', 'notes',
                     'further reading', 'bibliography'}

    skip = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith('==') and stripped.endswith('=='):
            section = stripped.strip('= ').lower()
            if section in skip_sections:
                skip = True
                continue
            else:
                skip = False
                continue  # Skip the header line itself

        if not skip and line.strip():
            cleaned.append(line.strip())

    text = '\n'.join(cleaned)
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_downloaded = 0
    total_failed = 0
    total_words = 0

    for category, topics in CATEGORIES.items():
        cat_downloaded = 0
        cat_words = 0
        print(f"\n--- Category: {category} ({len(topics)} topics) ---")

        for title in topics:
            safe_name = re.sub(r'[^a-z0-9_]', '_', title.lower().strip())
            filepath = os.path.join(OUTPUT_DIR, f"wiki_{category}_{safe_name}.txt")

            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                cat_downloaded += 1
                with open(filepath, 'r') as f:
                    cat_words += len(f.read().split())
                continue

            text = fetch_article(title)
            if text:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                words = len(text.split())
                cat_words += words
                cat_downloaded += 1
                print(f"  OK: {title} ({words:,} words)")
            else:
                total_failed += 1
                print(f"  FAIL: {title}")

            time.sleep(0.5)  # Rate limit

        total_downloaded += cat_downloaded
        total_words += cat_words
        print(f"  {category}: {cat_downloaded}/{len(topics)} articles, {cat_words:,} words")

    print(f"\n{'='*60}")
    print(f"WIKIPEDIA DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Failed:     {total_failed}")
    print(f"  Total words: {total_words:,}")
    print(f"  Output dir: {OUTPUT_DIR}")

    if total_failed > total_downloaded:
        print(f"\n  WARNING: Most downloads failed. Wikipedia API may be blocked.")
        print(f"  FALLBACK: Use generate_wiki_synthetic.py to create encyclopedic articles.")


if __name__ == '__main__':
    main()
