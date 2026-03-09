"""Download and prepare Quran English translation for training."""
import json
import os
import time
import urllib.request
import urllib.error


QURAN_BASE_URL = "https://raw.githubusercontent.com/semarketir/quranjson/master/source/translation/en/en_translation_{}.json"
QURAN_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
QURAN_TEXT_PATH = os.path.join(QURAN_DATA_DIR, 'quran_en.txt')


def download_quran(output_path=None, max_retries=3):
    """Download all 114 surahs and save as plain text."""
    if output_path is None:
        output_path = QURAN_TEXT_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_text = []
    for surah_num in range(1, 115):
        url = QURAN_BASE_URL.format(surah_num)
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode('utf-8'))

                name = data.get('name', f'Surah {surah_num}')
                verses = data.get('verse', {})

                # Build text: each surah as a paragraph
                surah_text = f"Surah {surah_num}: {name}\n"
                for key in sorted(verses.keys(), key=lambda k: int(k.split('_')[1])):
                    surah_text += verses[key] + " "
                surah_text = surah_text.strip()
                all_text.append(surah_text)

                if surah_num % 10 == 0:
                    print(f"  Downloaded {surah_num}/114 surahs")
                break

            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retry {attempt+1} for surah {surah_num}: {e}")
                    time.sleep(wait)
                else:
                    print(f"  Failed to download surah {surah_num}: {e}")

    full_text = "\n\n".join(all_text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    word_count = len(full_text.split())
    print(f"Quran downloaded: {len(all_text)} surahs, {word_count:,} words")
    return output_path


def load_quran_text(path=None):
    """Load Quran text, downloading if needed."""
    if path is None:
        path = QURAN_TEXT_PATH
    if not os.path.exists(path):
        print("Quran text not found, downloading...")
        download_quran(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


if __name__ == '__main__':
    download_quran()
