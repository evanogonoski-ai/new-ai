"""Data loading — combines synthetic stories + Quran text for richer training."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Re-export from V1
from neurogenesis.training.data import (
    TextDataset,
    generate_synthetic_stories,
    SyntheticTransformDataset,
)
from neurogenesis.tokenizer.bpe import train_tokenizer, load_tokenizer, TOKENIZER_PATH


def generate_enhanced_stories(num_stories: int = 10000, seed: int = 42) -> list:
    """Generate richer synthetic stories with more diverse vocabulary and structures."""
    import random
    rng = random.Random(seed)

    names = [
        "Lily", "Tom", "Sara", "Max", "Emma", "Ben", "Mia", "Jack",
        "Anna", "Leo", "Lucy", "Sam", "Kate", "Dan", "Ella", "Tim",
        "Zoe", "Finn", "Ivy", "Noah", "Ava", "Luke", "Ruby", "Owen",
        "Grace", "Ethan", "Sofia", "Liam", "Chloe", "Oliver", "Aria",
        "James", "Nora", "Henry", "Layla", "Kai", "Maya", "Theo",
        "Amira", "Yusuf", "Fatima", "Ali", "Zahra", "Ibrahim", "Hana",
    ]
    animals = [
        "cat", "dog", "bird", "rabbit", "fish", "frog", "bear",
        "fox", "deer", "owl", "duck", "mouse", "turtle", "bee",
        "eagle", "dolphin", "wolf", "horse", "butterfly", "sparrow",
        "lamb", "dove", "camel", "lion", "gazelle", "nightingale",
    ]
    colors = [
        "red", "blue", "green", "yellow", "pink", "purple", "orange",
        "white", "black", "brown", "golden", "silver", "crimson",
        "emerald", "sapphire", "ivory", "scarlet", "turquoise",
    ]
    places = [
        "park", "garden", "forest", "river", "hill", "beach", "meadow",
        "village", "school", "house", "lake", "mountain", "farm", "cave",
        "valley", "desert", "oasis", "temple", "market", "bridge",
        "harbor", "tower", "orchard", "courtyard", "path", "summit",
    ]
    objects = [
        "ball", "flower", "stone", "star", "book", "cake", "toy",
        "hat", "box", "cup", "kite", "bell", "ring", "leaf",
        "lamp", "mirror", "pearl", "feather", "scroll", "compass",
        "lantern", "seed", "key", "crown", "blanket", "bread",
    ]
    feelings = [
        "happy", "sad", "excited", "surprised", "brave", "kind",
        "proud", "curious", "cheerful", "gentle", "friendly", "calm",
        "grateful", "humble", "peaceful", "hopeful", "patient",
        "merciful", "generous", "wise", "faithful", "determined",
    ]
    actions = [
        "walked", "ran", "jumped", "danced", "played", "sang",
        "laughed", "smiled", "skipped", "climbed", "swam", "flew",
        "whispered", "wondered", "helped", "shared", "carried",
        "discovered", "remembered", "forgave", "prayed", "traveled",
    ]
    weather = [
        "sunny", "rainy", "cloudy", "windy", "snowy", "warm",
        "cool", "misty", "stormy", "clear", "bright", "calm",
    ]
    times = [
        "morning", "afternoon", "evening", "night", "dawn", "dusk",
        "sunrise", "sunset", "midnight", "daybreak",
    ]
    virtues = [
        "kindness", "patience", "courage", "honesty", "generosity",
        "forgiveness", "gratitude", "wisdom", "mercy", "compassion",
        "faith", "justice", "humility", "perseverance", "love",
    ]

    templates = [
        "Once upon a time, there was a {feeling} {name} who lived near a {place}. One day, {name} found a {color} {object} by the {place}. {name} picked it up and {action} all the way home. It was the best day ever.",
        "{name} had a little {animal}. The {animal} was very {feeling}. They {action} together in the {place} every day. {name} loved the {animal} so much.",
        "One {time}, {name} woke up and saw a {color} {animal} in the {place}. The {animal} was looking for a {object}. {name} helped the {animal} find it. The {animal} was so {feeling}.",
        "There was a {color} {object} in the {place}. {name} wanted to find it. {name} {action} through the {place} until finding the {object}. {name} felt very {feeling}.",
        "{name} and {name2} were friends. They liked to play in the {place}. One day they found a {color} {object}. They shared it and were both {feeling}.",
        "The {color} {animal} {action} across the {place}. {name} watched and felt {feeling}. Then {name} {action} too. They became good friends.",
        "It was a {weather} day. {name} went to the {place} with a {color} {object}. A {animal} came and wanted to play. {name} and the {animal} {action} together. Everyone was {feeling}.",
        "{name} had a dream about a {color} {place}. In the dream, a {feeling} {animal} gave {name} a special {object}. When {name} woke up, {name} felt {feeling}.",
        "In a little {place}, there lived a {feeling} {animal}. The {animal} liked to collect {color} {object}s. One day, {name} came and they {action} together.",
        "Once, {name} lost a {color} {object} in the {place}. {name} was {feeling}. But a kind {animal} found it and brought it back. {name} was so {feeling}.",
        "The {place} was quiet. Then {name} came with a {object}. Soon everyone was playing. The {animal} {action} and {name} {action}. It was a {feeling} day.",
        "{name} wanted to be brave. {name} {action} to the big {place}. There was a {color} {animal} there. But the {animal} was {feeling}. They became friends.",
        "At {time}, the {weather} sky stretched over the {place}. {name} sat quietly, thinking about {virtue}. A {color} {animal} appeared and seemed to understand. Together they {action} toward the {place}, where {name2} was waiting with a warm {object}.",
        "{name} learned about {virtue} from watching the {animal} in the {place}. The {animal} always {action} when others needed help. This taught {name} to be {feeling}. From that day forward, {name} tried to show {virtue} in everything.",
        "The old {place} held many memories. {name} {action} through it one {time}, finding a {color} {object} left by {name2} long ago. Holding it close, {name} felt {feeling}. Some things are worth remembering.",
        "Under the {weather} sky, {name} {action} along the {place}. The wind carried the sound of a {animal} singing. {name} stopped to listen and felt {feeling}. Nature has its own way of teaching {virtue}.",
        "Every {time}, {name} would visit the {place} to feed the {animal}s. One day, a small {color} {animal} followed {name} home. {name2} said they could keep it. The {animal} brought so much joy and taught them about {virtue}.",
        "{name} and {name2} had an argument about the {object}. {name} felt {feeling} but then remembered the importance of {virtue}. {name} {action} back to {name2} and said sorry. {name2} smiled and they were friends again.",
        "A {feeling} traveler named {name} arrived at the {place} one {time}. The people there were {feeling} and shared their {object} with {name}. {name} learned that {virtue} exists everywhere if you look for it.",
        "The {color} {object} glowed softly in the {time} light. {name} held it up and {action} with wonder. In the {place} nearby, a {animal} watched with {feeling} eyes. Some moments are simply perfect.",
        "Long ago, in a distant {place}, there lived a {feeling} {name} who believed in {virtue}. {name} {action} far and wide, helping those in need. The {animal}s of the {place} became {name}'s companions on this journey.",
        "The {weather} {time} brought everyone to the {place}. {name} carried a basket of {object}s to share. {name2} brought stories of {virtue}. The {animal}s gathered too, as if they knew something special was happening.",
        "What is {virtue}? {name} asked one {time}. {name2} pointed to the {animal} caring for its young in the {place}. That, said {name2}, is {virtue}. {name} nodded, feeling {feeling} and grateful.",
    ]

    stories = []
    for _ in range(num_stories):
        template = rng.choice(templates)
        story = template.format(
            name=rng.choice(names),
            name2=rng.choice(names),
            animal=rng.choice(animals),
            color=rng.choice(colors),
            place=rng.choice(places),
            object=rng.choice(objects),
            feeling=rng.choice(feelings),
            action=rng.choice(actions),
            weather=rng.choice(weather),
            time=rng.choice(times),
            virtue=rng.choice(virtues),
        )
        stories.append(story)

    return stories


def load_quran_paragraphs(path=None):
    """Load Quran text and split into paragraph-sized chunks for training."""
    from neurogenesis_v2.training.fetch_quran import load_quran_text
    text = load_quran_text(path)
    # Split by double newline (each surah is a paragraph)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    # Further split long surahs into chunks of ~200 words
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= 250:
            chunks.append(para)
        else:
            for i in range(0, len(words), 200):
                chunk = ' '.join(words[i:i+200])
                if len(chunk.split()) > 20:
                    chunks.append(chunk)
    return chunks


def prepare_combined_corpus(num_synthetic=50000, include_quran=True):
    """Build combined corpus from enhanced synthetic stories + Quran text."""
    stories = generate_enhanced_stories(num_synthetic)
    print(f"Generated {len(stories)} enhanced synthetic stories")

    corpus = list(stories)

    if include_quran:
        try:
            quran_chunks = load_quran_paragraphs()
            # Repeat Quran chunks to balance with synthetic data
            for _ in range(5):
                corpus.extend(quran_chunks)
            print(f"Added {len(quran_chunks)} Quran chunks (x5 repetitions)")
        except Exception as e:
            print(f"Could not load Quran text: {e}")
            print("Continuing with synthetic data only")

    return corpus
