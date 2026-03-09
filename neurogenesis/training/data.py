import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SyntheticTransformDataset(Dataset):
    """Generates synthetic vector transformation tasks for Phase 1 primitive bootstrapping."""

    TASK_TYPES = [
        'identity', 'negate', 'shift', 'scale', 'normalize',
        'threshold', 'reverse', 'sort_dims', 'add_noise', 'abs',
        'square_root', 'double', 'halve', 'clip', 'center',
        'pattern_repeat',
    ]

    def __init__(self, d_model: int = 128, num_samples: int = 10000, seed: int = 42):
        self.d_model = d_model
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        # Pre-generate data
        self.inputs = torch.randn(num_samples, d_model)
        self.task_indices = torch.randint(0, len(self.TASK_TYPES), (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.inputs[idx]
        task_idx = self.task_indices[idx].item()
        target = self._apply_transform(x, task_idx)
        return x, target, task_idx

    def _apply_transform(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        task = self.TASK_TYPES[task_idx]
        if task == 'identity':
            return x.clone()
        elif task == 'negate':
            return -x
        elif task == 'shift':
            return torch.roll(x, 1, dims=0)
        elif task == 'scale':
            return x * 0.5
        elif task == 'normalize':
            return x - x.mean()
        elif task == 'threshold':
            return torch.relu(x)
        elif task == 'reverse':
            return x.flip(0)
        elif task == 'sort_dims':
            return x.sort()[0]
        elif task == 'add_noise':
            return x + torch.randn_like(x) * 0.1
        elif task == 'abs':
            return x.abs()
        elif task == 'square_root':
            return x.abs().sqrt() * x.sign()
        elif task == 'double':
            return x * 2.0
        elif task == 'halve':
            return x * 0.5
        elif task == 'clip':
            return x.clamp(-1.0, 1.0)
        elif task == 'center':
            return x - x.mean()
        elif task == 'pattern_repeat':
            half = self.d_model // 2
            return torch.cat([x[:half], x[:half]])
        else:
            return x.clone()


class TextDataset(Dataset):
    """Dataset for language modeling from tokenized text."""

    def __init__(self, token_ids: list, seq_len: int = 64):
        self.seq_len = seq_len
        # Flatten all token_ids into one long sequence
        if isinstance(token_ids[0], list):
            self.data = torch.tensor(
                [t for seq in token_ids for t in seq], dtype=torch.long
            )
        else:
            self.data = torch.tensor(token_ids, dtype=torch.long)
        self.num_sequences = max(0, len(self.data) - seq_len - 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target


def load_tinystories(tokenizer, seq_len: int = 64, max_samples: int = None):
    """Load and tokenize TinyStories dataset. Falls back to synthetic stories."""
    try:
        from datasets import load_dataset
        print("Loading TinyStories dataset...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds['text']
        if max_samples:
            texts = texts[:max_samples]
    except Exception as e:
        print(f"Could not load TinyStories: {e}")
        print("Generating synthetic training stories instead...")
        texts = generate_synthetic_stories(max_samples or 10000)

    print(f"Tokenizing {len(texts)} stories...")
    all_ids = []
    for i, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        all_ids.extend(encoded.ids)
        if (i + 1) % 10000 == 0:
            print(f"  Tokenized {i+1}/{len(texts)}")

    print(f"Total tokens: {len(all_ids)}")
    return TextDataset(all_ids, seq_len=seq_len)


def load_tinystories_validation(tokenizer, seq_len: int = 64, max_samples: int = None):
    """Load validation set. Falls back to synthetic stories."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="validation")
        texts = ds['text']
        if max_samples:
            texts = texts[:max_samples]
    except Exception:
        texts = generate_synthetic_stories(max_samples or 1000, seed=9999)

    all_ids = []
    for text in texts:
        encoded = tokenizer.encode(text)
        all_ids.extend(encoded.ids)

    return TextDataset(all_ids, seq_len=seq_len)


def generate_synthetic_stories(num_stories: int = 10000, seed: int = 42) -> list:
    """Generate simple synthetic stories for training when TinyStories is unavailable."""
    import random
    rng = random.Random(seed)

    names = ["Lily", "Tom", "Sara", "Max", "Emma", "Ben", "Mia", "Jack",
             "Anna", "Leo", "Lucy", "Sam", "Kate", "Dan", "Ella", "Tim",
             "Zoe", "Finn", "Ivy", "Noah", "Ava", "Luke", "Ruby", "Owen"]
    animals = ["cat", "dog", "bird", "rabbit", "fish", "frog", "bear",
               "fox", "deer", "owl", "duck", "mouse", "turtle", "bee"]
    colors = ["red", "blue", "green", "yellow", "pink", "purple", "orange",
              "white", "black", "brown", "golden", "silver"]
    places = ["park", "garden", "forest", "river", "hill", "beach", "meadow",
              "village", "school", "house", "lake", "mountain", "farm", "cave"]
    objects = ["ball", "flower", "stone", "star", "book", "cake", "toy",
               "hat", "box", "cup", "kite", "bell", "ring", "leaf"]
    feelings = ["happy", "sad", "excited", "surprised", "brave", "kind",
                "proud", "curious", "cheerful", "gentle", "friendly", "calm"]
    actions = ["walked", "ran", "jumped", "danced", "played", "sang",
               "laughed", "smiled", "skipped", "climbed", "swam", "flew"]

    templates = [
        "Once upon a time, there was a {feeling} {name} who lived near a {place}. One day, {name} found a {color} {object} by the {place}. {name} picked it up and {action} all the way home. It was the best day ever.",
        "{name} had a little {animal}. The {animal} was very {feeling}. They {action} together in the {place} every day. {name} loved the {animal} so much.",
        "One morning, {name} woke up and saw a {color} {animal} in the {place}. The {animal} was looking for a {object}. {name} helped the {animal} find it. The {animal} was so {feeling}.",
        "There was a {color} {object} in the {place}. {name} wanted to find it. {name} {action} through the {place} until finding the {object}. {name} felt very {feeling}.",
        "{name} and {name2} were friends. They liked to play in the {place}. One day they found a {color} {object}. They shared it and were both {feeling}.",
        "The {color} {animal} {action} across the {place}. {name} watched and felt {feeling}. Then {name} {action} too. They became good friends.",
        "It was a sunny day. {name} went to the {place} with a {color} {object}. A {animal} came and wanted to play. {name} and the {animal} {action} together. Everyone was {feeling}.",
        "{name} had a dream about a {color} {place}. In the dream, a {feeling} {animal} gave {name} a special {object}. When {name} woke up, {name} felt {feeling}.",
        "In a little {place}, there lived a {feeling} {animal}. The {animal} liked to collect {color} {object}s. One day, {name} came and they {action} together.",
        "Once, {name} lost a {color} {object} in the {place}. {name} was {feeling}. But a kind {animal} found it and brought it back. {name} was so {feeling}.",
        "The {place} was quiet. Then {name} came with a {object}. Soon everyone was playing. The {animal} {action} and {name} {action}. It was a {feeling} day.",
        "{name} wanted to be brave. {name} {action} to the big {place}. There was a {color} {animal} there. But the {animal} was {feeling}. They became friends.",
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
        )
        stories.append(story)

    return stories
