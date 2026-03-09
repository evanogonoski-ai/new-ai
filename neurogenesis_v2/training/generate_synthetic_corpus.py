#!/usr/bin/env python3
"""Generate structurally diverse synthetic text for corpus expansion.

Produces 25,000 texts across 7 formats:
- Narrative stories (5,000)
- Dialogues (5,000)
- Q&A pairs (3,000)
- Descriptive passages (3,000)
- Instructional text (3,000)
- Argumentative text (3,000)
- Letters/correspondence (3,000)
"""
import os
import random


# Shared vocabulary pools (much expanded from V2)
NAMES = [
    "Lily", "Tom", "Sara", "Max", "Emma", "Ben", "Mia", "Jack",
    "Anna", "Leo", "Lucy", "Sam", "Kate", "Dan", "Ella", "Tim",
    "Zoe", "Finn", "Ivy", "Noah", "Ava", "Luke", "Ruby", "Owen",
    "Grace", "Ethan", "Sofia", "Liam", "Chloe", "Oliver", "Aria",
    "James", "Nora", "Henry", "Layla", "Kai", "Maya", "Theo",
    "Amira", "Yusuf", "Fatima", "Ali", "Zahra", "Ibrahim", "Hana",
    "Marcus", "Seneca", "Plato", "Aristotle", "Helen", "Arthur",
    "Clara", "George", "Margaret", "William", "Elizabeth", "Thomas",
    "Catherine", "Robert", "Rose", "Edward", "Florence", "Charles",
]

TOPICS = [
    "kindness", "patience", "courage", "honesty", "generosity",
    "forgiveness", "gratitude", "wisdom", "mercy", "compassion",
    "justice", "humility", "perseverance", "love", "friendship",
    "truth", "beauty", "nature", "knowledge", "freedom",
    "responsibility", "respect", "loyalty", "hope", "faith",
    "creativity", "discipline", "empathy", "integrity", "peace",
]

PLACES = [
    "park", "garden", "forest", "river", "hill", "beach", "meadow",
    "village", "school", "house", "lake", "mountain", "farm", "cave",
    "valley", "desert", "oasis", "temple", "market", "bridge",
    "harbor", "tower", "orchard", "courtyard", "path", "summit",
    "library", "workshop", "kitchen", "harbor", "cliff", "grove",
    "cathedral", "ruins", "cottage", "castle", "monastery", "inn",
]

ANIMALS = [
    "cat", "dog", "bird", "rabbit", "fish", "frog", "bear",
    "fox", "deer", "owl", "duck", "mouse", "turtle", "bee",
    "eagle", "dolphin", "wolf", "horse", "butterfly", "sparrow",
    "lamb", "dove", "camel", "lion", "gazelle", "nightingale",
    "hawk", "crow", "swan", "elephant", "tiger", "whale",
]

COLORS = [
    "red", "blue", "green", "yellow", "pink", "purple", "orange",
    "white", "black", "brown", "golden", "silver", "crimson",
    "emerald", "sapphire", "ivory", "scarlet", "turquoise",
    "amber", "violet", "indigo", "coral", "bronze", "grey",
]

OBJECTS = [
    "ball", "flower", "stone", "star", "book", "cake", "toy",
    "hat", "box", "cup", "kite", "bell", "ring", "leaf",
    "lamp", "mirror", "pearl", "feather", "scroll", "compass",
    "lantern", "seed", "key", "crown", "blanket", "bread",
    "candle", "sword", "shield", "quill", "map", "telescope",
    "violin", "painting", "clock", "journal", "coin", "gem",
]

FEELINGS = [
    "happy", "sad", "excited", "surprised", "brave", "kind",
    "proud", "curious", "cheerful", "gentle", "friendly", "calm",
    "grateful", "humble", "peaceful", "hopeful", "patient",
    "merciful", "generous", "wise", "faithful", "determined",
    "anxious", "thoughtful", "serene", "melancholy", "resolute",
    "weary", "delighted", "solemn", "tender", "fierce",
]

ACTIONS = [
    "walked", "ran", "jumped", "danced", "played", "sang",
    "laughed", "smiled", "skipped", "climbed", "swam", "flew",
    "whispered", "wondered", "helped", "shared", "carried",
    "discovered", "remembered", "forgave", "prayed", "traveled",
    "studied", "built", "painted", "wrote", "listened", "observed",
    "gathered", "planted", "cooked", "healed", "taught", "learned",
]

WEATHER = [
    "sunny", "rainy", "cloudy", "windy", "snowy", "warm",
    "cool", "misty", "stormy", "clear", "bright", "calm",
    "foggy", "humid", "crisp", "mild", "chilly", "blazing",
]

TIMES = [
    "morning", "afternoon", "evening", "night", "dawn", "dusk",
    "sunrise", "sunset", "midnight", "daybreak", "noon", "twilight",
]

SAID_VERBS = [
    "said", "asked", "replied", "whispered", "exclaimed", "murmured",
    "answered", "wondered", "suggested", "explained", "argued",
    "declared", "admitted", "noted", "observed", "recalled",
    "insisted", "proposed", "questioned", "reflected",
]

ADJECTIVES = [
    "small", "large", "ancient", "young", "quiet", "loud",
    "bright", "dark", "warm", "cold", "deep", "shallow",
    "narrow", "wide", "simple", "complex", "gentle", "fierce",
    "smooth", "rough", "fragile", "sturdy", "rare", "common",
    "bitter", "sweet", "sharp", "soft", "heavy", "light",
]

ABSTRACT_NOUNS = [
    "truth", "beauty", "justice", "freedom", "wisdom", "knowledge",
    "virtue", "honor", "courage", "faith", "hope", "love",
    "reason", "nature", "time", "memory", "silence", "power",
    "harmony", "balance", "order", "chaos", "change", "growth",
    "purpose", "meaning", "identity", "destiny", "choice", "consequence",
]

PROFESSIONS = [
    "teacher", "farmer", "healer", "scholar", "merchant", "sailor",
    "blacksmith", "weaver", "painter", "musician", "cook", "builder",
    "shepherd", "scribe", "philosopher", "astronomer", "gardener",
    "carpenter", "potter", "storyteller",
]

FOODS = [
    "bread", "soup", "cheese", "fruit", "honey", "rice",
    "fish", "tea", "milk", "nuts", "berries", "dates",
    "olives", "figs", "grapes", "lentils", "herbs", "spices",
]


def _c(rng, lst):
    """Choose random item from list."""
    return rng.choice(lst)


def generate_narrative_stories(n, rng):
    """Generate narrative stories with varied templates."""
    templates = [
        "Once upon a time, there was a {feeling} {name} who lived near a {place}. One day, {name} found a {color} {object} by the {place}. {name} picked it up and {action} all the way home. It was the best day ever.",
        "{name} had a little {animal}. The {animal} was very {feeling}. They {action} together in the {place} every day. {name} loved the {animal} so much.",
        "One {time}, {name} woke up and saw a {color} {animal} in the {place}. The {animal} was looking for a {object}. {name} helped the {animal} find it. The {animal} was so {feeling}.",
        "At {time}, the {weather} sky stretched over the {place}. {name} sat quietly, thinking about {topic}. A {color} {animal} appeared and seemed to understand. Together they {action} toward the {place2}, where {name2} was waiting with a warm {object}.",
        "{name} learned about {topic} from watching the {animal} in the {place}. The {animal} always {action} when others needed help. This taught {name} to be {feeling}. From that day forward, {name} tried to show {topic} in everything.",
        "The old {place} held many memories. {name} {action} through it one {time}, finding a {color} {object} left by {name2} long ago. Holding it close, {name} felt {feeling}. Some things are worth remembering.",
        "Long ago, in a distant {place}, there lived a {feeling} {profession} named {name} who believed in {topic}. {name} {action} far and wide, helping those in need. The {animal}s of the {place} became {name}'s companions on this journey.",
        "The {weather} {time} brought everyone to the {place}. {name} carried a basket of {food} to share. {name2} brought stories of {topic}. The {animal}s gathered too, as if they knew something special was happening.",
        "What is {topic}? {name} asked one {time}. {name2} pointed to the {animal} caring for its young in the {place}. That, {name2} {said}, is {topic}. {name} nodded, feeling {feeling} and grateful.",
        "{name} was a {profession} in a {adj} {place}. Every {time}, {name} would {action} to the {place2} to work. One day, a {feeling} stranger named {name2} arrived, carrying a {color} {object}. Their meeting changed everything.",
        "The {adj} {place} had been quiet for years. Then {name}, a young {profession}, arrived with nothing but a {object} and a dream. {name} believed that {topic} could transform even the most {adj} of places. And slowly, it did.",
    ]

    stories = []
    for _ in range(n):
        t = _c(rng, templates)
        stories.append(t.format(
            name=_c(rng, NAMES), name2=_c(rng, NAMES), animal=_c(rng, ANIMALS),
            color=_c(rng, COLORS), place=_c(rng, PLACES), place2=_c(rng, PLACES),
            object=_c(rng, OBJECTS), feeling=_c(rng, FEELINGS), action=_c(rng, ACTIONS),
            weather=_c(rng, WEATHER), time=_c(rng, TIMES), topic=_c(rng, TOPICS),
            said=_c(rng, SAID_VERBS), adj=_c(rng, ADJECTIVES),
            profession=_c(rng, PROFESSIONS), food=_c(rng, FOODS),
        ))
    return stories


def generate_dialogues(n, rng):
    """Generate two-character dialogues with turn-taking."""
    openers = [
        '"{name}, do you ever think about {topic}?" {name2} {said} one {time}.',
        '{name} and {name2} sat by the {place}. "{name2}," {name} began, "I have been wondering about {topic}."',
        '"What do you think {topic} really means?" {name} {said}, looking at the {color} {object} in the {place}.',
        'The {weather} {time} found {name} and {name2} walking through the {place}. {name} broke the silence first.',
    ]
    exchanges = [
        '"{topic} is like a {object}," {name2} {said}. "You must hold it carefully, or it breaks."',
        '"I disagree," {name} {said}. "I think {topic} is more like the {weather} — you cannot control it, only respond to it."',
        '{name2} paused. "Perhaps you are right. But have you considered that {topic} and {topic2} are connected?"',
        '"Tell me more," {name} {said}, leaning forward with {feeling} eyes.',
        '"When I was young," {name2} {said}, "my {profession} taught me that {topic} begins with {topic2}."',
        '"That is a {adj} thought," {name} replied. "But what about {topic2}? Does it not matter as well?"',
        '{name2} smiled. "You ask good questions. A true {profession} would be proud."',
        '"I learned this from watching the {animal}s," {name} explained. "They understand {topic} without words."',
        '"Interesting," {name2} {said}. "I never thought of it that way before."',
        '"Perhaps we are both right," {name} suggested. "There is more than one path to {topic}."',
    ]
    closers = [
        'They sat in {feeling} silence as the {time} settled around them. Some conversations need no ending.',
        '{name} and {name2} looked at each other and smiled. The {animal} nearby seemed to understand.',
        '"Thank you, {name2}," {name} {said}. "This has been a {feeling} conversation." {name2} nodded.',
        'As the {weather} sky darkened, they walked back together, each carrying new understanding of {topic}.',
    ]

    dialogues = []
    for _ in range(n):
        parts = [_c(rng, openers)]
        for _ in range(rng.randint(2, 5)):
            parts.append(_c(rng, exchanges))
        parts.append(_c(rng, closers))

        n1, n2 = _c(rng, NAMES), _c(rng, NAMES)
        while n2 == n1:
            n2 = _c(rng, NAMES)

        text = ' '.join(parts).format(
            name=n1, name2=n2, topic=_c(rng, TOPICS), topic2=_c(rng, TOPICS),
            said=_c(rng, SAID_VERBS), time=_c(rng, TIMES), place=_c(rng, PLACES),
            color=_c(rng, COLORS), object=_c(rng, OBJECTS), weather=_c(rng, WEATHER),
            feeling=_c(rng, FEELINGS), profession=_c(rng, PROFESSIONS),
            animal=_c(rng, ANIMALS), adj=_c(rng, ADJECTIVES),
        )
        dialogues.append(text)
    return dialogues


def generate_qa_pairs(n, rng):
    """Generate Q&A pairs with varying complexity."""
    templates = [
        'What is {topic}? {topic} is the quality of being {feeling} and showing concern for others. It can be seen in the way a {profession} treats those around them, or in the simple act of sharing {food} with a stranger in the {place}.',
        'Why is {topic} important? Without {topic}, communities cannot thrive. A {place} without {topic} becomes a {adj} place where people lose {topic2}. History teaches us that {topic} and {topic2} are the foundations of a just society.',
        'How can one practice {topic}? Begin each {time} with a small act of {topic}. A {profession} might {action} to help a neighbor. A {name} might share a {object} with someone in need. Even watching a {animal} in the {place} can remind us of {topic}.',
        'Is {topic} the same as {topic2}? While {topic} and {topic2} are related, they are not identical. {topic} comes from the heart, while {topic2} comes from the mind. A truly {feeling} person cultivates both.',
        'Can {topic} be learned? Yes. Like any {adj} skill, {topic} can be developed through practice. A {profession} becomes better by working daily. Similarly, {topic} grows stronger each time we choose it over {topic2}.',
        'What did the ancients say about {topic}? Many {adj} thinkers wrote about {topic}. They believed that {topic} was essential for a {feeling} life. In their view, without {topic}, neither {topic2} nor {abstract} could truly flourish.',
        'What is the relationship between {topic} and {abstract}? They are deeply connected. {topic} enables {abstract}, and {abstract} strengthens {topic}. A {profession} who practices both will find their work more {feeling} and meaningful.',
    ]

    pairs = []
    for _ in range(n):
        pairs.append(_c(rng, templates).format(
            topic=_c(rng, TOPICS), topic2=_c(rng, TOPICS), feeling=_c(rng, FEELINGS),
            profession=_c(rng, PROFESSIONS), food=_c(rng, FOODS), place=_c(rng, PLACES),
            adj=_c(rng, ADJECTIVES), time=_c(rng, TIMES), action=_c(rng, ACTIONS),
            name=_c(rng, NAMES), object=_c(rng, OBJECTS), animal=_c(rng, ANIMALS),
            abstract=_c(rng, ABSTRACT_NOUNS),
        ))
    return pairs


def generate_descriptive_passages(n, rng):
    """Generate scene-setting passages with spatial, temporal, and sensory language."""
    templates = [
        'The {place} stretched out before them in the {time} light. To the left, a {adj} {object} caught the {color} rays of the sun. The air was {weather} and carried the scent of {food} from a nearby {place2}. A {animal} moved silently through the {adj2} grass, its {color2} feathers catching the breeze.',
        'It was a {weather} {time}. The {place} was bathed in {color} light that filtered through the {adj} canopy above. Below, the earth was {adj2} and cool. {name} could hear the distant call of a {animal}, and somewhere, water {action} over {adj} stones. The whole world felt {feeling}.',
        'The {place} in {time} was a different world entirely. {color} shadows stretched across the {adj} ground. The {weather} air was thick with the fragrance of {adj2} {object}s that grew along the path. In the distance, a {animal} called out, its voice echoing across the {place2}.',
        'From the top of the {place}, {name} could see everything: the {color} {place2} far below, the {adj} line of the horizon, and the {weather} sky above. The wind was {adj2}, carrying with it the sound of {animal}s and the taste of salt. It was the most {feeling} view {name} had ever seen.',
        'Inside the {adj} {place}, the air was still and {weather}. {color} light fell through narrow windows onto {adj2} floors. Shelves lined the walls, holding {object}s and {object2}s of every description. The silence was {feeling}, broken only by the soft sound of a {animal} somewhere in the shadows.',
    ]

    passages = []
    for _ in range(n):
        passages.append(_c(rng, templates).format(
            place=_c(rng, PLACES), place2=_c(rng, PLACES), time=_c(rng, TIMES),
            adj=_c(rng, ADJECTIVES), adj2=_c(rng, ADJECTIVES), color=_c(rng, COLORS),
            color2=_c(rng, COLORS), object=_c(rng, OBJECTS), object2=_c(rng, OBJECTS),
            weather=_c(rng, WEATHER), animal=_c(rng, ANIMALS), name=_c(rng, NAMES),
            feeling=_c(rng, FEELINGS), action=_c(rng, ACTIONS), food=_c(rng, FOODS),
        ))
    return passages


def generate_instructional_text(n, rng):
    """Generate how-to style text with sequential steps and conditionals."""
    templates = [
        'How to practice {topic}: First, find a {adj} {place} where you can think without distraction. Then, take a {object} and write down what {topic} means to you. If you feel {feeling}, that is normal. Next, {action} to someone you trust and share your thoughts. If they respond with {topic2}, listen carefully. Finally, try to show {topic} in one small way each {time}.',
        'To become a better {profession}, follow these steps. Begin by {action} every {time} before the day starts. Second, observe how a {adj} {profession} handles {adj2} situations. Third, practice {topic} even when it is difficult. If you find yourself feeling {feeling}, remember that growth requires patience. The most important step is the last: never stop learning.',
        'Making {food} requires attention and {topic}. Start with {adj} ingredients from the {place}. If the {weather} is too {adj2}, adjust your approach. Stir carefully until the mixture is {adj}. Then, let it rest while you prepare the {object}. If the result is not perfect, do not worry. As any {profession} will tell you, mastery comes through practice and {topic}.',
        'Learning {topic} is a journey, not a destination. Step one: read what others have written about {topic}. Step two: observe {topic} in daily life, especially in the {place} and {place2}. Step three: if you meet someone who embodies {topic}, ask them how they learned. Step four: practice. If you fail, that is part of the process. Step five: teach what you have learned to someone else.',
    ]

    texts = []
    for _ in range(n):
        texts.append(_c(rng, templates).format(
            topic=_c(rng, TOPICS), topic2=_c(rng, TOPICS), feeling=_c(rng, FEELINGS),
            profession=_c(rng, PROFESSIONS), place=_c(rng, PLACES), place2=_c(rng, PLACES),
            adj=_c(rng, ADJECTIVES), adj2=_c(rng, ADJECTIVES), object=_c(rng, OBJECTS),
            weather=_c(rng, WEATHER), time=_c(rng, TIMES), action=_c(rng, ACTIONS),
            food=_c(rng, FOODS),
        ))
    return texts


def generate_argumentative_text(n, rng):
    """Generate persuasive text with evidence and counterpoints."""
    templates = [
        '{topic} is more important than {topic2}, and here is why. First, consider the {profession} who must choose between them daily. When faced with a {adj} decision, {topic} provides a clear path forward, while {topic2} can sometimes lead to confusion. Critics might argue that {topic2} is equally vital, and they make a fair point. However, without {topic}, {topic2} lacks a foundation. The evidence from watching any {place} flourish or decline confirms this.',
        'Some believe that {topic} is a {adj} ideal, impossible to achieve in practice. I disagree. Every {profession} in every {place} demonstrates {topic} through small acts: sharing {food}, helping a {feeling} neighbor, or simply listening. While it is true that {topic} requires effort, and that {topic2} sometimes seems easier, the rewards of choosing {topic} far outweigh the cost.',
        'The question of whether {topic} or {topic2} matters more has been debated for centuries. Those who favor {topic2} point to its practical benefits: a {place} built on {topic2} is {adj} and efficient. But is efficiency enough? A {place} without {topic} may function, but it will never truly thrive. The {animal} in the {place2} teaches us this: even in nature, {topic} is essential for survival.',
        'It could be argued that {topic} is merely a {adj} sentiment, not a guiding principle. However, this view ignores the evidence. The most {feeling} societies in history were those that valued {topic} above {topic2}. A {profession} who practices {topic} earns the trust of others, while one who relies on {topic2} alone will eventually find themselves isolated.',
    ]

    texts = []
    for _ in range(n):
        texts.append(_c(rng, templates).format(
            topic=_c(rng, TOPICS), topic2=_c(rng, TOPICS), feeling=_c(rng, FEELINGS),
            profession=_c(rng, PROFESSIONS), place=_c(rng, PLACES), place2=_c(rng, PLACES),
            adj=_c(rng, ADJECTIVES), object=_c(rng, OBJECTS), animal=_c(rng, ANIMALS),
            food=_c(rng, FOODS),
        ))
    return texts


def generate_letters(n, rng):
    """Generate formal and informal correspondence."""
    templates = [
        'Dear {name2}, I hope this letter finds you well. I am writing from the {place}, where the {weather} days have given me time to think about {topic}. Do you remember when we {action} together in the {place2}? That {time} taught me something important about {topic}. I have enclosed a {color} {object} that reminded me of you. With warm regards, {name}.',
        'My dear friend {name2}, it has been too long since we last spoke. Much has changed here in the {place}. A new {profession} has arrived and brought with them ideas about {topic} that I find both {feeling} and {adj}. I would value your thoughts on this matter. Write back when you can. Your {feeling} friend, {name}.',
        '{name2}, I must tell you what happened today. I was walking through the {place} when I saw a {color} {animal} behaving in the most {adj} way. It made me think of what you once said about {topic}. You were right, of course. You always are about these things. I miss our conversations about {topic} and {topic2}. Please visit soon. Yours, {name}.',
        'To {name2}, esteemed {profession}, I write to you regarding the matter of {topic} that we discussed at the {place}. After much reflection, I have come to believe that your position on {topic} is {adj} and well-reasoned. However, I would like to suggest that {topic2} also deserves consideration. I look forward to continuing this exchange of ideas. Respectfully, {name}, {profession2}.',
        'Dearest {name2}, the {time} here is {weather} and {adj}. I sit by the window of the {place}, watching the {animal}s and thinking of you. {topic} is on my mind today, as it so often is. Do you recall the {color} {object} we found together? I keep it still, a reminder of what {topic} truly means. All my {topic2}, {name}.',
    ]

    letters = []
    for _ in range(n):
        letters.append(_c(rng, templates).format(
            name=_c(rng, NAMES), name2=_c(rng, NAMES), topic=_c(rng, TOPICS),
            topic2=_c(rng, TOPICS), place=_c(rng, PLACES), place2=_c(rng, PLACES),
            weather=_c(rng, WEATHER), time=_c(rng, TIMES), action=_c(rng, ACTIONS),
            color=_c(rng, COLORS), object=_c(rng, OBJECTS), feeling=_c(rng, FEELINGS),
            adj=_c(rng, ADJECTIVES), profession=_c(rng, PROFESSIONS),
            profession2=_c(rng, PROFESSIONS), animal=_c(rng, ANIMALS),
        ))
    return letters


def generate_all_synthetic(seed=42, output_dir='corpus/raw/synthetic'):
    """Generate all synthetic texts and save to files."""
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    generators = [
        ("narrative", generate_narrative_stories, 5000),
        ("dialogue", generate_dialogues, 5000),
        ("qa", generate_qa_pairs, 3000),
        ("descriptive", generate_descriptive_passages, 3000),
        ("instructional", generate_instructional_text, 3000),
        ("argumentative", generate_argumentative_text, 3000),
        ("letter", generate_letters, 3000),
    ]

    all_texts = []
    for format_type, generator, count in generators:
        texts = generator(count, rng)
        all_texts.extend([(t, format_type) for t in texts])

        # Save to file
        filepath = os.path.join(output_dir, f"{format_type}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n\n')

        word_count = sum(len(t.split()) for t in texts)
        print(f"  {format_type}: {count} texts, {word_count:,} words")

    total_words = sum(len(t.split()) for t, _ in all_texts)
    print(f"\n  TOTAL synthetic: {len(all_texts)} texts, {total_words:,} words")
    return all_texts


if __name__ == '__main__':
    generate_all_synthetic()
