#!/usr/bin/env python3
"""Generate structurally diverse synthetic texts across 7 formats.

Formats:
  1. Narrative stories (5,000) — enhanced templates with wider vocabulary
  2. Dialogues (5,000) — turn-taking with varied attribution
  3. Q&A pairs (3,000) — question + explanatory answer
  4. Descriptive passages (3,000) — spatial, temporal, sensory language
  5. Instructional text (3,000) — sequential steps, conditionals
  6. Argumentative text (3,000) — evidence, counterpoints
  7. Letters/correspondence (3,000) — formal and informal registers
"""
import os
import random


# ─── Shared vocabulary pools ────────────────────────────────────────────────

NAMES = [
    "Lily", "Tom", "Sara", "Max", "Emma", "Ben", "Mia", "Jack",
    "Anna", "Leo", "Lucy", "Sam", "Kate", "Dan", "Ella", "Tim",
    "Zoe", "Finn", "Ivy", "Noah", "Ava", "Luke", "Ruby", "Owen",
    "Grace", "Ethan", "Sofia", "Liam", "Chloe", "Oliver", "Aria",
    "James", "Nora", "Henry", "Layla", "Kai", "Maya", "Theo",
    "Amira", "Yusuf", "Fatima", "Ali", "Zahra", "Ibrahim", "Hana",
    "Clara", "Felix", "Rose", "Amir", "Leila", "Hassan",
]

ANIMALS = [
    "cat", "dog", "bird", "rabbit", "fish", "frog", "bear",
    "fox", "deer", "owl", "duck", "mouse", "turtle", "bee",
    "eagle", "dolphin", "wolf", "horse", "butterfly", "sparrow",
    "lamb", "dove", "camel", "lion", "gazelle", "nightingale",
    "hawk", "raven", "salmon", "ant", "spider", "crane",
]

COLORS = [
    "red", "blue", "green", "yellow", "pink", "purple", "orange",
    "white", "black", "brown", "golden", "silver", "crimson",
    "emerald", "sapphire", "ivory", "scarlet", "turquoise",
    "amber", "violet", "indigo", "coral",
]

PLACES = [
    "park", "garden", "forest", "river", "hill", "beach", "meadow",
    "village", "school", "house", "lake", "mountain", "farm", "cave",
    "valley", "desert", "oasis", "temple", "market", "bridge",
    "harbor", "tower", "orchard", "courtyard", "path", "summit",
    "library", "workshop", "cathedral", "lighthouse", "canyon", "island",
]

OBJECTS = [
    "ball", "flower", "stone", "star", "book", "cake", "toy",
    "hat", "box", "cup", "kite", "bell", "ring", "leaf",
    "lamp", "mirror", "pearl", "feather", "scroll", "compass",
    "lantern", "seed", "key", "crown", "blanket", "bread",
    "map", "sword", "shield", "flute", "candle", "basket",
]

FEELINGS = [
    "happy", "sad", "excited", "surprised", "brave", "kind",
    "proud", "curious", "cheerful", "gentle", "friendly", "calm",
    "grateful", "humble", "peaceful", "hopeful", "patient",
    "merciful", "generous", "wise", "faithful", "determined",
    "anxious", "amazed", "thoughtful", "relieved",
]

ACTIONS = [
    "walked", "ran", "jumped", "danced", "played", "sang",
    "laughed", "smiled", "skipped", "climbed", "swam", "flew",
    "whispered", "wondered", "helped", "shared", "carried",
    "discovered", "remembered", "forgave", "prayed", "traveled",
    "gazed", "listened", "searched", "built", "planted", "waited",
]

WEATHER = [
    "sunny", "rainy", "cloudy", "windy", "snowy", "warm",
    "cool", "misty", "stormy", "clear", "bright", "calm",
    "foggy", "humid", "frosty", "breezy",
]

TIMES = [
    "morning", "afternoon", "evening", "night", "dawn", "dusk",
    "sunrise", "sunset", "midnight", "daybreak",
]

VIRTUES = [
    "kindness", "patience", "courage", "honesty", "generosity",
    "forgiveness", "gratitude", "wisdom", "mercy", "compassion",
    "faith", "justice", "humility", "perseverance", "love",
    "integrity", "loyalty", "discipline", "hope", "charity",
]

TOPICS = [
    "friendship", "learning", "nature", "adventure", "family",
    "discovery", "growing up", "seasons", "animals", "music",
    "art", "cooking", "gardening", "building", "traveling",
    "the ocean", "the stars", "forests", "mountains", "rivers",
]

FOODS = [
    "bread", "soup", "rice", "fruit", "cheese", "honey",
    "fish", "vegetables", "stew", "pie", "porridge", "dates",
    "olives", "lentils", "tea", "milk", "apples", "grapes",
]

MATERIALS = [
    "wood", "stone", "clay", "metal", "glass", "silk",
    "cotton", "leather", "paper", "wool", "iron", "copper",
]

SPEECH_VERBS = [
    "said", "replied", "whispered", "exclaimed", "asked",
    "murmured", "shouted", "answered", "sighed", "wondered",
    "declared", "suggested", "agreed", "insisted", "admitted",
]

CONNECTORS = [
    "However", "Moreover", "Furthermore", "In addition",
    "On the other hand", "Nevertheless", "Consequently",
    "Therefore", "Meanwhile", "Similarly", "In contrast",
]


def _fill(template, rng):
    """Fill a template string with random vocabulary."""
    return template.format(
        name=rng.choice(NAMES),
        name2=rng.choice(NAMES),
        name3=rng.choice(NAMES),
        animal=rng.choice(ANIMALS),
        animal2=rng.choice(ANIMALS),
        color=rng.choice(COLORS),
        color2=rng.choice(COLORS),
        place=rng.choice(PLACES),
        place2=rng.choice(PLACES),
        object=rng.choice(OBJECTS),
        object2=rng.choice(OBJECTS),
        feeling=rng.choice(FEELINGS),
        feeling2=rng.choice(FEELINGS),
        action=rng.choice(ACTIONS),
        action2=rng.choice(ACTIONS),
        weather=rng.choice(WEATHER),
        time=rng.choice(TIMES),
        virtue=rng.choice(VIRTUES),
        virtue2=rng.choice(VIRTUES),
        topic=rng.choice(TOPICS),
        food=rng.choice(FOODS),
        food2=rng.choice(FOODS),
        material=rng.choice(MATERIALS),
        connector=rng.choice(CONNECTORS),
        speech_verb=rng.choice(SPEECH_VERBS),
    )


# ─── 1. Narrative stories ──────────────────────────────────────────────────

NARRATIVE_TEMPLATES = [
    "Once upon a time, there was a {feeling} {name} who lived near a {place}. One day, {name} found a {color} {object} by the {place}. {name} picked it up and {action} all the way home. It was the best day ever.",
    "{name} had a little {animal}. The {animal} was very {feeling}. They {action} together in the {place} every day. {name} loved the {animal} so much.",
    "One {time}, {name} woke up and saw a {color} {animal} in the {place}. The {animal} was looking for a {object}. {name} helped the {animal} find it. The {animal} was so {feeling}.",
    "There was a {color} {object} hidden in the {place}. {name} wanted to find it. {name} {action} through the {place} until finding the {object}. {name} felt very {feeling} and showed it to {name2}.",
    "{name} and {name2} were friends. They liked to play in the {place}. One day they found a {color} {object}. They shared it and were both {feeling}. The {weather} sky made everything beautiful.",
    "The {color} {animal} {action} across the {place}. {name} watched and felt {feeling}. Then {name} {action} too. They became good friends and met every {time}.",
    "It was a {weather} day. {name} went to the {place} with a {color} {object}. A {animal} came and wanted to play. {name} and the {animal} {action} together until {time}. Everyone was {feeling}.",
    "{name} had a dream about a {color} {place}. In the dream, a {feeling} {animal} gave {name} a special {object}. When {name} woke up, {name} felt {feeling} and told {name2} about it.",
    "In a little {place}, there lived a {feeling} {animal}. The {animal} liked to collect {color} {object}s. One day, {name} came and they {action} together. It was the beginning of a wonderful friendship.",
    "Once, {name} lost a {color} {object} in the {place}. {name} was {feeling}. But a kind {animal} found it and brought it back. {name} was so {feeling2} that {name} {action} with joy.",
    "The {place} was quiet at {time}. Then {name} came with a {object}. Soon everyone was playing. The {animal} {action} and {name} {action2}. It was a {feeling} day that nobody would forget.",
    "{name} wanted to be brave. {name} {action} to the big {place}. There was a {color} {animal} there. But the {animal} was {feeling}. They became friends and {action2} together every day.",
    "At {time}, the {weather} sky stretched over the {place}. {name} sat quietly, thinking about {virtue}. A {color} {animal} appeared and seemed to understand. Together they {action} toward the {place2}, where {name2} was waiting with a warm {food}.",
    "{name} learned about {virtue} from watching the {animal} in the {place}. The {animal} always {action} when others needed help. This taught {name} to be {feeling}. From that day forward, {name} tried to show {virtue} in everything.",
    "The old {place} held many memories. {name} {action} through it one {time}, finding a {color} {object} left by {name2} long ago. Holding it close, {name} felt {feeling}. Some things are worth remembering forever.",
    "Under the {weather} sky, {name} {action} along the {place}. The wind carried the sound of a {animal} singing. {name} stopped to listen and felt {feeling}. Nature has its own way of teaching {virtue}.",
    "Every {time}, {name} would visit the {place} to feed the {animal}s. One day, a small {color} {animal} followed {name} home. {name2} said they could keep it. The {animal} brought so much joy and taught them about {virtue}.",
    "{name} and {name2} had an argument about the {object}. {name} felt {feeling} but then remembered the importance of {virtue}. {name} {action} back to {name2} and said sorry. {name2} smiled and they were friends again.",
    "A {feeling} traveler named {name} arrived at the {place} one {time}. The people there were {feeling2} and shared their {food} with {name}. {name} learned that {virtue} exists everywhere if you look for it.",
    "Long ago, in a distant {place}, there lived a {feeling} {name} who believed in {virtue}. {name} {action} far and wide, helping those in need. The {animal}s of the {place} became {name}'s companions on this journey of {virtue2}.",
]


def generate_narratives(count, rng):
    return [_fill(rng.choice(NARRATIVE_TEMPLATES), rng) for _ in range(count)]


# ─── 2. Dialogues ──────────────────────────────────────────────────────────

DIALOGUE_TEMPLATES = [
    '"{name}, have you seen the {color} {object}?" {speech_verb} {name2}.\n"{name2}, I think it is in the {place}," {name} replied.\n"Let us go find it together," {name2} {speech_verb}.\nThey {action} to the {place} and found it near the {animal}.',

    '"What do you think about {topic}?" {name} {speech_verb} to {name2}.\n{name2} paused and then replied, "I think {topic} is important because it teaches us {virtue}."\n"I agree," {name} said with a {feeling} smile. "We should talk about this more often."',

    '"I am {feeling}," {name} {speech_verb} quietly.\n"Why?" {name2} asked, sitting down beside {name}.\n"Because I lost my {color} {object} in the {place}," {name} explained.\n"Do not worry," {name2} replied. "I will help you find it. That is what friends are for."',

    '"{name3}, come quickly!" {name} shouted from the {place}.\n{name3} {action} over and asked, "What happened?"\n"Look," {name} whispered, pointing at a {color} {animal}. "Is it not beautiful?"\n"It is," {name3} {speech_verb} in amazement. "I have never seen one so {feeling} before."',

    '"Do you remember the {time} we spent at the {place}?" {name} {speech_verb}.\n{name2} nodded. "Of course. That was when we learned about {virtue}."\n"Those were good times," {name} said. "I wish we could go back."\n"We can always make new memories," {name2} replied with {virtue2}.',

    '"Would you like some {food}?" {name} offered.\n"Yes, please," {name2} {speech_verb}. "I have been {action} all day and I am hungry."\n{name} smiled and served the {food}. "Eat well. You deserve it."\n"Thank you for your {virtue}," {name2} said gratefully.',

    '"I want to learn about {topic}," {name} told {name2}.\n"That is wonderful," {name2} {speech_verb}. "I know someone at the {place} who can teach you."\n"Really? When can we go?" {name} asked eagerly.\n"Tomorrow at {time}," {name2} answered. "I will take you there myself."',

    '"The {weather} sky looks different today," {name} observed.\n"Indeed," {name2} {speech_verb}. "It reminds me of the day we found the {object} in the {place}."\n{name} laughed. "That was quite an adventure."\n"Every day is an adventure if you look at it the right way," {name2} replied.',

    '"Father, what is {virtue}?" the child asked.\n{name} knelt down and said, "It is when you do what is right, even when it is difficult."\n"Like when {name2} shared the {food} with everyone?" the child asked.\n"Exactly like that," {name} {speech_verb} with a {feeling} smile.',

    '"We should help the {animal} in the {place}," {name} {speech_verb}.\n"But how?" {name2} asked.\n"We can bring it {food} and make a shelter from {material}," {name} suggested.\n"That is a {feeling} idea," {name2} agreed. "Let us start right away."',
]


def generate_dialogues(count, rng):
    return [_fill(rng.choice(DIALOGUE_TEMPLATES), rng) for _ in range(count)]


# ─── 3. Q&A pairs ──────────────────────────────────────────────────────────

QA_TEMPLATES = [
    "Question: Why is {virtue} important in daily life?\n\nAnswer: {virtue} is important because it helps us build stronger relationships with others. When we practice {virtue}, people around us feel valued and respected. For example, showing {virtue} to a friend like {name} can make a difficult day much better. {connector}, {virtue} creates a positive environment where everyone can thrive.",

    "Question: What can we learn from observing {animal}s in nature?\n\nAnswer: Observing {animal}s teaches us many valuable lessons. First, {animal}s demonstrate {virtue} through their daily activities. They work together, care for their young, and adapt to changing {weather} conditions. {name} once watched a {animal} at the {place} and learned that even small creatures show great {virtue2}.",

    "Question: How does the {weather} affect people's mood?\n\nAnswer: Weather has a significant impact on how people feel. On {weather} days, many people feel {feeling} and are more likely to spend time outdoors at places like the {place}. In contrast, when the weather turns stormy, people often feel more {feeling2}. {name} noticed that on {weather} days, the {place} was always full of {feeling} people.",

    "Question: What makes the {place} a special location?\n\nAnswer: The {place} is special for several reasons. First, it is home to many {color} {animal}s that cannot be found elsewhere. Second, the {place} has a long history dating back many generations. {name} and {name2} often visit the {place} during {time} to enjoy its {feeling} atmosphere and learn about {topic}.",

    "Question: How can someone practice {virtue} every day?\n\nAnswer: Practicing {virtue} every day starts with small actions. You can begin by being {feeling} toward others, even strangers. For instance, {name} practices {virtue} by helping {name2} with tasks at the {place}. {connector}, sharing {food} with neighbors is another simple way to demonstrate {virtue}. Over time, these small acts become habits.",

    "Question: Why do some people enjoy {topic}?\n\nAnswer: People enjoy {topic} for many reasons. Some find it {feeling}, while others see it as a way to express {virtue}. {name} became interested in {topic} after visiting the {place} one {time}. The experience was so {feeling2} that {name} decided to learn more. As {name2} once said, {topic} helps us understand ourselves better.",

    "Question: What is the difference between a {animal} and a {animal2}?\n\nAnswer: While both the {animal} and the {animal2} are fascinating creatures, they differ in several ways. The {animal} is typically found near the {place}, while the {animal2} prefers the {place2}. The {animal} is known for being {feeling}, whereas the {animal2} is more {feeling2}. {name} studied both animals and found that each has unique strengths.",
]


def generate_qa(count, rng):
    return [_fill(rng.choice(QA_TEMPLATES), rng) for _ in range(count)]


# ─── 4. Descriptive passages ───────────────────────────────────────────────

DESCRIPTIVE_TEMPLATES = [
    "The {place} stretched out before {name} like a painting. {color} light filtered through the trees, casting long shadows across the ground. The air smelled of {food} and fresh earth. A gentle {weather} breeze stirred the leaves, and somewhere in the distance, a {animal} called out to its companion. {name} stood still, breathing deeply, feeling {feeling} for the first time in weeks.",

    "At {time}, the {place} transformed completely. The sky turned {color}, then slowly shifted to {color2}. The water in the river reflected every change, creating a mirror of the heavens above. {name} watched as a {animal} glided across the surface, barely disturbing the stillness. The whole world seemed to hold its breath, waiting for something beautiful to happen.",

    "The old {place} had walls of rough {material} and a floor worn smooth by countless feet. A {color} {object} hung near the entrance, catching the light from a single window. The room smelled of old {material} and {food}. In the corner, a {color2} {object2} sat on a shelf, gathering dust. {name} touched it gently, feeling the weight of memories.",

    "The {weather} {time} brought a special quality to the {place}. Everything seemed brighter, sharper, more alive. The {color} flowers opened their petals wide, and the {animal}s in the {place} sang their most beautiful songs. {name} could feel the warmth on the skin and taste the sweetness of the air. It was a moment of perfect {virtue}.",

    "Walking through the {place}, {name} noticed details that others might miss. The way the {color} moss grew in patterns on the {material} walls. The sound of water dripping somewhere deep inside the {place}. The faint smell of {food} carried on the {weather} air. Each sensation told a story, and {name} listened with a {feeling} heart.",

    "The {place} in {time} was unlike any other time. The {color2} shadows grew long and deep, hiding secrets in their folds. A {animal} perched on a branch of {material}, its eyes bright and watchful. The ground was covered in fallen leaves of every {color} imaginable. {name} walked slowly, each step crunching softly beneath careful feet.",
]


def generate_descriptive(count, rng):
    return [_fill(rng.choice(DESCRIPTIVE_TEMPLATES), rng) for _ in range(count)]


# ─── 5. Instructional text ─────────────────────────────────────────────────

INSTRUCTIONAL_TEMPLATES = [
    "How to Build a {object} from {material}\n\nStep 1: Gather your materials. You will need {material}, a sharp {object2}, and some {color} thread.\nStep 2: Measure carefully. The {object} should be about the size of your hand.\nStep 3: Cut the {material} into the right shape. If the {material} is too thick, try soaking it in water first.\nStep 4: Assemble the pieces. Make sure each joint is secure before moving on.\nStep 5: Let it dry in a {weather} spot. This usually takes one full day.\nNote: If the {material} cracks during Step 3, start over with a fresh piece.",

    "How to Prepare {food}\n\nFirst, wash your hands and clean the {place} where you will be working. Next, gather the following ingredients: fresh {food2}, clean water, and a pinch of salt.\n\nBegin by heating water in a large pot. While waiting, prepare the {food2} by cutting it into small pieces. Once the water is boiling, add the ingredients one at a time.\n\nImportant: Do not rush this process. If the {food} is cooked too quickly, it will lose its flavor. Stir gently and let it simmer until {time}.\n\nServe warm and enjoy with friends like {name} and {name2}.",

    "How to Care for a {animal}\n\nStep 1: Create a comfortable space. Your {animal} needs a {feeling} environment with enough room to move freely.\nStep 2: Provide fresh {food} and water daily. A {animal} eats about twice a day.\nStep 3: Spend time with your {animal} every {time}. They need companionship and attention.\nStep 4: If your {animal} seems {feeling2}, check for common problems. Sometimes they just need more exercise.\nStep 5: Visit the {place} regularly for health checks.\n\nRemember: Every {animal} has a unique personality. Be {feeling} and show {virtue}.",

    "A Guide to Exploring the {place}\n\nBefore you begin, make sure you have a {color} {object} for navigation and plenty of {food} for energy.\n\n1. Start your journey at {time} when the light is best.\n2. Follow the main path until you reach the first clearing.\n3. If you encounter a {animal}, remain calm and give it space.\n4. Look for {color2} markers on the trees — they show the safest route.\n5. Take breaks every hour. Drink water and eat a small amount of {food2}.\n6. If the weather turns {weather}, find shelter immediately.\n\n{name} recommends bringing a friend. The {place} is always better when shared.",

    "How to Practice {virtue} Daily\n\nMorning: Begin each day with a moment of quiet reflection. Think about one way you can show {virtue} today.\n\nMidday: During your activities at the {place}, look for opportunities to help others. Even small gestures matter. If {name} needs help carrying a {object}, offer your assistance.\n\nEvening: Before rest, consider what went well and what could improve. Did you show {virtue2} when it was difficult? Were you {feeling} toward those around you?\n\nWeekly: Set aside time to visit the {place} and spend time with people who inspire {virtue} in you. {name2} says that surrounding yourself with {feeling} people makes all the difference.",
]


def generate_instructional(count, rng):
    return [_fill(rng.choice(INSTRUCTIONAL_TEMPLATES), rng) for _ in range(count)]


# ─── 6. Argumentative text ─────────────────────────────────────────────────

ARGUMENTATIVE_TEMPLATES = [
    "{virtue} is more important than {virtue2} in building a strong community. While both qualities have their place, {virtue} creates the foundation upon which all other values rest. Consider the example of {name}, who lived in a small {place}. Through consistent practice of {virtue}, {name} brought together neighbors who had been divided for years. {connector}, without {virtue}, even the most talented individuals struggle to work together effectively.",

    "Some argue that spending time in the {place} is better than staying in the {place2}. There is strong evidence for this view. The {place} offers fresh air, natural beauty, and the chance to observe {animal}s in their habitat. {name} found that after spending time at the {place}, both mood and health improved significantly. {connector}, the {place2} has its own advantages, including shelter from {weather} conditions and access to {food}. The ideal approach, as {name2} suggests, is to balance both.",

    "Learning about {topic} should be a priority for everyone. First, {topic} develops critical thinking skills that are useful in all areas of life. Second, understanding {topic} helps us appreciate the world around us more deeply. {name} began studying {topic} at a young age and found that it made every experience richer and more meaningful. {connector}, some people argue that {topic} is impractical. However, the evidence suggests otherwise. Communities that value {topic} tend to be more {feeling} and show greater {virtue}.",

    "The {animal} is a better companion than the {animal2}. While both animals have loyal followers, the {animal} offers several distinct advantages. First, the {animal} is known for being {feeling}, which makes it ideal for families. Second, {animal}s require less space than {animal2}s, making them suitable for life in a small {place}. {name} has raised both and confirms that the {animal} is easier to care for. {connector}, {animal2} enthusiasts like {name2} point out that {animal2}s are more {feeling2} and better for outdoor activities at the {place}.",

    "Is {virtue} something we are born with, or is it learned? This question has been debated for generations. Those who believe {virtue} is innate point to young children like {name}, who naturally show {virtue} without being taught. On the other hand, {name2} argues that {virtue} must be practiced and developed over time, much like any skill. The truth likely lies somewhere between these positions. While we may have a natural inclination toward {virtue}, it is through practice and experience at places like the {place} that it truly develops.",

    "{food} is better than {food2} as a daily staple. First, {food} provides more energy and nutrition. {name} switched from {food2} to {food} and noticed improved health within weeks. Second, {food} is easier to prepare and stores longer. {connector}, supporters of {food2} argue that it tastes better and has more cultural significance. While these points have merit, when considering overall health and practicality, {food} remains the superior choice. As {name2} from the {place} often says, good food is the foundation of a {feeling} life.",
]


def generate_argumentative(count, rng):
    return [_fill(rng.choice(ARGUMENTATIVE_TEMPLATES), rng) for _ in range(count)]


# ─── 7. Letters/correspondence ─────────────────────────────────────────────

LETTER_TEMPLATES = [
    "Dear {name2},\n\nI hope this letter finds you well. I am writing to tell you about my recent visit to the {place}. The weather has been {weather} here, and the {color} flowers are in full bloom.\n\nYesterday, I saw the most beautiful {animal} near the {place2}. It reminded me of the time we spent together last {time}. I felt very {feeling} thinking about those memories.\n\nI have been practicing {virtue} as you suggested, and I must say it has made a great difference. The people at the {place} are very {feeling2} and welcoming.\n\nPlease give my regards to {name3}. I hope to visit soon.\n\nWith {virtue2},\n{name}",

    "Dear Friend,\n\nIt has been too long since we last spoke. I wanted to share some news from the {place}.\n\nThe {animal}s have returned for the season, and {name2} has been busy preparing the {place2} for the celebration. We are making {food} and {food2} for everyone.\n\nI have learned something important about {virtue} recently. {name3} taught me that true {virtue} is not about grand gestures but about the small things we do every day. I thought you would appreciate this wisdom.\n\nThe {weather} season is approaching, so please dress warmly. I look forward to your reply.\n\nYour {feeling} friend,\n{name}",

    "To {name2},\n\nI must apologize for not writing sooner. Things at the {place} have been very busy.\n\n{name3} and I have been working on a project together — building a new {object} from {material}. It has been challenging but rewarding. We learned that {virtue} is essential when working with others.\n\nI also wanted to ask your advice about {topic}. You have always been so {feeling} and knowledgeable about these things. When you have a moment, please share your thoughts.\n\nThe {animal}s in the {place} send their greetings. The {color} one you liked has grown quite large.\n\nHoping to hear from you soon,\n{name}",

    "My dear {name2},\n\nThank you for the {color} {object} you sent. It arrived safely and now sits in a place of honor at the {place}.\n\nLife here continues at its usual pace. Every {time}, I walk to the {place2} and think of your words about {virtue}. They have stayed with me through many {weather} days.\n\n{name3} asks about you often. We all miss your {feeling} presence and your wonderful stories about {topic}.\n\nI am sending you some {food} from our garden. I hope it reminds you of home.\n\nWith deep {virtue2} and affection,\n{name}",

    "{name2},\n\nQuick note — I found the {color} {object} you were looking for! It was at the {place}, just where {name3} said it would be.\n\nAlso, the {animal} is doing much better now. The {food} you recommended worked perfectly.\n\nCome visit when you can. The {place} is beautiful this time of year, especially at {time}.\n\n— {name}",

    "Respected {name2},\n\nI write to inform you of developments at the {place}. The community has decided to organize a gathering focused on {topic} and {virtue}.\n\nWe would be honored by your presence. The event will take place during {time}, and {name3} has volunteered to prepare {food} for all attendees.\n\nYour {feeling} guidance on matters of {virtue2} would be invaluable to our discussions. The young people especially would benefit from your wisdom about {topic}.\n\nPlease confirm your availability at your earliest convenience.\n\nWith great respect,\n{name}",
]


def generate_letters(count, rng):
    return [_fill(rng.choice(LETTER_TEMPLATES), rng) for _ in range(count)]


# ─── Main generation function ──────────────────────────────────────────────

FORMAT_SPECS = [
    ("narrative", 5000, generate_narratives),
    ("dialogue", 5000, generate_dialogues),
    ("qa", 3000, generate_qa),
    ("descriptive", 3000, generate_descriptive),
    ("instructional", 3000, generate_instructional),
    ("argumentative", 3000, generate_argumentative),
    ("letter", 3000, generate_letters),
]


def generate_all_synthetic(output_dir='corpus/raw/synthetic', seed=42):
    """Generate all 25,000 diverse synthetic texts and save to files."""
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    total_texts = 0
    total_words = 0

    for format_name, count, generator in FORMAT_SPECS:
        texts = generator(count, rng)
        total_texts += len(texts)

        # Save as one file per format
        filepath = os.path.join(output_dir, f"{format_name}.txt")
        combined = '\n\n---\n\n'.join(texts)
        word_count = len(combined.split())
        total_words += word_count

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(combined)

        results[format_name] = {'count': len(texts), 'words': word_count}
        print(f"  {format_name}: {len(texts)} texts, {word_count:,} words → {filepath}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SYNTHETIC GENERATION SUMMARY")
    print(f"{'='*60}")
    for fmt, info in results.items():
        print(f"  {fmt:<15} {info['count']:>6} texts, {info['words']:>10,} words")
    print(f"  {'TOTAL':<15} {total_texts:>6} texts, {total_words:>10,} words")

    return results


def load_synthetic_texts(input_dir='corpus/raw/synthetic'):
    """Load all synthetic texts as a list of individual texts."""
    texts = []
    for format_name, _, _ in FORMAT_SPECS:
        filepath = os.path.join(input_dir, f"{format_name}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            parts = content.split('\n\n---\n\n')
            texts.extend(parts)
    return texts


if __name__ == '__main__':
    generate_all_synthetic()
