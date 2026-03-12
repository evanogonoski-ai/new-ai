"""
Generate 50,000 longer synthetic texts (~150+ words each) using extended templates.
Uses vocabulary pools and 7 scaled formats. Target: ~11M words total.
"""

import os
import random
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

NAMES = [
    "Alexander", "Beatrice", "Catherine", "Dmitri", "Eleanor", "Frederick",
    "Genevieve", "Heinrich", "Isabella", "Johannes", "Katharina", "Leopold",
    "Magdalena", "Nathaniel", "Ophelia", "Percival", "Quintessa", "Raphael",
    "Seraphina", "Theodore", "Ursula", "Valentina", "Wolfgang", "Xiomara",
    "Yolanda", "Zacharias", "Ambrose", "Brunhilde", "Cornelius", "Dorothea",
    "Edmund", "Francesca", "Gertrude", "Horatio", "Ingrid", "Julius",
    "Lysander", "Miriam", "Nikolai", "Octavia", "Penelope", "Roland",
    "Sylvia", "Tobias", "Vivienne", "Winston", "Artemis", "Bartholomew",
    "Clementine", "Desmond",
]

PLACES = [
    "Vienna", "Prague", "Constantinople", "Alexandria", "Kyoto", "Samarkand",
    "Venice", "Bruges", "Timbuktu", "Hangzhou", "Lisbon", "Marrakech",
    "Edinburgh", "Barcelona", "Dubrovnik", "Cusco", "Varanasi", "Isfahan",
    "Zanzibar", "Heidelberg", "Salzburg", "Florence", "Seville", "Krakow",
    "Fez", "Luang Prabang", "Petra", "Angkor", "Lhasa", "Havana",
    "Cartagena", "Jaipur", "Tallinn", "Bruges", "Ghent", "Ravenna",
    "Siena", "Toledo", "Cordoba", "Granada", "Bratislava", "Ljubljana",
    "Riga", "Vilnius", "Bergen", "Reykjavik", "Valletta", "Kotor",
    "Ohrid", "Tbilisi",
]

ADJECTIVES = [
    "magnificent", "peculiar", "ancient", "ethereal", "somber", "radiant",
    "intricate", "formidable", "delicate", "enigmatic", "resplendent",
    "austere", "luminous", "verdant", "tranquil", "melancholic", "vibrant",
    "profound", "sublime", "majestic", "serene", "turbulent", "graceful",
    "imposing", "mysterious", "elegant", "rugged", "harmonious", "vast",
    "intimate", "solemn", "exquisite", "brooding", "crystalline", "venerable",
    "primordial", "celestial", "ephemeral", "monumental", "pastoral",
]

PROFESSIONS = [
    "astronomer", "cartographer", "apothecary", "blacksmith", "philosopher",
    "architect", "botanist", "weaver", "scribe", "physician", "alchemist",
    "navigator", "sculptor", "clockmaker", "librarian", "diplomat",
    "mathematician", "theologian", "merchant", "composer", "antiquarian",
    "geographer", "chronicler", "metallurgist", "linguist", "herbalist",
    "glassblower", "engraver", "surveyor", "naturalist", "mineralogist",
    "archivist", "curator", "translator", "lexicographer", "ornithologist",
    "entomologist", "ceramicist", "illuminator", "calligrapher",
]

ABSTRACT_NOUNS = [
    "truth", "justice", "beauty", "wisdom", "courage", "harmony", "liberty",
    "compassion", "integrity", "resilience", "curiosity", "serenity",
    "gratitude", "humility", "perseverance", "dignity", "empathy", "virtue",
    "honor", "fortitude", "temperance", "prudence", "devotion", "clarity",
    "patience", "solitude", "transcendence", "consciousness", "existence",
    "perception", "mortality", "eternity", "purpose", "destiny", "paradox",
    "ambiguity", "duality", "absurdity", "authenticity", "melancholy",
]

SEASONS = [
    "spring", "summer", "autumn", "winter", "early spring", "late summer",
    "mid-autumn", "deep winter", "the vernal equinox", "the summer solstice",
    "the autumnal equinox", "the winter solstice", "the harvest season",
    "the planting season", "the monsoon season", "the dry season",
    "the rainy season", "the frost season", "the blooming season",
    "the dormant season",
]

NATURAL_FEATURES = [
    "mountain range", "river valley", "coastal cliff", "dense forest",
    "vast desert", "frozen tundra", "volcanic island", "coral reef",
    "rolling prairie", "deep canyon", "alpine meadow", "mangrove swamp",
    "limestone cave", "glacial lake", "peat bog", "salt flat",
    "sandstone arch", "basalt column", "hot spring", "barrier island",
    "fjord", "estuary", "wetland", "steppe", "taiga", "savanna",
    "rainforest canopy", "tide pool", "sand dune", "obsidian field",
]

EMOTIONS = [
    "joy", "sorrow", "wonder", "nostalgia", "longing", "contentment",
    "apprehension", "elation", "melancholy", "awe", "reverence", "doubt",
    "resolve", "tenderness", "fury", "despair", "hope", "regret",
    "fascination", "bewilderment",
]

MATERIALS = [
    "marble", "granite", "oak", "bronze", "silk", "iron", "glass",
    "porcelain", "leather", "copper", "ivory", "jade", "obsidian",
    "sandstone", "limestone", "mahogany", "cedar", "linen", "wool",
    "parchment",
]

COLORS = [
    "crimson", "azure", "emerald", "amber", "violet", "indigo", "scarlet",
    "cerulean", "ochre", "vermillion", "cobalt", "saffron", "burgundy",
    "teal", "ivory", "charcoal", "pewter", "russet", "sienna", "alabaster",
]

TIME_PERIODS = [
    "the Renaissance", "the Enlightenment", "the Middle Ages",
    "the Classical era", "the Victorian age", "the Romantic period",
    "the Industrial Revolution", "the Age of Exploration",
    "the Bronze Age", "the Iron Age", "antiquity", "the Baroque period",
    "the Reformation", "the Scientific Revolution", "the Gilded Age",
    "the Meiji era", "the Mughal period", "the Tang Dynasty",
    "the Ottoman era", "the Colonial period",
]

SCIENCES = [
    "astronomy", "botany", "chemistry", "geology", "physics",
    "mathematics", "biology", "meteorology", "zoology", "ecology",
    "anatomy", "pharmacology", "mineralogy", "hydrology", "optics",
    "acoustics", "thermodynamics", "electromagnetism", "taxonomy",
    "crystallography",
]

VERBS_PAST = [
    "discovered", "examined", "contemplated", "traversed", "constructed",
    "observed", "documented", "analyzed", "cultivated", "restored",
    "transformed", "illuminated", "deciphered", "catalogued", "preserved",
    "synthesized", "theorized", "pioneered", "revolutionized", "established",
]


def _pick(pool):
    """Pick a random item from a pool."""
    return random.choice(pool)


def _fill_extended(template):
    """Fill a template string by replacing {placeholders} with random vocabulary."""
    mapping = {
        "name": lambda: _pick(NAMES),
        "name2": lambda: _pick(NAMES),
        "name3": lambda: _pick(NAMES),
        "place": lambda: _pick(PLACES),
        "place2": lambda: _pick(PLACES),
        "adj": lambda: _pick(ADJECTIVES),
        "adj2": lambda: _pick(ADJECTIVES),
        "adj3": lambda: _pick(ADJECTIVES),
        "profession": lambda: _pick(PROFESSIONS),
        "profession2": lambda: _pick(PROFESSIONS),
        "abstract": lambda: _pick(ABSTRACT_NOUNS),
        "abstract2": lambda: _pick(ABSTRACT_NOUNS),
        "abstract3": lambda: _pick(ABSTRACT_NOUNS),
        "season": lambda: _pick(SEASONS),
        "feature": lambda: _pick(NATURAL_FEATURES),
        "feature2": lambda: _pick(NATURAL_FEATURES),
        "emotion": lambda: _pick(EMOTIONS),
        "emotion2": lambda: _pick(EMOTIONS),
        "material": lambda: _pick(MATERIALS),
        "material2": lambda: _pick(MATERIALS),
        "color": lambda: _pick(COLORS),
        "color2": lambda: _pick(COLORS),
        "period": lambda: _pick(TIME_PERIODS),
        "science": lambda: _pick(SCIENCES),
        "verb_past": lambda: _pick(VERBS_PAST),
        "verb_past2": lambda: _pick(VERBS_PAST),
    }
    result = template
    for key, fn in mapping.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, fn(), 1)
    return result


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

NARRATIVE_EXTENDED_TEMPLATES = [
    (
        "The {adj} city of {place} had always been known for its {adj2} architecture "
        "and the quiet devotion of its inhabitants. Among them, {name}, a {profession} "
        "of considerable renown, spent long hours in a workshop overlooking the central "
        "square. The scent of {material} and old parchment filled the air as {name} "
        "worked tirelessly on a commission that had occupied months of careful labor. "
        "Outside, the streets of {place} hummed with the rhythms of daily life. "
        "Market vendors called out their wares, children chased pigeons across the "
        "cobblestones, and elderly couples sat beneath the shade of chestnut trees, "
        "speaking in low voices about the changes they had witnessed over the decades. "
        "{name2}, a fellow {profession2}, often visited in the evenings, bringing news "
        "from {place2} and engaging in long conversations about {abstract} and the "
        "nature of {abstract2}. Their friendship, forged during {period}, had weathered "
        "many storms, and both understood that {abstract3} was not something easily won "
        "but rather cultivated through years of honest effort and mutual respect. "
        "As {season} settled over the rooftops, {name} reflected on how much had changed "
        "since those early days, and how much remained beautifully, stubbornly the same."
    ),
    (
        "It was during {season} that {name} first arrived in {place}, carrying nothing "
        "but a leather satchel and a letter of introduction to the renowned {profession} "
        "{name2}. The journey from {place2} had been long and arduous, crossing "
        "{feature} after {feature2}, but the sight of the city rising from the morning "
        "mist filled {name} with a sense of {emotion} that would never fully fade. "
        "{name2} received the young traveler with characteristic warmth, offering lodging "
        "in an {adj} room above the workshop. In the weeks that followed, {name} learned "
        "the fundamentals of {science} under {name2}'s patient guidance, discovering "
        "that the discipline required not only technical skill but also a deep appreciation "
        "for {abstract}. The other apprentices were a diverse group: {name3}, who had come "
        "from distant {place2}, brought a {adj2} perspective that challenged conventional "
        "thinking. Together they spent evenings debating questions of {abstract2} and "
        "{abstract3}, their voices rising and falling in the candlelit study. {name2} "
        "would listen with a {adj3} smile, occasionally offering a remark that reframed "
        "the entire discussion. Years later, {name} would remember these formative months "
        "as the period when {emotion2} first gave way to genuine understanding."
    ),
    (
        "{name} had spent the better part of a decade in {place}, working as a {profession} "
        "and slowly building a reputation for {adj} craftsmanship. The workshop on the "
        "eastern side of the old quarter was modest but well-equipped, its walls lined with "
        "tools of {material} and shelves of reference texts accumulated over the years. "
        "Each morning began the same way: a walk along the {feature} at the edge of town, "
        "where the air was fresh and the mind could settle before the demands of the day. "
        "It was on one such morning, during {season}, that {name} encountered {name2}, "
        "a {profession2} from {place2} who had traveled extensively and carried with "
        "them stories of {adj2} discoveries and {adj3} failures in equal measure. "
        "Their conversation that day lasted well past noon, ranging from the practical "
        "details of {science} to the broader questions of {abstract} that had occupied "
        "thinkers since {period}. {name2} spoke of a theory that linked {abstract2} "
        "to the observable patterns in nature, an idea that {name} found both compelling "
        "and unsettling. Over the following weeks, as {name2} remained in {place} to "
        "complete research, the two met regularly, and what began as intellectual "
        "curiosity deepened into a collaboration that would eventually produce work "
        "of lasting significance. The {emotion} they shared in those early discussions "
        "never entirely disappeared, even as the work grew more rigorous and demanding."
    ),
    (
        "The household of {name} in {place} was known throughout the district for its "
        "{adj} hospitality and the lively gatherings that took place every {season}. "
        "As a respected {profession}, {name} had long maintained that {abstract} was "
        "best understood not through solitary reflection but through the exchange of "
        "ideas among thoughtful people. The drawing room, with its {color} curtains "
        "and furniture of polished {material}, had witnessed debates on every subject "
        "from {science} to the meaning of {abstract2}. {name2}, a {profession2} who "
        "had recently returned from an extended stay in {place2}, was among the most "
        "frequent guests. Having {verb_past} several important texts during the journey, "
        "{name2} brought a {adj2} energy to the conversations that others found both "
        "stimulating and challenging. One evening, the discussion turned to the question "
        "of {abstract3} and whether it could be reconciled with the demands of modern "
        "life. {name3}, a younger {profession2} with strong opinions and a {adj3} "
        "manner of expression, argued passionately that the old frameworks had outlived "
        "their usefulness. {name} listened carefully, weighing each point with the "
        "patience of someone who had seen many intellectual fashions come and go. When "
        "at last {name} spoke, the room fell quiet, for everyone knew that what followed "
        "would be worth hearing. The words were measured but carried the weight of deep "
        "conviction, and even {name3} had to concede that there was much still to learn "
        "from the traditions of the past."
    ),
    (
        "In the {adj} harbor of {place}, where trading vessels arrived from as far as "
        "{place2}, {name} worked as a {profession} whose skills were sought by captains "
        "and merchants alike. The waterfront was a world unto itself, alive with the "
        "sounds of {material} being loaded and unloaded, the shouts of dockworkers, and "
        "the ever-present cry of gulls wheeling overhead. {name} had learned the trade "
        "during {period}, apprenticed to a stern but fair master who believed that "
        "{abstract} was the foundation of all good work. Now, years later, {name} "
        "trained others with the same exacting standards, though always tempered by "
        "a {adj2} understanding of human limitation. Among the current apprentices, "
        "{name2} showed the most promise. Young and full of {emotion}, {name2} had "
        "arrived from a small village near the {feature} with little more than raw "
        "talent and determination. Under {name}'s guidance, {name2} began to understand "
        "that excellence in {science} required not just skill but also {abstract2} and "
        "a willingness to accept failure as part of the process. The two often worked "
        "late into the evening, the lamp casting {color} shadows across the workbench, "
        "discussing not only technical matters but also the larger questions that haunted "
        "anyone who thought deeply about {abstract3}. It was during one such evening, "
        "with a {adj3} wind blowing in from the sea, that {name2} first articulated the "
        "idea that would eventually change how the entire guild approached its craft."
    ),
]

DIALOGUE_EXTENDED_TEMPLATES = [
    (
        'The study was dimly lit by a single lamp of {color} glass. {name}, a '
        "{adj} {profession}, sat across from {name2}, who had traveled from {place} "
        "specifically for this conversation.\n\n"
        '"{name2}, I must confess that your letter surprised me," {name} began, '
        "setting aside a volume on {science}. "
        '"The question of {abstract} is not one I expected to revisit at this stage '
        'of my career."\n\n'
        '"{name2} leaned forward. "And yet it is precisely because of your experience '
        "that I seek your counsel. In {place2}, the prevailing view is that {abstract2} "
        "can be fully explained through empirical observation alone. But I have come to "
        'doubt this."\n\n'
        '"You doubt empiricism?" {name} raised an eyebrow. "That is a {adj2} position '
        'for someone trained in {science}."\n\n'
        '"Not empiricism itself," {name2} clarified, "but the assumption that it '
        "excludes all other ways of knowing. Consider the {feature} near my home. "
        "I can measure its depth, catalog its species, map its currents. But none of "
        "that captures the {emotion} one feels standing at its edge during {season}. "
        'Is that experience not also a form of knowledge?"\n\n'
        "{name} was quiet for a long moment. The clock on the mantle ticked steadily. "
        '"What you describe," {name} said at last, "is something my old teacher in '
        "{place} used to call the gap between measurement and meaning. She was a "
        "{profession2}, brilliant and {adj3}, and she argued that {abstract3} required "
        "us to hold both in tension rather than choosing one over the other. I confess "
        'I did not fully appreciate her point at the time."\n\n'
        '"{name2} nodded slowly. "Then perhaps we are both students still."'
    ),
    (
        "The market square of {place} was crowded with the bustle of {season}. "
        "{name}, a {profession} known for {adj} opinions, stood beside a fountain "
        "of {material}, deep in discussion with {name2}.\n\n"
        '"You cannot seriously believe," {name2} said, shaking their head, "that '
        "the developments in {science} will have no effect on how we understand "
        '{abstract}."\n\n'
        '"{name} smiled. "I believe they will have an effect, but not the one you '
        "expect. Every generation thinks its discoveries are revolutionary. During "
        "{period}, people said exactly the same things about their innovations, and "
        'yet the fundamental questions remained."\n\n'
        '"That is because the questions are eternal," interjected {name3}, a young '
        '{profession2} who had been listening from a nearby bench. "The answers, '
        'however, must evolve."\n\n'
        '{name} turned to regard {name3} with a {adj2} expression. "And what answers '
        'do you propose?"\n\n'
        '"{name3} stood and approached. "Consider the {feature} that separates {place} '
        "from {place2}. For centuries it was seen as a barrier. Now we understand it as "
        "an ecosystem, a network of relationships more {adj3} than any single mind can "
        "comprehend. That shift in perspective did not invalidate the old knowledge. It "
        'enriched it."\n\n'
        '{name2} nodded. "That is precisely my point. {abstract2} is not a fixed thing. '
        'It grows as we grow."\n\n'
        "{name} looked from one to the other, then laughed softly. "
        '"Very well. You have convinced me that there is more to discuss. Shall we '
        'continue over dinner? I know a place that serves excellent food and tolerates '
        'long arguments."'
    ),
    (
        "The library of the old university in {place} was nearly empty, as it often "
        "was during {season}. {name} and {name2} sat at opposite ends of a long "
        "table of {material}, surrounded by towers of books on {science}.\n\n"
        '"{name2} broke the silence first. "I have read your latest paper three times, '
        "and I still cannot agree with your central claim about {abstract}. The evidence "
        'you cite from {place2} is compelling, but your interpretation troubles me."\n\n'
        '"{name} set down a pen. "Troubles you how?"\n\n'
        '"You argue that {abstract2} arises naturally from the conditions you describe. '
        "But what about the role of individual choice? The {profession} who {verb_past} "
        "those early records in {place2} was making deliberate decisions about what to "
        'include and what to omit."\n\n'
        '"{name} considered this. "You raise a valid point. But consider that the '
        "{profession} was operating within a framework shaped by {period}. Individual "
        "choice is never truly free of context. The {adj} traditions of that era "
        "constrained what was even thinkable, let alone expressible. We must account "
        'for that."\n\n'
        '"{name3} appeared at the doorway, arms full of manuscripts. "Am I interrupting?"\n\n'
        '"On the contrary," {name2} said. "We need a fresh perspective. {name3}, what '
        "is your view on the relationship between {abstract3} and historical "
        'circumstance?"\n\n'
        "{name3}, a {adj2} {profession2} with a reputation for unconventional thinking, "
        "set down the manuscripts and pulled up a chair. What followed was a conversation "
        "that lasted well into the evening, as the light through the {color} windows "
        "shifted and faded, and the three scholars found themselves, despite their "
        "differences, moving toward something that resembled, if not agreement, then "
        "at least a deeper mutual understanding."
    ),
    (
        "The observatory at {place} offered a {adj} view of the night sky, unobstructed "
        "by the lights of the lower city. {name}, a senior {profession}, had invited "
        "{name2} to observe a celestial event that occurred only once every few decades.\n\n"
        '"There," {name} said, adjusting the telescope. "Do you see it?"\n\n'
        "{name2} peered through the eyepiece and drew a sharp breath. "
        '"It is extraordinary. The {color} light is far more vivid than the texts '
        'described."\n\n'
        '"{name} nodded. "Text can only approximate experience. That is why I brought '
        "you here rather than simply sending a report. Some aspects of {science} must "
        'be witnessed to be understood."\n\n'
        "They stood in silence for several minutes, watching the sky. The {adj2} air "
        "of {season} carried the scent of the {feature} that lay beyond the city walls.\n\n"
        '"It makes one think about {abstract}," {name2} said quietly. "About how small '
        'our concerns seem against something like this."\n\n'
        '"{name} smiled. "My teacher, {name3}, used to say that {abstract2} was the '
        "natural response of any thinking being confronted with the scale of the "
        "universe. Not despair, as some might expect, but a kind of {adj3} "
        '{emotion}."\n\n'
        '"I think I understand that now," {name2} replied. "In {place2}, where I grew '
        "up, we had a saying that {abstract3} is the beginning of all real knowledge. "
        'I always thought it was just a platitude. Now I am not so sure."\n\n'
        "The telescope hummed faintly as {name} redirected it. "
        '"There is still much to see tonight. And I find that the best conversations '
        'happen when one keeps looking upward."'
    ),
]

ESSAY_EXTENDED_TEMPLATES = [
    (
        "The relationship between {abstract} and {abstract2} has been a subject of "
        "sustained inquiry since {period}, when thinkers in {place} first began to "
        "articulate the tension between individual aspiration and collective obligation. "
        "This tension, far from being resolved by subsequent developments in {science} "
        "and philosophy, has only deepened as societies have grown more complex and "
        "interconnected. The {adj} frameworks developed during {period} provided a "
        "vocabulary for discussing these issues, but the underlying questions remain as "
        "pressing as ever. What does it mean to pursue {abstract} in a world that "
        "increasingly demands {abstract2}? How should institutions balance the needs of "
        "the individual against the requirements of the community? These are not merely "
        "academic questions. They shape policy, influence education, and determine the "
        "character of public life. The {profession} who works in isolation and the "
        "{profession2} who works within a large organization face different versions "
        "of the same dilemma, and neither can afford to ignore it. Recent scholarship, "
        "drawing on evidence from {place2} and elsewhere, suggests that the most "
        "productive approach is neither pure individualism nor uncritical collectivism, "
        "but rather a {adj2} synthesis that acknowledges the legitimate claims of both. "
        "This synthesis, sometimes called the principle of {abstract3}, holds that "
        "genuine {abstract} is only possible within a framework of mutual respect and "
        "shared commitment to the common good. While this view has its critics, its "
        "growing influence in academic and professional circles indicates that the old "
        "dichotomies may finally be giving way to more nuanced understanding."
    ),
    (
        "The study of {science} has undergone a remarkable transformation over the past "
        "century, shifting from a discipline concerned primarily with classification and "
        "description to one that seeks to understand underlying mechanisms and principles. "
        "This shift, which began during {period} in institutions across {place} and "
        "{place2}, was driven by several factors: the development of new instruments, "
        "the increasing availability of data, and a growing recognition that the {adj} "
        "categories inherited from earlier eras were insufficient for capturing the "
        "complexity of the natural world. The {profession} of today operates in a "
        "landscape that would be almost unrecognizable to predecessors working just a "
        "few generations ago. Where once the primary task was to observe and catalogue, "
        "the modern practitioner must also model, predict, and test hypotheses against "
        "an ever-expanding body of evidence. This has brought {science} into closer "
        "dialogue with adjacent fields, including {abstract} and applied {abstract2}, "
        "creating productive intersections but also new tensions. The question of how "
        "to maintain disciplinary rigor while remaining open to interdisciplinary "
        "insight is one that occupies many leading researchers. Some argue for a return "
        "to {adj2} first principles; others advocate for a more {adj3} approach that "
        "embraces complexity and uncertainty. What is clear is that the field cannot "
        "stand still. The challenges posed by contemporary problems, from environmental "
        "change to the ethical implications of new technologies, demand that the study "
        "of {science} continue to evolve, drawing on the best of its traditions while "
        "remaining responsive to the needs of the present."
    ),
    (
        "Education has long been understood as the primary vehicle through which "
        "societies transmit {abstract} from one generation to the next. Yet the methods "
        "by which this transmission occurs have varied enormously across cultures and "
        "historical periods. In {place} during {period}, the dominant model emphasized "
        "apprenticeship and direct instruction: a young {profession} would learn by "
        "working alongside an experienced master, absorbing not only technical skills "
        "but also the values and habits of mind that defined the craft. This model, "
        "while effective in many respects, had significant limitations. Access was "
        "restricted, curricula were narrow, and the pace of learning was dictated by "
        "the master rather than the student. The rise of formal institutions in {place2} "
        "and elsewhere introduced new possibilities but also new challenges. Large "
        "classrooms required standardized methods, and the {adj} intimacy of the "
        "apprenticeship gave way to more impersonal forms of instruction. The tension "
        "between {abstract2} and efficiency has shaped educational debate ever since. "
        "Contemporary reformers, influenced by developments in {science} and the "
        "practical insights of experienced {profession2}s, have proposed a variety of "
        "approaches intended to combine the strengths of both models. Some emphasize "
        "project-based learning, in which students engage with real-world problems that "
        "require the application of multiple skills. Others advocate for personalized "
        "instruction, made increasingly feasible by new technologies. What remains "
        "constant is the conviction that {abstract3} is not merely the accumulation of "
        "facts but the development of {adj2} judgment, the capacity to think clearly "
        "under conditions of uncertainty, and the {adj3} commitment to lifelong inquiry."
    ),
    (
        "The concept of {abstract} occupies a central position in the intellectual "
        "history of {place} and, by extension, in the broader tradition of Western "
        "thought. From the earliest philosophical texts produced during {period} to "
        "the most recent contributions of contemporary scholars, the attempt to define, "
        "defend, and critique {abstract} has generated an enormous body of literature. "
        "At its core, the debate revolves around a deceptively simple question: what "
        "constitutes a good life? The {adj} answer provided by the ancients, which "
        "emphasized {abstract2} and the cultivation of character, held sway for "
        "centuries but was eventually challenged by thinkers who argued that external "
        "conditions, not just internal dispositions, were decisive. The {profession} "
        "working in {place2} during the height of this debate {verb_past} a series of "
        "texts that sought to reconcile these perspectives, arguing that {abstract3} "
        "required both personal virtue and a just social order. This synthesis, while "
        "influential, did not settle the matter. Subsequent generations of scholars "
        "have continued to refine and contest it, drawing on evidence from {science}, "
        "comparative cultural studies, and the practical experience of {profession2}s "
        "working in diverse settings. The {adj2} insight that emerges from this long "
        "history is that {abstract} is not a destination but a process, not a fixed "
        "state but an ongoing negotiation between the individual and the world. "
        "Understanding this may not resolve all disagreements, but it provides a "
        "foundation for more {adj3} and productive conversation."
    ),
]

NATURE_EXTENDED_TEMPLATES = [
    (
        "The {feature} that stretches beyond {place} is a landscape of extraordinary "
        "diversity, shaped over millennia by the forces of wind, water, and geological "
        "upheaval. During {season}, the region takes on a {adj} character that draws "
        "visitors from as far as {place2}, each seeking something different: solitude, "
        "inspiration, or simply the chance to stand in the presence of something vast "
        "and indifferent to human concern. The flora of the area includes species found "
        "nowhere else, their {color} blossoms emerging in patterns that have puzzled "
        "the study of {science} for generations. Near the water's edge, where the "
        "{feature2} meets the lowland plain, the ecosystem transitions abruptly, "
        "creating a mosaic of habitats that supports an astonishing variety of life. "
        "Birds nest in the {adj2} canopy overhead, their calls echoing across the "
        "valley in the early morning hours. Below, the undergrowth teems with insects "
        "and small mammals, each occupying a niche refined by countless generations "
        "of adaptation. The {material} outcroppings that punctuate the landscape bear "
        "the marks of ancient processes: layers of sediment compressed and folded by "
        "tectonic forces, then exposed by erosion over spans of time that dwarf all "
        "human history. To walk through this place during {season} is to experience "
        "a profound sense of {emotion}, a recognition that the world operates on "
        "scales and timelines that exceed our capacity for comprehension. And yet there "
        "is also {emotion2}, for the beauty of the {adj3} light filtering through the "
        "canopy, the clarity of the water, and the simple persistence of life in all "
        "its forms speak to something deeply hopeful."
    ),
    (
        "The coastline near {place} has been shaped by centuries of tidal action, "
        "creating a series of {adj} formations that are unique in the region. The "
        "{material} cliffs rise sharply from the water, their surfaces worn into "
        "{adj2} patterns by the relentless motion of the sea. During {season}, the "
        "light strikes these formations at angles that produce {color} reflections, "
        "and the entire shoreline seems to glow with an almost otherworldly quality. "
        "Fishermen from {place} have worked these waters for generations, their "
        "knowledge of currents and weather patterns passed down through oral tradition "
        "long before the formal study of {science} provided systematic explanations. "
        "The {feature} that lies offshore is home to a complex web of marine life, "
        "from the smallest invertebrates to the larger predators that patrol the deeper "
        "channels. The health of this ecosystem depends on a delicate balance of "
        "temperature, salinity, and nutrient flow, a balance that researchers from "
        "{place2} have monitored with increasing concern in recent years. Changes in "
        "the {season} patterns have introduced new stresses, and species that once "
        "thrived in the {adj3} waters are now found less frequently. The local community, "
        "whose livelihood depends on the continued vitality of the coast, has responded "
        "with a combination of traditional stewardship and modern conservation practices. "
        "Walking along the shore at low tide, one can still find evidence of the region's "
        "extraordinary biodiversity: tide pools teeming with life, nesting sites carved "
        "into the cliff face, and the occasional glimpse of something rare and beautiful "
        "moving just beneath the surface."
    ),
    (
        "Deep within the interior of the landmass that extends south of {place}, the "
        "{feature} presents one of the most {adj} landscapes on Earth. Stretching for "
        "hundreds of kilometers, it encompasses terrain that ranges from arid scrubland "
        "to lush {adj2} forest, with transitional zones that support unique communities "
        "of plants and animals. The geological history of the region is written in the "
        "{material} formations that jut from the earth at dramatic angles, each layer "
        "representing an era stretching back millions of years. Researchers in {science} "
        "from the university in {place2} have spent decades mapping these formations, "
        "finding evidence of ancient seas, volcanic eruptions, and climatic shifts that "
        "transformed the landscape repeatedly. During {season}, the {feature2} that runs "
        "through the center of the region swells with meltwater, creating temporary "
        "wetlands that attract migratory species from across the continent. The {color} "
        "hues of the vegetation during this period are striking, a palette that has "
        "inspired artists and poets since {period}. Local communities have long "
        "recognized the importance of this seasonal cycle, organizing their agricultural "
        "and cultural calendars around the rhythms of the natural world. The sense of "
        "{emotion} that pervades the landscape during {season} is not merely aesthetic "
        "but rooted in a deep practical understanding of interdependence. Every element, "
        "from the {adj3} soil to the migrating flocks overhead, plays a role in a system "
        "whose complexity continues to reveal new dimensions with each passing year."
    ),
]

PHILOSOPHICAL_EXTENDED_TEMPLATES = [
    (
        "The question of {abstract} has occupied thinkers in {place} and beyond since "
        "at least {period}, and the debate shows no sign of resolution. This is not, "
        "as some have argued, because the question is meaningless, but rather because "
        "it touches on aspects of human experience that resist easy categorization. "
        "When the {profession} {name} wrote about {abstract} in the context of "
        "{science}, the argument was that empirical investigation could illuminate "
        "but never fully resolve the deeper issues at stake. The {adj} insight at the "
        "heart of this position is that some forms of knowledge are irreducibly personal, "
        "grounded in experience rather than observation alone. This does not make them "
        "less valid, only differently accessible. {name2}, writing a generation later "
        "from the very different vantage point of {place2}, offered a {adj2} response: "
        "that the personal and the empirical were not separate domains but aspects of "
        "a single, unified reality. The tension between these views has been enormously "
        "productive, generating a rich body of work that continues to inform contemporary "
        "discussions of {abstract2}. What is perhaps most striking about this intellectual "
        "tradition is its refusal to settle for simple answers. The easy path would be to "
        "declare one side or the other correct and move on. But the most thoughtful "
        "contributors, from {name} to the present day, have recognized that {abstract3} "
        "demands a {adj3} engagement with complexity, a willingness to hold contradictory "
        "ideas in mind without rushing to resolve them. This, more than any specific "
        "conclusion, may be the tradition's most valuable legacy."
    ),
    (
        "If one were to ask what single concept has generated the most sustained "
        "philosophical debate in the history of human thought, a strong case could be "
        "made for {abstract}. From the marketplaces of {place} during {period} to the "
        "seminar rooms of modern universities, the question of what {abstract} truly "
        "means and how it relates to {abstract2} has provoked endless argument, "
        "reflection, and revision. The {adj} tradition holds that {abstract} is "
        "something objective, existing independently of human perception, discoverable "
        "through reason and careful analysis. Against this, a competing school, "
        "particularly influential in {place2}, maintains that {abstract} is always "
        "situated, always shaped by the cultural and historical circumstances of the "
        "person who seeks it. The {profession} {name}, whose writings remain required "
        "reading for students of {science}, attempted to bridge these positions by "
        "arguing that while {abstract} in its fullest sense might be beyond human "
        "grasp, the pursuit of it was itself the highest form of {abstract2}. This "
        "{adj2} formulation did not satisfy everyone, of course. {name2}, working "
        "independently on similar questions, argued that the very distinction between "
        "objective and subjective was a product of a specific intellectual tradition "
        "and could not be taken for granted. The debate continues, enriched by each "
        "new generation of thinkers who bring fresh perspectives and, inevitably, "
        "new questions. Perhaps the most {adj3} lesson to emerge from this long "
        "conversation is that the value of philosophy lies not in the answers it "
        "provides but in the quality of attention it brings to the questions themselves."
    ),
    (
        "The experience of {emotion} is universal, yet its interpretation varies "
        "enormously across cultures and historical periods. In {place} during {period}, "
        "{emotion} was understood primarily as a response to the {adj} order of the "
        "natural world, a recognition of forces and patterns that exceeded human "
        "comprehension. The {profession} {name} wrote extensively about this experience, "
        "connecting it to the study of {science} and arguing that genuine understanding "
        "began not with certainty but with {emotion2}. This view, which found support "
        "among scholars in {place2} and elsewhere, had {adj2} implications for how "
        "knowledge was pursued and valued. If {emotion} was the starting point of "
        "inquiry, then the confident assertion of complete understanding was not a "
        "sign of wisdom but of its absence. {name2}, a {profession2} of the next "
        "generation, took this argument further, suggesting that {abstract} itself "
        "was best understood not as a destination but as a mode of engagement with the "
        "world. To possess {abstract} was not to have all the answers but to ask better "
        "questions, to approach each new problem with the {adj3} awareness that one's "
        "current understanding was necessarily incomplete. The practical consequences "
        "of this view were significant. Institutions in {place} reformed their methods "
        "of instruction, placing greater emphasis on critical thinking and less on rote "
        "memorization. The study of {abstract2} was reframed as an ongoing conversation "
        "rather than a fixed body of doctrine. And the relationship between {abstract3} "
        "and daily life was reconceived as something dynamic and evolving, shaped by "
        "experience as much as by theory."
    ),
]

HISTORICAL_EXTENDED_TEMPLATES = [
    (
        "The history of {place} during {period} is marked by a series of transformations "
        "that reshaped not only the city itself but also the broader region in which it "
        "was situated. The catalyst for many of these changes was the arrival of "
        "{name}, a {profession} whose {adj} vision for the future attracted both fervent "
        "supporters and determined opponents. Prior to {name}'s arrival, {place} had been "
        "a relatively quiet center of trade and learning, its reputation built on the "
        "work of {profession2}s who had established a tradition of {adj2} scholarship "
        "stretching back several generations. The {material} buildings of the old quarter "
        "housed workshops, libraries, and meeting halls where the principles of {science} "
        "and {abstract} were debated with equal vigor. But the political landscape was "
        "shifting. Developments in {place2} had created new pressures, and the old "
        "equilibrium could no longer hold. {name}'s contribution was to articulate a "
        "path forward that honored the best of the existing tradition while acknowledging "
        "the need for change. This was not achieved without conflict. {name2}, who led "
        "the conservative faction, argued passionately that {abstract2} demanded fidelity "
        "to established practices. The resulting debate, which played out in public "
        "forums, private correspondence, and the pages of the city's leading journal, "
        "lasted the better part of a decade. When it finally resolved, the settlement "
        "reflected a {adj3} compromise: the old institutions were preserved but their "
        "mandates were expanded, and new positions were created to ensure that fresh "
        "perspectives would always have a voice."
    ),
    (
        "The trade routes that connected {place} to {place2} during {period} were "
        "more than commercial arteries; they were channels through which ideas, "
        "technologies, and cultural practices flowed in both directions. The {adj} "
        "caravans that traversed the {feature} between the two cities carried not "
        "only goods of {material} and spices but also manuscripts, instruments, and "
        "the personal expertise of traveling {profession}s whose knowledge was sought "
        "by rulers and scholars alike. {name}, who made the journey several times "
        "during a long career, {verb_past} detailed accounts of the communities "
        "encountered along the way, each with its own distinctive approach to "
        "{science} and its own understanding of {abstract}. These accounts, rediscovered "
        "by researchers in {place2} centuries later, provide an invaluable record of "
        "a world that might otherwise have been lost. {name2}, a {profession2} who "
        "edited and published the accounts during a later period, noted that what "
        "struck {name2} most was the {adj2} diversity of thought that coexisted "
        "along a single route. The assumption that cultural exchange leads inevitably "
        "to homogeneity was, {name2} argued, contradicted by the evidence at every "
        "turn. Instead, exposure to different traditions seemed to sharpen rather than "
        "erode local identities, as communities found new ways to articulate what "
        "made them distinctive. The {adj3} legacy of these routes is visible even "
        "today, in the architectural styles, linguistic patterns, and intellectual "
        "traditions that bear the unmistakable imprint of centuries of creative exchange."
    ),
    (
        "In the archives of {place}, a collection of letters written during {period} "
        "offers a {adj} window into the daily lives of ordinary people navigating "
        "extraordinary circumstances. The correspondence, primarily between {name}, "
        "a {profession} stationed in the city, and {name2}, a {profession2} living "
        "in {place2}, covers a span of several years and touches on subjects ranging "
        "from the practical details of household management to the most profound "
        "questions of {abstract} and {abstract2}. What emerges from these letters is "
        "a portrait of two individuals attempting to maintain intellectual and emotional "
        "connection across distance, at a time when communication was slow and the "
        "outcome of events deeply uncertain. {name} writes with a {adj2} precision "
        "that reflects professional training in {science}, describing the {feature} "
        "visible from the city walls, the changing quality of light during {season}, "
        "and the mood of the populace as news arrived from distant fronts. {name2}'s "
        "responses are warmer, more personal, filled with observations about the "
        "{adj3} beauty of the countryside and reflections on what {abstract3} might "
        "mean in a world that seemed determined to test every principle that decent "
        "people held dear. Together, the letters form a document of remarkable "
        "depth, one that historians have described as essential reading for anyone "
        "seeking to understand how individuals preserved their humanity in the face "
        "of forces that threatened to overwhelm it."
    ),
]

TECHNICAL_EXTENDED_TEMPLATES = [
    (
        "The process of extracting and refining {material} has evolved considerably "
        "since its origins in {place} during {period}. Early practitioners, often "
        "working as {profession}s under the patronage of local authorities, developed "
        "techniques that were {adj} in their ingenuity but limited by the available "
        "technology. The basic principle involved heating raw ore to extreme "
        "temperatures, a process that required both specialized furnaces and an "
        "intimate understanding of the material's physical properties. In {place2}, "
        "a parallel tradition emerged that approached the problem from a different "
        "angle, emphasizing chemical rather than thermal methods. The {profession2} "
        "{name}, whose work during this period was particularly influential, "
        "{verb_past} a series of experiments that demonstrated the viability of "
        "combining both approaches. The resulting technique, which became the standard "
        "in the field of {science}, achieved significantly higher yields while reducing "
        "the environmental impact that had concerned observers for decades. Modern "
        "practitioners build on this foundation but have introduced additional "
        "refinements. The use of {adj2} analytical instruments allows for precise "
        "monitoring of the process at every stage, from initial extraction through "
        "final purification. Quality control, once dependent on the subjective "
        "judgment of experienced workers, is now supported by quantitative methods "
        "that can detect impurities at levels previously unimaginable. The {adj3} "
        "challenge that remains is scaling these techniques to meet growing demand "
        "without compromising the standards of {abstract} that define professional "
        "practice. Research groups in {place} and {place2} are currently exploring "
        "several promising avenues, including the application of principles from "
        "{science} to optimize the most resource-intensive stages of the process."
    ),
    (
        "The construction of large-scale {material} structures in {place} during "
        "{period} represented a significant advance in the field of {science} and "
        "engineering. The {adj} techniques employed by builders, many of whom were "
        "trained as {profession}s, required a sophisticated understanding of load "
        "distribution, material fatigue, and the effects of environmental exposure "
        "over extended timescales. The primary challenge was to create structures "
        "that could withstand the {adj2} conditions of {season} while maintaining "
        "their aesthetic qualities throughout the year. The solution, developed over "
        "several generations of incremental improvement, involved a combination of "
        "structural innovation and careful material selection. {name}, whose treatise "
        "on the subject remains a standard reference, {verb_past} the fundamental "
        "principles that guided this work, drawing on observations made during "
        "extensive travels through {place2} and the surrounding regions. The key "
        "insight was that the properties of {material} varied significantly depending "
        "on its source and preparation, and that optimal results required matching "
        "the right variant to the specific demands of each structural element. This "
        "principle, which may seem obvious to the modern {profession2}, was genuinely "
        "revolutionary in its time. It shifted the field from an approach based on "
        "received wisdom and tradition to one grounded in empirical testing and "
        "systematic documentation. The {adj3} legacy of this shift is visible in "
        "the surviving structures, many of which have endured for centuries in "
        "conditions that would have destroyed less carefully designed buildings. "
        "Contemporary engineers studying these structures continue to find lessons "
        "applicable to modern challenges, particularly in the areas of {abstract} "
        "and sustainable construction practices."
    ),
    (
        "Advances in the study of {science} during {period} were driven in large "
        "part by the development of new instruments capable of measurements that "
        "earlier generations of {profession}s could only dream of. In {place}, "
        "the workshop of {name} became a center of innovation, producing devices "
        "of {adj} precision that were exported to research institutions across the "
        "continent. The most significant of these was an instrument for measuring "
        "minute variations in the properties of {material}, a capability that opened "
        "entirely new areas of investigation. {name2}, a {profession2} working in "
        "{place2}, was among the first to recognize the instrument's potential. "
        "Using it to conduct a systematic survey of specimens gathered from the "
        "{feature} region, {name2} {verb_past} patterns that had eluded researchers "
        "for decades. The findings, published in a series of {adj2} papers that "
        "attracted considerable attention, demonstrated that the conventional "
        "understanding of {abstract} in the context of {science} was incomplete. "
        "The data suggested a more {adj3} relationship between structure and function "
        "than previously assumed, one that required new theoretical frameworks to "
        "explain. This prompted a period of intense collaborative work between "
        "experimentalists and theorists, a collaboration that proved remarkably "
        "productive. Within a few years, the field had been transformed, its "
        "foundational assumptions revised and its methods upgraded to accommodate "
        "the new level of detail that {name}'s instruments made possible. The story "
        "illustrates a broader principle that recurs throughout the history of "
        "{science}: that theoretical progress and instrumental innovation are "
        "deeply interdependent, each enabling and constraining the other in ways "
        "that shape the direction of discovery."
    ),
]


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def gen_narrative_extended():
    return _fill_extended(random.choice(NARRATIVE_EXTENDED_TEMPLATES))


def gen_dialogue_extended():
    return _fill_extended(random.choice(DIALOGUE_EXTENDED_TEMPLATES))


def gen_essay_extended():
    return _fill_extended(random.choice(ESSAY_EXTENDED_TEMPLATES))


def gen_nature_extended():
    return _fill_extended(random.choice(NATURE_EXTENDED_TEMPLATES))


def gen_philosophical_extended():
    return _fill_extended(random.choice(PHILOSOPHICAL_EXTENDED_TEMPLATES))


def gen_historical_extended():
    return _fill_extended(random.choice(HISTORICAL_EXTENDED_TEMPLATES))


def gen_technical_extended():
    return _fill_extended(random.choice(TECHNICAL_EXTENDED_TEMPLATES))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("gen_narrative_extended", 12000, gen_narrative_extended),
    ("gen_dialogue_extended", 8000, gen_dialogue_extended),
    ("gen_essay_extended", 8000, gen_essay_extended),
    ("gen_nature_extended", 6000, gen_nature_extended),
    ("gen_philosophical_extended", 6000, gen_philosophical_extended),
    ("gen_historical_extended", 5000, gen_historical_extended),
    ("gen_technical_extended", 5000, gen_technical_extended),
]


def main():
    random.seed(42)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "raw_texts" / "synthetic_scaled"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_texts = 0
    total_words = 0
    global_counter = 0
    category_stats = {}

    for cat_name, count, generator in CATEGORIES:
        cat_words = 0
        print(f"\nGenerating {count} texts for category: {cat_name}")

        for i in range(count):
            text = generator()
            word_count = len(text.split())
            cat_words += word_count

            filename = f"synthetic_{cat_name}_{i:06d}.txt"
            filepath = output_dir / filename
            filepath.write_text(text, encoding="utf-8")

            global_counter += 1
            if global_counter % 5000 == 0:
                print(f"  Progress: {global_counter} texts generated so far...")

        total_texts += count
        total_words += cat_words
        category_stats[cat_name] = {"count": count, "words": cat_words}
        print(f"  {cat_name}: {count} texts, {cat_words:,} words "
              f"(avg {cat_words // count} words/text)")

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total texts generated: {total_texts:,}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per text: {total_words // total_texts}")
    print("\nPer-category breakdown:")
    for cat_name, stats in category_stats.items():
        print(f"  {cat_name}: {stats['count']:,} texts, {stats['words']:,} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
