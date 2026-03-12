"""
Generate 24,000 Wikipedia-style encyclopedic articles using lambda templates.
7 categories with encyclopedic tone. Target: ~6.3M words total.
"""

import os
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

SCIENTISTS = [
    "Johannes Kepler", "Marie Curie", "Dmitri Mendeleev", "Gregor Mendel",
    "Niels Bohr", "Max Planck", "Rosalind Franklin", "Michael Faraday",
    "Carl Linnaeus", "Antoine Lavoisier", "James Clerk Maxwell", "Emmy Noether",
    "Lise Meitner", "Ernest Rutherford", "Leonhard Euler", "Henri Poincare",
    "Bernhard Riemann", "Srinivasa Ramanujan", "Ada Lovelace", "Blaise Pascal",
]

HISTORICAL_FIGURES = [
    "Charlemagne", "Saladin", "Genghis Khan", "Elizabeth I", "Akbar the Great",
    "Simón Bolívar", "Catherine the Great", "Suleiman the Magnificent",
    "Ashoka the Great", "Mansa Musa", "Hatshepsut", "Cyrus the Great",
    "Augustus Caesar", "Tokugawa Ieyasu", "Peter the Great", "Mehmed II",
    "Frederick the Great", "Isabella of Castile", "Harun al-Rashid",
    "Ramesses II",
]

PHILOSOPHERS = [
    "Aristotle", "Plato", "Confucius", "Immanuel Kant", "David Hume",
    "Friedrich Nietzsche", "Simone de Beauvoir", "John Locke", "Baruch Spinoza",
    "Thomas Aquinas", "René Descartes", "John Stuart Mill", "Hannah Arendt",
    "Søren Kierkegaard", "Ludwig Wittgenstein", "Jean-Paul Sartre",
    "Epicurus", "Zeno of Citium", "Al-Farabi", "Ibn Rushd",
]

PLACES = [
    "Vienna", "Prague", "Constantinople", "Alexandria", "Kyoto", "Samarkand",
    "Venice", "Bruges", "Timbuktu", "Hangzhou", "Lisbon", "Marrakech",
    "Edinburgh", "Barcelona", "Cusco", "Varanasi", "Isfahan", "Florence",
    "Seville", "Krakow", "Athens", "Rome", "Beijing", "Cairo", "Baghdad",
    "Delhi", "London", "Paris", "Moscow", "Tokyo",
]

YEARS_RANGE = list(range(1200, 1950))

ELEMENTS = [
    "hydrogen", "helium", "lithium", "carbon", "nitrogen", "oxygen",
    "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur",
    "iron", "copper", "zinc", "silver", "gold", "mercury", "lead", "uranium",
]

ORGANS = [
    "heart", "liver", "brain", "lungs", "kidneys", "stomach", "pancreas",
    "spleen", "thyroid", "adrenal glands", "pituitary gland", "hypothalamus",
    "cerebellum", "cerebral cortex", "bone marrow", "lymph nodes",
    "small intestine", "large intestine", "gallbladder", "appendix",
]

COUNTRIES = [
    "Japan", "Brazil", "Egypt", "India", "Norway", "Peru", "Morocco",
    "Thailand", "Greece", "Mexico", "Kenya", "Chile", "Indonesia",
    "Portugal", "Turkey", "Vietnam", "Argentina", "Ethiopia", "Poland",
    "Colombia", "Iran", "Sweden", "Nepal", "Tanzania", "Mongolia",
]

RIVERS = [
    "the Nile", "the Amazon", "the Danube", "the Yangtze", "the Ganges",
    "the Mississippi", "the Rhine", "the Mekong", "the Volga", "the Congo",
    "the Tigris", "the Euphrates", "the Zambezi", "the Indus", "the Loire",
]

MOUNTAINS = [
    "the Alps", "the Himalayas", "the Andes", "the Rockies", "the Caucasus",
    "the Pyrenees", "the Carpathians", "the Atlas Mountains",
    "the Appalachian Mountains", "the Urals", "Mount Kilimanjaro",
    "Mount Fuji", "Mount Olympus", "the Dolomites", "the Sierra Nevada",
]

ART_MOVEMENTS = [
    "Impressionism", "Baroque", "Renaissance art", "Romanticism",
    "Neoclassicism", "Expressionism", "Cubism", "Surrealism",
    "Art Nouveau", "Gothic art", "Mannerism", "Realism",
    "Pre-Raphaelitism", "Symbolism", "Post-Impressionism",
]

MUSICAL_FORMS = [
    "the symphony", "the concerto", "the sonata", "the fugue", "the opera",
    "the string quartet", "the oratorio", "chamber music", "choral music",
    "the art song",
]

MATH_FIELDS = [
    "number theory", "topology", "abstract algebra", "differential geometry",
    "combinatorics", "probability theory", "mathematical logic",
    "functional analysis", "algebraic geometry", "graph theory",
    "set theory", "category theory", "numerical analysis", "measure theory",
    "dynamical systems",
]

DISEASES = [
    "tuberculosis", "malaria", "cholera", "typhoid fever", "smallpox",
    "influenza", "pneumonia", "hepatitis", "diabetes mellitus",
    "hypertension", "anemia", "asthma", "epilepsy", "gout",
    "rheumatic fever", "scurvy", "rickets", "pellagra", "beriberi",
    "dysentery",
]

ECOSYSTEMS = [
    "tropical rainforest", "temperate deciduous forest", "boreal forest",
    "savanna grassland", "tundra", "coral reef", "mangrove wetland",
    "alpine meadow", "desert scrubland", "freshwater marsh",
    "kelp forest", "seagrass bed", "peat bog", "salt marsh",
    "Mediterranean scrubland",
]

ADJECTIVES = [
    "significant", "notable", "substantial", "considerable", "fundamental",
    "distinctive", "remarkable", "extensive", "prominent", "influential",
    "complex", "diverse", "systematic", "comprehensive", "critical",
]

CENTURIES = [
    "the 12th century", "the 13th century", "the 14th century",
    "the 15th century", "the 16th century", "the 17th century",
    "the 18th century", "the 19th century", "the 20th century",
]

PERIODS = [
    "the Renaissance", "the Enlightenment", "the Middle Ages",
    "the Classical era", "the Victorian age", "the Romantic period",
    "the Industrial Revolution", "the Age of Exploration",
    "the Bronze Age", "the Iron Age", "antiquity", "the Baroque period",
]


def _pick(pool):
    return random.choice(pool)


def _year():
    return str(random.choice(YEARS_RANGE))


# ---------------------------------------------------------------------------
# Wiki article generators (lambda-style using inner functions)
# ---------------------------------------------------------------------------

WIKI_SCIENCE_GENERATORS = [
    lambda: (
        f"Atomic Structure of {_pick(ELEMENTS).title()}\n\n"
        f"{_pick(ELEMENTS).title()} is a chemical element that has been the subject of "
        f"{_pick(ADJECTIVES)} scientific investigation since its isolation in {_year()}. "
        f"The element occupies a {_pick(ADJECTIVES)} position in the periodic table, and "
        f"its atomic structure exhibits properties that have proven essential to the "
        f"understanding of chemical bonding and molecular interaction. Early research "
        f"conducted in {_pick(PLACES)} established the fundamental characteristics of the "
        f"element, including its atomic mass, electron configuration, and most common "
        f"oxidation states. The work of {_pick(SCIENTISTS)} was particularly influential "
        f"in elucidating the quantum mechanical behavior of this element's electron "
        f"shells, demonstrating that the distribution of electrons followed patterns "
        f"that could be predicted mathematically but often defied classical intuition. "
        f"Subsequent experiments in {_pick(PLACES)} confirmed these theoretical predictions "
        f"and extended them to include the behavior of the element under extreme conditions "
        f"of temperature and pressure. The isotopic variants of {_pick(ELEMENTS)} have "
        f"attracted particular attention due to their applications in medical imaging, "
        f"industrial processes, and fundamental research. The most stable isotope has a "
        f"half-life that makes it suitable for dating geological formations, while "
        f"shorter-lived variants have found use in tracer studies and therapeutic "
        f"applications. The spectroscopic signature of the element is {_pick(ADJECTIVES)}, "
        f"featuring emission lines that have been used to identify its presence in stellar "
        f"atmospheres and interstellar gas clouds. Current research focuses on the "
        f"element's behavior in novel materials, including superconductors, catalysts, "
        f"and nanoscale structures, where its unique properties continue to yield "
        f"unexpected results that challenge existing theoretical models."
    ),
    lambda: (
        f"Principles of Stellar Nucleosynthesis\n\n"
        f"Stellar nucleosynthesis is the process by which elements are created within "
        f"the cores of stars through nuclear fusion reactions. This {_pick(ADJECTIVES)} "
        f"process is responsible for the production of nearly all elements heavier than "
        f"hydrogen and helium, beginning with the conversion of {_pick(ELEMENTS)} into "
        f"progressively heavier nuclei as a star evolves through its life cycle. The "
        f"theory of stellar nucleosynthesis was developed through the collaborative "
        f"efforts of researchers in {_pick(PLACES)} and {_pick(PLACES)} during "
        f"{_pick(CENTURIES)}, building on earlier work by {_pick(SCIENTISTS)} concerning "
        f"the energy source of stars. The process begins in the main sequence phase, "
        f"where hydrogen nuclei fuse to form helium through the proton-proton chain "
        f"or the carbon-nitrogen-oxygen cycle, depending on the star's mass and core "
        f"temperature. As the hydrogen fuel is depleted, stars of sufficient mass "
        f"undergo gravitational contraction, raising core temperatures to levels where "
        f"helium fusion becomes possible. This produces {_pick(ELEMENTS)} and oxygen "
        f"through the triple-alpha process, a reaction whose rate is extremely sensitive "
        f"to temperature. More massive stars continue this sequence, fusing progressively "
        f"heavier elements in concentric shells surrounding an inert core. The process "
        f"terminates with the production of iron-group elements, beyond which fusion "
        f"becomes endothermic and can no longer sustain the star against gravitational "
        f"collapse. Elements heavier than iron are produced primarily through neutron "
        f"capture processes, either the slow process occurring in asymptotic giant "
        f"branch stars or the rapid process associated with supernovae and neutron "
        f"star mergers. The abundances of elements observed in the solar system and "
        f"in stellar atmospheres provide {_pick(ADJECTIVES)} evidence for these "
        f"theoretical predictions and continue to inform models of galactic chemical "
        f"evolution."
    ),
    lambda: (
        f"Mechanisms of Enzyme Catalysis\n\n"
        f"Enzyme catalysis represents one of the most {_pick(ADJECTIVES)} areas of "
        f"biochemistry, encompassing the study of how biological macromolecules "
        f"accelerate chemical reactions with extraordinary specificity and efficiency. "
        f"The fundamental principles of enzyme catalysis were established through "
        f"research conducted in {_pick(PLACES)} and {_pick(PLACES)} during "
        f"{_pick(CENTURIES)}, when investigators first demonstrated that biological "
        f"extracts could catalyze reactions outside of living cells. The lock-and-key "
        f"model, proposed by researchers studying the decomposition of {_pick(ELEMENTS)}-"
        f"containing compounds, provided an initial framework for understanding enzyme "
        f"specificity. However, subsequent work revealed that the induced fit model, "
        f"in which both enzyme and substrate undergo conformational changes upon binding, "
        f"more accurately described the majority of enzymatic reactions. Modern "
        f"understanding recognizes several mechanisms by which enzymes achieve catalysis. "
        f"These include proximity and orientation effects, which position reactive groups "
        f"in optimal geometric arrangements; acid-base catalysis, involving the transfer "
        f"of protons between the enzyme and substrate; covalent catalysis, in which "
        f"temporary covalent bonds form between enzyme and substrate; and metal ion "
        f"catalysis, where {_pick(ELEMENTS)} or similar elements serve as cofactors. "
        f"The kinetic behavior of enzymes is described by the Michaelis-Menten equation "
        f"and its extensions, which relate reaction velocity to substrate concentration "
        f"through parameters that reflect the enzyme's affinity for its substrate and "
        f"its maximum catalytic rate. Allosteric regulation, in which molecules binding "
        f"at sites distant from the active site modulate enzymatic activity, provides "
        f"an additional layer of control that is {_pick(ADJECTIVES)} for cellular "
        f"metabolism. Research continues to reveal new aspects of enzyme function, "
        f"including the role of quantum mechanical tunneling in hydrogen transfer "
        f"reactions and the importance of protein dynamics in facilitating catalysis."
    ),
]

WIKI_HISTORY_GENERATORS = [
    lambda: (
        f"The Consolidation of Power in {_pick(PLACES)}\n\n"
        f"The political consolidation of {_pick(PLACES)} during {_pick(PERIODS)} "
        f"represents a {_pick(ADJECTIVES)} chapter in the broader history of state "
        f"formation. Prior to this period, the region was characterized by a "
        f"fragmented political landscape in which competing factions vied for control "
        f"of trade routes, agricultural land, and access to resources. The emergence "
        f"of a centralized authority was catalyzed by the military and diplomatic "
        f"efforts of {_pick(HISTORICAL_FIGURES)}, whose strategic vision transformed "
        f"the region from a collection of loosely affiliated territories into a "
        f"coherent political entity. The process was neither swift nor peaceful. "
        f"Resistance from local elites, who stood to lose autonomy under a centralized "
        f"system, was {_pick(ADJECTIVES)} and required a combination of military force, "
        f"strategic marriage alliances, and carefully negotiated concessions. The "
        f"administrative reforms that followed consolidation drew on models from "
        f"{_pick(PLACES)} and adapted them to local conditions. A professional "
        f"bureaucracy was established, tax collection was systematized, and a legal "
        f"code was promulgated that sought to balance the interests of the ruling "
        f"authority with the customary rights of subject populations. The cultural "
        f"consequences of consolidation were equally {_pick(ADJECTIVES)}. Patronage of "
        f"the arts, architecture, and learning flourished under the new regime, as "
        f"rulers sought to legitimize their authority through the visible display of "
        f"wealth and sophistication. Scholars from {_pick(PLACES)} were invited to "
        f"establish academies, and the resulting intellectual environment produced works "
        f"of lasting importance. The legacy of this period of consolidation continued "
        f"to shape the political and cultural character of the region for centuries, "
        f"and its effects remain discernible in contemporary institutions and social "
        f"structures."
    ),
    lambda: (
        f"The Trade Networks of {_pick(CENTURIES)}\n\n"
        f"The commercial networks that linked {_pick(PLACES)} to {_pick(PLACES)} during "
        f"{_pick(CENTURIES)} constituted one of the most {_pick(ADJECTIVES)} systems of "
        f"economic exchange in the pre-modern world. These networks were sustained by "
        f"a combination of overland caravan routes and maritime shipping lanes, each "
        f"with its own logistical challenges and commercial opportunities. The goods "
        f"that moved along these routes included textiles, precious metals, spices, "
        f"ceramics, and manuscripts, but the exchange was never purely material. Ideas, "
        f"technologies, religious practices, and artistic styles traveled alongside "
        f"physical commodities, creating patterns of cultural diffusion that profoundly "
        f"influenced the societies involved. The role of {_pick(HISTORICAL_FIGURES)} "
        f"in facilitating and regulating this trade has been the subject of "
        f"{_pick(ADJECTIVES)} scholarly attention. Under their patronage, trading posts "
        f"were established, standardized weights and measures were introduced, and "
        f"legal frameworks were developed to adjudicate commercial disputes between "
        f"merchants of different origins. The economic impact of these networks extended "
        f"well beyond the immediate participants. Secondary and tertiary markets "
        f"developed in {_pick(PLACES)} and surrounding regions, creating multiplier "
        f"effects that stimulated local production and encouraged specialization. "
        f"Agricultural practices adapted to meet the demands of distant consumers, "
        f"and artisanal industries emerged that catered specifically to the export "
        f"market. The decline of these networks, brought about by shifting political "
        f"circumstances, the opening of alternative routes, and changes in consumer "
        f"demand, had {_pick(ADJECTIVES)} consequences for the communities that had "
        f"depended on them, triggering economic restructuring that in some cases "
        f"took generations to complete."
    ),
    lambda: (
        f"Military Campaigns of {_pick(HISTORICAL_FIGURES)}\n\n"
        f"The military campaigns conducted by {_pick(HISTORICAL_FIGURES)} during "
        f"{_pick(CENTURIES)} rank among the most {_pick(ADJECTIVES)} military operations "
        f"in recorded history. Beginning with the consolidation of power in the "
        f"home territory near {_pick(PLACES)}, these campaigns eventually extended "
        f"across a vast geographic area, encompassing diverse terrain from mountain "
        f"ranges to river valleys and from arid plains to densely forested regions. "
        f"The military strategy employed was characterized by a combination of rapid "
        f"movement, flexible tactics, and the effective integration of different arms "
        f"of service. Cavalry, infantry, and siege engineers operated in coordinated "
        f"fashion, adapting their methods to the specific challenges posed by each "
        f"theatre of operations. The logistical infrastructure required to sustain "
        f"these campaigns was itself a {_pick(ADJECTIVES)} achievement, involving "
        f"the construction of supply depots, the establishment of communication "
        f"networks, and the management of large numbers of non-combatant support "
        f"personnel. The political consequences of the campaigns were far-reaching. "
        f"Defeated territories were incorporated into an expanding administrative "
        f"system that drew on practices from {_pick(PLACES)} and other conquered "
        f"regions. Local elites were often co-opted rather than displaced, a policy "
        f"that facilitated rapid pacification but created long-term tensions that "
        f"would eventually contribute to the fragmentation of the empire. The "
        f"cultural impact was equally {_pick(ADJECTIVES)}, as the movement of armies "
        f"and administrators created channels for the exchange of ideas, technologies, "
        f"and artistic traditions between previously isolated communities. Historians "
        f"continue to debate the balance between destruction and creation in these "
        f"campaigns, recognizing that the same forces that caused immense suffering "
        f"also laid the groundwork for new forms of political organization and "
        f"cultural synthesis."
    ),
]

WIKI_PHILOSOPHY_GENERATORS = [
    lambda: (
        f"The Problem of {_pick(['Free Will', 'Universals', 'Induction', 'Other Minds', 'Personal Identity', 'Evil', 'Consciousness', 'Causation'])}\n\n"
        f"The problem of {_pick(['free will', 'universals', 'induction', 'other minds', 'personal identity', 'evil', 'consciousness', 'causation'])} "
        f"is one of the most enduring questions in the history of philosophy, "
        f"attracting {_pick(ADJECTIVES)} attention from thinkers across multiple "
        f"traditions and historical periods. The classical formulation, associated "
        f"with the work of {_pick(PHILOSOPHERS)}, framed the problem in terms that "
        f"remain recognizable today, though subsequent developments have introduced "
        f"new dimensions and complications. At its core, the problem asks whether "
        f"and how certain fundamental features of human experience can be reconciled "
        f"with our best understanding of the natural world. The tension arises because "
        f"ordinary experience seems to presuppose conditions that are difficult to "
        f"establish on purely empirical grounds. {_pick(PHILOSOPHERS)}, writing during "
        f"{_pick(PERIODS)}, offered what many consider the most {_pick(ADJECTIVES)} "
        f"analysis of this tension, arguing that the apparent conflict dissolves once "
        f"we recognize that the categories of understanding through which we interpret "
        f"experience are themselves conditions of the possibility of experience. This "
        f"approach, while influential, was challenged by subsequent thinkers who argued "
        f"that it conceded too much to skepticism or, conversely, that it did not "
        f"concede enough. The contemporary debate has been enriched by contributions "
        f"from cognitive science, neuroscience, and philosophy of mind, which have "
        f"provided new empirical data relevant to the problem without resolving its "
        f"philosophical dimensions. Scholars in {_pick(PLACES)} and {_pick(PLACES)} "
        f"continue to produce {_pick(ADJECTIVES)} work on this topic, and the volume "
        f"of published research shows no sign of diminishing. What remains {_pick(ADJECTIVES)} "
        f"is the degree to which the problem resists definitive resolution, suggesting "
        f"that it touches on something genuinely fundamental about the relationship "
        f"between mind and world."
    ),
    lambda: (
        f"Ethical Theory in {_pick(PERIODS)}\n\n"
        f"The development of ethical theory during {_pick(PERIODS)} represents a "
        f"{_pick(ADJECTIVES)} chapter in the history of moral philosophy, marked by "
        f"the emergence of systematic frameworks that sought to ground moral judgment "
        f"in rational principles rather than custom, authority, or revelation. The "
        f"intellectual environment of {_pick(PLACES)} provided particularly fertile "
        f"ground for this project, as thinkers were exposed to a diversity of moral "
        f"traditions through trade, diplomacy, and the circulation of translated texts "
        f"from {_pick(PLACES)} and other centers of learning. {_pick(PHILOSOPHERS)} "
        f"is widely credited with formulating the most {_pick(ADJECTIVES)} version of "
        f"the rationalist approach, arguing that moral obligations could be derived "
        f"from principles accessible to any reasoning agent regardless of cultural "
        f"background or personal inclination. This universalist aspiration was "
        f"challenged by {_pick(PHILOSOPHERS)}, who maintained that moral sentiments, "
        f"rather than abstract reason, constituted the true foundation of ethical "
        f"life. The resulting debate between rationalism and sentimentalism proved "
        f"extraordinarily productive, generating a body of work that continues to "
        f"inform contemporary ethical theory. The practical implications of these "
        f"competing frameworks were {_pick(ADJECTIVES)}, influencing debates over "
        f"political legitimacy, the rights of individuals, the obligations of "
        f"governments, and the proper scope of law. Educational institutions in "
        f"{_pick(PLACES)} incorporated these discussions into their curricula, ensuring "
        f"that successive generations of students were exposed to the full range of "
        f"positions. The legacy of this period is evident in the structure of "
        f"contemporary moral philosophy, which continues to organize itself around "
        f"the fundamental question of whether morality is ultimately a matter of "
        f"reason, sentiment, or some combination of both."
    ),
    lambda: (
        f"Epistemology and the Limits of Knowledge\n\n"
        f"Epistemology, the branch of philosophy concerned with the nature, scope, "
        f"and limits of human knowledge, has been a central preoccupation of Western "
        f"philosophy since {_pick(PERIODS)}. The foundational question, what can we "
        f"know and how can we know it, was posed with particular urgency by "
        f"{_pick(PHILOSOPHERS)}, whose systematic investigation of the conditions of "
        f"knowledge established a framework that dominated philosophical inquiry for "
        f"centuries. The key insight was that knowledge required more than mere true "
        f"belief; it required justification, a condition whose precise nature has been "
        f"the subject of {_pick(ADJECTIVES)} debate ever since. The challenge of "
        f"skepticism, which questions whether justification is ever truly achievable, "
        f"has provided a persistent counterpoint to constructive epistemological "
        f"projects. {_pick(PHILOSOPHERS)}, working in {_pick(PLACES)} during a period "
        f"of {_pick(ADJECTIVES)} intellectual ferment, reformulated the skeptical "
        f"challenge in terms that forced subsequent thinkers to take it seriously as "
        f"more than a mere intellectual exercise. The response to skepticism has taken "
        f"many forms, from foundationalism, which seeks to identify beliefs that are "
        f"immune to doubt, to coherentism, which argues that beliefs are justified by "
        f"their relationships to other beliefs rather than by any foundational "
        f"certainty. More recently, virtue epistemology has shifted attention from "
        f"the properties of beliefs to the intellectual character of the knower, "
        f"arguing that knowledge is best understood as the product of epistemically "
        f"virtuous inquiry. Researchers in {_pick(PLACES)} have contributed "
        f"{_pick(ADJECTIVES)} work to this emerging field, drawing on resources from "
        f"both analytic and continental traditions. The practical relevance of "
        f"epistemology has become increasingly apparent in an age characterized by "
        f"information abundance and the challenge of distinguishing reliable knowledge "
        f"from misinformation."
    ),
]

WIKI_MEDICAL_GENERATORS = [
    lambda: (
        f"Clinical Overview of {_pick(DISEASES).title()}\n\n"
        f"{_pick(DISEASES).title()} is a medical condition that has been recognized "
        f"since antiquity, with descriptions appearing in texts from {_pick(PLACES)} "
        f"dating to {_pick(CENTURIES)}. The condition is characterized by a constellation "
        f"of symptoms affecting primarily the {_pick(ORGANS)}, though systemic "
        f"manifestations involving other organ systems are common in advanced cases. "
        f"The etiology of the condition was poorly understood until {_pick(CENTURIES)}, "
        f"when investigators in {_pick(PLACES)} established the pathophysiological "
        f"mechanisms underlying the disease process. The primary pathology involves "
        f"disruption of normal cellular function in the affected tissues, leading to "
        f"inflammation, tissue damage, and, if untreated, progressive organ dysfunction. "
        f"Diagnosis relies on a combination of clinical assessment, laboratory findings, "
        f"and imaging studies. The characteristic presentation includes changes in the "
        f"function of the {_pick(ORGANS)} and associated structures, often accompanied "
        f"by constitutional symptoms such as fatigue, weight loss, and intermittent "
        f"fever. Laboratory investigations typically reveal {_pick(ADJECTIVES)} "
        f"abnormalities in markers of inflammation and organ function. Treatment has "
        f"evolved {_pick(ADJECTIVES)}ly over the past century. Historical approaches, "
        f"which included dietary modification, herbal preparations, and various "
        f"physical therapies, have been largely supplanted by pharmacological "
        f"interventions targeting specific disease mechanisms. Current first-line "
        f"therapy aims to control the underlying disease process while managing "
        f"symptoms and preventing complications. Prognosis depends on the stage at "
        f"which treatment is initiated, the presence of comorbid conditions, and "
        f"individual patient factors. Research conducted in institutions in "
        f"{_pick(PLACES)} and {_pick(PLACES)} continues to advance understanding of "
        f"the condition, with particular focus on identifying biomarkers for early "
        f"detection and developing targeted therapies with improved efficacy and "
        f"fewer adverse effects."
    ),
    lambda: (
        f"Anatomy and Function of the {_pick(ORGANS).title()}\n\n"
        f"The {_pick(ORGANS)} is a vital organ whose structure and function have been "
        f"the subject of {_pick(ADJECTIVES)} medical and scientific investigation "
        f"since the earliest anatomical studies conducted in {_pick(PLACES)} during "
        f"{_pick(PERIODS)}. Situated within the body cavity, the organ performs "
        f"multiple functions essential to maintaining homeostasis, including metabolic "
        f"regulation, filtration, and the production of substances required by other "
        f"organ systems. The gross anatomy of the {_pick(ORGANS)} reflects its "
        f"functional complexity. The organ is divided into distinct regions, each "
        f"specialized for particular physiological tasks. The vascular supply is "
        f"{_pick(ADJECTIVES)}, reflecting the organ's high metabolic demands and its "
        f"role in processing and distributing substances throughout the body. At the "
        f"microscopic level, the tissue architecture reveals a highly organized "
        f"arrangement of specialized cell types, each contributing to the organ's "
        f"overall function. The discovery of these cellular subtypes, which began with "
        f"the work of researchers in {_pick(PLACES)} during {_pick(CENTURIES)}, has "
        f"been {_pick(ADJECTIVES)} for understanding both normal physiology and the "
        f"pathological processes that lead to disease. Dysfunction of the {_pick(ORGANS)} "
        f"can result from a variety of causes, including infection, autoimmune "
        f"processes, toxic exposure, and genetic abnormalities. The clinical "
        f"manifestations of such dysfunction vary depending on which specific functions "
        f"are impaired and the severity of the underlying pathology. Modern diagnostic "
        f"techniques, including advanced imaging and molecular assays developed by "
        f"research teams in {_pick(PLACES)}, have greatly improved the ability to "
        f"detect and characterize organ dysfunction at early stages, facilitating "
        f"more timely and targeted therapeutic interventions."
    ),
    lambda: (
        f"History of Surgical Technique\n\n"
        f"The history of surgical technique spans millennia, from the earliest "
        f"trepanation procedures performed in prehistoric communities to the "
        f"{_pick(ADJECTIVES)} minimally invasive approaches of contemporary practice. "
        f"Archaeological evidence from {_pick(PLACES)} and surrounding regions suggests "
        f"that surgical intervention was practiced as early as several thousand years "
        f"before the common era, with varying degrees of sophistication. The classical "
        f"period saw {_pick(ADJECTIVES)} advances, particularly in {_pick(PLACES)}, "
        f"where systematic observation and documentation of surgical procedures laid "
        f"the groundwork for later developments. Techniques for treating wounds, "
        f"setting fractures, and managing conditions of the {_pick(ORGANS)} were "
        f"codified in texts that remained authoritative for centuries. The medieval "
        f"period is often characterized as one of stagnation in surgical practice, "
        f"though recent scholarship has challenged this view, highlighting "
        f"{_pick(ADJECTIVES)} innovations developed in {_pick(PLACES)} and transmitted "
        f"to Europe through translated texts. The modern era of surgery began with "
        f"the introduction of anesthesia and antiseptic technique during "
        f"{_pick(CENTURIES)}, developments that dramatically reduced surgical mortality "
        f"and expanded the range of operable conditions. The subsequent development "
        f"of techniques for operating on the {_pick(ORGANS)} and other deep structures "
        f"required not only improved anesthetic methods but also a more detailed "
        f"understanding of anatomy and physiology. The twentieth century saw the "
        f"introduction of microsurgical techniques, transplantation, and image-guided "
        f"procedures, each representing a {_pick(ADJECTIVES)} advance in the surgeon's "
        f"ability to intervene with precision and safety. Contemporary research, "
        f"conducted in centers across {_pick(PLACES)} and other leading medical "
        f"institutions, continues to push the boundaries of what is surgically "
        f"achievable."
    ),
]

WIKI_GEOGRAPHY_GENERATORS = [
    lambda: (
        f"Geography and Climate of {_pick(COUNTRIES)}\n\n"
        f"{_pick(COUNTRIES)} occupies a {_pick(ADJECTIVES)} position in its geographic "
        f"region, encompassing terrain that ranges from coastal lowlands to mountainous "
        f"interiors. The country's topography has been shaped by geological processes "
        f"operating over millions of years, including tectonic uplift, volcanic activity, "
        f"and the erosive forces of water and wind. The drainage system is dominated "
        f"by {_pick(RIVERS)}, which traverses the country from its headwaters in "
        f"{_pick(MOUNTAINS)} to its outlet along the coast. This river system and its "
        f"tributaries provide water for agriculture, industry, and domestic use, and "
        f"have historically served as important transportation corridors. The climate "
        f"varies {_pick(ADJECTIVES)}ly across the country's territory. Coastal regions "
        f"experience maritime influences that moderate temperature extremes and bring "
        f"regular precipitation. Interior areas, sheltered from oceanic moisture by "
        f"mountain barriers, tend toward more continental conditions with greater "
        f"temperature variation and lower rainfall. The southern portions of the "
        f"country receive the highest levels of solar radiation and support vegetation "
        f"typical of {_pick(ECOSYSTEMS)} environments. The northern regions, by contrast, "
        f"exhibit characteristics more typical of {_pick(ECOSYSTEMS)} biomes. This "
        f"climatic diversity supports a correspondingly diverse array of ecosystems, "
        f"from {_pick(ECOSYSTEMS)} environments at lower elevations to alpine zones "
        f"at the highest points. The country's biodiversity has attracted "
        f"{_pick(ADJECTIVES)} scientific attention, with research stations established "
        f"by institutions from {_pick(PLACES)} and other countries conducting ongoing "
        f"surveys of flora and fauna. Conservation efforts have expanded in recent "
        f"decades, driven by recognition of the ecological significance of the "
        f"country's natural heritage and the threats posed by development, resource "
        f"extraction, and climatic change."
    ),
    lambda: (
        f"The Hydrology of {_pick(RIVERS)}\n\n"
        f"{_pick(RIVERS).replace('the ', '').title()} is one of the world's most "
        f"{_pick(ADJECTIVES)} river systems, draining a basin that encompasses "
        f"portions of several countries and supports millions of people. The river "
        f"originates in {_pick(MOUNTAINS)}, where snowmelt and glacial runoff feed "
        f"its upper tributaries, and flows for thousands of kilometers before reaching "
        f"its delta. The hydrological regime of the river is characterized by seasonal "
        f"variation in discharge, with peak flows typically occurring during the months "
        f"following the snowmelt season and minimum flows during the dry period. This "
        f"seasonal pattern has shaped human settlement along the river for millennia, "
        f"with agricultural communities in {_pick(PLACES)} and surrounding areas "
        f"organizing their planting and harvesting cycles around the river's rhythms. "
        f"The river's floodplain supports a {_pick(ADJECTIVES)} {_pick(ECOSYSTEMS)} "
        f"ecosystem, characterized by high biodiversity and complex ecological "
        f"interactions. Fish species adapted to the river's variable flow conditions "
        f"undertake migrations that are among the most {_pick(ADJECTIVES)} in the "
        f"freshwater realm. Riparian vegetation along the river's banks provides "
        f"habitat for terrestrial species and stabilizes the banks against erosion. "
        f"The construction of dams and diversions during {_pick(CENTURIES)} altered "
        f"the river's natural flow regime, with {_pick(ADJECTIVES)} consequences for "
        f"downstream ecosystems and communities. Sedimentation patterns changed, "
        f"fish migration routes were disrupted, and the seasonal flooding that had "
        f"renewed agricultural soils was curtailed. Contemporary river management "
        f"efforts seek to balance the competing demands of agriculture, industry, "
        f"urban water supply, and ecological conservation, a challenge that requires "
        f"coordination across political boundaries and competing interests."
    ),
    lambda: (
        f"Ecosystems of {_pick(MOUNTAINS)}\n\n"
        f"{_pick(MOUNTAINS)} constitute one of the most {_pick(ADJECTIVES)} mountain "
        f"systems on Earth, extending across multiple climate zones and supporting a "
        f"remarkable diversity of plant and animal communities. The ecological zonation "
        f"of these mountains follows a pattern determined primarily by elevation, with "
        f"distinct vegetation bands replacing one another as altitude increases and "
        f"temperatures decline. The lower slopes, where conditions are relatively "
        f"warm and moisture is abundant, support {_pick(ECOSYSTEMS)} communities "
        f"characterized by dense canopy cover and high species richness. As elevation "
        f"increases, the forest composition shifts toward species adapted to cooler "
        f"temperatures and shorter growing seasons. Above the treeline, which varies "
        f"in altitude depending on latitude and aspect, {_pick(ECOSYSTEMS)} environments "
        f"prevail, dominated by grasses, sedges, and low-growing shrubs. The highest "
        f"elevations support only sparse vegetation adapted to extreme conditions of "
        f"cold, wind, and ultraviolet radiation. The fauna of {_pick(MOUNTAINS)} "
        f"includes numerous endemic species whose distributions are restricted to "
        f"specific elevational bands. Research conducted by teams from {_pick(PLACES)} "
        f"and {_pick(PLACES)} has documented {_pick(ADJECTIVES)} levels of endemism, "
        f"particularly among invertebrates and lower plants. The glaciers and "
        f"permanent snowfields that cap the highest peaks serve as important sources "
        f"of freshwater, feeding rivers that support populations in the lowlands below. "
        f"The retreat of these glaciers, documented through systematic monitoring over "
        f"recent decades, has raised concerns about future water security and the "
        f"long-term viability of ecosystems that depend on glacial meltwater."
    ),
]

WIKI_ARTS_GENERATORS = [
    lambda: (
        f"{_pick(ART_MOVEMENTS)} and Its Cultural Context\n\n"
        f"{_pick(ART_MOVEMENTS)} emerged as a {_pick(ADJECTIVES)} artistic movement "
        f"during {_pick(CENTURIES)}, primarily in {_pick(PLACES)} before spreading to "
        f"{_pick(PLACES)} and other cultural centers. The movement was both a response "
        f"to and a reaction against the prevailing aesthetic conventions of the period, "
        f"which many artists and critics considered exhausted or inadequate for "
        f"expressing the realities of contemporary experience. The founders of the "
        f"movement, working initially in relative isolation, developed a distinctive "
        f"visual vocabulary characterized by innovative approaches to composition, "
        f"color, and the representation of light and space. The theoretical foundations "
        f"of the movement were articulated in manifestos, critical essays, and private "
        f"correspondence, documents that reveal the {_pick(ADJECTIVES)} ambition of its "
        f"practitioners. Central to the movement's philosophy was the conviction that "
        f"art should not merely reproduce external reality but should reveal underlying "
        f"truths about human perception and experience. This emphasis on subjective "
        f"experience represented a {_pick(ADJECTIVES)} departure from the academic "
        f"tradition, which had emphasized technical mastery and fidelity to observable "
        f"fact. The critical reception of the movement was initially mixed. "
        f"Conservative critics in {_pick(PLACES)} dismissed the new work as technically "
        f"deficient and aesthetically incoherent, while more sympathetic observers "
        f"recognized its originality and its potential to revitalize artistic practice. "
        f"Over time, the movement's influence proved {_pick(ADJECTIVES)}, affecting not "
        f"only painting and sculpture but also architecture, literature, and the "
        f"decorative arts. The legacy of {_pick(ART_MOVEMENTS)} is visible in virtually "
        f"every subsequent artistic development, and its core insights about the nature "
        f"of perception and representation continue to inform contemporary practice "
        f"and theory."
    ),
    lambda: (
        f"The Development of {_pick(MUSICAL_FORMS).title()}\n\n"
        f"{_pick(MUSICAL_FORMS).replace('the ', '').title()} is a musical form that "
        f"emerged during {_pick(PERIODS)} and subsequently became one of the most "
        f"{_pick(ADJECTIVES)} vehicles for musical expression in the Western classical "
        f"tradition. The form's origins can be traced to {_pick(PLACES)}, where "
        f"composers and performers experimented with new ways of organizing musical "
        f"material in response to changing aesthetic values and technological "
        f"developments in instrument construction. The early development of the form "
        f"was shaped by the patronage system, which connected composers to courts, "
        f"churches, and aristocratic households in {_pick(PLACES)} and throughout the "
        f"region. The structural principles underlying the form evolved over several "
        f"generations, with each major composer contributing refinements that expanded "
        f"its expressive range. The classical period saw the establishment of standard "
        f"formal conventions, including specific patterns of key relationships, "
        f"thematic development, and proportional balance between sections. These "
        f"conventions provided a framework within which composers could exercise "
        f"{_pick(ADJECTIVES)} creativity, and the tension between formal constraint "
        f"and individual expression became one of the defining characteristics of the "
        f"form. The romantic period brought {_pick(ADJECTIVES)} expansion of the form's "
        f"dimensions, harmonic language, and orchestral resources, as composers sought "
        f"to express emotional and philosophical content of unprecedented scope and "
        f"intensity. The twentieth century saw both radical transformation and "
        f"conscious preservation of the form, as composers in {_pick(PLACES)} and "
        f"elsewhere either deconstructed traditional conventions or reinterpreted "
        f"them through the lens of new aesthetic philosophies. The form continues to "
        f"attract composers and audiences, demonstrating a vitality that suggests its "
        f"expressive possibilities remain far from exhausted."
    ),
    lambda: (
        f"Architectural Traditions of {_pick(PLACES)}\n\n"
        f"The architectural heritage of {_pick(PLACES)} reflects a building tradition "
        f"that spans {_pick(ADJECTIVES)} periods of cultural development, from the "
        f"earliest permanent structures to the sophisticated monuments of later eras. "
        f"The city's built environment is characterized by a layered quality, with "
        f"structures from different periods coexisting and sometimes interpenetrating, "
        f"creating a streetscape that documents the successive waves of construction, "
        f"renovation, and adaptation that have shaped the urban fabric. The oldest "
        f"surviving structures date to {_pick(CENTURIES)} and display construction "
        f"techniques that reflect both local innovation and the influence of building "
        f"traditions from {_pick(PLACES)} and other centers. The use of locally "
        f"available materials, including stone, brick, and timber, determined the "
        f"basic vocabulary of the architectural tradition, while the region's climate "
        f"imposed constraints that shaped building form and orientation. Religious "
        f"architecture constitutes some of the most {_pick(ADJECTIVES)} examples of "
        f"the tradition, with places of worship designed to embody theological "
        f"principles through spatial organization, proportional systems, and the "
        f"manipulation of natural light. Secular architecture, while often less "
        f"monumental, displays its own {_pick(ADJECTIVES)} characteristics, including "
        f"sophisticated approaches to ventilation, water management, and the creation "
        f"of private space within dense urban environments. The colonial period "
        f"introduced new building types and aesthetic models from {_pick(PLACES)}, "
        f"creating a hybrid architectural language that combined imported forms with "
        f"local materials and craftsmanship. Contemporary preservation efforts seek "
        f"to maintain the historical integrity of the city's architectural heritage "
        f"while accommodating the demands of modern urban life."
    ),
]

WIKI_MATHEMATICS_GENERATORS = [
    lambda: (
        f"Foundations of {_pick(MATH_FIELDS).title()}\n\n"
        f"{_pick(MATH_FIELDS).title()} is a branch of mathematics that has developed "
        f"into one of the most {_pick(ADJECTIVES)} areas of mathematical research, "
        f"with applications spanning pure mathematics, theoretical physics, computer "
        f"science, and engineering. The historical roots of the field can be traced "
        f"to problems studied by mathematicians in {_pick(PLACES)} during "
        f"{_pick(CENTURIES)}, though the systematic development of the subject as a "
        f"distinct discipline occurred primarily in {_pick(CENTURIES)}. The foundational "
        f"contributions of {_pick(SCIENTISTS)} established the basic definitions, "
        f"axioms, and proof techniques that continue to form the core of the subject. "
        f"The central objects of study in the field are characterized by specific "
        f"structural properties that can be analyzed using a combination of algebraic, "
        f"analytic, and geometric methods. The interplay between these different "
        f"approaches has been one of the most {_pick(ADJECTIVES)} features of the "
        f"field's development, as results obtained through one method often suggest "
        f"unexpected connections to problems in other areas. The major theorems of the "
        f"field include results that are both technically demanding and conceptually "
        f"illuminating, providing deep insight into the structure of mathematical "
        f"objects that might appear simple at first glance. The proof techniques "
        f"developed within the field have found {_pick(ADJECTIVES)} application "
        f"elsewhere in mathematics, and several of the field's central problems "
        f"remain open, continuing to attract the attention of researchers at "
        f"institutions in {_pick(PLACES)} and mathematical centers worldwide. The "
        f"computational aspects of the field have become increasingly important as "
        f"advances in computer technology have made it possible to explore examples "
        f"and test conjectures at a scale that was previously impossible. This "
        f"interaction between theoretical development and computational exploration "
        f"represents one of the most {_pick(ADJECTIVES)} trends in contemporary "
        f"mathematical research."
    ),
    lambda: (
        f"The History and Applications of {_pick(MATH_FIELDS).title()}\n\n"
        f"The development of {_pick(MATH_FIELDS)} represents one of the most "
        f"{_pick(ADJECTIVES)} intellectual achievements in the history of mathematics. "
        f"The field originated from practical problems encountered in {_pick(PLACES)} "
        f"during {_pick(CENTURIES)}, but rapidly evolved into a sophisticated abstract "
        f"discipline with far-reaching implications. The early pioneers, including "
        f"{_pick(SCIENTISTS)} and contemporaries working in {_pick(PLACES)}, recognized "
        f"that the specific problems they were solving were instances of more general "
        f"mathematical structures, and the identification and characterization of "
        f"these structures became the primary focus of the field. The axiomatic "
        f"foundations of the subject were established through a process of gradual "
        f"refinement, as mathematicians sought to identify the minimal set of "
        f"assumptions from which the known results could be derived. This axiomatic "
        f"approach, characteristic of modern mathematics, brought {_pick(ADJECTIVES)} "
        f"clarity to the subject and revealed connections to other branches of "
        f"mathematics that had previously gone unnoticed. The applications of the "
        f"field are remarkably diverse. In physics, the mathematical structures "
        f"studied in the field provide the natural language for describing symmetries, "
        f"conservation laws, and the geometry of spacetime. In computer science, "
        f"the field's algorithmic aspects have informed the development of efficient "
        f"methods for solving large-scale computational problems. In engineering, "
        f"the field's tools are used to analyze the stability and behavior of complex "
        f"systems. The ongoing interaction between pure mathematical development and "
        f"applied problem-solving ensures that the field remains vibrant, with new "
        f"questions arising from applications that challenge and extend the theoretical "
        f"framework. Research groups in {_pick(PLACES)} and other mathematical centers "
        f"continue to make {_pick(ADJECTIVES)} contributions to both the theoretical "
        f"foundations and the practical applications of the field."
    ),
    lambda: (
        f"Major Theorems in {_pick(MATH_FIELDS).title()}\n\n"
        f"The major theorems of {_pick(MATH_FIELDS)} constitute some of the most "
        f"{_pick(ADJECTIVES)} results in all of mathematics, combining technical "
        f"depth with conceptual beauty in ways that have inspired generations of "
        f"mathematicians. The earliest significant results in the field were obtained "
        f"by researchers in {_pick(PLACES)} during {_pick(CENTURIES)}, who established "
        f"foundational results using methods that, while limited by modern standards, "
        f"demonstrated remarkable ingenuity. The development of more powerful tools "
        f"during subsequent periods, particularly in {_pick(PLACES)} and {_pick(PLACES)}, "
        f"made it possible to prove results of much greater generality and depth. "
        f"The work of {_pick(SCIENTISTS)} was particularly {_pick(ADJECTIVES)} in this "
        f"regard, introducing techniques that unified previously disparate areas of "
        f"the field and opened entirely new directions for investigation. Among the "
        f"most celebrated results is a theorem establishing the relationship between "
        f"local and global properties of the structures studied in the field. This "
        f"result, which required the development of new mathematical machinery for "
        f"its proof, has had {_pick(ADJECTIVES)} consequences not only within the "
        f"field itself but also in adjacent areas of mathematics. The proof techniques "
        f"employed in establishing this theorem have been adapted and generalized by "
        f"subsequent researchers, leading to results that extend and refine the "
        f"original insight. Other major theorems address the classification of objects "
        f"in the field, providing systematic criteria for determining when two "
        f"apparently different objects are in fact structurally identical. These "
        f"classification results are among the deepest in mathematics, often requiring "
        f"extended collaborative efforts by teams of mathematicians working at "
        f"institutions across multiple countries. The field continues to generate "
        f"conjectures that resist proof, ensuring that the next generation of "
        f"mathematicians will inherit a rich supply of challenging open problems."
    ),
]


# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("wiki_science", 5000, WIKI_SCIENCE_GENERATORS),
    ("wiki_history", 4000, WIKI_HISTORY_GENERATORS),
    ("wiki_philosophy", 3000, WIKI_PHILOSOPHY_GENERATORS),
    ("wiki_medical", 3000, WIKI_MEDICAL_GENERATORS),
    ("wiki_geography", 3000, WIKI_GEOGRAPHY_GENERATORS),
    ("wiki_arts", 3000, WIKI_ARTS_GENERATORS),
    ("wiki_mathematics", 3000, WIKI_MATHEMATICS_GENERATORS),
]


def main():
    random.seed(42)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "raw_texts" / "wikipedia_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_texts = 0
    total_words = 0
    global_counter = 0
    category_stats = {}

    for cat_name, count, generators in CATEGORIES:
        cat_words = 0
        print(f"\nGenerating {count} articles for category: {cat_name}")

        for i in range(count):
            generator = random.choice(generators)
            text = generator()
            word_count = len(text.split())
            cat_words += word_count

            filename = f"wikipedia_synthetic_{cat_name}_{i:06d}.txt"
            filepath = output_dir / filename
            filepath.write_text(text, encoding="utf-8")

            global_counter += 1
            if global_counter % 5000 == 0:
                print(f"  Progress: {global_counter} articles generated so far...")

        total_texts += count
        total_words += cat_words
        category_stats[cat_name] = {"count": count, "words": cat_words}
        print(f"  {cat_name}: {count} articles, {cat_words:,} words "
              f"(avg {cat_words // count} words/article)")

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total articles generated: {total_texts:,}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per article: {total_words // total_texts}")
    print("\nPer-category breakdown:")
    for cat_name, stats in category_stats.items():
        print(f"  {cat_name}: {stats['count']:,} articles, {stats['words']:,} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
