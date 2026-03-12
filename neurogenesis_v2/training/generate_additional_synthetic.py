"""
Generate 28,000 additional academic/professional texts.
4 formats: textbook passages, formal essays, historical narratives, technical descriptions.
Target: ~9.6M words total.
"""

import os
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

NAMES = [
    "Alexander", "Beatrice", "Catherine", "Dmitri", "Eleanor", "Frederick",
    "Genevieve", "Heinrich", "Isabella", "Johannes", "Katharina", "Leopold",
    "Magdalena", "Nathaniel", "Ophelia", "Percival", "Raphael", "Seraphina",
    "Theodore", "Ursula", "Valentina", "Wolfgang", "Ambrose", "Cornelius",
    "Dorothea", "Edmund", "Francesca", "Horatio", "Ingrid", "Julius",
    "Lysander", "Miriam", "Nikolai", "Octavia", "Penelope", "Roland",
    "Sylvia", "Tobias", "Vivienne", "Winston", "Artemis", "Bartholomew",
    "Clementine", "Desmond", "Emmeline", "Gilbert", "Helena", "Isadora",
    "Jasper", "Marguerite",
]

PLACES = [
    "Vienna", "Prague", "Constantinople", "Alexandria", "Kyoto", "Samarkand",
    "Venice", "Bruges", "Timbuktu", "Hangzhou", "Lisbon", "Marrakech",
    "Edinburgh", "Barcelona", "Dubrovnik", "Cusco", "Varanasi", "Isfahan",
    "Florence", "Seville", "Krakow", "Athens", "Rome", "Beijing", "Cairo",
    "Baghdad", "Delhi", "London", "Paris", "Moscow", "Stockholm", "Zurich",
    "Geneva", "Heidelberg", "Salamanca", "Bologna", "Leiden", "Uppsala",
    "Coimbra", "Leuven",
]

ADJECTIVES = [
    "fundamental", "significant", "comprehensive", "systematic", "rigorous",
    "nuanced", "substantial", "meticulous", "thorough", "sophisticated",
    "innovative", "traditional", "empirical", "theoretical", "practical",
    "analytical", "critical", "comparative", "interdisciplinary", "seminal",
    "groundbreaking", "influential", "controversial", "pioneering", "decisive",
    "profound", "complex", "intricate", "elegant", "robust",
]

SCIENCES = [
    "astronomy", "botany", "chemistry", "geology", "physics", "mathematics",
    "biology", "meteorology", "zoology", "ecology", "anatomy", "pharmacology",
    "mineralogy", "hydrology", "optics", "acoustics", "thermodynamics",
    "electromagnetism", "taxonomy", "crystallography", "paleontology",
    "epidemiology", "genetics", "neuroscience", "immunology",
]

ABSTRACT_NOUNS = [
    "truth", "justice", "beauty", "wisdom", "knowledge", "harmony",
    "liberty", "compassion", "integrity", "resilience", "curiosity",
    "understanding", "equality", "progress", "innovation", "tradition",
    "authority", "legitimacy", "sovereignty", "governance", "morality",
    "consciousness", "identity", "causation", "evidence", "methodology",
    "objectivity", "subjectivity", "rationality", "empiricism",
]

TIME_PERIODS = [
    "the Renaissance", "the Enlightenment", "the Middle Ages",
    "the Classical era", "the Victorian age", "the Romantic period",
    "the Industrial Revolution", "the Age of Exploration",
    "the Bronze Age", "the Iron Age", "antiquity", "the Baroque period",
    "the Reformation", "the Scientific Revolution", "the Gilded Age",
    "the Meiji era", "the Mughal period", "the Tang Dynasty",
    "the Ottoman era", "the Colonial period", "the Hellenistic period",
    "the Medieval period", "the Early Modern period",
]

PROFESSIONS = [
    "astronomer", "cartographer", "apothecary", "philosopher", "architect",
    "botanist", "scribe", "physician", "alchemist", "navigator", "sculptor",
    "librarian", "diplomat", "mathematician", "theologian", "merchant",
    "composer", "geographer", "chronicler", "metallurgist", "linguist",
    "historian", "engineer", "chemist", "naturalist", "physicist",
    "biologist", "geologist", "archaeologist", "anthropologist",
]

MATERIALS = [
    "marble", "granite", "oak", "bronze", "silk", "iron", "glass",
    "porcelain", "leather", "copper", "steel", "aluminium", "concrete",
    "timber", "brick", "sandstone", "limestone", "basalt", "quartz",
    "obsidian",
]

NATURAL_FEATURES = [
    "mountain range", "river valley", "coastal plain", "dense forest",
    "vast desert", "frozen tundra", "volcanic island", "coral reef",
    "rolling prairie", "deep canyon", "alpine meadow", "mangrove swamp",
    "limestone cave", "glacial lake", "salt flat", "sandstone arch",
    "fjord", "estuary", "wetland", "steppe",
]

CENTURIES = [
    "the 12th century", "the 13th century", "the 14th century",
    "the 15th century", "the 16th century", "the 17th century",
    "the 18th century", "the 19th century", "the 20th century",
]

VERBS_PAST = [
    "discovered", "examined", "contemplated", "constructed", "observed",
    "documented", "analyzed", "cultivated", "restored", "transformed",
    "illuminated", "deciphered", "catalogued", "preserved", "synthesized",
    "theorized", "pioneered", "revolutionized", "established", "formulated",
]


def _pick(pool):
    return random.choice(pool)


def _fill(template):
    """Fill a template string by replacing {placeholders} with random vocabulary."""
    mapping = {
        "name": lambda: _pick(NAMES),
        "name2": lambda: _pick(NAMES),
        "name3": lambda: _pick(NAMES),
        "place": lambda: _pick(PLACES),
        "place2": lambda: _pick(PLACES),
        "place3": lambda: _pick(PLACES),
        "adj": lambda: _pick(ADJECTIVES),
        "adj2": lambda: _pick(ADJECTIVES),
        "adj3": lambda: _pick(ADJECTIVES),
        "adj4": lambda: _pick(ADJECTIVES),
        "science": lambda: _pick(SCIENCES),
        "science2": lambda: _pick(SCIENCES),
        "abstract": lambda: _pick(ABSTRACT_NOUNS),
        "abstract2": lambda: _pick(ABSTRACT_NOUNS),
        "abstract3": lambda: _pick(ABSTRACT_NOUNS),
        "period": lambda: _pick(TIME_PERIODS),
        "period2": lambda: _pick(TIME_PERIODS),
        "profession": lambda: _pick(PROFESSIONS),
        "profession2": lambda: _pick(PROFESSIONS),
        "material": lambda: _pick(MATERIALS),
        "material2": lambda: _pick(MATERIALS),
        "feature": lambda: _pick(NATURAL_FEATURES),
        "feature2": lambda: _pick(NATURAL_FEATURES),
        "century": lambda: _pick(CENTURIES),
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
# Textbook passage templates
# ---------------------------------------------------------------------------

TEXTBOOK_TEMPLATES = [
    (
        "Chapter Overview: The Foundations of {science}\n\n"
        "The study of {science} rests upon a set of {adj} principles that have been "
        "refined over centuries of observation, experimentation, and theoretical "
        "development. In this chapter, we examine the core concepts that underpin the "
        "discipline, beginning with the historical context in which they emerged and "
        "tracing their evolution to the present day. The earliest systematic "
        "investigations in {science} were conducted in {place} during {period}, when "
        "scholars first began to distinguish between empirical observation and "
        "speculative reasoning. This distinction, which may seem obvious to the modern "
        "student, represented a {adj2} intellectual achievement that transformed the "
        "way knowledge was produced and validated. The key figures of this early period, "
        "including {name} and {name2}, established methodological standards that "
        "continue to influence practice in the field. Their insistence on reproducibility, "
        "careful documentation, and the systematic testing of hypotheses laid the "
        "groundwork for what would later become the scientific method. As the discipline "
        "matured, new areas of investigation opened up. The work of {name3} in {place2} "
        "during {century} demonstrated that the principles of {science} could be applied "
        "to problems of increasing complexity, and the resulting discoveries had "
        "{adj3} implications for related fields. The tools and techniques available to "
        "practitioners also evolved, from the simple instruments of the early period "
        "to the {adj4} apparatus of the modern laboratory. Each advance in instrumentation "
        "made it possible to observe phenomena that had previously been invisible, "
        "expanding the domain of the discipline and challenging existing theoretical "
        "frameworks. Students approaching {science} for the first time should be aware "
        "that the field is characterized by an ongoing dialogue between theory and "
        "experiment, and that the most productive researchers are those who maintain a "
        "critical awareness of both the power and the limitations of the methods they "
        "employ. The exercises at the end of this chapter are designed to reinforce these "
        "foundational concepts and to develop the analytical skills that will be required "
        "in subsequent chapters."
    ),
    (
        "Key Concepts in {science}: A Textbook Introduction\n\n"
        "This section introduces the {adj} theoretical framework that organizes our "
        "understanding of {science} and provides the vocabulary needed for more advanced "
        "study. The framework is built upon several key concepts, each of which has a "
        "precise technical meaning that must be distinguished from its everyday usage. "
        "The first of these concepts concerns the relationship between structure and "
        "function. In {science}, as in many related disciplines, the properties of a "
        "system are determined not only by the identity of its components but by the way "
        "those components are organized. This principle, first articulated by researchers "
        "in {place} during {period}, has proven to be one of the most {adj2} ideas in "
        "the field, applicable at every scale from the microscopic to the macroscopic. "
        "The second key concept is the principle of conservation, which states that "
        "certain measurable quantities remain constant as a system changes over time. "
        "The identification of conserved quantities has been a {adj3} tool in the "
        "development of {science}, as it allows researchers to constrain the range of "
        "possible outcomes and thereby simplify the analysis of complex systems. The "
        "work of {name} in {place2} during {century} provided the first {adj4} "
        "demonstration of this principle in the context of the field, and subsequent "
        "research has extended it to an increasingly wide range of phenomena. A third "
        "concept, equilibrium, describes the state in which competing processes balance "
        "one another, producing a condition of apparent stability. Understanding "
        "equilibrium and the conditions under which it is maintained or disrupted is "
        "essential for analyzing both natural and engineered systems. Together, these "
        "concepts form the conceptual foundation upon which the remainder of this "
        "textbook is built. Each subsequent chapter will elaborate on one or more of "
        "these ideas, introducing the mathematical formalism needed to apply them "
        "quantitatively and exploring their implications through worked examples and "
        "case studies drawn from current research."
    ),
    (
        "Practical Applications of {science}\n\n"
        "The transition from theoretical understanding to practical application is one "
        "of the most {adj} aspects of {science}, requiring practitioners to bridge the "
        "gap between abstract principles and the complexities of real-world systems. "
        "This chapter examines several case studies that illustrate how the concepts "
        "developed in earlier chapters have been applied to solve problems of "
        "practical significance. The first case study concerns the development of "
        "improved methods for processing {material}, a challenge that occupied "
        "researchers in {place} for much of {century}. The problem was that existing "
        "techniques, while effective on a small scale, produced inconsistent results "
        "when scaled up to industrial quantities. The solution, developed by {name} "
        "and a team of collaborators at the institute in {place2}, involved applying "
        "principles from {science} to redesign the processing sequence. The {adj2} "
        "insight was that temperature control during a critical phase of the process "
        "was far more important than had been previously recognized, and that small "
        "variations in this parameter could produce {adj3} differences in the quality "
        "of the final product. By developing instruments capable of monitoring "
        "temperature with greater precision and incorporating feedback mechanisms to "
        "maintain optimal conditions, the team achieved a level of consistency that "
        "had eluded earlier practitioners. The second case study examines the "
        "application of {science} principles to environmental monitoring in the region "
        "surrounding {place}. The {feature} ecosystem had been under increasing stress "
        "from human activity, and conventional monitoring methods provided an incomplete "
        "picture of the changes taking place. Researchers from {place2} introduced a "
        "new approach based on the {adj4} analysis of multiple data streams, integrating "
        "information from satellite imagery, ground-based sensors, and biological "
        "indicators. The resulting monitoring system proved far more effective at "
        "detecting early signs of ecosystem degradation, enabling more timely and "
        "targeted intervention. These examples illustrate a general principle: that the "
        "most successful applications of {science} are those that combine rigorous "
        "theoretical understanding with a {adj3} appreciation for the specific "
        "conditions of the problem at hand."
    ),
    (
        "Historical Development of {science}: From Observation to Theory\n\n"
        "The evolution of {science} from a descriptive enterprise to a mature "
        "theoretical discipline is one of the most {adj} stories in the history of "
        "knowledge. The earliest practitioners were careful observers who recorded "
        "natural phenomena with {adj2} precision but lacked the conceptual tools "
        "needed to explain what they saw. In {place} during {period}, scholars "
        "maintained detailed records of observations over extended periods, creating "
        "data sets that would later prove invaluable for testing theoretical "
        "predictions. The transition to explanatory science began in {place2} during "
        "{century}, when {name} proposed a {adj3} framework that connected previously "
        "isolated observations into a coherent account. This framework, while "
        "imperfect by modern standards, demonstrated for the first time that the "
        "phenomena studied by {science} were governed by regular principles amenable "
        "to mathematical description. The subsequent refinement of this framework "
        "involved contributions from researchers across multiple institutions. {name2}, "
        "working in {place3} under the influence of the {period2} intellectual "
        "tradition, introduced modifications that addressed known deficiencies and "
        "extended the theory's predictive reach. The experimental confirmation of "
        "these predictions, achieved through {adj4} measurements using newly developed "
        "instruments, established the credibility of the theoretical approach and "
        "attracted new practitioners to the field. By the end of {century}, {science} "
        "had achieved the status of a mature discipline with established methods, a "
        "growing body of confirmed predictions, and a community of practitioners "
        "engaged in both fundamental research and practical application. The story of "
        "this development is instructive for students because it illustrates several "
        "general features of scientific progress, including the role of anomalous "
        "observations in stimulating theoretical innovation, the importance of "
        "instrumentation in enabling new discoveries, and the gradual, collaborative "
        "nature of scientific advance."
    ),
]

# ---------------------------------------------------------------------------
# Formal essay templates
# ---------------------------------------------------------------------------

ESSAY_TEMPLATES = [
    (
        "The relationship between {abstract} and {abstract2} constitutes one of the "
        "most {adj} problems in contemporary intellectual life. While the two concepts "
        "are frequently invoked in public discourse, their precise meanings remain "
        "contested, and the nature of their relationship is far from settled. This "
        "essay argues that a proper understanding of {abstract} requires attending to "
        "its historical development, its dependence on institutional context, and its "
        "complex interaction with {abstract2}. The argument proceeds in three stages. "
        "First, a historical overview traces the evolution of {abstract} as a concept "
        "from its origins in {place} during {period} to its current usage in academic "
        "and professional settings. This overview reveals that {abstract} has never "
        "been a static concept but has been continuously redefined in response to "
        "changing social, political, and intellectual conditions. Second, the essay "
        "examines the institutional structures that shape how {abstract} is understood "
        "and practiced. Drawing on evidence from {place2} and comparable settings, it "
        "is shown that the meaning of {abstract} is partly determined by the norms, "
        "incentives, and power structures of the institutions in which it operates. "
        "The work of {name}, a {profession} whose {adj2} analysis of institutional "
        "dynamics remains influential, provides a useful framework for this analysis. "
        "Third, the essay considers the relationship between {abstract} and {abstract2}, "
        "arguing that the two are neither identical nor opposed but rather mutually "
        "constitutive. The pursuit of {abstract} without attention to {abstract2} "
        "produces outcomes that are technically correct but socially impoverished, while "
        "an emphasis on {abstract2} without grounding in {abstract} risks descending "
        "into sentimentality. The conclusion draws on insights from {name2}, a "
        "{profession2} working in {place}, to sketch a framework that integrates both "
        "values without collapsing the distinction between them. It is suggested that "
        "this integrative approach, while demanding, offers the most {adj3} path forward "
        "for those who seek to act with both {abstract} and {abstract2} in a world of "
        "irreducible complexity."
    ),
    (
        "The question of whether {abstract} can be reconciled with the demands of "
        "modern institutional life is not new, but it acquires particular urgency in "
        "the present moment. As organizations grow larger, more complex, and more "
        "deeply integrated into global systems, the conditions under which individuals "
        "can exercise genuine {abstract} are increasingly constrained. This essay "
        "examines the nature and extent of these constraints, drawing on case studies "
        "from {place} and {place2}, and argues that the preservation of {abstract} "
        "requires not only individual commitment but also {adj} structural reform. "
        "The argument begins with a conceptual analysis of {abstract}, distinguishing "
        "between its procedural and substantive dimensions. Procedurally, {abstract} "
        "refers to the capacity to make decisions free from external coercion. "
        "Substantively, it refers to the capacity to make decisions that are genuinely "
        "one's own, reflecting considered values and informed judgment. Both dimensions "
        "are important, but the essay argues that contemporary discussions tend to "
        "overemphasize the procedural at the expense of the substantive. The "
        "institutional analysis that follows draws on the work of {name}, a "
        "{profession} whose research in {place} during {century} {verb_past} the "
        "mechanisms through which organizational structures shape individual decision-"
        "making. The findings were {adj2}: even in institutions formally committed "
        "to {abstract}, subtle pressures of hierarchy, conformity, and routine "
        "significantly narrowed the range of options that individuals felt empowered "
        "to consider. The essay then turns to the question of reform, examining "
        "several proposals for restructuring institutions to better support {abstract2}. "
        "Drawing on the comparative analysis of {name2}, who studied similar "
        "institutions in {place2}, the essay identifies three principles that appear "
        "to be {adj3} for effective reform: transparency in decision-making processes, "
        "meaningful participation by all stakeholders, and accountability mechanisms "
        "that ensure stated values are reflected in actual practice. The essay concludes "
        "by acknowledging the difficulty of implementing these principles and by "
        "arguing that the effort is nonetheless essential if {abstract3} is to remain "
        "more than an aspiration."
    ),
    (
        "In recent decades, the concept of {abstract} has attracted {adj} attention "
        "from scholars across a range of disciplines, including philosophy, political "
        "science, and the social sciences more broadly. This interdisciplinary interest "
        "reflects a growing recognition that {abstract} is not merely an abstract "
        "ideal but a practical condition whose presence or absence has {adj2} "
        "consequences for individuals and communities alike. This essay contributes "
        "to the ongoing discussion by examining the concept from a comparative "
        "perspective, drawing on historical evidence from {place} during {period} "
        "and from {place2} during {period2}. The comparison reveals both striking "
        "similarities and instructive differences in how {abstract} has been "
        "understood and institutionalized across different cultural and historical "
        "contexts. In {place}, the dominant understanding of {abstract} emphasized "
        "its connection to {abstract2}, and the institutions developed to promote it "
        "reflected this emphasis. {name}, a {profession} who played a {adj3} role "
        "in shaping these institutions, argued that {abstract} without {abstract2} "
        "was incomplete and ultimately self-defeating. In {place2}, by contrast, "
        "the emphasis fell more heavily on the relationship between {abstract} and "
        "{abstract3}, reflecting a different set of historical circumstances and "
        "intellectual traditions. {name2}, whose writings from this period remain "
        "widely studied, maintained that {abstract3} was the prerequisite for "
        "genuine {abstract}, and that any attempt to promote the latter without "
        "securing the former was doomed to superficiality. The essay argues that "
        "both perspectives capture important truths, and that a {adj4} understanding "
        "of {abstract} must incorporate insights from both traditions. The concluding "
        "section proposes a synthetic framework that draws on the strengths of each "
        "approach while addressing their respective limitations. It is argued that "
        "this synthesis is not merely of historical interest but has direct relevance "
        "to contemporary debates about the design of institutions and the conditions "
        "of a good society."
    ),
    (
        "The tension between {abstract} and {abstract2} has been a recurring theme "
        "in the intellectual history of the West, and its resolution, or at least "
        "its management, remains one of the central challenges of political and "
        "moral philosophy. This essay offers a {adj} reexamination of this tension, "
        "drawing on classical sources, modern theoretical developments, and "
        "contemporary empirical research. The classical formulation of the problem, "
        "associated with thinkers in {place} during {period}, posed it in terms "
        "of a fundamental opposition: either {abstract} or {abstract2} must take "
        "precedence, and the attempt to serve both simultaneously leads to incoherence "
        "or hypocrisy. This framing, while intellectually clarifying, has been "
        "criticized by subsequent thinkers for its excessive rigidity. {name}, "
        "writing during {period2}, argued that the opposition was not genuine but "
        "rather the product of a {adj2} conceptual error. According to this view, "
        "{abstract} and {abstract2} are not competing values but complementary "
        "aspects of a single, more comprehensive ideal. The {profession} {name2}, "
        "working independently in {place2}, reached a similar conclusion through a "
        "different line of reasoning, drawing on evidence from the practical "
        "experience of {profession2}s who routinely navigated the tension in their "
        "daily work. The empirical evidence, drawn from studies conducted in "
        "{place} and {place2} over the past several decades, lends support to this "
        "more nuanced view. Individuals and institutions that successfully integrate "
        "{abstract} and {abstract2} tend to produce outcomes that are {adj3} on "
        "multiple dimensions, while those that prioritize one at the expense of the "
        "other tend to encounter problems that could have been predicted by the "
        "theoretical framework outlined above. The essay concludes by drawing out "
        "the practical implications of this analysis, arguing that the pursuit of "
        "{abstract3} requires a {adj4} commitment to both {abstract} and {abstract2}, "
        "and that institutions designed to foster one should always be evaluated in "
        "terms of their effects on the other."
    ),
]

# ---------------------------------------------------------------------------
# Historical narrative templates
# ---------------------------------------------------------------------------

HISTORICAL_NARRATIVE_TEMPLATES = [
    (
        "The city of {place} in {century} stood at a crossroads of cultural and "
        "political forces that would reshape the region for generations to come. The "
        "ruling authority, having consolidated power through a combination of military "
        "prowess and diplomatic skill, faced the {adj} challenge of governing a diverse "
        "population with competing interests and loyalties. The administration, drawing "
        "on models from {place2} and adapting them to local conditions, established a "
        "bureaucratic system that was {adj2} in its attention to detail and remarkably "
        "effective in its execution. Tax revenues were collected with regularity, public "
        "works were maintained, and a system of courts adjudicated disputes according "
        "to a legal code that, while imperfect by modern standards, represented a "
        "genuine advance over the arbitrary justice that had prevailed previously. "
        "The cultural life of {place} during this period was equally vibrant. {name}, "
        "a {profession} of exceptional talent, attracted students and collaborators "
        "from across the region, transforming the city into a center of learning that "
        "rivaled {place2} in prestige if not in size. The workshops and studios that "
        "lined the principal streets produced works of {material} and other materials "
        "that were traded as far as {place3}. The intellectual atmosphere was "
        "characterized by a {adj3} openness to new ideas, tempered by respect for "
        "established traditions. Scholars debated the merits of competing approaches "
        "to {science}, {abstract}, and the proper organization of society, and the "
        "resulting body of written work provides an invaluable record of a civilization "
        "at the height of its creative powers. The decline that followed, brought on "
        "by a combination of external pressures and internal contradictions, was "
        "gradual rather than sudden. Over the course of several decades, the conditions "
        "that had sustained the cultural flowering deteriorated, and the city entered "
        "a period of contraction from which it would not fully recover for centuries. "
        "The memory of the golden age persisted, however, serving as an inspiration "
        "and a reproach to subsequent generations."
    ),
    (
        "The expedition that departed from {place} in the early decades of {century} "
        "represented one of the most {adj} ventures of the age, combining scientific "
        "ambition with political calculation in roughly equal measure. The sponsoring "
        "authority, motivated by a desire to extend territorial claims and to gather "
        "intelligence about the resources and peoples of distant regions, assembled a "
        "company of specialists that included {profession}s, {profession2}s, soldiers, "
        "and translators. At the head of this diverse group stood {name}, whose "
        "previous experience in {place2} had demonstrated both leadership ability and "
        "a capacity for improvisation under difficult circumstances. The route chosen "
        "took the expedition through the {feature}, a region known for its {adj2} "
        "terrain and unpredictable weather patterns. The early stages of the journey "
        "proceeded according to plan, with the company making steady progress and "
        "{name} sending regular dispatches back to the capital. Encounters with local "
        "populations were generally peaceful, facilitated by the translators and by "
        "{name}'s practice of offering gifts and demonstrations of European technology. "
        "The scientific members of the expedition, particularly {name2}, a {profession} "
        "with a keen interest in {science}, took full advantage of the opportunity, "
        "collecting specimens, recording observations, and sketching maps of the "
        "landscape. The materials gathered during this period would later prove "
        "{adj3} for the development of {science} in {place} and beyond. Difficulties "
        "mounted as the expedition penetrated deeper into unknown territory. Supply "
        "lines stretched thin, illness thinned the ranks, and the terrain grew "
        "increasingly hostile. {name}'s leadership was tested repeatedly, and the "
        "decisions made during this critical phase reveal both the strengths and "
        "limitations of the expedition's planning. The eventual return to {place}, "
        "after an absence of more than two years, was met with celebrations that "
        "masked the human cost of the venture. The published accounts, carefully "
        "edited to emphasize success and minimize failure, became influential texts "
        "that shaped public understanding of the regions traversed and influenced "
        "subsequent expeditions for decades."
    ),
    (
        "The reforms instituted in {place} during {period} fundamentally altered "
        "the relationship between the governing authority and the governed population, "
        "setting in motion a process of institutional transformation that would unfold "
        "over the better part of a century. The immediate catalyst for reform was a "
        "crisis of {abstract}, precipitated by a series of military setbacks that "
        "exposed the {adj} inadequacy of existing administrative structures. The "
        "reform program, championed by {name} and a circle of like-minded {profession}s, "
        "drew inspiration from developments in {place2}, where similar challenges had "
        "been met with {adj2} reorganization of state institutions. The first phase of "
        "reform targeted the military establishment, introducing standardized training, "
        "professional appointment procedures, and a system of rank based on merit "
        "rather than birth. The second phase addressed the civilian administration, "
        "creating new ministries, rationalizing the tax system, and establishing "
        "educational institutions intended to produce a class of competent public "
        "servants. The third phase, which proved the most contentious, attempted to "
        "reform the legal system, replacing the patchwork of customary law and "
        "royal decree with a unified code based on principles of {abstract2} and "
        "equal treatment. Resistance to the reforms was {adj3} and came from multiple "
        "directions. The traditional aristocracy, whose privileges were threatened by "
        "meritocratic appointment, mobilized their considerable resources in opposition. "
        "Religious authorities, wary of the secular tendencies of the reform movement, "
        "raised objections grounded in theological arguments about the proper ordering "
        "of society. Even among the reformers themselves, disagreements over pace, "
        "priorities, and methods created fissures that opponents were quick to exploit. "
        "{name2}, who had been a {adj4} ally in the early stages, broke with {name} "
        "over the question of legal reform, arguing that the proposed changes went too "
        "far too fast. The eventual outcome was a compromise that preserved the core "
        "of the reform program while accommodating some of the concerns raised by "
        "its critics. The legacy of this period continues to be debated by historians, "
        "with some emphasizing the genuine advances achieved and others pointing to "
        "the costs imposed on those who bore the burden of change."
    ),
    (
        "The intellectual life of {place} during {period} was shaped by a convergence "
        "of factors that made the city one of the most {adj} centers of learning in "
        "the known world. The presence of a wealthy and relatively tolerant ruling "
        "class created conditions in which scholars, artists, and practitioners of "
        "{science} could pursue their work with a degree of freedom unusual for the "
        "time. The city's geographic position, at the intersection of trade routes "
        "linking {place2} to {place3}, ensured a steady influx of new ideas, texts, "
        "and technologies. Libraries and academies attracted scholars from diverse "
        "backgrounds, and the resulting intellectual exchanges produced a body of work "
        "whose influence extended far beyond the city's walls. {name}, a {profession} "
        "who settled in {place} after years of study in {place2}, was among the most "
        "productive scholars of this period. {name}'s contributions to {science} were "
        "{adj2}, including the {verb_past} compilation of observational data that "
        "corrected errors in earlier texts and the development of new methods for "
        "analyzing complex phenomena. {name2}, a {profession2} whose interests ranged "
        "from {abstract} to the practical arts, collaborated with {name} on several "
        "projects and independently {verb_past2} works that synthesized knowledge from "
        "multiple traditions into coherent and accessible treatises. The methods "
        "employed by these scholars reflected a commitment to {adj3} inquiry that "
        "combined respect for earlier authorities with a willingness to challenge "
        "received wisdom when evidence warranted it. This balance between tradition "
        "and innovation proved remarkably productive, and the works produced during "
        "this period continued to be studied, copied, and translated for centuries "
        "after the conditions that produced them had vanished."
    ),
]

# ---------------------------------------------------------------------------
# Technical description templates
# ---------------------------------------------------------------------------

TECHNICAL_DESCRIPTION_TEMPLATES = [
    (
        "The process of manufacturing {material} products has undergone {adj} "
        "transformation since its origins in {place} during {period}. Early methods "
        "relied on manual techniques that, while producing items of considerable "
        "quality, were limited in scale and consistency. The transition to more "
        "systematic production methods began in {century}, when {name}, a {profession} "
        "working in {place2}, developed a series of innovations that addressed the "
        "principal bottlenecks in the existing process. The most significant of these "
        "was a method for controlling the thermal profile during the critical formation "
        "stage, which had previously been the primary source of defects and batch-to-"
        "batch variation. By introducing a staged heating sequence with precise "
        "temperature targets at each stage, {name} achieved a level of consistency "
        "that had been impossible with earlier methods. The second major innovation "
        "concerned the preparation of raw materials. Traditional practice involved "
        "minimal preprocessing, with the result that impurities in the starting "
        "materials often compromised the quality of the finished product. {name}'s "
        "approach, which drew on principles from {science}, introduced a purification "
        "step that removed the most problematic contaminants without significantly "
        "increasing processing time or cost. The combination of improved thermal "
        "control and material purification produced results that were {adj2} by the "
        "standards of the time, and the method was rapidly adopted by manufacturers "
        "in {place} and throughout the region. Subsequent developments have built "
        "on this foundation, incorporating advances in automation, process monitoring, "
        "and quality assurance. Modern production facilities employ sensors and "
        "feedback systems that maintain process parameters within tolerances that "
        "would have been unimaginable to earlier practitioners. The {adj3} challenge "
        "of scaling production while maintaining quality has been addressed through "
        "the application of statistical process control methods, which enable "
        "operators to detect and correct deviations before they affect the final "
        "product. Environmental considerations have also become increasingly "
        "important, with {adj4} efforts to reduce waste, minimize energy consumption, "
        "and ensure that production processes are compatible with sustainable "
        "resource management."
    ),
    (
        "The engineering of structures built from {material} requires a {adj} "
        "understanding of the material's mechanical properties and how those properties "
        "are affected by environmental conditions. The fundamental challenge is to "
        "design structures that can safely bear their intended loads throughout their "
        "service life while remaining economically feasible to construct and maintain. "
        "This challenge has been addressed by engineers since {period}, when builders "
        "in {place} first developed empirical rules for proportioning structural "
        "elements based on accumulated experience. The formalization of these rules "
        "into mathematical models began in {century}, when {name}, a {profession} "
        "working at the academy in {place2}, {verb_past} the first {adj2} theory "
        "of structural behavior under load. This theory, while simplified by modern "
        "standards, captured the essential relationship between applied force, material "
        "properties, and structural geometry, and provided a basis for rational design "
        "that replaced the trial-and-error approach of earlier periods. The theory "
        "was extended and refined by subsequent researchers, including {name2}, whose "
        "contributions during the late {century} addressed the behavior of structures "
        "under dynamic loads, a problem of particular importance for structures "
        "exposed to wind, seismic, or traffic-induced forces. Modern structural "
        "analysis employs computational methods that can model the behavior of "
        "structures with {adj3} fidelity, accounting for nonlinear material behavior, "
        "geometric effects, and the interaction between structural elements and their "
        "foundations. These methods have made it possible to design structures that "
        "are both more efficient and more reliable than their historical predecessors. "
        "The inspection and maintenance of {material} structures remain essential "
        "components of engineering practice, as even well-designed structures are "
        "subject to degradation processes including corrosion, fatigue, and the "
        "cumulative effects of environmental exposure. Current research focuses on "
        "the development of smart monitoring systems that can detect early signs of "
        "damage and alert maintenance personnel before structural integrity is "
        "compromised."
    ),
    (
        "Water management systems have been essential to human civilization since "
        "the earliest settled communities in {place} during {period}. The fundamental "
        "engineering challenge of delivering clean water for domestic and agricultural "
        "use while safely disposing of waste water has generated a body of technical "
        "knowledge that spans millennia. The earliest systems were {adj} in their "
        "simplicity, relying on gravity-fed channels to direct water from elevated "
        "sources to points of use. In {place}, archaeological evidence reveals "
        "networks of channels, reservoirs, and settling basins that demonstrate a "
        "{adj2} understanding of hydraulic principles, even in the absence of formal "
        "mathematical analysis. The Roman aqueduct system, portions of which supplied "
        "cities comparable to {place2}, represented a {adj3} advance in scale and "
        "engineering sophistication. The design of these systems required careful "
        "surveying to establish grades, the construction of bridges and tunnels to "
        "cross obstacles, and the development of materials capable of withstanding "
        "continuous contact with flowing water. The medieval period saw a relative "
        "decline in the scale of water infrastructure in some regions, though "
        "communities in {place} and elsewhere continued to maintain and extend "
        "systems suited to local needs. The modern era of water engineering began "
        "in {century}, when the connection between contaminated water and epidemic "
        "disease was established through the work of {name}, a {profession} whose "
        "investigations in {place2} {verb_past} the mechanisms of waterborne disease "
        "transmission. This discovery transformed water management from a convenience "
        "to a public health imperative, driving the construction of treatment "
        "facilities, distribution networks, and sewerage systems on an unprecedented "
        "scale. Contemporary water engineering faces the {adj4} challenge of "
        "maintaining aging infrastructure while adapting to changing demand patterns, "
        "emerging contaminants, and the effects of climate variability on water "
        "availability. Advanced treatment technologies, including membrane filtration "
        "and ultraviolet disinfection, have expanded the toolkit available to "
        "engineers, while computational modeling supports the design and optimization "
        "of distribution networks of increasing complexity."
    ),
    (
        "The development of precision instruments for measurement in {science} "
        "represents one of the most {adj} chapters in the history of technology. "
        "The ability to quantify natural phenomena with accuracy and reproducibility "
        "is fundamental to scientific practice, and advances in instrumentation have "
        "repeatedly enabled discoveries that transformed our understanding of the "
        "natural world. The earliest measuring instruments, developed in {place} "
        "during {period}, were designed to address practical needs in surveying, "
        "navigation, and timekeeping. These instruments, typically constructed from "
        "{material} and {material2}, achieved levels of precision that were {adj2} "
        "for their time, though modest by modern standards. The principle underlying "
        "most early instruments was the subdivision of a known standard into smaller "
        "units, a process that required both skilled craftsmanship and careful "
        "calibration. {name}, a {profession} active in {place2} during {century}, "
        "introduced several innovations that significantly improved instrument "
        "accuracy, including the use of vernier scales, micrometer screws, and "
        "temperature compensation mechanisms. These innovations, initially applied "
        "to instruments for {science}, were subsequently adapted for use in other "
        "fields, demonstrating the cross-disciplinary impact of instrumental "
        "development. The industrial era brought both new demands and new "
        "capabilities. Mass production required instruments capable of verifying "
        "dimensional tolerances with {adj3} precision, and the resulting development "
        "of gauge blocks, optical comparators, and coordinate measuring machines "
        "transformed manufacturing quality control. The twentieth century saw the "
        "introduction of electronic sensors and digital readout systems, which "
        "dramatically increased both the speed and the accuracy of measurement. "
        "Contemporary instrumentation incorporates microprocessors, laser-based "
        "measurement systems, and advanced signal processing algorithms that enable "
        "measurements of physical quantities at resolutions approaching fundamental "
        "limits. The ongoing development of measurement technology continues to "
        "be driven by the interplay between scientific curiosity, technological "
        "capability, and practical need, a dynamic that has characterized the field "
        "since its earliest days in {place}."
    ),
]


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def gen_textbook_passage():
    return _fill(random.choice(TEXTBOOK_TEMPLATES))


def gen_essay():
    return _fill(random.choice(ESSAY_TEMPLATES))


def gen_historical_narrative():
    return _fill(random.choice(HISTORICAL_NARRATIVE_TEMPLATES))


def gen_technical_description():
    return _fill(random.choice(TECHNICAL_DESCRIPTION_TEMPLATES))


# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("gen_textbook_passage", 8000, gen_textbook_passage),
    ("gen_essay", 8000, gen_essay),
    ("gen_historical_narrative", 6000, gen_historical_narrative),
    ("gen_technical_description", 6000, gen_technical_description),
]


def main():
    random.seed(42)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "raw_texts" / "additional_synthetic"
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

            filename = f"additional_{cat_name}_{i:06d}.txt"
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
