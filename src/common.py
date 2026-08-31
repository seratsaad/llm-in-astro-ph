"""Shared configuration: paths, the frozen marker basket, cohort definition."""
import os
import re

KG = os.environ.get("LLMH_KG", os.path.expanduser("~/Downloads/astroph_kg"))
ROOT = os.environ.get("LLMH_ROOT",
                      os.path.expanduser("~/astro_ph_llm_use_hierarchical"))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
LOGS = os.path.join(ROOT, "logs")

# ---------------------------------------------------------------- marker basket
# Frozen before any model fitting: the 38 tokens published in the abstract-only
# analysis (Saad 2026), themselves taken from Liang et al. 2024, Gray 2025 and
# Kobak et al. 2024. STRONG = rare, high-specificity; SOFT = commoner words with
# a higher background rate.
MARKERS_STRONG = [
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "intricate", "intricacies", "showcasing", "showcases", "boasts", "tapestry",
    "realm", "realms", "pivotal", "meticulous", "meticulously", "nuanced",
]
MARKERS_SOFT = [
    "comprehensive", "leverage", "leveraging", "harness", "harnessing",
    "notably", "crucial", "robust", "seamless", "seamlessly", "garner",
    "encompass", "encompasses", "multifaceted", "insights", "unravel",
    "elucidate", "myriad", "plethora", "holistic",
]
MARKERS = MARKERS_STRONG + MARKERS_SOFT
assert len(MARKERS) == 38, len(MARKERS)

# Neutral astronomy vocabulary: must not hockey-stick. Negative control.
CONTROL = [
    "observed", "measured", "obtained", "presented", "galaxy", "stellar",
    "sample", "temperature", "redshift", "spectra",
]

# Placebo basket: frequency-matched to MARKERS but with no LLM association.
# Chosen for comparable pre-2020 document frequency, not by any post-2022 signal.
PLACEBO = [
    "derived", "estimated", "corresponding", "significant", "consistent",
    "typical", "moderate", "apparent", "distinct", "substantial",
    "considerable", "evident", "notable", "reliable", "adequate",
]

WATCH = sorted(set(MARKERS + CONTROL + PLACEBO))

# ---------------------------------------------------------------- cohort
YEAR_MIN, YEAR_MAX = 2015, 2026
QUARTER_MAX = (2026, 2)          # last complete quarter in the release
KNOWN_NEGATIVE_BEFORE = 2020     # primary specification
ADOPTION_QUARTER = (2022, 4)     # ChatGPT public release, 2022-11-30

TOKEN_RE = re.compile(r"[a-z]+")
NEW_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")


def id_to_ym(arxiv_id):
    """First-submission (year, month) from a new-style arXiv identifier."""
    m = NEW_ID_RE.match(arxiv_id)
    if not m:
        return None
    return 2000 + int(m.group(1)), int(m.group(2))


def quarter_index(year, month):
    """Integer quarter index with 2015Q1 == 0."""
    q = (month - 1) // 3
    return (year - YEAR_MIN) * 4 + q


def quarter_label(idx):
    y = YEAR_MIN + idx // 4
    return f"{y}Q{idx % 4 + 1}"


N_QUARTERS = quarter_index(*QUARTER_MAX) + 1
