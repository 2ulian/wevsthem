"""
Othering Detector — Week 3 of "We vs Them" NLP project.

Detects "othering" language in social media posts using four pattern families:
  1. Dehumanizing metaphors  (invasion, flood, swarm…)
  2. Moral exclusion         (don't belong, go back…)
  3. Generalization          (all of them, these people always…)
  4. Threat framing          (taking over, replacing us…)

Main public API
---------------
detect_othering(text) -> dict
    Returns has_othering (bool), othering_score (int 0-4),
    matched_patterns (list[str])

apply_othering(df, text_col="clean_text") -> pd.DataFrame
    Adds has_othering, othering_score, matched_patterns columns to df.
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Pattern definitions — each family is a list of (label, compiled regex) pairs
# ---------------------------------------------------------------------------

def _build_patterns(raw: list[tuple[str, str]]) -> list[tuple[str, re.Pattern]]:
    """Compile raw (label, pattern) pairs into (label, re.Pattern) tuples."""
    return [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in raw]


DEHUMANIZING = _build_patterns([
    ("invasion",      r"\b(invasion|invade|invading)\b"),
    ("flood",         r"\b(flood(ing)?|flooded)\b"),
    ("swarm",         r"\b(swarm(ing)?|swarmed)\b"),
    ("infestation",   r"\b(infest(ation|ed|ing)?)\b"),
    ("plague",        r"\b(plague[ds]?|plaguing)\b"),
    ("vermin",        r"\b(vermin|cockroach(es)?|rat[s]?)\b"),
    ("animals",       r"\b(animals?|beast[s]?|savages?|barbarian[s]?)\b"),
    ("parasite",      r"\b(parasite[s]?|parasitic)\b"),
])

MORAL_EXCLUSION = _build_patterns([
    ("dont_belong",   r"\bdon'?t\s+belong\b"),
    ("go_back",       r"\bgo\s+back\s+(to\s+)?(where|your)\b"),
    ("not_like_us",   r"\bnot\s+like\s+us\b"),
    ("their_kind",    r"\btheir\s+kind\b"),
    ("not_welcome",   r"\bnot\s+welcome\s+here\b"),
    ("get_out",       r"\b(get\s+out|leave\s+our\s+(country|land|nation))\b"),
    ("no_place_here", r"\bno\s+place\s+(here|in\s+this\s+country)\b"),
    ("send_them_back",r"\bsend\s+them\s+back\b"),
])

GENERALIZATION = _build_patterns([
    ("all_of_them",      r"\ball\s+of\s+them\b"),
    ("these_ppl_always", r"\bthese\s+people\s+always\b"),
    ("they_never",       r"\bthey\s+never\b"),
    ("they_all",         r"\bthey'?re?\s+all\b"),
    ("all_they_do",      r"\ball\s+they\s+(do|want|care)\b"),
    ("every_single_one", r"\bevery\s+single\s+one\s+of\s+them\b"),
    ("those_people",     r"\bthose\s+(kind\s+of\s+)?people\b"),
    ("none_of_them",     r"\bnone\s+of\s+them\b"),
])

THREAT_FRAMING = _build_patterns([
    ("taking_over",   r"\btaking\s+over\b"),
    ("replacing_us",  r"\breplacing\s+us\b"),
    ("replacement",   r"\b(great\s+)?replacement\b"),
    ("destroying_our",r"\bdestroying\s+our\b"),
    ("under_attack",  r"\b(under\s+attack|being\s+attacked)\b"),
    ("erase_us",      r"\beras(e|ing|ed)\s+us\b"),
    ("our_culture",   r"\b(destroy|destroy(ing)?|threaten(ing)?|eradicate)\s+our\s+(culture|way\s+of\s+life|values|heritage)\b"),
    ("white_genocide",r"\b(white\s+genocide|demographic\s+replacement)\b"),
])

# Map family name → list of patterns
ALL_FAMILIES: dict[str, list[tuple[str, re.Pattern]]] = {
    "dehumanizing":    DEHUMANIZING,
    "moral_exclusion": MORAL_EXCLUSION,
    "generalization":  GENERALIZATION,
    "threat_framing":  THREAT_FRAMING,
}


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_othering(text: str) -> dict:
    """
    Analyse one text and return othering indicators.

    Parameters
    ----------
    text : str
        Raw or cleaned post text.

    Returns
    -------
    dict with keys:
        has_othering    : bool   — True if at least one pattern matched
        othering_score  : int    — number of families that matched (0-4)
        matched_patterns: list   — list of matched pattern labels
    """
    if not isinstance(text, str) or not text.strip():
        return {"has_othering": False, "othering_score": 0, "matched_patterns": []}

    matched: list[str] = []
    families_hit: set[str] = set()

    for family, patterns in ALL_FAMILIES.items():
        for label, regex in patterns:
            if regex.search(text):
                matched.append(label)
                families_hit.add(family)

    return {
        "has_othering":     len(families_hit) > 0,
        "othering_score":   len(families_hit),   # 0-4
        "matched_patterns": matched,
    }


# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------

def apply_othering(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Apply detect_othering to every row and add three columns in place.

    Parameters
    ----------
    df       : pd.DataFrame — must contain `text_col`
    text_col : str          — column holding the text to analyse

    Returns
    -------
    pd.DataFrame with added columns:
        has_othering, othering_score, matched_patterns
    """
    results = df[text_col].fillna("").apply(detect_othering)
    results_df = pd.DataFrame(results.tolist(), index=df.index)
    df = df.copy()
    df["has_othering"]     = results_df["has_othering"]
    df["othering_score"]   = results_df["othering_score"]
    df["matched_patterns"] = results_df["matched_patterns"]
    return df
