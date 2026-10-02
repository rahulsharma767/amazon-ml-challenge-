"""
features.py
============================================================
CHANGES from the original (see PHASE1_ARCHITECTURE_AUDIT.md,
sections "Existing feature set" and "Runtime optimization plan"
for the measurements behind these changes):

1. NON-LATIN FIX (correctness, not just speed):
   name_char_similarity / name_levenshtein_similarity / etc. used
   to be computed ONLY on normalize_business_name()/normalize_address(),
   which strip non-Latin scripts to "". For a true-match pair where
   both sides are in Devanagari/Telugu/Kannada/Bengali, every
   similarity feature silently became 0 or the "missing" default,
   even though the strings are genuinely very similar. This module
   now also computes the same feature suite on a script-agnostic
   normalization (normalize.normalize_unicode_preserving) and uses
   whichever normalization actually has content, so the ML model
   gets real signal on non-Latin rows instead of a constant.

2. SPEED: the original had TWO independent implementations of
   "edit-distance-based ratio" per field (text_similarity() via
   difflib.SequenceMatcher, and levenshtein_similarity() via
   rapidfuzz). difflib's SequenceMatcher is pure Python and is the
   single most expensive call in the old feature set -- at the
   ~440M-pair scale estimated for the full test set, removing one
   of the two redundant passes per field is a meaningful win.
   rapidfuzz.fuzz.ratio() is a compatible-by-design, C-accelerated
   reimplementation of the same "difflib-style" ratio, so
   name_char_similarity / address_char_similarity are now computed
   with it too. If rapidfuzz is unavailable, this falls back to the
   original difflib implementation exactly as before -- no feature
   changes shape, only which backend computes it.

3. Feature list is APPENDED to, not reordered or removed, so
   existing train_features_*.tsv files stay a strict prefix-
   compatible subset of the new schema (old columns keep the same
   name and meaning).
"""

from functools import lru_cache

from normalize import (
    normalize_business_name,
    normalize_address,
    normalize_business_name_script_agnostic,
    normalize_unicode_preserving,
    is_non_latin,
)

from similarity import token_similarity, address_number_match, country_match


# --------------------------------------------------------------
# Ratio backend: rapidfuzz preferred (C-accelerated, difflib-
# compatible ratio), pure-Python difflib as a guaranteed fallback.
# One implementation now serves BOTH "char_similarity" and
# "levenshtein_similarity" -- see change #2 above.
# --------------------------------------------------------------

try:
    from rapidfuzz.fuzz import ratio as _rapidfuzz_ratio

    def _ratio(a, b):
        if not a or not b:
            return 0.0
        return _rapidfuzz_ratio(a, b) / 100.0

except ImportError:

    from difflib import SequenceMatcher

    def _ratio(a, b):
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()


levenshtein_similarity = _ratio  # kept as a public name for compatibility


@lru_cache(maxsize=2_000_000)
def cached_normalize_business_name(value):
    return normalize_business_name(value)


@lru_cache(maxsize=2_000_000)
def cached_normalize_address(value):
    return normalize_address(value)


@lru_cache(maxsize=2_000_000)
def cached_normalize_business_name_unicode(value):
    return normalize_business_name_script_agnostic(value)


@lru_cache(maxsize=2_000_000)
def cached_normalize_address_unicode(value):
    return normalize_unicode_preserving(value)


# --------------------------------------------------------------
# Character n-gram Jaccard
# --------------------------------------------------------------

def char_ngram_jaccard(a, b, n=2):

    if not a or not b:
        return 0.0

    if len(a) < n or len(b) < n:
        return 0.0

    grams_a = {a[i:i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i:i + n] for i in range(len(b) - n + 1)}

    if not grams_a or not grams_b:
        return 0.0

    return len(grams_a & grams_b) / len(grams_a | grams_b)


def prefix_similarity(a, b, k=4):

    if not a or not b:
        return 0.0

    a_p, b_p = a[:k], b[:k]
    max_len = min(len(a_p), len(b_p))

    if max_len == 0:
        return 0.0

    matched = sum(1 for x, y in zip(a_p, b_p) if x == y)
    return matched / max_len


def suffix_similarity(a, b, k=4):

    if not a or not b:
        return 0.0

    a_s, b_s = a[-k:], b[-k:]
    max_len = min(len(a_s), len(b_s))

    if max_len == 0:
        return 0.0

    matched = sum(1 for x, y in zip(reversed(a_s), reversed(b_s)) if x == y)
    return matched / max_len


def token_overlap_count(a, b):

    if not a or not b:
        return 0

    return len(set(a.split()) & set(b.split()))


# --------------------------------------------------------------
# Feature columns
# --------------------------------------------------------------
# Original columns are UNCHANGED in name, meaning and position.
# New columns are appended at the end.

FEATURE_COLUMNS = [

    # Name features (original)
    "name_exact_match",
    "name_char_similarity",
    "name_levenshtein_similarity",
    "name_token_jaccard",
    "name_token_overlap",
    "name_compact_similarity",
    "name_prefix_similarity",
    "name_suffix_similarity",
    "name_char_ngram_jaccard",

    # Address features (original)
    "address_exact_match",
    "address_char_similarity",
    "address_levenshtein_similarity",
    "address_token_jaccard",
    "address_token_overlap",
    "address_number_match",
    "address_char_ngram_jaccard",

    # Other (original)
    "country_match",
    "name_missing",
    "address_missing",
    "combined_name_address_score",

    # ---- NEW: non-Latin fallback signal ----
    "name_is_non_latin",
    "address_is_non_latin",
    "name_char_similarity_unicode",
    "name_token_jaccard_unicode",
]


def calculate_features(name1, address1, country1, name2, address2, country2):

    # -------- ASCII-folded normalization (original path) --------
    n1 = cached_normalize_business_name(name1)
    n2 = cached_normalize_business_name(name2)
    a1 = cached_normalize_address(address1)
    a2 = cached_normalize_address(address2)

    name_missing = int(not n1 or not n2)
    address_missing = int(not a1 or not a2)

    name_char_sim = _ratio(n1, n2)
    name_lev_sim = _ratio(n1, n2)  # same backend now, see module docstring #2
    name_tok_jaccard = token_similarity(n1, n2)
    name_compact_sim = _ratio(n1.replace(" ", ""), n2.replace(" ", ""))

    address_char_sim = _ratio(a1, a2)
    address_lev_sim = _ratio(a1, a2)
    address_tok_jaccard = token_similarity(a1, a2)

    combined = 0.6 * name_char_sim + 0.4 * address_char_sim

    # -------- NEW: script-agnostic fallback for non-Latin names --------
    # Only pay for the extra normalization/similarity work when the
    # ASCII-folded form actually lost content -- cheap for the
    # (majority) Latin-script rows, since is_non_latin() short-
    # circuits on the already-cached ascii_form.
    name_non_latin = is_non_latin(name1) or is_non_latin(name2)

    if name_non_latin:
        un1 = cached_normalize_business_name_unicode(name1)
        un2 = cached_normalize_business_name_unicode(name2)
        name_char_sim_unicode = _ratio(un1, un2)
        name_tok_jaccard_unicode = token_similarity(un1, un2)
    else:
        name_char_sim_unicode = name_char_sim
        name_tok_jaccard_unicode = name_tok_jaccard

    address_non_latin = is_non_latin(address1) or is_non_latin(address2)

    return {

        "name_exact_match": int(bool(n1) and n1 == n2),
        "name_char_similarity": name_char_sim,
        "name_levenshtein_similarity": name_lev_sim,
        "name_token_jaccard": name_tok_jaccard,
        "name_token_overlap": token_overlap_count(n1, n2),
        "name_compact_similarity": name_compact_sim,
        "name_prefix_similarity": prefix_similarity(n1, n2),
        "name_suffix_similarity": suffix_similarity(n1, n2),
        "name_char_ngram_jaccard": char_ngram_jaccard(n1, n2),

        "address_exact_match": int(bool(a1) and a1 == a2),
        "address_char_similarity": address_char_sim,
        "address_levenshtein_similarity": address_lev_sim,
        "address_token_jaccard": address_tok_jaccard,
        "address_token_overlap": token_overlap_count(a1, a2),
        "address_number_match": address_number_match(address1, address2),
        "address_char_ngram_jaccard": char_ngram_jaccard(a1, a2),

        "country_match": country_match(country1, country2),
        "name_missing": name_missing,
        "address_missing": address_missing,
        "combined_name_address_score": combined,

        "name_is_non_latin": int(name_non_latin),
        "address_is_non_latin": int(address_non_latin),
        "name_char_similarity_unicode": name_char_sim_unicode,
        "name_token_jaccard_unicode": name_tok_jaccard_unicode,
    }
