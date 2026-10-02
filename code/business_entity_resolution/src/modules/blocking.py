"""
blocking.py
============================================================
Blocking-key generation for Business Entity Resolution.

This module is SOURCE-INDEPENDENT: the same functions are used
for Source 1, Source 2 and Source 3, and for both the 10K
development data and the full 5M+ record production data.

Nothing in this file is hard-coded to any dataset size. Frequency
thresholds are NOT computed here -- they are computed by indexer.py
against whatever data was actually indexed (10K or full), which is
what makes this pipeline safe to move from lab -> production.

Key types produced:

  Name keys   : NP6, NP8, NS6, FL, FL3, F2, FML
  Address keys: AN, A2, AW2, AL2, NL
  A4 keys     : A4W (numeric address component + nearby address word,
                windowed so cost is O(n) in the number of address
                tokens, not O(n^2))
"""

import re

from normalize import normalize_business_name, normalize_address



# ============================================================
# UNICODE NAME BLOCKING
# ============================================================

def make_unicode_name_keys(name):
    """
    Unicode-preserving name keys.

    Keeps Hindi, Kannada, Gujarati, etc. instead of deleting
    non-ASCII characters.
    """
    if not name:
        return []

    import unicodedata

    n = unicodedata.normalize("NFKC", str(name)).casefold()
    n = " ".join(n.split())

    if not n:
        return []

    keys = []

    compact = "".join(ch for ch in n if not ch.isspace())

    if len(compact) >= 4:
        keys.append(("UNP4", compact[:4]))

    if len(compact) >= 6:
        keys.append(("UNP6", compact[:6]))

    words = n.split()

    if len(words) >= 2:
        first = words[0]
        last = words[-1]

        if len(first) >= 2 and len(last) >= 2:
            keys.append(("UFL", first[:3] + "|" + last[:3]))

    return keys


# ============================================================
# NAME BLOCKING
# ============================================================

def make_name_keys(name):
    """
    Generate selective name-blocking keys.

    We avoid generic single-word keys because they create
    enormous candidate buckets. Logic preserved from the
    original candidate_generator.py (already measured at
    ~92% S2 recall on the 1,000-row dev sample).
    """

    n = normalize_business_name(name)

    if not n:
        return []

    keys = []
    compact = n.replace(" ", "")

    if len(compact) >= 6:
        keys.append(("NP6", compact[:6]))

    if len(compact) >= 8:
        keys.append(("NP8", compact[:8]))

    if len(compact) >= 6:
        keys.append(("NS6", compact[-6:]))

    words = n.split()

    if len(words) >= 2:
        first, last = words[0], words[-1]

        if len(first) >= 3 and len(last) >= 3:
            keys.append(("FL", first[:4] + "|" + last[:4]))
            keys.append(("FL3", first[:3] + "|" + last[:5]))

        w1, w2 = words[0], words[1]

        if len(w1) >= 3 and len(w2) >= 3:
            keys.append(("F2", w1[:5] + "|" + w2[:5]))

    if len(words) >= 3:
        first, middle, last = words[0], words[-2], words[-1]

        if len(first) >= 3 and len(last) >= 3:
            keys.append(
                ("FML", first[:3] + "|" + middle[:3] + "|" + last[:3])
            )

    return keys


# ============================================================
# ADDRESS BLOCKING (original key set)
# ============================================================

def make_address_keys(address):
    """
    Generate selective address-blocking keys.

    We deliberately do NOT use a bare number as a key -- logic
    preserved from the original candidate_generator.py.
    """

    a = normalize_address(address)

    if not a:
        return []

    keys = []
    words = a.split()

    if not words:
        return []

    numbers = [w for w in words if w.isdigit()]
    non_numeric = [w for w in words if not w.isdigit()]

    if numbers:
        number = numbers[0]

        for word in words:
            if not word.isdigit() and len(word) >= 3:
                keys.append(("AN", number + "|" + word[:6]))
                break

        if len(non_numeric) >= 2:
            keys.append(
                (
                    "A2",
                    number + "|" + non_numeric[0][:4] + "|" + non_numeric[1][:4],
                )
            )

    if len(non_numeric) >= 2:
        w1, w2 = non_numeric[0], non_numeric[1]

        if len(w1) >= 3 and len(w2) >= 3:
            keys.append(("AW2", w1[:5] + "|" + w2[:5]))

        w1, w2 = non_numeric[-2], non_numeric[-1]

        if len(w1) >= 3 and len(w2) >= 3:
            keys.append(("AL2", w1[:5] + "|" + w2[:5]))

    if numbers and non_numeric:
        number = numbers[0]
        last_word = non_numeric[-1]

        if len(last_word) >= 3:
            keys.append(("NL", number + "|" + last_word[:6]))

    return keys



# ============================================================
# AT2 BLOCKING (informative address-token pairs)
# ============================================================

# Very common address words are poor blocking signals.
# Keep this list deliberately small.
GENERIC_ADDRESS_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave",
    "lane", "ln", "drive", "dr", "boulevard", "blvd",
    "highway", "hwy", "near", "opposite", "opp",
    "building", "bldg", "floor", "fl", "block",
    "district", "city", "state", "india",
    "private", "limited", "ltd", "pvt",
}

def make_at2_keys(address):
    """
    Generate a small number of informative address-token-pair keys.

    Unlike AW2/AL2, this does not depend on token position.
    It selects informative tokens and creates only adjacent pairs
    after sorting by their original position.

    The goal is to recover matches where the business name is in
    another script but the address is shared.
    """
    a = normalize_address(address)

    if not a:
        return []

    tokens = a.split()

    informative = []

    for tok in tokens:
        if tok.isdigit():
            continue

        if len(tok) < 4:
            continue

        if tok in GENERIC_ADDRESS_WORDS:
            continue

        informative.append(tok)

    if len(informative) < 2:
        return []

    # Keep at most 6 informative tokens per address.
    # Prefer longer tokens because they are generally more distinctive.
    informative = sorted(
        informative,
        key=lambda x: (-len(x), x)
    )[:6]

    keys = []
    seen = set()

    # Only 5-6 pairs, rather than O(n^2) over the whole address.
    for i in range(len(informative) - 1):
        w1 = informative[i][:8]
        w2 = informative[i + 1][:8]

        if w1 == w2:
            continue

        a1, a2 = sorted((w1, w2))
        key_value = a1 + "|" + a2

        if key_value not in seen:
            seen.add(key_value)
            keys.append(("AT2", key_value))

    return keys


# ============================================================
# A4 BLOCKING (numeric address component + NEARBY words)
# ============================================================
#
# The original A4 experiment (test_a4_overhead_fast.py) paired every
# number in the address with EVERY pair of words (O(n^2) per record).
# That is fine for a one-off 1,000-row experiment but is not safe to
# run unmodified against 5M+ records: addresses with many tokens
# would generate a large number of keys per record.
#
# This production version keeps the same idea -- "numeric address
# component + nearby normalized address words" -- but only looks at
# a small window of words around each number (WINDOW on each side),
# making key generation O(n) per record instead of O(n^2). Recall
# on the dev set should stay close to the original experiment
# because true matches almost always keep numbers and street words
# adjacent to one another.

A4_WINDOW = 3  # words examined on each side of a number


def make_a4_keys(address, window=A4_WINDOW):

    a = normalize_address(address)

    if not a:
        return []

    tokens = a.split()

    number_positions = [i for i, t in enumerate(tokens) if t.isdigit()]

    if not number_positions:
        return []

    keys = []
    seen = set()

    for pos in number_positions:

        number = tokens[pos]

        nearby = []

        for offset in range(1, window + 1):

            for idx in (pos - offset, pos + offset):

                if 0 <= idx < len(tokens):

                    tok = tokens[idx]

                    if not tok.isdigit() and len(tok) >= 2:
                        nearby.append(tok)

        # de-dup while preserving order, cap to the 3 closest words
        nearby = list(dict.fromkeys(nearby))[:3]

        for i in range(len(nearby)):
            for j in range(i + 1, len(nearby)):

                w1, w2 = sorted((nearby[i][:6], nearby[j][:6]))
                key_value = number + "|" + w1 + "|" + w2

                if key_value not in seen:
                    seen.add(key_value)
                    keys.append(("A4W", key_value))

    return keys


# ============================================================
# COMBINED
# ============================================================

def make_all_keys(name, address):
    """
    Convenience helper returning every (key_type, key_value) pair
    for a record's name + address, used by indexer.py.
    """

    return make_name_keys(name) + make_unicode_name_keys(name) + make_address_keys(address) + make_a4_keys(address)


NAME_KEY_TYPES = {"NP6", "NP8", "NS6", "FL", "FL3", "F2", "FML", "UNP4", "UNP6", "UFL"}
ADDRESS_KEY_TYPES = {"AN", "A2", "AW2", "AL2", "NL", "A4W", "AT2"}
ALL_KEY_TYPES = NAME_KEY_TYPES | ADDRESS_KEY_TYPES


# Default per-key-type maximum bucket size ("frequency cap").
# If a specific key's bucket (for the dataset actually indexed --
# 10K dev OR full production) grows past this size, that key is
# treated as too broad and skipped for that record. This is what
# keeps AL2 / NS6 usable without deleting them outright, and is
# recomputed fresh every time build_index() runs, so it is always
# evaluated against the CURRENT index, never against stale 10K
# statistics.
DEFAULT_FREQ_CAPS = {
    "NP6": 300,
    "NP8": 300,
    "NS6": 300,
    "FL": 400,
    "FL3": 400,
    "F2": 400,
    "FML": 400,
    "AN": 400,
    "A2": 400,
    "AW2": 400,
    "AL2": 120,
    "NL": 400,
    "A4W": 15,
    "AT2": 30,
    "UNP4": 300,
    "UNP6": 300,
    "UFL": 300,
}
