"""
normalize.py
============================================================
PRESERVED: normalize_text / normalize_business_name /
normalize_address are UNCHANGED from the original implementation
and are still what blocking.py's ASCII-based keys (NP6, NP8, NS6,
FL, FL3, F2, FML, AN, A2, AW2, AL2, NL, A4W, AT2) and the
frequency-cap tuning in blocking.DEFAULT_FREQ_CAPS were built and
measured against. Do not change their behavior -- that would
silently invalidate every recall number in output/*.txt.

ADDED: normalize_unicode_preserving(), used by features.py (not
blocking.py, which already had its own equivalent logic inlined
in make_unicode_name_keys). This closes a real gap found during
the Phase 1 audit:

  normalize_business_name() / normalize_address() strip every
  character outside [a-z0-9\\s] AFTER NFKD decomposition. NFKD
  decomposition only helps scripts where the accent is a separate
  combining character on top of a Latin base letter (e.g. "e" +
  combining acute). It does nothing for Devanagari, Telugu,
  Kannada, Bengali, etc. -- those characters simply have no
  a-z0-9 equivalent, so they are deleted outright.

  Example measured in output/s2_missed_matches_inspection.tsv:
    "एसएस सिस्टम्स लिमिटेड"  -->  normalize_business_name()  -->  ""

  blocking.py's make_unicode_name_keys() already generates
  candidate-blocking keys straight from the untouched Unicode
  string, which is why these pairs DO show up as candidates.
  But features.py currently computes every name/address
  similarity feature from normalize_business_name() /
  normalize_address() alone. Once both sides normalize to "",
  every similarity feature for that pair becomes 0 (or the
  hard-coded "missing" default) regardless of whether the two
  Unicode strings are near-identical or completely unrelated.
  The ML model in Phase 2 would then have no signal at all to
  distinguish a true match from a false positive on this
  subset -- i.e. blocking recall is fine for non-Latin scripts,
  but downstream precision/recall on those rows would collapse.

  normalize_unicode_preserving() keeps letters/digits/combining
  marks in ANY script (via unicodedata category, not a Latin
  regex) so features.py can compute a second, script-agnostic
  set of similarity features and fall back to it whenever the
  ASCII-folded normalization is empty on either side.
"""

import re
import unicodedata


def normalize_text(text):
    """
    Basic normalization for business names and addresses.
    UNCHANGED -- do not modify, see module docstring.
    """

    if not text:
        return ""

    text = str(text)
    text = text.lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(name):
    """
    UNCHANGED -- do not modify, see module docstring.
    """

    name = normalize_text(name)

    suffixes = [
        "incorporated",
        "inc",
        "corporation",
        "corp",
        "limited",
        "ltd",
        "llc",
        "company",
        "co",
        "private limited",
        "pvt ltd",
    ]

    words = name.split()

    while words and " ".join(words[-2:]) in suffixes:
        words = words[:-2]

    while words and words[-1] in suffixes:
        words = words[:-1]

    return " ".join(words)


def normalize_address(address):
    """
    UNCHANGED -- do not modify, see module docstring.
    """

    address = normalize_text(address)

    replacements = {
        " street ": " st ",
        " road ": " rd ",
        " avenue ": " ave ",
        " boulevard ": " blvd ",
        " drive ": " dr ",
        " lane ": " ln ",
        " highway ": " hwy ",
        " apartment ": " apt ",
    }

    address = " " + address + " "

    for old, new in replacements.items():
        address = address.replace(old, new)

    return address.strip()


# ============================================================
# NEW: script-agnostic normalization for FEATURE calculation
# ============================================================

# A small set of business-suffix WORDS that are meaningful in
# transliterated/English form even inside an otherwise non-Latin
# string (e.g. "Pvt Ltd" tacked onto a Devanagari name). Matching
# the ASCII suffix list keeps this consistent with
# normalize_business_name() rather than inventing a second list.
_ASCII_SUFFIX_WORDS = {
    "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "llc", "company", "co", "private", "pvt",
}


def normalize_unicode_preserving(text):
    """
    Like normalize_text(), but keeps letters/digits/marks from
    ANY script instead of only [a-z0-9]. Used by features.py as a
    fallback whenever normalize_business_name()/normalize_address()
    collapse to "" for a non-Latin string, so similarity features
    still carry real signal for India-language business names.

    NOT used by blocking.py: blocking already has its own,
    independently-tuned Unicode key logic (make_unicode_name_keys)
    and changing that would move the measured recall numbers.
    """

    if not text:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).casefold()

    out_chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith("L") or category.startswith("N") or category.startswith("M"):
            out_chars.append(ch)
        else:
            out_chars.append(" ")

    text = re.sub(r"\s+", " ", "".join(out_chars)).strip()

    return text


def normalize_business_name_script_agnostic(name):
    """
    normalize_unicode_preserving() plus trailing-suffix stripping,
    for business names specifically. Suffix stripping only removes
    ASCII suffix words (see _ASCII_SUFFIX_WORDS); non-Latin legal
    suffixes are left in place since a hard-coded ASCII list can't
    identify them, and leaving them in does not hurt similarity
    scoring the way an empty string does.
    """

    text = normalize_unicode_preserving(name)
    if not text:
        return ""

    words = text.split()
    while words and words[-1] in _ASCII_SUFFIX_WORDS:
        words = words[:-1]

    return " ".join(words)


def is_non_latin(text):
    """
    True if the ASCII-folded normalization would lose real content
    -- i.e. normalize_business_name(text) would be "" or much
    shorter than normalize_unicode_preserving(text). Used by
    features.py to decide whether to trust the ASCII-folded
    features or fall back to the script-agnostic ones.
    """

    ascii_form = normalize_text(text)
    unicode_form = normalize_unicode_preserving(text)

    if not unicode_form:
        return False

    return len(ascii_form.replace(" ", "")) < 0.5 * len(unicode_form.replace(" ", ""))


if __name__ == "__main__":

    examples = [
        "Payne Enterprises",
        "एसएस सिस्टम्स लिमिटेड",
        "శివ మేనేజ్‌మెంట్ ప్రైవేట్ లిమిటెడ్",
        "ಬಾಂಬೆ ಪವರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
    ]

    for name in examples:
        print(f"{name!r}")
        print(f"  ascii-folded : {normalize_business_name(name)!r}")
        print(f"  script-agnostic: {normalize_business_name_script_agnostic(name)!r}")
        print(f"  is_non_latin : {is_non_latin(name)}")
