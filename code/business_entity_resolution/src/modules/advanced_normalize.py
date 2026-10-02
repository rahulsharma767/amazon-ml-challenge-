"""Additional normalization views for the Phase-2 matcher.

These views are deliberately separate from normalize.py's legacy functions so
previous recall measurements remain reproducible. They are used only by the
new feature/retrieval layer.
"""
import re
import unicodedata

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}
_ORDINAL_WORDS = {
    "first":"1", "second":"2", "third":"3", "fourth":"4", "fifth":"5",
    "sixth":"6", "seventh":"7", "eighth":"8", "ninth":"9", "tenth":"10",
    "eleventh":"11", "twelfth":"12", "thirteenth":"13", "fourteenth":"14",
    "fifteenth":"15", "sixteenth":"16", "seventeenth":"17", "eighteenth":"18",
    "nineteenth":"19", "twentieth":"20", "thirtieth":"30", "fortieth":"40",
    "fiftieth":"50", "sixtieth":"60", "seventieth":"70", "eightieth":"80",
    "ninetieth":"90", "hundredth":"100",
}
_SUFFIXES = {"st","nd","rd","th"}

def normalize_address_numbers(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    # Ordinal digits: 21st -> 21.
    s = re.sub(r"(?<!\w)(\d{1,6})(?:st|nd|rd|th)(?!\w)", r"\1", s)
    tokens = re.findall(r"\d+|[a-z]+|[^\w\s]+", s)
    out=[]
    i=0
    while i < len(tokens):
        t=tokens[i]
        if t in _ORDINAL_WORDS:
            out.append(_ORDINAL_WORDS[t]); i+=1; continue
        # two-word ordinals/numbers such as twenty first / twenty one
        if i+1 < len(tokens) and tokens[i] in {"twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"}:
            a=_NUMBER_WORDS.get(tokens[i]); b=_NUMBER_WORDS.get(tokens[i+1]) or _ORDINAL_WORDS.get(tokens[i+1])
            if a and b and b.isdigit() and int(b) < 10:
                out.append(str(int(a)+int(b))); i+=2; continue
        if t.isdigit():
            out.append(str(int(t)))  # strips leading zeroes
        else:
            out.append(t)
        i+=1
    return re.sub(r"\s+", " ", " ".join(out)).strip()

def token_sorted(text: str) -> str:
    if not text: return ""
    s=unicodedata.normalize("NFKC", str(text)).casefold()
    toks=re.findall(r"[\w]+", s, flags=re.UNICODE)
    return " ".join(sorted(set(toks)))

def compact_alnum(text: str) -> str:
    if not text: return ""
    s=unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(ch for ch in s if ch.isalnum())

def transliterate_ascii(text: str) -> str:
    """Best-effort local transliteration without external identity data.

    Unidecode is optional. If unavailable, retain Unicode rather than silently
    deleting it. This is a feature view, never a blocking hard constraint.
    """
    if not text: return ""
    try:
        from unidecode import unidecode
        return re.sub(r"\s+", " ", unidecode(text).casefold()).strip()
    except Exception:
        return str(text).casefold().strip()
