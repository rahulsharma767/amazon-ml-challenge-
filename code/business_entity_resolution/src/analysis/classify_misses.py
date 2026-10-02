import json, re, unicodedata, difflib
from collections import Counter
import sys
sys.path.insert(0, '.')
from normalize import normalize_business_name, normalize_address

data = json.load(open('/home/claude/work/miss_records.json'))
s1r, s2r, s3r = data['s1'], data['s2'], data['s3']

s2ck = json.load(open('/home/claude/work/checkpoint/checkpoints/recall_s2_full_DONE.json'))
s3ck = json.load(open('/home/claude/work/checkpoint/checkpoints/recall_s3_full_DONE.json'))

def is_non_latin(s):
    for ch in s:
        if ch.isalpha() and ord(ch) > 0x2FF:
            return True
    return False

def tok_jaccard(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def char_ratio(a, b):
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def classify(s1_id, other_id, other_recs, source_ck_examples):
    s1 = s1r.get(s1_id)
    o = other_recs.get(other_id)
    if s1 is None or o is None:
        return "record_not_found_in_scan"

    name1, addr1, c1 = s1.get('business_name',''), s1.get('business_address',''), s1.get('country','')
    name2, addr2, c2 = o.get('business_name',''), o.get('business_address',''), o.get('country','')

    n1n, n2n = normalize_business_name(name1), normalize_business_name(name2)
    a1n, a2n = normalize_address(addr1), normalize_address(addr2)

    non_latin = is_non_latin(name1) or is_non_latin(name2)

    country_mismatch = bool(c1) and bool(c2) and c1.strip().upper() != c2.strip().upper()
    country_missing = (not c1) or (not c2)

    name_tok_j = tok_jaccard(n1n, n2n)
    addr_tok_j = tok_jaccard(a1n, a2n)
    name_char = char_ratio(n1n, n2n)
    addr_char = char_ratio(a1n, a2n)

    if non_latin and (name_tok_j == 0 and name_char < 0.4):
        return "non_latin_script_no_ascii_overlap"
    if country_mismatch:
        return "country_string_mismatch"
    if country_missing:
        return "country_missing_one_side"
    if name_tok_j == 0 and name_char < 0.3 and (addr_tok_j > 0.4 or addr_char > 0.6):
        return "address_only_match_name_diverges"
    if addr_tok_j == 0 and addr_char < 0.3 and (name_tok_j > 0.4 or name_char > 0.6):
        return "name_only_match_address_diverges"
    if name_tok_j == 0 and addr_tok_j == 0 and name_char < 0.3 and addr_char < 0.3:
        return "no_lexical_overlap_either_field"
    if name_char > 0.85 and name_tok_j < 0.5:
        return "name_close_edit_distance_but_low_token_overlap(likely_typo)"
    if name_tok_j > 0 and name_tok_j < 0.5 and addr_tok_j > 0 and addr_tok_j < 0.5:
        return "partial_overlap_both_fields_below_blocking_threshold"
    return "other_partial_overlap_needs_manual_review"

counts_s2 = Counter()
counts_s3 = Counter()
examples_by_cat = {}

for ex in s2ck['missed_examples']:
    s1_id = ex['s1_id']
    for m in ex['missed']:
        cat = classify(s1_id, m, s2r, None)
        counts_s2[cat] += 1
        examples_by_cat.setdefault(('S2',cat), []).append((s1_id, m))

for ex in s3ck['missed_examples']:
    s1_id = ex['s1_id']
    for m in ex['missed']:
        cat = classify(s1_id, m, s3r, None)
        counts_s3[cat] += 1
        examples_by_cat.setdefault(('S3',cat), []).append((s1_id, m))

print("=== S2 missed-match taxonomy (n=%d) ===" % sum(counts_s2.values()))
for cat, n in counts_s2.most_common():
    print(f"  {n:4d}  ({100*n/sum(counts_s2.values()):5.1f}%)  {cat}")

print("\n=== S3 missed-match taxonomy (n=%d) ===" % sum(counts_s3.values()))
for cat, n in counts_s3.most_common():
    print(f"  {n:4d}  ({100*n/sum(counts_s3.values()):5.1f}%)  {cat}")

json.dump({'s2': dict(counts_s2), 's3': dict(counts_s3),
           'examples': {f"{k[0]}|{k[1]}": v[:3] for k, v in examples_by_cat.items()}},
          open('/home/claude/work/miss_taxonomy.json','w'), indent=2)
