"""High-signal pair features for the entity-resolution classifier."""
import re
from math import log1p
from rapidfuzz.fuzz import ratio, WRatio, token_set_ratio, token_sort_ratio
from advanced_normalize import normalize_address_numbers, token_sorted, compact_alnum, transliterate_ascii
from normalize import normalize_business_name, normalize_address

FEATURE_COLUMNS = [
    "name_ratio","name_wratio","name_token_set_ratio","name_token_sort_ratio",
    "name_exact_norm","name_sorted_exact","name_compact_ratio","name_len_ratio",
    "name_token_jaccard","name_token_overlap","name_prefix_ratio","name_suffix_ratio",
    "name_unicode_ratio","name_translit_ratio","name_translit_token_ratio",
    "address_ratio","address_token_set_ratio","address_token_sort_ratio",
    "address_exact_norm","address_numbers_norm_ratio","address_numbers_exact",
    "address_numeric_overlap","address_token_jaccard","address_token_overlap",
    "address_prefix_ratio","address_suffix_ratio","address_len_ratio",
    "country_exact","country_missing_either","country_both_missing",
    "name_missing_either","address_missing_either","both_fields_missing",
    "name_address_weighted","name_address_min","name_address_max",
    "name_address_product","name_address_harmonic","name_minus_address_abs",
    "blocking_num_keys","blocking_min_freq_log","blocking_name_key","blocking_address_key",
    "candidate_rank","candidate_count","rank_reciprocal","rank_log",
]

def _safe_ratio(a,b):
    return ratio(a,b)/100.0 if a and b else 0.0

def _jacc(a,b):
    A=set(a.split()) if a else set(); B=set(b.split()) if b else set()
    return len(A&B)/len(A|B) if A|B else 0.0

def _overlap(a,b):
    A=set(a.split()) if a else set(); B=set(b.split()) if b else set()
    return len(A&B)

def _lenratio(a,b):
    if not a or not b: return 0.0
    return min(len(a),len(b))/max(len(a),len(b))

def _numbers(s): return re.findall(r"\d+", normalize_address_numbers(s))

def pair_features(s1,s2,meta=None):
    meta=meta or {}
    n1=normalize_business_name(s1.get("business_name","")); n2=normalize_business_name(s2.get("business_name",""))
    u1=str(s1.get("business_name","")).casefold().strip(); u2=str(s2.get("business_name","")).casefold().strip()
    t1=transliterate_ascii(u1); t2=transliterate_ascii(u2)
    a1=normalize_address(s1.get("business_address","")); a2=normalize_address(s2.get("business_address",""))
    an1=normalize_address_numbers(a1); an2=normalize_address_numbers(a2)
    nt1=token_sorted(n1); nt2=token_sorted(n2)
    nums1=_numbers(a1); nums2=_numbers(a2)
    nr=_safe_ratio(n1,n2); ar=_safe_ratio(a1,a2)
    vals={
      "name_ratio":nr,"name_wratio":WRatio(n1,n2)/100 if n1 and n2 else 0.0,
      "name_token_set_ratio":token_set_ratio(n1,n2)/100 if n1 and n2 else 0.0,
      "name_token_sort_ratio":token_sort_ratio(n1,n2)/100 if n1 and n2 else 0.0,
      "name_exact_norm":int(bool(n1 and n1==n2)),"name_sorted_exact":int(bool(nt1 and nt1==nt2)),
      "name_compact_ratio":_safe_ratio(compact_alnum(n1),compact_alnum(n2)),"name_len_ratio":_lenratio(n1,n2),
      "name_token_jaccard":_jacc(n1,n2),"name_token_overlap":_overlap(n1,n2),
      "name_prefix_ratio":_safe_ratio(n1[:6],n2[:6]),"name_suffix_ratio":_safe_ratio(n1[-6:],n2[-6:]),
      "name_unicode_ratio":_safe_ratio(u1,u2),"name_translit_ratio":_safe_ratio(t1,t2),
      "name_translit_token_ratio":token_set_ratio(t1,t2)/100 if t1 and t2 else 0.0,
      "address_ratio":ar,"address_token_set_ratio":token_set_ratio(a1,a2)/100 if a1 and a2 else 0.0,
      "address_token_sort_ratio":token_sort_ratio(a1,a2)/100 if a1 and a2 else 0.0,
      "address_exact_norm":int(bool(a1 and a1==a2)),"address_numbers_norm_ratio":_safe_ratio(an1,an2),
      "address_numbers_exact":int(bool(nums1 and nums2 and nums1==nums2)),
      "address_numeric_overlap":len(set(nums1)&set(nums2)),"address_token_jaccard":_jacc(a1,a2),
      "address_token_overlap":_overlap(a1,a2),"address_prefix_ratio":_safe_ratio(a1[:10],a2[:10]),
      "address_suffix_ratio":_safe_ratio(a1[-10:],a2[-10:]),"address_len_ratio":_lenratio(a1,a2),
    }
    c1=str(s1.get("country","")).strip().casefold(); c2=str(s2.get("country","")).strip().casefold()
    vals.update({"country_exact":int(bool(c1 and c2 and c1==c2)),"country_missing_either":int(not c1 or not c2),"country_both_missing":int(not c1 and not c2)})
    nm=int(not n1 or not n2); am=int(not a1 or not a2)
    vals.update({"name_missing_either":nm,"address_missing_either":am,"both_fields_missing":int(nm and am)})
    vals.update({"name_address_weighted":.65*nr+.35*ar,"name_address_min":min(nr,ar),"name_address_max":max(nr,ar),
                 "name_address_product":nr*ar,"name_address_harmonic":2*nr*ar/(nr+ar) if nr+ar else 0.0,
                 "name_minus_address_abs":abs(nr-ar)})
    vals.update({"blocking_num_keys":float(meta.get("blocking_num_keys",0)),"blocking_min_freq_log":log1p(float(meta.get("blocking_min_freq",0))),
                 "blocking_name_key":float(meta.get("blocking_name_key",0)),"blocking_address_key":float(meta.get("blocking_address_key",0)),
                 "candidate_rank":float(meta.get("candidate_rank",0)),"candidate_count":float(meta.get("candidate_count",0)),
                 "rank_reciprocal":1.0/(1.0+float(meta.get("candidate_rank",0))),"rank_log":log1p(float(meta.get("candidate_rank",0)))})
    return vals
