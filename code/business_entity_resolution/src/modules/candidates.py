"""
candidates.py
============================================================
CHANGE from the original: index.bucket(key_type, composed) is now
called as index.bucket(key_type, composed, limit=cap + 1).

Why cap + 1 and not cap: we still need to distinguish "bucket size
== cap exactly" (keep it) from "bucket size > cap" (drop it), so we
ask for one more row than the cap allows. For MemoryIndex this is
free (already in RAM). For SqliteIndex this turns a potentially
huge, wasted `SELECT ... ` + fetchall() on a hot key into a bounded
`SELECT ... LIMIT cap+1` -- seeSee indexer.py's module docstring
for the full reasoning and why this matters specifically at full
(5M+ row) production scale, not at the 100K-row dev scale the
original numbers were measured on.

Everything else in this file (the candidate dict shape, key_types/
min_freq bookkeeping, generate_candidate_pairs_for_row) is
UNCHANGED.
"""

from blocking import (
    make_name_keys,
    make_unicode_name_keys,
    make_address_keys,
    make_a4_keys,
    make_at2_keys,
    DEFAULT_FREQ_CAPS,
)
from indexer import compose_key


def find_candidates(s1_row, index, freq_caps=None):

    freq_caps = freq_caps or DEFAULT_FREQ_CAPS

    country = s1_row.get("country", "").strip()
    name = s1_row.get("business_name", "")
    address = s1_row.get("business_address", "")

    all_keys = (
        make_name_keys(name)
        + make_unicode_name_keys(name)
        + make_address_keys(address)
        + make_a4_keys(address)
        + make_at2_keys(address)
    )

    candidates = {}

    for key_type, key_value in all_keys:

        cap = freq_caps.get(key_type, DEFAULT_FREQ_CAPS.get(key_type, 500))

        composed = compose_key(country, key_value)

        # Ask for at most cap + 1 rows. If we get back cap + 1, the
        # true bucket is > cap somewhere at or beyond that row, so
        # it's still correctly treated as "too broad" below -- we
        # just never had to materialize the rest of it.
        bucket = index.bucket(key_type, composed, limit=cap + 1)
        bucket_size = len(bucket)

        if bucket_size == 0:
            continue

        if bucket_size > cap:
            continue

        for suffix in bucket:

            entry = candidates.get(suffix)

            if entry is None:
                candidates[suffix] = {
                    "key_types": {key_type},
                    "min_freq": bucket_size,
                }
            else:
                entry["key_types"].add(key_type)
                entry["min_freq"] = min(entry["min_freq"], bucket_size)

    return candidates


def generate_candidate_pairs_for_row(s1_row, indexes_by_source, freq_caps=None):

    s1_id = s1_row.get("entity_id", "")
    rows = []

    for source, index in indexes_by_source.items():

        candidates = find_candidates(s1_row, index, freq_caps=freq_caps)

        for suffix, info in candidates.items():

            candidate_id = f"{source}-{suffix}"

            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "source": source,
                    "blocking_key_types": "|".join(sorted(info["key_types"])),
                    "blocking_num_keys": len(info["key_types"]),
                    "blocking_min_freq": info["min_freq"],
                }
            )

    return rows
