"""
labels.py
============================================================
PRESERVED: load_ground_truth() / label_for() are UNCHANGED and
still correct -- keep using them for train_10k-scale work where
their extra convenience (pre-split "S2"/"S3"/"all" sets) is cheap.

ADDED: load_ground_truth_compact(), a memory-lean alternative for
the full 2.2M-row train_ground_truth.tsv.

MEASURED (this sandbox, 3.9 GB total RAM, Python 3.12):
  load_ground_truth() on the full file -> 2,206,821 entries,
  peak RSS ~2.84 GB, ~25s.

That is the ENTIRE per-S1-row ground truth structure sitting in
RAM before candidate generation, feature calculation, indexes, or
anything else has been loaded -- on a Kaggle instance with, say,
13-16 GB total RAM, this is a meaningful fixed cost, and on a
smaller instance it can be the difference between fitting in
memory and an OOM kill mid-run (which is exactly what truncated a
larger validation run attempted during this audit).

The cost comes from storing THREE Python sets per S1 row
("S2", "S3", "all") instead of one -- each empty/small set has
real fixed overhead (hash table allocation) independent of how
many ids it holds, so building three per row roughly triples the
per-row container overhead versus storing the raw id set once and
filtering by prefix on demand (which is O(len(entry)), and entries
have on average only a handful of matches).

load_ground_truth_compact() stores exactly one frozenset per row.
matches_for_source()/label_for_compact() below reconstruct the
"S2 only" / "S3 only" view on demand, at effectively no extra cost
since per-row match counts are small (median well under 10, per
analyze_ground_truth.py's own distribution check).
"""

import csv


def load_ground_truth(path):
    """
    UNCHANGED. Returns dict: source1_entity_id -> {
        "S2": set of matched S2 ids,
        "S3": set of matched S3 ids,
        "all": set of every matched id (any source),
    }
    """

    ground_truth = {}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            s1_id = row["source1_entity_id"]
            raw = (row.get("matched_entity_ids") or "").strip()

            if raw:
                all_ids = {x.strip() for x in raw.split(",") if x.strip()}
            else:
                all_ids = set()

            ground_truth[s1_id] = {
                "S2": {x for x in all_ids if x.startswith("S2-")},
                "S3": {x for x in all_ids if x.startswith("S3-")},
                "all": all_ids,
            }

    return ground_truth


def label_for(s1_id, candidate_id, ground_truth):
    entry = ground_truth.get(s1_id)
    if not entry:
        return 0
    return int(candidate_id in entry["all"])


# ============================================================
# NEW: memory-lean loader for full-scale (2.2M+ row) ground truth
# ============================================================

_EMPTY_FROZENSET = frozenset()


def load_ground_truth_compact(path):
    """
    Returns dict: source1_entity_id -> frozenset of matched ids
    (mixed S2-*/S3-*). One container per row instead of three.

    frozenset (vs set) is used because these are never mutated
    after loading, and CPython's frozenset has slightly less
    per-object overhead than a mutable set for read-only use.
    """

    ground_truth = {}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            s1_id = row["source1_entity_id"]
            raw = (row.get("matched_entity_ids") or "").strip()

            if raw:
                ground_truth[s1_id] = frozenset(
                    x.strip() for x in raw.split(",") if x.strip()
                )
            else:
                ground_truth[s1_id] = _EMPTY_FROZENSET

    return ground_truth


def matches_for_source(entry, source_prefix):
    """
    entry: a frozenset from load_ground_truth_compact(). Filters
    by prefix on demand -- cheap, since entry is typically only a
    handful of ids (see analyze_ground_truth.py's match_count
    distribution: median match count is small).
    """
    if not entry:
        return _EMPTY_FROZENSET
    return frozenset(x for x in entry if x.startswith(source_prefix))


def label_for_compact(s1_id, candidate_id, ground_truth):
    entry = ground_truth.get(s1_id)
    if not entry:
        return 0
    return int(candidate_id in entry)
