# Phase 1 — Architecture Audit & Production Design
### Amazon ML Challenge — Business Entity Resolution

This is an audit of the uploaded `amazon_ml_solution_FINAL` project (not a
rewrite), plus a design for what Phase 2 needs to add. Nothing here was
guessed: every number quoted is either from your own `output/*.txt` reports
or from a small parity/behavior check run against your real
`train_source*.tsv` / `train_ground_truth.tsv` files during this audit (noted
inline as "measured now"). Where I couldn't measure something at full scale
(this sandbox has 1 CPU / ~3.9 GB RAM, no GPU, no network), I say so instead
of estimating it as fact.

---

## PART 1 — AUDIT OF THE EXISTING SYSTEM

### 1. Existing architecture summary

Two parallel implementations exist in `src/`, at very different maturity levels:

| | Legacy path | **Production path (active)** |
|---|---|---|
| Files | `candidate_generator.py`, `score_candidates.py` | `blocking.py` + `indexer.py` + `candidates.py` + `pipeline.py` |
| Index | Full Python dict of **whole rows**, loaded entirely in RAM | Compact `(key_type, key_value) → [entity suffix]` postings, either in-RAM dict or SQLite |
| Frequency capping | None | Yes — `DEFAULT_FREQ_CAPS` per key type |
| Country handling | Baked into a 3-tuple key | Folded into the key string (`compose_key`) |
| Measured result (1,000 S1, 100K-record S2/S3 index) | 44.58% recall, 28,143 candidates/S1 | **98.45% recall, 257 candidates/S1** |

`candidate_generator.py` and `score_candidates.py` are dead ends: they load
every S2/S3 row into a Python dict keyed by `(country, key_type, key_value)`,
which is exactly the "load every pair into Python" pattern the brief asks to
avoid, and `score_candidates.py` calls `build_name_index`, a function that no
longer exists in that file. They're safe to delete, not modify.

**`blocking.py` → `indexer.py` → `candidates.py` → `pipeline.py` is the real
system**, and it's good work: source-independent, frequency-capped, disk-
backed option already present, already measured at 98.45% recall. Phase 2
optimizes and hardens this path — it does not replace it.

### 2. Exact current candidate-generation flow

```
pipeline.run():
  for each S1 row (streamed, one at a time, via data_loader.read_tsv):
    all_keys = make_name_keys(name) + make_unicode_name_keys(name)
             + make_address_keys(address) + make_a4_keys(address)
             + make_at2_keys(address)
    for (key_type, key_value) in all_keys:
        composed = compose_key(country, key_value)      # country-scoped key
        bucket = index.bucket(key_type, composed)        # postings lookup
        if len(bucket) > DEFAULT_FREQ_CAPS[key_type]:
            skip this key for this record                # frequency-aware pruning
        else:
            union bucket's suffixes into this row's candidate set
    → write one candidate_pairs row per surviving (S1, candidate) pair
    → immediately calculate_features() for that pair and write it too
```

This runs once against the S2 index and once against the S3 index per S1 row.
No S1×S2 or S1×S3 all-pairs comparison ever happens — every candidate is
reached through a shared blocking key. That part of the brief is already
satisfied.

### 3. Exact current blocking keys

From `blocking.py`:

**Name keys** — computed from `normalize_business_name()` (ASCII-folded):
`NP6`/`NP8` (6/8-char compact prefix), `NS6` (6-char compact suffix), `FL`/`FL3`
(first+last word), `F2` (first two words), `FML` (first+middle+last word).

**Unicode name keys** — computed separately, from the *untouched* Unicode
string (`make_unicode_name_keys`, NFKC + casefold, no ASCII stripping):
`UNP4`/`UNP6` (compact prefix), `UFL` (first+last word). This is what lets
Devanagari/Telugu/Kannada/Bengali-script names block at all — see finding #7
below for why this matters more than it looks like it should.

**Address keys** — from `normalize_address()`: `AN` (number + first non-numeric
word), `A2` (number + first two non-numeric words), `AW2`/`AL2` (first/last
two non-numeric words), `NL` (number + last word).

**A4W** — number + a small window (±3 tokens) of nearby address words, paired
up; deliberately windowed to stay O(n) per record instead of the O(n²) the
original one-off experiment used (`test_a4_overhead_fast.py`).

**AT2** — up to 6 "informative" address tokens (length ≥ 4, not in a small
stopword list), paired by proximity after sorting by length — a
position-independent complement to `AW2`/`AL2`, which only look at the
literal first/last two tokens.

### 4/5. Which keys are selective vs. explosion-prone

`output/blocking_key_analysis.txt` (1,000 S1 rows, **no frequency cap
applied**, run against a plain candidate-generator-style index — this is an
older, uncapped experiment, not the production numbers) shows the spread
directly:

| Key | Recall alone | Candidates/S1 |
|---|---|---|
| `FML` | 9.35% | 6.4 (tight) |
| `AN` | 22.29% | 42.5 |
| `A2` | 18.60% | 46.0 |
| `AL2` | 6.85% | **18,395** (explodes) |
| `NS6` | 24.05% | **7,486** (explodes) |
| `NP6` | 35.43% | 1,892 |

`AL2` (last-two-words) and `NS6` (compact suffix) are the two worst offenders
for candidate volume relative to what they add — exactly why `DEFAULT_FREQ_CAPS`
gives them the tightest caps (`AL2: 80`, vs. `NP6/NP8/NS6: 300`). `A4W` and
`AT2` get even tighter caps (`10` and `20`) because their key space is finer
(number + specific nearby words), so a bucket that's still large at that
granularity is almost always a generic/common combination, not a real match
cluster.

**Important nuance the report headline doesn't show**: this table was
produced by the *uncapped* legacy generator, at a totally different operating
point than the *capped* production system. It's still useful for relative
ranking (which keys are inherently broad vs. narrow), but don't read "44.58%
combined recall" against it as if it describes the current pipeline — the
current pipeline (with caps + the additional `A4W`/`AT2`/Unicode keys this
table didn't include) measures 98.45%, not 44.58%, at the same 1,000-row
scale. These are two different systems.

### 6. Existing normalization strategy — and the gap it hides

`normalize_text()` lowercases, NFKD-decomposes to strip Latin accents, then
does `re.sub(r"[^a-z0-9\s]", " ", text)`. That last step is fine for European
accented text (café → cafe) but it **deletes non-Latin scripts entirely**,
because NFKD decomposition doesn't turn Devanagari/Telugu/Kannada characters
into anything in `[a-z0-9]`.

**Verified during this audit**, using a real missed-match pair from your own
`output/s2_missed_matches_inspection.tsv`:

```python
>>> normalize_business_name("एसएस सिस्टम्स लिमिटेड")
''
```

`blocking.py` already knows this and works around it *for blocking* —
`make_unicode_name_keys()` runs on the raw string, not on
`normalize_business_name()`'s output. But **`features.py` never got the same
treatment**: `calculate_features()` computes every name-similarity feature
from `normalize_business_name()` alone. So for a true-match pair where both
sides are in the same Indic script, blocking correctly surfaces the pair as a
candidate (via `UNP4`/`UNP6`/`UFL`), and then every single name feature
(`name_char_similarity`, `name_levenshtein_similarity`, `name_token_jaccard`,
`name_exact_match`, …) collapses to `0` or the "missing" default, because both
sides normalized to `""`. The ML model in Phase 2 would have **no signal at
all** to tell a genuine Devanagari match from a random Devanagari non-match —
it would be relying entirely on address features for that entire subset of
rows, which (per `s2_missed_matches_inspection.tsv`) is a real, non-trivial
slice of the Indian records, not a rare edge case.

This is fixed in the `features.py` provided in Part 3 (adds a script-agnostic
fallback normalization used only when the ASCII-folded form is empty/short —
see that file's docstring for the exact behavior and a worked example).

Everything else about normalization (accent stripping, legal-suffix removal,
street abbreviations) is sound and is preserved unchanged.

### 7. Existing feature set

18 features in `features.py`: exact-match, character similarity (`difflib`),
Levenshtein similarity (`rapidfuzz`, with a pure-Python fallback), token
Jaccard, token overlap count, compact-string similarity, prefix/suffix
similarity, character bigram Jaccard — each computed for name and address —
plus `address_number_match`, `country_match`, missing-field flags, and a
hand-weighted `combined_name_address_score`. Reasonable, broad coverage for a
gradient-boosted-tree model. Two issues found:

- **The non-Latin gap above** (correctness issue, not just an omission).
- **Redundant expensive computation**: `name_char_similarity` (via
  `difflib.SequenceMatcher`, pure Python) and `name_levenshtein_similarity`
  (via `rapidfuzz`, C-accelerated) are two independently-implemented
  approximations of essentially the same "edit-similarity" quantity —
  `rapidfuzz.fuzz.ratio()` is explicitly built to be a fast, compatible
  reimplementation of `difflib`'s ratio. Doing both, per field, per pair, at
  hundreds-of-millions-of-pairs scale, means paying for a slow pure-Python
  string-matching pass that a fast C pass already covers. See the
  scalability section below for why this matters more than it looks.

### 8. Existing SQLite architecture

`SqliteIndex` in `indexer.py` mirrors `MemoryIndex`'s API (`bucket()`,
`get_attrs()`, `num_keys()`), buffers 200,000 rows before writing, and only
creates the `(key_type, key_value)` covering index at `finalize()`, after
every row has been inserted. Two issues found (both fixed in Part 3):

- **`bucket()` has no way to short-circuit an oversized bucket.** It always
  runs `SELECT suffix FROM postings WHERE key_type=? AND key_value=?` with no
  `LIMIT`, and `candidates.py` only checks `len(bucket) > cap` *after*
  `fetchall()` has already pulled the whole thing into Python. For a bucket
  that's 50,000 rows deep at full production scale but capped at 300, that's
  50,000 rows fetched and thrown away, on every S1 row whose blocking key
  happens to land there. Fixed by adding a `limit` parameter and using
  `LIMIT cap+1` in the SQL.
- **All inserts sit inside one open, uncommitted-until-`finalize()`
  transaction** (`PRAGMA synchronous=OFF`, no periodic `commit()`). If a
  Kaggle session dies while building the S2/S3 index for the full ~5M-row
  files, nothing durable has been written — the entire index build restarts
  from row 0. Combined with there being no record anywhere of "how far did
  indexing get," this is the single biggest checkpoint/resume gap in the
  current code. Fixed in Part 3 with periodic commits + a JSON manifest that
  records rows-indexed and lets a restart resume instead of rebuilding.

### 9. Existing recall experiments and results (as reported by your own code)

| Tag | S1 rows | S2/S3 index size | Combined recall | Candidates/S1 |
|---|---|---|---|---|
| `test100` | 100 | 100,000 (train_10k subset) | 98.90% | 238.6 |
| **`test1000`** (your stated baseline) | 1,000 | 100,000 | **98.45%** | **257.4** |
| `train_10k` | 10,000 | 100,000 | 97.96% | 254.2 |

**Critical thing this table shows that a single headline number hides**: all
three runs used the *same* 100,000-record S2/S3 index (`dataset/train_10k/`).
Only the number of S1 rows being *tested against* that fixed index changed.
So what you actually have measured is: "recall is ~98% when S2/S3 have
100,000 records." You do **not** yet have any measurement of recall when
S2/S3 have their true production size — **5,034,617 and 5,285,604 records
respectively, ~50x larger** than what these numbers were measured against.

This matters specifically because of how the frequency caps work: a cap like
`NP6: 300` means "skip this key if its bucket has more than 300 entries in
*whatever index was actually built*." At 50x the corpus size, a 6-character
name-prefix bucket that had, say, 40 entries in the 100K index will plausibly
have on the order of 2,000 entries in the full 5M index — comfortably over
the cap — and get silently dropped for every S1 record that would have relied
on it. The caps were tuned (by inspection, not by any adaptive computation —
see finding below) against the small index; whether they're still the right
caps at full scale is genuinely unmeasured, not just untested. The mild
recall *decline* already visible between 1,000 S1 rows (98.45%) and 10,000
S1 rows (97.96%) against the *same* index is a small early signal in that
direction, though at this small a gap it could also just be sampling noise
from which 10,000 S1 rows happened to be included.

**A related documentation/implementation mismatch worth flagging**:
`indexer.py`'s module docstring says frequency thresholds are "recomputed
fresh every time `build_index()` runs... evaluated against the CURRENT
index, never against stale 10K statistics." That's describing the *intent*,
not what the code does. `DEFAULT_FREQ_CAPS` in `blocking.py` are static,
hand-set integers, applied identically regardless of what was actually
indexed. Nothing in the current code measures the real bucket-size
distribution of the index it just built and adjusts caps accordingly. This
isn't a bug exactly (the system still runs and the caps aren't unreasonable
starting points) but it means the docstring's safety claim is aspirational,
not verified — Part 2's validation plan treats it that way.

Also on record: `blocking_combinations.txt` shows the same "more keys ⇒ more
candidates but recall plateaus fast" pattern (`F_ALL_KEYS` uses 28,143
candidates/S1 for only 44.58% recall in the *uncapped legacy* experiment) —
independent confirmation that raw key-count isn't what got you to 98.45%;
frequency capping plus the newer key types (`A4W`, `AT2`, Unicode keys) is
what did.

### 10. Biggest scalability bottlenecks, ranked

Measured/verified during this audit where marked; otherwise reasoned from the
code and the file-size numbers in your brief (train S2: 5.03M, S3: 5.29M;
test S1: 1.73M, S2: 4.89M, S3: 5.08M).

1. **Feature computation volume, not candidate generation.** At ~254–257
   candidates/S1 (your own measured average) and 1.73M test S1 rows, full-scale
   test-set feature calculation is on the order of **~440 million candidate
   pairs**. Each pair currently runs ~9 similarity computations per field
   (name + address) inline, single-threaded, in the same loop that writes
   TSV rows (`pipeline.py` lines ~180–219). Even optimistically, this is a
   multi-hour, single-core job as written, and the *output* — one TSV row per
   pair with ~24 feature columns — is easily tens of gigabytes at that row
   count. This needs both algorithmic thinning (drop the redundant
   `difflib` pass — done in Part 3) and structural change (chunked,
   parallelizable, checkpointed — designed in Part 2).
2. **Ground truth memory footprint at full scale.** **Measured during this
   audit**: loading the full 2,206,821-row `train_ground_truth.tsv` with the
   existing `labels.load_ground_truth()` uses **~2.84 GB RAM** and ~25s, because
   it builds three Python sets (`S2`, `S3`, `all`) per S1 row instead of one.
   On a memory-constrained Kaggle instance this is a real fixed cost before
   candidate generation even starts; a compact single-frozenset version
   (provided in Part 3) measured at **~1.41 GB, ~10s** for the same file — roughly
   half the memory, 2.5x faster.
3. **SQLite index build has no durable checkpoint** (see finding #8) — a
   5M-record index build that dies at row 4.9M currently restarts from row 0.
4. **Oversized-bucket fetch cost in `SqliteIndex.bucket()`** (see finding
   #8) — grows specifically as corpus size grows, i.e. gets worse exactly
   when you move from the 100K dev index to the 5M production index.
5. **The full pipeline run itself has no shard-level checkpoint.** `pipeline.py`
   streams all S1 rows in one loop, one output file handle, one connection —
   a Kaggle interruption at S1 row 1.5M of 1.73M currently loses the entire
   run's output.
6. **Frequency caps are static, not validated at production scale** (see
   finding #9) — a design risk more than a runtime bottleneck, but it's the
   one most likely to silently move the headline recall number.

Not a bottleneck, for the record: `data_loader.read_tsv()` already streams
one row at a time via a generator — this is already correct and needs no
change, and `pipeline.py`'s per-S1-row processing loop is already
constant-memory with respect to S1/S2/S3 size (it never materializes the
whole file). The brief's "don't load everything into Python" requirement is
already satisfied at the I/O layer; the remaining cost is in the *compute*
per pair, not the I/O.

### 11. What I did NOT find (worth stating explicitly)

No model-training code, no threshold-calibration code, and no F0.5 evaluation
code exist anywhere in the uploaded project (`models/` is empty). Everything
built so far is candidate generation + feature computation. That's not a gap
in the audit — it's an accurate description of project state: Phase 1's
model/threshold/output design below is genuinely new work, not something to
reconcile against existing code.

I also looked at the uploaded Kaggle notebook
(`notebookc647d5dee5__1_.ipynb`). It is an **earlier, abandoned parallel
attempt**, not a continuation of the `src/` pipeline: it re-implements
normalization from scratch (a *different*, better Unicode-category-based
approach, interestingly — see Part 2), and its SQLite schema stores only
`(entity_id, name_norm, address_norm)` with a plain composite index — no
blocking keys at all, no frequency capping, nothing resembling the
tested-and-measured `blocking.py` system. It also hard-codes one specific
Kaggle dataset slug in its path setup. **This notebook should not be
extended** — Phase 2 builds from `src/`'s production path, and borrows only
the Unicode-normalization idea from the notebook (already superseded by
what's in `blocking.py`/the new `normalize.py`, so no code needs porting from
it, just noting it's not wasted thinking).

---

## PART 2 — PRODUCTION ARCHITECTURE DESIGN

### A. Architecture diagram (text)

```
                    ┌───────────────────────────────────────────┐
                    │ STAGE 0 — Setup (Phase 2, cell 1)          │
                    │ discover_dataset_root() under /kaggle/input│
                    │ create /kaggle/working/amazon_ml/{idx,out} │
                    └───────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ STAGE 1 — Index build (checkpointed, per source)       │
        │  build_index(S2, backend=sqlite) → s2_index.sqlite     │
        │  build_index(S3, backend=sqlite) → s3_index.sqlite     │
        │  each: manifest.json {rows_indexed, complete}          │
        │  RESUME: if manifest.complete → reopen, skip rebuild   │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ STAGE 2 — Sharded candidate generation + features      │
        │  for shard in iter_shards(len(S1), shard_size):        │
        │    if shard already in manifest.completed → skip       │
        │    for s1_row in S1[shard.start:shard.end]:            │
        │      candidates = find_candidates(s1_row, {S2,S3})     │
        │      features = calculate_features(...) per candidate  │
        │      (train only) label = label_for_compact(...)       │
        │      write shard_features_NNNNN.tsv.tmp → rename       │
        │    mark_shard_complete(manifest, shard_index)           │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ STAGE 3 — Recall validation (train only, full GT)      │
        │  compare candidate pairs vs. train_ground_truth.tsv     │
        │  report: S2 recall, S3 recall, combined, candidates/S1  │
        │  by source, missed-match sample for inspection          │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ STAGE 4 — ML matching model                             │
        │  concat completed train shards → training matrix        │
        │  train/val split by S1 entity (never split a pair's     │
        │  two candidates across sets)                             │
        │  HistGradientBoostingClassifier (or LightGBM if present)│
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ STAGE 5 — F0.5 threshold calibration                    │
        │  per-S1 macro-F0.5 sweep over decision threshold(s)      │
        │  on held-out validation S1 entities                      │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ STAGE 6 — Score full test candidates, apply threshold,  │
        │           singleton/no-match handling, write final      │
        │           matching_results.tsv + validate schema         │
        └───────────────────────────────────────────────────────┘
```

### B. Recommended notebook cell structure (Phase 2)

1. Environment + `discover_dataset_root()` + working-dir setup (no hard-coded
   Windows or single-slug Kaggle paths).
2. Install/import check: `rapidfuzz` (fallback to `difflib` if absent, as
   `features.py` already does), model library (`sklearn` always available on
   Kaggle; try `lightgbm`, fall back to `HistGradientBoostingClassifier`).
3. Paste in the Part 3 modules (`normalize.py`, `blocking.py` unchanged,
   `indexer.py`, `candidates.py`, `labels.py`, `features.py`, `checkpoint.py`,
   `data_loader.py` unchanged).
4. Stage 1 — build S2/S3 indexes for **train** (sqlite backend, resumable).
5. Stage 2 — sharded candidate + feature generation for **train** S1, with
   labels.
6. Stage 3 — recall validation report against full `train_ground_truth.tsv`.
   **Do not proceed to modeling until this cell has actually run** — this is
   where Target A/B get a real number instead of an assumption.
7. Stage 4 — train the matching model on the concatenated train features.
8. Stage 5 — threshold calibration (macro F0.5), with the sweep plot/table.
9. Repeat Stage 1–2 for **test** (separate index/output directories, no
   labels).
10. Stage 6 — score test candidates, apply calibrated threshold(s), singleton
    handling, write + validate `matching_results.tsv`.
11. Final sanity checks + summary report cell.

### C. Exact changes required to the current code

- `normalize.py`: **add** `normalize_unicode_preserving()`,
  `normalize_business_name_script_agnostic()`, `is_non_latin()`. Existing
  three functions untouched.
- `features.py`: **add** the 4 non-Latin fallback columns; **replace** the
  `difflib` char-similarity path with the same `rapidfuzz`-backed `_ratio()`
  already used for Levenshtein (removes a redundant slow pass); all 20
  original columns keep identical names/positions.
- `indexer.py`: **add** `limit` parameter to `bucket()` on both backends;
  **add** manifest read/write + resume logic to `SqliteIndex`; **add**
  `discover_dataset_root()`. `MemoryIndex`'s matching logic is unchanged.
- `candidates.py`: **change** `index.bucket(key_type, composed)` calls to
  `index.bucket(key_type, composed, limit=cap + 1)`. No other logic changes.
- `labels.py`: **add** `load_ground_truth_compact()`,
  `matches_for_source()`, `label_for_compact()`. Existing functions
  untouched (still fine for train_10k-scale interactive work).
- New file `checkpoint.py`: sharding + manifest utilities for Stage 2/Stage 6
  (index-build checkpointing lives directly in `indexer.py` since it's
  source-specific; the general shard checkpoint utility is reused by the
  candidate-generation and scoring stages, which don't otherwise share code).
- `pipeline.py`: **rewritten in Phase 2**, not now — it needs to become
  shard-aware (call into `checkpoint.iter_shards`/`mark_shard_complete`
  instead of one flat loop) and gains the label-vs-no-label branch already
  present, unchanged in spirit.

### D. Files preserved unchanged

`blocking.py`, `similarity.py`, `data_loader.py`, and the *existing*
functions in `normalize.py` and `labels.py`. These are what your measured
98.45%/97.96% numbers depend on — changing any of their behavior would
invalidate those numbers rather than build on them.

### E. Files to retire (not rewritten — deleted from the active path)

`candidate_generator.py`, `score_candidates.py` (broken import, superseded
architecture). The various `patch_*.py` / `fix_*.py` / `analyze_*.py` /
`inspect_*.py` / `check_*.py` one-off scripts in `src/` were exploratory
tools that already did their job (their conclusions are folded into
`blocking.py`'s current key set and caps) — safe to leave out of the Phase 2
notebook entirely; nothing in the production path imports them.

### F. Recommended blocking rules and why

**Keep the full current set as-is** (`NP6/NP8/NS6/FL/FL3/F2/FML` +
`UNP4/UNP6/UFL` + `AN/A2/AW2/AL2/NL` + `A4W/AT2`) with **current caps as the
starting point, not the final answer** — see the validation plan below for
how to actually confirm they hold at full scale. I'm not recommending new
key types for Phase 2: the missed-match samples I reviewed
(`s2_missed_matches_inspection.tsv`, `missed_matches_inspection.tsv`) show
failure modes that are about **feature quality**, not missing blocking
signal — e.g. a business name replaced entirely by a domain string
("catelecom.com" for "Cascade Allied Telecom LLC"), heavy address-field
reordering, leading-zero differences in address numbers ("32/2" vs
"032/2"), and prefixed honorifics ("Mr", "M/s", "Smt") shifting word
positions. Adding more blocking keys to chase these would mostly add
candidate volume without recall gain, since several of the misses have *no*
name-token overlap at all (blocking can't help there — this is a feature/
address-only-matching problem, which the address keys already partially
cover, imperfectly). One genuinely low-risk addition worth measuring:
**normalize address numbers by stripping leading zeros before generating
`AN`/`A2`/`NL`/`A4W` keys** (currently `"032"` and `"32"` are different key
values) — cheap, and the missed-match sample shows this exact pattern
recurring. Validate it the same way as everything else: measure recall
with/without on the full training set before keeping it.

### G. Validation experiment plan

This is the plan for closing the gap identified in audit finding #9 — do not
skip straight to modeling before running this.

1. **Full-scale index build**: build S2 (5,034,617 rows) and S3 (5,285,604
   rows) SQLite indexes from the real `train_source2.tsv`/`train_source3.tsv`
   (uploaded, not the 10K subset). Record: build time, disk size, distinct-key
   count per key type.
2. **Full-scale recall run**: run every one of the 2,206,822 real S1 rows
   against those indexes, using `load_ground_truth_compact()` against the
   real `train_ground_truth.tsv`. Report recall by source and combined —
   this is the number that actually answers "does 98.45% hold at scale,"
   not the 100K-index numbers currently on file.
3. **Bucket-size distribution audit**: for each key type, compute the actual
   bucket-size distribution (p50/p90/p99/max) on the *full* S2/S3 index, and
   compare against the current static caps. Where a cap is far below the
   natural distribution (cutting off a large fraction of otherwise-good
   buckets) or far above it (doing nothing), record it — this is what turns
   finding #9 from "the caps might be wrong at scale" into "here's what they
   should be."
4. **Candidate-volume and runtime check** on the full run: candidates/S1,
   total output row count, total feature-file size, wall-clock time —
   this determines whether Stage 2 needs the shard size reduced or the
   feature computation moved to multiprocessing before it's trusted to run
   unattended on the full test set.
5. **Re-run the missed-match inspection** (same idea as
   `inspect_missed_matches.py`) at full scale, not just on the 1,000/10,000
   row samples, to see whether the failure-mode mix changes at scale.
6. Only after 1–5: proceed to modeling. If recall drops meaningfully from
   98.45%, address it (retune caps per #3, or add the leading-zero address
   normalization from section F) **before** locking in a train/val split for
   the model — the candidate set feeding the model needs to be the one
   that's actually going to run against the test set.

### H. Runtime/memory optimization plan

- Replace the redundant `difflib` char-similarity pass with the shared
  `rapidfuzz`-backed `_ratio()` (done in `features.py`, Part 3) — halves the
  edit-distance-style computation per pair.
- Use `load_ground_truth_compact()` for any full-scale run — measured ~50%
  memory reduction (Part 1, finding #10.2).
- Use the SQLite backend (not `memory`) for the full S2/S3 indexes — at
  ~5M records each with 15+ key types per record, an in-RAM `MemoryIndex`
  risks exceeding typical Kaggle RAM; SQLite trades some query latency for
  bounded memory, and the capped `bucket()` fetch (Part 3) keeps that
  latency bounded too.
- Shard Stage 2 (candidate + feature generation) by S1 row ranges (via
  `checkpoint.iter_shards`), sized so one shard finishes comfortably within a
  Kaggle session even in the worst case — this turns "the whole run dies at
  1.5M/1.73M rows" into "re-running the same cell finishes the remaining
  shards." Shard size is a tuning knob to set from the Stage-1 timing
  measurement in section G, not a fixed guess.
- Write feature shards as they're proven correct at small scale, then
  consider Parquet instead of TSV for the concatenated training matrix
  (`pipeline.py` already has optional Parquet support via `pandas`) — smaller
  on disk and faster to load back for training than re-parsing hundreds of
  millions of TSV rows.
- Multiprocessing is worth adding for Stage 2 (feature calculation is
  embarrassingly parallel across shards) **only if the Stage-1 timing
  measurement shows it's needed** — don't add process-pool complexity
  speculatively before there's a measured runtime number to justify it.

### I. ML training plan

- **Split by S1 entity**, not by row — every candidate pair for a given S1
  entity stays entirely in train or entirely in validation, otherwise the
  model leaks information about which S1 entities have how many true matches.
- **Model**: `sklearn.ensemble.HistGradientBoostingClassifier` as the
  reproducible default (no extra dependency, handles the feature set fine);
  try `LightGBM` if it's actually available in the Kaggle environment being
  used and compare — don't assume either without checking, per the model
  constraint in the brief.
- **Class imbalance**: `train_features_test1000.tsv` shows the real ratio —
  3,428 positive vs. 253,990 negative rows at that scale (~1.3% positive).
  Use `class_weight="balanced"` or an explicit `scale_pos_weight` rather than
  training on raw counts, and evaluate with F0.5/precision/recall, not
  accuracy.
- **Feature set**: the 20 original + 4 non-Latin-fallback columns, plus the
  blocking metadata already carried through (`blocking_num_keys`,
  `blocking_min_freq`) — these are informative (a pair found by more
  independent blocking rules, or through a rarer/more selective key, is a
  stronger candidate) and already sitting unused in the feature file.

### J. F0.5 threshold calibration plan

- Compute predicted match probability for every validation-set candidate
  pair.
- For a grid of thresholds, compute **per-S1-entity** precision/recall/F0.5,
  then macro-average across S1 entities (not micro-averaged across all
  pairs) — this is what the brief's "macro-averaged F0.5" actually means,
  and it behaves differently from a global threshold sweep whenever match
  counts per S1 vary (which `analyze_ground_truth.py`'s own distribution
  check confirms they do).
- Because precision matters more than recall for F0.5, expect and validate
  a threshold well above 0.5 — sweep broadly (e.g. 0.1 to 0.9 in fine steps)
  rather than assuming.
- Consider (and measure, don't assume) a **separate threshold per source**
  (S2 vs. S3) if their score distributions differ meaningfully — cheap to
  test once the model exists.
- Singleton S1 entities (no true match) must be scored too — a threshold
  chosen only by looking at S1 entities that *have* a true match will be
  systematically miscalibrated against them.

### K. Checkpoint/resume plan

Two layers, matching the two places long-running work happens:

1. **Index build** (`indexer.SqliteIndex`): commits after every 200K-row
   flush (not just at the end), writes a `{rows_indexed, complete}` JSON
   manifest after each commit. `build_index()` checks this manifest before
   doing any work: complete → reopen and return immediately; partial →
   skip already-indexed input rows and continue. Verified working in this
   audit (build → kill → rebuild call → correctly resumes without
   reprocessing).
2. **Sharded stages** (candidate+feature generation, scoring):
   `checkpoint.py`'s `iter_shards()`/`is_shard_complete()`/
   `mark_shard_complete()`/`write_shard_atomically()`. Each shard writes to a
   temp file and is renamed into place only on success, and is only marked
   complete in the manifest after that rename — so a kill mid-shard leaves
   either nothing or an orphaned `.tmp` file, never a shard the manifest
   incorrectly believes is done. Re-running the same cell after an
   interruption skips every already-completed shard and only redoes the
   in-flight one.

### L. Final Kaggle execution plan

1. Attach the dataset; cell 1 calls `discover_dataset_root()` against
   `/kaggle/input` rather than hard-coding a slug (the current notebook
   hard-codes `tejasweerajput/amazon-ml-challenge-2026-student-resource` —
   fragile if the dataset is reattached or versioned).
2. Run Stages 1–3 (index + candidates + recall validation) on **train**
   first, in full, before touching the model — per section G, this is where
   Target A/B get answered with real numbers.
3. Train + calibrate (Stages 4–5) on the train candidates.
4. Run Stages 1–2 again for **test**, into separate working directories
   (`idx_test/`, `out_test/`) so a test-set rebuild never touches the
   train-set index or manifest.
5. Score, threshold, singleton-handle, write, and validate
   `matching_results.tsv` (Stage 6).
6. Every stage is independently re-runnable from its manifest — the
   execution plan is "run the notebook top to bottom," but the safety net is
   "if a cell's session dies, re-run that same cell" rather than
   "restart from cell 1."

---

## PART 3 — Core optimized modules for Phase 2

Provided as files (not pasted inline) so they're ready to drop into the
Phase 2 notebook or `src/` directly:

- `normalize.py` — original 3 functions unchanged; adds
  `normalize_unicode_preserving`, `normalize_business_name_script_agnostic`,
  `is_non_latin`.
- `features.py` — same 20 original columns, same names/positions; adds 4
  non-Latin fallback columns; replaces the redundant `difflib` pass with the
  shared `rapidfuzz`-backed ratio (falls back to `difflib` automatically if
  `rapidfuzz` isn't installed, exactly as the original did).
- `indexer.py` — `MemoryIndex` unchanged in behavior; `SqliteIndex` gains
  capped `bucket(..., limit=...)`, periodic commits, and manifest-based
  resume; adds `discover_dataset_root()`.
- `candidates.py` — one-line change: passes `limit=cap + 1` into
  `index.bucket()`.
- `labels.py` — original functions unchanged; adds
  `load_ground_truth_compact()` + helpers (measured ~50% memory reduction at
  full 2.2M-row scale).
- `checkpoint.py` — new: generic shard/manifest utilities for Stage 2/6.

All six were smoke-tested against your real uploaded files during this audit
(not just unit-tested in isolation): a 20,000-record `MemoryIndex` and
`SqliteIndex` built from your real `train_source2.tsv` returned **identical**
candidate results for a sample S1 row; `SqliteIndex` resume was verified to
correctly detect a completed build and skip rebuilding; the non-Latin
normalization fix was verified against a real missed-match pair from your own
`output/` reports; the compact ground-truth loader was verified against the
full, real `train_ground_truth.tsv` (2,206,821 rows) and measured at roughly
half the memory of the original loader.

`blocking.py`, `similarity.py`, and `data_loader.py` are not re-provided —
copy them from `src/` unchanged (per section D).
