You are continuing an Amazon ML Challenge 2026 Business Entity Resolution
project. DO NOT START FROM SCRATCH.

First read, in this order:
1. STATUS.md (this package's root)
2. MEASURED_RESULTS.md            <- this session's exact Experiment A methodology + numbers
3. BEST_RESULT.md                 <- the single current-best config, short form
4. BEST_CONFIG.md                 <- exact parameter values to reproduce it
5. DATA_STATUS.md
6. EXPERIMENT_LOG.md
7. This file's "Next experiments" section below

## Current objective

Same as before: maximize candidate recall without unbounded candidate-volume
blowup, THEN move to feature engineering / hard negatives / model training —
none of that has started. Current best: **95.21% combined candidate recall**
(see BEST_RESULT.md).

## What is COMPLETE this session

- **Experiment A (cap loosening) — ACCEPTED and promoted into
  `modules/blocking.py`**. Full detail in MEASURED_RESULTS.md. This is now
  the best-known-good configuration; do not revert it without a measured
  reason.
- **Missed-match taxonomy** built from real records (not the raw
  unclassified IDs the previous session left). See `analysis/miss_taxonomy.json`
  for counts and `analysis/classify_misses.py` for the exact classification
  logic (heuristic, based on token/char overlap + script detection — treat
  the categories as directionally correct, not a hand-labeled gold set).

## What is NOT done (in priority order)

1. **Address leading-zero normalization.** The taxonomy caught a real
   example (`"01528"` vs `"1528"`) exactly matching what
   `PHASE1_ARCHITECTURE_AUDIT.md` predicted from the previous session, and
   it's cheap: strip leading zeros from numeric tokens before generating
   `AN`/`A2`/`NL`/`A4W` keys in `blocking.py`. Measure recall with/without
   before keeping it, same methodology as Experiment A (rebuild full S2 index,
   measure with the 30K-row sample; do NOT skip the S3 side just because
   S2 looks good — Experiment A's S2/S3 gains were different sizes,
   +0.34pp vs +0.20pp, so don't assume one number for both).
2. **Ordinal-number word↔digit normalization** ("12th" vs "twelfth",
   "21st" vs "twenty first") — seen twice independently in the real miss
   examples pulled this session (`analysis/miss_records.json` has the raw
   records if you want to re-inspect them: search for `S2-993815983` and
   `S2-951619159`). This is a genuine, evidenced pattern, not a hypothesis
   from a public repo. Build a small ordinal-word→digit map (one/first,
   two/second, ... — needs only a couple dozen entries for realistic
   address text) and apply it during address normalization before key
   generation. Measure the same way.
3. **Character n-gram / TF-IDF retrieval as an additional candidate layer**
   (never implemented) — union with the existing deterministic keys.
   Targets the ~36-47% "other_partial_overlap_needs_manual_review" +
   "no_lexical_overlap" buckets in the taxonomy that no cap tuning will fix,
   because several of those misses share almost no token overlap at all
   (see the real examples in `analysis/miss_taxonomy.json`'s `examples` field
   — e.g. `"@Fóot"` for `"Foot & Ankle Care Associates LP"`, or
   `"mumbaitelecommunication.com"` for `"Mumbai Telecommunication Company"` —
   blocking cannot recover these from token/prefix keys no matter how the
   caps are tuned).
4. **Experiment B / C from the master prompt** (AL2=150/200, A4W=20/30,
   AT2=40/60) — untested. Experiment A already showed diminishing-but-real
   returns from loosening; it is not yet established whether B/C keep
   paying off or start adding candidate volume for no further recall gain.
   Test B before assuming C is worth the cost.
5. Everything past blocking, unchanged from before: 40–70 feature
   engineering, hard-negative mining, model training, GroupKFold validation,
   Macro F0.5 threshold optimization, singleton calibration, multi-match
   policy, test inference, final outputs, official validator. NONE started.

## Reproducing the full-scale Experiment A measurement yourself

The 6+ GB SQLite index files are NOT in this zip (too large; also this
sandbox class cannot hold S2-full and S3-full simultaneously, ~10GB disk
total). Rebuild from the raw TSVs — this is genuinely resumable, verified
twice this session (a kill at row 3.54M/5.03M and again effectively at
5.28M/5.28M both resumed correctly):

```bash
cd modules/
# ONE full index at a time. Re-run the SAME command if it gets cut off
# (every environment this project has run in kills long commands around
# 300s) -- build_index()'s manifest-based resume picks up where it left off.
python3 -c "
from indexer import build_index
build_index('/path/to/train_source2.tsv', 'S2', limit=None, backend='sqlite', db_path='./s2.sqlite')
"
# repeat the exact same command until you see 'Finished S2: 5,034,616 records'

cd ../measurements/
python3 -c "
import sys; sys.path.insert(0, '../modules')
sys.argv = ['x', 'S2', '../modules/s2.sqlite', './recall_s2.json', '30000']
exec(open('measure_full_single_source.py').read())
"
# also resumable the same way; re-run until it prints 'DONE'

# then free disk and repeat both steps for S3 (source_prefix='S3', sample=15000)
rm ../modules/s2.sqlite*
```

**Important operational note discovered this session, not in prior
handoffs:** if you're in a similarly constrained sandbox, do NOT background
long jobs with `nohup ... &` across separate tool calls — the process is
killed when the tool call that started it returns; only the filesystem
persists between calls, not the process tree. Run the build as a plain
foreground command, let it get killed at the 300s wall-clock limit if it
must, then just re-invoke the identical command — the manifest resume
makes this safe and it's actually the faster path (foreground indexing here
ran at roughly 11,800 rows/sec; there is no benefit to backgrounding it).

## What to upload to continue

Required every time (this sandbox does not persist them across sessions):
```
train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
```
Also re-upload this zip so the next Claude has this session's accepted
Experiment A caps, taxonomy, and full experiment log instead of starting
over from the 94.92%/pre-taxonomy state.
Needed once you reach feature/model work against the real test set (not
needed for further blocking experiments):
```
test_source1.tsv, test_source2.tsv, test_source3.tsv
```

## What NOT to repeat

- Don't re-measure recall against a truncated (partial) index and present
  it as "the" number — full index only, per the last two sessions' explicit
  corrections.
- Don't background long-running builds with `nohup`/`&` across tool calls
  in a sandbox like this one — verified this session that it silently loses
  all progress. Foreground + resume is the only pattern that actually works
  here.
- Don't try to hold S2-full and S3-full SQLite indexes on disk
  simultaneously (~10GB disk ceiling, each index is ~6GB) — sequential,
  one at a time, freeing the previous one's file before starting the next.
- Don't fabricate an F0.5 number, and don't claim the taxonomy percentages
  in `analysis/miss_taxonomy.json` are a hand-verified gold standard — they
  come from an automated heuristic classifier (`classify_misses.py`), useful
  for prioritization, not a precise ground truth breakdown.
