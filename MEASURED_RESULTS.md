# MEASURED_RESULTS.md — this session's exact methodology and numbers

Supersedes nothing about the previous session's 94.92% baseline measurement
itself (that methodology was sound) — this session re-used the identical
measurement script and sampling parameters, changing only the frequency
caps, so the two numbers are directly comparable.

## Environment

1 CPU core, ~3.9GB RAM, ~10GB free disk, no GPU, no internet. Every bash
command is hard-killed at 300 seconds wall-clock. Confirmed experimentally
this session: background (`nohup ... &`) processes do **not** survive
between separate tool invocations — only the filesystem persists, not the
process tree. All long jobs below were run as repeated **foreground**,
manifest-resumed calls.

## Change under test: Experiment A frequency caps

```
AL2: 80  -> 120
A4W: 10  -> 15
AT2: 20  -> 30
```
(all other `DEFAULT_FREQ_CAPS` values in `blocking.py` unchanged)

This was the top-ranked, previously-untested recommendation left by the
prior session's `CHECKPOINT_MANIFEST.txt`.

## Step 1 — S2 full index build (Experiment A caps)

Built from the real, complete `train_source2.tsv` (5,034,616 rows), sqlite
backend, no row limit. Took 2 foreground calls: first call ran 300s and
reached row 3,542,286 before being killed (confirmed safe via the manifest
— commits happen every 200,000-row flush); second call resumed from there
and finished, reaching all 5,034,616 rows / 26,696,614 distinct blocking
keys.

## Step 2 — S2 recall measurement (Experiment A caps)

Same script (`measure_full_single_source.py`), same 30,000-row S1 sample,
same stride (≈73) across the full 2,206,821-row `train_source1.tsv` as the
documented baseline measurement — so this is an apples-to-apples comparison
against the 95.14% baseline number, not a re-sample.

Took 2 foreground calls (one partial, cut at row 20,838 of 30,000; one
resumed to completion).

**Result: 47,724 found / 49,982 reachable-actual = 95.4824% recall**
(baseline was 47,551 / 49,982 = 95.14%).

## Step 3 — S3 full index build (Experiment A caps)

Same procedure. Freed the ~6GB S2 sqlite file first (disk ceiling — both
full indexes don't fit at once). Built from the real, complete
`train_source3.tsv` (5,285,603 rows). Took 3 foreground calls (killed at
row 3,371,243; resumed and appeared to finish reading all rows but the
call was cut before `finalize()` ran — manifest showed `rows_indexed:
5,285,603, complete: false`; one more call correctly detected all rows
already present, skipped straight to `finalize()`, and completed in
seconds).

## Step 4 — S3 recall measurement (Experiment A caps)

Same script, same 15,000-row S1 sample, same stride (≈147) as the
documented S3 baseline. One foreground call, ran to completion in ~217s
(no interruption needed).

**Result: 25,581 found / 27,007 reachable-actual = 94.7199% recall**
(baseline was 25,527 / 27,007 = 94.52%).

## Combined result

```
combined_actual_baseline   = 49,982 + 27,007 = 76,989
combined_found_baseline    = 47,551 + 25,527 = 73,078
combined_recall_baseline   = 73,078 / 76,989 = 94.92%

combined_actual_expA       = 49,982 + 27,007 = 76,989   (same S1 sample, same GT)
combined_found_expA        = 47,724 + 25,581 = 73,305
combined_recall_expA       = 73,305 / 76,989 = 95.21%
```

**Net effect: +0.29 percentage points combined recall.**

## Candidate volume cost

| | S2 baseline | S2 Exp A | S3 baseline | S3 Exp A |
|---|---:|---:|---:|---:|
| mean candidates/S1 | 201.6 | 204.7 (+1.5%) | 206.5 | 210.8 (+2.1%) |
| median | 170 | 173 | 174 | 178 |
| p95 | 508 | 513 | 512 | 518 |
| p99 | 688 | 693 | 707 | 714 |
| max | 1,218 | 1,223 | 1,122 | 1,122 |

Candidate-volume growth is small and roughly proportional across the whole
distribution (not just the tail) — no sign of a runaway bucket appearing
under the loosened caps.

## Decision

**ACCEPT.** Promoted into `modules/blocking.py`'s `DEFAULT_FREQ_CAPS` as
the new best-known-good. Reason: recall improved at both sources
individually (S2 +0.34pp, S3 +0.20pp) for a candidate-volume cost under
2.5% — a clearly favorable tradeoff, and directionally consistent with the
previous session's hypothesis that the original caps (tuned on a 100K-row
index) were too tight at full 5M+ scale.

## What this does / doesn't tell you

**Does**: at full production index scale, loosening these three caps is a
real, measured, positive-ROI change.

**Doesn't**: whether Experiments B (AL2=150, A4W=20, AT2=40) or C (AL2=200,
A4W=30, AT2=60) from the master prompt keep paying off, plateau, or start
costing more candidates than they're worth — untested. Also doesn't tell
you anything about precision or F0.5 — no model exists yet.

## Missed-match taxonomy (built this session, not previously done)

Pulled the actual S1/S2/S3 records behind the 200 (S2) + 200 (S3) raw miss
examples saved in the previous session's `recall_s2_full_DONE.json` /
`recall_s3_full_DONE.json` checkpoints, and classified each miss with a
heuristic classifier (token-Jaccard, character-ratio, script detection —
see `analysis/classify_misses.py`). This is a rough automated taxonomy for
prioritization, not a hand-verified gold labeling.

```
S2 (n=225 missed pairs):
  90 (40.0%)  non_latin_script_no_ascii_overlap
  82 (36.4%)  other_partial_overlap_needs_manual_review
  29 (12.9%)  name_only_match_address_diverges
  12 ( 5.3%)  partial_overlap_both_fields_below_blocking_threshold
   8 ( 3.6%)  name_close_edit_distance_low_token_overlap (likely typo)
   4 ( 1.8%)  address_only_match_name_diverges

S3 (n=234 missed pairs):
  111 (47.4%)  other_partial_overlap_needs_manual_review
   50 (21.4%)  non_latin_script_no_ascii_overlap
   24 (10.3%)  partial_overlap_both_fields_below_blocking_threshold
   21 ( 9.0%)  name_only_match_address_diverges
   17 ( 7.3%)  name_close_edit_distance_low_token_overlap (likely typo)
   11 ( 4.7%)  address_only_match_name_diverges
```

Real examples pulled during manual review of the "needs manual review"
bucket surfaced two concrete, recurring, actionable failure modes not
previously documented:

1. **Spelled-out ordinal numbers** defeating address blocking — seen
   independently in two separate pairs:
   `"146 12th Street"` vs `"146 TWELFTH ST"` (S2-993815983)
   `"321 21st Street"` vs `"321 Twenty First St"` (S2-951619159)
2. **Leading-zero address numbers** — `"No. 1528"` vs `"No. 01528"`
   (S3-383257895) — exactly the pattern
   `PHASE1_ARCHITECTURE_AUDIT.md` predicted from the previous session but
   had not yet found a concrete instance of.

Also re-confirmed from real examples: legal-suffix word reordering
(`"LLC Vanguard Peak Services"` vs `"Vanguard Peak Services LLC"`), and
business-name-replaced-by-domain-string
(`"mumbaitelecommunication.com"` for `"Mumbai Telecommunication Company"`)
— this last one is a feature/model problem, not fixable by blocking.

Full raw records are in `analysis/miss_records.json`; full categorized
output with 3 example pairs per category is in `analysis/miss_taxonomy.json`.
