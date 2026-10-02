# STATUS — read this first

## The one number you asked for: current measured candidate recall

**95.21% combined candidate recall**, full-scale (all 5,034,616 S2 rows /
5,285,603 S3 rows indexed), with the caps below. This supersedes the
94.92% baseline from the previous session.

| | Baseline caps (AL2=80, A4W=10, AT2=20) | **Experiment A caps (AL2=120, A4W=15, AT2=30) — CURRENT BEST** |
|---|---:|---:|
| S2 recall | 95.14% | **95.48%** |
| S3 recall | 94.52% | **94.72%** |
| Combined (weighted) | 94.92% | **95.21%** |
| Mean candidates/S1 (S2) | 201.6 | 204.7 (+1.5%) |
| Mean candidates/S1 (S3) | 206.5 | 210.8 (+2.1%) |
| p99 candidates/S1 (S2 / S3) | 688 / 707 | 693 / 714 |

This is still **candidate recall only** (did blocking retrieve the true
match at all) — not precision, not F0.5, not a trained model's accuracy.
**No ML matcher has been trained yet.** Never report the number above as
a leaderboard score.

## What changed this session

1. **Missed-match taxonomy** — pulled the actual S1/S2/S3 records behind
   the 200+200 saved miss examples from the previous session's checkpoints
   and classified them (see `analysis/miss_taxonomy.json` and
   `analysis/classify_misses.py`). This was evidence-based, not guessed.
   Headline findings:
   - S2: 40% non-Latin script with zero ASCII overlap, 13% name-only-match
     (address diverges or missing), rest partial-overlap/typo cases.
   - S3: 47% partial-overlap "needs review", 21% non-Latin script, 10%
     both fields below the blocking threshold.
   - Two concrete, recurring, actionable patterns found in real examples:
     **spelled-out ordinal numbers** defeating address blocking ("12th
     Street" vs "TWELFTH ST", "21st" vs "TWENTY FIRST" — seen independently
     twice) and **leading-zero address numbers** ("1528" vs "01528",
     confirming what `PHASE1_ARCHITECTURE_AUDIT.md` predicted). Also
     confirmed: legal-suffix word reordering ("LLC X Y" vs "X Y LLC") and
     business-name-replaced-by-domain-string ("mumbaitelecommunication.com").
2. **Experiment A: cap loosening** (the top recommended next step from the
   previous session's `CHECKPOINT_MANIFEST.txt`) — tested, measured, and
   **ACCEPTED**. Full-scale S2 and S3 indexes were rebuilt from the real
   uploaded training files with `AL2: 80→120`, `A4W: 10→15`, `AT2: 20→30`,
   and recall was re-measured with the exact same sampling methodology as
   the documented baseline (same 30,000/15,000-row strides), so the
   comparison is apples-to-apples. Net effect: **+0.29pp combined recall
   for ~1.5-2% more candidates per S1** — a clearly favorable tradeoff.
   These new caps are now in `modules/blocking.py`'s `DEFAULT_FREQ_CAPS`
   and are the current best-known-good configuration.

## What's actually done (cumulative, across sessions)

1. Full data audit (row counts, referential integrity, match-count
   distribution, country distribution) — `DATA_STATUS.md`.
2. Phase 1 code audit + 6 optimized modules, smoke-tested against real
   data — `PHASE1_ARCHITECTURE_AUDIT.md`.
3. Full-scale (not truncated) candidate recall measured for S2 and S3
   independently, at both the original caps and Experiment A's caps.
4. A missed-match taxonomy with real, categorized examples (new this
   session).
5. Cells 1–3 of the 27-cell production notebook (`notebook_progress/`) —
   still genuinely unfinished, unchanged from the previous session.

## What's NOT done yet (unchanged from before, still true)

- Cells 4–27 of the notebook.
- Character n-gram / TF-IDF retrieval layer (recommended next step,
  never implemented).
- The address leading-zero and ordinal-number normalization fixes the
  taxonomy surfaced this session (implicated by real evidence, not yet
  coded or measured).
- Any feature engineering beyond the existing 18-column `features.py`.
- **No model has been trained.** No accuracy/precision/F0.5 number
  exists anywhere in this project. Only candidate recall.
- Test-file audit beyond row counts + country distribution.
- `matching_results.tsv` / `candidate_pairs.tsv` do not exist.

## Sandbox constraints this session ran under (for context)

Same class of sandbox as the previous session: 1 CPU core, ~3.9GB RAM,
~10GB free disk, no GPU, **no internet access this time** (previous
session apparently had none either, based on its own notes). Additionally
observed this session: **every bash command is hard-killed at 300
seconds**, and **background/nohup processes do not survive between tool
calls** (the process tree is not preserved across invocations, only the
filesystem is) — so anything long-running MUST be done as a foreground,
resumable, checkpoint-based call, repeated across multiple tool
invocations. This is not a new constraint, just newly confirmed
experimentally; `indexer.py`'s manifest-based resume already handles it
correctly (verified: a build killed mid-run at row 3.54M correctly
resumed and finished from there, twice, once for S2 and once for S3).

See `HANDOFF_TO_NEXT_CLAUDE.md` for exactly what to do next.
