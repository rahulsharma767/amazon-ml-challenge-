# RECOVERY_PLAN.md

If this project needs to resume in a fresh sandbox:

1. Re-upload train_source1/2/3.tsv, train_ground_truth.tsv (and test_*.tsv
   once you reach test-scoring stages).
2. Re-upload this zip.
3. Read STATUS.md, then HANDOFF_TO_NEXT_CLAUDE.md.
4. The full 5M+-row SQLite indexes are NOT preserved in this zip (too
   large, and no single sandbox in this project's history has had disk
   for both S2-full and S3-full simultaneously) -- rebuild per the exact
   commands in HANDOFF_TO_NEXT_CLAUDE.md's reproduction section. Building
   is resumable and safe to interrupt; the manifest tracks real progress.
5. Do not re-derive DEFAULT_FREQ_CAPS from scratch -- BEST_CONFIG.md has
   the exact accepted values, already applied in modules/blocking.py.
6. Continue from NEXT_EXPERIMENTS.md, in order, unless new evidence changes
   the priority.
