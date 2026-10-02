# EXPERIMENT_LOG.md

| experiment_id | date | baseline | change | dataset | sample | candidate recall | candidates/S1 (mean) | precision | recall | Macro F0.5 | runtime | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp_00_baseline | prior session | -- | (documented baseline, not re-run this session) | full S2 (5,034,616) + full S3 (5,285,603) | 30,000 S1 (S2) / 15,000 S1 (S3) | S2 95.14%, S3 94.52%, combined 94.92% | S2 201.6, S3 206.5 | N/A | N/A | N/A | prior session | KEPT (reference baseline) |
| exp_A_cap_loosen | this session | exp_00 | AL2 80->120, A4W 10->15, AT2 20->30 | full S2 + full S3 (rebuilt this session, same row counts) | 30,000 S1 (S2) / 15,000 S1 (S3), identical stride to exp_00 | S2 95.48%, S3 94.72%, combined 95.21% | S2 204.7 (+1.5%), S3 210.8 (+2.1%) | N/A | N/A | N/A | ~10 min wall-clock across several 300s-limited resumed calls | **ACCEPTED, promoted to best_known_good** |

No other experiments run this session. Precision/recall/F0.5 columns are
N/A everywhere because no model has been trained in this project yet --
every row so far is candidate-generation (blocking) recall only.
