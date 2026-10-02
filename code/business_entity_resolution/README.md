# Amazon ML Challenge 2026 — Business Entity Resolution

This package extends the audited Phase-1 blocking system into a complete, reproducible Phase-2 matching pipeline. The legacy blocking keys and measured Experiment-A caps are preserved. The new stages add advanced pair features, hard-negative training, group-aware validation, OOF Macro F0.5 threshold calibration, singleton/margin protection, frozen test inference, and strict output validation.

## Architecture

1. Multi-view normalization (legacy views preserved; additional address-number/ordinal view and transliteration feature view).
2. Deterministic blocking using the audited SQLite index and Experiment-A caps.
3. Candidate union across S2 and S3.
4. Pairwise feature generation.
5. Positive + hard-negative sampling.
6. LightGBM classifier (MIT-licensed library; no external identity data).
7. GroupKFold by S1 entity.
8. OOF threshold and top-candidate margin calibration against exact challenge-style Macro F0.5.
9. Frozen test inference.
10. `matching_results.tsv` and `candidate_pairs.tsv` generation.
11. Structural validator.

## Compliance

No external business identity lookup, geocoding, government registry, commercial ER API, or external business-data augmentation is used. Test data is never used for supervised optimization.

## Important status rule

The supplied checkpoint had measured candidate recall but no trained model. Therefore a numerical final F0.5 cannot honestly be claimed until `train_phase2.py` is executed on the real training files. The code in this package is the complete implementation of the previously missing ML/output stages; measured results must be recorded only after actual execution.
