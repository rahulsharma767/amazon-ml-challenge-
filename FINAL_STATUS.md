# FINAL STATUS — implementation-complete checkpoint

## What was added

The original checkpoint had a mature blocking/candidate-recall stage but no trained matcher or submission-generation stage. This package adds the previously missing implementation:

- advanced pairwise features (`modules/advanced_features.py`)
- address leading-zero and ordinal normalization feature view (`modules/advanced_normalize.py`)
- hard-negative pair construction
- LightGBM pair classifier
- S1-grouped 5-fold OOF validation
- exact challenge-style Macro F0.5 scoring
- OOF threshold calibration
- optional top-candidate margin calibration
- frozen test inference
- exact `matching_results.tsv` generation
- exact `candidate_pairs.tsv` generation
- structural output validator
- reproducible run instructions and methodology

## Existing best-known measured baseline retained

Experiment A deterministic blocking:

- S2 candidate recall: 95.48%
- S3 candidate recall: 94.72%
- combined candidate recall: 95.21%
- caps: AL2=120, A4W=15, AT2=30

These are candidate-recall measurements, not final F0.5.

## Important honesty statement

The real competition TSV datasets were not included in the uploaded checkpoint ZIPs available for this build session. Therefore the final competition model cannot be trained here, and no real competition `matching_results.tsv` can be truthfully generated in this session.

The new code was syntax-checked and exercised end-to-end on a synthetic smoke dataset. The smoke run produced both required output files and the structural validator returned PASS. This validates the implementation path, not competition performance.

Run `train_phase2.py` on the real training TSVs, then `infer_test.py` on the real test TSVs, then `validate_outputs.py` before submission.
