# Methodology — Amazon ML Challenge 2026

## Objective
Find all S2/S3 records matching every S1 record. The optimization target is the official macro-averaged F0.5, which is precision-heavy and includes singleton S1 entities.

## Candidate generation
The audited deterministic blocking system is preserved. Experiment A frequency caps are retained: AL2=120, A4W=15, AT2=30. SQLite indexes make full-source lookup resumable and disk-backed. Candidate lists are the exact lists subsequently scored by the model and are written to `candidate_pairs.tsv`.

## Feature layer
The classifier uses normalized exact/similarity signals for business names and addresses, token overlap, character similarity, numeric-address agreement, country agreement/missingness, transliteration similarity, non-Latin-safe Unicode similarity, blocking evidence, candidate rank and candidate-set size. The legacy normalization behavior used for measured blocking recall is not changed.

## Hard negatives
For each sampled S1 entity, retrieved non-matches are ranked by a cheap name/address similarity pre-score. The hardest negatives are retained together with every retrieved positive. This concentrates training on plausible false merges, which is appropriate for F0.5.

## Validation
Rows are split by S1 entity using GroupKFold. This prevents candidate pairs for one reference entity from appearing in both train and validation folds. Threshold calibration is performed only on out-of-fold predictions.

## Decision calibration
The primary decision variable is the classifier probability threshold. A small top-candidate margin is optionally selected on OOF data. Multiple matches remain possible; there is no global one-to-one constraint because the challenge explicitly permits one-to-many matches.

## Test isolation
Test files are used only after the model, threshold and decision rules are frozen. No test labels or external identity data are used.

## Reproducibility
The pipeline stores the feature list, model, threshold, margin and training configuration in the model sidecar JSON. SQLite index builds are resumable.
