# Phase-2 implementation log

## P2-001 — Advanced matcher implementation
Status: IMPLEMENTED / SMOKE-TESTED / NOT YET MEASURED ON COMPETITION DATA.

Added pairwise name/address/country/transliteration/numeric-address/blocking/rank features, hard-negative sampling, LightGBM classifier, GroupKFold OOF calibration, frozen inference and output validation.

## P2-002 — Exact ID preservation
Status: IMPLEMENTED / SMOKE-TESTED.

Fixed identifier handling so numeric-looking suffixes are kept as strings. This prevents a dangerous transformation such as `S2-00047 -> S2-47`, which would violate the submission contract when source IDs contain leading zeroes.

## P2-003 — Address-number normalization feature view
Status: IMPLEMENTED / SMOKE-TESTED / NOT YET MEASURED ON COMPETITION DATA.

Added leading-zero stripping and common ordinal-word/digit normalization as a feature representation. Legacy measured blocking normalization remains unchanged.
