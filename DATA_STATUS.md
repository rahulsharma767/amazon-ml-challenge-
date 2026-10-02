# DATA_STATUS.md — verified this session, full files (not samples)

## Row counts (exact, `wc -l` minus header)

```
TRAIN
train_source1.tsv        2,206,821 rows   AVAILABLE
train_source2.tsv        5,034,616 rows   AVAILABLE
train_source3.tsv        5,285,603 rows   AVAILABLE
train_ground_truth.tsv   2,206,821 rows   AVAILABLE  (1:1 with S1)

TEST
test_source1.tsv         1,732,545 rows   AVAILABLE
test_source2.tsv         4,887,274 rows   AVAILABLE
test_source3.tsv         5,082,317 rows   AVAILABLE
```

Columns (all 4 source files): `entity_id, business_name, business_address, country`.
Ground truth columns: `source1_entity_id, matched_entity_ids`.

## Integrity checks (full files, exact)

- Zero duplicate `entity_id` in `train_source1.tsv`.
- Zero empty `business_name` or `business_address` in `train_source1.tsv`.
- `train_ground_truth.tsv` S1 IDs are exactly the same set as
  `train_source1.tsv` IDs (2,206,821 == 2,206,821, set-equal).
- **100% referential integrity**: all 7,638,365 match references across
  every ground-truth row resolve to a real ID in S2 or S3. Zero invalid
  references. (Checked by loading full S2/S3 ID sets into memory, ~46s.)

## Ground-truth match-count distribution (exact, all 2,206,821 rows)

```
0 matches (singleton):  123,247   (5.6%)
1 match:                119,157
2 matches:               375,212
3 matches:                530,841   <- mode
4 matches:                484,115
5 matches:                321,957
6 matches:                164,868
7 matches:                 63,968
8 matches:                 18,680
9 matches:                  4,205
10 matches:                   534
11 matches:                    37   <- max
```

Implication for later stages: singletons are only 5.6% of entities —
tune singleton/no-match calibration accordingly; don't optimize as if
"no match" were the common case, and don't force one-to-one matching
(median is 3-4 matches, i.e. multi-match is normal, not an edge case).

## Country distribution

```
train_source1.tsv:  US 1,323,633  |  India 883,188   (no France in training)
test_source1.tsv:   India 809,986 | US 663,106 | France 259,452  (~15% France)
```

Confirms the problem statement's warning: country must stay an open-set
string field. France is genuinely unseen in training — any hard-coded
US/India logic (address format assumptions, etc.) needs to degrade
gracefully, not break, on France records.

## Not yet done from the full data-audit checklist

- Script/language character-set breakdown (Latin vs Devanagari vs other)
- Name/address length distribution
- Near-duplicate / fuzzy-duplicate detection within a single source
- Full missed-match failure-mode taxonomy (only ~200 raw miss examples
  saved so far, in the recall checkpoints — not yet categorized)
- Test-file-side audit (missingness, duplicates, unusual Unicode) — only
  row counts and country distribution done for test files so far
