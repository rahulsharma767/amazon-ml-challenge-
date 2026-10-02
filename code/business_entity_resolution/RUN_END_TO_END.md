# End-to-end run

## Training

From the `code/business_entity_resolution` directory:

```bash
python train_phase2.py --train-dir /PATH/dataset/train --work-dir artifacts --sample-s1 60000 --hard-negatives 40 --max-candidates 120
```

This builds/reuses full SQLite indexes, constructs positive + hard-negative training pairs, uses GroupKFold by S1 entity, generates OOF predictions, optimizes the official Macro F0.5 on OOF data, and saves `artifacts/final_model.joblib` plus its JSON metadata.

For a final high-compute run, increase `--sample-s1` up to the full S1 training population if memory/time permit. Do not use test data in this stage.

## Frozen test inference

```bash
python infer_test.py --test-dir /PATH/dataset/test --model artifacts/final_model.joblib --out output
```

This builds test-only indexes and performs frozen inference. It does not read ground truth (none exists) and does not tune any parameter.

## Validation

```bash
python validate_outputs.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir /PATH/dataset/test
```

Only a PASS from this validator should be treated as structurally submission-ready.


## Windows full-scale note
The Phase-2 training path was patched to make `--max-candidates` an actual bound, batch SQLite attribute lookups, preserve reachable ground-truth positives, and print the transition into training-pair generation. This avoids the previous one-SQL-query-per-candidate behavior that could make full-scale training appear stuck after S3 indexing.
