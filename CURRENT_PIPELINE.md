# CURRENT_PIPELINE.md

```
S1 row (streamed)
  -> make_name_keys + make_unicode_name_keys(name)
  -> make_address_keys + make_a4_keys + make_at2_keys(address)
  -> for each (key_type, key_value):
       composed = compose_key(country, key_value)
       bucket = index.bucket(key_type, composed, limit=cap+1)   # cap = DEFAULT_FREQ_CAPS[key_type]
       if len(bucket) > cap: skip this key   # too broad, dropped
       else: union bucket's suffixes into this row's candidate set
  -> runs once against S2 index, once against S3 index
  -> candidate_pairs rows: one per (S1, candidate) that survived
```

Current DEFAULT_FREQ_CAPS: see BEST_CONFIG.md (Experiment A values, applied
in modules/blocking.py).

Nothing downstream of candidate generation exists yet: no feature
computation is wired into a training pipeline, no model, no threshold, no
singleton/multi-match policy, no test inference, no output files. The
`features.py` module has 18 pairwise similarity features implemented and
smoke-tested (from the previous session's Phase 1 audit) but is not yet
called end-to-end over the candidate set produced above -- that wiring,
plus everything in NEXT_EXPERIMENTS.md item 4 onward, is the actual next
phase of work.
