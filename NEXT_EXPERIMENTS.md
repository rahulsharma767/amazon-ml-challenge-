# NEXT_EXPERIMENTS.md — ranked

1. Address leading-zero normalization (strip leading zeros from numeric
   address tokens before AN/A2/NL/A4W key generation). Real evidenced miss
   found this session (S3-383257895: "1528" vs "01528").
2. Ordinal-number word<->digit normalization ("12th" vs "twelfth"). Real
   evidenced misses found this session (S2-993815983, S2-951619159).
3. Experiment B caps (AL2=150, A4W=20, AT2=40) -- test whether loosening
   further keeps paying off or has plateaued after Experiment A's gain.
4. Character n-gram / TF-IDF retrieval as a new candidate channel, unioned
   with existing deterministic keys -- targets the ~36-47% "other partial
   overlap" / "no lexical overlap" taxonomy buckets that no cap tuning can
   reach.
5. Full-scale test-file audit (only row counts + country distribution done
   so far, from the previous session).

Each of these should be measured with the same before/after, same-sample
methodology used for Experiment A -- see MEASURED_RESULTS.md as the
template.
