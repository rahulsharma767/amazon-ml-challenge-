# Amazon ML Challenge 2026 — Finalized Engineering Checkpoint

This is the completed implementation built by extending the supplied v3 checkpoint rather than replacing it.

The deterministic blocking stage and its measured Experiment-A baseline are preserved. The previously missing Phase-2 matcher, validation, calibration, test inference, output generation and structural validation are now implemented under `code/business_entity_resolution/`.

**Do not interpret the supplied 95.21% number as final F0.5.** It is candidate recall only. The real competition model must be trained on the real training TSVs and then run on the frozen test TSVs.

See `FINAL_STATUS.md` and `code/business_entity_resolution/RUN_END_TO_END.md`.
