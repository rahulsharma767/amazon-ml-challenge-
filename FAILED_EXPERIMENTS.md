# FAILED_EXPERIMENTS.md

Nothing rejected this session. Experiment A (cap loosening) was tested and
ACCEPTED (see EXPERIMENT_LOG.md / MEASURED_RESULTS.md). No experiment run
this session came back worse than baseline.

One operational failure, not a modeling one: an initial attempt to build
the S2 index via `nohup ... &` in the background, intending to poll it
across several tool calls, silently lost all progress -- the background
process does not survive past the tool call that started it in this
sandbox. Corrected by switching to foreground, manifest-resumed calls
instead (see HANDOFF_TO_NEXT_CLAUDE.md's operational note). No data or
measured result was lost, only a small amount of wall-clock time.
