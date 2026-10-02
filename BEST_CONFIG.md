# BEST_CONFIG.md — exact parameters to reproduce the current best

File: `modules/blocking.py`, `DEFAULT_FREQ_CAPS` dict:

```python
DEFAULT_FREQ_CAPS = {
    "NP6": 300,
    "NP8": 300,
    "NS6": 300,
    "FL": 400,
    "FL3": 400,
    "F2": 400,
    "FML": 400,
    "AN": 400,
    "A2": 400,
    "AW2": 400,
    "AL2": 120,   # was 80 -- Experiment A, ACCEPTED this session
    "NL": 400,
    "A4W": 15,    # was 10 -- Experiment A, ACCEPTED this session
    "AT2": 30,    # was 20 -- Experiment A, ACCEPTED this session
    "UNP4": 300,
    "UNP6": 300,
    "UFL": 300,
}
```

Everything else (key generation logic, normalization, indexer, candidate
union logic) is unchanged from the previous session's audited/optimized
modules. No feature engineering, model, or threshold config exists yet --
there is nothing else to pin down.
