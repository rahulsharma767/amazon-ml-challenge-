"""
checkpoint.py
============================================================
Generic checkpoint/resume support for long-running, shard-based
stages of the pipeline (index building, candidate generation +
feature calculation, scoring).

Design goals:
  - A Kaggle session interruption at ANY point must lose at most
    one shard's worth of work, never the whole run.
  - Resuming must be a no-op decision ("skip, already done") for
    every shard that already finished, so restarting the same
    notebook cell is always safe to re-run.
  - No external state beyond plain files already in
    /kaggle/working, so it survives a kernel restart on the same
    persisted working directory.

This module knows nothing about business logic (blocking,
features, etc.) -- it only tracks "which shards of which stage
are done", so it is shared by every stage of the pipeline.

Manifest layout (one JSON file per stage):

    {
      "stage": "candidates_test_full",
      "shard_size": 50000,
      "total_rows": 1730000,
      "completed_shards": [0, 1, 2, ...],
      "updated_at": "2026-09-26T12:00:00"
    }

Each completed shard's OUTPUT is written to its own file
(e.g. candidate_pairs_shard_0007.tsv). On resume, a shard is
only re-run if it is missing from completed_shards, so half-
written output files from a killed process are simply
overwritten -- there is never a "partially valid" shard file.
"""

import json
import os
import tempfile
from datetime import datetime, timezone


def _atomic_write_json(path, data):
    """
    Write JSON atomically: write to a temp file in the same
    directory, then os.replace() it over the target. This means
    a kill -9 mid-write can never leave a corrupt manifest --
    readers always see either the old or the new manifest, never
    a half-written one.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".ckpt_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_manifest(manifest_path, stage, shard_size, total_rows):
    """
    Load an existing manifest, or create a fresh in-memory one if
    none exists yet or it doesn't match the current run's shape
    (stage name / shard_size / total_rows). A shape mismatch means
    the run configuration changed, so we start over rather than
    silently reusing shard boundaries computed under a different
    shard_size.
    """
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if (
            data.get("stage") == stage
            and data.get("shard_size") == shard_size
            and data.get("total_rows") == total_rows
        ):
            data["completed_shards"] = set(data.get("completed_shards", []))
            return data

        print(
            f"[checkpoint] manifest at {manifest_path} does not match "
            f"current run shape -- starting a fresh manifest for '{stage}'"
        )

    return {
        "stage": stage,
        "shard_size": shard_size,
        "total_rows": total_rows,
        "completed_shards": set(),
        "updated_at": None,
    }


def save_manifest(manifest_path, manifest):
    to_write = dict(manifest)
    to_write["completed_shards"] = sorted(manifest["completed_shards"])
    to_write["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(manifest_path, to_write)


def mark_shard_complete(manifest_path, manifest, shard_index):
    """
    Call this AFTER a shard's output file has been fully written
    and closed (renamed into place if you write-then-rename), so
    the manifest can never claim a shard is done while its output
    file is still incomplete.
    """
    manifest["completed_shards"].add(shard_index)
    save_manifest(manifest_path, manifest)


def is_shard_complete(manifest, shard_index):
    return shard_index in manifest["completed_shards"]


def iter_shards(total_rows, shard_size):
    """
    Yield (shard_index, start_row, end_row) triples covering
    [0, total_rows) in fixed-size, deterministic chunks. Using a
    fixed shard_size (rather than "N shards") means adding rows
    to a source file never changes the boundaries of shards that
    already completed.
    """
    shard_index = 0
    start = 0
    while start < total_rows:
        end = min(start + shard_size, total_rows)
        yield shard_index, start, end
        start = end
        shard_index += 1


def shard_output_path(out_dir, prefix, shard_index):
    return os.path.join(out_dir, f"{prefix}_shard_{shard_index:05d}.tsv")


def write_shard_atomically(final_path, write_fn):
    """
    write_fn(tmp_path) must fully write the shard's content to
    tmp_path. This function then renames it into place, so any
    reader only ever sees a complete file at final_path, and a
    process killed mid-write leaves only an orphaned .tmp file
    (never a truncated final_path).
    """
    directory = os.path.dirname(final_path) or "."
    os.makedirs(directory, exist_ok=True)

    tmp_path = final_path + ".tmp"
    write_fn(tmp_path)
    os.replace(tmp_path, final_path)


def concat_completed_shards(out_dir, prefix, manifest, header_line, final_path):
    """
    Convenience for the last step of a stage: stitch every
    completed shard's file into one final output, in shard order.
    Safe to call multiple times (idempotent) since it always
    rewrites final_path from the shard files, and only ever reads
    shards recorded as complete in the manifest.
    """
    shard_indices = sorted(manifest["completed_shards"])

    def _write(tmp_path):
        with open(tmp_path, "w", encoding="utf-8", newline="") as out_f:
            out_f.write(header_line.rstrip("\n") + "\n")
            for shard_index in shard_indices:
                shard_path = shard_output_path(out_dir, prefix, shard_index)
                with open(shard_path, "r", encoding="utf-8") as in_f:
                    next(in_f, None)  # skip that shard's own header
                    for line in in_f:
                        out_f.write(line)

    write_shard_atomically(final_path, _write)
    return final_path
