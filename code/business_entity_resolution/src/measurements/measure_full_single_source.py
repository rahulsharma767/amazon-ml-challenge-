"""
measure_full_single_source.py
============================================================
Measures candidate recall for ONE source (S2 or S3) against its
FULL-SCALE index (all rows, not a 2.5M-row truncation). Because
this sandbox has ~10GB free disk and a full S2 or S3 index is
~6GB, only one full index fits at a time -- so this script
measures S2 and S3 independently (never needs both indexes
loaded simultaneously), unlike the earlier combined script.

Because the index is FULL SCALE for whichever source is loaded,
every ground-truth match for that source is reachable by
definition -- there is no "unreachable, excluded" bucket like the
2.5M-row measurement had. This is a genuine full-scale recall
number for that source, not a partial-index approximation.

Resumable: checkpoints every CHECKPOINT_EVERY rows to a JSON file
so a background run can be killed/restarted without losing
progress. Run repeatedly (or just leave it running via nohup)
until state["done"] is True.

Usage:
    python3 measure_full_single_source.py S2 /path/to/s2_full.sqlite /path/to/ckpt_s2.json [sample_size]
    python3 measure_full_single_source.py S3 /path/to/s3_full.sqlite /path/to/ckpt_s3.json [sample_size]
"""
import sys
import time
import json
import os

from indexer import SqliteIndex
from candidates import find_candidates
from data_loader import read_tsv
from labels import load_ground_truth_compact, matches_for_source

S1_PATH = "/mnt/user-data/uploads/train_source1_1.tsv"
GT_PATH = "/mnt/user-data/uploads/train_ground_truth_1.tsv"
TOTAL_S1 = 2206821


def load_state(ckpt_path, sample_target, stride):
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {
        "sample_target": sample_target,
        "stride": stride,
        "next_sample_index": 0,
        "n_processed": 0,
        "n_with_gt": 0,
        "actual": 0,
        "found": 0,
        "candidates_per_s1": [],
        "missed_examples": [],
        "done": False,
    }


def save_state(state, ckpt_path):
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, ckpt_path)


def main():
    source = sys.argv[1]           # "S2" or "S3"
    db_path = sys.argv[2]
    ckpt_path = sys.argv[3]
    sample_target = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
    max_seconds = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9  # effectively unlimited; we run via nohup

    stride = max(1, TOTAL_S1 // sample_target)
    state = load_state(ckpt_path, sample_target, stride)

    if state["done"]:
        print(f"[{source}] already complete -- see {ckpt_path}")
        return

    print(f"[{source}] resuming at sample_index={state['next_sample_index']} "
          f"({state['n_processed']:,}/{state['sample_target']:,})", flush=True)

    t_load0 = time.time()
    gt = load_ground_truth_compact(GT_PATH)
    print(f"[{source}] ground truth loaded: {len(gt):,} rows in {time.time()-t_load0:.1f}s", flush=True)

    index = SqliteIndex(source, db_path)

    target_row_offset = state["next_sample_index"] * stride
    next_target_sample_index = state["next_sample_index"]

    t0 = time.time()
    last_ckpt = time.time()

    for i, row in enumerate(read_tsv(S1_PATH)):

        if i < target_row_offset:
            continue
        if (i - target_row_offset) % stride != 0:
            continue
        if next_target_sample_index >= state["sample_target"]:
            state["done"] = True
            break

        s1_id = row["entity_id"]
        gt_entry = gt.get(s1_id)
        gt_matches = matches_for_source(gt_entry, source) if gt_entry else set()

        if gt_matches:
            state["n_with_gt"] += 1
            state["actual"] += len(gt_matches)

        cands = find_candidates(row, index)  # dict: suffix -> {"key_types":..., "min_freq":...}
        cand_id_set = {f"{source}-{suf}" for suf in cands.keys()}

        found = gt_matches & cand_id_set
        missed = gt_matches - cand_id_set
        state["found"] += len(found)

        if missed and len(state["missed_examples"]) < 200:
            state["missed_examples"].append({
                "s1_id": s1_id,
                "missed": list(missed)[:5],
            })

        state["candidates_per_s1"].append(len(cand_id_set))
        state["n_processed"] += 1
        next_target_sample_index += 1
        state["next_sample_index"] = next_target_sample_index

        if state["n_processed"] % 500 == 0:
            elapsed = time.time() - t0
            rate = state["n_processed"] / elapsed if elapsed > 0 else 0
            recall = state["found"] / state["actual"] if state["actual"] else 0
            print(f"[{source}] processed {state['n_processed']:,}/{state['sample_target']:,} "
                  f"recall_so_far={recall:.4f} rate={rate:.1f} rows/s elapsed={elapsed:.0f}s", flush=True)

        if time.time() - last_ckpt > 30:
            save_state(state, ckpt_path)
            last_ckpt = time.time()

        if time.time() - t0 > max_seconds:
            print(f"[{source}] hit time budget, checkpointing and stopping for this call.", flush=True)
            break

    save_state(state, ckpt_path)

    if state["done"]:
        recall = state["found"] / state["actual"] if state["actual"] else 0
        print(f"[{source}] DONE. n_processed={state['n_processed']:,} "
              f"actual={state['actual']:,} found={state['found']:,} recall={recall:.4%}", flush=True)
    else:
        print(f"[{source}] paused at {state['n_processed']:,}/{state['sample_target']:,} -- rerun to continue.", flush=True)


if __name__ == "__main__":
    main()
