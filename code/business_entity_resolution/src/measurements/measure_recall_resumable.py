import time
import json
import os
import statistics
from indexer import SqliteIndex
from candidates import generate_candidate_pairs_for_row
from data_loader import read_tsv
from labels import load_ground_truth_compact, matches_for_source

S1_PATH = "/mnt/user-data/uploads/train_source1.tsv"
GT_PATH = "/mnt/user-data/uploads/train_ground_truth.tsv"
TOTAL_S1 = 2206821
SAMPLE_TARGET = 12000
STRIDE = max(1, TOTAL_S1 // SAMPLE_TARGET)
CKPT_PATH = "/home/claude/work/run/recall_checkpoint.json"
MAX_SECONDS = 250  # stay well under the 300s hard call limit

DEFAULT_STATE = {
    "next_sample_index": 0,   # how many stride-steps we've consumed so far
    "n_processed": 0,
    "n_with_gt": 0,
    "s2_actual": 0, "s2_found": 0,
    "s3_actual": 0, "s3_found": 0,
    "combined_actual": 0, "combined_found": 0,
    "candidates_per_s1": [],
    "missed_examples": [],
    "total_elapsed": 0.0,
    "done": False,
}


def load_state():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_STATE)


def save_state(state):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CKPT_PATH)


def main():
    state = load_state()

    if state["done"]:
        print("Already complete -- see recall_checkpoint.json. Run summarize_recall.py.")
        return

    print(f"Resuming at sample_index={state['next_sample_index']} "
          f"(processed so far: {state['n_processed']:,}/{SAMPLE_TARGET:,})")

    t_load0 = time.time()
    gt = load_ground_truth_compact(GT_PATH)
    print(f"  ground truth loaded: {len(gt):,} rows in {time.time()-t_load0:.1f}s")

    s2 = SqliteIndex("S2", "/home/claude/work/run/s2_full.sqlite")
    s3 = SqliteIndex("S3", "/home/claude/work/run/s3_full.sqlite")
    indexes = {"S2": s2, "S3": s3}

    target_row_offset = state["next_sample_index"] * STRIDE
    next_target_sample_index = state["next_sample_index"]

    t0 = time.time()
    processed_this_call = 0

    for i, row in enumerate(read_tsv(S1_PATH)):

        if i < target_row_offset:
            continue

        if (i - target_row_offset) % STRIDE != 0:
            continue

        if next_target_sample_index >= SAMPLE_TARGET:
            state["done"] = True
            break

        s1_id = row["entity_id"]
        pairs = generate_candidate_pairs_for_row(row, indexes)

        state["n_processed"] += 1
        processed_this_call += 1
        state["candidates_per_s1"].append(len(pairs))

        got_by_source = {"S2": set(), "S3": set()}
        for p in pairs:
            got_by_source[p["source"]].add(p["candidate_entity_id"])
        got_all = got_by_source["S2"] | got_by_source["S3"]

        entry = gt.get(s1_id)
        if entry:
            state["n_with_gt"] += 1

            gt_s2 = matches_for_source(entry, "S2-")
            gt_s3 = matches_for_source(entry, "S3-")

            state["s2_actual"] += len(gt_s2)
            state["s2_found"] += len(gt_s2 & got_by_source["S2"])

            state["s3_actual"] += len(gt_s3)
            state["s3_found"] += len(gt_s3 & got_by_source["S3"])

            state["combined_actual"] += len(entry)
            state["combined_found"] += len(entry & got_all)

            missed = entry - got_all
            if missed and len(state["missed_examples"]) < 30:
                state["missed_examples"].append({
                    "s1_id": s1_id, "name": row["business_name"],
                    "address": row["business_address"], "country": row["country"],
                    "missed_ids": sorted(missed)[:5],
                })

        next_target_sample_index += 1
        state["next_sample_index"] = next_target_sample_index

        if time.time() - t0 > MAX_SECONDS:
            print(f"  time budget reached this call ({processed_this_call:,} rows) -- "
                  f"checkpointing and stopping for now.")
            break
    else:
        state["done"] = True

    if next_target_sample_index >= SAMPLE_TARGET:
        state["done"] = True

    state["total_elapsed"] += time.time() - t0
    save_state(state)

    print(f"This call: processed {processed_this_call:,} rows in {time.time()-t0:.1f}s")
    print(f"Cumulative: {state['n_processed']:,}/{SAMPLE_TARGET:,} sample rows, "
          f"done={state['done']}")


if __name__ == "__main__":
    main()
