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
STRIDE = max(1, TOTAL_S1 // SAMPLE_TARGET)  # SAME stride as before -> same sampled rows
CKPT_PATH = "/home/claude/work/run/reachable_checkpoint.json"
MAX_SECONDS = 250

DEFAULT_STATE = {
    "next_sample_index": 0,
    "n_processed": 0,
    "n_with_gt": 0,
    # "reachable" = ground-truth match ids that were actually indexed
    # (i.e. exist in our 2.5M-record S2/S3 SQLite attrs table)
    "s2_reachable_actual": 0, "s2_reachable_found": 0,
    "s3_reachable_actual": 0, "s3_reachable_found": 0,
    "combined_reachable_actual": 0, "combined_reachable_found": 0,
    "combined_unreachable_actual": 0,  # out-of-scope matches (not our fault, not indexed)
    "candidates_per_s1": [],
    "genuine_misses": [],   # reachable but NOT found -> real blocking misses
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
        print("Already complete.")
        return

    print(f"Resuming at sample_index={state['next_sample_index']} "
          f"(processed so far: {state['n_processed']:,}/{SAMPLE_TARGET:,})")

    t_load0 = time.time()
    gt = load_ground_truth_compact(GT_PATH)
    print(f"  ground truth loaded: {len(gt):,} rows in {time.time()-t_load0:.1f}s")

    s2 = SqliteIndex("S2", "/home/claude/work/run/s2_full.sqlite")
    s3 = SqliteIndex("S3", "/home/claude/work/run/s3_full.sqlite")
    indexes = {"S2": s2, "S3": s3}

    def is_indexed(entity_id, source_prefix, index):
        if not entity_id.startswith(source_prefix + "-"):
            return False
        suffix = entity_id[len(source_prefix) + 1:]
        return index.get_attrs(suffix) is not None

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

        entry = gt.get(s1_id)
        if entry:
            state["n_with_gt"] += 1

            for mid in entry:
                src = mid.split("-", 1)[0]
                if src not in ("S2", "S3"):
                    continue
                idx = s2 if src == "S2" else s3
                reachable = is_indexed(mid, src, idx)

                if not reachable:
                    state["combined_unreachable_actual"] += 1
                    continue

                found = mid in got_by_source[src]

                if src == "S2":
                    state["s2_reachable_actual"] += 1
                    state["s2_reachable_found"] += int(found)
                else:
                    state["s3_reachable_actual"] += 1
                    state["s3_reachable_found"] += int(found)

                state["combined_reachable_actual"] += 1
                state["combined_reachable_found"] += int(found)

                if not found and len(state["genuine_misses"]) < 50:
                    state["genuine_misses"].append({
                        "s1_id": s1_id, "name": row["business_name"],
                        "address": row["business_address"], "country": row["country"],
                        "missed_id": mid,
                    })

        next_target_sample_index += 1
        state["next_sample_index"] = next_target_sample_index

        if time.time() - t0 > MAX_SECONDS:
            print(f"  time budget reached ({processed_this_call:,} rows this call) -- checkpointing.")
            break
    else:
        state["done"] = True

    if next_target_sample_index >= SAMPLE_TARGET:
        state["done"] = True

    state["total_elapsed"] += time.time() - t0
    save_state(state)

    print(f"This call: processed {processed_this_call:,} rows in {time.time()-t0:.1f}s")
    print(f"Cumulative: {state['n_processed']:,}/{SAMPLE_TARGET:,}, done={state['done']}")


if __name__ == "__main__":
    main()
