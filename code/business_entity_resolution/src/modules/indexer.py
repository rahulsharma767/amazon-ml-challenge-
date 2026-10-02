"""
indexer.py
============================================================
CHANGES from the original (see PHASE1_ARCHITECTURE_AUDIT.md,
"Existing SQLite architecture" and "Checkpoint/resume plan"):

1. bucket() now accepts an optional `limit`. candidates.py passes
   limit = cap + 1. Previously SqliteIndex.bucket() always ran
   `SELECT suffix FROM postings WHERE key_type=? AND key_value=?`
   with no LIMIT and called fetchall() -- for a hot key (a common
   6-character name prefix, say) whose bucket is far larger than
   its frequency cap, this fetched and materialized the ENTIRE
   oversized bucket just to immediately discard it in
   candidates.py's `if bucket_size > cap: continue`. At full
   production scale (S2/S3 ~5M rows each, vs. the 100K-row index
   the 98.45%/97.96% numbers were measured against) hot buckets
   are proportionally larger, so this was on track to become one
   of the biggest avoidable costs in candidate generation. With a
   LIMIT of cap+1, an oversized bucket now costs O(cap) to detect
   as oversized, not O(bucket size).

2. SqliteIndex now commits periodically (every flush(), not only
   at finalize()) and writes a small JSON manifest recording how
   many source rows have been indexed. On restart, build_index()
   checks the manifest: if a source's index previously finished,
   it is reused as-is (no rebuild); if it was left partway
   through, indexing resumes from the last completed row instead
   of starting over. This is what makes a Kaggle session
   interruption DURING index-building recoverable -- previously
   the whole build sat inside one open SQLite transaction with no
   recorded progress, so a kill at row 4,900,000 of 5,000,000 lost
   the entire build.

3. Kaggle-path discovery: `discover_dataset_root()` walks
   /kaggle/input for a directory containing a train_source1.tsv,
   instead of a hard-coded local path. Falls back to a path the
   caller supplies, so this remains usable outside Kaggle too.

Everything else (MemoryIndex, split_entity_id, join_entity_id,
compose_key, the key-generation call sites) is UNCHANGED --
these are what the measured recall numbers depend on.
"""

import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

from blocking import (
    make_name_keys,
    make_unicode_name_keys,
    make_address_keys,
    make_a4_keys,
    make_at2_keys,
)
from data_loader import read_tsv


def split_entity_id(entity_id):
    """Split an ID without changing the suffix representation.

    Entity IDs are identifiers, not numbers: preserving leading zeroes is
    mandatory because the submission must reference the exact IDs present in
    the TSV files. Older code converted numeric suffixes to int, which could
    silently turn S2-00047 into S2-47.
    """
    if "-" not in entity_id:
        return "", entity_id
    prefix, suffix = entity_id.split("-", 1)
    return prefix, suffix


def join_entity_id(prefix, suffix):
    return f"{prefix}-{suffix}"


def compose_key(country, key_value):
    return (country or "").strip().upper() + "||" + key_value


# ============================================================
# KAGGLE PATH DISCOVERY
# ============================================================

def discover_dataset_root(required_files=("train_source1.tsv",), search_root="/kaggle/input"):
    """
    Walk search_root looking for a directory that contains every
    file in required_files. Returns the first match, or None.

    Avoids hard-coding a specific Kaggle dataset slug (the
    notebook skeleton in notebookc647d5dee5.ipynb hard-codes
    ".../tejasweerajput/amazon-ml-challenge-2026-student-resource/
    student_resource" -- if the dataset is re-attached under a
    different slug, or a new version, or the person forks the
    notebook, every path breaks silently). This walks the tree
    once at startup instead.
    """
    root = Path(search_root)
    if not root.exists():
        return None

    for dirpath, _dirnames, filenames in os.walk(root):
        filenames = set(filenames)
        if all(f in filenames for f in required_files):
            return dirpath

    return None


# ============================================================
# IN-MEMORY BACKEND (unchanged)
# ============================================================

class MemoryIndex:

    def __init__(self, source_prefix):
        self.source_prefix = source_prefix
        self.postings = defaultdict(lambda: defaultdict(list))
        self.attrs = {}
        self.count = 0

    def add_record(self, entity_id, name, address, country):

        _, suffix = split_entity_id(entity_id)
        self.attrs[suffix] = (name, address, country)

        for key_type, key_value in make_name_keys(name) + make_unicode_name_keys(name):
            self.postings[key_type][compose_key(country, key_value)].append(suffix)

        for key_type, key_value in make_address_keys(address):
            self.postings[key_type][compose_key(country, key_value)].append(suffix)

        for key_type, key_value in make_a4_keys(address) + make_at2_keys(address):
            self.postings[key_type][compose_key(country, key_value)].append(suffix)

        self.count += 1

    def bucket(self, key_type, key_value, limit=None):
        result = self.postings.get(key_type, {}).get(key_value, [])
        if limit is not None:
            # Already resident in RAM either way; slicing just
            # avoids handing an oversized list to the caller.
            return result[:limit]
        return result

    def get_attrs(self, suffix):
        return self.attrs.get(suffix)

    def num_keys(self):
        return sum(len(v) for v in self.postings.values())


# ============================================================
# SQLITE (DISK-BACKED) BACKEND -- with capped fetch + checkpoint
# ============================================================

class SqliteIndex:

    def __init__(self, source_prefix, db_path, manifest_path=None):

        self.source_prefix = source_prefix
        self.db_path = db_path
        self.manifest_path = manifest_path or (str(db_path) + ".manifest.json")
        self.count = 0

        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = WAL")  # survives a crash mid-write

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS postings "
            "(key_type TEXT, key_value TEXT, suffix TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS attrs "
            "(suffix TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT)"
        )
        self.conn.commit()

        self._buffer = []
        self._attr_buffer = []
        self._index_built = False

    # -------- manifest / resume --------

    def _read_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_manifest(self, rows_indexed, complete):
        manifest = {
            "source_prefix": self.source_prefix,
            "rows_indexed": rows_indexed,
            "complete": complete,
        }
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        os.replace(tmp_path, self.manifest_path)

    def resume_point(self):
        """
        Returns (rows_already_indexed, already_complete). The
        caller (build_index) uses this to decide how many input
        rows to skip, and whether to skip building this source's
        index at all.
        """
        manifest = self._read_manifest()
        if manifest is None:
            return 0, False
        return manifest.get("rows_indexed", 0), manifest.get("complete", False)

    # -------- ingest --------

    def add_record(self, entity_id, name, address, country):

        _, suffix = split_entity_id(entity_id)
        suffix = str(suffix)

        self._attr_buffer.append((suffix, name, address, country))

        for key_type, key_value in make_name_keys(name) + make_unicode_name_keys(name):
            self._buffer.append((key_type, compose_key(country, key_value), suffix))

        for key_type, key_value in make_address_keys(address):
            self._buffer.append((key_type, compose_key(country, key_value), suffix))

        for key_type, key_value in make_a4_keys(address) + make_at2_keys(address):
            self._buffer.append((key_type, compose_key(country, key_value), suffix))

        self.count += 1

        if len(self._buffer) >= 200_000:
            self._flush()

    def _flush(self):

        if self._buffer:
            self.conn.executemany("INSERT INTO postings VALUES (?,?,?)", self._buffer)
            self._buffer = []

        if self._attr_buffer:
            self.conn.executemany(
                "INSERT OR REPLACE INTO attrs VALUES (?,?,?,?)", self._attr_buffer
            )
            self._attr_buffer = []

        # Commit on every flush (not just at finalize()) so a kill
        # between flushes loses at most one buffer's worth of rows,
        # and record how far we got so a restart knows where to
        # resume from.
        self.conn.commit()
        self._write_manifest(rows_indexed=self.count, complete=False)

    def finalize(self):

        self._flush()

        if not self._index_built:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_postings "
                "ON postings(key_type, key_value)"
            )
            self.conn.commit()
            self._index_built = True

        self._write_manifest(rows_indexed=self.count, complete=True)

    # -------- query --------

    def bucket(self, key_type, key_value, limit=None):
        if limit is not None:
            cur = self.conn.execute(
                "SELECT suffix FROM postings WHERE key_type=? AND key_value=? LIMIT ?",
                (key_type, key_value, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT suffix FROM postings WHERE key_type=? AND key_value=?",
                (key_type, key_value),
            )
        return [row[0] for row in cur.fetchall()]

    def get_attrs(self, suffix):
        cur = self.conn.execute(
            "SELECT name, address, country FROM attrs WHERE suffix=?",
            (str(suffix),),
        )
        return cur.fetchone()

    def get_attrs_many(self, suffixes, chunk_size=500):
        """Fetch many attribute rows in bounded SQL batches.

        The old phase-2 path issued one SQLite SELECT per candidate. At
        full 5M-row scale that can become millions of tiny queries. This
        helper keeps the same data semantics while reducing the query count
        by orders of magnitude. Returned mapping preserves string suffixes.
        """
        vals = [str(x) for x in suffixes]
        out = {}
        for start in range(0, len(vals), chunk_size):
            chunk = vals[start:start + chunk_size]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            cur = self.conn.execute(
                f"SELECT suffix, name, address, country FROM attrs WHERE suffix IN ({placeholders})",
                chunk,
            )
            for suffix, name, address, country in cur.fetchall():
                out[str(suffix)] = (name, address, country)
        return out

    def num_keys(self):
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT key_type, key_value FROM postings)"
        )
        return cur.fetchone()[0]


# ============================================================
# BUILD (with resume)
# ============================================================

def build_index(path, source_prefix, limit=None, backend="memory", db_path=None,
                 manifest_path=None):
    """
    Same signature and behavior as before for backend="memory"
    (no persistence is possible for an in-RAM structure across a
    kernel restart -- if the run dies, a memory-backend index must
    be rebuilt from scratch; use backend="sqlite" for anything
    that needs to survive a Kaggle interruption, per the
    checkpoint/resume plan).

    backend="sqlite" now resumes: if db_path + its manifest show a
    previously COMPLETE build for this source, the existing
    database is reopened and returned immediately (no re-read of
    `path` at all). If the manifest shows a partial build, only
    the remaining rows of `path` are read and appended.
    """

    if backend == "sqlite":
        if db_path is None:
            raise ValueError("db_path is required for the sqlite backend")

        index = SqliteIndex(source_prefix, db_path, manifest_path=manifest_path)
        rows_done, complete = index.resume_point()

        if complete:
            print(
                f"[indexer] {source_prefix} index at {db_path} is already "
                f"complete ({rows_done:,} rows) -- reusing, no rebuild."
            )
            index.count = rows_done
            return index

        if rows_done:
            print(
                f"[indexer] Resuming {source_prefix} index at {db_path}: "
                f"{rows_done:,} rows already indexed, continuing from there."
            )
        index.count = rows_done

    else:
        index = MemoryIndex(source_prefix)
        rows_done = 0

    print(f"[indexer] Building {backend} index for {source_prefix} from: {path}")

    for i, row in enumerate(read_tsv(path)):

        if i < rows_done:
            continue  # already indexed in a previous, interrupted run
            # NOTE: this still parses every skipped row (read_tsv is a
            # plain csv.DictReader generator) -- resume avoids redoing
            # key generation + inserts, which dominate cost, but not
            # the comparatively cheap TSV parse. Acceptable trade-off
            # for correctness/simplicity; revisit only if profiling
            # shows the parse itself is significant.

        if limit is not None and i >= limit:
            break

        index.add_record(
            entity_id=row.get("entity_id", ""),
            name=row.get("business_name", ""),
            address=row.get("business_address", ""),
            country=row.get("country", "").strip(),
        )

        if index.count % 500_000 == 0:
            print(f"[indexer]   indexed {index.count:,} {source_prefix} records...")

    if backend == "sqlite":
        index.finalize()

    print(
        f"[indexer] Finished {source_prefix}: {index.count:,} records, "
        f"{index.num_keys():,} distinct blocking keys."
    )

    return index
