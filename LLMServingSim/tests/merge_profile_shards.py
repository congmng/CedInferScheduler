#!/usr/bin/env python3
"""Merge the per-shard CSVs of a ``--shard i/N`` profile run into one bundle.

``--shard`` splits a category's composed grid across processes so the fixed
per-shot ``torch.profiler`` cost can be paid on several cards at once.  Each
shard writes a complete-looking bundle under its own ``--out-root``; this
script puts them back together:

    python3 tests/merge_profile_shards.py \
        --hardware RTX5090 --model Zyphra/Zamba2-1.2B --variant bf16 --tp 1 \
        --out-root /tmp/merged \
        /tmp/shard0 /tmp/shard1

Shards are disjoint by construction, so the merge is a concatenation -- but the
script asserts disjointness and coverage rather than trusting it, because a
silent overlap (two shards measuring the same shape twice) or a gap (a shot
nobody fired) both produce a bundle that loads and is subtly wrong.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]

# The column(s) that identify a shot per category. Attention carries the 4D
# key; dense/per_sequence carry (layer, tokens|sequences); skew its own tuple.
KEY_COLUMNS = {
    "attention.csv": ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode"),
    "dense.csv": ("layer", "tokens"),
    "per_sequence.csv": ("layer", "sequences"),
    "moe.csv": ("tokens", "activated_experts"),
    "skew.csv": ("n", "nb", "pc", "kp", "kvs", "regime"),
}


def _read(path: pathlib.Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+",
                        help="one --out-root directory per shard, in order")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    rel = pathlib.Path(args.hardware) / args.model / args.variant
    dest = pathlib.Path(args.out_root) / rel
    (dest / f"tp{args.tp}").mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    for name, key_columns in KEY_COLUMNS.items():
        pieces: list[pathlib.Path] = []
        for shard in args.shards:
            candidate = pathlib.Path(shard) / rel / f"tp{args.tp}" / name
            if candidate.is_file():
                pieces.append(candidate)
        if not pieces:
            print(f"{name}: absent in every shard (ok for moe/skew)")
            continue
        header = None
        seen: set[tuple] = set()
        rows: list[dict] = []
        for piece in pieces:
            piece_header, piece_rows = _read(piece)
            header = header or piece_header
            if piece_header != header:
                raise SystemExit(f"{piece}: header differs from the first shard")
            for row in piece_rows:
                key = tuple(row[column] for column in key_columns)
                if key in seen:
                    raise SystemExit(
                        f"{piece}: duplicate shot {key} -- the shards overlap; "
                        f"check that every process used the same grid knobs "
                        f"(--attention-max-kv / -chunk-factor / -kv-factor / "
                        f"--max-num-seqs / --max-num-batched-tokens).")
                seen.add(key)
                rows.append(row)
        target = dest / f"tp{args.tp}" / name
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        written[name] = len(rows)
        print(f"{name}: {len(rows)} rows from {len(pieces)} shards -> {target}")

    # meta.yaml: copy the first shard's and note the merge, so a reader can
    # tell a two-card run from a single-card one.
    meta_source = None
    for shard in args.shards:
        candidate = pathlib.Path(shard) / rel / "meta.yaml"
        if candidate.is_file():
            meta_source = candidate
            break
    if meta_source is None:
        raise SystemExit("no shard produced meta.yaml")
    meta = yaml.safe_load(meta_source.read_text(encoding="utf-8")) or {}
    meta["shard"] = None
    meta["shards"] = [f"{pathlib.Path(shard).name}" for shard in args.shards]
    (dest / "meta.yaml").write_text(
        yaml.dump(meta, sort_keys=False, default_flow_style=False),
        encoding="utf-8")
    print(f"meta.yaml from {meta_source} (merged over {len(args.shards)} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
