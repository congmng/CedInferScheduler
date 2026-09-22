#!/usr/bin/env python3
"""Merge P-15B's per-block-type bundles into one, checking the overlap agrees.

The profiler always builds ``num_hidden_layers=1``, so P-15B's three block
types (r0 / r4 / r128) are three separate runs.  Their canonical layer names
were split *per type* in ``profiler/models/p15b.yaml`` -- but only for the
pieces that actually differ (``compressor_csa`` vs ``compressor_hca`` and
friends).  Everything else (``embedding``, ``qkv_down``, ``q_up``, ``o_lora_*``,
``moe``, the norms) is the same shape in all three and therefore appears in all
three CSVs under the *same* row key.

That overlap is useful rather than awkward: the shared layers are three
independent measurements of the same thing on the same card, so their spread is
a free run-to-run noise estimate.  This tool keeps the first row for a
duplicated key and reports the largest relative difference it saw -- anything
large means the runs were not comparable (different grid knobs, throttling, a
shared card).

    python3 tests/merge_profile_types.py --out-root profiler/perf \
        --hardware RTX4090 --model casr/P15B --variant bf16 --tp 1 \
        --type-root /tmp/p15b-formal
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Row identity per category -- mirrors ``merge_profile_shards.py`` so the two
#: tools agree about what "the same shot" means.
KEY_COLUMNS = {
    "attention.csv": ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode"),
    "dense.csv": ("layer", "tokens"),
    "per_sequence.csv": ("layer", "sequences"),
    "moe.csv": ("tokens", "activated_experts"),
    "skew.csv": ("n", "nb", "pc", "kp", "kvs", "regime"),
}

#: ``attention`` is the one category the three block types genuinely disagree
#: about: r0 attends causally over the prefix, r4 over a window of 8 plus top-k
#: states, r128 over a window of 128.  Measured on a small grid, the same shot
#: key differs by up to 77% between types -- so they cannot share a table, and
#: merging them by key would silently pick one type's operator for all three.
#: They are written side by side instead, which also leaves the evidence in the
#: bundle for whoever teaches the simulator to pick per block type.
PER_TYPE = ("attention.csv",)


def _read(path: pathlib.Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def _bundle(type_root: pathlib.Path, block: str, hardware: str, model: str,
            variant: str, tp: int) -> pathlib.Path:
    """One block type's bundle *directory holding the CSVs* (``.../tp<N>``)."""
    return (type_root / block / hardware
            / model.replace("P15B", f"P15B-{block}") / variant / f"tp{tp}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--model", default="casr/P15B",
                        help="destination model id; sources are derived per type")
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--type-root", required=True,
                        help="directory holding one out-root per block type")
    parser.add_argument("--blocks", default="r0,r4,r128")
    parser.add_argument("--max-spread", type=float, default=0.15,
                        help="fail if a shared layer's *median* relative "
                             "difference across its rows exceeds this")
    args = parser.parse_args()

    blocks = [b for b in args.blocks.split(",") if b]
    type_root = pathlib.Path(args.type_root)
    sources = {block: _bundle(type_root, block, args.hardware, args.model,
                              args.variant, args.tp) for block in blocks}
    missing = [b for b, path in sources.items() if not path.is_dir()]
    if missing:
        raise SystemExit(f"missing bundle(s) for {missing}: "
                         + ", ".join(str(sources[b]) for b in missing))

    dest = (pathlib.Path(args.out_root) / args.hardware / args.model / args.variant
            / f"tp{args.tp}")
    dest.mkdir(parents=True, exist_ok=True)

    for name, key_columns in KEY_COLUMNS.items():
        if name in PER_TYPE:
            for block in blocks:
                piece = sources[block] / name
                if not piece.is_file():
                    continue
                suffix = "" if block == blocks[0] else f"_{block}"
                target = dest / f"{name[:-4]}{suffix}.csv"
                target.write_bytes(piece.read_bytes())
                rows = sum(1 for _ in csv.DictReader(
                    target.open(newline="", encoding="utf-8")))
                print(f"{name}: kept {rows} rows of {block} as {target.name} "
                      f"(per block type -- the operator differs)")
            continue
        pieces = [sources[b] / name for b in blocks if (sources[b] / name).is_file()]
        if not pieces:
            print(f"{name}: absent in every type (ok for moe/skew)")
            continue
        header = None
        rows: list[dict] = []
        seen: dict[tuple, tuple[dict, str]] = {}
        duplicates = 0
        worst = (0.0, None)
        # Per-row spreads are noisy for small ops (one 448-token RMSNorm can
        # swing 28% at --measurement-iterations 1).  What says "these runs are
        # comparable" is a *systematic* difference, so aggregate per layer and
        # judge the median; the worst single row is reported for context only.
        per_group: dict[str, list[float]] = {}
        for piece in pieces:
            piece_header, piece_rows = _read(piece)
            header = header or piece_header
            if piece_header != header:
                raise SystemExit(f"{piece}: header differs from the first bundle")
            for row in piece_rows:
                key = tuple(row[column] for column in key_columns)
                if key not in seen:
                    seen[key] = (row, piece.parent.parent.parent.name)
                    rows.append(row)
                    continue
                duplicates += 1
                first = seen[key][0]
                for column in header:
                    if column in key_columns:
                        continue
                    try:
                        a, b = float(first[column]), float(row[column])
                    except ValueError:
                        continue
                    if a == b == 0:
                        continue
                    spread = abs(b - a) / max(abs(a), abs(b), 1e-9)
                    group = key[0]
                    per_group.setdefault(group, []).append(spread)
                    if spread > worst[0]:
                        worst = (spread, f"{column}@{key}")
        target = dest / name
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        medians = {group: sorted(values)[len(values) // 2]
                   for group, values in per_group.items()}
        offenders = {g: v for g, v in medians.items() if v > args.max_spread}
        print(f"{name}: {len(rows)} rows ({duplicates} duplicated keys kept once; "
              f"worst row {worst[0]:.1%}, worst layer median "
              f"{max(medians.values(), default=0.0):.1%}) -> {target}")
        if offenders:
            raise SystemExit(
                f"these shared layers disagree systematically across block types "
                f"(median > {args.max_spread:.0%}): "
                + ", ".join(f"{g} {v:.1%}" for g, v in sorted(offenders.items())))

    # meta.yaml sits at the variant level, one directory above the tp folder
    # that holds the CSVs.
    meta = yaml.safe_load(
        (sources[blocks[0]].parent / "meta.yaml").read_text(encoding="utf-8"))
    meta["model"] = args.model
    meta["merged_types"] = blocks
    meta["source_bundles"] = [str(sources[b].relative_to(type_root)) for b in blocks]
    meta["architectures"] = None
    (dest.parent / "meta.yaml").write_text(
        yaml.dump(meta, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"meta.yaml written for {args.model} from {blocks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
