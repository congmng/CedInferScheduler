#!/usr/bin/env python3
"""Archive per-request arena statistics that ``summary.json`` does not carry.

``run_hetero_arena``'s summary keeps latency and the Prefill landing, but the
two experiments the paper review asked for need two more columns that only the
per-request CSV has:

* ``exchange`` — whether the request took the P/D handoff (``transfer``) or was
  served locally on its Decode (``local``).  That is the Q2 decision, and the
  output-length sweep plots its share.
* ``output`` — the *realized* output length, so a sweep over
  ``--max-output-tokens`` can report the mean actually produced rather than the
  cap (the pool's own outputs run 13..297 tokens, so asking for 256 mostly
  yields ~70).

    python3 tests/collect_arena_extras.py \
        --r8 /tmp/r8-out{16,64,128,256} --r9 /tmp/r9-tiebreak \
        --out ../docs/arena-summaries

Writes ``r8-output-sweep.json`` and ``r9-tiebreak.json`` next to the other
archived summaries, so the figures are regenerable without the raw run dirs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics


def summarise_csv(path: pathlib.Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    lat = sorted(float(row["latency"]) / 1e6 for row in rows)      # ns -> ms
    exchange = collections.Counter(row["exchange"] for row in rows)
    return {
        "n": len(rows),
        "mean_output_tokens": round(
            statistics.mean(int(row["output"]) for row in rows), 2),
        "local_share": round(exchange.get("local", 0) / len(rows), 4),
        "exchange": dict(exchange),
        "e2e_mean_ms": round(statistics.mean(lat), 1),
        "e2e_p50_ms": round(statistics.median(lat), 1),
        "e2e_p95_ms": round(lat[int(0.95 * len(lat)) - 1], 1),
        "ttft_p50_ms": round(statistics.median(
            float(row["TTFT"]) / 1e6 for row in rows), 1),
        "prefills": dict(collections.Counter(
            row["prefill_instance_id"] for row in rows).most_common(6)),
        "decodes": dict(collections.Counter(
            row["decode_instance_id"] for row in rows).most_common(6)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8", nargs="*", default=[],
                        help="one run dir per --max-output-tokens setting")
    parser.add_argument("--r9", default="", help="the tie-break comparison dir")
    parser.add_argument("--out", default="../docs/arena-summaries")
    parser.add_argument("--name", default="r8-output-sweep",
                        help="basename for the R8 archive (no .json)")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.r8:
        sweep = {}
        for run_dir in args.r8:
            root = pathlib.Path(run_dir)
            for csv_path in sorted(root.glob("*.csv")):
                arm = csv_path.stem
                sweep[f"{root.name}/{arm}"] = summarise_csv(csv_path)
        (out / f"{args.name}.json").write_text(
            json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out / f"{args.name}.json", f"({len(sweep)} entries)")

    if args.r9:
        root = pathlib.Path(args.r9)
        tie = {csv_path.stem: summarise_csv(csv_path)
               for csv_path in sorted(root.glob("*.csv"))}
        (out / "r9-tiebreak.json").write_text(
            json.dumps(tie, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out / "r9-tiebreak.json", f"({len(tie)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
