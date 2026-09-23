#!/usr/bin/env python3
"""Add request-level SLO budgets to an existing trace, in place of rewriting it.

The ``casr_*`` traces predate the SLO flags on their generator, so the solver's
``p_slo`` term had nothing to price in every experiment that used them.  Rather
than regenerate (which would redraw arrivals and prefixes and make the new run
incomparable with the recorded ones), this rewrites only the two SLO fields:
every other column, including ``arrival_time_ns`` and the token ids, is copied
byte-for-byte from the source row.

    python3 tests/annotate_trace_slo.py \
        --input workloads/casr_hetero_hot_cold.jsonl \
        --output workloads/casr_hetero_hot_cold-slo500-50.jsonl \
        --slo-ttft-ms 500 --slo-tpot-ms 50

    # Two classes of service in one file: rows tagged decode_tier=slow get the
    # looser budget, so the control plane sees a per-class SLO instead of a
    # single global threshold.
    python3 tests/annotate_trace_slo.py --input ... --tiered
      --slo-ttft-ms 250 --slo-tpot-ms 50 --slow-ttft-ms 800 --slow-tpot-ms 80
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slo-ttft-ms", type=float, default=0.0, dest="slo_ttft_ms")
    parser.add_argument("--slo-tpot-ms", type=float, default=0.0, dest="slo_tpot_ms")
    parser.add_argument("--tiered", action="store_true",
                        help="Give rows tagged decode_tier=slow (or region=edge, "
                             "when no tier tag exists) the second budget.")
    parser.add_argument("--slow-ttft-ms", type=float, default=800.0,
                        dest="slow_ttft_ms")
    parser.add_argument("--slow-tpot-ms", type=float, default=80.0,
                        dest="slow_tpot_ms")
    parser.add_argument("--slow-tag", default="slow",
                        help="Value of decode_tier that counts as the slow class.")
    args = parser.parse_args()

    if args.slo_ttft_ms <= 0 and args.slo_tpot_ms <= 0:
        raise SystemExit("nothing to write: pass --slo-ttft-ms and/or --slo-tpot-ms")

    src = pathlib.Path(args.input)
    dst = pathlib.Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = fast = slow = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tier = str(row.get("decode_tier", ""))
            is_slow = args.tiered and tier == args.slow_tag
            ttft = args.slow_ttft_ms if is_slow else args.slo_ttft_ms
            tpot = args.slow_tpot_ms if is_slow else args.slo_tpot_ms
            if ttft > 0:
                row["slo_ttft_ms"] = float(ttft)
            if tpot > 0:
                row["slo_tpot_ms"] = float(tpot)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            slow += 1 if is_slow else 0
            fast += 0 if is_slow else 1

    print(f"{src.name}: {rows} rows -> {dst}")
    print(f"  tight budget {args.slo_ttft_ms}ms/{args.slo_tpot_ms}ms on {fast} rows"
          f"{f', loose {args.slow_ttft_ms}ms/{args.slow_tpot_ms}ms on {slow}' if args.tiered else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
