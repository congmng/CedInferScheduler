#!/usr/bin/env python3
"""Fail a simulator run that never actually made progress.

The 2026-09-15 deadlock produced a log that looked like a *performance*
problem: instances held dozens of waiting requests, the simulated clock ran to
4314 s, and the run was killed after eleven minutes -- with **zero** requests
completed.  Nothing in the artifacts said "this run is broken"; the numbers
simply looked catastrophic.

Two cheap invariants separate "slow" from "stuck":

* **completion** -- every offered request has an end time (a run that finishes
  200 of 200 requests is not deadlocked), and
* **span** -- the simulated wall clock needed to drain the trace must stay
  within ``--max-slowdown`` of the trace's own arrival span (a backlogged but
  live system stretches; a dead system stretches without bound).

Usage:
    python3 tests/check_sim_liveness.py --csv run.csv \
        --trace workloads/cnndm-short-real-qwen3-8b-600-sps14.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="simulator output csv")
    parser.add_argument("--trace", default="",
                        help="trace replayed (for the arrival span / offered rate)")
    parser.add_argument("--max-slowdown", type=float, default=8.0,
                        dest="max_slowdown",
                        help="allow the drain to take this multiple of the "
                             "trace's arrival span before calling it stuck")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if not rows:
        print("LIVENESS FAILED: no rows in the result csv")
        return 1
    finished = [row for row in rows
                if row.get("end_time") and float(row["end_time"]) > 0]
    arrivals = [float(row["arrival"]) for row in rows if row.get("arrival")]
    ends = [float(row["end_time"]) for row in finished]
    if not finished:
        print("LIVENESS FAILED: 0 requests completed -- this is the deadlock shape")
        return 1
    span_s = (max(ends) - min(arrivals)) / 1e9 if ends and arrivals else 0.0
    trace_span_s = None
    if args.trace:
        with open(args.trace, encoding="utf-8") as handle:
            stamps = [json.loads(line)["arrival_time_ns"] for line in handle
                      if line.strip()]
        if stamps:
            trace_span_s = (max(stamps) - min(stamps)) / 1e9
    report = {"requests": len(rows), "completed": len(finished),
              "span_s": round(span_s, 2),
              "trace_span_s": None if trace_span_s is None else round(trace_span_s, 2),
              "achieved_rps": round(len(finished) / span_s, 2) if span_s else None}
    failures = []
    if len(finished) != len(rows):
        failures.append(f"only {len(finished)}/{len(rows)} requests completed")
    if trace_span_s and span_s > args.max_slowdown * trace_span_s:
        failures.append(
            f"drain took {span_s:.1f}s for a {trace_span_s:.1f}s trace "
            f"(> {args.max_slowdown}x) -- the backlog never cleared")
    if failures:
        print("LIVENESS FAILED: " + "; ".join(failures))
        print(json.dumps(report, ensure_ascii=False))
        return 1
    if not args.quiet:
        print("LIVENESS OK: " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
