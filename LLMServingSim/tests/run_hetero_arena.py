#!/usr/bin/env python3
"""Run the routing/placement arms on the large heterogeneous arena.

The cluster is six domains, one hardware type each (5090, 5090, 4090, 3090,
3090, A100) with a Prefill and a Decode instance per domain, and the workload is
1250-token prompts in a 0.5 -> 4.0 -> 0.5 req/s phase.  Every per-instance cost
the control plane prices (service time, capacity, router capacity, token rate)
comes from the profiler bundles the timeline executes with, so the comparison
is about *placement*, not about mismatched constants.

    python3 tests/run_hetero_arena.py --out /tmp/hetero-arena

Arms: the baselines (``load``, ``cache_aware``, ``rr``), the algorithm with the
small static pool (``casr_lp``), its structural-elasticity variant
(``casr_full``), and the same algorithm with the whole pool already up
(``casr_all6``) -- the last one separates "placement quality" from "how fast the
pool can grow".
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DOMAINS = "5090,5090,4090,3090,3090,a100"
TRACE = "workloads/cnndm-long-hetero6-4rps.jsonl"
NUM_REQS = 376

ARMS = (
    ("load", "baseline", ["--request-routing-policy", "LOAD"]),
    ("cache_aware", "baseline", ["--request-routing-policy", "CACHE_AWARE"]),
    ("rr", "baseline", ["--request-routing-policy", "RR"]),
    ("casr_lp", "algorithm", ["--enable-casr", "--casr-solver", "lp",
                              "--casr-control-interval-ms", "1000"]),
    ("casr_full", "algorithm", ["--enable-casr", "--casr-solver", "lp",
                               "--casr-control-interval-ms", "1000"]),
    ("casr_all6", "algorithm", ["--enable-casr", "--casr-solver", "lp",
                                "--casr-control-interval-ms", "1000"]),
)


def build_configs(out_dir):
    """Three variants of the arena: static-2, elastic, and all-six-active."""
    configs = {}
    for tag, extra in (("static", []), ("elastic", ["--structural"]),
                       ("all6", ["--min-active", "6"])):
        path = out_dir / f"hetero6-{tag}.json"
        subprocess.run([sys.executable, str(REPO / "tests" / "make_hetero_cluster.py"),
                        "--domains", DOMAINS, "--out", str(path), *extra],
                       cwd=REPO, check=True, capture_output=True, text=True)
        configs[tag] = path
    return configs


def ensure_trace():
    trace = REPO / TRACE
    if not trace.exists():
        subprocess.run([sys.executable, str(REPO / "tests" / "make_phased_trace.py"),
                        "--input", "workloads/cnndm-long-pool-qwen3-8b.jsonl",
                        "--rates", "0.5,4.0,0.5", "--durations", "15,90,15",
                        "--names", "warmup,peak,cool", "--output", TRACE],
                       cwd=REPO, check=True)
    return trace


def summarise(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    lat = [float(row["latency"]) / 1e6 for row in rows]
    span = (max(float(row["end_time"]) for row in rows)
            - min(float(row["arrival"]) for row in rows)) / 1e9
    return {
        "n": len(rows),
        "e2e_mean": round(statistics.mean(lat), 1),
        "e2e_p50": round(statistics.median(lat), 1),
        "e2e_p95": round(sorted(lat)[int(0.95 * len(rows)) - 1], 1),
        "ttft_p50": round(statistics.median(
            [float(row["TTFT"]) / 1e6 for row in rows]), 1),
        "tpot_p50": round(statistics.median(
            [float(row["TPOT"]) / 1e6 for row in rows]), 1),
        "span_s": round(span, 1),
        "prefills": dict(collections.Counter(
            row["prefill_instance_id"] for row in rows).most_common(8)),
        "decodes": dict(collections.Counter(
            row["decode_instance_id"] for row in rows).most_common(8)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--arms", default="",
                        help="comma separated subset of " +
                             ",".join(arm for arm, _, _ in ARMS))
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path("/tmp/hetero-arena")
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(out_dir)
    trace = ensure_trace()
    config_for = {"casr_full": configs["elastic"], "casr_all6": configs["all6"]}

    wanted = {name.strip() for name in args.arms.split(",") if name.strip()}
    report = {}
    for arm, kind, extra_args in ARMS:
        if wanted and arm not in wanted:
            continue
        config = config_for.get(arm, configs["static"])
        csv_path = out_dir / f"{arm}.csv"
        command = [sys.executable, "-m", "serving",
                   "--cluster-config", str(config),
                   "--dataset", str(TRACE), "--num-reqs", str(NUM_REQS),
                   "--dtype", "bfloat16", "--block-size", "16",
                   "--max-output-tokens", "16", "--max-num-seqs", "16",
                   "--log-level", "WARNING", "--output", str(csv_path),
                   "--inputs-root", str(out_dir / f"{arm}-inputs"), *extra_args]
        print(f"== {arm} ({kind})", flush=True)
        try:
            subprocess.run(command, cwd=REPO, check=True,
                           stdout=(out_dir / f"{arm}.log").open("w"),
                           stderr=subprocess.STDOUT, timeout=args.timeout_s or None)
        except subprocess.TimeoutExpired:
            print(f"   timeout after {args.timeout_s}s")
            continue
        report[arm] = {"kind": kind, **summarise(csv_path)}

    header = (f"{'arm':<12}{'kind':<11}{'E2E mean':>10}{'p50':>10}{'p95':>10}"
              f"{'TTFT p50':>10}{'TPOT p50':>10}{'span':>8}   Prefill landing")
    print("\n" + header)
    for arm, item in report.items():
        print(f"{arm:<12}{item['kind']:<11}{item['e2e_mean']:>10.1f}"
              f"{item['e2e_p50']:>10.1f}{item['e2e_p95']:>10.1f}"
              f"{item['ttft_p50']:>10.1f}{item['tpot_p50']:>10.1f}"
              f"{item['span_s']:>7.1f}s   {item['prefills']}")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
