#!/usr/bin/env python3
"""Compare routing arms on one cluster config + trace, with the KV path visible.

The three-arm comparison harness (``run_casr_comparison.sh``) fixes the arms at
static / greedy / lp; the questions that keep coming up need a *routing policy*
arm (``load`` vs ``cache_aware`` vs ``kv_aware``) or a different transport
setting, and they need to see **how the KV actually travelled** -- a "win" that
comes from recomputing locally is a different claim from one that comes from
moving compressed KV across the fabric.

    python3 tests/run_route_arms.py \
        --cluster-config configs/cluster/casr_p15b_three_domain.json \
        --dataset workloads/casr-peak30-1250tok-slo-tiered.jsonl \
        --arms load,cache_aware,kv_aware,casr_lp --out-root /tmp/arms

Each arm runs ``python -m serving`` in its own output directory; the summary
reports latency/TTFT/TPOT, SLO attainment and the ``exchange`` split (what the
router's recompute-vs-transfer decision chose).
"""

from __future__ import annotations

import argparse
import copy
import collections
import csv
import json
import pathlib
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

#: arm name -> extra CLI flags
ARMS = {
    "load": ["--request-routing-policy", "LOAD"],
    "load_rr": ["--request-routing-policy", "LOAD_RR"],
    "cache_aware": ["--request-routing-policy", "CACHE_AWARE"],
    "kv_aware": ["--request-routing-policy", "KV_AWARE"],
    "rr": ["--request-routing-policy", "RR"],
    "static": ["--no-enable-casr"],
    "greedy": ["--enable-casr", "--casr-solver", "greedy"],
    "casr_lp": ["--enable-casr", "--casr-solver", "lp"],
    # The same solver with a *fixed single-worker pool*: the arena's
    # ``casr_lp`` (static) against its ``casr_full`` (elastic).  The pool size
    # is a config property, not a flag, so these two need ``--static-config``.
    "casr_static": ["--enable-casr", "--casr-solver", "lp", "@static"],
    "casr_elastic": ["--enable-casr", "--casr-solver", "lp"],
    # A *fixed* pool the same size as the baseline's, so the comparison isolates
    # placement at equal capacity (the arena's casr_lp vs casr_full conflates the
    # two; pairing a 1-worker pool against a 3-worker baseline measures capacity).
    "casr_plan3": ["--enable-casr", "--casr-solver", "lp", "@fixed"],
    # -- SOTA baselines added 2026-09-24 (docs/SOTA覆盖与模拟器基线映射.md) --
    # DOPD-style autoscaling: pool size follows utilisation/queue thresholds
    # with hysteresis instead of a counterfactual gain evaluation.
    "dopd": ["--enable-casr", "--casr-solver", "lp", "@overlay"],
    # PrfaaS-style cross-DC Prefill offload: local DC first, then a hard
    # transfer budget.  ``prfaas_tight`` lowers that budget so the comparison
    # shows what the threshold itself buys.
    "prfaas": ["--request-routing-policy", "PRFAAS"],
    "prfaas_tight": ["--request-routing-policy", "PRFAAS",
                     "--prfaas-max-offload-ms", "150"],
    # ServerlessLLM-style fast loading: the same elastic pool, with the boot
    # modelled as engine init + weights / storage bandwidth (see
    # ``serving/core/boot_model.py``).
    "serverless_elastic": ["--enable-casr", "--casr-solver", "lp", "@overlay"],
    # Llumnix-style migration on top of the load arm: queue rebalancing
    # between Decodes, charged as the KV copy it is.
    "llumnix": ["--request-routing-policy", "LOAD", "@overlay"],
    # DistServe/Splitwise-style P:D ratio: the pool the search picked
    # (``tests/pd_ratio_search.py --write-config``), with the baseline's own
    # routing so the arm isolates the *ratio* decision.
    "distserve": ["--request-routing-policy", "LOAD", "@fixed"],
    "distserve_lp": ["--enable-casr", "--casr-solver", "lp", "@fixed"],
}

#: Per-arm cluster-config overlays, deep-merged into a copy of
#: ``--cluster-config``.  Keeping them here (instead of duplicating the whole
#: cluster JSON per arm) is what makes "one variable changed" checkable.
OVERLAYS = {
    "dopd": {"casr": {"structural": {
        "rule": "dopd",
        "scale_out_utilization": 0.8,
        "scale_in_utilization": 0.3,
        "scale_ticks": 3,
        "queue_scale_out_ratio": 0.5,
        "threshold_hold_ms": 5000,
    }}},
    "serverless_elastic": {"casr": {"boot": {
        "model": "weight_load",
        # engine init (17.2 s) and the P-15B weights (15.26 GiB) read from
        # local NVMe at 3 GB/s, overlapped with initialisation -> 17.2 s.
        "load_bandwidth_bytes_per_s": 3.0e9,
        "overlap": True,
    }}},
    "llumnix": {"casr": {
        # Rebalance every simulated 10 ms (a scheduling round), with loose
        # thresholds: the *cost/benefit* gate is what decides a move, and the
        # arm's finding on this fabric is that nothing clears it.
        "llumnix_interval_ms": 10.0,
        "llumnix_hot_ms": 10.0,
        "llumnix_cold_ms": 5.0,
        "llumnix_min_gain_ms": 5.0,
        "llumnix_gain_ratio": 2.0,
        "llumnix_batch": 4,
    }},
}


def _merge(base, overlay):
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def summarise(path: pathlib.Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {path}")

    def scaled(name):
        return sorted(float(row[name]) / 1e6 for row in rows)

    def q(values, frac):
        return values[min(len(values) - 1, int(frac * len(values)))]

    latency, ttft, tpot = scaled("latency"), scaled("TTFT"), scaled("TPOT")
    judged = [row for row in rows if str(row.get("slo_ok", "")).strip() != ""]
    met = sum(1 for row in judged if str(row["slo_ok"]).strip().lower() == "true")
    arrivals = [float(row["arrival"]) / 1e9 for row in rows]
    ends = [float(row["end_time"]) / 1e9 for row in rows]
    return {
        "n": len(rows),
        "mean": statistics.fmean(latency),
        "p50": q(latency, 0.50),
        "p95": q(latency, 0.95),
        "ttft_p50": q(ttft, 0.50),
        "tpot_p50": q(tpot, 0.50),
        "attainment": (100.0 * met / len(judged)) if judged else None,
        "span_s": max(ends) - min(arrivals),
        "exchange": dict(collections.Counter(row.get("exchange", "") for row in rows)),
        "prefills": dict(collections.Counter(row.get("prefill_instance_id", "")
                                             for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arms", default="load,cache_aware,casr_lp")
    parser.add_argument("--out-root", default="/tmp/route-arms")
    parser.add_argument("--num-reqs", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--control-interval-ms", type=int, default=100)
    parser.add_argument("--client-concurrency", type=int, default=0)
    parser.add_argument("--fixed-config", default=None,
                        help="Cluster config used by arms flagged '@fixed' "
                             "(a fixed pool the same size as the baseline's).")
    parser.add_argument("--static-config", default=None,
                        help="Cluster config used by arms flagged '@static' "
                             "(a fixed-size Prefill pool).")
    args = parser.parse_args()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
    num_reqs = args.num_reqs or sum(1 for line in open(args.dataset) if line.strip())
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for arm in arms:
        flags = list(ARMS[arm])
        cluster_config = args.cluster_config
        if "@static" in flags:
            if not args.static_config:
                raise SystemExit(f"arm {arm} needs --static-config")
            flags.remove("@static")
            cluster_config = args.static_config
        if "@fixed" in flags:
            if not args.fixed_config:
                raise SystemExit(f"arm {arm} needs --fixed-config")
            flags.remove("@fixed")
            cluster_config = args.fixed_config
        if "@overlay" in flags:
            flags.remove("@overlay")
            overlay = OVERLAYS.get(arm)
            if overlay:
                base = json.loads(pathlib.Path(cluster_config).read_text(
                    encoding="utf-8"))
                cluster_config = str(out_root / f"{arm}-cluster.json")
                pathlib.Path(cluster_config).write_text(
                    json.dumps(_merge(copy.deepcopy(base), overlay),
                               ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
        out_csv = out_root / f"{arm}.csv"
        cmd = [
            sys.executable, "-m", "serving",
            "--cluster-config", cluster_config,
            "--dataset", args.dataset,
            "--num-reqs", str(num_reqs),
            "--dtype", "bfloat16", "--block-size", "16",
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--log-level", "WARNING",
            "--casr-control-interval-ms", str(args.control_interval_ms),
            "--output", str(out_csv),
            "--inputs-root", str(out_root / f"{arm}-inputs"),
            *flags,
        ]
        if args.client_concurrency:
            cmd += ["--client-concurrency", str(args.client_concurrency)]
        print(f"== {arm} ==", flush=True)
        with (out_root / f"{arm}.log").open("w") as log:
            subprocess.run(cmd, cwd=REPO, check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        results[arm] = summarise(out_csv)

    print()
    print(f"{'arm':<12}{'mean':>9}{'p50':>8}{'p95':>9}{'TTFT p50':>10}"
          f"{'TPOT p50':>10}{'SLO%':>7}{'span s':>8}   exchange / prefills")
    for arm in arms:
        row = results[arm]
        attainment = "n/a" if row["attainment"] is None else f"{row['attainment']:.1f}"
        print(f"{arm:<12}{row['mean']:>9.0f}{row['p50']:>8.0f}{row['p95']:>9.0f}"
              f"{row['ttft_p50']:>10.0f}{row['tpot_p50']:>10.1f}{attainment:>7}"
              f"{row['span_s']:>8.1f}   {row['exchange']} / {row['prefills']}")
    (out_root / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_root/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
