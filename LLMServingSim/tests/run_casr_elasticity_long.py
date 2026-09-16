#!/usr/bin/env python3
"""Acceptance run: structural elasticity must *pay* on a sustained overload.

The simulator could already scale workers out, but three separate plumbing
defects meant the extra workers never received traffic, so every elasticity
comparison came out identical to the static pool (see
``docs/模拟器与真机一致性核查.md`` 附九).  This driver is the criterion: same
configuration, same routing policy (CASR-LP), same trace -- only the pool
policy differs -- and the elastic arm has to beat the static one.

The trace's high phase must outlast the measured container boot (45 s) or the
new worker cannot serve anything before the burst ends; 240 s is used here,
which is also what makes the comparison meaningful rather than a
millisecond-level tie.

Usage:
    python3 tests/run_casr_elasticity_long.py                 # run + assert
    python3 tests/run_casr_elasticity_long.py --min-gain 0.05 # looser gate
    python3 tests/run_casr_elasticity_long.py --workdir /tmp/casr-elastic-e2e
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs" / "cluster" / "casr_real_qwen3_8b_generated_kvheavy.json"
POOL_TRACE = REPO / "workloads" / "cnndm-long-pool-qwen3-8b.jsonl"
TRACE = REPO / "workloads" / "cnndm-long-elastic-240s-qwen3-8b.jsonl"
RATES = "0.5,1.8,0.5"
DURATIONS = "20,240,20"


def ensure_trace() -> pathlib.Path:
    """Regenerate the phased trace when it is missing (it is checked in)."""
    if TRACE.exists():
        return TRACE
    subprocess.run([sys.executable, str(REPO / "tests" / "make_phased_trace.py"),
                    "--input", str(POOL_TRACE), "--rates", RATES,
                    "--durations", DURATIONS, "--names", "low_a,high,low_c",
                    "--output", str(TRACE)], cwd=REPO, check=True)
    return TRACE


def write_arm_config(tag: str, min_active: int, max_active: int) -> pathlib.Path:
    """Cluster config for one arm.

    The simulator resolves cluster configs relative to ``astra-sim/``, so the
    file has to live under ``configs/cluster/``; it is removed afterwards.
    """
    config = json.loads(_base_config().read_text(encoding="utf-8"))
    config["casr"]["lifecycle"]["min_active_prefill"] = min_active
    config["casr"]["lifecycle"]["max_active_prefill"] = max_active
    path = REPO / "configs" / "cluster" / f"_tmp_elastic_{tag}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


_BASE_CONFIG_OVERRIDE = None


def _base_config() -> pathlib.Path:
    """The cluster the arms run on.

    Defaults to the 4-domain ``kvheavy`` config; ``--base-config`` lets an
    acceptance run pick another (the 3-domain small-cluster config, say, which
    is what the real comparisons use -- the deployment's A100 is shared with
    another tenant, so it is neither usable nor representative).
    """
    return pathlib.Path(_BASE_CONFIG_OVERRIDE or BASE_CONFIG)


def run_arm(tag: str, config: pathlib.Path, workdir: pathlib.Path) -> dict:
    out_csv = workdir / f"{tag}.csv"
    state = workdir / f"{tag}-state.jsonl"
    cmd = [sys.executable, "-m", "serving",
           "--cluster-config", str(config.relative_to(REPO)),
           "--dataset", str(TRACE.relative_to(REPO)),
           "--num-reqs", str(sum(1 for _ in TRACE.open(encoding="utf-8"))),
           "--dtype", "bfloat16", "--block-size", "16", "--log-level", "WARNING",
           "--enable-casr", "--casr-solver", "lp", "--casr-control-interval-ms", "100",
           "--casr-state-output", str(state), "--output", str(out_csv),
           "--inputs-root", str(workdir / f"{tag}-inputs")]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "SIM_ET_PAIRING_CHECK": "1"})
    if proc.returncode != 0:
        raise RuntimeError(f"arm {tag} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"arm {tag} produced no rows")
    lat = [float(row["latency"]) / 1e6 for row in rows]
    ttft = [float(row["TTFT"]) / 1e6 for row in rows]
    span = (max(float(row["end_time"]) for row in rows)
            - min(float(row["arrival"]) for row in rows)) / 1e9
    peak_active = 0
    for line in state.open(encoding="utf-8"):
        snapshot = json.loads(line)
        active = sum(1 for key, value in (snapshot.get("instances") or {}).items()
                     if value.get("admission_state") == "ACTIVE" and int(key) % 2 == 0)
        peak_active = max(peak_active, active)
    return {
        "requests": len(rows),
        "mean_ms": round(statistics.mean(lat), 1),
        "p50_ms": round(sorted(lat)[len(lat) // 2], 1),
        "p95_ms": round(sorted(lat)[int(0.95 * len(lat))], 1),
        "ttft_mean_ms": round(statistics.mean(ttft), 1),
        "drain_s": round(span, 1),
        "throughput_rps": round(len(rows) / span, 3),
        "peak_active_prefills": peak_active,
        "landing": dict(collections.Counter(row["prefill_instance_id"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="")
    parser.add_argument("--min-gain", type=float, default=0.10, dest="min_gain",
                        help="required mean-latency improvement of elastic over static")
    parser.add_argument("--base-config", default="",
                        help="cluster config to vary (default: the 4-domain kvheavy one)")
    args = parser.parse_args()

    global _BASE_CONFIG_OVERRIDE
    if args.base_config:
        _BASE_CONFIG_OVERRIDE = args.base_config

    ensure_trace()
    workdir = pathlib.Path(args.workdir) if args.workdir else pathlib.Path(
        tempfile.mkdtemp(prefix="casr-elastic-e2e."))
    workdir.mkdir(parents=True, exist_ok=True)
    configs = {}
    try:
        configs["static"] = write_arm_config("static", 1, 1)
        configs["elastic"] = write_arm_config("elastic", 1, 4)
        results = {tag: run_arm(tag, path, workdir) for tag, path in configs.items()}
    finally:
        for path in configs.values():
            path.unlink(missing_ok=True)

    static, elastic = results["static"], results["elastic"]
    gain = (static["mean_ms"] - elastic["mean_ms"]) / static["mean_ms"]
    print(f"{'arm':<9}{'mean':>9}{'p50':>9}{'p95':>9}{'TTFT':>9}{'drain':>9}"
          f"{'req/s':>8}{'peakP':>7}  landing")
    for label, item in (("static", static), ("elastic", elastic)):
        print(f"{label:<9}{item['mean_ms']:>9.0f}{item['p50_ms']:>9.0f}"
              f"{item['p95_ms']:>9.0f}{item['ttft_mean_ms']:>9.0f}"
              f"{item['drain_s']:>9.1f}{item['throughput_rps']:>8.2f}"
              f"{item['peak_active_prefills']:>7}  {item['landing']}")
    print(f"\nelasticity gain (mean latency): {gain * 100:+.1f}% "
          f"(gate: >= {args.min_gain * 100:.0f}%)")
    print(f"artifacts: {workdir}")

    failures = []
    if gain < args.min_gain:
        failures.append(f"elastic only gained {gain * 100:.1f}% on mean latency")
    if elastic["peak_active_prefills"] < 2:
        failures.append(f"elastic arm never ran more than {elastic['peak_active_prefills']} Prefills")
    # The ceiling has to bind end to end, not just inside the evaluator: the
    # first version of this driver declared ``max_active_prefill=1`` for the
    # static arm and still measured three active Prefills, which silently
    # turned the "elasticity gain" into a three-worker-vs-two comparison.
    if static["peak_active_prefills"] > 1:
        failures.append("the max_active_prefill=1 arm ran more than one Prefill "
                        f"(peak {static['peak_active_prefills']})")
    if static["requests"] != elastic["requests"]:
        failures.append("the two arms completed different numbers of requests")
    if failures:
        print("ELASTICITY ACCEPTANCE FAILED: " + "; ".join(failures))
        return 1
    print("ELASTICITY ACCEPTANCE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
