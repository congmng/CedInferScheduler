#!/usr/bin/env python3
"""Measure the per-card Prefill overhead the layer profile does not time.

``step_cost_ns`` and the trace generator agree to <0.1% on a step's layer
charge (verify with ``--charge-only``), but a *serving pair* executes a prompt
more slowly than that charge: each chunk-step also pays the P/D handoff, the
Decode's own first step and scheduler/host bookkeeping.  The CASR plan prices a
PreFill, so if it prices the bare charge it overstates the card -- measured
2026-09-23, one 5090 was priced at 17.2 reference req/s against an executed
13.6, and the 16 rps P-15B peak queued on it for 8.7 s.

This script produces the table used by
``hw_service.PREFILL_PIPELINE_OVERHEAD_MS_BY_HARDWARE``: for each card it
saturates a 1P1D instance with ``--num-reqs`` identical ``--tokens``-token
prompts released at t=0 and reports

    charge_ms   -- the layer charge for that prompt (``prefill_charge_ns``)
    period_ms   -- the steady-state TTFT period, i.e. 1/throughput
    overhead_ms -- the per-step difference

    python3 tests/probe_prefill_pipeline.py --tokens 1024 --num-reqs 64
    python3 tests/probe_prefill_pipeline.py --charge-only

The probe rewrites the cluster config to a single card and drops the intra-node
link keys (a one-node topology cannot express them), so it measures the engine,
not the fabric.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE_CONFIG = "configs/cluster/casr_p15b_three_domain_plan3.json"
#: node index in SOURCE_CONFIG -> the card it holds
NODES = {0: "RTX3090", 1: "RTX4090", 2: "RTX5090"}
MODEL = "casr/P15B"


def single_card_config(destination: pathlib.Path, node_index: int) -> pathlib.Path:
    base = json.loads((REPO / SOURCE_CONFIG).read_text(encoding="utf-8"))
    config = {key: value for key, value in base.items()
              if key not in ("nodes", "intra_node_link_bw",
                             "intra_node_link_latency", "_intra_link_comment")}
    config["nodes"] = [base["nodes"][node_index]]
    config["num_nodes"] = 1
    config["casr"] = dict(base["casr"])
    config["casr"]["lifecycle"] = {"min_active_prefill": 1,
                                   "max_active_prefill": 1,
                                   "scale_on_demand": False, "warmup_ms": 250}
    destination.write_text(json.dumps(config, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    return destination


def write_trace(path: pathlib.Path, tokens: int, num_reqs: int) -> None:
    """``num_reqs`` distinct-prefix prompts, all due at t=0 (so the pool is
    saturated from the first step); shared prefixes would collapse under the
    prefix cache and measure nothing."""
    with path.open("w", encoding="utf-8") as handle:
        for index in range(num_reqs):
            handle.write(json.dumps({
                "input_toks": tokens, "output_toks": 1, "arrival_time_ns": 0,
                "input_tok_ids": [1000 * index + token for token in range(tokens)],
                "output_tok_ids": [7], "category": "probe",
                "source_id": f"probe-{index}",
            }) + "\n")


def charge_ms(hardware: str, tokens: int, chunk: int) -> float:
    """Layer charge from the bundle, the same number ``step_cost_ns`` quotes."""
    import os
    cwd = pathlib.Path.cwd()
    os.chdir(REPO / "astra-sim")
    sys.path.insert(0, str(REPO))
    try:
        from serving.core.hw_service import (prefill_charge_ns,
                                             prefill_period_ms,
                                             prefill_pipeline_overhead_ms)
        return (prefill_charge_ns(hardware, MODEL, tp=1, tokens=tokens,
                                  chunk=chunk) / 1e6,
                prefill_period_ms(hardware, MODEL, tp=1, tokens=tokens,
                                  reference=tokens, chunk=chunk),
                prefill_pipeline_overhead_ms(hardware))
    finally:
        os.chdir(cwd)


def run_probe(hardware: str, node_index: int, tokens: int, num_reqs: int,
              workdir: pathlib.Path) -> float:
    config = single_card_config(workdir / f"probe-{hardware}.json", node_index)
    trace = workdir / f"probe-{hardware}-{tokens}.jsonl"
    write_trace(trace, tokens, num_reqs)
    out_csv = workdir / f"probe-{hardware}.csv"
    subprocess.run([
        sys.executable, "-m", "serving",
        "--cluster-config", str(config), "--dataset", str(trace),
        "--num-reqs", str(num_reqs), "--dtype", "bfloat16",
        "--block-size", "16", "--max-num-seqs", str(num_reqs),
        "--max-num-batched-tokens", str(min(1024, tokens)),
        "--log-level", "WARNING", "--no-enable-casr",
        "--output", str(out_csv),
        "--inputs-root", str(workdir / f"probe-{hardware}-inputs"),
    ], cwd=REPO, check=True, stdout=subprocess.DEVNULL,
       stderr=subprocess.STDOUT)
    ttft = sorted(float(row["TTFT"]) / 1e6
                  for row in csv.DictReader(out_csv.open()))
    # Steady-state period: the last request's TTFT minus the first's, over the
    # intervals between them.  The first request pays the pipeline, not a queue.
    return (ttft[-1] - ttft[0]) / max(1, len(ttft) - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--num-reqs", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=1024,
                        help="the run's --max-num-batched-tokens")
    parser.add_argument("--charge-only", action="store_true",
                        help="print the model's charge/period without running")
    args = parser.parse_args()

    print(f"{'card':<10}{'charge_ms':>11}{'period_ms':>11}{'overhead_ms':>13}"
          f"{'req/s':>9}")
    rows = {}
    with tempfile.TemporaryDirectory(prefix="prefill-probe-") as tmp:
        workdir = pathlib.Path(tmp)
        for node_index, hardware in NODES.items():
            charge, period, overhead = charge_ms(hardware, args.tokens,
                                                 args.chunk)
            if args.charge_only:
                print(f"{hardware:<10}{charge:>11.2f}{period:>11.2f}"
                      f"{overhead:>13.2f}{1000.0 / period:>9.2f}"
                      f"   (model only)")
                continue
            measured = run_probe(hardware, node_index, args.tokens,
                                 args.num_reqs, workdir)
            rows[hardware] = measured
            print(f"{hardware:<10}{charge:>11.2f}{measured:>11.2f}"
                  f"{measured - charge:>13.2f}{1000.0 / measured:>9.2f}")
    if rows:
        print()
        print("Paste into hw_service.PREFILL_PIPELINE_OVERHEAD_MS_BY_HARDWARE:")
        for hardware, measured in rows.items():
            charge, _, _ = charge_ms(hardware, args.tokens, args.chunk)
            print(f'    "{hardware}": {measured - charge:.1f},')
        spread = [round(measured - charge_ms(hardware, args.tokens, args.chunk)[0], 1)
                  for hardware, measured in rows.items()]
        print(f"  # spread {min(spread):.1f}-{max(spread):.1f} ms "
              f"(median {statistics.median(spread):.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
