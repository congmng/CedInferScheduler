#!/usr/bin/env python3
"""Sweep the Decode tail-pricing weight on one trace and print the trade-off.

``c_ijk`` prices a Decode's queue as a *fraction* (``queue_weight``), which is
blind to how long that queue takes to drain; ``tail_weight`` converts it into
seconds using the instance's own step time (see ``FlowSolverConfig``).  This
runs the CASR LP arm at several weights on the same trace and reports mean,
p50/p95 and SLO attainment, so the knob is tuned against measurements rather
than taste.

    python3 tests/sweep_tail_weight.py \
        --cluster-config configs/cluster/casr_p15b_three_domain_slo.json \
        --dataset workloads/casr-peak30-1250tok-slo-tiered.jsonl \
        --weights 0,0.5,1,2,5 --out-root /tmp/tail-sweep

The comparator runs ``tests/compare_casr.py``'s summary directly, so the
numbers are the same ones the comparison tables print.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def summarise(path: pathlib.Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    def scaled(name):
        return sorted(float(row[name]) / 1e6 for row in rows)
    lat, ttft, tpot = scaled("latency"), scaled("TTFT"), scaled("TPOT")
    judged = [row for row in rows if str(row.get("slo_ok", "")).strip() != ""]
    met = sum(1 for row in judged if str(row["slo_ok"]).strip().lower() == "true")
    decodes: dict[str, int] = {}
    for row in rows:
        key = row.get("decode_instance_id", "")
        decodes[key] = decodes.get(key, 0) + 1
    return {
        "n": len(rows),
        "mean": statistics.fmean(lat),
        "p50": lat[len(lat) // 2],
        "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "ttft_p50": ttft[len(ttft) // 2],
        "tpot_p50": tpot[len(tpot) // 2],
        "attainment": (100.0 * met / len(judged)) if judged else None,
        "decodes": dict(sorted(decodes.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", default="0,0.5,1,2,5")
    parser.add_argument("--out-root", default="/tmp/tail-sweep")
    parser.add_argument("--arm", default="lp", choices=["lp", "greedy"])
    parser.add_argument("--num-reqs", type=int, default=0)
    parser.add_argument("--control-interval-ms", type=int, default=100)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    args = parser.parse_args()

    base = json.loads(pathlib.Path(args.cluster_config).read_text(encoding="utf-8"))
    num_reqs = args.num_reqs or sum(1 for line in open(args.dataset) if line.strip())
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw_weight in args.weights.split(","):
        weight = float(raw_weight)
        tag = f"tail{weight:g}"
        config_path = out_root / f"{tag}.json"
        config = json.loads(json.dumps(base))
        config["casr"]["tail_weight"] = weight
        config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
        out_csv = out_root / f"{tag}.csv"
        print(f"== tail_weight={weight:g} ==", flush=True)
        cmd = [
            sys.executable, "-m", "serving",
            "--cluster-config", str(config_path),
            "--dataset", args.dataset,
            "--num-reqs", str(num_reqs),
            "--dtype", "bfloat16", "--block-size", "16",
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--log-level", "WARNING",
            "--enable-casr", "--casr-solver", args.arm,
            "--casr-control-interval-ms", str(args.control_interval_ms),
            "--output", str(out_csv),
            "--casr-state-output", str(out_root / f"{tag}.jsonl"),
            "--inputs-root", str(out_root / f"{tag}-inputs"),
        ]
        with (out_root / f"{tag}.log").open("w") as log:
            subprocess.run(cmd, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
        summary = summarise(out_csv)
        summary["tail_weight"] = weight
        rows.append(summary)

    print()
    header = (f"{'tail_w':>7}{'mean ms':>10}{'p50':>9}{'p95':>10}"
              f"{'TTFT p50':>10}{'TPOT p50':>10}{'SLO %':>8}   decodes")
    print(header)
    for row in rows:
        attainment = "n/a" if row["attainment"] is None else f"{row['attainment']:.1f}"
        print(f"{row['tail_weight']:>7g}{row['mean']:>10.0f}{row['p50']:>9.0f}"
              f"{row['p95']:>10.0f}{row['ttft_p50']:>10.0f}{row['tpot_p50']:>10.1f}"
              f"{attainment:>8}   {row['decodes']}")
    (out_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_root/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
