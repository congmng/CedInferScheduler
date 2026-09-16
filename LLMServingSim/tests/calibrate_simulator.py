#!/usr/bin/env python3
"""Measure how far the simulator's timing is from the profiles and from reality.

The simulator's *routing* is shared with the real router, but its timing model
is not yet aligned with measurements: on 2026-09-12 a single RTX5090 P/D pair
simulated a 14 req/s Dolly replay at 1232 ms TTFT while the real pair did
243 ms.  Before that gap is closed, no simulated performance claim is
trustworthy, and closing it needs a reproducible anchor set -- which is what
this script produces.

It reports, per anchor:

* ``profile_ms``    -- the per-layer sum of the measured ``dense.csv`` /
  ``attention.csv`` for that hardware and prompt length.  This is what the
  simulator *should* cost if it simply added the profiled kernels.
* ``sim_prefill_ms`` -- what the simulator actually charged (TTFT minus its own
  reported queuing delay), i.e. the compute it believes a prefill takes.
* ``sim_tpot_ms`` / ``TTFT`` / achieved throughput.
* ``real_*``        -- optional measured anchors loaded from a JSON file
  (``router_config.json`` service times, ``tests/calibrate_pd_throughput.py``
  saturation), so the ratio to the real system is visible in the same row.

Usage::

    python3 tests/calibrate_simulator.py --hardware RTX5090 --tokens 256 \
        --num-reqs 1 8 32 --out /tmp/sim-calib.json
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
LAYERS = 36
ONCE = ("embedding", "final_layernorm")


def profile_prefill_ms(hardware: str, tokens: int, model="Qwen/Qwen3-8B",
                       variant="bf16") -> float:
    """Per-layer sum of the measured profile for one prefill chunk.

    Mirrors what ``trace_generator`` looks up: every dense entry except the
    once-per-model ones is paid per layer, and the attention term is looked up
    at ``(prefill_chunk=tokens, kv_prefill=0)``.
    """
    root = REPO / "profiler" / "perf" / hardware / model / variant / "tp1"
    dense = list(csv.DictReader((root / "dense.csv").open()))
    at = {row["layer"]: float(row["time_us"])
          for row in dense if int(row["tokens"]) == tokens}
    once = sum(at.get(name, 0.0) for name in ONCE)
    per_layer = sum(v for k, v in at.items() if k not in ONCE)
    attention = [float(r["time_us"])
                 for r in csv.DictReader((root / "attention.csv").open())
                 if int(r["prefill_chunk"]) == tokens
                 and int(r["kv_prefill"]) == 0]
    return (once + per_layer * LAYERS + (attention[0] if attention else 0.0)
            * LAYERS) / 1000.0


def synth_trace(path, tokens, output_tokens, num_reqs, spacing_s=0.0):
    """``num_reqs`` identical requests, so the batch is controlled exactly."""
    ids = list(range(1, tokens + 1))
    with open(path, "w", encoding="utf-8") as out:
        for index in range(num_reqs):
            out.write(json.dumps({
                "input_toks": tokens, "output_toks": output_tokens,
                "arrival_time_ns": int(index * spacing_s * 1e9),
                "input_tok_ids": ids, "output_tok_ids": list(range(output_tokens)),
            }) + "\n")


def run_sim(cluster, dataset, num_reqs, workdir, extra=()):
    out_csv = pathlib.Path(workdir) / "out.csv"
    cmd = [sys.executable, "-m", "serving",
           "--cluster-config", str(cluster), "--dataset", str(dataset),
           "--num-reqs", str(num_reqs), "--dtype", "bfloat16",
           "--block-size", "16", "--max-num-seqs", "64",
           "--max-num-batched-tokens", "2048", "--log-level", "WARNING",
           "--request-routing-policy", "RR", "--no-enable-casr",
           "--output", str(out_csv), *extra]
    # NOTE: this used to export ``PYTHONPATH=/tmp/chakra-install.*/build/lib``,
    # a directory from an abandoned install that no longer exists.  Shadowing
    # the installed chakra from the tree is *not* an option either: the tree's
    # generated protobufs are stale ("Descriptors cannot be created directly"),
    # so the supported sync is copying the changed Python sources into
    # site-packages -- guarded by tests/test_chakra_runtime_sync.py.
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**__import__("os").environ,
                               "SIM_ET_PAIRING_CHECK": "1"})
    if proc.returncode != 0:
        raise RuntimeError(f"simulator failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    rows = list(csv.DictReader(out_csv.open()))
    # Liveness gate: a deadlocked run reports zero completions and an unbounded
    # simulated clock, which otherwise reads as "very slow" rather than broken.
    if not rows or not any(float(r["end_time"] or 0) > 0 for r in rows):
        raise RuntimeError(
            f"simulator produced no completed request for {num_reqs} arrivals "
            f"(deadlock shape); graph provenance:\n"
            + "\n".join(line for line in proc.stdout.splitlines()
                        if "chakra converter" in line))
    lat = sorted(float(r["latency"]) / 1e6 for r in rows)
    ttft = sorted(float(r["TTFT"]) / 1e6 for r in rows)
    que = [float(r["queuing_delay"]) / 1e6 for r in rows]
    tpot = [float(r["TPOT"]) / 1e6 for r in rows]
    span = (max(float(r["end_time"]) for r in rows)
            - min(float(r["arrival"]) for r in rows)) / 1e9
    return {
        "requests": len(rows),
        "completed": sum(1 for r in rows if float(r["end_time"] or 0) > 0),
        "ttft_p50_ms": round(statistics.median(ttft), 1),
        "queuing_p50_ms": round(statistics.median(que), 1),
        "sim_prefill_ms": round(statistics.median(t - q for t, q in zip(ttft, que)), 1),
        "tpot_p50_ms": round(statistics.median(tpot), 1),
        "latency_p50_ms": round(lat[len(lat) // 2], 1),
        "achieved_rps": round(len(rows) / span, 2) if span else None,
        "chakra_converter": next((line.split(":", 1)[1].strip()
                                  for line in proc.stdout.splitlines()
                                  if "chakra converter" in line), None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--num-reqs", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--spacing-s", type=float, default=0.0,
                        help="arrival spacing; 0 means all in one batch")
    parser.add_argument("--real-anchors", default="",
                        help="JSON: {hardware: {ttft_p50_ms, tpot_p50_ms, capacity_rps}}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    predicted = profile_prefill_ms(args.hardware, args.tokens)
    real = {}
    if args.real_anchors:
        real = (json.loads(pathlib.Path(args.real_anchors).read_text())
                .get(args.hardware, {}))

    report = {"hardware": args.hardware, "tokens": args.tokens,
              "profile_prefill_ms": round(predicted, 1), "anchors": []}
    print(f"{args.hardware} {args.tokens} tok: profile predicts "
          f"{predicted:.1f} ms prefill ({predicted / args.tokens:.3f} ms/token)")
    print(f"{'N':>4} {'TTFT':>8} {'queue':>7} {'sim_prefill':>12} "
          f"{'x profile':>10} {'TPOT':>7} {'rps':>7}")
    with tempfile.TemporaryDirectory() as tmp:
        for num in args.num_reqs:
            dataset = pathlib.Path(tmp) / f"trace-{num}.jsonl"
            # The simulator resolves datasets relative to the repo.
            local = REPO / "workloads" / f"calib-sim-{num}.jsonl"
            synth_trace(local, args.tokens, args.output_tokens, num, args.spacing_s)
            row = run_sim(args.cluster_config, local.relative_to(REPO), num, tmp)
            local.unlink()
            ratio = (row["sim_prefill_ms"] / predicted) if predicted else None
            row["ratio_vs_profile"] = round(ratio, 2) if ratio else None
            report["anchors"].append({"num_reqs": num, **row})
            print(f"{num:>4} {row['ttft_p50_ms']:>8.1f} {row['queuing_p50_ms']:>7.1f} "
                  f"{row['sim_prefill_ms']:>12.1f} "
                  f"{(f'{ratio:.2f}x' if ratio else '-'):>10} "
                  f"{row['tpot_p50_ms']:>7.1f} {row['achieved_rps'] or 0:>7.2f}")

    if real:
        best = min(report["anchors"], key=lambda a: abs(a["sim_prefill_ms"] - real.get("prefill_ms", 0)))
        report["real"] = real
        print(f"\nreal anchor: TTFT p50 {real.get('ttft_p50_ms')} ms, "
              f"TPOT p50 {real.get('tpot_p50_ms')} ms, capacity "
              f"{real.get('capacity_rps')} req/s")
        print(f"   simulator TPOT matches ({report['anchors'][0]['tpot_p50_ms']} ms); "
              f"TTFT is off by {report['anchors'][0]['ttft_p50_ms'] / real['ttft_p50_ms']:.1f}x"
              if real.get("ttft_p50_ms") else "")
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
