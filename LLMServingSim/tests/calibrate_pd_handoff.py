#!/usr/bin/env python3
"""Measure what the simulator charges for a P/D KV handoff, by domain.

The real router prices a (Prefill, Decode) pair as
``rtt + kv_bytes / bandwidth``, and treats a same-domain pair as free because
the KV never leaves the host.  The simulator only reproduces that if its
network model can tell a same-node handoff from a cross-node one, which is what
``intra_node_link_bw`` / ``intra_node_link_latency`` plus the
``[tp, slots_per_node, num_nodes]`` topology switch on.  See
``docs/模拟器与真机一致性核查.md`` 附四 and 附七.

This probe builds a *homogeneous* two-node cluster (the same hardware on both
nodes, so compute cost is identical) and replays one synthetic prompt length
through it.  With a single prompt length every request pushes the same number
of KV bytes, so the difference between the same-domain and cross-domain
requests isolates the handoff: it is the number the solver's cross-domain
``rtt_ms`` has to be set to.

Usage::

    python3 tests/calibrate_pd_handoff.py --hardware RTX5090 --tokens 512 \
        --num-reqs 24 --out /tmp/pd-handoff.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKLOAD_DIR = REPO / "workloads"
CONFIG_DIR = REPO / "configs" / "cluster"
CHAKRA_LIB = os.environ.get("CHAKRA_PYTHONPATH", "/tmp/chakra-install.uAi1HQ/build/lib")
LAYERS = 36


def synth_trace(path, tokens, output_tokens, num_reqs, spacing_s):
    ids = list(range(1, tokens + 1))
    with open(path, "w", encoding="utf-8") as out:
        for index in range(num_reqs):
            out.write(json.dumps({
                "input_toks": tokens, "output_toks": output_tokens,
                "arrival_time_ns": int(index * spacing_s * 1e9),
                "input_tok_ids": ids, "output_tok_ids": list(range(output_tokens)),
            }) + "\n")


def build_cluster(template_path, hardware, nodes, inter_bw, inter_latency_ns,
                  intra_bw, intra_latency_ns):
    """Build N identical nodes, each holding one Prefill and one Decode.

    Both nodes run the same hardware, so a same-domain and a cross-domain
    handoff differ in *nothing* but the link the KV crosses.
    """
    template = json.loads(pathlib.Path(template_path).read_text())
    proto = template["nodes"][0]["instances"]
    prefill = next(i for i in proto if i["pd_type"] == "prefill")
    decode = next(i for i in proto if i["pd_type"] == "decode")

    cluster = {k: v for k, v in template.items() if k not in ("nodes",)}
    cluster.update({
        "num_nodes": nodes,
        "link_bw": inter_bw,
        "link_latency": inter_latency_ns,
        "intra_node_link_bw": intra_bw,
        "intra_node_link_latency": intra_latency_ns,
        "nodes": [],
    })
    instance_id = 0
    for node in range(nodes):
        instances = []
        for role, base in (("prefill", prefill), ("decode", decode)):
            inst = json.loads(json.dumps(base))
            inst["instance_id"] = instance_id
            inst["pd_type"] = role
            inst["hardware"] = hardware
            instances.append(inst)
            instance_id += 1
        cluster["nodes"].append({
            "node_id": node,
            "num_instances": len(instances),
            "cpu_mem": json.loads(json.dumps(template["nodes"][0]["cpu_mem"])),
            "instances": instances,
        })
    return cluster


def run_sim(cluster_rel, dataset_rel, num_reqs, output_csv, policy):
    cmd = [sys.executable, "-m", "serving",
           "--cluster-config", str(cluster_rel), "--dataset", str(dataset_rel),
           "--num-reqs", str(num_reqs), "--dtype", "bfloat16",
           "--block-size", "16", "--max-num-seqs", "64",
           "--max-num-batched-tokens", "2048", "--log-level", "WARNING",
           "--request-routing-policy", policy, "--no-enable-casr",
           "--run-id", "pd_handoff", "--keep-inputs",
           "--output", str(output_csv)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": CHAKRA_LIB})
    if proc.returncode != 0:
        raise RuntimeError(f"simulator failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return list(csv.DictReader(open(output_csv)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="RTX5090")
    parser.add_argument("--template", default=str(CONFIG_DIR / "casr_1p1d_rtx5090_aligned.json"))
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--num-reqs", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--spacing-s", type=float, default=0.05)
    parser.add_argument("--policy", default="RAND",
                        help="RAND (the default, seeded) spreads the same/cross "
                             "handoffs evenly; LOAD rotates Prefill and Decode "
                             "in lockstep on a homogeneous probe and picks "
                             "same-domain only")
    parser.add_argument("--inter-bw", type=float, default=2.0,
                        help="inter-node link bandwidth in GB/s (as ASTRA-Sim reads it)")
    parser.add_argument("--inter-latency-ms", type=float, default=0.3)
    parser.add_argument("--intra-bw", type=float, default=50.0)
    parser.add_argument("--intra-latency-ms", type=float, default=0.001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cluster = build_cluster(args.template, args.hardware, args.nodes,
                            args.inter_bw, args.inter_latency_ms * 1e6,
                            args.intra_bw, args.intra_latency_ms * 1e6)
    config_rel = pathlib.Path("configs") / "cluster" / "_calib_pd_handoff.json"
    trace_rel = pathlib.Path("workloads") / "_calib_pd_handoff.jsonl"
    (REPO / config_rel).write_text(json.dumps(cluster, indent=1))
    synth_trace(REPO / trace_rel, args.tokens, args.output_tokens,
                args.num_reqs, args.spacing_s)

    node_of = {}
    for node in cluster["nodes"]:
        for inst in node["instances"]:
            node_of[inst["instance_id"]] = node["node_id"]

    report = {
        "hardware": args.hardware,
        "prompt_tokens": args.tokens,
        "nodes": args.nodes,
        "inter_node": {"bandwidth_gb_s": args.inter_bw,
                       "latency_ms": args.inter_latency_ms},
        "intra_node": {"bandwidth_gb_s": args.intra_bw,
                       "latency_ms": args.intra_latency_ms},
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = pathlib.Path(tmp) / "out.csv"
            rows = run_sim(config_rel, trace_rel, args.num_reqs, out_csv, args.policy)
    finally:
        (REPO / config_rel).unlink(missing_ok=True)
        (REPO / trace_rel).unlink(missing_ok=True)

    buckets = {"same": [], "cross": []}
    for row in rows:
        p = int(row["prefill_instance_id"])
        d = int(row["decode_instance_id"])
        kind = "same" if node_of.get(p) == node_of.get(d) else "cross"
        buckets[kind].append(row)
    if not buckets["same"] or not buckets["cross"]:
        raise RuntimeError(
            f"probe did not exercise both handoff kinds: "
            f"same={len(buckets['same'])} cross={len(buckets['cross'])}. "
            "Raise --num-reqs or --spacing-s."
        )

    def stat(kind, column):
        values = [float(r[column]) / 1e6 for r in buckets[kind]]
        return {"n": len(values), "mean_ms": round(statistics.mean(values), 2),
                "p50_ms": round(statistics.median(values), 2)}

    for kind in ("same", "cross"):
        report[kind] = {
            "ttft": stat(kind, "TTFT"),
            "latency": stat(kind, "latency"),
            "kv_bytes_mean": round(statistics.mean(
                float(r["pd_kv_bytes"] or 0) for r in buckets[kind]), 0),
        }
    delta_ttft = report["cross"]["ttft"]["mean_ms"] - report["same"]["ttft"]["mean_ms"]
    kv_bytes = max(report["cross"]["kv_bytes_mean"], 1.0)
    report["handoff_penalty"] = {
        "ttft_delta_mean_ms": round(delta_ttft, 2),
        "kv_bytes_mean": kv_bytes,
        "implied_bandwidth_gb_s": (round(kv_bytes / (delta_ttft * 1e6), 3)
                                   if delta_ttft > 0 else None),
        "per_layer_latency_ms": round(delta_ttft / LAYERS, 3),
    }

    print(f"{args.hardware}: {args.nodes} nodes x (1P+1D), {args.tokens}-token prompts, "
          f"{len(rows)} requests")
    print(f"  intra-node link {args.intra_bw} GB/s / {args.intra_latency_ms} ms, "
          f"inter-node {args.inter_bw} GB/s / {args.inter_latency_ms} ms")
    for kind in ("same", "cross"):
        row = report[kind]
        print(f"  {kind:5s} domain n={row['ttft']['n']:>3} "
              f"TTFT mean {row['ttft']['mean_ms']:>8.2f} ms   "
              f"e2e mean {row['latency']['mean_ms']:>8.2f} ms   "
              f"kv {row['kv_bytes_mean'] / 1e6:>6.1f} MB")
    print(f"  cross-domain penalty: +{delta_ttft:.2f} ms TTFT "
          f"({delta_ttft / LAYERS:.3f} ms per layer)")
    print("  -> set the solver's cross-domain pair cost to "
          f"rtt_ms={delta_ttft:.2f} with bandwidth_bytes_per_s={int(args.inter_bw * 1e9)}")
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
