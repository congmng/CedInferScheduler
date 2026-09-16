#!/usr/bin/env python3
"""Build a large heterogeneous simulated cluster from the profiler bundles.

The small-cluster comparisons are three domains and one workload shape; the
algorithm's job -- deciding *where* a request's Prefill and Decode run when the
domains differ in accelerator and in link -- needs a bigger, more varied board.
This generator takes one hardware per domain and emits a config whose CASR
control block is derived from the profiler bundles the simulator executes
with (``serving/core/hw_service.py``), so the plan is priced against the same
engine the timeline runs:

    python3 tests/make_hetero_cluster.py \
        --domains 5090,5090,4090,3090,3090,a100 \
        --out configs/cluster/hetero6_generated.json

Every domain is one node with a Prefill and a Decode instance (``--min-active``
of them start active, the rest are the structural-elasticity spares), the
same-host P/D handoff is charged at the measured 0.257 GB/s, and cross-domain
hops at the deployment's standard 0.11 GB/s.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# hardware -> (memory GiB, memory GB/s) as the simulator's memory model wants it
HARDWARE = {
    "5090": ("RTX5090", 32, 1792),
    "4090": ("RTX4090", 24, 1008),
    "3090": ("RTX3090", 24, 936),
    "a100": ("A100", 80, 2039),
}
# Measured same-host P/D handoff: 585 ms per 1000 prompt tokens = 252 MB/s.
SAME_HOST_GBPS = 0.257
SAME_HOST_LATENCY_NS = 1000
# Deployment's standard cross-domain link (0.88 Gbps) and slowest RTT.
CROSS_GBPS = 0.11
CROSS_LATENCY_NS = 48_000_000
# Producer-side KV push ceiling measured on the deployment (239-314 MB/s).
KV_EGRESS_GBPS = 0.26
PD_BUFFER_BYTES = 34359738368
CONTAINER_BOOT_MS = 45000
MODEL = "Qwen/Qwen3-8B"


def profiled_costs(hardware, tp=1, prefill_reference=1024, decode_reference=16):
    """(decode_service_ms, prefill_service_ms, decode_rps, prefill_rps).

    ``*_service_ms`` is the per-*request* service at the reference length, which
    is the unit the plan's cost function uses; the capacity is the reciprocal,
    in reference-length requests per second.
    """
    import os

    from serving.core.hw_service import step_cost_ns

    # The profiler root is resolved relative to the simulator's working
    # directory (``astra-sim/``); this generator runs from the repo root.
    cwd = os.getcwd()
    os.chdir(REPO / "astra-sim")
    try:
        decode_ms = step_cost_ns(hardware, MODEL, tp=tp, tokens=1) / 1e6
        prefill_ms = step_cost_ns(hardware, MODEL, tp=tp,
                                  tokens=prefill_reference) / 1e6
    finally:
        os.chdir(cwd)
    return (decode_reference * decode_ms, prefill_ms,
            1000.0 / (decode_reference * decode_ms), 1000.0 / prefill_ms)


def build(domains, min_active=2, max_active=None, structural=False,
          prefix_caching=True):
    nodes = []
    prefill_ids = {index: index * 2 for index in range(len(domains))}
    decode_ids = {index: index * 2 + 1 for index in range(len(domains))}

    for index, name in enumerate(domains):
        hardware, mem_gb, mem_bw = HARDWARE[name]
        nodes.append({
            "num_instances": 2,
            "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
            "instances": [
                {"instance_id": prefill_ids[index], "model_name": MODEL,
                 "hardware": hardware,
                 "npu_mem": {"mem_size": mem_gb, "mem_bw": mem_bw,
                             "mem_latency": 0, "mem_util": 0.9},
                 "pd_type": "prefill", "tp_size": 1,
                 "enable_prefix_caching": prefix_caching},
                {"instance_id": decode_ids[index], "model_name": MODEL,
                 "hardware": hardware,
                 "npu_mem": {"mem_size": mem_gb, "mem_bw": mem_bw,
                             "mem_latency": 0, "mem_util": 0.9},
                 "pd_type": "decode", "tp_size": 1,
                 "enable_prefix_caching": prefix_caching},
            ],
        })

    prefill_capacity, decode_capacity = {}, {}
    prefill_service, decode_service = {}, {}
    router_capacity, prefill_tokens_per_s = {}, {}
    for index, name in enumerate(domains):
        hardware = HARDWARE[name][0]
        decode_ms, prefill_ms, decode_rps, prefill_rps = profiled_costs(hardware)
        prefill_capacity[str(prefill_ids[index])] = round(prefill_rps, 3)
        decode_capacity[str(decode_ids[index])] = round(decode_rps, 3)
        prefill_service[str(prefill_ids[index])] = round(prefill_ms, 3)
        decode_service[str(decode_ids[index])] = round(decode_ms, 3)
        # The real router's ``capacity`` field is a per-instance requests/s
        # figure measured with its own concurrency, i.e. the engine rate times
        # max_num_seqs; keep that convention so the ``load`` baseline spreads
        # the way it would on the deployment.
        router_capacity[str(prefill_ids[index])] = round(prefill_rps * 16, 1)
        router_capacity[str(decode_ids[index])] = round(decode_rps * 16, 1)
        prefill_tokens_per_s[str(prefill_ids[index])] = round(prefill_rps * 1024, 1)

    shared_links = []
    for index, name in enumerate(domains):
        hardware = HARDWARE[name][0]
        shared_links.append({
            "id": f"kvlink-d{index}-{name}",
            "capacity_bytes_per_s": KV_EGRESS_GBPS * 1e9,
            "pairs": [[prefill_ids[index], decode_ids[other]]
                      for other in prefill_ids],
        })

    return {
        "num_nodes": len(nodes),
        "link_bw": CROSS_GBPS,
        "link_latency": CROSS_LATENCY_NS,
        "intra_node_link_bw": SAME_HOST_GBPS,
        "intra_node_link_latency": SAME_HOST_LATENCY_NS,
        "kv_egress_gbps": KV_EGRESS_GBPS,
        "pd_buffer_bytes": PD_BUFFER_BYTES,
        "_generated": (
            "Derived from the profiler bundles by tests/make_hetero_cluster.py: "
            "the per-instance service times and capacities are the profiled "
            "engine rates, so the plan is priced against what the timeline "
            "executes.  Link numbers are the deployment's measured ones."),
        "_link_comment": (
            f"Same-host P/D handoff {SAME_HOST_GBPS} GB/s (measured 585 ms per "
            f"1000 prompt tokens), cross-domain {CROSS_GBPS} GB/s at "
            f"{CROSS_LATENCY_NS/1e6:.0f} ms RTT, producer push ceiling "
            f"{KV_EGRESS_GBPS} GB/s (measured 239-314 MB/s)."),
        "nodes": nodes,
        "casr": {
            "solver": "lp",
            "control_interval_s": 1.0,
            "lifecycle": {
                "min_active_prefill": int(min_active),
                "max_active_prefill": int(max_active or len(nodes)),
                "warmup_ms": CONTAINER_BOOT_MS,
                "scale_on_demand": False,
                "prefill_capacity": prefill_capacity,
                "prefill_tokens_per_s": prefill_tokens_per_s,
                "capacity_reference_tokens": 1024,
                "kv_egress_gbps": KV_EGRESS_GBPS,
            },
            "structural": {
                "evaluation_window_ms": 60000.0,
                "gain_threshold_abs": 0.001,
                "gain_threshold_rel": 0.01,
                "dwell_time_ms": 3000.0,
                "startup_cost": 0.0,
                "warm_cost": 0.0,
                "enabled": bool(structural),
                "startup_s": CONTAINER_BOOT_MS / 1000.0,
                "idle_cost_fraction": 0.1,
                "enable_warm_counterfactual": False,
            },
            "resources": {
                "startup_ms": CONTAINER_BOOT_MS,
                "reclaim_ms": 50,
                "nodes": {str(index): {"gpu_count": 2,
                                       "gpu_mem_gb": [HARDWARE[name][1]] * 2}
                          for index, name in enumerate(domains)},
            },
            "prefill_capacity": prefill_capacity,
            "decode_capacity": decode_capacity,
            "prefill_service_ms": prefill_service,
            "decode_service_ms": decode_service,
            "router_capacity": router_capacity,
            "prefill_tokens_per_s": prefill_tokens_per_s,
            "capacity_reference_tokens": 1024,
            "decode_reference_tokens": 16,
            "overflow_penalty": 10,
            "utilization_weight": 1.0,
            "utilization_segments": 8,
            "compute_weight": 1.0,
            "network_weight": 1.0,
            "single_home_below_rps": 1.0,
            "kv_bytes_per_token": 147456,
            "shared_links": shared_links,
            "local_prefill": "never",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="5090,5090,4090,3090,3090,a100",
                        help="comma separated hardware per domain (" +
                             ", ".join(sorted(HARDWARE)) + ")")
    parser.add_argument("--min-active", type=int, default=2,
                        help="Prefills that start ACTIVE (the rest are spares)")
    parser.add_argument("--max-active", type=int, default=0)
    parser.add_argument("--structural", action="store_true",
                        help="enable structural scale-out (casr_full arm)")
    parser.add_argument("--out", default="configs/cluster/hetero6_generated.json")
    args = parser.parse_args()

    domains = [part.strip() for part in args.domains.split(",") if part.strip()]
    unknown = [name for name in domains if name not in HARDWARE]
    if unknown:
        raise SystemExit(f"unknown domain hardware: {unknown}; "
                         f"known: {sorted(HARDWARE)}")
    config = build(domains, min_active=args.min_active,
                   max_active=args.max_active or len(domains),
                   structural=args.structural)
    path = pathlib.Path(args.out)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"wrote {path} ({len(domains)} domains, "
          f"{len(config['nodes']) * 2} instances, structural="
          f"{config['casr']['structural']['enabled']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
