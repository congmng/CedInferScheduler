#!/usr/bin/env python3
"""DistServe/Splitwise-style P:D ratio search, on this cluster's own numbers.

DistServe's resource decision is *how many* Prefill and Decode instances to run
for a given traffic mix and SLO, chosen to maximise goodput -- the request rate
that is actually served inside the TTFT/TPOT budgets.  Splitwise makes the same
choice at machine granularity.  Everything the comparison has run so far fixes
that ratio (3 P + 3 D on the three-domain cluster, 6 + 6 on the heterogeneous
one) and only varies *placement inside* the pools, so the ratio axis was
untested.

This script is that search:

1. price every instance from the profiler bundles (the same
   ``hw_service.resolve_runtime_capacities`` the controller uses), so a
   candidate ratio is scored with the capacities the run will really execute;
2. for every candidate (nP, nD), compute the pool's *effective* capacity for
   this trace's prompt/output mix -- a Prefill's reference-rate scales with
   ``1024 / prompt_tokens``, a Decode's with ``decode_reference_tokens /
   output_tokens`` -- and take ``goodput = min(offered, P_cap, D_cap)``;
3. write the cluster config for the best ratio (the machines are ranked by
   capacity, so the search also decides *which* instances stay up).

The estimate is deliberately the same accounting the simulator does, not a
queueing formula: the point of the script is to pick the ratio, and the run
that follows measures the goodput that ratio actually achieves.  Use
``--verify`` to run the top candidates through the simulator and print both.

**Known bias, inherited from the simulator**: the Decode capacity the profiler
resolves is a *serial* figure (``1000 / (reference_tokens x step)``), which
understates a batched engine by roughly ``max_num_seqs`` (see
``serving/core/hw_service.py``).  The search therefore learns "Decode is the
bottleneck" earlier than the executed timeline does.  Ratios are still ranked
against each other with one consistent model, and the ``--verify`` runs are the
authority on what a ratio achieves.

    python3 tests/pd_ratio_search.py \
        --cluster-config configs/cluster/hetero6_generated.json \
        --dataset workloads/matrix-16rps-slo1500.jsonl \
        --write-config /tmp/distserve-cluster.json --verify 3
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def read_trace(path):
    prompts, outputs, arrivals, slo_ttft, slo_tpot = [], [], [], [], []
    peak_rate = 0.0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(float(row.get("input_toks", 0)))
            # The trace file's ``output_toks`` is the *generated* count; the
            # simulator's request object carries the total (input + output).
            outputs.append(max(1.0, float(row.get("output_toks", 0))))
            arrivals.append(float(row.get("arrival_time_ns", 0)) / 1e9)
            if row.get("slo_ttft_ms"):
                slo_ttft.append(float(row["slo_ttft_ms"]))
            if row.get("slo_tpot_ms"):
                slo_tpot.append(float(row["slo_tpot_ms"]))
    meta_path = pathlib.Path(str(path) + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rates = meta.get("phase_rates_rps") or []
        peak_rate = max((float(value) for value in rates), default=0.0)
    span = max(arrivals) - min(arrivals) if arrivals else 1.0
    return {
        "n": len(prompts),
        "prompt_tokens": statistics.fmean(prompts) if prompts else 0.0,
        "output_tokens": statistics.fmean(outputs) if outputs else 1.0,
        # The *peak* phase is what a ratio has to be sized for; the trace's own
        # average over warm-up/peak/cool-down understates it by ~15%.
        "offered_rps": peak_rate or ((len(prompts) / span) if span > 0 else 0.0),
        "mean_rps": (len(prompts) / span) if span > 0 else 0.0,
        "slo_ttft_ms": statistics.median(slo_ttft) if slo_ttft else 0.0,
        "slo_tpot_ms": statistics.median(slo_tpot) if slo_tpot else 0.0,
    }


def resolve_capacities(cluster, config_path):
    """Per-instance capacities as the controller prices them."""
    casr = copy.deepcopy(cluster.get("casr") or {})
    instances = [inst for node in cluster["nodes"] for inst in node.get("instances", [])]
    cwd = pathlib.Path.cwd()
    astra = REPO / "astra-sim"
    os.chdir(astra)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(astra))
    try:
        from serving.core.hw_service import rescale_capacities, rescale_service_times
        reference = int(casr.get("capacity_reference_tokens", 1024) or 1024)
        rescale_service_times(casr, instances, "prefill_service_ms",
                              tokens=reference, verbose=False)
        rescale_service_times(casr, instances, "decode_service_ms", tokens=1,
                              verbose=False)
        rescale_capacities(casr, instances, verbose=False)
    finally:
        os.chdir(cwd)
    prefill = {int(key): float(value)
               for key, value in (casr.get("prefill_capacity") or {}).items()}
    decode = {int(key): float(value)
              for key, value in (casr.get("decode_capacity") or {}).items()}
    # The producer's KV egress caps a long-prompt Prefill below its compute
    # rate; the controller applies the same term before it solves.
    kv_per_token = float(casr.get("kv_bytes_per_token") or 0.0)
    links = casr.get("shared_links") or ()
    if kv_per_token > 0 and links:
        reference = int(casr.get("reference_tokens", 1250) or 1250)
        budgets = {}
        for link in links:
            capacity = float(link.get("capacity_bytes_per_s") or 0.0)
            for pair in (link.get("pairs") or ()):
                producer = int(pair[0])
                budgets[producer] = max(budgets.get(producer, 0.0), capacity)
        for producer, budget in budgets.items():
            if producer in prefill and budget > 0:
                prefill[producer] = min(prefill[producer],
                                        budget / (kv_per_token * reference))
    return prefill, decode


def cluster_nodes(cluster, prefill_caps, decode_caps):
    """``[(node_index, [(prefill_id, cap)...], [(decode_id, cap)...])]``."""
    nodes = []
    for index, node in enumerate(cluster.get("nodes", ())):
        prefills, decodes = [], []
        for inst in node.get("instances", ()):
            instance_id = int(inst["instance_id"])
            if str(inst.get("pd_type", "")).lower() == "prefill":
                prefills.append((instance_id, prefill_caps.get(instance_id, 0.0)))
            elif str(inst.get("pd_type", "")).lower() == "decode":
                decodes.append((instance_id, decode_caps.get(instance_id, 0.0)))
        nodes.append((index, prefills, decodes))
    return nodes


def is_node_uniform(cluster):
    """True when every node owns one Prefill and one Decode (the domain model).

    The config builder refuses an asymmetric layout whenever the cluster
    declares an intra-node link, because the topology then cannot express a
    separate same-node fabric.  Every multi-node config in this repo is
    node-uniform, so the *expressible* ratios there are ``n:n``.
    """
    if "intra_node_link_bw" not in cluster:
        return False
    shapes = set()
    for node in cluster.get("nodes", ()):
        roles = tuple(sorted(str(inst.get("pd_type", "")).lower()
                             for inst in node.get("instances", ())))
        shapes.add(roles)
    return shapes == {("decode", "prefill")}


def candidate_ratios(ratios, uniform, n_prefill, n_decode, n_nodes):
    out = []
    for ratio in ratios:
        left, _, right = ratio.partition(":")
        n_p, n_d = int(left), int(right)
        if not (1 <= n_p <= n_prefill and 1 <= n_d <= n_decode):
            continue
        if uniform and (n_p != n_d or n_p > n_nodes):
            continue
        out.append((n_p, n_d))
    return sorted(set(out))


def evaluate(n_p, n_d, trace, decode_reference, nodes, uniform,
             decode_batch_factor=1.0):
    """``(goodput, p_capacity, d_capacity, prefills, decodes)`` for a ratio.

    On a node-uniform cluster the decision is *which nodes stay up*: the ratio
    cannot split a node's two roles, so nodes are ranked by the goodput they
    contribute (their weaker leg) and the top ``n`` are kept.
    """
    prompt_work = max(1.0, trace["prompt_tokens"] / 1024.0)
    output_work = max(1.0, trace["output_tokens"] / max(1.0, decode_reference))
    if uniform:
        ranked = sorted(
            nodes,
            key=lambda item: (-min(sum(cap for _, cap in item[1]) / prompt_work,
                                   sum(cap for _, cap in item[2]) / output_work),
                              item[0]))
        chosen = ranked[:n_p]
        best_p = [entry for _, prefills, _ in chosen for entry in prefills]
        best_d = [entry for _, _, decodes in chosen for entry in decodes]
    else:
        best_p = sorted(
            [(entry[0], entry[1]) for _, prefills, _ in nodes for entry in prefills],
            key=lambda item: (-item[1], item[0]))[:n_p]
        best_d = sorted(
            [(entry[0], entry[1]) for _, _, decodes in nodes for entry in decodes],
            key=lambda item: (-item[1], item[0]))[:n_d]
    p_capacity = sum(value for _, value in best_p) / prompt_work
    # ``decode_batch_factor`` converts the profiler's *serial* Decode rate into
    # the batched rate the engine actually reaches (see the module docstring):
    # 1.0 keeps the raw figure, the measured value for an environment is what
    # makes the search agree with the executed timeline.
    d_capacity = (decode_batch_factor * sum(value for _, value in best_d)
                  / output_work)
    goodput = min(trace["offered_rps"], p_capacity, d_capacity)
    return goodput, p_capacity, d_capacity, best_p, best_d


def prune_config(cluster, prefill_ids, decode_ids):
    keep = set(prefill_ids) | set(decode_ids)
    pruned = copy.deepcopy(cluster)
    for node in pruned["nodes"]:
        node["instances"] = [inst for inst in node.get("instances", [])
                             if int(inst["instance_id"]) in keep]
        # The node's own count is what the config builder validates against.
        if "num_instances" in node:
            node["num_instances"] = len(node["instances"])
    pruned["nodes"] = [node for node in pruned["nodes"] if node.get("instances")]
    pruned["num_nodes"] = len(pruned["nodes"])
    casr = pruned.setdefault("casr", {})
    for link in casr.get("shared_links", []) or []:
        link["pairs"] = [list(pair) for pair in link.get("pairs", [])
                         if int(pair[0]) in keep and int(pair[1]) in keep]
    casr["shared_links"] = [link for link in (casr.get("shared_links") or [])
                            if link.get("pairs")]
    lifecycle = casr.setdefault("lifecycle", {})
    for key in ("min_active_prefill", "max_active_prefill"):
        if lifecycle.get(key):
            lifecycle[key] = min(int(lifecycle[key]), len(prefill_ids))
    if lifecycle.get("initial_active_prefill"):
        lifecycle["initial_active_prefill"] = [
            value for value in lifecycle["initial_active_prefill"]
            if int(value) in keep]
    return pruned


def verify(cluster_path, dataset, num_reqs, out_dir, tag):
    """Run the simulator once and return the measured SLO attainment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{tag}.csv"
    cmd = [sys.executable, "-m", "serving",
           "--cluster-config", str(cluster_path),
           "--dataset", dataset,
           "--num-reqs", str(num_reqs),
           "--dtype", "bfloat16", "--block-size", "16",
           "--max-num-seqs", "64", "--max-num-batched-tokens", "1024",
           "--log-level", "WARNING",
           "--output", str(csv_path),
           "--inputs-root", str(out_dir / f"{tag}-inputs"),
           "--request-routing-policy", "LOAD"]
    with (out_dir / f"{tag}.log").open("w") as log:
        subprocess.run(cmd, cwd=REPO, check=True, stdout=log,
                       stderr=subprocess.STDOUT)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    latency = sorted(float(row["latency"]) / 1e6 for row in rows)
    judged = [row for row in rows if str(row.get("slo_ok", "")).strip() != ""]
    met = sum(1 for row in judged if str(row["slo_ok"]).strip().lower() == "true")
    arrivals = [float(row["arrival"]) / 1e9 for row in rows]
    ends = [float(row["end_time"]) / 1e9 for row in rows]
    span = max(ends) - min(arrivals)
    served = sum(1 for row in rows if row["end_time"] != row["arrival"])
    return {
        "mean_ms": statistics.fmean(latency),
        "p95_ms": latency[min(len(latency) - 1, int(0.95 * len(latency)))],
        "attainment": (100.0 * met / len(judged)) if judged else None,
        "goodput_rps": served / span if span > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ratios", default="",
                        help="candidate nP:nD pairs, comma separated "
                             "(default: every pair up to the fleet size)")
    parser.add_argument("--write-config", default=None,
                        help="write the winning ratio's cluster config here")
    parser.add_argument("--verify", type=int, default=0,
                        help="run the top N candidates through the simulator")
    parser.add_argument("--verify-reqs", type=int, default=0,
                        help="cap the requests each verification run replays "
                             "(default: the whole trace)")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="offered rate to size for (default: the trace's "
                             "peak phase rate)")
    parser.add_argument("--decode-batch-factor", type=float, default=1.0,
                        help="multiply the profiler's serial Decode capacity by "
                             "this to describe a batched engine (1.0 = raw)")
    parser.add_argument("--out-root", default="/tmp/pd-ratio-search")
    args = parser.parse_args()

    cluster = json.loads(pathlib.Path(args.cluster_config).read_text(
        encoding="utf-8"))
    trace = read_trace(args.dataset)
    if args.rate:
        trace["offered_rps"] = float(args.rate)
    prefill_caps, decode_caps = resolve_capacities(cluster, args.cluster_config)
    decode_reference = float((cluster.get("casr") or {}).get(
        "decode_reference_tokens", 16) or 16)
    wanted = ([item for item in args.ratios.split(",") if item.strip()]
              if args.ratios else None)
    ratios = (wanted if wanted else
              [f"{n_p}:{n_d}" for n_p in range(1, len(prefill_caps) + 1)
               for n_d in range(1, len(decode_caps) + 1)])
    uniform = is_node_uniform(cluster)
    nodes = cluster_nodes(cluster, prefill_caps, decode_caps)
    candidates = candidate_ratios(ratios, uniform, len(prefill_caps),
                                  len(decode_caps), len(nodes))
    skipped = sorted(set(ratios) - {f"{n_p}:{n_d}" for n_p, n_d in candidates})

    print(f"trace: {trace['n']} requests, {trace['prompt_tokens']:.0f} prompt / "
          f"{trace['output_tokens']:.1f} output tokens, sizing for "
          f"{trace['offered_rps']:.2f} req/s, SLO {trace['slo_ttft_ms']:.0f} ms TTFT / "
          f"{trace['slo_tpot_ms']:.0f} ms TPOT (trace mean "
          f"{trace['mean_rps']:.2f} req/s)")
    print(f"fleet: {len(prefill_caps)} Prefill / {len(decode_caps)} Decode"
          + (" on a node-uniform topology: the expressible ratios are n:n and "
             "*which* nodes stay up is part of the decision"
             if uniform else ""))
    if skipped:
        print(f"skipped (not expressible here): {', '.join(skipped)}")
    rows = []
    for n_p, n_d in candidates:
        goodput, p_cap, d_cap, best_p, best_d = evaluate(
            n_p, n_d, trace, decode_reference, nodes, uniform,
            args.decode_batch_factor)
        rows.append({
            "ratio": f"{n_p}:{n_d}", "n_p": n_p, "n_d": n_d,
            "goodput_rps": goodput, "p_capacity": p_cap, "d_capacity": d_cap,
            "prefills": [instance for instance, _ in best_p],
            "decodes": [instance for instance, _ in best_d],
            "bound": ("prefill" if p_cap <= min(d_cap, trace["offered_rps"])
                      else "decode" if d_cap <= min(p_cap, trace["offered_rps"])
                      else "offer"),
        })
    rows.sort(key=lambda row: (-row["goodput_rps"], row["n_p"] + row["n_d"],
                              row["n_p"], row["n_d"]))
    print(f"\n{'ratio':>7}{'P cap':>9}{'D cap':>9}{'goodput':>9}  bound     instances")
    for row in rows[:12]:
        print(f"{row['ratio']:>7}{row['p_capacity']:>9.2f}{row['d_capacity']:>9.2f}"
              f"{row['goodput_rps']:>9.2f}  {row['bound']:<9} "
              f"P{row['prefills']} D{row['decodes']}")

    winner = rows[0]
    print(f"\nbest ratio {winner['ratio']} "
          f"(goodput {winner['goodput_rps']:.2f} req/s, bound by {winner['bound']})")
    if args.write_config:
        pruned = prune_config(cluster, winner["prefills"], winner["decodes"])
        pathlib.Path(args.write_config).write_text(
            json.dumps(pruned, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        print(f"wrote {args.write_config}")
    if args.verify:
        out_root = pathlib.Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        num_reqs = args.verify_reqs or sum(
            1 for line in open(args.dataset, encoding="utf-8") if line.strip())
        print(f"\nverifying the top {args.verify} candidates in the simulator")
        for row in rows[:args.verify]:
            pruned = prune_config(cluster, row["prefills"], row["decodes"])
            path = out_root / f"ratio-{row['n_p']}p{row['n_d']}d.json"
            path.write_text(json.dumps(pruned, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
            measured = verify(path, args.dataset, num_reqs, out_root,
                              f"ratio-{row['n_p']}p{row['n_d']}d")
            attainment = (f"{measured['attainment']:.1f}%"
                          if measured["attainment"] is not None else "n/a")
            print(f"  {row['ratio']:>7}  predicted {row['goodput_rps']:>6.2f}  "
                  f"measured goodput {measured['goodput_rps']:>6.2f} req/s  "
                  f"mean {measured['mean_ms']:>7.0f} ms  p95 {measured['p95_ms']:>7.0f} ms  "
                  f"SLO {attainment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
