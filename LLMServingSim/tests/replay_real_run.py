#!/usr/bin/env python3
"""Replay a recorded real run in the simulator and diff the two.

A real comparison directory (``tests/run_real_multidomain_comparison.sh``)
contains everything a replay needs: ``run-config.json`` (trace, request count,
output cap, policies), ``metrics-<policy>.jsonl`` (per-request TTFT/TPOT/E2E,
the instances each request used, and whether it took the local or the handoff
path) and ``state-<policy>.jsonl`` for plan policies.  This tool reads that
bundle, runs the same arms against a deployment-derived simulator config, and
prints the side-by-side table, so alignment claims are reproducible rather than
restated from a one-off experiment.

    tests/replay_real_run.py --real-dir /mnt/home/casr/results/small3-0916 \
        --cluster-config configs/cluster/casr_real_small3_generated.json \
        --policies load casr_lp

``--max-output-tokens`` defaults to the recorded run's cap: the real client
truncates generation (16 tokens in every archived comparison), and a replay
that ignored it would compare a different workload.
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

# Real policy name -> simulator flags.
POLICY_ARGS = {
    "load": ["--request-routing-policy", "LOAD"],
    "cache_aware": ["--request-routing-policy", "CACHE_AWARE"],
    "rr": ["--request-routing-policy", "RR"],
    "random": ["--request-routing-policy", "RAND"],
}


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def sim_args(policy, control_interval_s):
    if policy in POLICY_ARGS:
        return list(POLICY_ARGS[policy])
    # Everything else in the real stack is a CASR plan policy (casr, casr_lp,
    # casr_full, and the ablations): the shared solver runs inside the router.
    solver = "lp" if policy.startswith("casr_lp") or policy in ("casr", "casr_full") else "greedy"
    return ["--enable-casr", "--casr-solver", solver,
            "--casr-control-interval-ms", str(int(control_interval_s * 1000))]


def load_real_metrics(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("request_id", "")).startswith("kvcheck"):
            continue          # the correctness gate's probes, not the workload
        if row.get("status") not in (200, "200", None):
            continue
        rows.append(row)
    return rows


def summarise_real(rows):
    # The correctness gate fires a handful of probe requests before the
    # workload; they are not part of the compared trace.
    rows = [r for r in rows
            if not str(r.get("request_id", "")).startswith("kvcheck")]
    e2e = [float(r["total_ms"]) for r in rows if r.get("total_ms")]
    ttft = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms")]
    tpot = [float(r["tpot_ms"]) for r in rows if r.get("tpot_ms")]
    return {
        "n": len(rows),
        "e2e_p50": quantile(e2e, 0.50) if e2e else None,
        "e2e_p95": quantile(e2e, 0.95) if e2e else None,
        "ttft_p50": quantile(ttft, 0.50) if ttft else None,
        "tpot_p50": quantile(tpot, 0.50) if tpot else None,
        "exchange": dict(collections.Counter(r.get("exchange") for r in rows)),
        "served": dict(collections.Counter(r["decode"] for r in rows if r.get("decode"))),
    }


def load_sim_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def summarise_sim(rows, instance_names):
    """Summarise a simulator arm, naming instances the way the real config does.

    For a locally-recomputed request the simulator records the instance that
    actually ran it in ``prefill_instance_id`` (the ``decode_instance_id``
    column is the *pair* the router named), so the placement column has to come
    from whichever one served the request.
    """
    e2e = [float(r["latency"]) / 1e6 for r in rows]
    ttft = [float(r["TTFT"]) / 1e6 for r in rows]
    tpot = [float(r["TPOT"]) / 1e6 for r in rows]
    served = collections.Counter()
    for row in rows:
        instance_id = (row["prefill_instance_id"] if row.get("exchange") == "local"
                       else row["decode_instance_id"])
        served[instance_names.get(int(instance_id), instance_id)] += 1
    return {
        "n": len(rows),
        "e2e_p50": quantile(e2e, 0.50) if e2e else None,
        "e2e_p95": quantile(e2e, 0.95) if e2e else None,
        "ttft_p50": quantile(ttft, 0.50) if ttft else None,
        "tpot_p50": quantile(tpot, 0.50) if tpot else None,
        "exchange": dict(collections.Counter(r.get("exchange") for r in rows)),
        "served": dict(served),
    }


def instance_names(cluster_config):
    """simulator instance id -> the deployment's instance name (``p5090`` ...).

    The simulator renumbers instances when it builds a topology, so a replay's
    landing counts are only comparable to the recorded run after mapping back
    by (hardware, role).  The deployment config declares which hardware each
    domain runs, and the generated simulator config names the hardware of every
    instance.
    """
    import gen_sim_config_from_real as gen                       # noqa: E402

    config = json.loads(pathlib.Path(cluster_config).read_text(encoding="utf-8"))
    sim_key = {}
    for node in config["nodes"]:
        for instance in node["instances"]:
            sim_key[int(instance["instance_id"])] = (
                instance.get("hardware"), instance.get("pd_type"))

    real = json.loads(gen.REAL.read_text(encoding="utf-8"))
    available = collections.defaultdict(list)
    for key, role in (("prefills", "prefill"), ("decodes", "decode")):
        for item in real[key]:
            hardware = gen.HARDWARE[item["domain"]][0]
            available[(hardware, role)].append(item["id"])
    for names in available.values():
        names.sort()

    mapping = {}
    for instance_id in sorted(sim_key):
        candidates = available.get(sim_key[instance_id])
        if candidates:
            mapping[instance_id] = candidates.pop(0)
    return mapping


def control_interval_s():
    """The plan policies' control period, from the deployment config."""
    config = json.loads((REPO / "deploy" / "real_lmcache_pd"
                         / "router_config.json").read_text(encoding="utf-8"))
    return float((config.get("casr") or {}).get("control_interval_s", 1.0))


def deployment_max_num_seqs():
    """The engine's concurrency budget, which the deployment sets explicitly.

    ``start_multidomain_pd.sh`` launches every instance with ``max_num_seqs``
    (16 on this deployment).  A replay that leaves the simulator's default in
    place runs with a different engine: measured 2026-09-16, a forced-transfer
    arm piled 100+ sequences into one Decode and paid the worst-case batch
    lookup on every step.
    """
    config = json.loads((REPO / "deploy" / "real_lmcache_pd"
                         / "router_config.json").read_text(encoding="utf-8"))
    values = [int(item.get("max_num_seqs", 0) or 0)
              for key in ("prefills", "decodes") for item in config[key]]
    values = [value for value in values if value > 0]
    return min(values) if values else 0


def arm_config_from_recording(cluster_config, real_dir, policy, names, out_dir):
    """The Prefill pool the cluster had up, written into the simulator config.

    ``--replay-placement`` pins every request's (Prefill, Decode) pair, but the
    simulator's own lifecycle still decides which workers start ACTIVE -- and
    its demand heuristic ranks workers by instantaneous load, so an arm can
    drain exactly the worker the recording used first (measured 2026-09-16: 247
    of 300 handoffs replayed across domains, although the cluster pushed 290 of
    them over its intra-node link).  The recording knows the answer: every
    Prefill that served a request is a worker the run had (p5090 alone at the
    start of the elasticity arm, with p3090a scaled in at 52 s), and a pinned
    pair can only be replayed if its Prefill is in the pool.
    """
    by_name = {name: instance_id for instance_id, name in names.items()}
    rows = [row for row in load_real_metrics(
        pathlib.Path(real_dir) / f"metrics-{policy}.jsonl")
        if str(row.get("request_id", "")).startswith("ds-")]
    rows.sort(key=lambda row: float(row.get("ts") or 0.0))
    pool = []
    for row in rows:
        instance_id = by_name.get(row.get("prefill"))
        if instance_id is not None and instance_id not in pool:
            pool.append(instance_id)
    if not pool:
        return pathlib.Path(cluster_config)
    config = json.loads(pathlib.Path(cluster_config).read_text(encoding="utf-8"))
    lifecycle = config.setdefault("casr", {}).setdefault("lifecycle", {})
    lifecycle["initial_active_prefill"] = sorted(pool)
    lifecycle["min_active_prefill"] = len(pool)
    # Absolute: the simulator resolves a relative ``--cluster-config`` from one
    # directory up (see ``build_cluster_config``).
    path = (pathlib.Path(out_dir) / "arm-config.json").resolve()
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"   recorded Prefill pool: {sorted(pool)} "
          f"(min_active_prefill={len(pool)}, written to {path.name})")
    return path


def run_sim_arm(policy, args, run_config, sim_config, out_dir):
    csv_path = out_dir / f"{policy}.csv"
    command = [sys.executable, "-m", "serving",
               "--cluster-config", str(sim_config),
               "--dataset", str(run_config["TRACE"]),
               "--num-reqs", str(run_config["NUM_REQS"]),
               "--dtype", "bfloat16", "--block-size", "16",
               "--log-level", "WARNING",
               "--max-output-tokens", str(args.max_output_tokens),
               "--output", str(csv_path),
               "--inputs-root", str(out_dir / f"{policy}-inputs")]
    budget = deployment_max_num_seqs()
    if budget:
        command += ["--max-num-seqs", str(budget)]
    if args.client_concurrency:
        command += ["--client-concurrency", str(args.client_concurrency)]
    command += sim_args(policy, control_interval_s())
    if args.replay_placement:
        command += ["--replay-placement", str(out_dir / f"{policy}-placement.json")]
    print("   $ " + " ".join(command[1:8]) + " ...")
    subprocess.run(command, cwd=REPO, check=True,
                   stdout=(out_dir / f"{policy}.log").open("w"),
                   stderr=subprocess.STDOUT)
    return csv_path


def build_placement(real_dir, policy, names):
    """``{trace index: [sim prefill id, sim decode id]}`` from the recording.

    The real client names its requests ``ds-<index>``, and the simulator numbers
    the trace rows the same way, so the cluster's per-request placement can be
    handed to the simulator verbatim (after mapping instance *names* to the
    simulator's renumbered ids).
    """
    by_name = {name: instance_id for instance_id, name in names.items()}
    placement = {}
    path = pathlib.Path(real_dir) / f"metrics-{policy}.jsonl"
    for row in load_real_metrics(path):
        request_id = str(row.get("request_id", ""))
        if not request_id.startswith("ds-"):
            continue
        prefill = by_name.get(row.get("prefill"))
        decode = by_name.get(row.get("decode"))
        if prefill is None or decode is None:
            continue
        placement[int(request_id.split("-", 1)[1])] = [prefill, decode]
    return placement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=0,
                        help="generation cap; the client's cap is applied by default")
    parser.add_argument("--skip-sim", action="store_true",
                        help="compare recorded results only (no simulator run)")
    parser.add_argument("--replay-placement", action="store_true",
                        help="route every request to the instance the cluster "
                             "used, isolating execution fidelity from the "
                             "controller's choices")
    parser.add_argument("--client-concurrency", type=int, default=8,
                        help="closed-loop arrival cap; the recorded client uses 8")
    args = parser.parse_args()

    real_dir = pathlib.Path(args.real_dir)
    run_config = json.loads((real_dir / "run-config.json").read_text(encoding="utf-8"))
    out_dir = pathlib.Path(args.out) if args.out else real_dir / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = args.policies or [
        name for name in (run_config.get("POLICIES") or "load").split() if name]
    if not args.max_output_tokens:
        args.max_output_tokens = int(run_config.get("MAX_OUTPUT_TOKENS") or 0)
    names = instance_names(args.cluster_config)

    report = {}
    print(f"{'policy':<12}{'side':<5}{'n':>5}{'E2E p50':>10}{'E2E p95':>10}"
          f"{'TTFT p50':>10}{'TPOT p50':>10}   exchange / served")
    for policy in policies:
        real_path = real_dir / f"metrics-{policy}.jsonl"
        if not real_path.exists():
            print(f"{policy:<12} (no recorded metrics)")
            continue
        real = summarise_real(load_real_metrics(real_path))
        report.setdefault(policy, {})["real"] = real
        print(f"{policy:<12}{'real':<5}{real['n']:>5}{real['e2e_p50']:>10.1f}"
              f"{real['e2e_p95']:>10.1f}{real['ttft_p50']:>10.1f}{real['tpot_p50']:>10.1f}"
              f"   {real['exchange']} / {real['served']}")
        if args.skip_sim:
            continue
        if args.replay_placement:
            placement = build_placement(real_dir, policy, names)
            (out_dir / f"{policy}-placement.json").write_text(
                json.dumps(placement), encoding="utf-8")
            print(f"   replayed placement: {len(placement)} requests")
            sim_config = arm_config_from_recording(
                args.cluster_config, real_dir, policy, names, out_dir)
        else:
            sim_config = args.cluster_config
        csv_path = run_sim_arm(policy, args, run_config, sim_config, out_dir)
        sim = summarise_sim(load_sim_csv(csv_path), names)
        report[policy]["sim"] = sim
        ratio = sim["e2e_p50"] / real["e2e_p50"] if real["e2e_p50"] else float("nan")
        print(f"{'':<12}{'sim':<5}{sim['n']:>5}{sim['e2e_p50']:>10.1f}"
              f"{sim['e2e_p95']:>10.1f}{sim['ttft_p50']:>10.1f}{sim['tpot_p50']:>10.1f}"
              f"   {sim['exchange']} / {sim['served']}   ({ratio:.2f}x)")
    (out_dir / "alignment.json").write_text(json.dumps(report, indent=2),
                                            encoding="utf-8")
    print(f"\nwrote {out_dir / 'alignment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
