#!/usr/bin/env python3
"""Derive a simulator cluster config from the real deployment config.

The hand-written ``casr_real_*`` configs drifted from the deployment they claim
to mirror (for example the three-domain "aligned" config models six RTX3090
instances while the fabric is a mix of RTX5090/3090/4090).  A drifted topology
silently invalidates any simulator-vs-real comparison, so the layout, the link
parameters and the P/D budgets are now generated from
``deploy/real_lmcache_pd/router_config.json`` instead of being retyped.

What is derived vs reused:

* **derived** -- node count and per-domain instances (one node per domain, one
  prefill/decode slot per instance), hardware per domain, instance ids, the
  inter-node link (min bandwidth / max RTT over the deployment's links), the
  intra-node link (measured same-host handoff), the producer egress ceiling
  (the deployment's ``kvlink-*`` budget) and the PD staging buffer;
* **reused** -- the solver block's simulator-only keys (``lifecycle``,
  ``resources``, ``kv_bytes_per_token`` ...), taken from an existing simulator
  config so the two halves stay in one place.

Usage:
    python3 tests/gen_sim_config_from_real.py --out configs/cluster/<name>.json
    python3 tests/gen_sim_config_from_real.py --check      # drift detector
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
REAL = REPO / "deploy" / "real_lmcache_pd" / "router_config.json"
TEMPLATE = REPO / "configs" / "cluster" / "casr_real_qwen3_8b_three_domain_aligned.json"
DEFAULT_OUT = REPO / "configs" / "cluster" / "casr_real_qwen3_8b_generated.json"

# Simulator hardware names and their memory model (size GiB, bandwidth GB/s).
HARDWARE = {
    "5090": ("RTX5090", {"mem_size": 32, "mem_bw": 1792}),
    "3090a": ("RTX3090", {"mem_size": 24, "mem_bw": 936}),
    "3090b": ("RTX3090", {"mem_size": 24, "mem_bw": 936}),
    "4090": ("RTX4090", {"mem_size": 24, "mem_bw": 1008}),
    "a100": ("A100", {"mem_size": 80, "mem_bw": 2039}),
}
MODEL = "Qwen/Qwen3-8B"


def has_profile(hardware, model=MODEL, variant="bf16"):
    """The simulator can only model a domain whose profile bundle exists.

    The deployment's A100 domain has no ``A100/Qwen/Qwen3-8B`` bundle (the
    node was reclaimed before one was captured), and generating a config that
    names it makes every run fail at trace-generation time.
    """
    return (REPO / "profiler" / "perf" / hardware / model / variant).is_dir()
# Same-host P/D handoff measured on the deployment: 585 ms per 1000 prompt
# tokens.  One prompt token is 147 456 B of KV (Qwen3-8B bf16), so 1000 tokens
# is 147.5 MB and the rate is 252 MB/s -- the deployment record's own table says
# 257 MB/s for this pair.  (An earlier revision wrote 0.31 GB/s by dividing the
# 1250-token KV size, 184 MB, by the *1000-token* time; that made every same-host
# push 17% too fast, which is the residual the recorded-pacing replay showed:
# sim TTFT 3839 ms against the cluster's 4795 ms.)  Cross-domain uses the
# deployment's links.
SAME_HOST_HANDOFF_GBPS = 0.257
SAME_HOST_LATENCY_NS = 1000
# Measured container restart -> serving time (2026-09-14/15, both the real
# router's WARMING gate and the p4090/p3090b scale-out).
MEASURED_CONTAINER_BOOT_MS = 45000


def build(real=None, template=None, keep_nonuniform=False, kv_heavy=False,
          return_maps=False, exclude=()):
    real = real or json.loads(REAL.read_text(encoding="utf-8"))
    template = template or json.loads(TEMPLATE.read_text(encoding="utf-8"))
    exclude = {str(item) for item in exclude}

    # Pass 1: which domains can be modelled, and which instances they own.
    domains, skipped = [], []
    excluded_by_request = []
    for domain, spec in real["hosts"].items():
        if domain in exclude:
            # Small-cluster comparisons take a subset of the deployment on
            # purpose (e.g. everything except the shared A100 host).
            excluded_by_request.append(domain)
            continue
        hardware, mem = HARDWARE[domain]
        if not has_profile(hardware):
            skipped.append(f"{domain} ({hardware}: no {MODEL} profile bundle)")
            continue
        owned = []
        for role, prefix in (("prefill", "p"), ("decode", "d")):
            source = next((item for item in
                           (real.get("prefills", []) if role == "prefill"
                            else real.get("decodes", []))
                           if item["id"] in (f"{prefix}_{domain}", f"{prefix}{domain}")), None)
            if source is not None:
                owned.append((role, source))
        if owned:
            domains.append({"domain": domain, "hardware": hardware, "mem": mem,
                            "owned": owned})

    # Pass 2: the domain-aware topology needs one instance count for every
    # node, so a domain whose P/D pairing is incomplete is dropped here (see
    # the note further down).
    excluded = []
    if not keep_nonuniform:
        counts = [len(item["owned"]) for item in domains]
        if len(set(counts)) > 1:
            target = max(set(counts), key=counts.count)
            for item in list(domains):
                if len(item["owned"]) == target:
                    continue
                domains.remove(item)
                for role, source in item["owned"]:
                    excluded.append(f"{role}#{source.get('instance_id')} ({item['hardware']})")

    # Pass 3: assign simulator instance ids and build the real->sim maps, so
    # the solver's capacity maps and link budgets follow the domains instead of
    # keeping the deployment's numbering.  Without this the capacities are
    # silently attached to the wrong hardware (verified 2026-09-15: the A100
    # slot inherited p4090's 59 req/s while the 4090 slot fell back to
    # max_num_seqs), which would invalidate any elasticity comparison.
    nodes, prefill_id_map, decode_id_map, instance_id = [], {}, {}, 0
    for item in domains:
        instances = []
        for role, source in item["owned"]:
            (prefill_id_map if role == "prefill" else decode_id_map)[
                int(source.get("instance_id", -1))] = instance_id
            instances.append({
                "instance_id": instance_id,
                "model_name": MODEL,
                "hardware": item["hardware"],
                "npu_mem": {**item["mem"], "mem_latency": 0, "mem_util": 0.9},
                "pd_type": role,
                "tp_size": 1,
                "enable_prefix_caching": True,
            })
            instance_id += 1
        nodes.append({"num_instances": len(instances),
                      "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
                      "instances": instances})

    # The simulator's multi-dim topology takes ONE bandwidth per dimension, so
    # a heterogeneous fabric cannot be expressed link-by-link.  Use the
    # *standard* inter-domain link (the value shared by the majority of pairs)
    # rather than the slowest one: otherwise a single slow host (the reclaimed
    # A100 over a 0.118 Gbps path) throttles every other pair by 7x.  The
    # exception is recorded in the comment below so a run cannot silently
    # assume uniform links.
    import collections
    cross = [link for link in real.get("links", ())
             if link["bw_gbps"] < 1.0] or real.get("links", [])
    counts = collections.Counter(item["bw_gbps"] for item in cross)
    standard_bw = max(counts, key=lambda bw: (counts[bw], bw)) if counts else 0.88
    link_bw = standard_bw / 8.0
    link_latency = max(item["rtt_ms"] for item in cross) * 1e6 if cross else 4.8e7
    kv_caps = [link["capacity_bytes_per_s"]
               for link in (real.get("casr", {}).get("shared_links") or ())]
    standard_kv = max(kv_caps) if kv_caps else None
    # Which regime the config models decides the handoff charge.  At 254-token
    # prompts a handoff measures 40-90 ms (37 MB, i.e. 0.4-0.9 GB/s) and the
    # producer ceiling does NOT bind; at 1250 tokens it is 585-731 ms for
    # 184 MB (0.25-0.31 GB/s) and it does.  Default to the wire value the
    # short-prompt comparisons were taken with; ``--kv-heavy`` switches to the
    # measured producer ceiling.
    kv_egress = (standard_kv / 1e9) if (kv_heavy and standard_kv) else link_bw

    config = {
        "num_nodes": len(nodes),
        "link_bw": round(link_bw, 4),
        "link_latency": round(link_latency, 1),
        "_generated": ("Derived from deploy/real_lmcache_pd/router_config.json by "
                       "tests/gen_sim_config_from_real.py -- do not hand-edit; "
                       "re-run the generator instead."),
        "_link_comment": (
            f"link_bw={round(link_bw, 4)} GB/s is the deployment's standard cross-domain "
            f"link ({standard_bw} Gbps, {counts.get(standard_bw, 0)} pairs) and "
            f"link_latency={round(link_latency, 1)} ns the slowest RTT; the fabric is NOT "
            f"uniform (A100 pairs run 0.118-0.146 Gbps, kvlink budget {sorted(kv_caps) if kv_caps else '-'} B/s), "
            "and the simulator's one-value-per-dimension topology cannot express that. "
            f"intra_node_link_bw={SAME_HOST_HANDOFF_GBPS} GB/s is the measured same-host "
            "P/D handoff (585 ms per 1000 prompt tokens), and kv_egress_gbps is the "
            "producer-side push ceiling from the deployment's standard kvlink budget."),
        "kv_egress_gbps": round(kv_egress, 4),
        "pd_buffer_bytes": 34359738368,
        "casr": dict(template.get("casr", {})),
        "nodes": nodes,
    }
    # The domain-aware link model needs a node-uniform layout: the simulator
    # splits the instance dimension into [slots_per_node, num_nodes] and gives
    # the inner dimensions the intra-node link.  The real fabric is not uniform
    # (3090b runs a Prefill with no same-domain Decode), so the choice is
    # explicit rather than silent: keep every domain and lose the same-host
    # discount, or drop the odd domain and keep it.
    if excluded:
        config["_excluded_for_uniformity"] = excluded
        config["_excluded_comment"] = (
            "Dropped so every node owns the same number of instances: the "
            "domain-aware topology cannot otherwise distinguish a same-host "
            "handoff from a cross-domain one, and charging both the "
            "cross-domain link overstates latency ~40x on short prompts. "
            "Use --keep-nonuniform to keep them and lose the domain model.")
    slots = {node["num_instances"] for node in nodes}
    uniform = len(slots) == 1
    if uniform:
        config["intra_node_link_bw"] = SAME_HOST_HANDOFF_GBPS
        config["intra_node_link_latency"] = SAME_HOST_LATENCY_NS
    else:
        config["_layout_warning"] = (
            f"nodes carry {sorted(slots)} instances, so the layout is not "
            "node-uniform and the intra-node link model is disabled: a same-host "
            "P/D handoff is charged the cross-domain link. Real same-host handoff "
            f"measures {SAME_HOST_HANDOFF_GBPS} GB/s (585 ms per 1000 tokens).")
    if skipped:
        config["_skipped_domains"] = (
            "Excluded because the simulator has no profile bundle for them: "
            + "; ".join(skipped)
            + ". Add a profiler run for that hardware/model pair to include it.")
    if excluded_by_request:
        config["_excluded_by_request"] = (
            "Removed by --exclude for a small-cluster run: "
            + ", ".join(sorted(excluded_by_request))
            + ". The real arms must disable the same instance ids.")
    # The solver block must price the real capacities / links, not the ones the
    # template happened to carry.
    casr = config["casr"]
    real_casr = real.get("casr", {})
    casr["prefill_capacity"] = {
        str(prefill_id_map[int(key)]): value
        for key, value in (real_casr.get("prefill_capacity") or {}).items()
        if int(key) in prefill_id_map}
    casr["decode_capacity"] = {
        str(decode_id_map[int(key)]): value
        for key, value in (real_casr.get("decode_capacity") or {}).items()
        if int(key) in decode_id_map}
    # Per-instance compute cost.  The real control loop seeds its objective with
    # each instance's *measured* single-request service time
    # (``casr_control.py``: ``prefill_service.setdefault(id, instance.service_ms)``),
    # so the simulator has to carry the same numbers.  Inheriting the
    # template's copy instead left them keyed to the *template's* topology:
    # measured 2026-09-16, a 5-domain build priced 3090b with 4090's 39.2 ms and
    # gave the A100 and 4090 instances no entry at all (they silently fell back
    # to the scheduler's default service time).
    real_prefill_service = {int(item["instance_id"]): float(item["service_ms"])
                            for item in (real.get("prefills") or [])
                            if item.get("service_ms")}
    real_decode_service = {int(item["instance_id"]): float(item["service_ms"])
                           for item in (real.get("decodes") or [])
                           if item.get("service_ms")}
    if real_prefill_service:
        casr["prefill_service_ms"] = {
            str(prefill_id_map[key]): value
            for key, value in real_prefill_service.items() if key in prefill_id_map}
    if real_decode_service:
        casr["decode_service_ms"] = {
            str(decode_id_map[key]): value
            for key, value in real_decode_service.items() if key in decode_id_map}
    # Per-pair network price, mirroring the real control loop
    # (``casr_control.py``: same-domain handoffs are free, cross-domain ones pay
    # the measured RTT, and capacity is charged once through ``shared_links``).
    # Without this the simulator's LP has no per-pair cost at all and falls back
    # to ``|NPU index distance| * 0.001`` -- an arbitrary proxy that made the
    # cheapest edge depend on the order instances happen to be listed in.
    # Measured 2026-09-16: the CASR plan single-homed every class onto
    # 3090a, while the real router (with pair costs) chose the 5090 pair.
    link_rtt = {(link.get("src"), link.get("dst")): float(link.get("rtt_ms", 0.0))
                for link in (real.get("links") or [])}
    pair_costs = {}
    for prefill_item in (real.get("prefills") or ()):
        source_id = int(prefill_item["instance_id"])
        if source_id not in prefill_id_map:
            continue
        for decode_item in (real.get("decodes") or ()):
            target_id = int(decode_item["instance_id"])
            if target_id not in decode_id_map:
                continue
            same_domain = prefill_item.get("domain") == decode_item.get("domain")
            rtt_ms = 0.0 if same_domain else link_rtt.get(
                (prefill_item.get("domain"), decode_item.get("domain")), 0.0)
            pair_costs[f"{prefill_id_map[source_id]},{decode_id_map[target_id]}"] = {
                "rtt_ms": rtt_ms,
                # Capacity lives in ``shared_links``; charging bytes/bandwidth
                # per pair as well double-counts it (same reasoning as the real
                # control loop).
                "bandwidth_bytes_per_s": 0.0,
            }
    if pair_costs:
        casr["pair_costs"] = pair_costs
    # The fast router's own load denominator.  ``_pick_load`` ranks candidates
    # by ``(inflight + 1) / capacity`` using the deployment's ``capacity`` field
    # (p5090 220, p3090a 114, p4090 228 ...), which is *not* the solver capacity
    # the LP prices with (63/45/59).  Without it the simulator's ``load``
    # baseline picked a different instance than the recorded run -- visible on
    # the forced-transfer replay, where the real arm put 104/120 on p4090 while
    # the simulator spread them (measured 2026-09-16).
    router_capacity = {}
    for key in ("prefills", "decodes"):
        for item in (real.get(key) or ()):
            source_id = int(item["instance_id"])
            mapping = (prefill_id_map if key == "prefills" else decode_id_map)
            if source_id in mapping and item.get("capacity"):
                router_capacity[str(mapping[source_id])] = float(item["capacity"])
    if router_capacity:
        casr["router_capacity"] = router_capacity
    links = []
    for link in (real_casr.get("shared_links") or []):
        pairs = [[prefill_id_map[int(p)], decode_id_map[int(d)]]
                 for p, d in (link.get("pairs") or [])
                 if int(p) in prefill_id_map and int(d) in decode_id_map]
        if not pairs:
            continue
        links.append({**link, "pairs": pairs,
                      "_id_comment": "pairs re-keyed to simulator instance ids"})
    if links:
        casr["shared_links"] = links
    # Model/economics parameters must travel with the deployment.
    #
    # The simulator configs carry a *simulator-only* ``structural`` block which
    # has ``enabled: false`` (the counterfactual evaluator is opt-in there) and
    # none of the length-normalised capacity parameters.  Copying the template
    # verbatim therefore produced a config whose structural evaluator never ran
    # -- measured 2026-09-15: the elastic arm was byte-identical to the static
    # one (mean 4400 ms, TTFT 1814 ms) while an always-on 4-Prefill pool showed
    # 2684 ms / 100 ms, i.e. the opportunity was real and was never taken.
    structural = dict(casr.get("structural") or {})
    structural.update({key: value for key, value in (real_casr.get("structural") or {}).items()
                       if not str(key).startswith("_")})
    # ``enabled`` is the deployment's intent, not the template's: the real
    # router runs the evaluator whenever the config does not say otherwise
    # (default True), while the simulator's own configs switch it off.
    structural["enabled"] = bool(
        (real_casr.get("structural") or {}).get("enabled", True))
    casr["structural"] = structural
    for key in ("capacity_reference_tokens", "prefill_fixed_ms",
                "prefill_ms_per_1k_tokens", "decode_reference_tokens",
                "decode_fixed_ms", "decode_ms_per_1k_tokens", "work_ceiling",
                "backlog_rps_multiple", "single_home_below_rps",
                "class_demand_floor_rps", "plan_uncovered_penalty"):
        if key in real_casr:
            casr[key] = real_casr[key]
    # P/D handoff vs local recompute.  The real router decides this per request
    # (``disagg_router.kv_exchange_decision``) and defaults to ``auto``; the
    # constants are the ones calibrated on the deployment
    # (``tests/calibrate_native_pd.py``, 2026-09-13) and exposed as env knobs
    # there.  They belong in the config so a generated simulator run makes the
    # same choice the recorded real run made: measured 2026-09-16, the real
    # ``load`` arm answered 200/200 requests by recomputing locally
    # (``prefill_ms`` p50 = 0) while the simulator shipped every KV.
    casr.setdefault("local_prefill", real_casr.get("local_prefill", "auto"))
    for key, default in (("local_prefill_ms_per_1k", 93.0),
                         ("transfer_ms_per_1k_local", 585.0),
                         ("transfer_ms_per_1k_cross", 1351.0),
                         ("transfer_fixed_ms_cross", 48.0),
                         ("local_prefill_queue_weight", 1.0),
                         ("local_prefill_queue_cap", 3.0)):
        casr.setdefault(key, real_casr.get(key, default))
    tokens_per_s = real_casr.get("prefill_tokens_per_s") or {}
    if tokens_per_s:
        casr["prefill_tokens_per_s"] = {
            str(prefill_id_map[int(key)]): value
            for key, value in tokens_per_s.items()
            if not str(key).startswith("_") and int(key) in prefill_id_map}
    elif tokens_per_s.get("_comment"):
        casr["_prefill_tokens_comment"] = tokens_per_s["_comment"]
    # The worker lifecycle has to describe *this* topology: the template it is
    # copied from may declare a different pool size, and its default warmup
    # (250 ms) is two orders of magnitude below the measured container restart.
    # ``min_active_prefill`` comes from the deployment; the ceiling is however
    # many Prefills the generated topology actually owns; the warmup is the
    # measured restart-to-serving time (45 s, 2026-09-14/15) rather than the
    # deployment's unvalidated placeholder value.
    prefill_count = sum(1 for node in nodes for instance in node["instances"]
                        if instance["pd_type"] == "prefill")
    lifecycle = dict(casr.get("lifecycle") or {})
    lifecycle["min_active_prefill"] = int(real_casr.get("min_active_prefill", 1) or 1)
    lifecycle["max_active_prefill"] = prefill_count
    lifecycle["warmup_ms"] = MEASURED_CONTAINER_BOOT_MS
    # The lifecycle's own demand->worker-count heuristic reads per-Prefill
    # capacity; the template's copy is keyed by the *template's* topology and
    # would rate wrong instances (or fall back to max_num_seqs for the rest).
    lifecycle["prefill_capacity"] = dict(casr["prefill_capacity"])
    casr["lifecycle"] = lifecycle
    # Resource boundary: one node per modelled domain, with the GPUs that
    # domain actually owns.  The template's block described the *template's*
    # topology (three nodes, two GPUs each), so a scale-out asked for a GPU on
    # a node that had none left and the ``+P`` decision -- which did fire, gain
    # 137 at t+0.12 s in the 2026-09-15 run -- could never be executed.
    resources = dict(casr.get("resources") or {})
    resources["startup_ms"] = MEASURED_CONTAINER_BOOT_MS
    resources["nodes"] = {
        str(index): {"gpu_count": len(item["owned"]),
                     "gpu_mem_gb": [item["mem"]["mem_size"]] * len(item["owned"])}
        for index, item in enumerate(domains)}
    casr["resources"] = resources
    if not casr.get("prefill_capacity"):
        casr["prefill_capacity"] = {"0": 1.0}
    if return_maps:
        # The deployment->simulator instance maps are what every per-instance
        # block has to be re-keyed through; expose them so tests can rebuild the
        # expected values from the deployment instead of restating the numbers.
        return config, {"prefill": prefill_id_map, "decode": decode_id_map}
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--keep-nonuniform", action="store_true",
                        dest="keep_nonuniform",
                        help="keep domains whose P/D pairing is incomplete "
                             "(loses the same-host handoff discount)")
    parser.add_argument("--kv-heavy", action="store_true", dest="kv_heavy",
                        help="charge the handoff at the measured producer ceiling "
                             "(0.26 GB/s) instead of the wire: correct for "
                             "1250-token prompts, ~3x pessimistic at 254")
    parser.add_argument("--check", action="store_true",
                        help="compare the checked-in config with a fresh build")
    parser.add_argument("--exclude", default="",
                        help="comma-separated domains to drop (small-cluster runs, "
                             "e.g. --exclude a100,3090b)")
    args = parser.parse_args()
    exclude = [item.strip() for item in args.exclude.split(",") if item.strip()]
    generated = build(keep_nonuniform=args.keep_nonuniform, kv_heavy=args.kv_heavy,
                      exclude=exclude)
    text = json.dumps(generated, ensure_ascii=False, indent=2) + "\n"
    path = pathlib.Path(args.out)
    if args.check:
        if not path.exists():
            print(f"MISSING: {path} -- run the generator")
            return 1
        if path.read_text(encoding="utf-8") != text:
            print(f"DRIFT: {path} differs from the deployment-derived config")
            print("  regenerate with: python3 tests/gen_sim_config_from_real.py "
                  f"--out {path}")
            return 1
        print(f"OK: {path} matches deploy/real_lmcache_pd/router_config.json")
        return 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({generated['num_nodes']} nodes, "
          f"{sum(node['num_instances'] for node in generated['nodes'])} instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
