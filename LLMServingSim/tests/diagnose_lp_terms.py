#!/usr/bin/env python3
"""Offline LP term decomposition for the CASR flow solver.

Answers one question: given the demand a real run actually showed, which cost
term does the LP react to?  The router only publishes the *last* tick's
diagnostics (``state-<policy>.json``), which is written after the load has
drained, so peak-time behaviour cannot be read back from the artifacts.  This
tool rebuilds the solver from the deployment config and solves it for a
synthetic demand vector, printing every term separately.

Usage:
    python3 tests/diagnose_lp_terms.py --rate 1.41 --prompt-tokens 1211 \
        --prefills p5090 --decodes d5090,d3090a,d4090
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "deploy" / "real_lmcache_pd"))
sys.path.insert(0, str(REPO / "serving"))

from casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig  # noqa: E402


class Sched:
    """Minimal duck type the solver reads (mirrors ``casr_control.RealInstance``)."""

    def __init__(self, instance_id, role, *, max_num_seqs=16, running=0,
                 waiting=0, capacity=1.0, service_ms=0.0, domain_index=0):
        self.instance_id = instance_id
        self.pd_type = role
        self.max_num_seqs = max_num_seqs
        self.running = [None] * running
        self.waiting = [None] * waiting
        self.capacity = capacity
        self.service_ms = service_ms
        self.start_npu = domain_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "deploy" / "real_lmcache_pd"
                                                / "router_config.json"))
    parser.add_argument("--rate", type=float, required=True,
                        help="total offered demand (req/s) to distribute over classes")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=1211)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prefills", default="p5090")
    parser.add_argument("--decodes", default="d5090,d3090a,d4090")
    parser.add_argument("--workers", type=int, default=0,
                        help="override max_num_seqs for every instance")
    parser.add_argument("--prefill-capacity", default="",
                        help="override, e.g. 'p5090=63' or 'all=6.8'")
    parser.add_argument("--work-scale", type=float, default=1.0,
                        help="multiply every class's work (simulates a "
                             "length-aware work model)")
    parser.add_argument("--kv-bytes-per-token", type=float, default=0.0,
                        dest="kv_bytes_per_token",
                        help="override the per-token KV size used for the "
                             "observed classes (default: the config's weights block)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--flows", action="store_true",
                        help="print the per-class (prefill, decode) flow split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    casr_block = dict(raw.get("casr") or {})
    config = FlowSolverConfig.from_dict(casr_block)
    kv_bytes_per_token = float(args.kv_bytes_per_token
                               or (raw.get("weights") or {}).get("kv_bytes_per_token")
                               or casr_block.get("default_kv_bytes")
                               or 0.0)

    want_p = {name.strip() for name in args.prefills.split(",") if name.strip()}
    want_d = {name.strip() for name in args.decodes.split(",") if name.strip()}
    prefill_capacity = dict(config.prefill_capacity)
    if args.prefill_capacity:
        for part in args.prefill_capacity.split(","):
            name, _, value = part.partition("=")
            value = float(value)
            for item in raw["prefills"]:
                if name == "all" or item["id"] == name.strip():
                    prefill_capacity[int(item.get("instance_id"))] = value
    config = FlowSolverConfig(**{**config.__dict__, "prefill_capacity": prefill_capacity})

    prefills, decodes = [], []
    for item in raw["prefills"]:
        if item["id"] in want_p:
            prefills.append(Sched(int(item.get("instance_id")), "prefill",
                                  max_num_seqs=args.workers or int(item.get("max_num_seqs", 16)),
                                  capacity=float(item.get("capacity", 1.0)),
                                  service_ms=float(item.get("service_ms", 0.0))))
    for item in raw["decodes"]:
        if item["id"] in want_d:
            decodes.append(Sched(int(item.get("instance_id")), "decode",
                                 max_num_seqs=args.workers or int(item.get("max_num_seqs", 16)),
                                 capacity=float(item.get("capacity", 1.0)),
                                 service_ms=float(item.get("service_ms", 0.0))))
    if not prefills or not decodes:
        raise SystemExit("no instance matched --prefills/--decodes")

    per_class = args.rate / max(1, args.classes)
    # The controller publishes one row per (prefill, class) holding the arrivals
    # that *that* Prefill actually served, and ``_aggregate_rows`` sums them, so
    # the rows must partition the class rate instead of repeating it.
    share = per_class / max(1, len(prefills))
    rows = []
    for index in range(args.classes):
        for prefill in prefills:
            rows.append({
                "class_id": f"probe-{index:03d}",
                "prefill_instance_id": prefill.instance_id,
                "arrival_rate_ewma": share,
                "hit_tokens_ewma": 0.0,
                "requested_tokens": args.prompt_tokens,
                "requested_tokens_ewma": args.prompt_tokens,
                "kv_bytes_per_request": args.prompt_tokens * kv_bytes_per_token,
            })

    solver = CapacityAwareFlowSolver(config)
    if args.work_scale != 1.0:
        overrides = {}
        for row in rows:
            for prefill in prefills:
                overrides[(prefill.instance_id, row["class_id"])] = args.work_scale
        assignments = solver.solve(rows, prefills, decodes, overrides)
    else:
        assignments = solver.solve(rows, prefills, decodes)

    diag = dict(solver.diagnostics)
    report = {
        "offered_rate": args.rate,
        "classes": args.classes,
        "prompt_tokens": args.prompt_tokens,
        "work_scale": args.work_scale,
        "kv_bytes_per_token": kv_bytes_per_token,
        "prefill_capacity": {p.instance_id: prefill_capacity.get(p.instance_id) for p in prefills},
        "declared_prefill_capacity": {p.instance_id: config.prefill_capacity.get(p.instance_id)
                                      for p in prefills},
        "objective": diag.get("objective"),
        "prefill_overflow": diag.get("prefill_overflow"),
        "decode_overflow": diag.get("decode_overflow"),
        "link_overflow": diag.get("link_overflow"),
        "plan_prefill": diag.get("plan_prefill") or diag.get("prefill"),
        "work": diag.get("work"),
        "assignments": [{"class_id": a.class_id, "prefill_id": a.prefill_id,
                         "decode_id": a.decode_id, "flow": a.flow, "cost": a.cost}
                        for a in assignments],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(f"offered {args.rate} req/s over {args.classes} classes "
          f"({args.prompt_tokens} prompt tokens, work_scale {args.work_scale})")
    print(f"  objective        : {report['objective']}")
    print(f"  prefill_overflow : {report['prefill_overflow']}")
    print(f"  decode_overflow  : {report['decode_overflow']}")
    print(f"  link_overflow    : {report['link_overflow']}")
    print(f"  prefill_capacity : {report['prefill_capacity']} "
          f"(declared {report['declared_prefill_capacity']})")
    for prefill_id, capacity in report["prefill_capacity"].items():
        ratio = (report.get("work") or {})
        samples = [v[prefill_id] for k, v in ratio.items() if str(prefill_id) in v]
        if samples and capacity:
            work = samples[0]
            print(f"  p{prefill_id}: work={work:.2f} -> effective capacity "
                  f"{capacity / max(1e-9, work):.2f} req/s for this prompt length")
    if args.flows:
        for assignment in sorted(report["assignments"], key=lambda a: -a["flow"]):
            print(f"  flow {assignment['class_id']:>10s} p{assignment['prefill_id']}"
                  f" -> d{assignment['decode_id']}: {assignment['flow']:.4f} req/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
