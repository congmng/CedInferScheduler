#!/usr/bin/env python3
"""Report a structural-elasticity run: decisions, landing, per-phase latency.

``run_real_multidomain_comparison.sh`` already writes everything needed, but the
scale-out story is spread across four files:

* ``state-<policy>.json``   -- the controller's ``structural`` decision and the
  ``actions`` audit trail (including ``scale_out`` executions);
* ``metrics-<policy>.jsonl``-- per request Prefill/Decode landing and the
  measured segment times;
* ``client-<policy>.jsonl`` -- per request TTFT/TPOT/latency, indexed by trace
  row so the low/high/low phases can be separated;
* ``watch-*.jsonl`` (optional) -- the polled ``/routing-state`` timeline, which
  is the only place that shows *when* a scaled-out Prefill became ACTIVE.

Usage:
    python3 tests/analyze_elastic_scaleout.py --dir /tmp/casr-md/<tag> \
        --trace workloads/cnndm-long-elastic-real-qwen3-8b.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="result directory")
    parser.add_argument("--trace", default="", help="trace used for the run")
    parser.add_argument("--policies", default="",
                        help="comma separated subset (default: every summary-*.json)")
    parser.add_argument("--watch", default="",
                        help="optional /routing-state timeline JSONL")
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)
    return ordered[max(0, index)]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def phase_of_index(trace_path):
    phases = {}
    if not trace_path:
        return phases
    for index, line in enumerate(pathlib.Path(trace_path).open(encoding="utf-8")):
        if line.strip():
            phases[index] = json.loads(line).get("phase", "?")
    return phases


def report_policy(result_dir, policy, phases):
    out = {"policy": policy}
    summary_path = pathlib.Path(result_dir) / f"summary-{policy}.json"
    if summary_path.exists():
        out["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    client_path = pathlib.Path(result_dir) / f"client-{policy}.jsonl"
    if client_path.exists():
        rows = load_jsonl(client_path)
        ok = [row for row in rows if row.get("status") == 200]
        out["requests"] = len(rows)
        out["ok"] = len(ok)
        out["latency_mean_ms"] = round(sum(r["latency_ms"] for r in ok) / len(ok), 1) if ok else None
        out["latency_p50_ms"] = round(percentile([r.get("latency_ms") for r in ok], 0.5), 1)
        out["latency_p95_ms"] = round(percentile([r.get("latency_ms") for r in ok], 0.95), 1)
        out["ttft_p50_ms"] = round(percentile([r.get("ttft_ms") for r in ok], 0.5), 1)
        out["ttft_p95_ms"] = round(percentile([r.get("ttft_ms") for r in ok], 0.95), 1)
        slo = [r for r in ok if r.get("slo_ok") is not None]
        out["slo_ok_pct"] = (round(100.0 * sum(1 for r in slo if r["slo_ok"]) / len(slo), 1)
                             if slo else None)
        stamps = sorted(r["ts"] for r in rows if r.get("ts"))
        if stamps:
            window = stamps[-1] - stamps[0]
            out["wall_s"] = round(window, 1)
            out["goodput_rps"] = round(len(ok) / window, 2) if window > 0 else None
        out["glitch_gt5s"] = sum(1 for r in ok if (r.get("latency_ms") or 0) > 5000)
        per_phase = collections.defaultdict(list)
        for row in ok:
            per_phase[phases.get(row.get("index"), "?")].append(row)
        out["phases"] = {
            name: {
                "n": len(items),
                "latency_mean_ms": round(sum(i["latency_ms"] for i in items) / len(items), 1),
                "latency_p95_ms": round(percentile([i.get("latency_ms") for i in items], 0.95), 1),
                "ttft_mean_ms": round(sum(i["ttft_ms"] for i in items if i.get("ttft_ms")) /
                                      max(1, sum(1 for i in items if i.get("ttft_ms"))), 1),
            }
            for name, items in sorted(per_phase.items())
        }

    metrics_path = pathlib.Path(result_dir) / f"metrics-{policy}.jsonl"
    if metrics_path.exists():
        rows = load_jsonl(metrics_path)
        out["prefill_landing"] = dict(collections.Counter(r.get("prefill") for r in rows))
        out["decode_landing"] = dict(collections.Counter(r.get("decode") for r in rows))
        out["exchange"] = dict(collections.Counter(r.get("exchange") for r in rows))
        stamps = sorted(r["ts"] for r in rows if r.get("ts"))
        if stamps:
            start = stamps[0]
            buckets = collections.defaultdict(collections.Counter)
            for row in rows:
                if row.get("ts"):
                    buckets[int((row["ts"] - start) // 30)][row.get("prefill")] += 1
            out["landing_timeline_30s"] = {f"t+{k*30}s": dict(v) for k, v in sorted(buckets.items())}

    state_path = pathlib.Path(result_dir) / f"state-{policy}.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        casr = state.get("casr") or {}
        out["controller"] = {
            "version": casr.get("version"),
            "plan_versions": len(casr.get("plan_events") or []),
            "final_structural": casr.get("structural"),
            "actions": casr.get("actions") or [],
            "prefills": casr.get("prefills"),
        }
    return out


def watch_timeline(path, limit_each=3):
    rows = load_jsonl(path)
    if not rows:
        return {"samples": 0}
    start = rows[0]["ts"]
    transitions, seen, gains = [], {}, []
    for row in rows:
        states = tuple((p['id'], p['state']) for p in (row.get("prefills") or []))
        if states and states != seen.get("states"):
            transitions.append({"t_s": round(row["ts"] - start, 1), "states": states})
            seen["states"] = states
        structural = row.get("structural") or {}
        if structural.get("gain") is not None:
            gains.append({"t_s": round(row["ts"] - start, 1),
                          "action": structural.get("action"),
                          "gain": round(structural["gain"], 4),
                          "base_objective": round(structural.get("base_objective", 0.0), 3)})
    return {"samples": len(rows),
            "span_s": round(rows[-1]["ts"] - start, 1),
            "state_transitions": transitions,
            "gain_head": gains[:limit_each],
            "gain_tail": gains[-limit_each:]}


def main() -> int:
    args = parse_args()
    phases = phase_of_index(args.trace)
    available = sorted(pathlib.Path(p).name[len("summary-"):-len(".json")]
                       for p in glob.glob(str(pathlib.Path(args.dir) / "summary-*.json")))
    policies = [p.strip() for p in args.policies.split(",") if p.strip()] or available
    report = {"dir": args.dir, "trace": args.trace, "policies": {}}
    for policy in policies:
        report["policies"][policy] = report_policy(args.dir, policy, phases)
    if args.watch:
        report["watch"] = watch_timeline(args.watch)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
