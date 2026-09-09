"""Run the CASR algorithm ablation matrix on one fixed workload."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


CASES = (
    ("static", None),
    ("ours", "ours"),
    ("no_cache_capacity", "no_cache_capacity"),
    ("no_warm_counterfactual", "no_warm_counterfactual"),
    ("no_structural_gain", "no_structural_gain"),
    ("no_network", "no_network"),
    ("no_hysteresis", "no_hysteresis"),
)


def _config_for(base, variant):
    config = json.loads(json.dumps(base))
    casr = config.setdefault("casr", {})
    if variant == "no_cache_capacity":
        casr["use_cache_capacity"] = False
    elif variant == "no_warm_counterfactual":
        casr.setdefault("structural", {})["enable_warm_counterfactual"] = False
    elif variant == "no_structural_gain":
        casr.setdefault("structural", {})["enabled"] = False
    elif variant == "no_network":
        casr["network_weight"] = 0.0
    elif variant == "no_hysteresis":
        structural = casr.setdefault("structural", {})
        structural["gain_threshold_abs"] = 0.0
        structural["gain_threshold_rel"] = 0.0
        structural["dwell_time_ms"] = 0.0
    return config


def _latency_summary(path):
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["latency"]) / 1_000_000)
    values.sort()
    if not values:
        return {"requests": 0, "mean_ms": 0.0, "p95_ms": 0.0}
    return {
        "requests": len(values),
        "mean_ms": sum(values) / len(values),
        "p95_ms": values[min(len(values) - 1, int(0.95 * (len(values) - 1)))],
    }


def _state_summary(path):
    actions = []
    warmups = 0
    warmup_requested_bytes = 0
    warmup_new_bytes = 0
    objectives = []
    resource_actions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            structural = row.get("structural", {})
            if structural.get("action") not in (None, "keep"):
                actions.append(structural)
            for warmup in row.get("warmups", ()):
                warmups += 1
                warmup_requested_bytes += int(warmup.get("requested_bytes", 0))
                warmup_new_bytes += int(warmup.get("bytes", 0))
            objective = row.get("solver", {}).get("objective")
            if objective is not None:
                objectives.append(float(objective))
            resource_actions.extend(event.get("action") for event in row.get("lifecycle", ()))
    return {
        "action_counts": dict(Counter(item["action"] for item in actions)),
        "selected_actions": actions,
        "warmup_count": warmups,
        "warmup_requested_bytes": warmup_requested_bytes,
        "warmup_new_bytes": warmup_new_bytes,
        "mean_solver_objective": sum(objectives) / len(objectives) if objectives else 0.0,
        "resource_action_counts": dict(Counter(resource_actions)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cluster/casr_hetero_rtx4090_rtxpro6000.json")
    parser.add_argument("--dataset", default="workloads/casr_elasticity_low_high_low.jsonl")
    parser.add_argument("--num-reqs", type=int, default=111)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    with (root / args.config).open(encoding="utf-8") as handle:
        base = json.load(handle)
    summary = {}

    for name, variant in CASES:
        config_path = root / "configs" / f".casr_ablation_{os.getpid()}_{name}.json"
        output_path = result_dir / f"{name}.csv"
        state_path = result_dir / f"{name}.jsonl"
        config_path.write_text(json.dumps(_config_for(base, variant), indent=2), encoding="utf-8")
        command = [
            sys.executable, "-m", "serving",
            "--dataset", args.dataset,
            "--num-reqs", str(args.num_reqs),
            "--dtype", "bfloat16",
            "--block-size", "16",
            "--log-level", "WARNING",
            "--cluster-config", str(config_path.relative_to(root)),
            "--output", str(output_path),
            "--inputs-root", str(result_dir / f"{name}-inputs"),
        ]
        if variant is not None:
            command.extend([
                "--enable-casr", "--casr-control-interval-ms", "100",
                "--casr-state-output", str(state_path),
            ])
        else:
            command.append("--no-enable-casr")
        try:
            with (result_dir / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL,
                               stderr=log)
        finally:
            config_path.unlink(missing_ok=True)
        item = {"latency": _latency_summary(output_path)}
        if variant is not None:
            item["state"] = _state_summary(state_path)
        summary[name] = item

    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"artifacts: {result_dir}")


if __name__ == "__main__":
    main()
