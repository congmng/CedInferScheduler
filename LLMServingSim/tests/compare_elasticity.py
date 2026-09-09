"""Compare fixed and elastic CASR runs over low/high/low phases."""

from __future__ import annotations

import argparse
import csv
import json
import statistics


def _quantile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _phase(arrival_ns, boundaries):
    if arrival_ns < boundaries[0]:
        return "low"
    if arrival_ns < boundaries[1]:
        return "high"
    return "low"


def _latency_metrics(rows, boundaries):
    grouped = {"low": [], "high": []}
    for row in rows:
        grouped[_phase(int(row["arrival"]), boundaries)].append(
            float(row["latency"]) / 1_000_000)
    return {
        phase: {
            "requests": len(values),
            "mean_ms": statistics.fmean(values) if values else 0.0,
            "p95_ms": _quantile(values, 0.95) if values else 0.0,
        }
        for phase, values in grouped.items()
    }


def _static_resources(rows, gpu_count, memory_gb):
    horizon_ns = max(int(row["end_time"]) for row in rows)
    return {
        "avg_gpu_count": float(gpu_count),
        "peak_gpu_count": int(gpu_count),
        "avg_gpu_seconds": gpu_count * horizon_ns / 1_000_000_000,
        "avg_gpu_mem_gb": float(memory_gb),
        "peak_gpu_mem_gb": float(memory_gb),
        "startup_count": 0,
        "startup_cost_ms": 0.0,
        "release_count": 0,
        "reject_count": 0,
    }


def _dynamic_resources(rows, state_path):
    snapshots = []
    events = []
    with open(state_path, encoding="utf-8") as handle:
        for line in handle:
            snapshot = json.loads(line)
            snapshots.append(snapshot)
            events.extend(snapshot.get("lifecycle", []))
    snapshots.sort(key=lambda value: int(value["time_ns"]))
    if not snapshots:
        raise ValueError(f"no resource snapshots in {state_path}")
    gpu_counts = []
    mem_counts = []
    gpu_seconds = 0.0
    for current, following in zip(snapshots, snapshots[1:]):
        node_rows = current["resources"]["nodes"].values()
        gpu_count = sum(int(node["gpu_count"]) - int(node["free_gpu_count"])
                        for node in node_rows)
        mem_count = sum(float(node["used_gpu_mem_gb"]) for node in node_rows)
        duration = (int(following["time_ns"]) - int(current["time_ns"])) / 1_000_000_000
        gpu_seconds += gpu_count * max(0.0, duration)
        gpu_counts.append(gpu_count)
        mem_counts.append(mem_count)
    scale_out = [event for event in events if event.get("action") == "resource_acquire"
                 and event.get("reason") == "scale out"]
    releases = [event for event in events if event.get("action") == "resource_release"]
    rejects = [event for event in events if event.get("action") == "resource_reject"]
    startup_ns = int(snapshots[0]["resources"].get("startup_ns", 0))
    return {
        "avg_gpu_count": statistics.fmean(gpu_counts) if gpu_counts else 0.0,
        "peak_gpu_count": max(gpu_counts) if gpu_counts else 0,
        "avg_gpu_seconds": gpu_seconds,
        "avg_gpu_mem_gb": statistics.fmean(mem_counts) if mem_counts else 0.0,
        "peak_gpu_mem_gb": max(mem_counts) if mem_counts else 0.0,
        "startup_count": len(scale_out),
        "startup_cost_ms": len(scale_out) * startup_ns / 1_000_000,
        "release_count": len(releases),
        "reject_count": len(rejects),
    }


def report(label, rows, boundaries, resources):
    metrics = _latency_metrics(rows, boundaries)
    print(f"[{label}]")
    for phase in ("low", "high"):
        item = metrics[phase]
        print(f"{phase:5s} requests={item['requests']:3d} "
              f"mean_latency_ms={item['mean_ms']:.3f} "
              f"p95_latency_ms={item['p95_ms']:.3f}")
    print(f"resources avg_gpu={resources['avg_gpu_count']:.3f} "
          f"peak_gpu={resources['peak_gpu_count']} "
          f"gpu_seconds={resources['avg_gpu_seconds']:.3f} "
          f"avg_mem_gb={resources['avg_gpu_mem_gb']:.3f} "
          f"startup_count={resources['startup_count']} "
          f"startup_cost_ms={resources['startup_cost_ms']:.3f} "
          f"release_count={resources['release_count']} "
          f"reject_count={resources['reject_count']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", default="1000000000,3000000000")
    parser.add_argument("--fixed2", required=True)
    parser.add_argument("--fixed1", required=True)
    parser.add_argument("--dynamic", required=True)
    parser.add_argument("--dynamic-state", required=True)
    args = parser.parse_args()
    boundaries = [int(value) for value in args.boundaries.split(",")]
    fixed2 = _rows(args.fixed2)
    fixed1 = _rows(args.fixed1)
    dynamic = _rows(args.dynamic)
    report("fixed-2P", fixed2, boundaries, _static_resources(fixed2, 4, 168))
    report("fixed-1P", fixed1, boundaries, _static_resources(fixed1, 3, 144))
    report("dynamic-CASR", dynamic, boundaries,
           _dynamic_resources(dynamic, args.dynamic_state))


if __name__ == "__main__":
    main()
