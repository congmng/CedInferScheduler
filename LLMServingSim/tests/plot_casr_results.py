"""Plot CASR ablation and structural-decision artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_state(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--state", required=True,
                        help="JSONL state file for the timeline plot")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(summary)
    means = [summary[label]["latency"]["mean_ms"] for label in labels]
    p95s = [summary[label]["latency"]["p95_ms"] for label in labels]
    positions = list(range(len(labels)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar([pos - width / 2 for pos in positions], means, width, label="mean")
    axis.bar([pos + width / 2 for pos in positions], p95s, width, label="p95")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("CASR algorithm ablation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "latency_ablation.png", dpi=160)
    plt.close(figure)

    states = _load_state(Path(args.state))
    times = [row["time_ns"] / 1_000_000_000 for row in states]
    objectives = [row.get("solver", {}).get("objective", 0.0) for row in states]
    active_prefill = [sum(1 for item in row.get("instances", {}).values()
                          if item.get("admission_state") == "ACTIVE" and
                          item.get("resource_mem_gb", 0) < 64)
                      for row in states]
    actions = [(time, row.get("structural", {}).get("action"))
               for time, row in zip(times, states)
               if row.get("structural", {}).get("action") not in (None, "keep")]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, objectives, label="solver objective")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Objective")
    secondary = axis.twinx()
    secondary.step(times, active_prefill, where="post", color="tab:orange",
                   label="active Prefill")
    secondary.set_ylabel("Active Prefill count")
    for time, action in actions:
        axis.axvline(time, color="tab:red", linestyle="--", alpha=0.5)
        axis.text(time, axis.get_ylim()[1], action, rotation=90,
                  va="top", ha="right", fontsize=8)
    axis.set_title("CASR control timeline")
    figure.tight_layout()
    figure.savefig(output_dir / "control_timeline.png", dpi=160)
    plt.close(figure)
    print(f"plots: {output_dir}")


if __name__ == "__main__":
    main()
