#!/usr/bin/env bash
# Evaluate CASR under a constant-rate A -> B prefix hotspot drift.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-casr-drift.XXXXXX)}"
mkdir -p "$result_dir"
trace="workloads/.casr_hotspot_drift_runtime.jsonl"
trap 'rm -f "$trace" "$trace.meta.json"' EXIT

python -m workloads.generators casr \
  --output "$trace" --num-reqs 120 --seed 42 \
  --hotspot-mode drift --num-prefixes 2 --reuse-rate 1.0 \
  --prefix-len 64 --input-len 64 --output-len 64 \
  --arrival-model uniform --sps 30 --edge-fraction 0.5

python -m serving --dataset "$trace" --num-reqs 120 \
  --dtype bfloat16 --block-size 16 --log-level WARNING \
  --cluster-config configs/cluster/casr_hetero_rtx4090_rtxpro6000.json \
  --enable-casr --casr-solver lp --casr-control-interval-ms 100 \
  --casr-state-output "$result_dir/casr.jsonl" \
  --output "$result_dir/casr.csv" --inputs-root "$result_dir/inputs" \
  >"$result_dir/casr.log" 2>&1

python - "$trace" "$result_dir/casr.jsonl" <<'PY'
import json
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8") as handle:
    trace = [json.loads(line) for line in handle]
with open(sys.argv[2], encoding="utf-8") as handle:
    states = [json.loads(line) for line in handle]

actions = [row["structural"] for row in states
           if row.get("structural", {}).get("action") != "keep"]
print(json.dumps({
    "trace_hotspots_first_10": [row["hotspot_id"] for row in trace[:10]],
    "trace_hotspots_last_10": [row["hotspot_id"] for row in trace[-10:]],
    "class_counts_first_half": dict(Counter(row["hotspot_id"] for row in trace[:60])),
    "class_counts_second_half": dict(Counter(row["hotspot_id"] for row in trace[60:])),
    "solver_objective_first": states[0].get("solver", {}).get("objective"),
    "solver_objective_last": states[-1].get("solver", {}).get("objective"),
    "active_prefill_first": sorted({row.get("prefill_instance_id")
                                     for row in states[0].get("prefix_states", [])}),
    "active_prefill_last": sorted({row.get("prefill_instance_id")
                                    for row in states[-1].get("prefix_states", [])}),
    "selected_actions": actions,
}, ensure_ascii=False, indent=2))
print(f"artifacts: {sys.argv[2]}")
PY
