#!/usr/bin/env bash
# Compare a static pool with CASR structural-gain evaluation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-casr-structural.XXXXXX)}"
mkdir -p "$result_dir"

common=(--dataset workloads/casr_elasticity_low_high_low.jsonl --num-reqs 111
        --dtype bfloat16 --block-size 16 --log-level WARNING)
config=configs/cluster/casr_structural_mismatch.json

python -m serving "${common[@]}" --cluster-config "$config" \
  --no-enable-casr --output "$result_dir/static.csv" \
  --inputs-root "$result_dir/static-inputs" >"$result_dir/static.log" 2>&1

python -m serving "${common[@]}" --cluster-config "$config" \
  --enable-casr --casr-solver greedy --casr-control-interval-ms 100 \
  --casr-state-output "$result_dir/casr.jsonl" --output "$result_dir/casr.csv" \
  --inputs-root "$result_dir/casr-inputs" >"$result_dir/casr.log" 2>&1

python - "$result_dir/casr.jsonl" <<'PY'
import json
import sys
from collections import Counter

actions = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        decision = row.get("structural", {})
        if decision.get("action") != "keep":
            actions.append({
                "time_ns": row.get("time_ns"),
                "action": decision.get("action"),
                "mode": decision.get("mode"),
                "gain": decision.get("gain"),
                "base_objective": decision.get("base_objective"),
                "candidate_objective": decision.get("candidate_objective"),
                "wanted_ids": decision.get("wanted_ids"),
            })
print(json.dumps({"action_counts": Counter(item["action"] for item in actions),
                  "selected_actions": actions}, ensure_ascii=False, indent=2))
print(f"artifacts: {sys.argv[1]}")
PY
