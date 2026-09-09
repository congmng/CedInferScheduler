#!/usr/bin/env bash
# Compare fixed 2P, fixed 1P, and resource-aware dynamic CASR over low/high/low load.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-casr-elasticity.XXXXXX)}"
mkdir -p "$result_dir"

common=(--dataset workloads/casr_elasticity_low_high_low.jsonl --num-reqs 111
        --dtype bfloat16 --block-size 16 --log-level WARNING)

python -m serving "${common[@]}" \
  --cluster-config configs/cluster/casr_hetero_rtx4090_rtxpro6000.json \
  --no-enable-casr --output "$result_dir/fixed2.csv" \
  --inputs-root "$result_dir/fixed2-inputs" >"$result_dir/fixed2.log" 2>&1

python -m serving "${common[@]}" \
  --cluster-config configs/cluster/casr_hetero_fixed1p_rtx4090_rtxpro6000.json \
  --no-enable-casr --output "$result_dir/fixed1.csv" \
  --inputs-root "$result_dir/fixed1-inputs" >"$result_dir/fixed1.log" 2>&1

python -m serving "${common[@]}" \
  --cluster-config configs/cluster/casr_hetero_rtx4090_rtxpro6000.json \
  --enable-casr --casr-solver lp --casr-control-interval-ms 100 \
  --casr-state-output "$result_dir/dynamic.jsonl" \
  --output "$result_dir/dynamic.csv" \
  --inputs-root "$result_dir/dynamic-inputs" >"$result_dir/dynamic.log" 2>&1

python tests/compare_elasticity.py \
  --fixed2 "$result_dir/fixed2.csv" \
  --fixed1 "$result_dir/fixed1.csv" \
  --dynamic "$result_dir/dynamic.csv" \
  --dynamic-state "$result_dir/dynamic.jsonl"
echo "artifacts: $result_dir"
