#!/usr/bin/env bash
# Heterogeneous CASR advantage reproduction:
#   2x RTX4090 Prefill, 1x RTXPRO6000 fast Decode, 1x RTX4090 slow Decode.
#   Zipf hot/cold request classes with identical full-input prefixes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-casr-hetero.XXXXXX)}"
mkdir -p "$result_dir"

common=(--cluster-config configs/cluster/casr_hetero_rtx4090_rtxpro6000.json
        --dataset workloads/casr_hetero_hot_cold.jsonl --num-reqs 80
        --dtype bfloat16 --block-size 16 --log-level WARNING)

python -m serving "${common[@]}" --no-enable-casr \
  --output "$result_dir/baseline.csv" --inputs-root "$result_dir/baseline-inputs" \
  >"$result_dir/baseline.log" 2>&1
python -m serving "${common[@]}" --enable-casr --casr-solver greedy \
  --casr-control-interval-ms 100 --output "$result_dir/greedy.csv" \
  --casr-state-output "$result_dir/greedy.jsonl" --inputs-root "$result_dir/greedy-inputs" \
  >"$result_dir/greedy.log" 2>&1
python -m serving "${common[@]}" --enable-casr --casr-solver lp \
  --casr-control-interval-ms 100 --output "$result_dir/lp.csv" \
  --casr-state-output "$result_dir/lp.jsonl" --inputs-root "$result_dir/lp-inputs" \
  >"$result_dir/lp.log" 2>&1

echo "baseline vs greedy"
python tests/compare_casr.py "$result_dir/baseline.csv" "$result_dir/greedy.csv"
echo
echo "baseline vs LP"
python tests/compare_casr.py "$result_dir/baseline.csv" "$result_dir/lp.csv"
echo
echo "artifacts: $result_dir"
