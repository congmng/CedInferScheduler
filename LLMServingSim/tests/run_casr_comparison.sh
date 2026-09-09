#!/usr/bin/env bash
# Reproducible baseline / greedy / LP comparison for CASR P/D scheduling.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-casr.XXXXXX)}"
mkdir -p "$result_dir"

cluster_config="${CLUSTER_CONFIG:-configs/cluster/casr_two_prefill_two_decode.json}"

common=(--cluster-config "$cluster_config"
        --dataset workloads/casr_hotspot_two_class.jsonl --num-reqs 8
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
