#!/usr/bin/env bash
# Compare Static routing, CASR greedy, and CASR LP on the real-profile Qwen3
# topology represented by four RTX 5090 and four RTX 4090 GPUs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-qwen3-hetero.XXXXXX)}"
num_reqs="${NUM_REQS:-80}"
mkdir -p "$result_dir"

common=(
  --cluster-config configs/cluster/casr_real_qwen3_tp4_heterogeneous.json
  --dataset workloads/casr_hetero_hot_cold.jsonl
  --num-reqs "$num_reqs"
  --dtype bfloat16
  --block-size 16
  --max-num-seqs 64
  --max-num-batched-tokens 1024
  --log-level WARNING
)

python -m serving "${common[@]}" --no-enable-casr \
  --output "$result_dir/static.csv" --inputs-root "$result_dir/static-inputs" \
  >"$result_dir/static.log" 2>&1

python -m serving "${common[@]}" --enable-casr --casr-solver greedy \
  --casr-control-interval-ms 100 --output "$result_dir/greedy.csv" \
  --casr-state-output "$result_dir/greedy.jsonl" --inputs-root "$result_dir/greedy-inputs" \
  >"$result_dir/greedy.log" 2>&1

python -m serving "${common[@]}" --enable-casr --casr-solver lp \
  --casr-control-interval-ms 100 --output "$result_dir/lp.csv" \
  --casr-state-output "$result_dir/lp.jsonl" --inputs-root "$result_dir/lp-inputs" \
  >"$result_dir/lp.log" 2>&1

echo "Static vs CASR greedy"
python tests/compare_casr.py "$result_dir/static.csv" "$result_dir/greedy.csv"
echo
echo "Static vs CASR LP"
python tests/compare_casr.py "$result_dir/static.csv" "$result_dir/lp.csv"
echo
echo "artifacts: $result_dir"
