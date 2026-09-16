#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-qwen3-8b-three-domain.XXXXXX)}"
num_reqs="${NUM_REQS:-80}"
dataset="${DATASET:-workloads/casr_hetero_hot_cold.jsonl}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-1024}"
mkdir -p "$result_dir"

common=(
  --cluster-config configs/cluster/casr_real_qwen3_8b_three_domain.json
  --dataset "$dataset"
  --num-reqs "$num_reqs"
  --dtype bfloat16
  --block-size 16
  --max-num-seqs 64
  --max-num-batched-tokens "$max_num_batched_tokens"
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

python tests/compare_casr.py "$result_dir/static.csv" "$result_dir/greedy.csv"
python tests/compare_casr.py "$result_dir/static.csv" "$result_dir/lp.csv"
echo "artifacts: $result_dir"
