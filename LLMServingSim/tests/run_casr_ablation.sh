#!/usr/bin/env bash
# Baseline and ablation matrix for the low/high/low elasticity workload.
# Cases:
#   fixed2      fixed 2P, no CASR
#   fixed1      fixed 1P, no CASR
#   dynamic_lp  full CASR with LP + resource lifecycle
#   dynamic_greedy full CASR with greedy flow solver
#   routing_only CASR routing/prefix state with no lifecycle/resources
#   no_prefix   full CASR against a no-reuse trace
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-casr-ablation.XXXXXX)}"
mkdir -p "$result_dir"

common=(--dataset workloads/casr_elasticity_low_high_low.jsonl --num-reqs 111
        --dtype bfloat16 --block-size 16 --log-level WARNING)

run_case() {
    local name="$1" config="$2" enable="$3" solver="$4"
    if [[ "$enable" == "yes" ]]; then
        python -m serving "${common[@]}" --cluster-config "$config" \
          --enable-casr --casr-solver "$solver" --casr-control-interval-ms 100 \
          --casr-state-output "$result_dir/$name.jsonl" --output "$result_dir/$name.csv" \
          --inputs-root "$result_dir/$name-inputs" >"$result_dir/$name.log" 2>&1
    else
        python -m serving "${common[@]}" --cluster-config "$config" \
          --no-enable-casr --output "$result_dir/$name.csv" \
          --inputs-root "$result_dir/$name-inputs" >"$result_dir/$name.log" 2>&1
    fi
}

run_case fixed2 configs/cluster/casr_hetero_rtx4090_rtxpro6000.json no lp
run_case fixed1 configs/cluster/casr_hetero_fixed1p_rtx4090_rtxpro6000.json no lp
run_case dynamic_lp configs/cluster/casr_hetero_rtx4090_rtxpro6000.json yes lp
run_case dynamic_greedy configs/cluster/casr_hetero_rtx4090_rtxpro6000.json yes greedy
run_case routing_only configs/cluster/casr_hetero_no_elasticity.json yes lp

python -m serving "${common[@]}" \
  --cluster-config configs/cluster/casr_hetero_rtx4090_rtxpro6000.json \
  --dataset workloads/casr_elasticity_no_prefix.jsonl --enable-casr --casr-solver lp \
  --casr-control-interval-ms 100 --casr-state-output "$result_dir/no_prefix.jsonl" \
  --output "$result_dir/no_prefix.csv" --inputs-root "$result_dir/no_prefix-inputs" \
  >"$result_dir/no_prefix.log" 2>&1

python tests/compare_elasticity.py --fixed2 "$result_dir/fixed2.csv" \
  --fixed1 "$result_dir/fixed1.csv" --dynamic "$result_dir/dynamic_lp.csv" \
  --dynamic-state "$result_dir/dynamic_lp.jsonl"
python tests/compare_elasticity.py --fixed2 "$result_dir/fixed2.csv" \
  --fixed1 "$result_dir/fixed1.csv" --dynamic "$result_dir/dynamic_greedy.csv" \
  --dynamic-state "$result_dir/dynamic_greedy.jsonl"
python tests/compare_elasticity.py --fixed2 "$result_dir/fixed2.csv" \
  --fixed1 "$result_dir/fixed1.csv" --dynamic "$result_dir/routing_only.csv" \
  --dynamic-state "$result_dir/routing_only.jsonl"
python tests/compare_elasticity.py --fixed2 "$result_dir/fixed2.csv" \
  --fixed1 "$result_dir/fixed1.csv" --dynamic "$result_dir/no_prefix.csv" \
  --dynamic-state "$result_dir/no_prefix.jsonl"
echo "artifacts: $result_dir"
