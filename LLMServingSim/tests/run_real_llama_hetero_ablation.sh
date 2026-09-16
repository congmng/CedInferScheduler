#!/usr/bin/env bash
# Run controlled ablations for the measured RTX5090/RTX4090 Llama topology.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/llmservingsim-llama-ablation.XXXXXX)}"
num_reqs="${NUM_REQS:-80}"
mkdir -p "$result_dir"

base_config="configs/cluster/casr_real_llama_heterogeneous.json"
variant_dir="configs/cluster/.real_llama_ablation"
mkdir -p "$variant_dir"
trap 'rm -rf "$variant_dir"' EXIT
common=(
  --dataset workloads/casr_hetero_hot_cold.jsonl
  --num-reqs "$num_reqs"
  --dtype bfloat16
  --block-size 16
  --max-num-seqs 64
  --max-num-batched-tokens 1024
  --log-level WARNING
)

make_variant() {
  local mode="$1" output="$2"
  python3 - "$base_config" "$output" "$mode" <<'PY'
import json
import sys

source, target, mode = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)

if mode == "no_cache":
    for node in config["nodes"]:
        for instance in node["instances"]:
            instance["enable_prefix_caching"] = False
elif mode == "no_network":
    config["link_bw"] = 1_000_000
    config["link_latency"] = 0
    for link in config["casr"]["shared_links"]:
        link["capacity"] = 1_000_000_000
elif mode == "no_elasticity":
    config["casr"]["lifecycle"]["max_active_prefill"] = 1
else:
    raise SystemExit(f"unknown variant: {mode}")

with open(target, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
}

run_static() {
  python -m serving "${common[@]}" --cluster-config "$base_config" \
    --no-enable-casr --output "$result_dir/static.csv" \
    --inputs-root "$result_dir/static-inputs" >"$result_dir/static.log" 2>&1
}

run_casr() {
  local name="$1" config="$2"
  local args=(python -m serving "${common[@]}" --cluster-config "$config"
    --enable-casr --casr-solver lp --casr-control-interval-ms 100
    --casr-state-output "$result_dir/$name.jsonl" --output "$result_dir/$name.csv"
    --inputs-root "$result_dir/$name-inputs")
  "${args[@]}" >"$result_dir/$name.log" 2>&1
}

run_static
run_casr casr_lp_cached "$base_config"
make_variant no_cache "$variant_dir/no-cache.json"
python -m serving "${common[@]}" --cluster-config "$variant_dir/no-cache.json" \
  --no-enable-casr --output "$result_dir/static_no_cache.csv" \
  --inputs-root "$result_dir/static-no-cache-inputs" \
  >"$result_dir/static_no_cache.log" 2>&1
run_casr casr_lp_no_cache "$variant_dir/no-cache.json"

make_variant no_network "$variant_dir/no-network.json"
run_casr casr_lp_no_network "$variant_dir/no-network.json"

make_variant no_elasticity "$variant_dir/no-elasticity.json"
run_casr casr_lp_no_elasticity "$variant_dir/no-elasticity.json"

for name in casr_lp_cached casr_lp_no_cache casr_lp_no_network casr_lp_no_elasticity; do
  echo "Static vs $name"
  python tests/compare_casr.py "$result_dir/static.csv" "$result_dir/$name.csv"
  echo
done
echo "No-cache Static vs casr_lp_no_cache"
python tests/compare_casr.py "$result_dir/static_no_cache.csv" "$result_dir/casr_lp_no_cache.csv"
echo
echo "artifacts: $result_dir"
