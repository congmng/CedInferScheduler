#!/usr/bin/env bash
# Profile P-15B on one hardware label: three block types, one run each.
#
# Why three runs instead of one: the profiler always builds
# ``num_hidden_layers=1``, and P-15B's three block types differ in shape
# (compressor 2048 / 1024 / none).  Each run therefore covers one type, and the
# canonical layer names were made distinct per type in
# ``profiler/models/p15b.yaml`` so the three bundles merge without colliding on
# a row key.
#
# Run inside the vLLM image with the repo mounted at /work:
#
#     PYTHONPATH=/work/deploy/vllm_p15b tests/run_p15b_profile.sh RTX4090 /out 0 2
#
# Arguments: <hardware> <out-root> [shard_index shard_count]
# Set BLOCKS to run a subset (default: r0,r4,r128) -- that is how the three
# types get spread over a host's two cards instead of run back to back.
#
# Set CATEGORIES to re-measure only some CSVs (dense / per_sequence /
# attention / moe).  Combined with the profiler's --force, this rewrites just
# those files inside an existing bundle: the 2026-09-22 catalog fix changed two
# dense rows, and
#
#     CATEGORIES=dense tests/run_p15b_profile.sh RTX4090 /out
#
# refreshes dense.csv for all three types in ~2 minutes each without touching
# a four-hour attention.csv.
set -euo pipefail

HARDWARE="${1:?usage: run_p15b_profile.sh <hardware> <out-root> [i n]}"
OUT_ROOT="${2:?usage: run_p15b_profile.sh <hardware> <out-root> [i n]}"
SHARD_INDEX="${3:-}"
SHARD_COUNT="${4:-}"

# Same recipe as the 2026-09-18 re-profile, so P-15B bundles are comparable
# with the Qwen3-8B ones in profiler/perf/.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
BLOCKS="${BLOCKS:-r0,r4,r128}"
VARIANT="${VARIANT:-bf16}"
# Empty = every category (the profiler's default).
CATEGORIES="${CATEGORIES:-}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SHARD_ARGS=()
[[ -n "$SHARD_INDEX" ]] && SHARD_ARGS+=(--shard "$SHARD_INDEX/$SHARD_COUNT")
CATEGORY_ARGS=()
[[ -n "$CATEGORIES" ]] && CATEGORY_ARGS+=(--categories "$CATEGORIES")

for BLOCK in ${BLOCKS//,/ }; do
    echo "### P-15B ${BLOCK} on ${HARDWARE}${SHARD_INDEX:+ (shard $SHARD_INDEX/$SHARD_COUNT)}"
    python3 -m profiler profile "casr/P15B-${BLOCK}" \
        --hardware "$HARDWARE" --tp 1 --dtype bfloat16 \
        --out-root "${OUT_ROOT}/${BLOCK}" --variant "$VARIANT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --attention-max-kv "$ATTENTION_MAX_KV" \
        --measurement-iterations "$MEASUREMENT_ITERATIONS" \
        --skip-skew --force "${SHARD_ARGS[@]}" "${CATEGORY_ARGS[@]}"
done

echo "### done: ${OUT_ROOT}/{r0,r4,r128}/${HARDWARE}/casr/P15B-*/${VARIANT}"
