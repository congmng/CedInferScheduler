#!/usr/bin/env bash
# Refresh one TP degree's categories for P-15B without redoing tp1.
#
# ``profile`` always sweeps tp=1 as well, and a tp=1 attention sweep is ~4 h.
# ``slice`` refreshes a single (tp, category) pair and then replicates the
# tp_stable rows from tp1, which is what a tp=2/4 bundle needs.
#
#     TP=2 tests/run_p15b_slice.sh RTX4090 /out
#     TP=2 GROUPS=dense,per_sequence tests/run_p15b_slice.sh RTX4090 /out
#
# Run inside the vLLM image with the repo mounted at /work.
set -euo pipefail

HARDWARE="${1:?usage: run_p15b_slice.sh <hardware> <out-root>}"
OUT_ROOT="${2:?usage: run_p15b_slice.sh <hardware> <out-root>}"

TP="${TP:-2}"
BLOCKS="${BLOCKS:-r0,r4,r128}"
GROUPS="${GROUPS:-dense,per_sequence,attention}"
VARIANT="${VARIANT:-bf16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for BLOCK in ${BLOCKS//,/ }; do
    for GROUP in ${GROUPS//,/ }; do
        echo "### P-15B ${BLOCK} tp=${TP} ${GROUP} on ${HARDWARE}"
        python3 -m profiler slice "casr/P15B-${BLOCK}" \
            --hardware "$HARDWARE" --tp-refresh "$TP" --group "$GROUP" \
            --tp "1,${TP}" --dtype bfloat16 --variant "$VARIANT" \
            --out-root "$OUT_ROOT" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --attention-max-kv "$ATTENTION_MAX_KV" \
            --measurement-iterations "$MEASUREMENT_ITERATIONS" \
            --skip-skew --force
    done
done

echo "### done: ${OUT_ROOT}/{r0,r4,r128}/${HARDWARE}/casr/P15B-*/${VARIANT}/tp${TP}"
