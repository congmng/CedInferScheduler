#!/usr/bin/env bash
set -euo pipefail

HARDWARE="${HARDWARE:?set HARDWARE to the measured GPU name}"
TP_DEGREES="${TP_DEGREES:-1,2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

args=(
  Qwen/Qwen3-8B
  --hardware "$HARDWARE"
  --tp "$TP_DEGREES"
  --dtype bfloat16
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --attention-max-kv "$ATTENTION_MAX_KV"
  --attention-chunk-factor "${ATTENTION_CHUNK_FACTOR:-2.0}"
  --attention-kv-factor "${ATTENTION_KV_FACTOR:-2.0}"
  --measurement-iterations "$MEASUREMENT_ITERATIONS"
)

[[ -n "${SKIP_SKEW:-}" ]] && args+=(--skip-skew)
[[ -n "${FORCE:-}" ]] && args+=(--force)
[[ -n "${VARIANT:-}" ]] && args+=(--variant "$VARIANT")

python3 -m profiler profile "${args[@]}"
