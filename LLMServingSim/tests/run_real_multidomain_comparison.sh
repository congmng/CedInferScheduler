#!/usr/bin/env bash
# Compare real-system routing policies on the multi-domain LMCache P/D topology.
#
# Each policy is measured from a cold cache: the P/D containers are restarted
# before the run so prefix-cache state cannot leak between policies.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_dir="${1:-$(mktemp -d /tmp/casr-md-compare.XXXXXX)}"
# Two concurrent runs tear down each other's router container
# (``docker rm -f casr-md-router``) and each other's P/D instances, which shows
# up as "Connection refused" inside the correctness gate and silently mixes
# result directories -- measured 2026-09-13.  Take an exclusive lock so a
# second invocation fails fast instead of corrupting both runs.
exec 9>"/tmp/casr-md-compare.lock"
if ! flock -n 9; then
  echo "another comparison run is active (lock /tmp/casr-md-compare.lock)" >&2
  echo "wait for it, or check for stale run_real_multidomain_comparison.sh" >&2
  exit 1
fi
# Re-using a result directory silently mixes runs: the client appends to
# ``client-<policy>.jsonl`` / ``metrics-<policy>.jsonl``, and the aggregator
# keeps the *first* row per request id, i.e. the stale one.  Two runs that
# share a directory therefore produce numbers that belong to neither.  Refuse
# unless the caller explicitly opts in.
if compgen -G "$result_dir/metrics-*.jsonl" >/dev/null \
    && [[ "${ALLOW_RESULT_REUSE:-0}" != "1" ]]; then
  echo "result dir '$result_dir' already contains metrics for:" >&2
  ls "$result_dir"/metrics-*.jsonl >&2
  echo "use a fresh directory (recommended) or set ALLOW_RESULT_REUSE=1" >&2
  exit 1
fi
num_reqs="${NUM_REQS:-96}"
num_clients="${NUM_CLIENTS:-4}"
prefix_tokens="${PREFIX_TOKENS:-512}"
output_tokens="${OUTPUT_TOKENS:-16}"
concurrency="${CONCURRENCY:-8}"
# ``CLIENT_PACING=closed`` is the real client's semaphore; ``trace`` submits
# purely on the trace's clock, which is what makes two policies comparable (see
# tests/real_dataset_client.py).
policies="${POLICIES:-rr load casr}"
# Hotspot shape: ``HOT_PREFIXES`` below ``NUM_CLIENTS`` makes several clients
# reuse one cached prompt (single-Prefill KV-egress congestion), and
# ``DRIFT_AFTER_FRACTION`` moves every client to a brand-new prompt part-way
# through the run (hotspot migration).
hot_prefixes="${HOT_PREFIXES:-0}"
drift_after_fraction="${DRIFT_AFTER_FRACTION:-0.0}"
# Classic-dataset replay.  When ``TRACE`` points at a generator trace with
# ``input_text`` (produced by ``--emit-text``), every policy replays that trace
# through ``real_dataset_client.py`` instead of the synthetic-prefix client.
trace="${TRACE:-}"
max_output_tokens="${MAX_OUTPUT_TOKENS:-16}"
time_scale="${TIME_SCALE:-1.0}"
# ``STREAM=1`` replays through SSE so the client can record TTFT/TPOT (and the
# router can attach them to ``metrics-<policy>.jsonl``).  Off by default so the
# trajectory of earlier non-streaming rounds stays comparable.
stream="${STREAM:-0}"
stream_args=()
if [[ "$stream" == "1" ]]; then
  stream_args+=(--stream)
fi
if [[ "${IGNORE_TRACE_SLO:-0}" == "1" ]]; then
  stream_args+=(--ignore-trace-slo)
fi
# The router needs the shared CASR package plus an OR-Tools-enabled Python.
# ``deploy/real_lmcache_pd/router_image.Dockerfile`` builds that image.
router_image="${ROUTER_IMAGE:-casr-router:latest}"
config="${ROUTER_CONFIG:-$repo_root/deploy/real_lmcache_pd/router_config.json}"
mkdir -p "$result_dir"
# Transient topology outages are a run parameter, not checked-in state: the
# caller passes ``DISABLED_INSTANCES=p3090b,p4090`` and every consumer below
# (health gate, warmup pairs, launcher, router) honours it.
disabled_instances="${DISABLED_INSTANCES:-}"
export DISABLED_INSTANCES="$disabled_instances"
# LMCache's PD sender PUSHes a "KV has landed" notification to the
# orchestrator, and the router can bind that PULL socket (PD_PROXY_PORT) to
# hold the Decode leg back until it arrives.  Leave both unset for now: on
# 2026-09-13 the notification path made the transfer happen but the Decode
# still produced garbage from the received KV (docs/五台异构实验环境部署记录.md,
# "KV 搬运本身没有生效").  With no proxy endpoint the sender's LMCache engine
# fails init, the Decode recomputes locally, and the answers are correct.
export PD_PROXY_HOST="${PD_PROXY_HOST:-${ROUTER_HOST:-10.212.67.167}}"
export PD_PROXY_PORT="${PD_PROXY_PORT:-}"
# Which KV transport the instances run and the router speaks.  ``native`` is
# vLLM's own NixlConnector -- measured working on 2026-09-13 (real transfers of
# 168 MB at ~276 MB/s, correct answers), while LMCache's PD backend delivered
# KV the Decode could not use.  Set to ``lmcache`` to go back.
export KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-native}"
export PD_TRANSFER_BACKEND="${PD_TRANSFER_BACKEND:-$KV_TRANSFER_BACKEND}"
# Instances that must keep their existing container (a node whose NVIDIA
# userspace/kernel modules drifted apart can serve from an already-running
# container but cannot create a new one).  They still have to pass the health
# gate below.
preserve_instances="${PRESERVE_INSTANCES:-}"
export PRESERVE_INSTANCES="$preserve_instances"
# Prefills whose container must be *stopped* after the warmup and before the
# router starts.  A topology whose every enabled Prefill is running leaves the
# structural evaluator with no ``+P`` candidate at all -- ``inactive`` is empty,
# the only counterfactual is ``-P``, and its gain is never positive -- so a
# scale-out experiment has to start from an intentionally shrunk pool plus one
# stopped spare.  Comma separated Prefill ids (``p5090,p4090``).  The instance
# stays configured/enabled, hence eligible, so this is exactly the "warm spare"
# case the lifecycle is built for.
stop_before_run="${STOP_BEFORE_RUN:-}"
export STOP_BEFORE_RUN="$stop_before_run"
python3 - "$config" "$result_dir/topology.json" <<'PY'
import json, os, sys
config = json.load(open(sys.argv[1]))
disabled = {t.strip() for t in os.environ.get("DISABLED_INSTANCES", "").split(",") if t.strip()}
def enabled(item):
    return bool(item.get("enabled", True)) and item["id"] not in disabled
json.dump({"prefills": [i for i in config["prefills"] if enabled(i)],
           "decodes": [i for i in config["decodes"] if enabled(i)]},
          open(sys.argv[2], "w"), indent=2)
PY
topology="$result_dir/topology.json"
python3 - "$result_dir/run-config.json" <<'PY'
import json, os, sys
keys = ("NUM_REQS", "NUM_CLIENTS", "PREFIX_TOKENS", "OUTPUT_TOKENS", "CONCURRENCY",
        "POLICIES", "HOT_PREFIXES", "DRIFT_AFTER_FRACTION", "PD_BUFFER_SIZE",
        "TRACE", "MAX_OUTPUT_TOKENS", "TIME_SCALE", "STREAM",
        "SLO_TTFT_MS", "SLO_TPOT_MS", "SLO_PENALTY",
        "OVERFLOW_PENALTY", "PLAN_TTL_S", "CONTROL_INTERVAL_S",
        "DISABLED_INSTANCES", "PRESERVE_INSTANCES",
        "STOP_BEFORE_RUN", "LOCAL_PREFILL", "PD_TRANSFER_BACKEND",
        "REQUEST_TIMEOUT_S", "CLIENT_PACING",
        "WARMUP_S", "STRUCTURAL_STARTUP_COST", "STRUCTURAL_EVAL_WINDOW_MS",
        "STRUCTURAL_DWELL_MS",
        "STRUCTURAL_STARTUP_S", "STRUCTURAL_HOLDING_COST",
        "STRUCTURAL_IDLE_COST_FRACTION", "MAX_ACTIVE_PREFILL",
        "LOCAL_PREFILL_QUEUE_WEIGHT", "LOCAL_PREFILL_QUEUE_CAP",
        "LOCAL_PREFILL_BASELINES",
        "PROMPT_TOKEN_CALIBRATION_REQS",
        "EQUALIZE_PREFILL_CAPACITY", "EQUALIZE_DECODE_CAPACITY",
        "PREFILL_CLASS_LIMIT")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({key: os.environ.get(key) for key in keys}, handle, indent=2)
PY

check_ray_topology() {
  # Ray is the control plane: refuse to run an experiment whose configured
  # domains are not the ones registered as ``domain:<name>`` labels.  Health is
  # skipped here because the P/D containers are (re)started further down.
  echo "-- Ray domain gate --"
  python3 deploy/ray/domain_inventory.py --check --skip-health
  echo
}

health_urls() {
  python3 -c "
import json;d=json.load(open('$topology'))
for item in (*d['prefills'], *d['decodes']):
    print(f\"http://{item['host']}:{item['port']}/health\")"
}

count_healthy() {
  local ready=0 code
  for url in $(health_urls); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" || true)
    [[ "$code" == "200" ]] && ready=$((ready + 1))
  done
  echo "$ready"
}

wait_healthy() {
  local expected="$1" attempt ready
  for attempt in $(seq 1 36); do
    ready=$(count_healthy)
    [[ "$ready" -eq "$expected" ]] && return 0
    sleep 5
  done
  return 1
}

restart_pd() {
  local expected
  # Only enabled instances are launched, so the readiness gate must ignore
  # entries that are deliberately disabled (e.g. a contended shared node).
  expected=$(python3 -c "import json;d=json.load(open('$topology'));print(len(d['prefills'])+len(d['decodes']))")
  # A restarted Decode can race the previous container's GPU memory release
  # (vLLM then aborts with "Available KV cache memory: -x GiB").  Give the
  # topology three launches before declaring the run broken.
  local launch
  for launch in 1 2 3; do
    CONFIG="$config" bash deploy/real_lmcache_pd/start_multidomain_pd.sh >/dev/null
    wait_healthy "$expected" && return 0
    echo "  P/D not healthy after launch $launch ($(count_healthy)/$expected); retrying" >&2
    sleep 10
  done
  echo "P/D instances did not become healthy" >&2
  return 1
}

warmup_instances() {
  # Each (Prefill, Decode) pair compiles its remote-KV code path on first use
  # with a fixed ~50s cost.  Exercise every pair once before measuring so the
  # comparison reflects steady-state routing instead of cold-start artifacts.
  #
  # LMCache occasionally deadlocks that first handshake (both EngineCores spin
  # at 100% and the request never returns), which used to make the whole batch
  # hang forever.  Bound each pair so a stalled handshake aborts the batch and
  # surfaces instead of masking the run.
  local warmup_timeout="${WARMUP_TIMEOUT:-300}"
  local rc
  while IFS='|' read -r p d ip ap qp; do
    rc=0
    timeout "$warmup_timeout" python3 tests/calibrate_real_pd.py --prefill "$p" --decode "$d" \
      --init-port "$ip" --alloc-port "$ap" --query-port "$qp" \
      --num-reqs 2 --output-tokens 4 >/dev/null || rc=$?
    if [[ "$rc" == "124" ]]; then
      echo "warmup stalled for $p -> $d after ${warmup_timeout}s (LMCache handshake deadlock)" >&2
      return 1
    fi
    [[ "$rc" == "0" ]] || echo "warmup for $p -> $d exited $rc" >&2
    echo "  warmed $p -> $d"
  done < <(python3 -c "
import json;d=json.load(open('$topology'))
for p in d['prefills']:
    for x in d['decodes']:
        print(f\"{p['host']}:{p['port']}|{x['host']}:{x['port']}|{x['init_port']}|{x['alloc_port']}|{x['query_port']}\")")
}

warmup_all_pairs() {
  # A stalled handshake leaves both engines spinning, so the only recovery is a
  # fresh container pair.  Restart and re-warm *every* pair (not just the
  # stalled one) because a restart drops the warm state of the pairs that had
  # already succeeded.  The launch itself is retried here too: a shared node can
  # lose a GPU between two policies, and ``restart_pd`` already tolerates the
  # transient out-of-memory race on the first launch.
  local attempts="${WARMUP_RESTARTS:-3}" attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if restart_pd && warmup_instances; then
      return 0
    fi
    if ((attempt < attempts)); then
      echo "retrying P/D launch + warmup (attempt $((attempt + 1))/$attempts)" >&2
      sleep 10
    fi
  done
  echo "P/D did not come up and warm up after $attempts attempts" >&2
  return 1
}

stop_configured_prefills() {
  # Shrink the running pool after the warmup: see ``STOP_BEFORE_RUN`` above.
  # The containers are stopped (not removed) so the router's ``+P`` can start
  # the very same container again -- a vLLM restart, not a cold image pull.
  [[ -n "$stop_before_run" ]] || return 0
  echo "-- shrinking Prefill pool (stop: $stop_before_run) --"
  python3 - "$config" "$stop_before_run" <<'PY'
import json
import subprocess
import sys

config_path, raw = sys.argv[1], sys.argv[2]
config = json.load(open(config_path))
prefix = str((config.get("casr") or {}).get("container_prefix", "casr-md-"))
hosts = config.get("hosts", {}) or {}
specs = {str(item["id"]): item for item in config.get("prefills", ())}
failed = []
for target in [t.strip() for t in raw.split(",") if t.strip()]:
    spec = specs.get(target)
    if spec is None:
        print(f"  {target}: not a configured Prefill", file=sys.stderr)
        failed.append(target)
        continue
    domain = str(spec.get("domain", spec.get("host", "")))
    host = hosts.get(domain, {}) or {}
    ssh_target = str(host.get("ssh", ""))
    if not ssh_target:
        print(f"  {target}: no ssh target for domain {domain}", file=sys.stderr)
        failed.append(target)
        continue
    name = f"{prefix}{target}"
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-p", str(int(host.get("ssh_port", 22) or 22)), ssh_target,
            f"docker stop -t 10 {name} >/dev/null 2>&1; "
            f"docker inspect -f '{{{{.State.Status}}}}' {name}"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(f"  {name}: ssh timed out", file=sys.stderr)
        failed.append(target)
        continue
    status = (done.stdout or "").strip()
    if done.returncode != 0 or status != "exited":
        detail = (done.stderr or done.stdout or "").strip()[-200:]
        print(f"  {name}: stop failed ({status or 'no status'}) {detail}",
              file=sys.stderr)
        failed.append(target)
        continue
    print(f"  {name}: {status} (container kept for +P)")
sys.exit(1 if failed else 0)
PY
}

start_router() {
  local policy="$1"
  local scale_backend=""
  local -a mounts=(-v "$repo_root/deploy/real_lmcache_pd:/router-config:ro"
                   -v "$repo_root/serving:/casr-serving:ro"
                   -v "$result_dir:/results")
  local -a args
  if [[ "$policy" == "casr_full" ]]; then
    # +P/-P starts/stops real Prefill containers, including containers on the
    # other domains, so the router needs ssh credentials to every worker host.
    scale_backend="docker"
    mounts+=(-v "$HOME/.ssh:/host-ssh:ro")
    local bootstrap="install -m 700 -d /root/.ssh && cp -a /host-ssh/. /root/.ssh/ &&"
    bootstrap+=" chown -R root:root /root/.ssh && chmod 700 /root/.ssh &&"
    bootstrap+=" chmod 600 /root/.ssh/config /root/.ssh/known_hosts /root/.ssh/id_* 2>/dev/null;"
    bootstrap+=" exec python3 /router-config/disagg_router.py"
    bootstrap+=" --config /router-config/$(basename "$config") --policy $policy"
    bootstrap+=" --scale-backend $scale_backend --port 9000"
    bootstrap+=" --metrics /results/metrics-$policy.jsonl"
    # Control-plane snapshot per tick, in the simulator's schema, so the run
    # can be replayed offline (`--casr-state-output` on the simulator side).
    bootstrap+=" --state-output /results/state-$policy.jsonl"
    args=(--entrypoint bash "$router_image" -c "$bootstrap")
  else
    args=(--entrypoint python3 "$router_image" /router-config/disagg_router.py
          --config "/router-config/$(basename "$config")" --policy "$policy"
          --scale-backend "$scale_backend" --port 9000
          --metrics "/results/metrics-$policy.jsonl"
          --state-output "/results/state-$policy.jsonl")
  fi
  if ! docker image inspect "$router_image" >/dev/null 2>&1; then
    echo "router image $router_image is missing; build it with:" >&2
    echo "  docker build -f deploy/real_lmcache_pd/router_image.Dockerfile -t $router_image deploy/real_lmcache_pd" >&2
    return 1
  fi
  docker rm -f casr-md-router >/dev/null 2>&1 || true
  # ``LOCAL_PREFILL_BASELINES`` defaults to ``auto``: the baselines get the
  # same "recompute locally vs hand the KV over" choice the CASR policies have
  # had since 2026-09-13.  Set it to ``never`` to reproduce the earlier
  # asymmetry (baselines pinned to the handoff path) -- a valid A/B, but not a
  # defensible default, because a local recompute is 6-14x cheaper on this
  # fabric and the old default handed our own policies that advantage.
  docker run -d --name casr-md-router --network host \
    "${mounts[@]}" \
    -e PYTHONPATH=/casr-serving \
    -e DISABLED_INSTANCES="${DISABLED_INSTANCES:-}" \
    -e SLO_TTFT_MS="${SLO_TTFT_MS:-}" \
    -e SLO_PENALTY="${SLO_PENALTY:-}" \
    -e OVERFLOW_PENALTY="${OVERFLOW_PENALTY:-}" \
    -e WARMUP_S="${WARMUP_S:-}" \
    -e STRUCTURAL_STARTUP_COST="${STRUCTURAL_STARTUP_COST:-}" \
    -e STRUCTURAL_EVAL_WINDOW_MS="${STRUCTURAL_EVAL_WINDOW_MS:-}" \
    -e STRUCTURAL_DWELL_MS="${STRUCTURAL_DWELL_MS:-}" \
    -e STRUCTURAL_STARTUP_S="${STRUCTURAL_STARTUP_S:-}" \
    -e STRUCTURAL_HOLDING_COST="${STRUCTURAL_HOLDING_COST:-}" \
    -e STRUCTURAL_IDLE_COST_FRACTION="${STRUCTURAL_IDLE_COST_FRACTION:-}" \
    -e MAX_ACTIVE_PREFILL="${MAX_ACTIVE_PREFILL:-}" \
    -e PLAN_TTL_S="${PLAN_TTL_S:-}" \
    -e CONTROL_INTERVAL_S="${CONTROL_INTERVAL_S:-}" \
    -e PD_PROXY_PORT="${PD_PROXY_PORT:-}" \
    -e PD_PROXY_WAIT_S="${PD_PROXY_WAIT_S:-60}" \
    -e PD_TRANSFER_BACKEND="${PD_TRANSFER_BACKEND:-native}" \
    -e LOCAL_PREFILL="${LOCAL_PREFILL:-auto}" \
    -e LOCAL_PREFILL_BASELINES="${LOCAL_PREFILL_BASELINES:-auto}" \
    -e LOCAL_PREFILL_QUEUE_WEIGHT="${LOCAL_PREFILL_QUEUE_WEIGHT:-}" \
    -e LOCAL_PREFILL_QUEUE_CAP="${LOCAL_PREFILL_QUEUE_CAP:-}" \
    -e PROMPT_TOKEN_CALIBRATION_REQS="${PROMPT_TOKEN_CALIBRATION_REQS:-}" \
    -e LOCAL_PREFILL_MS_PER_1K="${LOCAL_PREFILL_MS_PER_1K:-}" \
    -e TRANSFER_MS_PER_1K_LOCAL="${TRANSFER_MS_PER_1K_LOCAL:-}" \
    -e TRANSFER_MS_PER_1K_CROSS="${TRANSFER_MS_PER_1K_CROSS:-}" \
    -e TRANSFER_FIXED_MS_CROSS="${TRANSFER_FIXED_MS_CROSS:-}" \
    -e DECODE_DEBUG="${DECODE_DEBUG:-}" \
    -e PREFILL_CLASS_LIMIT="${PREFILL_CLASS_LIMIT:-}" \
    -e EQUALIZE_PREFILL_CAPACITY="${EQUALIZE_PREFILL_CAPACITY:-}" \
    -e EQUALIZE_DECODE_CAPACITY="${EQUALIZE_DECODE_CAPACITY:-}" \
    "${args[@]}" >/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 5 http://127.0.0.1:9000/health >/dev/null && return 0
    sleep 2
  done
  echo "router failed to start for policy $policy" >&2
  docker logs casr-md-router 2>&1 | tail -20 >&2
  return 1
}

check_ray_topology

for policy in $policies; do
  echo "== policy: $policy =="
  warmup_all_pairs
  stop_configured_prefills
  start_router "$policy"
  # Timings alone cannot tell a working handoff from a fast one that serves
  # stale KV: on 2026-09-13 every instance passed health checks and answered in
  # ~300 ms while returning newline spam, because the Decode read a staging
  # slot the sender had not written yet.  Ask real articles for a code they
  # carry, and abort the policy if too few answers can reproduce their own.
  correctness_args=(--endpoint http://127.0.0.1:9000
    --requests "${CORRECTNESS_REQUESTS:-4}"
    --max-fail-ratio "${CORRECTNESS_MAX_FAIL_RATIO:-0.25}"
    --model "${MODEL_NAME:-qwen3-8b}")
  if [[ "${CORRECTNESS_MODE:-code}" == "consistency" ]]; then
    # Model-agnostic oracle for base models: the same prompt must generate the
    # same text locally and through the P/D pair.
    consistency_pair=$(python3 -c "
import json
d=json.load(open('$topology'))
p=[x for x in d['prefills'] if x.get('enabled', True)][0]
q=[x for x in d['decodes'] if x.get('enabled', True)][0]
print(f\"http://{p['host']}:{p['port']} http://{q['host']}:{q['port']}\")")
    read -r c_p c_d <<<"$consistency_pair"
    correctness_args+=(--prefill "$c_p" --decode "$c_d" --native-pd --consistency)
  fi
  if [[ "${SKIP_CORRECTNESS:-0}" == "1" ]]; then
    echo "correctness gate skipped by SKIP_CORRECTNESS=1" \
      > "$result_dir/correctness-$policy.txt"
  elif ! python3 tests/check_pd_correctness.py "${correctness_args[@]}" \
    > "$result_dir/correctness-$policy.txt" 2>&1; then
    echo "P/D output correctness gate failed for policy $policy" >&2
    cat "$result_dir/correctness-$policy.txt" >&2
    exit 1
  fi
  if [[ -n "$trace" ]]; then
    python3 tests/real_dataset_client.py --url http://127.0.0.1:9000 \
      --trace "$trace" --num-reqs "$num_reqs" --model "${MODEL_NAME:-qwen3-8b}" \
      --max-output-tokens "$max_output_tokens" --concurrency "$concurrency" \
      --pacing "${CLIENT_PACING:-closed}" \
      --request-timeout-s "${REQUEST_TIMEOUT_S:-180}" \
      --time-scale "$time_scale" \
      --slo-ttft-ms "${SLO_TTFT_MS:-0}" --slo-tpot-ms "${SLO_TPOT_MS:-0}" \
      "${stream_args[@]}" \
      --output "$result_dir/client-$policy.jsonl" | tee "$result_dir/summary-$policy.json"
  else
    python3 tests/real_multidomain_client.py --url http://127.0.0.1:9000 \
      --num-reqs "$num_reqs" --num-clients "$num_clients" --model "${MODEL_NAME:-qwen3-8b}" \
      --prefix-tokens "$prefix_tokens" --output-tokens "$output_tokens" \
      --concurrency "$concurrency" --hot-prefixes "$hot_prefixes" \
      --drift-after-fraction "$drift_after_fraction" \
      --output "$result_dir/client-$policy.jsonl" | tee "$result_dir/summary-$policy.json"
  fi
  curl -s --max-time 5 http://127.0.0.1:9000/routing-state \
    > "$result_dir/state-$policy.json" || true
  echo
done

echo "artifacts: $result_dir"
