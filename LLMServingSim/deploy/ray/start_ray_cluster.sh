#!/usr/bin/env bash
# Start a lightweight Ray control plane across the five heterogeneous experiment
# hosts.  Ray only provides discovery, resource labels, and health/state
# collection; vLLM P/D containers stay under the separate LMCache launcher so
# Ray never silently claims a GPU already held by an inference container.
set -euo pipefail

HEAD_HOST="${HEAD_HOST:-10.212.67.167}"
HEAD_USER="${HEAD_USER:-ubuntu}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_VERSION="${RAY_VERSION:-2.55.1}"

# Ray keeps its session state, object spool and *all* of its logs under
# ``--temp-dir``.  The default is /tmp/ray, i.e. the system disk, which is at
# 99% on 3090a; every host therefore points Ray at a data-disk directory
# (``ray_temp_dir`` in deploy/real_lmcache_pd/router_config.json, same paths).
HEAD_RAY_TEMP="${HEAD_RAY_TEMP:-/mnt/home/casr/ray}"

# worker spec: "user@host|domain|gpu_type|ssh_port|node_ip|ray_temp_dir"
WORKERS=(
  "${WORKER_3090:-buaa@10.212.67.68}|3090a|RTX3090|${WORKER_3090_PORT:-22}|${WORKER_3090_IP:-10.212.67.68}|${WORKER_3090_TEMP:-/data/sdb/model/casr/ray}"
  "${WORKER_5090:-buaa@10.212.70.196}|5090|RTX5090|${WORKER_5090_PORT:-22}|${WORKER_5090_IP:-10.212.70.196}|${WORKER_5090_TEMP:-/data/casr/ray}"
)
if [[ -n "${WORKER_3090B:-buaa@10.212.70.38}" ]]; then
  WORKERS+=("${WORKER_3090B:-buaa@10.212.70.38}|3090b|RTX3090|${WORKER_3090B_PORT:-22}|${WORKER_3090B_IP:-10.212.70.38}|${WORKER_3090B_TEMP:-/data/casr/ray}")
fi
if [[ -n "${WORKER_A100:-buaa@10.66.0.15}" ]]; then
  WORKERS+=("${WORKER_A100:-buaa@10.66.0.15}|a100|A100|${WORKER_A100_PORT:-2222}|${WORKER_A100_IP:-10.66.0.15}|${WORKER_A100_TEMP:-/mnt/adminserver-nfsrdma/casr/ray}")
fi

ssh_node() {
  local target="${1%%|*}" port="$2"; shift 2
  ssh -o BatchMode=yes -p "$port" "$target" "$@"
}

install_ray() {
  local target="${1%%|*}" port="$2"
  if ssh_node "$target" "$port" "python3 -c 'import ray' >/dev/null 2>&1"; then
    echo "  ray already present on $target:$port"
    return 0
  fi
  ssh_node "$target" "$port" \
    "python3 -m pip install --user --quiet --upgrade 'ray[default]==$RAY_VERSION'"
}

echo "== installing Ray $RAY_VERSION =="
if ! ssh_node "$HEAD_USER@$HEAD_HOST|head" 22 "python3 -c 'import ray' >/dev/null 2>&1"; then
  ssh_node "$HEAD_USER@$HEAD_HOST|head" 22 \
    "python3 -m pip install --user --quiet --upgrade 'ray[default]==$RAY_VERSION'"
fi
for worker in "${WORKERS[@]}"; do
  IFS='|' read -r target domain gpu_type port node_ip <<<"$worker"
  echo "  - $target:$port ($domain/$gpu_type)"
  install_ray "$worker" "$port"
done

echo "== starting Ray head on $HEAD_HOST =="
ssh_node "$HEAD_USER@$HEAD_HOST|head" 22 \
  "export PATH=\$HOME/.local/bin:\$PATH; ray stop --force >/dev/null 2>&1 || true; \
   mkdir -p $HEAD_RAY_TEMP; \
   ray start --head --node-ip-address=$HEAD_HOST --port=$RAY_PORT \
     --temp-dir=$HEAD_RAY_TEMP \
     --dashboard-host=0.0.0.0 \
     --resources='{\"domain:4090\": 1, \"gpu_type:RTX4090\": 1}'"

for worker in "${WORKERS[@]}"; do
  IFS='|' read -r target domain gpu_type port node_ip temp_dir <<<"$worker"
  echo "== starting worker $node_ip ($domain) =="
  ssh_node "$worker" "$port" \
    "export PATH=\$HOME/.local/bin:\$PATH; ray stop --force >/dev/null 2>&1 || true; \
     mkdir -p $temp_dir; \
     ray start --address=$HEAD_HOST:$RAY_PORT --node-ip-address=$node_ip \
       --temp-dir=$temp_dir \
       --resources='{\"domain:$domain\": 1, \"gpu_type:$gpu_type\": 1}'"
done

echo "Ray head: $HEAD_HOST:$RAY_PORT"
echo "Dashboard: http://$HEAD_HOST:8265"
echo "Session state/logs (data disk): head=$HEAD_RAY_TEMP"
for worker in "${WORKERS[@]}"; do
  IFS='|' read -r _ domain _ _ node_ip temp_dir <<<"$worker"
  echo "  $node_ip ($domain): $temp_dir"
done
echo "== cluster status =="
ssh_node "$HEAD_USER@$HEAD_HOST|head" 22 \
  "export PATH=\$HOME/.local/bin:\$PATH; ray status --address=$HEAD_HOST:$RAY_PORT"
