#!/usr/bin/env bash
# Start a single-GPU Qwen3-8B vLLM 0.29.0 + LMCache 0.5.4 cross-host P/D pair.
# Run this script from a control host with SSH access to both nodes. The remote
# users must belong to the docker group so orchestration remains non-interactive.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

P_HOST="${P_HOST:-10.212.70.196}"
D_HOST="${D_HOST:-10.212.67.68}"
P_SSH_TARGET="${P_SSH_TARGET:-buaa@$P_HOST}"
D_SSH_TARGET="${D_SSH_TARGET:-buaa@$D_HOST}"
IMAGE="${IMAGE:-vllm/vllm-openai:casr029}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
P_MODEL="${P_MODEL:-/data/Models/Qwen3-8B}"
D_MODEL="${D_MODEL:-/data/sdb/model/casr/Qwen3-8B}"
P_GPU="${P_GPU:-1}"
D_GPU="${D_GPU:-1}"
P_UCX_NET_DEVICES="${P_UCX_NET_DEVICES:-ens6f0}"
D_UCX_NET_DEVICES="${D_UCX_NET_DEVICES:-ens18}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
P_CONFIG_HOST="${P_CONFIG_HOST:-/home/buaa/casr-lmcache-pd/lmcache-prefiller-crosshost.yaml}"
D_CONFIG_HOST="${D_CONFIG_HOST:-/home/buaa/casr-lmcache-pd/lmcache-decoder-crosshost.yaml}"
P_CONFIG="${P_CONFIG:-/pd-config/lmcache-prefiller-crosshost.yaml}"
D_CONFIG="${D_CONFIG:-/pd-config/lmcache-decoder-crosshost.yaml}"

remote_docker() {
  local host="$1"; shift
  ssh "$host" "docker $*"
}

scp "$SCRIPT_DIR/lmcache-prefiller-crosshost.yaml" "$P_SSH_TARGET:/tmp/casr-lmcache-prefiller.yaml"
scp "$SCRIPT_DIR/lmcache-decoder-crosshost.yaml" "$D_SSH_TARGET:/tmp/casr-lmcache-decoder.yaml"
scp "$SCRIPT_DIR/disagg_proxy_pd.py" "$P_SSH_TARGET:/tmp/casr-disagg-proxy-pd.py"
ssh "$P_SSH_TARGET" "mkdir -p $(dirname "$P_CONFIG_HOST") && mv /tmp/casr-lmcache-prefiller.yaml $P_CONFIG_HOST"
ssh "$D_SSH_TARGET" "mkdir -p $(dirname "$D_CONFIG_HOST") && mv /tmp/casr-lmcache-decoder.yaml $D_CONFIG_HOST"
ssh "$D_SSH_TARGET" "sed -i 's/10.212.70.196/$D_HOST/g' $D_CONFIG_HOST"
ssh "$P_SSH_TARGET" "mv /tmp/casr-disagg-proxy-pd.py $(dirname "$P_CONFIG_HOST")/disagg_proxy_pd.py"

remote_docker "$P_SSH_TARGET" rm -f casr-llama-api casr-pd-prefiller casr-pd-proxy || true
remote_docker "$D_SSH_TARGET" rm -f casr-llama-api casr-pd-decoder || true

remote_docker "$P_SSH_TARGET" "run -d --name casr-pd-prefiller --network host --ipc host --shm-size 16g --gpus 'device=$P_GPU' -e PYTHONHASHSEED=123 -e UCX_TLS=cuda_copy,tcp -e UCX_NET_DEVICES=$P_UCX_NET_DEVICES -e UCX_TCP_INTERFACE=$P_UCX_NET_DEVICES -e LMCACHE_CONFIG_FILE=$P_CONFIG -v $P_MODEL:/model:ro -v $(dirname "$P_CONFIG_HOST"):/pd-config:ro --entrypoint vllm $IMAGE serve /model --served-model-name $MODEL_NAME --host 0.0.0.0 --port 8100 --enforce-eager --dtype bfloat16 --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-model-len 4096 --max-num-seqs 16 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"producer1\"}}'"
remote_docker "$D_SSH_TARGET" "run -d --name casr-pd-decoder --network host --ipc host --shm-size 16g --gpus 'device=$D_GPU' -e PYTHONHASHSEED=123 -e UCX_TLS=cuda_copy,tcp -e UCX_NET_DEVICES=$D_UCX_NET_DEVICES -e UCX_TCP_INTERFACE=$D_UCX_NET_DEVICES -e LMCACHE_CONFIG_FILE=$D_CONFIG -v $D_MODEL:/model:ro -v $(dirname "$D_CONFIG_HOST"):/pd-config:ro --entrypoint vllm $IMAGE serve /model --served-model-name $MODEL_NAME --host 0.0.0.0 --port 8200 --enforce-eager --dtype bfloat16 --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-model-len 4096 --max-num-seqs 16 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"consumer1\"}}'"
remote_docker "$P_SSH_TARGET" "run -d --name casr-pd-proxy --network host --ipc host -v $(dirname "$P_CONFIG_HOST"):/pd-config:ro --entrypoint python3 $IMAGE /pd-config/disagg_proxy_pd.py --host 0.0.0.0 --port 9000 --prefiller-host $P_HOST --prefiller-port 8100 --decoder-host $D_HOST --decoder-port 8200"

echo "Prefill: http://$P_HOST:8100"
echo "Decode:  http://$D_HOST:8200"
echo "Proxy:   http://$P_HOST:9000"
