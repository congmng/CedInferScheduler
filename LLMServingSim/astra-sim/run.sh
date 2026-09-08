#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")
#./build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
BINARY="${SCRIPT_DIR:?}"/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
# BINARY="${SCRIPT_DIR:?}"/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default
# /inputs/workload/H100/meta-llama/Llama-3.1-70B/instance0_batch0/llm
WORKLOAD="${SCRIPT_DIR:?}"/inputs/workload/A6000/meta-llama/Llama-3.1-8B/instance0_batch0/llm
SYSTEM="${SCRIPT_DIR:?}"/inputs/system/system.json
NETWORK="${SCRIPT_DIR:?}"/inputs/network/network.yml
# NETWORK="${SCRIPT_DIR:?}"/extern/network_backend/ns-3/scratch/config/config.txt
MEMORY="${SCRIPT_DIR:?}"/inputs/memory/memory_expansion.json
# LOGICAL="${SCRIPT_DIR:?}"/inputs/logical_topology/logical_8nodes_1D.json

# running version
"${BINARY}" \
  --workload-configuration="${WORKLOAD}" \
  --system-configuration="${SYSTEM}" \
  --network-configuration="${NETWORK}" \
  --memory-configuration="${MEMORY}" \
  --start-npu-ids="0" \
  # --end-npu-ids="0" 
  # --logical-topology-configuration="${LOGICAL}"
