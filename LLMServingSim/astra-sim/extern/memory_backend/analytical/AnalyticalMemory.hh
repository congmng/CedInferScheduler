/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ANALYTICAL_MEMORY_HH__
#define __ANALYTICAL_MEMORY_HH__

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "astra-sim/system/AstraMemoryAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"

struct MemLevelConf {
  std::string name;            // "local_mem", "remote_mem", "cxl_mem" key name
  std::string memory_type;     // "PER_NPU_MEMORY_EXPANSION" / "PER_NODE_MEMORY_EXPANSION" / "MEMORY_POOL"
  std::string memory_location; // "LOCAL_MEMORY" / "REMOTE_MEMORY" / "CXL_MEMORY" 
  uint64_t mem_latency; 
  uint64_t mem_bw;      
  uint32_t num_devices;
  uint32_t pim_channels;
};

namespace Analytical {
enum MemoryArchitectureType {
  NO_MEMORY_EXPANSION = 0,
  PER_NODE_MEMORY_EXPANSION,
  PER_NPU_MEMORY_EXPANSION,
  MEMORY_POOL
};


class PendingMemoryRequest {
 public:
  PendingMemoryRequest(
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd)
    : tensor_size(tensor_size), wlhd(wlhd) {
  }

  uint64_t tensor_size;
  AstraSim::WorkloadLayerHandlerData* wlhd;
};

class AnalyticalMemory : public AstraSim::AstraMemoryAPI, public AstraSim::Callable{
 public:
  AnalyticalMemory(std::string memory_configuration) noexcept;
  void set_sys(int id, AstraSim::Sys* sys);
  void issue(
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd);
  void call(AstraSim::EventType type, AstraSim::CallData* data);
  uint64_t get_mem_runtime(uint64_t tensor_size);
  AstraSim::MemoryLocationType get_memory_location_type() const override { return mem_loc_type; }

 private:
  MemoryArchitectureType mem_type;
  AstraSim::MemoryLocationType mem_loc_type; 
  uint64_t mem_latency; // memory access latency in nanosec
  uint64_t mem_bw; // memory bandwidth in GB/sec
  uint32_t num_devices; // number of devices for this memory level
  uint32_t pim_channels; // number of PIM channels for this memory level
  std::vector<bool> ongoing_transaction;

  std::unordered_map<int, AstraSim::Sys*> sys_map;
  std::vector<std::deque<PendingMemoryRequest>> pending_requests;

  // pim implementation
  std::vector<bool> pim_ongoing_transaction;
  std::vector<std::deque<PendingMemoryRequest>> pim_pending_requests;
};
} // namespace Analytical

#endif /* __ANALYTICAL_MEMORY_HH__ */
