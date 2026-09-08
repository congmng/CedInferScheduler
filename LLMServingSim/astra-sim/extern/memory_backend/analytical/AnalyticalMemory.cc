/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

#include "extern/memory_backend/analytical/AnalyticalMemory.hh"
#include <json/json.hpp>
#include <fstream>
#include <iostream>
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/system/AstraMemoryAPI.hh"
using AstraSim::MemoryLocationType; 

using namespace std;
using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

AnalyticalMemory::AnalyticalMemory(
    string memory_configuration) noexcept {
  ifstream conf_file;

  conf_file.open(memory_configuration);
  if (!conf_file) {
    cerr << "Unable to open file: " << memory_configuration << endl;
    exit(1);
  }

  json j;
  conf_file >> j;

  if (j.contains("memory-type")) {
    string mem_type_str = j["memory-type"];
    if (mem_type_str.compare("NO_MEMORY_EXPANSION") == 0) {
      mem_type = NO_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NODE_MEMORY_EXPANSION") == 0) {
      mem_type = PER_NODE_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NPU_MEMORY_EXPANSION") == 0) {
      mem_type = PER_NPU_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("MEMORY_POOL") == 0) {
      mem_type = MEMORY_POOL;
    } else {
      cerr << "Unsupported memory type: " << mem_type_str << endl;
      exit(1);
    }
  }

  if (j.contains("memory-location")) {
    string mem_loc_type_str = j["memory-location"];
    if (mem_loc_type_str.compare("INVALID_MEMORY") == 0) {
      std::cout << "Detected INVALID_MEMORY" << std::endl;
      mem_loc_type = MemoryLocationType::INVALID_MEMORY;
    } else if (mem_loc_type_str.compare("LOCAL_MEMORY") == 0) {
      mem_loc_type = MemoryLocationType::LOCAL_MEMORY;
    } else if (mem_loc_type_str.compare("REMOTE_MEMORY") == 0) {
      mem_loc_type = MemoryLocationType::REMOTE_MEMORY;
    } else if (mem_loc_type_str.compare("CXL_MEMORY") == 0) {
      mem_loc_type = MemoryLocationType::CXL_MEMORY;
    } else if (mem_loc_type_str.compare("STORAGE_MEMORY") == 0) {
      mem_loc_type = MemoryLocationType::STORAGE_MEMORY;
    } else {
      cerr << "Unsupported memory location type: " << mem_loc_type_str << endl;
      exit(1);
    }
  } else {
    std::cout << "No memory location type specified. Defaulting to REMOTE_MEMORY" << std::endl;
    mem_loc_type = MemoryLocationType::REMOTE_MEMORY;
  }

  mem_latency = 0;
  if (j.contains("mem-latency")) {
    mem_latency = j["mem-latency"];
  }

  mem_bw = 0;
  if (j.contains("mem-bw")) {
    mem_bw = j["mem-bw"];
  }

  num_devices = 1;
  if (j.contains("num-devices")) {
    num_devices = j["num-devices"];
  }

  pim_channels = 0; // > 1 if pim enabled
  if (j.contains("pim-channels")) {
    pim_channels = j["pim-channels"];
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    for (int i = 0; i < num_devices; i++) {
      ongoing_transaction.push_back(false);
      deque<PendingMemoryRequest> dpmr;
      pending_requests.push_back(dpmr);

      // pim reqeust queue
      for (int i = 0; i < pim_channels; i++) {
        pim_ongoing_transaction.push_back(false);
        deque<PendingMemoryRequest> pim_dpmr;
        pim_pending_requests.push_back(pim_dpmr);
      }
    }
  } else if (mem_type == MEMORY_POOL) {
    for (int i = 0; i < num_devices; i++) {
      ongoing_transaction.push_back(false);
      deque<PendingMemoryRequest> dpmr;
      pending_requests.push_back(dpmr);

      // pim reqeust queue
      for (int i = 0; i < pim_channels; i++) {
        pim_ongoing_transaction.push_back(false);
        deque<PendingMemoryRequest> pim_dpmr;
        pim_pending_requests.push_back(pim_dpmr);
      }
    }
  }

  conf_file.close();
}

void AnalyticalMemory::set_sys(int id, Sys* sys) {
  sys_map[id] = sys;
}

void AnalyticalMemory::issue(
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
  
  int sys_id = wlhd->sys_id;
  int device_id = wlhd->device_id;
  bool pim_enabled = wlhd->pim_enabled;
  // PIM operation
  if (pim_enabled) {
    if (pim_channels == 0) {
      cerr << "PIM operation requested but pim_channels is set to 0" << endl;
      exit(1);
    }
    int pim_channel_id = wlhd->pim_channel_id;
    int queue_idx = num_devices * device_id + pim_channel_id;
    if (pim_ongoing_transaction[queue_idx]) {
      PendingMemoryRequest pmr(tensor_size, wlhd);
      pim_pending_requests[queue_idx].push_back(pmr);
    } else {
      uint64_t load_store_time = get_mem_runtime(tensor_size); // new tensors that need to be load/stored in pim
      uint64_t runtime = wlhd->pim_runtime + load_store_time;

      Sys* sys = sys_map[sys_id];

      sys->register_event(this, EventType::General, wlhd, runtime);

      sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);

      pim_ongoing_transaction[queue_idx] = true;
    }
    return;
  }
  // Ordinary memory access
  else {
    if (mem_type == NO_MEMORY_EXPANSION) {
      cerr << "Memory access is not supported in NO_MEMORY_EXPANSION"
          << endl;
      exit(1);
    } else if (mem_type == PER_NODE_MEMORY_EXPANSION) {
      if (ongoing_transaction[device_id]) {
        PendingMemoryRequest pmr(tensor_size, wlhd);
        pending_requests[device_id].push_back(pmr);
      } else {
        uint64_t runtime = get_mem_runtime(tensor_size);

        Sys* sys = sys_map[sys_id];

        sys->register_event(this, EventType::General, wlhd, runtime);

        sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);

        ongoing_transaction[device_id] = true;
      }
    } else if (mem_type == PER_NPU_MEMORY_EXPANSION) {
      uint64_t runtime = get_mem_runtime(tensor_size);
      Sys* sys = sys_map[sys_id];
      sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);
    } else if (mem_type == MEMORY_POOL) {
      if (ongoing_transaction[device_id]) {
        PendingMemoryRequest pmr(tensor_size, wlhd);
        pending_requests[device_id].push_back(pmr);
      } else {
        uint64_t runtime = get_mem_runtime(tensor_size);

        Sys* sys = sys_map[sys_id];

        sys->register_event(this, EventType::General, wlhd, runtime);

        sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);

        ongoing_transaction[device_id] = true;
      }
    }
  }
}

void AnalyticalMemory::call(EventType type, CallData* data) {
  WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
  int device_id = wlhd->device_id;
  bool pim_enabled = wlhd->pim_enabled;

  // PIM operation
  if (pim_enabled) {
    int pim_channel_id = wlhd->pim_channel_id;
    int queue_idx = num_devices * device_id + pim_channel_id;
    if (!pim_pending_requests[queue_idx].empty()) {
      PendingMemoryRequest pmr = pim_pending_requests[queue_idx].front();
      pim_pending_requests[queue_idx].pop_front();
      uint64_t load_store_time = get_mem_runtime(pmr.tensor_size); // new tensors that need to be load/stored in pim
      uint64_t runtime = pmr.wlhd->pim_runtime + load_store_time;
      Sys* sys = sys_map[pmr.wlhd->sys_id];

      sys->register_event(this, EventType::General, pmr.wlhd, runtime);

      sys->register_event(
          pmr.wlhd->workload, EventType::General, pmr.wlhd, runtime);

      pim_ongoing_transaction[queue_idx] = true;
    } else {
      pim_ongoing_transaction[queue_idx] = false;
    }
    return;
  }
  // Ordinary memory access
  else{
    if (mem_type == PER_NODE_MEMORY_EXPANSION) {
      if (!pending_requests[device_id].empty()) {
        PendingMemoryRequest pmr = pending_requests[device_id].front();
        pending_requests[device_id].pop_front();

        uint64_t runtime = get_mem_runtime(pmr.tensor_size);

        Sys* sys = sys_map[pmr.wlhd->sys_id];

        sys->register_event(this, EventType::General, pmr.wlhd, runtime);

        sys->register_event(
            pmr.wlhd->workload, EventType::General, pmr.wlhd, runtime);

        ongoing_transaction[device_id] = true;
      } else {
        ongoing_transaction[device_id] = false;
      }
    } else if (mem_type == MEMORY_POOL) {
      if (!pending_requests[device_id].empty()) {
        PendingMemoryRequest pmr = pending_requests[device_id].front();
        pending_requests[device_id].pop_front();

        uint64_t runtime = get_mem_runtime(pmr.tensor_size);

        Sys* sys = sys_map[pmr.wlhd->sys_id];

        sys->register_event(this, EventType::General, pmr.wlhd, runtime);

        sys->register_event(
            pmr.wlhd->workload, EventType::General, pmr.wlhd, runtime);

        ongoing_transaction[device_id] = true;
      } else {
        ongoing_transaction[device_id] = false;
      }
    }
  }
}

uint64_t AnalyticalMemory::get_mem_runtime(uint64_t tensor_size) {
  uint64_t runtime = mem_latency
      + static_cast<uint64_t>((static_cast<double>(tensor_size) / mem_bw));
  return runtime;
}
