/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "common/CmdLineParser.hh"
#include "congestion_unaware/CongestionUnawareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_unaware/Helper.h>
#include <memory_backend/analytical/AnalyticalMemory.hh>
#include <json/json.hpp>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionUnaware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;
using namespace std;
using json = nlohmann::json;


static std::string save_json_to_tmp(const json& j, const std::string& name) {
  const char* dir = "tmp__mem";
  if (::mkdir(dir, 0755) == -1) {
    if (errno != EEXIST) {
      std::perror("mkdir tmp_mem");
      std::exit(1);
    }
  }
  std::string path = std::string(dir) + "/" + name + ".json";
  std::ofstream ofs(path);
  if (!ofs) {
    std::cerr << "Unable to write tmp file: " << path << "\n";
    std::exit(1);
  }
  ofs << j.dump(2);
  return path;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);

    // Get command line arguments
    const auto workload_configuration =
        cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration =
        cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration =
        cmd_line_parser.get<std::string>("system-configuration");
    const auto memory_configuration =
        cmd_line_parser.get<std::string>("memory-configuration");
    const auto network_configuration =
        cmd_line_parser.get<std::string>("network-configuration");
    const auto logging_configuration =
        cmd_line_parser.get<std::string>("logging-configuration");
    const auto num_queues_per_dim =
        cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol =
        cmd_line_parser.get<bool>("rendezvous-protocol");
    auto start_npu_ids =
        cmd_line_parser.get<std::vector<int>>("start-npu-ids");
    auto end_npu_ids =
        cmd_line_parser.get<std::vector<int>>("end-npu-ids");

    // clear vector if default value is used
    if (start_npu_ids.size() == 1 && start_npu_ids[0] == -1) {
      start_npu_ids.clear();
    }
    if (end_npu_ids.size() == 1 && end_npu_ids[0] == -1) {
      end_npu_ids.clear();
    }

    AstraSim::LoggerFactory::init(logging_configuration);

    // Instantiate event queue
    const auto event_queue = std::make_shared<EventQueue>();

    // Generate topology
    const auto network_parser = NetworkParser(network_configuration);
    const auto topology = construct_topology(network_parser);

    // Get topology information
    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();

    // Set up Network API
    CongestionUnawareNetworkApi::set_event_queue(event_queue);
    CongestionUnawareNetworkApi::set_topology(topology);

    // Create ASTRA-sim related resources
    auto network_apis =
        std::vector<std::unique_ptr<CongestionUnawareNetworkApi>>();
    
    json mem_json;
    std::ifstream rm_ifs(memory_configuration);
    rm_ifs >> mem_json;

    std::vector<std::unique_ptr<AnalyticalMemory>> memory_levels;

    // Check if the configuration is for a single memory type
    const bool is_single =
      mem_json.is_object() &&
      mem_json.contains("memory-type") &&
      mem_json.contains("mem-latency") &&
      mem_json.contains("mem-bw");
    
    if (is_single) {
      std::cout << "Single Memory Configuration Detected" << std::endl;
      memory_levels.push_back(std::make_unique<AnalyticalMemory>(memory_configuration));
    } else {
      // local memory
      if (mem_json.contains("local_mem") && mem_json["local_mem"].is_object()) {
        json j = mem_json["local_mem"];
        j["memory-location"] = "LOCAL_MEMORY";
        auto path = save_json_to_tmp(j, "local_mem");
        memory_levels.push_back(std::make_unique<AnalyticalMemory>(path));
        std::remove(path.c_str()); 
      }

      // remote memory
      if (mem_json.contains("remote_mem") && mem_json["remote_mem"].is_object()) {
        json j = mem_json["remote_mem"];
        j["memory-location"] = "REMOTE_MEMORY";
        auto path = save_json_to_tmp(j, "remote_mem");
        memory_levels.push_back(std::make_unique<AnalyticalMemory>(path));
        std::remove(path.c_str()); 
      }

      // cxl memory
      if (mem_json.contains("cxl_mem") && mem_json["cxl_mem"].is_object()) {
        json j = mem_json["cxl_mem"];
        j["memory-location"] = "CXL_MEMORY";
        auto path = save_json_to_tmp(j, "cxl_mem");
        memory_levels.push_back(std::make_unique<AnalyticalMemory>(path));
        std::remove(path.c_str()); 
      }

      ::rmdir("tmp_mem");
    }

    auto memory_apis = std::vector<AstraMemoryAPI*>();
    for (auto& mem_api : memory_levels) {
      memory_apis.push_back(mem_api.get());
    }

    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }

    for (int i = 0; i < npus_count; i++) {
        // create network and system
        auto network_api = std::make_unique<CongestionUnawareNetworkApi>(i);
        auto* const system =
            new Sys(i, workload_configuration, comm_group_configuration,
                    system_configuration, memory_apis, network_api.get(),
                    npus_count_per_dim, queues_per_dim, injection_scale,
                    comm_scale, rendezvous_protocol);

        // push back network and system
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    // Map instance NPU IDs for proper workload management
    // Precompute the systems handled by each controller NPU
    std::vector<std::vector<Sys*>> managed_systems(start_npu_ids.size());

    for (std::size_t idx = 0; idx < start_npu_ids.size(); ++idx) {
      int npu_id = start_npu_ids[idx];

      // Determine the upper bound for this controller:
      // - If there's a next controller, stop before it
      // - Otherwise, go until npus_count
      int upper_bound_id;
      if (idx + 1 < start_npu_ids.size()) {
        upper_bound_id = start_npu_ids[idx + 1];
      } else {
        upper_bound_id = npus_count;  // last controller handles until the end
      }

      // Collect systems in the range (npu_id+1 .. upper_bound_id-1)
      for (int sid = npu_id + 1; sid < upper_bound_id; ++sid) {
        if (sid < 0 || sid >= npus_count) {
            AstraSim::LoggerFactory::get_logger("workload")
                ->critical("Skipping invalid system id {} while building managed_systems", sid);
        }
        if (std::find(end_npu_ids.begin(), end_npu_ids.end(), sid) != end_npu_ids.end()) {
          continue;
        }
        managed_systems[idx].push_back(systems[sid]);
      }
    }

    // Initiate simulation
    for (int i = 0; i < npus_count; i++) {
        systems[i]->workload->fire();
        // For debugging
        // systems[i]->workload->et_feeder->printGraph();
    }

    // run simulation
    // while (!event_queue->finished()) {
    //     event_queue->proceed();
    // }

    // There is deliberately no per-tick "Checking NPU ..." trace here. It
    // used to print once per NPU per event-queue tick regardless of whether
    // that NPU had anything to report, and the frontend has to read every
    // such line off the pipe before it reaches the "Waiting" it is blocked
    // on: 2,397,257 of 3,072,837 lines on a 10-request 8-NPU run (78%), and
    // 12,734,237 of 12,788,005 on an MoE DP+EP run (99.6%), because the
    // volume follows event-queue ticks rather than handshakes. `git log` has
    // it if a hang ever needs it back.
    //
    // "Checking Non-Exited Systems ..." further down is *protocol*, not
    // debug: Controller.read_wait terminates its read loop on that exact
    // string. Likewise "All Request Has Been Exited" / "ERROR: Some
    // Requests Remain" for Controller.check_end. Leave all three alone.

    // ---- Idle-NPU re-ask suppression ------------------------------------
    // A finished NPU that answered "pass" will answer "pass" again until
    // something changes the frontend's scheduling decision, and only two
    // things can: another NPU reporting an iteration the frontend has not
    // processed yet -- which is what frees KV blocks, hands off a P/D
    // prefill, or completes a DP round -- or simulated time reaching a
    // request arrival the frontend already knows about. Re-asking in
    // between is guaranteed to get "pass" back.
    //
    // That is nearly all of the traffic. Classified on the simulator
    // container, "pass" was 336,894 of 337,786 answers on a 10-request
    // 8-NPU run, and 336,890 of those were sent while another instance was
    // mid-batch; on pp=4, 32,117 of 32,154.
    //
    // So: remember the state generation at which each NPU passed, bump the
    // generation when an NPU reports an iteration it has not reported
    // before, and skip an NPU whose recorded generation still matches. The
    // frontend supplies the arrival deadline it already computes, as
    // "pass <tick>", for the one case the generation cannot cover: time
    // crossing a known arrival while every other instance is mid-batch.
    //
    // Liveness is guaranteed three ways, because a suppression bug here
    // would surface as a silent hang rather than an error -- the same
    // failure mode as a mis-cut pipeline stage boundary:
    //   1. an NPU that receives a workload is un-suppressed outright, so
    //      the NPU actually doing work is always asked when it finishes,
    //      and its report bumps the generation for everyone else;
    //   2. an empty event queue clears all suppression. That is also what
    //      keeps a fully idle cluster bit-identical to before: it still
    //      advances in add_current_time()'s 1 ms quanta;
    //   3. a bounded backstop, in case 1 and 2 are ever both wrong. It
    //      should never fire, and turns a hypothetical hang into a
    //      slowdown instead.
    constexpr long long kSuppressionBackstop = 1000;
    std::vector<long long> pass_gen(npus_count, -1);      // -1 = askable
    std::vector<long long> pass_deadline(npus_count, 0);  //  0 = no deadline
    std::vector<long long> last_reported_iter(npus_count, -1);
    long long state_gen = 0;
    long long unasked_rounds = 0;

    auto askable = [&](int npu_id) {
      if (pass_gen[npu_id] < 0) {
        return true;
      }
      if (pass_gen[npu_id] != state_gen) {
        return true;
      }
      if (pass_deadline[npu_id] > 0 &&
          static_cast<long long>(event_queue->get_current_time()) >=
              pass_deadline[npu_id]) {
        return true;
      }
      return false;
    };
    auto clear_suppression = [&]() {
      std::fill(pass_gen.begin(), pass_gen.end(), -1);
    };

    bool exit = false;
    while (!exit) {
      bool asked_any = false;
      if(!event_queue->finished()){
        event_queue->proceed();
      }
      else {
        event_queue->add_current_time();
        // Nothing is in flight, so no report can arrive to lift the
        // suppression; the 1 ms quantum is the only thing that moves.
        clear_suppression();
      }

      for (std::size_t idx = 0; idx < end_npu_ids.size(); ++idx) {
        int npu_id = end_npu_ids[idx];
        // Only proceed if the workload has finished its iteration
        if (!systems[npu_id]->workload->is_sleep &&
            systems[npu_id]->workload->is_finished && askable(npu_id)) {
          asked_any = true;
          if (static_cast<long long>(systems[npu_id]->workload->iteration) !=
              last_reported_iter[npu_id]) {
            // An iteration the frontend has not processed yet: its
            // scheduling decision for every other NPU may now differ.
            last_reported_iter[npu_id] = systems[npu_id]->workload->iteration;
            ++state_gen;
          }
          systems[npu_id]->workload->report();
          AstraSim::LoggerFactory::get_logger("workload")->info("Waiting");

          std::string new_filename;
          std::getline(std::cin, new_filename);

          if (new_filename.rfind("pass", 0) == 0) {
            // Nothing to run. Do not ask again until the frontend's answer
            // could differ: a new report from any NPU, or the arrival
            // deadline it optionally appended as "pass <tick>".
            const long long deadline = (new_filename.size() > 5)
                ? std::strtoll(new_filename.c_str() + 5, nullptr, 10)
                : 0;
            if (deadline < 0) {
              // "pass -1": the frontend changed scheduler state while
              // answering "pass" -- it joined a DP barrier round, or handed a
              // batch claim back. Such a pass is not idempotent, so re-open
              // every *other* NPU, exactly as a workload assignment does.
              // Without the bump the barrier stalls until the backstop fires,
              // which perturbed the dummy-wave count (moe_dp_pp 578 batches
              // -> 572, -4.0%).
              //
              // This NPU is suppressed at the *new* generation, not left
              // askable. Its own answer cannot change until somebody else
              // moves -- it is waiting on the rest of the round -- so leaving
              // it open just re-asked it every tick until the round closed.
              // That was 4.8 handshakes per DP wave against the 3 this
              // barrier needs (join, peer closes the round, pick up).
              ++state_gen;
              pass_gen[npu_id] = state_gen;
              pass_deadline[npu_id] = 0;
            } else {
              pass_gen[npu_id] = state_gen;
              pass_deadline[npu_id] = deadline;
            }
            continue;
          }

          // Any other answer changed the frontend's state, so every
          // suppressed NPU has to be re-asked -- not just this one. A
          // workload assignment is the case that matters: the instance's
          // other TP ranks have to pick that same batch up, and a DP peer
          // has to join the round it opens, and neither of those is
          // preceded by a report. Bumping only on reports left rank 1
          // suppressed until something else completed, which shifted
          // results on every tp>1 and DP configuration.
          ++state_gen;
          pass_gen[npu_id] = -1;

          if (new_filename == "exit") {
            // Terminate the entire simulator
            exit = true;
            break;
          }
          else if (new_filename == "done") {
            // This instance is done. Go to sleep until exit
            systems[npu_id]->workload->is_sleep = true;
          }
          else {
            // Add new workload to this system
            systems[npu_id]->workload
                ->add_workload(new_filename, {});
          }
        }
      }
      
      if (exit) {
        break;
      }

      for (std::size_t idx = 0; idx < start_npu_ids.size(); ++idx) {
        int npu_id = start_npu_ids[idx];
        // Only proceed if the workload has finished its iteration
        if (!systems[npu_id]->workload->is_sleep &&
            systems[npu_id]->workload->is_finished && askable(npu_id)) {
          asked_any = true;
          if (static_cast<long long>(systems[npu_id]->workload->iteration) !=
              last_reported_iter[npu_id]) {
            // An iteration the frontend has not processed yet: its
            // scheduling decision for every other NPU may now differ.
            last_reported_iter[npu_id] = systems[npu_id]->workload->iteration;
            ++state_gen;
          }
          systems[npu_id]->workload->report();
          AstraSim::LoggerFactory::get_logger("workload")->info("Waiting");

          std::string new_filename;
          std::getline(std::cin, new_filename);

          if (new_filename.rfind("pass", 0) == 0) {
            // Nothing to run. Do not ask again until the frontend's answer
            // could differ: a new report from any NPU, or the arrival
            // deadline it optionally appended as "pass <tick>".
            const long long deadline = (new_filename.size() > 5)
                ? std::strtoll(new_filename.c_str() + 5, nullptr, 10)
                : 0;
            if (deadline < 0) {
              // "pass -1": the frontend changed scheduler state while
              // answering "pass" -- it joined a DP barrier round, or handed a
              // batch claim back. Such a pass is not idempotent, so re-open
              // every *other* NPU, exactly as a workload assignment does.
              // Without the bump the barrier stalls until the backstop fires,
              // which perturbed the dummy-wave count (moe_dp_pp 578 batches
              // -> 572, -4.0%).
              //
              // This NPU is suppressed at the *new* generation, not left
              // askable. Its own answer cannot change until somebody else
              // moves -- it is waiting on the rest of the round -- so leaving
              // it open just re-asked it every tick until the round closed.
              // That was 4.8 handshakes per DP wave against the 3 this
              // barrier needs (join, peer closes the round, pick up).
              ++state_gen;
              pass_gen[npu_id] = state_gen;
              pass_deadline[npu_id] = 0;
            } else {
              pass_gen[npu_id] = state_gen;
              pass_deadline[npu_id] = deadline;
            }
            continue;
          }

          // Any other answer changed the frontend's state, so every
          // suppressed NPU has to be re-asked -- not just this one. A
          // workload assignment is the case that matters: the instance's
          // other TP ranks have to pick that same batch up, and a DP peer
          // has to join the round it opens, and neither of those is
          // preceded by a report. Bumping only on reports left rank 1
          // suppressed until something else completed, which shifted
          // results on every tp>1 and DP configuration.
          ++state_gen;
          pass_gen[npu_id] = -1;

          if (new_filename == "exit") {
            // Terminate the entire simulator
            exit = true;
            break;
          }
          else if (new_filename == "done") {
            // This instance is done. Go to sleep until exit
            systems[npu_id]->workload->is_sleep = true;
          }
          else {
            // Add new workload to the systems handled by this npu
            systems[npu_id]->workload
                ->add_workload(new_filename, managed_systems[idx]);
          }
        }
      }

      if (asked_any) {
        unasked_rounds = 0;
      } else if (++unasked_rounds >= kSuppressionBackstop) {
        // Should be unreachable: an in-flight workload always produces a
        // report, and an empty queue clears suppression outright. Kept so a
        // suppression bug degrades to a slow run instead of a silent hang.
        unasked_rounds = 0;
        clear_suppression();
      }
    }

    // check non exited system
    cout << "Checking Non-Exited Systems ..." << endl;
    bool done = true;
    for (int npu_id = 0; npu_id < npus_count; npu_id++) {

      if (systems[npu_id]->workload->is_finished == false){
        cout << "sys[" << npu_id << "] " << endl;
        systems[npu_id]->workload->et_feeder->printGraph();
        done = false;
      }
    }
    if (done){
      cout << "---------------------------" << endl;
      cout << "All Request Has Been Exited" << endl;
      cout << "---------------------------" << endl;
    }
    else{
      cout << "---------------------------" << endl;
      cout << "ERROR: Some Requests Remain" << endl;
      cout << "---------------------------" << endl;
    }

    // terminate simulation
    AstraSim::LoggerFactory::shutdown();
    return 0;
}
