#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <json/json.hpp>

// monkey patch, the spdlog include <syslog.h> and define these macros, and
// break the ns3 log enum keys
#define NS3_LOG_COMPAT_UNDEF_SYSLOG
#include "astra-sim/network_frontend/ns3/ns3_log_monkey_patch.h"

#include "entry.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/rng-seed-manager.h"

#undef NS3_LOG_COMPAT_UNDEF_SYSLOG
#include "astra-sim/network_frontend/ns3/ns3_log_monkey_patch.h"

#include <chrono>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace ns3;
using json = nlohmann::json;

/**
 * @class NS3BackendCompletionTracker
 * @brief Tracks the completion status of ranks in the NS3 backend.
 *
 * Each ASTRASimNetwork instance only corresponds to one rank, so this tracker
 * provides the process-wide completion condition. It intentionally stops the
 * simulator only after any scheduled background microbursts complete, allowing
 * telemetry to flush in main instead of calling exit() from a callback.
 */
class NS3BackendCompletionTracker {
  public:
    NS3BackendCompletionTracker(int num_ranks) {
        num_ranks_ = num_ranks;
        num_unfinished_ranks_ = num_ranks;
        last_unfinished_rank_count_ = num_ranks;
        completion_tracker_ = vector<int>(num_ranks, 0);
    }

    void mark_rank_as_finished(int rank) {
        if (completion_tracker_[rank] == 0) {
            completion_tracker_[rank] = 1;
            num_unfinished_ranks_--;
            AstraSimNs3::experiment_telemetry.record_rank_completion(
                rank, Simulator::Now().GetNanoSeconds());
        }
        if (num_unfinished_ranks_ == 0) {
            AstraSim::LoggerFactory::get_logger("network")->debug(
                "All ranks have finished. Waiting for background traffic.");
            stop_when_ready();
        }
    }

    void start_liveness_checkpoints() {
        Simulator::Schedule(kLivenessCheckpointInterval,
                            &NS3BackendCompletionTracker::liveness_checkpoint,
                            this);
    }

    bool is_complete() const {
        return num_unfinished_ranks_ == 0 && active_qp_count() == 0 &&
               !has_pending_background_traffic();
    }

  private:
    static constexpr uint64_t kLivenessCheckpointIntervalNs = 10000000;
    // A run whose network is idle can still be making progress inside the
    // system layer, but only for as long as compute and memory events take.
    // Ten simulated seconds of an idle network with no completion at all is a
    // deadlock, not slow progress, and the checkpoint is the only event left
    // to observe it: without this the simulator advances its clock forever at
    // no wall-clock cost and the job dies on its CI timeout with no diagnosis.
    static constexpr uint64_t kQuiescentCheckpointsBeforeStall = 1000;
    static const Time kLivenessCheckpointInterval;

    // Deliberately ignores whether QPs are "active": the observed wedge is a
    // lone registered flow whose queue pair has no pending events at all, so
    // requiring active_qp_count() == 0 made this detector blind to exactly
    // the state it exists to catch. Ten simulated seconds without a single
    // QP or rank completing is a stall regardless of what the registry says.
    bool is_quiescent() const {
        return !has_pending_background_traffic() &&
               completed_qp_count() == last_completed_qp_count_ &&
               num_unfinished_ranks_ == last_unfinished_rank_count_;
    }

    void liveness_checkpoint() {
        // Self-profiling: the wall-clock and event-count deltas since the
        // previous checkpoint make every run its own profiler. Per-event wall
        // cost separates "each event became slower" (a code or build
        // regression) from "the same simulated interval now executes more
        // events" (a behavioral regression), a distinction no external
        // timeout can make after killing the process.
        const auto wall_now = std::chrono::steady_clock::now();
        const uint64_t events_now = Simulator::GetEventCount();
        // Resident set in MB, from statm page counts. A hosted CI runner that
        // runs out of memory kills the runner agent and reports only "lost
        // communication with the server"; this field is the only way to tell
        // that death from an infrastructure flake after the fact.
        uint64_t rss_mb = 0;
        if (FILE* statm = fopen("/proc/self/statm", "r")) {
            unsigned long size_pages = 0, resident_pages = 0;
            if (fscanf(statm, "%lu %lu", &size_pages, &resident_pages) == 2) {
                rss_mb = resident_pages *
                         static_cast<uint64_t>(sysconf(_SC_PAGESIZE)) >> 20;
            }
            fclose(statm);
        }
        const uint64_t wall_ms_delta =
            last_checkpoint_wall_.time_since_epoch().count() == 0
                ? 0
                : std::chrono::duration_cast<std::chrono::milliseconds>(
                      wall_now - last_checkpoint_wall_)
                      .count();
        const uint64_t events_delta =
            events_now >= last_checkpoint_events_
                ? events_now - last_checkpoint_events_
                : 0;
        last_checkpoint_wall_ = wall_now;
        last_checkpoint_events_ = events_now;
        AstraSim::LoggerFactory::get_logger("network")->info(
            "Liveness checkpoint: simulated_time_ns={} completed_qps={} "
            "active_qps={} completed_ranks={}/{} pending_background_flows={} "
            "wall_ms_delta={} events_delta={} rss_mb={}",
            Simulator::Now().GetNanoSeconds(), completed_qp_count(),
            active_qp_count(), num_ranks_ - num_unfinished_ranks_, num_ranks_,
            pending_background_flows, wall_ms_delta, events_delta, rss_mb);
        if (is_complete()) {
            return;
        }
        if (has_transport_failure() && active_qp_count() == 0 &&
            !has_pending_background_traffic()) {
            AstraSim::LoggerFactory::get_logger("network")->info(
                "Transport failure reached quiescence; stopping simulation.");
            Simulator::Stop();
            return;
        }
        quiescent_checkpoints_ = is_quiescent() ? quiescent_checkpoints_ + 1 : 0;
        last_completed_qp_count_ = completed_qp_count();
        last_unfinished_rank_count_ = num_unfinished_ranks_;
        if (quiescent_checkpoints_ >= kQuiescentCheckpointsBeforeStall) {
            AstraSim::LoggerFactory::get_logger("network")->critical(
                "Simulation stalled: no flow or rank completed in the last {} "
                "ns of simulated time while {} of {} ranks were unfinished "
                "and {} flows were registered as active.",
                kQuiescentCheckpointsBeforeStall * kLivenessCheckpointIntervalNs,
                num_unfinished_ranks_, num_ranks_, active_qp_count());
            for (const auto& entry : active_flow_registry) {
                const auto& flow = entry.second;
                AstraSim::LoggerFactory::get_logger("network")->critical(
                    "Stalled flow: {}->{} source_port={} logical_bytes={} "
                    "started_at_ns={}",
                    flow.src, flow.dst, flow.source_port, flow.logical_bytes,
                    flow.start_time_ns);
            }
            Simulator::Stop();
            return;
        }
        Simulator::Schedule(kLivenessCheckpointInterval,
                            &NS3BackendCompletionTracker::liveness_checkpoint,
                            this);
    }

    void stop_when_ready() {
        if (num_unfinished_ranks_ != 0) {
            return;
        }
        if (has_pending_background_traffic()) {
            Simulator::Schedule(MicroSeconds(1),
                                &NS3BackendCompletionTracker::stop_when_ready,
                                this);
            return;
        }
        Simulator::Stop();
    }

    int num_ranks_;
    int num_unfinished_ranks_;
    uint64_t last_completed_qp_count_ = 0;
    int last_unfinished_rank_count_ = 0;
    uint64_t quiescent_checkpoints_ = 0;
    std::chrono::steady_clock::time_point last_checkpoint_wall_{};
    uint64_t last_checkpoint_events_ = 0;
    vector<int> completion_tracker_;
};

const Time NS3BackendCompletionTracker::kLivenessCheckpointInterval =
    NanoSeconds(NS3BackendCompletionTracker::kLivenessCheckpointIntervalNs);

class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
  public:
    ASTRASimNetwork(int rank, NS3BackendCompletionTracker* completion_tracker)
        : AstraNetworkAPI(rank) {
        completion_tracker_ = completion_tracker;
    }

    ~ASTRASimNetwork() {}

    void sim_notify_finished() {
        // Output to file instead of stdout
        /*
        for (auto it = node_to_bytes_sent_map.begin();
             it != node_to_bytes_sent_map.end(); it++) {
            pair<int, int> p = it->first;
            if (p.second == 0) {
                cout << "All data sent from node " << p.first << " is "
                     << it->second << "\n";
            } else {
                cout << "All data received by node " << p.first << " is "
                     << it->second << "\n";
            }
        }
        */
        completion_tracker_->mark_rank_as_finished(rank);
        return;
    }

    double sim_time_resolution() {
        return 0;
    }

    void handleEvent(int dst, int cnt) {}

    AstraSim::timespec_t sim_get_time() {
        AstraSim::timespec_t timeSpec;
        timeSpec.time_res = AstraSim::NS;
        timeSpec.time_val = Simulator::Now().GetNanoSeconds();
        return timeSpec;
    }

    void sim_record_collective_completion(
        const AstraSim::OperationContext& operation,
        uint64_t logical_bytes,
        uint64_t start_time_ns,
        uint64_t end_time_ns) override {
        AstraSimNs3::experiment_telemetry.record_collective_completion(
            rank, operation, logical_bytes, start_time_ns, end_time_ns);
    }

    virtual void sim_schedule(AstraSim::timespec_t delta,
                              void (*fun_ptr)(void* fun_arg),
                              void* fun_arg) {
        Simulator::Schedule(NanoSeconds(delta.time_val), fun_ptr, fun_arg);
        return;
    }

    virtual int sim_send(void* buffer,
                         uint64_t message_size,
                         int type,
                         int dst_id,
                         int tag,
                         AstraSim::sim_request* request,
                         void (*msg_handler)(void* fun_arg),
                         void* fun_arg) {
        int src_id = rank;

        // Trigger ns3 to schedule RDMA QP event.
        if (request == nullptr) {
            throw runtime_error("ns-3 sim_send requires a sim_request");
        }
        // Sys::call_events swallows exceptions raised by a callable, so an
        // unstartable flow has to be recorded here or the run would deadlock
        // silently instead of failing.
        try {
            send_flow(src_id, dst_id, message_size, msg_handler, fun_arg, tag,
                      *request);
        } catch (const exception& error) {
            const string message = "Unable to start flow " +
                                   to_string(src_id) + "->" +
                                   to_string(dst_id) + ": " + error.what();
            AstraSim::LoggerFactory::get_logger("network")->critical(
                "ns-3 bridge failure at {} ns; stopping simulation: {}",
                Simulator::Now().GetNanoSeconds(), message);
            record_bridge_failure(message);
            throw;
        }
        return 0;
    }

    virtual int sim_recv(void* buffer,
                         uint64_t message_size,
                         int type,
                         int src_id,
                         int tag,
                         AstraSim::sim_request* request,
                         void (*msg_handler)(void* fun_arg),
                         void* fun_arg) {
        int dst_id = rank;
        MsgEvent recv_event =
            MsgEvent(src_id, dst_id, 1, message_size, fun_arg, msg_handler);
        MsgEventKey recv_event_key =
            make_pair(tag, make_pair(recv_event.src_id, recv_event.dst_id));

        if (received_msg_standby_hash.find(recv_event_key) !=
            received_msg_standby_hash.end()) {
            // 1) ns3 has already received some message before sim_recv is
            // called.
            uint64_t received_msg_bytes =
                received_msg_standby_hash[recv_event_key];
            if (received_msg_bytes == message_size) {
                // 1-1) The received message size is same as what we expect.
                // Exit.
                received_msg_standby_hash.erase(recv_event_key);
                recv_event.callHandler();
            } else if (received_msg_bytes > message_size) {
                // 1-2) The node received more than expected.
                // Do trigger the callback handler for this message, but wait
                // for Sys layer to call sim_recv for more messages.
                received_msg_standby_hash[recv_event_key] =
                    received_msg_bytes - message_size;
                recv_event.callHandler();
            } else {
                // 1-3) The node received less than what we expected.
                // Reduce the number of bytes we are waiting to receive.
                received_msg_standby_hash.erase(recv_event_key);
                recv_event.remaining_msg_bytes -= received_msg_bytes;
                sim_recv_waiting_hash[recv_event_key] = recv_event;
            }
        } else {
            // 2) ns3 has not yet received anything.
            if (sim_recv_waiting_hash.find(recv_event_key) ==
                sim_recv_waiting_hash.end()) {
                // 2-1) We have not been expecting anything.
                sim_recv_waiting_hash[recv_event_key] = recv_event;
            } else {
                // 2-2) We have already been expecting something.
                // Increment the number of bytes we are waiting to receive.
                uint64_t expecting_msg_bytes =
                    sim_recv_waiting_hash[recv_event_key].remaining_msg_bytes;
                recv_event.remaining_msg_bytes += expecting_msg_bytes;
                sim_recv_waiting_hash[recv_event_key] = recv_event;
            }
        }
        return 0;
    }

  private:
    NS3BackendCompletionTracker* completion_tracker_;
};

// Command line arguments and default values.
string workload_configuration;
string system_configuration;
string network_configuration;
string memory_configuration;
string comm_group_configuration = "empty";
string logical_topology_configuration;
string logging_configuration = "empty";
string experiment_configuration = "empty";
string experiment_output_dir = "empty";
string clr_mask_configuration = "empty";
int num_queues_per_dim = 1;
double comm_scale = 1;
double injection_scale = 1;
bool rendezvous_protocol = false;
uint32_t ns3_rng_seed = 1;
uint64_t ns3_rng_run = 1;
auto logical_dims = vector<int>();
int num_npus = 1;
auto queues_per_dim = vector<int>();

// TODO: Migrate to yaml
void read_logical_topo_config(string network_configuration,
                              vector<int>& logical_dims) {
    ifstream inFile;
    inFile.open(network_configuration);
    if (!inFile) {
        cerr << "Unable to open file: " << network_configuration << endl;
        exit(1);
    }

    // Find the size of each dimension.
    json j;
    inFile >> j;
    if (j.contains("logical-dims")) {
        vector<string> logical_dims_str_vec = j["logical-dims"];
        for (auto logical_dims_str : logical_dims_str_vec) {
            logical_dims.push_back(stoi(logical_dims_str));
        }
    }

    // Find the number of all npus.
    stringstream dimstr;
    for (auto num_npus_per_dim : logical_dims) {
        num_npus *= num_npus_per_dim;
        dimstr << num_npus_per_dim << ",";
    }
    cout << "There are " << num_npus << " npus: " << dimstr.str() << "\n";

    queues_per_dim = vector<int>(logical_dims.size(), num_queues_per_dim);
}

// Read command line arguments.
void parse_args(int argc, char* argv[]) {
    CommandLine cmd;
    cmd.AddValue("workload-configuration", "Workload configuration file.",
                 workload_configuration);
    cmd.AddValue("system-configuration", "System configuration file",
                 system_configuration);
    cmd.AddValue("network-configuration", "Network configuration file",
                 network_configuration);
    cmd.AddValue("remote-memory-configuration", "Memory configuration file",
                 memory_configuration);
    cmd.AddValue("comm-group-configuration",
                 "Communicator group configuration file",
                 comm_group_configuration);
    cmd.AddValue("logical-topology-configuration",
                 "Logical topology configuration file",
                 logical_topology_configuration);
    cmd.AddValue("logging-configuration", "Logging configuration file",
                 logging_configuration);
    cmd.AddValue("experiment-configuration",
                 "Experiment policy and microburst configuration file",
                 experiment_configuration);
    cmd.AddValue("experiment-output-dir",
                 "Directory for experiment flow and completion telemetry",
                 experiment_output_dir);
    cmd.AddValue("clr-mask-configuration",
                 "CSV mapping training steps to static CLR states",
                 clr_mask_configuration);
    cmd.AddValue("ns3-rng-seed", "Seed for ns-3 random streams", ns3_rng_seed);
    cmd.AddValue("ns3-rng-run", "Run number for ns-3 random streams", ns3_rng_run);

    cmd.AddValue("num-queues-per-dim", "Number of queues per each dimension",
                 num_queues_per_dim);
    cmd.AddValue("comm-scale", "Communication scale", comm_scale);
    cmd.AddValue("injection-scale", "Injection scale", injection_scale);
    cmd.AddValue("rendezvous-protocol", "Whether to enable rendezvous protocol",
                 rendezvous_protocol);

    cmd.Parse(argc, argv);
}

// The step plan, read out of ASTRA-sim's own collective code before anything
// launches. Every rank is asked what its DP All-Reduces would send and to
// whom; rank r's answer for peer p becomes p's owed from r, and the count of
// r's own collectives tells the ledger which completion certifies the step.
// One call installs the whole table, because a cell whose senders are still
// being added would measure its budget against a total that is not yet the
// total.
void install_collective_plan(const vector<AstraSim::Sys*>& systems) {
    if (!AstraSimNs3::experiment_config.enabled ||
        !AstraSimNs3::forgives(AstraSimNs3::experiment_config.domain)) {
        return;
    }
    AstraSimNs3::CollectivePlan plan;
    for (AstraSim::Sys* system : systems) {
        const uint32_t sender = static_cast<uint32_t>(system->id);
        for (const auto& entry : system->workload->plan_dp_all_reduce()) {
            const uint32_t step = entry.first;
            plan[{sender, step}].collectives += entry.second.collectives;
            for (const auto& peer : entry.second.bytes_by_peer) {
                if (peer.first < 0) {
                    throw runtime_error("a collective plan named a negative "
                                        "peer");
                }
                plan[{static_cast<uint32_t>(peer.first), step}]
                    .by_sender[sender] += peer.second;
            }
        }
    }
    AstraSimNs3::install_collective_plan(plan);
}

int main(int argc, char* argv[]) {
    LogComponentEnable("OnOffApplication", LOG_INFO);
    LogComponentEnable("PacketSink", LOG_INFO);

    cout << "ASTRA-sim + NS3" << endl;

    // Read network config and find logical dims.
    parse_args(argc, argv);
    if (ns3_rng_seed == 0 || ns3_rng_run == 0) {
        cerr << "ns3-rng-seed and ns3-rng-run must be nonzero\n";
        return 1;
    }
    RngSeedManager::SetSeed(ns3_rng_seed);
    RngSeedManager::SetRun(ns3_rng_run);
    AstraSim::LoggerFactory::init(logging_configuration);
    read_logical_topo_config(logical_topology_configuration, logical_dims);

    // Setup network & System layer.
    vector<ASTRASimNetwork*> networks(num_npus, nullptr);
    vector<AstraSim::Sys*> systems(num_npus, nullptr);
    Analytical::AnalyticalRemoteMemory* mem =
        new Analytical::AnalyticalRemoteMemory(memory_configuration);
    NS3BackendCompletionTracker* completion_tracker =
        new NS3BackendCompletionTracker(num_npus);

    for (int npu_id = 0; npu_id < num_npus; npu_id++) {
        networks[npu_id] = new ASTRASimNetwork(npu_id, completion_tracker);
        systems[npu_id] = new AstraSim::Sys(
            npu_id, workload_configuration, comm_group_configuration,
            system_configuration, mem, networks[npu_id], logical_dims,
            queues_per_dim, injection_scale, comm_scale, rendezvous_protocol);
    }

    // The experiment configuration is parsed first: the shedding domain
    // decides whether ns-3 is built with the recovery-verdict fork, so the
    // network cannot be set up before it is known.
    try {
        AstraSimNs3::configure_experiment(experiment_configuration,
                                          experiment_output_dir);
        AstraSimNs3::configure_clr_mask(clr_mask_configuration);
        AstraSimNs3::validate_experiment_contract();
        install_collective_plan(systems);
    } catch (const exception& error) {
        cerr << "Unable to configure experiment: " << error.what() << "\n";
        Simulator::Destroy();
        return 1;
    }

    // Initialize ns3 simulation.
    if (auto ok = setup_ns3_simulation(network_configuration); ok == -1) {
        std::cerr << "Fail to setup ns3 simulation." << std::endl;
        return -1;
    }

    // Telemetry last: a refused arm must not leave a run directory holding
    // headers and no rows, which every reader takes for a started run.
    try {
        AstraSimNs3::open_experiment_telemetry();
    } catch (const exception& error) {
        cerr << "Unable to open experiment telemetry: " << error.what() << "\n";
        Simulator::Destroy();
        return 1;
    }

    // Tell workload layer to schedule first events.
    for (int i = 0; i < num_npus; i++) {
        systems[i]->workload->fire();
    }
    completion_tracker->start_liveness_checkpoints();

    // Run the simulation by triggering the ns3 event queue.
    Simulator::Run();
    write_fabric_counters();
    AstraSimNs3::finalize_experiment_telemetry();
    Simulator::Destroy();
    if (has_bridge_failure()) {
        cerr << "Simulation stopped after an ns-3 bridge failure: "
             << current_bridge_failure_message() << "\n";
        return 1;
    }
    if (has_transport_failure()) {
        cerr << "Simulation stopped after explicit transport failure: "
             << current_transport_failure_message() << "\n";
        return 1;
    }
    if (!completion_tracker->is_complete()) {
        cerr << "Simulation ended before all ranks, QPs, and background flows completed\n";
        return 1;
    }
    return 0;
}
