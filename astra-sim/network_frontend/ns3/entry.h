#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include "astra-sim/network_frontend/ns3/ExperimentConfig.hh"
#include "common.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <stdexcept>
#include <vector>
#include <utility>

using namespace ns3;
using namespace std;

// This bridge owns the ns-3 side of ASTRA send/receive completion. A message
// remains logically complete only when both its sender and receiver callbacks
// have been resolved.
class MsgEvent {
  public:
    int src_id;
    int dst_id;
    int type;
    uint64_t remaining_msg_bytes;
    void* fun_arg;
    void (*msg_handler)(void* fun_arg);

    MsgEvent(int source,
             int destination,
             int event_type,
             uint64_t remaining_bytes,
             void* argument,
             void (*handler)(void*))
        : src_id(source),
          dst_id(destination),
          type(event_type),
          remaining_msg_bytes(remaining_bytes),
          fun_arg(argument),
          msg_handler(handler) {}

    MsgEvent()
        : src_id(0),
          dst_id(0),
          type(0),
          remaining_msg_bytes(0),
          fun_arg(nullptr),
          msg_handler(nullptr) {}

    void callHandler() {
        if (msg_handler == nullptr) {
            throw runtime_error("Missing ASTRA message completion callback");
        }
        msg_handler(fun_arg);
    }
};

using MsgEventKey = pair<int, pair<int, int>>;
using SenderKey = pair<int, pair<int, int>>;
using FlowKey = pair<uint16_t, pair<int, int>>;

map<SenderKey, int> sender_src_port_map;
map<pair<int, int>, uint64_t> node_to_bytes_sent_map;
map<pair<MsgEventKey, int>, MsgEvent> sim_send_waiting_hash;
map<MsgEventKey, MsgEvent> sim_recv_waiting_hash;
map<MsgEventKey, uint64_t> received_msg_standby_hash;
map<FlowKey, AstraSimNs3::FlowRecord> active_flow_registry;
uint64_t pending_background_flows = 0;
uint64_t completed_qps = 0;
bool transport_failure = false;
string transport_failure_message;
bool bridge_failure = false;
string bridge_failure_message;

FlowKey make_flow_key(uint16_t source_port, int src_id, int dst_id) {
    return make_pair(source_port, make_pair(src_id, dst_id));
}

// ns-3 gives every ordered host pair the source-port range [10000, 65535], and
// `common.h` seeds `portNumber` at the bottom of that range. A source port only
// names a live five-tuple: `qp_finish`/`qp_fail` delete both the sender and the
// receiver queue pair, after which the port is free again. Allocation therefore
// walks the range round-robin over the live set instead of consuming it
// monotonically, because a long chained collective sends every message of every
// step to the same ring neighbour and would otherwise cap one host pair at
// 55536 flows for the whole simulation.
constexpr uint16_t kFirstSourcePort = 10000;
constexpr uint32_t kSourcePortRange =
    static_cast<uint32_t>(numeric_limits<uint16_t>::max()) + 1 -
    kFirstSourcePort;

map<pair<int, int>, set<uint16_t>> live_source_ports;

uint16_t acquire_source_port(int src_id, int dst_id) {
    auto& live = live_source_ports[make_pair(src_id, dst_id)];
    uint16_t& cursor = portNumber[src_id][dst_id];
    for (uint32_t attempt = 0; attempt < kSourcePortRange; ++attempt) {
        if (cursor < kFirstSourcePort) {
            cursor = kFirstSourcePort;
        }
        const uint16_t candidate = cursor;
        cursor = candidate == numeric_limits<uint16_t>::max()
                     ? kFirstSourcePort
                     : static_cast<uint16_t>(candidate + 1);
        if (live.insert(candidate).second) {
            return candidate;
        }
    }
    throw runtime_error("Every source port of the host pair is already live");
}

void release_source_port(int src_id, int dst_id, uint16_t source_port) {
    const auto pair_ports = live_source_ports.find(make_pair(src_id, dst_id));
    if (pair_ports == live_source_ports.end() ||
        pair_ports->second.erase(source_port) != 1) {
        throw runtime_error("Released source port was not live");
    }
}

void account_physical_bytes(int src_id, int dst_id, uint64_t bytes) {
    node_to_bytes_sent_map[make_pair(src_id, 0)] += bytes;
    node_to_bytes_sent_map[make_pair(dst_id, 1)] += bytes;
}

bool has_pending_background_traffic() {
    return pending_background_flows != 0;
}

uint64_t completed_qp_count() {
    return completed_qps;
}

size_t active_qp_count() {
    return active_flow_registry.size();
}

bool has_transport_failure() {
    return transport_failure;
}

const string& current_transport_failure_message() {
    return transport_failure_message;
}

bool has_bridge_failure() {
    return bridge_failure;
}

const string& current_bridge_failure_message() {
    return bridge_failure_message;
}

// `AstraSim::Sys::call_events` wraps every callable in a catch-all that only
// logs, so an exception raised while starting a flow would otherwise leave the
// system layer waiting forever on a message ns-3 never carries. A bridge
// failure is an internal error, never a modeled transport outcome: it records
// the first cause and ends the run instead of producing a partial result that
// looks like a completed experiment.
void record_bridge_failure(const string& message) {
    if (!bridge_failure) {
        bridge_failure = true;
        bridge_failure_message = message;
    }
    Simulator::Stop();
}

void register_logical_send_event(int src_id,
                                 int dst_id,
                                 uint64_t bytes,
                                 void (*handler)(void*),
                                 void* argument,
                                 int tag,
                                 uint16_t source_port) {
    const auto key = make_pair(make_pair(tag, make_pair(src_id, dst_id)),
                               static_cast<int>(source_port));
    if (!sim_send_waiting_hash.emplace(
             key, MsgEvent(src_id, dst_id, 0, bytes, argument, handler))
             .second) {
        throw runtime_error("Duplicate logical ASTRA send event");
    }
}

// The transport's recovery-verdict callback. ns-3 knows a five-tuple and a
// byte range; this resolves the flow the way the telemetry join does, by
// (src, dst, source_port), and lets the experiment layer answer. An unknown
// five-tuple gets a repair with no allowance report: a range whose flow has
// already terminated cannot be charged to anything.
bool recovery_verdict(uint32_t sip,
                      uint32_t dip,
                      uint16_t sport,
                      uint16_t dport,
                      uint64_t seq,
                      uint32_t length) {
    (void)dport;
    const uint32_t src = ip_to_node_id(Ipv4Address(sip));
    const uint32_t dst = ip_to_node_id(Ipv4Address(dip));
    const FlowKey key = make_flow_key(sport, static_cast<int>(src),
                                      static_cast<int>(dst));
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        return false;
    }
    return AstraSimNs3::evaluate_forgiveness(active->second, seq, length);
}

// The transport's question about the report it is about to send: this flow is
// missing these bytes, is the step's allowance gone. Resolved the way the
// verdict callbacks are. An unknown five-tuple contributes no holes and
// reports none spent, which leaves its sender wherever it already was.
bool allowance_gone(uint32_t sip,
                    uint32_t dip,
                    uint16_t sport,
                    uint16_t dport,
                    uint64_t holes) {
    (void)dport;
    const uint32_t src = ip_to_node_id(Ipv4Address(sip));
    const uint32_t dst = ip_to_node_id(Ipv4Address(dip));
    const FlowKey key = make_flow_key(sport, static_cast<int>(src),
                                      static_cast<int>(dst));
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        return false;
    }
    return AstraSimNs3::note_holes(active->second, holes);
}

// The transport's remainder-verdict callback, asked when a receive queue pair
// has gone quiet with data still unsent. It resolves the flow the way the
// verdict callback does, and answers with the end offset to absorb. The
// transport supplies both edges of the hole, the cumulative sequence and the
// bytes accepted above it; only this side knows the flow's size. An unknown
// five-tuple refuses, which is what keeps a receive queue pair resurrected by
// late data inert: its flow is no longer registered.
uint64_t remainder_verdict(uint32_t sip,
                           uint32_t dip,
                           uint16_t sport,
                           uint16_t dport,
                           uint64_t next_expected,
                           uint64_t accepted_above) {
    (void)dport;
    const uint32_t src = ip_to_node_id(Ipv4Address(sip));
    const uint32_t dst = ip_to_node_id(Ipv4Address(dip));
    const FlowKey key = make_flow_key(sport, static_cast<int>(src),
                                      static_cast<int>(dst));
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        return 0;
    }
    return AstraSimNs3::evaluate_remainder(active->second, next_expected,
                                           accepted_above);
}

// Every logical flow this bridge starts is addressed to the same receiving
// port; the sender's port is what names the flow. Naming it once keeps the
// stop's five-tuple and the queue pair's the same by construction.
constexpr uint16_t kReceiverPort = 100;

// Stopping runs the remainder path, which acknowledges and mutates flow
// records; nothing in it receives data, so this guard is for a future change
// rather than for a path that exists.
bool stopping_sender = false;

// The receiver's budget says this sender has delivered 1 - p of what it owes
// for the step, so every open flow it still has into this rank is stopped now.
// The registry is walked rather than indexed: it holds the run's open flows,
// in the low thousands at 64 ranks, and a crossing happens once per (sender,
// step), so a walk costs less than a second index kept current on every start
// and completion. The keys are collected before any stop runs, because a
// stopped flow can complete and leave the registry.
void stop_sender_flows(uint32_t src, uint32_t dst, uint32_t step) {
    if (stopping_sender) {
        return;
    }
    stopping_sender = true;
    vector<FlowKey> stopping;
    for (const auto& entry : active_flow_registry) {
        const AstraSimNs3::FlowRecord& flow = entry.second;
        if (flow.src == static_cast<int>(src) &&
            flow.dst == static_cast<int>(dst) &&
            flow.operation.training_step == step &&
            AstraSimNs3::forgivable(flow)) {
            stopping.push_back(entry.first);
        }
    }
    for (const FlowKey& key : stopping) {
        StopFlowAtReceiver(dst, serverAddress[src].Get(),
                           serverAddress[dst].Get(), key.first, kReceiverPort);
    }
    stopping_sender = false;
}

// The transport's report that a receiving NIC accepted payload bytes. It
// resolves the flow the way the verdict callbacks do; an unknown five-tuple is
// a flow the experiment layer no longer holds, and nothing is credited to it.
void data_accepted(uint32_t sip,
                   uint32_t dip,
                   uint16_t sport,
                   uint16_t dport,
                   uint64_t bytes,
                   uint64_t late_forgiven) {
    (void)dport;
    const uint32_t src = ip_to_node_id(Ipv4Address(sip));
    const uint32_t dst = ip_to_node_id(Ipv4Address(dip));
    const FlowKey key = make_flow_key(sport, static_cast<int>(src),
                                      static_cast<int>(dst));
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        return;
    }
    if (AstraSimNs3::note_delivered(active->second, bytes, late_forgiven)) {
        stop_sender_flows(src, dst, active->second.operation.training_step);
    }
}

// The receiver's question about a flow it is about to acknowledge: may the
// experiment layer forgive it on this step. Asked once per receive queue pair
// and resolved the way the verdict callbacks are. An unknown five-tuple is not
// eligible, so its sender keeps obeying its controller.
bool forgiveness_eligible(uint32_t sip,
                          uint32_t dip,
                          uint16_t sport,
                          uint16_t dport) {
    (void)dport;
    const uint32_t src = ip_to_node_id(Ipv4Address(sip));
    const uint32_t dst = ip_to_node_id(Ipv4Address(dip));
    const FlowKey key = make_flow_key(sport, static_cast<int>(src),
                                      static_cast<int>(dst));
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        return false;
    }
    return AstraSimNs3::exemption_eligible(active->second);
}

void start_rdma_flow(AstraSimNs3::FlowRecord flow,
                     void (*msg_handler)(void*) = nullptr,
                     void* fun_arg = nullptr) {
    if (flow.src < 0 || flow.dst < 0 ||
        static_cast<size_t>(flow.src) >= serverAddress.size() ||
        static_cast<size_t>(flow.dst) >= serverAddress.size()) {
        throw runtime_error("Flow endpoints are outside the physical topology");
    }
    AstraSimNs3::validate_priority_group(flow.priority_group,
                                         "flow priority group");

    const uint16_t port = acquire_source_port(flow.src, flow.dst);
    flow.source_port = port;
    flow.start_time_ns = Simulator::Now().GetNanoSeconds();
    if (flow.kind != AstraSimNs3::FlowKind::BackgroundMicroburst) {
        sender_src_port_map.emplace(make_pair(port, make_pair(flow.src, flow.dst)),
                                    flow.tag);
        register_logical_send_event(flow.src, flow.dst, flow.logical_bytes,
                                    msg_handler, fun_arg, flow.tag, port);
    }

    const FlowKey key = make_flow_key(port, flow.src, flow.dst);
    if (!active_flow_registry.emplace(key, flow).second) {
        throw runtime_error("Duplicate active RDMA flow key");
    }
    flow_input.idx++;

    RdmaClientHelper client_helper(
        flow.priority_group, serverAddress[flow.src], serverAddress[flow.dst],
        port, kReceiverPort, flow.physical_bytes,
        has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(flow.src)][n.Get(flow.dst)])
                : 0,
        global_t == 1 ? maxRtt : pairRtt[flow.src][flow.dst], msg_handler,
        fun_arg, flow.tag, flow.src, flow.dst);
    ApplicationContainer applications = client_helper.Install(n.Get(flow.src));
    // Node::AddApplication schedules initialization at the current simulated
    // time. Application::DoInitialize then treats the configured start time as
    // a relative delay, so passing Simulator::Now() here makes every dynamic
    // QP wait for the current timestamp a second time. That compounds across
    // dependent collectives and can make a small run appear to hang.
    applications.Start(Time(0));
}

void start_background_flow(AstraSimNs3::MicroburstFlow* background) {
    try {
        AstraSimNs3::FlowRecord flow;
        flow.kind = AstraSimNs3::FlowKind::BackgroundMicroburst;
        flow.operation.transport_role = AstraSim::TransportRole::BackgroundTraffic;
        flow.origin_transport_role = AstraSim::TransportRole::BackgroundTraffic;
        flow.operation.training_step =
            AstraSimNs3::experiment_config.microburst_trigger_step;
        flow.src = static_cast<int>(background->src);
        flow.dst = static_cast<int>(background->dst);
        flow.priority_group = background->priority_group;
        flow.logical_bytes = background->size_bytes;
        flow.physical_bytes = background->size_bytes;
        start_rdma_flow(flow);
    } catch (const exception& error) {
        cerr << "Unable to start background microburst flow: " << error.what()
             << "\n";
        exit(EXIT_FAILURE);
    }
    delete background;
}

void maybe_trigger_microburst(const AstraSim::sim_request& request) {
    const auto& config = AstraSimNs3::experiment_config;
    if (!config.enabled || !config.microburst_enabled ||
        config.microburst_triggered ||
        !AstraSim::is_dp_all_reduce_payload(request.operation) ||
        request.operation.training_step != config.microburst_trigger_step) {
        return;
    }

    AstraSimNs3::experiment_config.microburst_triggered = true;
    for (const auto& flow : config.microburst_flows) {
        ++pending_background_flows;
        Simulator::Schedule(NanoSeconds(flow.offset_ns), &start_background_flow,
                            new AstraSimNs3::MicroburstFlow(flow));
    }
}

// Called by the ASTRA network API to start one reliable ns-3 RDMA QP.
void send_flow(int src_id,
               int dst_id,
               uint64_t message_size,
               void (*msg_handler)(void*),
               void* fun_arg,
               int tag,
               const AstraSim::sim_request& request) {
    maybe_trigger_microburst(request);
    const auto decision =
        AstraSimNs3::evaluate_shedding(request, src_id, dst_id, tag);

    AstraSimNs3::FlowRecord flow;
    flow.operation = request.operation;
    flow.origin_transport_role = request.operation.transport_role;
    flow.admission_eligible = decision.eligible;
    flow.decision_hash = decision.decision_hash;
    flow.src = src_id;
    flow.dst = dst_id;
    flow.tag = tag;
    flow.logical_bytes = message_size;
    flow.shed = decision.shed;

    if (decision.shed) {
        flow.kind = AstraSimNs3::FlowKind::ProvenanceControl;
        flow.operation.transport_role = AstraSim::TransportRole::ProvenanceControl;
        flow.priority_group =
            AstraSimNs3::experiment_config.provenance_priority_group;
        flow.physical_bytes =
            AstraSimNs3::experiment_config.provenance_control_bytes;
    } else {
        flow.kind = AstraSimNs3::FlowKind::ForegroundPayload;
        flow.priority_group = AstraSimNs3::priority_group_for_vnet(request.vnet);
        flow.physical_bytes = message_size;
    }
    // The budget is denominated in the bytes that were eligible for it, and
    // eligibility is decided here, at the sender, for the receiving rank that
    // will later be asked to forgive some of them.
    if (decision.eligible && dst_id >= 0) {
        const uint32_t dst = static_cast<uint32_t>(dst_id);
        const uint32_t step = request.operation.training_step;
        AstraSimNs3::forgiveness_ledger.register_eligible(dst, step,
                                                          message_size);
        if (decision.shed) {
            AstraSimNs3::forgiveness_ledger.register_shed(dst, step,
                                                          message_size);
        }
    }
    start_rdma_flow(flow, msg_handler, fun_arg);
}

void notify_receiver_receive_data(int src_id,
                                  int dst_id,
                                  uint64_t message_size,
                                  int tag,
                                  bool count_physical_bytes = true) {
    const MsgEventKey key = make_pair(tag, make_pair(src_id, dst_id));
    const auto expected = sim_recv_waiting_hash.find(key);
    if (expected != sim_recv_waiting_hash.end()) {
        MsgEvent event = expected->second;
        if (message_size == event.remaining_msg_bytes) {
            sim_recv_waiting_hash.erase(expected);
            event.callHandler();
        } else if (message_size > event.remaining_msg_bytes) {
            received_msg_standby_hash[key] =
                message_size - event.remaining_msg_bytes;
            sim_recv_waiting_hash.erase(expected);
            event.callHandler();
        } else {
            event.remaining_msg_bytes -= message_size;
            expected->second = event;
        }
    } else {
        received_msg_standby_hash[key] += message_size;
    }
    if (count_physical_bytes) {
        node_to_bytes_sent_map[make_pair(dst_id, 1)] += message_size;
    }
}

void notify_sender_sending_finished(int src_id,
                                    int dst_id,
                                    uint64_t message_size,
                                    int tag,
                                    uint16_t source_port) {
    const auto key = make_pair(make_pair(tag, make_pair(src_id, dst_id)),
                               static_cast<int>(source_port));
    const auto waiting = sim_send_waiting_hash.find(key);
    if (waiting == sim_send_waiting_hash.end()) {
        throw runtime_error("Cannot find the completed ASTRA send event");
    }
    if (waiting->second.remaining_msg_bytes != message_size) {
        throw runtime_error("Completed RDMA payload size does not match ASTRA");
    }
    MsgEvent event = waiting->second;
    sim_send_waiting_hash.erase(waiting);
    node_to_bytes_sent_map[make_pair(src_id, 0)] += message_size;
    event.callHandler();
}

void complete_logical_shed_sender(const AstraSimNs3::FlowRecord& flow) {
    const auto key = make_pair(make_pair(flow.tag, make_pair(flow.src, flow.dst)),
                               static_cast<int>(flow.source_port));
    const auto waiting = sim_send_waiting_hash.find(key);
    if (waiting == sim_send_waiting_hash.end()) {
        throw runtime_error("Cannot find provenance-controlled ASTRA send");
    }
    if (waiting->second.remaining_msg_bytes != flow.logical_bytes) {
        throw runtime_error("Provenance completion does not match logical size");
    }
    MsgEvent event = waiting->second;
    sim_send_waiting_hash.erase(waiting);
    event.callHandler();
}

void qp_finish_print_log(FILE* fout, Ptr<RdmaQueuePair> q) {
    const uint32_t sid = ip_to_node_id(q->sip);
    const uint32_t did = ip_to_node_id(q->dip);
    const uint64_t base_rtt = pairRtt[sid][did];
    const uint64_t bandwidth = pairBw[sid][did];
    const uint64_t total_bytes =
        q->m_size + ((q->m_size - 1) / packet_payload_size + 1) *
                        (CustomHeader::GetStaticWholeHeaderSize() -
                         IntHeader::GetStaticSize());
    const uint64_t standalone_fct =
        base_rtt + total_bytes * 8000000000ULL / bandwidth;
    fprintf(fout, "%08x %08x %u %u %llu %llu %llu %llu\n", q->sip.Get(),
            q->dip.Get(), q->sport, q->dport,
            static_cast<unsigned long long>(q->m_size),
            static_cast<unsigned long long>(q->startTime.GetTimeStep()),
            static_cast<unsigned long long>(
                (Simulator::Now() - q->startTime).GetTimeStep()),
            static_cast<unsigned long long>(standalone_fct));
    fflush(fout);
}

// Both terminal outcomes report the same transport counters, so they read them
// through one function: a counter added on one path and not the other would be
// a silent hole in exactly the failed flows that need diagnosing.
void copy_transport_counters(AstraSimNs3::FlowRecord& flow,
                             Ptr<RdmaQueuePair> q) {
    flow.physical_bytes = q->m_size;
    flow.data_attempted_bytes = q->m_data_attempted_bytes;
    flow.retransmitted_bytes = q->m_retransmitted_bytes;
    flow.trimmed_payload_bytes = q->m_trimmed_payload_bytes;
    flow.recovery_events = q->m_recovery_events;
    flow.trim_notifications = q->m_trim_notifications;
    flow.trim_lasthop_notifications = q->m_trim_lasthop_notifications;
    flow.trim_recovery_events = q->m_trim_recovery_events;
    flow.stale_trim_notifications = q->m_stale_trim_notifications;
    flow.timeouts = q->m_timeouts;
    flow.cnp_received = q->m_cnp_received;
    // Granted at some point, which is what the column means; the queue pair's
    // live flag is false again after a re-arm.
    flow.cc_exempt = q->m_cc_exempt_granted_ns != 0;
    flow.cc_exempt_granted_ns = q->m_cc_exempt_granted_ns;
    flow.cc_signal_withheld = q->m_cc_signals_withheld;
    flow.allowance_gone_reports = q->m_allowance_gone_reports;
    flow.cc_transitions = q->m_cc_transitions;
    // A queue pair still obeying when it ends closes its last stretch here,
    // because the transport only closes one when a report reopens it.
    flow.cc_obeying_ns =
        q->m_cc_obeying_ns +
        (q->m_cc_obey_since_ns != 0
             ? Simulator::Now().GetNanoSeconds() - q->m_cc_obey_since_ns
             : 0);
    flow.first_trim_ns = q->m_first_trim_ns;
    flow.first_repair_ns = q->m_first_repair_ns;
    flow.end_time_ns = Simulator::Now().GetNanoSeconds();
}

// Registered by common.h::SetupNetwork and invoked for every completed QP.
void qp_finish(FILE* fout, Ptr<RdmaQueuePair> q) {
    const uint32_t sid = ip_to_node_id(q->sip);
    const uint32_t did = ip_to_node_id(q->dip);
    qp_finish_print_log(fout, q);

    Ptr<Node> dst_node = n.Get(did);
    Ptr<RdmaDriver> rdma = dst_node->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);

    const FlowKey key = make_flow_key(q->sport, sid, did);
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        throw runtime_error("Completed QP has no active flow record");
    }
    // A terminal flow holds no holes: the cell's sum is over the flows still
    // receiving, and this one has stopped.
    AstraSimNs3::note_holes(active->second, 0);
    AstraSimNs3::FlowRecord flow = active->second;
    copy_transport_counters(flow, q);
    flow.terminal_outcome = AstraSimNs3::FlowTerminalOutcome::Completed;
    // The receiver credited every byte as it arrived, so by completion its
    // account must equal what the sender offered less what it was forgiven. A
    // mismatch means an arrival was counted twice or a forgiven range was
    // counted as data, either of which would move the budget the contract is
    // measured against.
    if (flow.accepted_bytes != AstraSimNs3::delivered_bytes(flow)) {
        throw runtime_error(
            "Receiver accepted " + to_string(flow.accepted_bytes) +
            " B for flow " + to_string(flow.src) + "->" + to_string(flow.dst) +
            " port " + to_string(flow.source_port) + " step " +
            to_string(flow.operation.training_step) + ", against " +
            to_string(q->m_size) + " B offered less " +
            to_string(flow.forgiven_bytes) + " B forgiven");
    }

    if (flow.kind == AstraSimNs3::FlowKind::BackgroundMicroburst) {
        account_physical_bytes(sid, did, q->m_size);
        if (pending_background_flows == 0) {
            throw runtime_error("Background flow completion accounting underflow");
        }
        --pending_background_flows;
    } else {
        const SenderKey sender_key =
            make_pair(static_cast<int>(q->sport), make_pair(sid, did));
        if (sender_src_port_map.erase(sender_key) != 1) {
            throw runtime_error("Completed QP has no logical sender tag");
        }
        if (flow.kind == AstraSimNs3::FlowKind::ProvenanceControl) {
            complete_logical_shed_sender(flow);
            notify_receiver_receive_data(sid, did, flow.logical_bytes,
                                         flow.tag, false);
            account_physical_bytes(sid, did, q->m_size);
        } else {
            notify_sender_sending_finished(sid, did, q->m_size, flow.tag,
                                           q->sport);
            notify_receiver_receive_data(sid, did, q->m_size, flow.tag);
        }
    }
    AstraSimNs3::experiment_telemetry.record_flow(flow);
    active_flow_registry.erase(active);
    // Release last: the ASTRA completion handlers above can start the next
    // message inline, and `RdmaHw::QpComplete` erases this five-tuple from the
    // sender only after this callback returns. Holding the port until here
    // keeps a nested send from reusing a port that is about to be torn down.
    release_source_port(sid, did, q->sport);
    ++completed_qps;
}

// Registered by common.h::SetupNetwork when bounded recovery exhausts a QP's
// retry budget. This is a terminal transport outcome, never a logical ASTRA
// message completion.
void qp_fail(FILE* fout, Ptr<RdmaQueuePair> q, uint32_t reason) {
    const uint32_t sid = ip_to_node_id(q->sip);
    const uint32_t did = ip_to_node_id(q->dip);
    const FlowKey key = make_flow_key(q->sport, sid, did);
    const auto active = active_flow_registry.find(key);
    if (active == active_flow_registry.end()) {
        throw runtime_error("Failed QP has no active flow record");
    }

    Ptr<Node> dst_node = n.Get(did);
    Ptr<RdmaDriver> rdma = dst_node->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);

    AstraSimNs3::note_holes(active->second, 0);
    AstraSimNs3::FlowRecord flow = active->second;
    copy_transport_counters(flow, q);
    flow.terminal_outcome = AstraSimNs3::FlowTerminalOutcome::Failed;
    flow.failure_reason =
        reason == static_cast<uint32_t>(RdmaFailureReason::TimeoutRetryExhausted)
            ? "retry_exhausted"
            : reason == static_cast<uint32_t>(RdmaFailureReason::TrimRetryExhausted)
                  ? "trim_retry_exhausted"
                  : reason == static_cast<uint32_t>(
                                  RdmaFailureReason::NoForwardProgress)
                        ? "no_forward_progress"
                        : "unknown";

    if (flow.kind == AstraSimNs3::FlowKind::BackgroundMicroburst) {
        if (pending_background_flows == 0) {
            throw runtime_error("Background flow failure accounting underflow");
        }
        --pending_background_flows;
    } else {
        const SenderKey sender_key =
            make_pair(static_cast<int>(q->sport), make_pair(sid, did));
        if (sender_src_port_map.erase(sender_key) != 1) {
            throw runtime_error("Failed QP has no logical sender tag");
        }
    }

    AstraSimNs3::experiment_telemetry.record_flow(flow);
    active_flow_registry.erase(active);
    release_source_port(sid, did, q->sport);
    transport_failure = true;
    transport_failure_message =
        "QP failed (" + flow.failure_reason + ") for " + to_string(sid) +
        "->" + to_string(did) + " source_port=" + to_string(q->sport) +
        " snd_una=" + to_string(q->snd_una) + "/" + to_string(q->m_size) +
        " recovery_events=" + to_string(q->m_recovery_events) +
        " trim_notifications=" + to_string(q->m_trim_notifications);
}

int setup_ns3_simulation(string network_configuration) {
    if (!ReadConf(network_configuration)) {
        return -1;
    }
    SetConfig();
    const bool recovery_domain =
        AstraSimNs3::experiment_config.enabled &&
        AstraSimNs3::forgives(AstraSimNs3::experiment_config.domain);
    const bool congestion_exempt =
        AstraSimNs3::experiment_config.enabled &&
        AstraSimNs3::experiment_config.domain ==
            AstraSimNs3::SheddingDomain::RecoveryExempt;
    if (recovery_domain) {
        // The experiment configuration asserts the transport contract; this is
        // where the assertion meets the transport that was actually built.
        if (selective_retransmission == 0 || packet_trim_mode != "ftd") {
            cerr << "Recovery domain requires SELECTIVE_RETRANSMISSION 1 and "
                    "PACKET_TRIM_MODE ftd in the network configuration\n";
            return -1;
        }
        if (load_balancing != "ecmp") {
            cerr << "Recovery domain and LOAD_BALANCING " << load_balancing
                 << " are mutually exclusive\n";
            return -1;
        }
    }
    // The step stop is on exactly when the profile asked for it, and the
    // callback being null is what tells the transport it is off.
    if (!SetupNetwork(qp_finish, qp_fail, recovery_verdict, recovery_domain,
                      forgiveness_eligible, congestion_exempt,
                      AstraSimNs3::experiment_config.step_stop
                          ? remainder_verdict
                          : nullptr,
                      data_accepted,
                      AstraSimNs3::experiment_config.reengage,
                      allowance_gone)) {
        return -1;
    }
    // The experiment's scale sizes the forgiveness ledger and bounds every
    // rank index it accepts, and the topology decides how many hosts exist.
    // A mismatch is silent otherwise: ranks above the ledger's width forgive
    // nothing, and the arm reads as a weak treatment effect.
    const uint32_t host_count = static_cast<uint32_t>(serverAddress.size());
    if (AstraSimNs3::experiment_config.rank_count != 0 &&
        AstraSimNs3::experiment_config.rank_count != host_count) {
        cerr << "Experiment scale.ranks "
             << AstraSimNs3::experiment_config.rank_count
             << " does not match the topology's " << host_count << " hosts\n";
        return -1;
    }
    return 0;
}
