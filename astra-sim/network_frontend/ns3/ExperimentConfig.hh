/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_NS3_EXPERIMENT_CONFIG_HH__
#define __ASTRA_SIM_NS3_EXPERIMENT_CONFIG_HH__

#include "astra-sim/system/Common.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <json/json.hpp>

namespace AstraSimNs3 {

constexpr uint64_t kDecisionScale = 1000000;

// One packet's payload, which is the largest range a single trim can carry.
// The generator writes it as `network.packet_payload_bytes` into
// network_config.txt and every profile in the tree uses this value; the spent
// report below is its only reader, and a smaller payload would only make that
// report fire marginally later.
constexpr uint64_t kPacketPayload = 4096;
constexpr uint16_t kPriorityGroupCount = 8;

inline uint64_t mix_hash(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

inline void hash_combine(uint64_t& hash, uint64_t value) {
    hash = mix_hash(hash ^ mix_hash(value));
}

enum class FlowKind : uint8_t {
    ForegroundPayload = 0,
    ProvenanceControl,
    BackgroundMicroburst,
};

enum class FlowTerminalOutcome : uint8_t {
    Pending = 0,
    Completed,
    Failed,
};

// Where the phase-aware budget is spent. Admission decides before a payload is
// offered and substitutes the whole message. Recovery decides after a switch
// has already trimmed a packet and lets that packet's bytes go, which is the
// only point at which the policy can act during a congestion episode.
// RecoveryExempt forgives the same way and additionally lets an eligible flow
// ignore congestion signals until the receiver refuses to forgive one of its
// trims, so the budget is paid in bounded loss rather than in rate.
enum class SheddingDomain : uint8_t {
    Admission = 0,
    Recovery,
    RecoveryExempt,
};

// Whether a domain decides after the trim. The one eliminator every test of
// "is this a forgiving domain" goes through, exhaustive so that a fourth
// variant fails to compile here rather than falling through to admission.
constexpr bool forgives(SheddingDomain domain) {
    switch (domain) {
        case SheddingDomain::Admission:
            return false;
        case SheddingDomain::Recovery:
            return true;
        case SheddingDomain::RecoveryExempt:
            return true;
    }
    return false;
}

// The semantics string a profile must name for its domain. Exhaustive for the
// same reason: the string is the contract the generator writes and this reads.
constexpr const char* selection_semantics(SheddingDomain domain) {
    switch (domain) {
        case SheddingDomain::Admission:
            return "logical_admission_selection";
        case SheddingDomain::Recovery:
            return "recovery_forgiveness";
        case SheddingDomain::RecoveryExempt:
            return "recovery_forgiveness_cc_exempt";
    }
    return "";
}

// What one sender owes a receiving rank for one step, and what of it has
// arrived. The step stop is a per-sender decision, so the receiver keeps the
// pair per sender as well as in the cell's totals.
struct SenderShare {
    uint64_t owed = 0;
    uint64_t delivered = 0;
};

// One receiving rank's budget for one training step. Every counter is
// monotone and forgiven + delivered never exceeds eligible.
struct StepLedger {
    // Bytes launched toward this rank this step. The budget law does not read
    // it; `close` and the analyzer certify against it.
    uint64_t eligible = 0;
    uint64_t shed = 0;
    uint64_t forgiven = 0;
    // Bytes of completed messages, less the bytes those messages were
    // forgiven. With forgiven, this is what the rank has accounted for, and
    // the budget law measures itself against the pair.
    uint64_t delivered = 0;
    // What the step's plan says this rank is owed, and by whom. The plan is a
    // function of the profile, so the generator computes it and the ledger
    // opens the cell with it; `close` refuses a run whose launches disagree.
    // Senders stay in the single digits per cell, so a map costs less than a
    // row per rank.
    uint64_t owed = 0;
    std::unordered_map<uint32_t, SenderShare> by_sender;
    // The bytes this rank is missing right now, summed over its receiving
    // flows: below the highest sequence each has seen, neither received nor
    // forgiven. It falls on its own as repairs land, which is what lets the
    // report turn green again.
    uint64_t holes = 0;
    // How many DP All-Reduce collectives the plan says this rank runs in this
    // step, and how many have completed. The step's bytes are accounted for
    // when the last one does, and that is when the cell is certified.
    uint32_t collectives = 0;
    uint32_t completed = 0;
    // Whether this cell affords against `owed` rather than against what the
    // rank has accounted for, which is the ablation. The plan is there either
    // way, because the report and the certification read it.
    bool owed_base = false;
};

// The pacing rule, as a closed sum. Pacing lets the receiver decline a
// forgivable trim so that allowance remains for later in the step, because
// the cap binds at the headline budget and first come first served spends it
// on the first burst. A Bernoulli probability is meaningful only under
// Bernoulli, so it lives inside that alternative and nowhere else.
struct NoPacing {};
struct Bernoulli {
    uint64_t threshold = 0;  // p * kDecisionScale
};
using Pacing = std::variant<NoPacing, Bernoulli>;

// std::visit over a lambda set. Every visit below lists both alternatives, so
// a third one fails to compile rather than defaulting.
template <class... Cases>
struct overloaded : Cases... {
    using Cases::operator()...;
};
template <class... Cases>
overloaded(Cases...) -> overloaded<Cases...>;

// Whether the rule declines a range the cap could still afford. The coin
// reserves allowance for the end of the step; NoPacing reserves nothing and
// never declines.
inline bool paces_out(const Pacing& pacing, uint64_t coin) {
    return std::visit(
        overloaded{
            [](const NoPacing&) { return false; },
            [&](const Bernoulli& rule) { return coin >= rule.threshold; },
        },
        pacing);
}

// The safety law, in one place and the same under every rule: shed and
// forgiven bytes share one budget, measured against the cell's base. The law
// of record is the vesting cap, the accounted base: the receiver measures the
// budget against the bytes it has accounted for, kept or forgiven. That law
// is
// `forgiven <= p x (delivered + forgiven)`, equivalently
// `forgiven <= p / (1 - p) x delivered`, and a receiving NIC holds both
// counters while it never sees what a sender launched. At the end of a step
// `delivered + forgiven = eligible`, so the ceiling is `p x eligible`, which
// is the v1 law; `eligible` survives only as the denominator the
// certification and the analyzer measure against. Under the owed base, which
// is the ablation, the cap is `forgiven <= p x owed` with the step's total
// from the plan, so the whole of it is available from the step's first
// packet. Pacing does not enter here, because the coin declines a range the
// budget affords rather than changing what it affords.
//
// The two agree at step end, where the certification holds the plan to the
// launches.
//
// Absorbing under both: neither term ever decreases, and under the accounted
// base a further spend adds kDecisionScale to the charge for every threshold
// it adds to the base, so spending never buys room.
inline bool affords_soft(const StepLedger& cell,
                         uint64_t threshold,
                         uint64_t bytes) {
    const uint64_t spent = cell.shed + cell.forgiven + bytes;
    const uint64_t base =
        cell.owed_base ? cell.owed : cell.delivered + spent;
    return spent * kDecisionScale <= base * threshold;
}

// The coin one trimmed arrival draws. A fresh draw per arrival, over the
// flow, the range and the flow's count of verdicts asked, so a range the coin
// refused is asked again on its next trim and meets the cap as it stands then,
// which under the receiver-local law has grown with delivery in the meantime.
// Nothing about a range is remembered between trims. No ns-3 random stream is
// consumed and the attempt counter is deterministic, so paired arms draw the
// same sequence.
inline uint64_t range_coin(uint64_t decision_hash,
                           uint64_t start,
                           uint32_t attempt) {
    uint64_t coin = decision_hash;
    hash_combine(coin, start);
    hash_combine(coin, attempt);
    return coin % kDecisionScale;
}

// The report the sender acts on: even if every byte the rank is missing right
// now were forgiven, this step's tolerance would be exceeded. Forgiven bytes
// and holes are exclusive by construction, because a forgiven range is
// absorbed as received, so nothing is counted twice; one packet's payload is
// added because that is the largest range the next trim can carry.
//
// It is not a latch. `forgiven` only grows, but the holes drain as repairs
// land, so a cell that reported gone reports room again once the fabric
// recovers, and the sender follows whichever report reached it last.
inline bool budget_gone(const StepLedger& cell, uint64_t threshold) {
    const uint64_t missing = cell.forgiven + cell.holes + kPacketPayload;
    return missing * kDecisionScale > cell.owed * threshold;
}

// Trim verdict. Pure and total: the cell, the rule and this range's coin in,
// forgive or repair and the cell to store out. The caller has already
// eliminated a missing cell and an ineligible flow. The report the sender
// acts on is not here: it is a property of the cell at the moment an
// acknowledgement or a repair request leaves the receiver, and a refusal says
// nothing about it either way.
inline std::pair<bool, StepLedger> trim_verdict(StepLedger cell,
                                                uint64_t threshold,
                                                const Pacing& pacing,
                                                uint64_t coin,
                                                uint64_t bytes) {
    const bool forgive =
        affords_soft(cell, threshold, bytes) && !paces_out(pacing, coin);
    if (forgive) {
        cell.forgiven += bytes;
    }
    return {forgive, cell};
}

// Whether the receiver has heard enough from one sender to stop it. The step
// stop's rule: once `1 - p` of what that sender owes this rank for this step
// has arrived, nothing it still holds is worth waiting for, so the receiver
// takes the rest. In integers at the decision scale, `delivered >= (1 - p) x
// owed` is `delivered x S >= (S - t) x owed`. A sender the step's plan does
// not name owes nothing and is never stopped.
inline bool sender_stopped(const StepLedger& cell,
                           uint32_t src,
                           uint64_t threshold) {
    const auto share = cell.by_sender.find(src);
    if (share == cell.by_sender.end() || share->second.owed == 0) {
        return false;
    }
    return share->second.delivered * kDecisionScale >=
           (kDecisionScale - threshold) * share->second.owed;
}

// Remainder verdict. Pure and total: the bytes to forgive, zero to refuse.
// Whole remainder or nothing, because the receiver knows the byte count and
// not which gradient elements matter, so it does not choose among them. The
// coin never applies here: pacing exists to keep allowance for the end of the
// step, and the stop fires at the end of the step, so the rule is not an
// argument.
//
// The stop is the trigger and the step pool is the ceiling. A remainder that
// fits the pool while its sender is still owed bytes is a flow the receiver
// still wants, so the sender must be stopped before the pool is even asked;
// the pool is what is left of the step's tolerance, `p x owed - forgiven`,
// and not the vesting cap, because the stop fires at the end of the step and
// takes what the step can still afford.
inline std::pair<uint64_t, StepLedger> remainder_verdict(StepLedger cell,
                                                         uint64_t threshold,
                                                         uint64_t remainder,
                                                         uint32_t src,
                                                         bool step_stop) {
    if (remainder == 0 || !step_stop ||
        !sender_stopped(cell, src, threshold) ||
        (cell.forgiven + remainder) * kDecisionScale >
            cell.owed * threshold) {
        return {uint64_t{0}, cell};
    }
    cell.forgiven += remainder;
    return {remainder, cell};
}

// Dense (receiving rank, step) budget table. Ranks stay under a few hundred
// and steps under a few hundred, so the whole table is well under a megabyte
// and every access is one multiply-add. A default-constructed ledger is empty
// and total: every query answers "do not forgive" and every update is a
// no-op, which is exactly the admission domain's behaviour.
class ForgivenessLedger {
  public:
    ForgivenessLedger() = default;

    static ForgivenessLedger make(uint32_t ranks, uint32_t steps) {
        ForgivenessLedger ledger;
        if (ranks == 0 || steps == 0) {
            return ledger;
        }
        ledger.ranks_ = ranks;
        ledger.steps_ = steps;
        ledger.cells_.assign(static_cast<size_t>(ranks) * steps, StepLedger{});
        return ledger;
    }

    bool empty() const {
        return cells_.empty();
    }

    // A launch into a cell the plan did not name is a plan that is wrong
    // about the collective schedule, and every budget decision downstream is
    // measured against that plan. It ends the run here rather than moving a
    // budget nothing certified.
    void register_eligible(uint32_t dst, uint32_t step, uint64_t bytes) {
        StepLedger* cell = find(dst, step);
        if (cell == nullptr) {
            return;
        }
        if (cell->owed == 0) {
            throw std::runtime_error(
                "the collective plan did not name rank " +
                std::to_string(dst) + " step " + std::to_string(step) +
                ", which a DP All-Reduce launched " + std::to_string(bytes) +
                " B into");
        }
        cell->eligible += bytes;
    }

    void register_shed(uint32_t dst, uint32_t step, uint64_t bytes) {
        if (StepLedger* cell = find(dst, step)) {
            cell->shed += bytes;
        }
    }

    // What one arrival added to the bytes this rank holds for the step, which
    // is half of what the budget law measures itself against, and which of its
    // senders sent them, which is what the step stop decides on.
    void register_delivered(uint32_t dst,
                            uint32_t step,
                            uint32_t src,
                            uint64_t bytes) {
        if (StepLedger* cell = find(dst, step)) {
            cell->delivered += bytes;
            cell->by_sender[src].delivered += bytes;
        }
    }

    // The step's plan for one (sender, receiver, step), read from ASTRA-sim's
    // own collective code before the run starts. Every forgiving run carries
    // it, because the report is measured against the step's total.
    void register_owed(uint32_t dst,
                       uint32_t step,
                       uint32_t src,
                       uint64_t bytes) {
        if (StepLedger* cell = find(dst, step)) {
            cell->owed += bytes;
            cell->by_sender[src].owed += bytes;
        }
    }

    // How many DP All-Reduce collectives the plan says this rank runs in this
    // step. The certification waits for the last of them, because a rank that
    // runs two has accounted for the step's bytes only when both are done.
    void register_collectives(uint32_t dst, uint32_t step, uint32_t count) {
        if (StepLedger* cell = find(dst, step)) {
            cell->collectives += count;
        }
    }

    // One flow's holes, as the receiver sees them at this moment. The cell
    // keeps the sum, so the caller hands over the new count and the flow's
    // previous one leaves with it.
    void replace_holes(uint32_t dst,
                       uint32_t step,
                       uint64_t before,
                       uint64_t after) {
        if (StepLedger* cell = find(dst, step)) {
            cell->holes -= before;
            cell->holes += after;
        }
    }

    // Measure affordability against the plan rather than against what the rank
    // has accounted for. One call from the parser, because the base is a
    // property of the profile and not of a cell.
    void use_owed_base() {
        for (StepLedger& cell : cells_) {
            cell.owed_base = true;
        }
    }

    // One of the step's DP All-Reduce collectives has completed at this rank.
    // The last one asserts the contract on the cell; the definition sits
    // beside the verdicts, below, because the threshold it holds the cell to
    // comes from the same mask they charged it against.
    void note_collective_completed(uint32_t dst, uint32_t step);

    // The one lookup. Null means the rank or step is outside the table, which
    // answers "repair, with nothing to report about the allowance".
    StepLedger* cell(uint32_t dst, uint32_t step) {
        return find(dst, step);
    }

    const StepLedger* cell(uint32_t dst, uint32_t step) const {
        return find(dst, step);
    }

  private:
    // Steps are one-based in every profile, mask, and telemetry row.
    StepLedger* find(uint32_t dst, uint32_t step) {
        return const_cast<StepLedger*>(
            const_cast<const ForgivenessLedger*>(this)->find(dst, step));
    }

    const StepLedger* find(uint32_t dst, uint32_t step) const {
        if (dst >= ranks_ || step == 0 || step > steps_) {
            return nullptr;
        }
        return &cells_[static_cast<size_t>(dst) * steps_ + (step - 1)];
    }

    uint32_t ranks_ = 0;
    uint32_t steps_ = 0;
    std::vector<StepLedger> cells_;
};

struct MicroburstFlow {
    uint32_t src = 0;
    uint32_t dst = 0;
    uint64_t size_bytes = 0;
    uint64_t offset_ns = 0;
    uint16_t priority_group = 3;
};

struct ExperimentConfig {
    bool enabled = false;
    uint64_t seed = 0;
    // Provenance only. The selection hash excludes it so two profiles that
    // differ only by name draw the same selection stream and stay matched.
    std::string run_id = "default";
    uint16_t default_priority_group = 3;
    uint16_t provenance_priority_group = 1;
    uint64_t provenance_control_bytes = 64;
    std::map<uint32_t, uint16_t> vnet_to_priority_group;
    std::map<uint32_t, uint64_t> shedding_threshold_by_step;
    bool selection_policy_configured = false;
    SheddingDomain domain = SheddingDomain::Admission;
    uint64_t p_low_threshold = 0;
    uint64_t p_high_threshold = 0;
    // The receiver's v2 policies. Pacing reserves allowance for later in the
    // step. `cap_base_owed` measures the cap against the step's plan rather
    // than against what the rank has accounted for, and `step_stop` lets the
    // receiver end a sender's step once `1 - p` of what that sender owes has
    // arrived. The stop reads the plan, which every forgiving domain carries,
    // so the two are independent.
    Pacing pacing = NoPacing{};
    // Which cap decides affordability. The soft one, the receiver-local
    // vesting cap, is the law of record; `cap_base = owed` is the ablation
    // that affords against the hard cap instead. Revocation reads the hard
    // side either way.
    bool cap_base_owed = false;
    bool step_stop = false;
    // Whether a spent report ends an exemption. False is the reference arm in
    // which the budget alone bounds the loss.
    bool reengage = true;
    uint32_t rank_count = 0;
    uint32_t step_count = 0;
    bool clr_mask_configured = false;
    std::map<uint32_t, bool> clr_mask_by_step;
    bool microburst_enabled = false;
    uint32_t microburst_trigger_step = 2;
    std::vector<MicroburstFlow> microburst_flows;
    bool microburst_triggered = false;
    // Where the telemetry will be written. Empty when no experiment is
    // configured. The files are not opened here: see open_experiment_telemetry.
    std::string telemetry_output_dir;
};

struct SheddingDecision {
    bool eligible = false;
    bool shed = false;
    bool is_clr = false;
    uint64_t decision_hash = 0;
};

struct FlowRecord {
    FlowKind kind = FlowKind::ForegroundPayload;
    bool shed = false;
    bool admission_eligible = false;
    AstraSim::TransportRole origin_transport_role =
        AstraSim::TransportRole::Unknown;
    AstraSim::OperationContext operation;
    uint64_t decision_hash = 0;
    int src = 0;
    int dst = 0;
    int tag = 0;
    uint16_t source_port = 0;
    uint16_t priority_group = 0;
    uint64_t logical_bytes = 0;
    uint64_t physical_bytes = 0;
    uint64_t data_attempted_bytes = 0;
    uint64_t retransmitted_bytes = 0;
    uint64_t trimmed_payload_bytes = 0;
    uint32_t recovery_events = 0;
    uint32_t trim_notifications = 0;
    uint32_t trim_lasthop_notifications = 0;
    uint32_t trim_recovery_events = 0;
    uint32_t stale_trim_notifications = 0;
    uint32_t timeouts = 0;
    uint32_t cnp_received = 0;
    // Payload bytes the receiving NIC accepted, counted as they arrived. A
    // completed flow must have accepted every byte it offered less those it
    // was forgiven, which is what the completion check asserts.
    uint64_t accepted_bytes = 0;
    // Bytes the receiver accepted without ever seeing them, and how many
    // trimmed ranges that took. Recovery domain only.
    uint64_t forgiven_bytes = 0;
    uint32_t forgiven_ranges = 0;
    // The subset of forgiven_bytes the receiver never waited for at all,
    // because the flow went quiet with data still unsent. Trimmed-forgiven
    // bytes are the difference, derived rather than counted.
    uint64_t forgiven_remainder_bytes = 0;
    // Trims the cap could have afforded and the pacing coin declined anyway.
    // Nothing else can see them: a cap refusal is allowance_gone_reports,
    // and a coin refusal leaves the cap untouched.
    uint32_t pacing_refusals = 0;
    // Bytes the soft cap declined for want of vested allowance. With the
    // coin's refusals and the forgiven bytes, this decomposes every trim the
    // receiver answered. Nothing reads it: it is the decomposition itself.
    uint64_t soft_refusals = 0;
    // Bytes that arrived for a range this flow was already forgiven. The
    // receiver drops them, so the budget charge stands; the analyzer reads
    // this beside the forgiven count to report what was actually lost.
    uint64_t late_forgiven_bytes = 0;
    // What the receiver is missing on this flow as of its last report, which
    // is the flow's share of the cell's holes. The cell keeps the sum, so the
    // previous value has to be here to leave it.
    uint64_t holes = 0;
    // Verdicts this flow has asked for. It is the coin's attempt number, so a
    // re-trimmed range draws again rather than repeating its refusal.
    uint32_t verdicts_asked = 0;
    // Congestion-exempt domain only. `cc_exempt` records that the receiver
    // granted this flow an exemption at some point, and is never withdrawn:
    // the exemption may have lapsed, but it was granted, and the telemetry is
    // the record of that. `cc_exempt_granted_ns` is when the first
    // acknowledgement carrying the grant arrived and
    // `allowance_gone_reports` counts the reports that said the budget was
    // gone. The sender follows the latest report either way, so what a flow
    // spends under its controller is a duration and not an instant:
    // `cc_transitions` counts the reports that changed the bit and
    // `cc_obeying_ns` the simulated time it spent delivering signals while
    // holding a grant.
    bool cc_exempt = false;
    uint64_t cc_exempt_granted_ns = 0;
    uint32_t cc_signal_withheld = 0;
    uint32_t allowance_gone_reports = 0;
    uint32_t cc_transitions = 0;
    uint64_t cc_obeying_ns = 0;
    // Zero means never; no packet can be trimmed or repaired at time zero.
    uint64_t first_trim_ns = 0;
    uint64_t first_repair_ns = 0;
    // Repair sends of a packet already repaired, where every data packet has
    // a send record. Zero under ECMP.
    uint32_t duplicate_repairs = 0;
    // Data packets that reached the receiver, duplicates included, and those
    // among them a source leaf carried on a spine other than the requested
    // one. The second is zero outside spray_uniform.
    uint64_t data_arrivals = 0;
    uint64_t folded_arrivals = 0;
    uint64_t start_time_ns = 0;
    uint64_t end_time_ns = 0;
    FlowTerminalOutcome terminal_outcome = FlowTerminalOutcome::Pending;
    std::string failure_reason;
};

inline ExperimentConfig experiment_config;
inline ForgivenessLedger forgiveness_ledger;

// What ASTRA-sim's own collective code says one receiving rank's step will
// carry: the bytes each sender contributes, and how many DP All-Reduce
// collectives the rank runs in the step. The planner walks every rank's trace
// before the first launch and hands the whole table over in one call, because
// a cell whose senders are still being added would report its budget gone on
// a total that is not yet the total.
struct PlannedStep {
    std::map<uint32_t, uint64_t> by_sender;
    uint32_t collectives = 0;
};
using CollectivePlan = std::map<std::pair<uint32_t, uint32_t>, PlannedStep>;

inline void install_collective_plan(const CollectivePlan& plan) {
    for (const auto& entry : plan) {
        const uint32_t dst = entry.first.first;
        const uint32_t step = entry.first.second;
        forgiveness_ledger.register_collectives(dst, step,
                                                entry.second.collectives);
        for (const auto& sender : entry.second.by_sender) {
            forgiveness_ledger.register_owed(dst, step, sender.first,
                                             sender.second);
        }
    }
}

// Offered minus forgiven. A completed queue pair in the recovery domain
// delivered fewer bytes than it offered, by exactly the forgiven count;
// physical_bytes stays the offered figure so it keeps joining fct.txt and
// keeps denominating W. The telemetry column and what the budget law counts
// as delivered are the same quantity, so they are the same function.
inline uint64_t delivered_bytes(const FlowRecord& flow) {
    return flow.physical_bytes >= flow.forgiven_bytes
        ? flow.physical_bytes - flow.forgiven_bytes
        : 0;
}

class ExperimentTelemetry {
  public:
    void initialize(const std::filesystem::path& output_dir) {
        std::filesystem::create_directories(output_dir);
        flow_events.open(output_dir / "flow_events.csv", std::ios::trunc);
        rank_completion.open(output_dir / "rank_completion.csv", std::ios::trunc);
        collective_events.open(output_dir / "collective_events.csv", std::ios::trunc);
        if (!flow_events || !rank_completion || !collective_events) {
            throw std::runtime_error("Unable to create experiment telemetry files");
        }
        flow_events
            << "flow_kind,decision,admission_eligible,parallelism_domain,"
               "origin_transport_role,transport_role,"
               "collective_type,training_step,workload_node_id,"
               "message_sequence,src,dst,tag,source_port,priority_group,"
               "logical_bytes,physical_bytes,data_attempted_bytes,"
               "retransmitted_bytes,trimmed_payload_bytes,recovery_events,"
               "trim_notifications,trim_lasthop_notifications,"
               "trim_recovery_events,stale_trim_notifications,terminal_outcome,"
               "failure_reason,decision_hash,start_time_ns,end_time_ns,"
               "timeouts,cnp_received,first_trim_ns,first_repair_ns,"
               "forgiven_bytes,forgiven_ranges,forgiven_remainder_bytes,"
               "pacing_refusals,soft_refusals,late_forgiven_bytes,"
               "delivered_bytes,cc_exempt,"
               "cc_exempt_granted_ns,cc_signal_withheld,"
               "allowance_gone_reports,cc_transitions,cc_obeying_ns,"
               "duplicate_repairs,data_arrivals,folded_arrivals\n";
        rank_completion << "rank,completion_time_ns\n";
        collective_events
            << "rank,parallelism_domain,collective_type,training_step,"
               "workload_node_id,logical_bytes,start_time_ns,end_time_ns\n";
    }

    bool enabled() const {
        return flow_events.is_open();
    }

    void record_flow(const FlowRecord& flow) {
        if (!enabled()) {
            return;
        }
        flow_events << flow_kind_name(flow.kind) << ','
                    << (flow.shed ? "shed" : "admitted") << ','
                    << (flow.admission_eligible ? "true" : "false") << ','
                    << parallelism_domain_name(flow.operation.parallelism_domain)
                    << ',' << transport_role_name(flow.origin_transport_role)
                    << ',' << transport_role_name(flow.operation.transport_role)
                    << ',' << collective_type_name(flow.operation.collective_type)
                    << ',' << flow.operation.training_step << ','
                    << flow.operation.workload_node_id << ','
                    << flow.operation.message_sequence << ',' << flow.src << ','
                    << flow.dst << ',' << flow.tag << ',' << flow.source_port
                    << ',' << flow.priority_group << ',' << flow.logical_bytes
                    << ',' << flow.physical_bytes << ','
                    << flow.data_attempted_bytes << ','
                    << flow.retransmitted_bytes << ','
                    << flow.trimmed_payload_bytes << ',' << flow.recovery_events
                    << ',' << flow.trim_notifications << ','
                    << flow.trim_lasthop_notifications << ','
                    << flow.trim_recovery_events << ','
                    << flow.stale_trim_notifications
                    << ',' << terminal_outcome_name(flow.terminal_outcome)
                    << ',' << flow.failure_reason << ',' << flow.decision_hash
                    << ',' << flow.start_time_ns << ',' << flow.end_time_ns
                    << ',' << flow.timeouts << ',' << flow.cnp_received << ','
                    << flow.first_trim_ns << ',' << flow.first_repair_ns << ','
                    << flow.forgiven_bytes << ',' << flow.forgiven_ranges
                    << ',' << flow.forgiven_remainder_bytes << ','
                    << flow.pacing_refusals << ',' << flow.soft_refusals
                    << ',' << flow.late_forgiven_bytes << ','
                    << delivered_bytes(flow) << ','
                    << (flow.cc_exempt ? "true" : "false") << ','
                    << flow.cc_exempt_granted_ns << ','
                    << flow.cc_signal_withheld << ','
                    << flow.allowance_gone_reports << ','
                    << flow.cc_transitions << ',' << flow.cc_obeying_ns
                    << ',' << flow.duplicate_repairs << ','
                    << flow.data_arrivals << ',' << flow.folded_arrivals
                    << '\n';
    }

    void record_collective_completion(
        int rank,
        const AstraSim::OperationContext& operation,
        uint64_t logical_bytes,
        uint64_t start_time_ns,
        uint64_t end_time_ns) {
        if (!collective_events.is_open()) {
            return;
        }
        if (end_time_ns < start_time_ns) {
            throw std::runtime_error("collective completion precedes its start time");
        }
        // One of the step's DP All-Reduce collectives is done at this rank.
        // The step's bytes are accounted for when the last one is, and that
        // is where the contract is asserted.
        if (rank >= 0 && AstraSim::is_dp_all_reduce_payload(operation)) {
            forgiveness_ledger.note_collective_completed(
                static_cast<uint32_t>(rank), operation.training_step);
        }
        collective_events << rank << ','
                          << parallelism_domain_name(operation.parallelism_domain)
                          << ',' << collective_type_name(operation.collective_type)
                          << ',' << operation.training_step << ','
                          << operation.workload_node_id << ',' << logical_bytes
                          << ',' << start_time_ns << ',' << end_time_ns << '\n';
    }

    void record_rank_completion(int rank, uint64_t completion_time_ns) {
        if (rank_completion.is_open()) {
            rank_completion << rank << ',' << completion_time_ns << '\n';
        }
    }

    void flush() {
        if (flow_events.is_open()) {
            flow_events.flush();
        }
        if (rank_completion.is_open()) {
            rank_completion.flush();
        }
        if (collective_events.is_open()) {
            collective_events.flush();
        }
    }

  private:
    static const char* flow_kind_name(FlowKind kind) {
        switch (kind) {
        case FlowKind::ForegroundPayload:
            return "foreground_payload";
        case FlowKind::ProvenanceControl:
            return "provenance_control";
        case FlowKind::BackgroundMicroburst:
            return "background_microburst";
        }
        return "unknown";
    }

    static const char* terminal_outcome_name(FlowTerminalOutcome outcome) {
        switch (outcome) {
        case FlowTerminalOutcome::Pending:
            return "pending";
        case FlowTerminalOutcome::Completed:
            return "completed";
        case FlowTerminalOutcome::Failed:
            return "failed";
        }
        return "unknown";
    }

    static const char* parallelism_domain_name(
        AstraSim::ParallelismDomain domain) {
        switch (domain) {
        case AstraSim::ParallelismDomain::Tensor:
            return "tp";
        case AstraSim::ParallelismDomain::Pipeline:
            return "pp";
        case AstraSim::ParallelismDomain::Data:
            return "dp";
        case AstraSim::ParallelismDomain::Unknown:
            return "unknown";
        }
        return "unknown";
    }

    static const char* transport_role_name(AstraSim::TransportRole role) {
        switch (role) {
        case AstraSim::TransportRole::CollectivePayload:
            return "collective_payload";
        case AstraSim::TransportRole::PointToPointPayload:
            return "point_to_point_payload";
        case AstraSim::TransportRole::RendezvousControl:
            return "rendezvous_control";
        case AstraSim::TransportRole::ProvenanceControl:
            return "provenance_control";
        case AstraSim::TransportRole::BackgroundTraffic:
            return "background_traffic";
        case AstraSim::TransportRole::Unknown:
            return "unknown";
        }
        return "unknown";
    }

    static const char* collective_type_name(AstraSim::ComType type) {
        switch (type) {
        case AstraSim::ComType::All_Reduce:
            return "all_reduce";
        case AstraSim::ComType::All_to_All:
            return "all_to_all";
        case AstraSim::ComType::All_Gather:
            return "all_gather";
        case AstraSim::ComType::Reduce_Scatter:
            return "reduce_scatter";
        case AstraSim::ComType::All_Reduce_All_to_All:
            return "all_reduce_all_to_all";
        case AstraSim::ComType::None:
            return "none";
        }
        return "unknown";
    }

    std::ofstream flow_events;
    std::ofstream rank_completion;
    std::ofstream collective_events;
};

inline ExperimentTelemetry experiment_telemetry;

inline void validate_priority_group(uint16_t priority_group,
                                    const std::string& field_name) {
    if (priority_group >= kPriorityGroupCount) {
        throw std::runtime_error(field_name + " must be in [0, 7]");
    }
}

inline uint64_t stable_operation_hash(const AstraSim::sim_request& request,
                                      int src,
                                      int dst,
                                      int tag) {
    uint64_t hash = experiment_config.seed;
    hash_combine(hash, request.operation.training_step);
    hash_combine(hash, request.operation.workload_node_id);
    hash_combine(hash, request.operation.message_sequence);
    hash_combine(hash, static_cast<uint64_t>(src));
    hash_combine(hash, static_cast<uint64_t>(dst));
    hash_combine(hash, static_cast<uint64_t>(tag));
    return hash;
}

inline SheddingDecision evaluate_shedding(const AstraSim::sim_request& request,
                                          int src,
                                          int dst,
                                          int tag) {
    SheddingDecision decision;
    if (!experiment_config.enabled ||
        !AstraSim::is_dp_all_reduce_payload(request.operation)) {
        return decision;
    }

    decision.eligible = true;
    // Before the domain branch: the hash identifies the operation, not the
    // decision taken on it, and every arm's flow_events.csv is joined on it.
    // Computing it only on the admission path left recovery rows carrying
    // zero, so a join across arms differed for a reason that is not the
    // domain.
    decision.decision_hash = stable_operation_hash(request, src, dst, tag);
    // The recovery domain spends the same budget after a trim, so shedding at
    // admission as well would double-spend it. Eligibility is still recorded:
    // it is what makes a flow forgivable later, and it keeps the two domains'
    // eligible populations identical for a matched comparison.
    if (forgives(experiment_config.domain)) {
        return decision;
    }
    if (experiment_config.clr_mask_configured) {
        const auto clr = experiment_config.clr_mask_by_step.find(
            request.operation.training_step);
        if (clr == experiment_config.clr_mask_by_step.end()) {
            throw std::runtime_error(
                "CLR mask does not define the request training step");
        }
        decision.is_clr = clr->second;
        const uint64_t threshold = decision.is_clr
            ? experiment_config.p_low_threshold
            : experiment_config.p_high_threshold;
        decision.shed = decision.decision_hash % kDecisionScale < threshold;
        return decision;
    }
    const auto threshold = experiment_config.shedding_threshold_by_step.find(
        request.operation.training_step);
    if (threshold == experiment_config.shedding_threshold_by_step.end()) {
        return decision;
    }
    decision.shed =
        decision.decision_hash % kDecisionScale < threshold->second;
    return decision;
}

// The step's threshold, and zero when the mask does not define the step. The
// sentinel and a configured zero coincide in meaning, because a step outside
// the mask and a step whose threshold is zero both forgive nothing and shed
// nothing, so nothing distinguishes them and the parser need not refuse a
// zero. Both verdict shells and the closing check resolve the threshold the
// same way, so they resolve it through the same function.
inline uint64_t step_threshold(uint32_t step) {
    const auto clr = experiment_config.clr_mask_by_step.find(step);
    if (clr == experiment_config.clr_mask_by_step.end()) {
        return 0;
    }
    return clr->second ? experiment_config.p_low_threshold
                       : experiment_config.p_high_threshold;
}

inline uint64_t step_threshold(const FlowRecord& flow) {
    return step_threshold(flow.operation.training_step);
}

// The contract, asserted where a cell stops changing: the receiving rank kept
// at least 1 - p(step) of what the step owed it. `affords` makes the throw
// unreachable, since eligible only grows and every charge was measured
// against it; the throw is what stops a later change to `affords`, to the
// remainder path, or to the ledger from shipping a run that broke the
// contract.
//
// It resolves the threshold rather than receiving one, because a caller that
// supplied it could measure a cell against a budget the cell was never
// charged under, and because the two-argument call keeps it total in what a
// caller knows: a rank and a step name a cell or they name nothing, and
// naming nothing certifies nothing. A step the mask does not define resolves
// to zero, which is the budget its verdicts already used, so a cell that
// spent nothing under it passes and one that spent anything throws.
//
// The step's bytes are accounted for when its last DP All-Reduce completes,
// not its first: a rank that runs two collectives in a step is still
// receiving the second while the first is done.
inline void ForgivenessLedger::note_collective_completed(uint32_t dst,
                                                         uint32_t step) {
    StepLedger* cell = find(dst, step);
    if (cell == nullptr) {
        return;
    }
    cell->completed++;
    if (cell->completed > cell->collectives) {
        throw std::runtime_error(
            "forgiveness ledger saw more DP All-Reduce collectives than its "
            "plan named at rank " +
            std::to_string(dst) + " step " + std::to_string(step) + ": " +
            std::to_string(cell->completed) + " completed, " +
            std::to_string(cell->collectives) + " planned");
    }
    if (cell->completed < cell->collectives) {
        return;
    }
    const uint64_t threshold = step_threshold(step);
    // The spent report was measured against the step's plan, so the plan has
    // to have been the launches. A cell the plan never named has an owed of
    // zero and fails this the moment anything was launched into it.
    if (cell->eligible != cell->owed) {
        throw std::runtime_error(
            "forgiveness ledger was launched bytes its plan did not predict "
            "at rank " +
            std::to_string(dst) + " step " + std::to_string(step) +
            ": eligible " + std::to_string(cell->eligible) + " B, owed " +
            std::to_string(cell->owed) + " B");
    }
    const uint64_t spent = cell->shed + cell->forgiven;
    if (spent * kDecisionScale > cell->eligible * threshold) {
        throw std::runtime_error(
            "forgiveness ledger broke the budget law at rank " +
            std::to_string(dst) + " step " + std::to_string(step) +
            ": eligible " + std::to_string(cell->eligible) + " B, shed " +
            std::to_string(cell->shed) + " B, forgiven " +
            std::to_string(cell->forgiven) + " B, threshold " +
            std::to_string(threshold) + " of " +
            std::to_string(kDecisionScale));
    }
}

// Whether the experiment layer may answer a receiver's question about this
// flow at all. Shared by both verdicts, because both ask the same thing of
// the flow; the exemption asks a narrower one and keeps its own guard.
inline bool forgivable(const FlowRecord& flow) {
    return experiment_config.enabled && forgives(experiment_config.domain) &&
           flow.kind == FlowKind::ForegroundPayload && flow.admission_eligible;
}

// The receiver's account of one arrival. A NIC credits a byte when it accepts
// it, so the budget grows per packet rather than per completed message, and a
// step's cap is available while that step is still arriving. The flow keeps
// the same count, so completion can check it against the bytes the sender
// offered. Same eligibility guard as `register_eligible`, so the two sides of
// the ratio cover the same population.
//
// True when this arrival is the one that carried the sending rank across
// `1 - p` of what it owes: the caller stops that sender's open flows then and
// there, because a flow waiting on a repair receives nothing and would
// otherwise be stopped only by its own timeout. The question is asked only
// where the step stop is on, since nothing else reads the answer.
inline bool note_delivered(FlowRecord& flow,
                           uint64_t bytes,
                           uint64_t late_forgiven) {
    flow.accepted_bytes += bytes;
    flow.late_forgiven_bytes += late_forgiven;
    if (!flow.admission_eligible || flow.dst < 0) {
        return false;
    }
    const uint32_t dst = static_cast<uint32_t>(flow.dst);
    const uint32_t src = static_cast<uint32_t>(flow.src);
    const uint32_t step = flow.operation.training_step;
    if (!experiment_config.step_stop) {
        forgiveness_ledger.register_delivered(dst, step, src, bytes);
        return false;
    }
    const uint64_t threshold = step_threshold(step);
    const StepLedger* before = forgiveness_ledger.cell(dst, step);
    const bool stopped_before =
        before != nullptr && sender_stopped(*before, src, threshold);
    forgiveness_ledger.register_delivered(dst, step, src, bytes);
    const StepLedger* after = forgiveness_ledger.cell(dst, step);
    return !stopped_before && after != nullptr &&
           sender_stopped(*after, src, threshold);
}

// The receiver's report, asked wherever an acknowledgement or a repair
// request leaves the receiver: this flow is missing `holes` bytes right now,
// is the step's allowance gone. The cell keeps the sum over its flows, so a
// repair that lands lowers it without anyone subtracting, and the report goes
// green again on its own. A flow whose cell or step the table does not name
// reports nothing, which leaves its sender under its controller.
inline bool note_holes(FlowRecord& flow, uint64_t holes) {
    if (!forgivable(flow) || flow.dst < 0) {
        return false;
    }
    const uint32_t dst = static_cast<uint32_t>(flow.dst);
    const uint32_t step = flow.operation.training_step;
    forgiveness_ledger.replace_holes(dst, step, flow.holes, holes);
    flow.holes = holes;
    const StepLedger* cell = forgiveness_ledger.cell(dst, step);
    if (cell == nullptr) {
        return false;
    }
    return budget_gone(*cell, step_threshold(step));
}

// The trim verdict the transport asks for. The transport is semantics-blind:
// it supplies a flow, a range start and a byte count, and learns nothing
// about steps, phases, or budgets. A thin shell over trim_verdict: it
// eliminates the cell, draws the coin, stores the answer back, and counts.
//
// The report the sender acts on is not here. It belongs to the cell and is
// asked for wherever the answer leaves the receiver, because a refusal the
// vesting cap made says nothing about the step and a range absorbed as
// forgiven has stopped being a hole by then.
inline bool evaluate_forgiveness(FlowRecord& flow,
                                 uint64_t start,
                                 uint64_t bytes) {
    if (!forgivable(flow)) {
        return false;
    }
    const uint64_t threshold = step_threshold(flow);
    if (threshold == 0) {
        return false;
    }
    StepLedger* cell = forgiveness_ledger.cell(
        static_cast<uint32_t>(flow.dst), flow.operation.training_step);
    if (cell == nullptr) {
        return false;
    }
    flow.verdicts_asked++;
    const uint64_t coin =
        range_coin(flow.decision_hash, start, flow.verdicts_asked);
    const bool paced_out = paces_out(experiment_config.pacing, coin);
    const bool affordable = affords_soft(*cell, threshold, bytes);
    // Only a range the cap could still have afforded: that is the forgiveness
    // the coin cost, and it makes the two refusals decompose without overlap,
    // the cap's into soft_refusals and the coin's into this.
    if (paced_out && affordable) {
        flow.pacing_refusals++;
    }
    if (!paced_out && !affordable) {
        flow.soft_refusals += bytes;
    }
    const auto answer =
        trim_verdict(*cell, threshold, experiment_config.pacing, coin, bytes);
    *cell = answer.second;
    if (answer.first) {
        flow.forgiven_bytes += bytes;
        flow.forgiven_ranges++;
    }
    return answer.first;
}

// The step stop's answer. The receive queue pair knows how far its cumulative
// sequence has reached and how much arrived above it, and not how large the
// flow is; this knows the size. The hole is what is left, and the answer is
// the end offset to absorb, or zero to refuse. The transport asks at every
// accepted arrival, so the refusals are what hold the stop to the end of the
// sender's step.
//
// Under selective repeat a stalled flow keeps accepting packets past the gap,
// so `accepted_above` is usually nonzero and charging the whole span above the
// cumulative sequence would spend the budget on bytes the receiver holds.
//
// forgiven_ranges stays untouched: it counts the trims a forgiveness spared a
// repair, and a remainder was never trimmed. That keeps one rate cut per trim
// as the identity a fixture can assert.
inline uint64_t evaluate_remainder(FlowRecord& flow,
                                   uint64_t next_expected,
                                   uint64_t accepted_above) {
    if (!forgivable(flow) ||
        flow.physical_bytes < next_expected + accepted_above) {
        return 0;
    }
    const uint64_t remainder =
        flow.physical_bytes - next_expected - accepted_above;
    const uint64_t threshold = step_threshold(flow);
    if (remainder == 0 || threshold == 0) {
        return 0;
    }
    StepLedger* cell = forgiveness_ledger.cell(
        static_cast<uint32_t>(flow.dst), flow.operation.training_step);
    if (cell == nullptr) {
        return 0;
    }
    const auto answer =
        remainder_verdict(*cell, threshold, remainder,
                          static_cast<uint32_t>(flow.src),
                          experiment_config.step_stop);
    *cell = answer.second;
    if (answer.first == 0) {
        return 0;
    }
    flow.forgiven_bytes += answer.first;
    flow.forgiven_remainder_bytes += answer.first;
    return flow.physical_bytes;
}

// Whether the receiver may forgive this flow on this step, which is what it
// marks its acknowledgements with. Pure and total: an unknown step or a step
// whose own budget is zero answers false, and so does every domain but the
// exempt one, which is how an arm without the exemption keeps every sender
// under its controller. A critical step grants like any other; its small p
// means the pool is gone after a few hundred kilobytes and the report brings
// the controller back almost at once. It reads no cell and spends no budget:
// whether the step still has allowance rides on the same acknowledgement, and
// that is the sender's other half of the grant.
inline bool exemption_eligible(const FlowRecord& flow) {
    return experiment_config.domain == SheddingDomain::RecoveryExempt &&
           forgivable(flow) && step_threshold(flow) > 0;
}

inline uint16_t priority_group_for_vnet(uint32_t vnet) {
    const auto mapping = experiment_config.vnet_to_priority_group.find(vnet);
    if (mapping != experiment_config.vnet_to_priority_group.end()) {
        return mapping->second;
    }
    return experiment_config.default_priority_group;
}

inline void reject_unknown_keys(const nlohmann::json& value,
                                std::initializer_list<const char*> allowed,
                                const std::string& object_name) {
    for (auto it = value.begin(); it != value.end(); ++it) {
        const auto allowed_key = std::find_if(
            allowed.begin(), allowed.end(), [&](const char* key) {
                return it.key() == key;
            });
        if (allowed_key == allowed.end()) {
            throw std::runtime_error("Unknown key '" + it.key() + "' in " +
                                     object_name);
        }
    }
}

inline uint64_t parse_uint64_key(const std::string& value,
                                 const std::string& field_name) {
    size_t consumed = 0;
    uint64_t result = 0;
    try {
        result = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(field_name + " key must be an unsigned integer");
    }
    if (consumed != value.size()) {
        throw std::runtime_error(field_name + " key must be an unsigned integer");
    }
    return result;
}

inline uint16_t parse_priority_group(const nlohmann::json& value,
                                     const std::string& field_name) {
    if (!value.is_number_unsigned() ||
        value.get<uint64_t>() >= kPriorityGroupCount) {
        throw std::runtime_error(field_name + " must be an integer in [0, 7]");
    }
    return static_cast<uint16_t>(value.get<uint64_t>());
}

inline uint64_t parse_probability_threshold(const nlohmann::json& value,
                                            const std::string& field_name) {
    if (!value.is_number()) {
        throw std::runtime_error(field_name + " must be a number in [0, 1]");
    }
    const double probability = value.get<double>();
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
        throw std::runtime_error(field_name + " must be a number in [0, 1]");
    }
    return static_cast<uint64_t>(
        std::llround(probability * static_cast<double>(kDecisionScale)));
}

// The generator writes the scaled integer beside the float it came from.
// Both are optional for a hand-written file, but a present one must agree
// with llround of the float, which is what the simulator itself rounds to.
inline void require_scaled_threshold(const nlohmann::json& policy,
                                     const char* key,
                                     uint64_t rounded) {
    if (!policy.contains(key)) {
        return;
    }
    const auto& value = policy.at(key);
    if (!value.is_number_unsigned() || value.get<uint64_t>() > kDecisionScale) {
        throw std::runtime_error(
            std::string("selection_policy.") + key +
            " must be an unsigned integer in [0, 1000000]");
    }
    if (value.get<uint64_t>() != rounded) {
        throw std::runtime_error(
            std::string("selection_policy.") + key + " is " +
            std::to_string(value.get<uint64_t>()) + " but the probability "
            "beside it rounds to " + std::to_string(rounded));
    }
}

// The one smart constructor for the pacing rule. A probability of one
// declines nothing, which is no pacing, and is refused. A probability of zero
// is a rule: the receiver forgives nothing while the exemption-eligible flag
// and the budget-exhausted flag behave as usual, which is the arm that asks
// whether forgiveness or the exemption recovers the time. A probability
// beside any other kind is a value that would be read by nothing.
inline void parse_pacing(const nlohmann::json& policy) {
    if (!policy.contains("pacing")) {
        return;
    }
    const auto& pacing = policy.at("pacing");
    if (!pacing.is_object()) {
        throw std::runtime_error("selection_policy.pacing must be an object");
    }
    reject_unknown_keys(pacing, {"kind", "p"}, "selection_policy.pacing");
    if (!pacing.contains("kind") || !pacing.at("kind").is_string()) {
        throw std::runtime_error("selection_policy.pacing requires kind");
    }
    const std::string kind = pacing.at("kind").get<std::string>();
    if (kind == "bernoulli") {
        if (!pacing.contains("p")) {
            throw std::runtime_error(
                "selection_policy.pacing kind 'bernoulli' requires p");
        }
        const uint64_t threshold = parse_probability_threshold(
            pacing.at("p"), "selection_policy.pacing.p");
        if (threshold >= kDecisionScale) {
            throw std::runtime_error(
                "selection_policy.pacing.p must be below 1; 1 is no pacing");
        }
        experiment_config.pacing = Bernoulli{threshold};
        return;
    }
    if (pacing.contains("p")) {
        throw std::runtime_error(
            "selection_policy.pacing.p belongs to kind 'bernoulli' alone");
    }
    if (kind == "none") {
        experiment_config.pacing = NoPacing{};
        return;
    }
    throw std::runtime_error(
        "selection_policy.pacing.kind must be none or bernoulli");
}

// Which bytes the cap is measured against, and whether the receiver may end a
// sender's step. The two are independent: the stop reads the step's plan,
// which every forgiving domain carries, so it composes with either base.
inline void parse_cap_base(const nlohmann::json& policy) {
    if (policy.contains("cap_base")) {
        const auto& base = policy.at("cap_base");
        if (base == "owed") {
            experiment_config.cap_base_owed = true;
        } else if (base != "accounted") {
            throw std::runtime_error(
                "selection_policy.cap_base must be accounted or owed");
        }
    }
    if (policy.contains("reengage")) {
        const auto& reengage = policy.at("reengage");
        if (!reengage.is_boolean()) {
            throw std::runtime_error(
                "selection_policy.reengage must be boolean");
        }
        experiment_config.reengage = reengage.get<bool>();
    }
    if (!policy.contains("step_stop")) {
        return;
    }
    const auto& stop = policy.at("step_stop");
    if (!stop.is_boolean()) {
        throw std::runtime_error("selection_policy.step_stop must be boolean");
    }
    experiment_config.step_stop = stop.get<bool>();
}

inline void configure_clr_mask(const std::string& configuration_path) {
    if (configuration_path.empty() || configuration_path == "empty") {
        return;
    }
    if (!experiment_config.enabled) {
        throw std::runtime_error(
            "--clr-mask-configuration requires an enabled experiment");
    }
    if (!experiment_config.selection_policy_configured) {
        throw std::runtime_error(
            "--clr-mask-configuration requires selection_policy in the experiment configuration");
    }

    std::ifstream input(configuration_path);
    if (!input) {
        throw std::runtime_error("Unable to open CLR mask: " + configuration_path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("CLR mask must contain a header");
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != "step_id,is_clr,probability") {
        throw std::runtime_error(
            "CLR mask header must be step_id,is_clr,probability");
    }

    uint64_t row_count = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            throw std::runtime_error("CLR mask must not contain blank rows");
        }
        std::stringstream stream(line);
        std::string step_text;
        std::string clr_text;
        std::string probability_text;
        std::string unexpected;
        if (!std::getline(stream, step_text, ',') ||
            !std::getline(stream, clr_text, ',') ||
            !std::getline(stream, probability_text, ',') ||
            std::getline(stream, unexpected, ',')) {
            throw std::runtime_error(
                "CLR mask rows must contain step_id,is_clr,probability");
        }
        const uint64_t step = parse_uint64_key(step_text, "CLR mask step_id");
        if (step == 0 || step > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("CLR mask step_id must be a nonzero uint32");
        }
        if (clr_text != "0" && clr_text != "1") {
            throw std::runtime_error("CLR mask is_clr must be 0 or 1");
        }
        size_t probability_length = 0;
        double probability = 0.0;
        try {
            probability = std::stod(probability_text, &probability_length);
        } catch (const std::exception&) {
            throw std::runtime_error("CLR mask probability must be in [0, 1]");
        }
        if (probability_length != probability_text.size() ||
            !std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0) {
            throw std::runtime_error("CLR mask probability must be in [0, 1]");
        }
        if (!experiment_config.clr_mask_by_step
                 .emplace(static_cast<uint32_t>(step), clr_text == "1")
                 .second) {
            throw std::runtime_error("CLR mask must not contain duplicate step_id values");
        }
        ++row_count;
    }
    if (row_count == 0) {
        throw std::runtime_error("CLR mask must contain at least one step");
    }
    experiment_config.clr_mask_configured = true;
}

inline void configure_experiment(const std::string& configuration_path,
                                 const std::string& output_dir) {
    experiment_config = ExperimentConfig{};
    forgiveness_ledger = ForgivenessLedger{};
    if (configuration_path.empty() || configuration_path == "empty") {
        return;
    }
    if (output_dir.empty() || output_dir == "empty") {
        throw std::runtime_error(
            "--experiment-output-dir is required with --experiment-configuration");
    }

    std::ifstream input(configuration_path);
    if (!input) {
        throw std::runtime_error("Unable to open experiment configuration: " +
                                 configuration_path);
    }

    nlohmann::json root;
    input >> root;
    if (!root.is_object()) {
        throw std::runtime_error("Experiment configuration must be a JSON object");
    }
    reject_unknown_keys(root,
                        {"schema_version", "enabled", "seed", "run_id",
                         "eligibility", "selection_probability_by_step",
                         "selection_policy", "scale",
                         "default_priority_group", "provenance",
                         "vnet_to_priority_group", "microburst"},
                        "experiment configuration");
    if (root.value("schema_version", 0) != 1) {
        throw std::runtime_error("Experiment schema_version must be 1");
    }
    if (!root.contains("enabled") || !root.at("enabled").is_boolean()) {
        throw std::runtime_error("Experiment configuration requires boolean enabled");
    }

    experiment_config.enabled = root.at("enabled").get<bool>();
    if (root.contains("seed")) {
        if (!root.at("seed").is_number_unsigned()) {
            throw std::runtime_error("seed must be an unsigned integer");
        }
        experiment_config.seed = root.at("seed").get<uint64_t>();
    }
    if (root.contains("run_id")) {
        if (!root.at("run_id").is_string()) {
            throw std::runtime_error("run_id must be a string");
        }
        experiment_config.run_id = root.at("run_id").get<std::string>();
    }

    if (experiment_config.enabled) {
        if (!root.contains("eligibility") ||
            root.at("eligibility") != "dp_all_reduce_only") {
            throw std::runtime_error(
                "enabled experiments require eligibility=dp_all_reduce_only");
        }
    }

    if (root.contains("default_priority_group")) {
        experiment_config.default_priority_group = parse_priority_group(
            root.at("default_priority_group"), "default_priority_group");
    }

    if (root.contains("provenance")) {
        const auto& provenance = root.at("provenance");
        if (!provenance.is_object()) {
            throw std::runtime_error("provenance must be an object");
        }
        reject_unknown_keys(provenance, {"control_bytes", "priority_group"},
                            "provenance");
        if (provenance.contains("control_bytes")) {
            if (!provenance.at("control_bytes").is_number_unsigned() ||
                provenance.at("control_bytes").get<uint64_t>() == 0) {
                throw std::runtime_error(
                    "provenance.control_bytes must be a nonzero unsigned integer");
            }
            experiment_config.provenance_control_bytes =
                provenance.at("control_bytes").get<uint64_t>();
        }
        if (provenance.contains("priority_group")) {
            experiment_config.provenance_priority_group = parse_priority_group(
                provenance.at("priority_group"), "provenance.priority_group");
        }
        if (experiment_config.provenance_priority_group == 0) {
            throw std::runtime_error(
                "provenance.priority_group must reserve priority group 0");
        }
    }

    if (root.contains("vnet_to_priority_group")) {
        const auto& mappings = root.at("vnet_to_priority_group");
        if (!mappings.is_object()) {
            throw std::runtime_error("vnet_to_priority_group must be an object");
        }
        for (auto it = mappings.begin(); it != mappings.end(); ++it) {
            const auto vnet = parse_uint64_key(it.key(), "vnet_to_priority_group");
            if (vnet > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("vnet_to_priority_group key exceeds uint32");
            }
            experiment_config.vnet_to_priority_group.emplace(
                static_cast<uint32_t>(vnet),
                parse_priority_group(it.value(),
                                     "vnet_to_priority_group priority group"));
        }
    }

    if (root.contains("selection_probability_by_step")) {
        const auto& probabilities = root.at("selection_probability_by_step");
        if (!probabilities.is_object()) {
            throw std::runtime_error(
                "selection_probability_by_step must be an object");
        }
        for (auto it = probabilities.begin(); it != probabilities.end(); ++it) {
            const auto step = parse_uint64_key(
                it.key(), "selection_probability_by_step");
            if (step == 0 || step > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "selection_probability_by_step keys must be nonzero uint32 values");
            }
            experiment_config.shedding_threshold_by_step.emplace(
                static_cast<uint32_t>(step),
                parse_probability_threshold(
                    it.value(), "selection_probability_by_step probability"));
        }
    }

    if (root.contains("selection_policy")) {
        const auto& policy = root.at("selection_policy");
        if (!policy.is_object()) {
            throw std::runtime_error("selection_policy must be an object");
        }
        reject_unknown_keys(policy,
                            {"semantics", "p_low", "p_high", "p_low_threshold",
                             "p_high_threshold", "domain", "transport",
                             "pacing", "cap_base", "step_stop",
                             "reengage"},
                            "selection_policy");
        if (policy.contains("domain")) {
            const auto& domain = policy.at("domain");
            if (domain == "admission") {
                experiment_config.domain = SheddingDomain::Admission;
            } else if (domain == "recovery") {
                experiment_config.domain = SheddingDomain::Recovery;
            } else if (domain == "recovery_exempt") {
                experiment_config.domain = SheddingDomain::RecoveryExempt;
            } else {
                throw std::runtime_error(
                    "selection_policy.domain must be admission, recovery, or "
                    "recovery_exempt");
            }
        }
        const char* expected_semantics =
            selection_semantics(experiment_config.domain);
        if (!policy.contains("semantics") ||
            policy.at("semantics") != expected_semantics) {
            throw std::runtime_error(
                std::string("selection_policy.semantics must be ") +
                expected_semantics);
        }
        if (forgives(experiment_config.domain)) {
            // The frontend cannot read network_config.txt, so the generator
            // asserts the transport contract here and entry.h checks the
            // assertion against the transport ns-3 actually built. Recovery
            // needs both: without ftd trimming nothing reaches the receiver to
            // forgive, and without selective repair the receiver never
            // consults the out-of-order range a forgiven hole becomes.
            if (!policy.contains("transport")) {
                throw std::runtime_error(
                    "recovery domain requires selection_policy.transport");
            }
            const auto& transport = policy.at("transport");
            if (!transport.is_object()) {
                throw std::runtime_error(
                    "selection_policy.transport must be an object");
            }
            reject_unknown_keys(transport,
                                {"selective_repair", "packet_trimming_ftd"},
                                "selection_policy.transport");
            for (const char* key : {"selective_repair", "packet_trimming_ftd"}) {
                if (!transport.contains(key) ||
                    !transport.at(key).is_boolean() ||
                    !transport.at(key).get<bool>()) {
                    throw std::runtime_error(
                        std::string("recovery domain requires "
                                    "selection_policy.transport.") + key);
                }
            }
        }
        for (const char* key : {"p_low", "p_high"}) {
            if (!policy.contains(key)) {
                throw std::runtime_error(
                    std::string("selection_policy requires ") + key);
            }
        }
        experiment_config.p_low_threshold = parse_probability_threshold(
            policy.at("p_low"), "selection_policy.p_low");
        experiment_config.p_high_threshold = parse_probability_threshold(
            policy.at("p_high"), "selection_policy.p_high");
        // The budget law is integer arithmetic on these thresholds, and the
        // analyzer checks the same law from the same integers in the file.
        // Refusing a disagreement is what keeps one law in two languages from
        // becoming two laws: a boundary cell must not be spent here and
        // reported violated there.
        require_scaled_threshold(policy, "p_low_threshold",
                                 experiment_config.p_low_threshold);
        require_scaled_threshold(policy, "p_high_threshold",
                                 experiment_config.p_high_threshold);
        // The strict-CLR ceiling on p_low (<= 0.01) is experiment-design
        // policy owned by the generator, which grants exactly one documented
        // exemption: the fixed-high comparison arm runs with p_low set to
        // the permissive rate. The simulator enforces only representability;
        // re-imposing the ceiling here rejected every fixed-high arm.
        // A p_low of zero is legal, and it asks the run to shed nothing and
        // forgive nothing on every step the mask marks critical.
        if (experiment_config.p_low_threshold >
            experiment_config.p_high_threshold) {
            throw std::runtime_error(
                "selection_policy.p_high must be at least p_low");
        }
        // Both v2 policies belong to the receiver, and only a forgiving
        // domain has a receiver that decides anything.
        if (!forgives(experiment_config.domain)) {
            for (const char* key :
                 {"pacing", "cap_base", "step_stop", "reengage"}) {
                if (policy.contains(key)) {
                    throw std::runtime_error(
                        std::string("selection_policy.") + key +
                        " requires a forgiving domain");
                }
            }
        }
        parse_pacing(policy);
        parse_cap_base(policy);
        experiment_config.selection_policy_configured = true;
    }

    if (root.contains("scale")) {
        const auto& scale = root.at("scale");
        if (!scale.is_object()) {
            throw std::runtime_error("scale must be an object");
        }
        reject_unknown_keys(scale, {"ranks", "steps"}, "scale");
        for (const char* key : {"ranks", "steps"}) {
            if (!scale.contains(key) || !scale.at(key).is_number_unsigned() ||
                scale.at(key).get<uint64_t>() == 0 ||
                scale.at(key).get<uint64_t>() >
                    std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("scale.") + key + " must be a nonzero uint32");
            }
        }
        experiment_config.rank_count =
            static_cast<uint32_t>(scale.at("ranks").get<uint64_t>());
        experiment_config.step_count =
            static_cast<uint32_t>(scale.at("steps").get<uint64_t>());
    }
    if (forgives(experiment_config.domain)) {
        if (experiment_config.rank_count == 0 ||
            experiment_config.step_count == 0) {
            throw std::runtime_error("recovery domain requires scale");
        }
        forgiveness_ledger = ForgivenessLedger::make(
            experiment_config.rank_count, experiment_config.step_count);
    }

    if (experiment_config.cap_base_owed) {
        forgiveness_ledger.use_owed_base();
    }

    if (root.contains("microburst")) {
        const auto& microburst = root.at("microburst");
        if (!microburst.is_object()) {
            throw std::runtime_error("microburst must be an object");
        }
        reject_unknown_keys(microburst, {"enabled", "trigger_step", "flows"},
                            "microburst");
        if (!microburst.contains("enabled") ||
            !microburst.at("enabled").is_boolean()) {
            throw std::runtime_error("microburst.enabled must be a boolean");
        }
        experiment_config.microburst_enabled =
            microburst.at("enabled").get<bool>();
        if (microburst.contains("trigger_step")) {
            if (!microburst.at("trigger_step").is_number_unsigned() ||
                microburst.at("trigger_step").get<uint64_t>() == 0 ||
                microburst.at("trigger_step").get<uint64_t>() >
                    std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "microburst.trigger_step must be a nonzero uint32");
            }
            experiment_config.microburst_trigger_step = static_cast<uint32_t>(
                microburst.at("trigger_step").get<uint64_t>());
        }
        if (experiment_config.microburst_enabled) {
            if (!microburst.contains("flows") || !microburst.at("flows").is_array() ||
                microburst.at("flows").empty()) {
                throw std::runtime_error(
                    "enabled microburst requires a nonempty flows array");
            }
            for (const auto& flow : microburst.at("flows")) {
                if (!flow.is_object()) {
                    throw std::runtime_error("microburst flow must be an object");
                }
                reject_unknown_keys(flow,
                                    {"src", "dst", "size_bytes", "offset_ns",
                                     "priority_group"},
                                    "microburst flow");
                for (const char* key : {"src", "dst", "size_bytes", "offset_ns"}) {
                    if (!flow.contains(key) || !flow.at(key).is_number_unsigned()) {
                        throw std::runtime_error(
                            std::string("microburst flow requires unsigned ") + key);
                    }
                }
                const uint64_t src = flow.at("src").get<uint64_t>();
                const uint64_t dst = flow.at("dst").get<uint64_t>();
                if (src > std::numeric_limits<uint32_t>::max() ||
                    dst > std::numeric_limits<uint32_t>::max() || src == dst ||
                    flow.at("size_bytes").get<uint64_t>() == 0) {
                    throw std::runtime_error(
                        "microburst flow requires distinct uint32 endpoints and nonzero size_bytes");
                }
                MicroburstFlow parsed_flow;
                parsed_flow.src = static_cast<uint32_t>(src);
                parsed_flow.dst = static_cast<uint32_t>(dst);
                parsed_flow.size_bytes = flow.at("size_bytes").get<uint64_t>();
                parsed_flow.offset_ns = flow.at("offset_ns").get<uint64_t>();
                parsed_flow.priority_group = flow.contains("priority_group")
                    ? parse_priority_group(flow.at("priority_group"),
                                           "microburst.priority_group")
                    : experiment_config.default_priority_group;
                experiment_config.microburst_flows.push_back(parsed_flow);
            }
        }
    }

    experiment_config.telemetry_output_dir = output_dir;
}

// Refusals the configuration alone cannot make, because the CLR mask arrives
// on its own command-line argument after the experiment is parsed. Run once,
// after both.
inline void validate_experiment_contract() {
    if (!experiment_config.enabled || !forgives(experiment_config.domain)) {
        return;
    }
    // Recovery reads the mask per trim and answers Pull for a step it does
    // not find. Without the mask the whole arm forgives nothing and reads as
    // "the mechanism did nothing", which is indistinguishable from a real
    // negative result. evaluate_shedding throws on the same miss, so the two
    // consumers of one map now refuse on the same terms.
    if (!experiment_config.clr_mask_configured) {
        throw std::runtime_error(
            "recovery domain requires --clr-mask-configuration");
    }
    for (uint32_t step = 1; step <= experiment_config.step_count; ++step) {
        if (experiment_config.clr_mask_by_step.count(step) == 0) {
            throw std::runtime_error(
                "CLR mask does not define training step " +
                std::to_string(step) + ", which the recovery domain requires");
        }
    }
}

// Opened only once ns-3 setup has succeeded. A refused arm must not leave a
// run directory carrying headers and no rows, which reads as started.
inline void open_experiment_telemetry() {
    if (!experiment_config.telemetry_output_dir.empty()) {
        experiment_telemetry.initialize(experiment_config.telemetry_output_dir);
    }
}

inline void finalize_experiment_telemetry() {
    experiment_telemetry.flush();
}

}  // namespace AstraSimNs3

#endif /* __ASTRA_SIM_NS3_EXPERIMENT_CONFIG_HH__ */
