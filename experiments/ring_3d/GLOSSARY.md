# Ring-3D glossary and configuration map

Read the [DBLP paper brief](../../docs/agents/ring-3d-paper-brief.md) and
[ASTRA-sim pivot](../../docs/agents/ring-3d-astra-pivot.md) before reusing a
paper term in a profile, report, or code symbol.

## Core terms

| Term | Meaning in this repository | Current 70B condition | Configuration / code | Measured artifact |
| --- | --- | --- | --- | --- |
| `clr_mask` | Immutable step-to-phase input, not simulator-detected gradients | Generated from a seeded decay/spike proxy or explicit profile labels | `generate_clr_schedule.py` to `clr_mask.csv` | `clr_mask.csv`, `manifest.json` |
| `clr_schedule.kind: "explicit_critical_steps"` | Exact one-based CLR labels imported into a profile | Phase-1 reference steps 1, 2, 153, 166 | `clr_schedule.critical_steps` | `clr_mask.csv`, `manifest.json` |
| CLR | `is_clr=1` selects the strict policy threshold | Step-dependent | `ExperimentConfig.hh` | `experiment.json`, flow rows |
| `selection_policy.p_low` | Logical-payload selection probability in CLR; **not paper $P_\mathrm{low}$ residual loss** | 0.5% | Profile and generated `experiment.json` | `decision`, `decision_hash` |
| `selection_policy.p_high` | Logical-payload selection probability outside CLR; **not paper $P_\mathrm{high}$ residual loss** | 10% | Profile and generated `experiment.json` | `decision`, `decision_hash` |
| $q$ | Packet-loss probability for `network.data_loss` data-plane impairment | 0 unless a profile explicitly enables `network.data_loss` | `network.data_loss.probability` | `transport_summary.csv` injected data drops |
| $D$ | Duration of the configured data-loss window | Unset unless a profile explicitly enables `network.data_loss` | `network.data_loss.start_ns`, `.duration_ns` | `manifest.json`, `network_config.txt` |
| Packet trimming (UEC 1.0.3 section 4.1) | A switch that fails buffer admission truncates a DSCP_TRIMMABLE packet to `MIN_TRIM_SIZE`, remarks it DSCP_TRIMMED, and forwards it on TC_med; its payload is not delivered | Enabled (`ftd`) on the flagship comparison profiles; disabled on a profile that omits it | `network.packet_trimming.mode` | Trim conversions, recovery controls, and terminal flow telemetry |
| FTD | Trim-and-forward-to-destination. This is the UEC 1.0.3 behavior: the trimmed packet reaches the destination, which returns a UET_TRIMMED NACK without accepting payload bytes | Enabled on the flagship comparison profiles | `network.packet_trimming.mode: "ftd"` | `trim_ftd_*` event and flow counters |
| BTS | Back-to-sender notification. UEC 1.0.3 section 4.1 explicitly excludes this ("Sending a trimmed packet back to the source ... is not part of this specification"); it models FastLane/P802.1Qdw and is research-only | Disabled | `network.packet_trimming.mode: "bts"` | `trim_bts_*` event and flow counters |
| DSCP_TRIMMED_LAST_HOP | Codepoint set when the trimming switch is the destination's own leaf. The source repairs the loss but does not treat it as a path or NSCC congestion signal | Enabled with trimming | `network.packet_trimming.last_hop_codepoint` | `trim_*_lasthop_*` events, `trim_lasthop_notifications` |
| Best-effort fabric | PFC disabled, zero headroom, shallow buffers. The only regime where trimming is meaningful (UEC 1.0.3 section 3.6.4.5) | On for the flagship comparison profiles (`pfc_enabled: false`); a profile that omits `network.fabric` keeps PFC on (lossless) | `network.fabric.pfc_enabled: false` | `switch_admission_drop`, `switch_trimmed_queue_drop` |
| PFC headroom | Buffer reserved to absorb packets in flight when a PAUSE is sent. Without PFC nothing drains it, so it becomes buffer that must fill before anything drops | 0 when PFC is off | `network.fabric.headroom_factor` | Effective drop threshold |
| Egress drop threshold | Per-queue byte bound on an egress queue, the `queue_trimmable.drop_threshold` / `queue_trimmed.drop_threshold` of UEC 1.0.3 section 4.1 | 4 MiB data / 1 MiB trimmed on the flagship comparison profiles; unbounded on a profile that omits it | `network.fabric.data_queue_bytes`, `.trimmed_queue_bytes` | Admission drops and trim conversions |
| TC_med | Egress tier for DSCP_TRIMMED, drained below TC_high control (queue 0) and ahead of the round-robin TC_low data queues, but capped at its configured bandwidth share | Queue 2 at 25% | `network.packet_trimming.trimmed_queue`, `.trimmed_queue_weight` | `switch_trimmed_queue_drop` |
| `trimmed_queue_weight` | Percent of egress bandwidth TC_med may take while TC_low has traffic. UEC 1.0.3 section 4.1 recommends WDRR at 25% and caps fair-queueing at 50%, because an unrestricted trimmed class can cause congestion collapse. 100 restores strict priority | 25 | `network.packet_trimming.trimmed_queue_weight` | Trim conversions versus data goodput |
| control plane | ACK (`0xFC`), NACK (`0xFD`), congestion notification (`0xFF`), PFC (`0xFE`), and named protocol/recovery control | No configured packet impairment in a lossless profile | Parsed before the QBB data-loss model; generated profiles set strict ACK/NACK priority at hosts and switches | Control attempts/delivery plus queue/drop totals in `transport_summary.csv` (per-packet rows in the `transport_events.csv.zst.NNN` segments; concatenate and decompress to reconstruct the stream) |
| data plane | RDMA UDP payload (`0x11`) subject to the explicit scoped impairment | No loss experiment is active | `network.data_loss` applies only after this wire classification | Data attempts, injected drops, retransmission bytes, and terminal flow telemetry |
| `microburst_bytes` | Bytes required by one synthetic background RDMA flow | 128 MiB | Profile JSON | Background `flow_events.csv` row |
| `microburst_flow_count` | Number of background flows | 7 | Profile JSON | Background flow rows |
| `microburst_offset_spacing_ns` | Start offset increment among background flows | 0 ns | Profile JSON | `start_time_ns` |
| `network.queue_monitor_interval_ns` | Period between switch egress-byte samples written to `qlen.txt` | 10 μs | Profile JSON | Queue peak in `summary.json`; cadence in the materialized network config |
| `provenance_control_bytes` | Physical payload for a selected logical payload's reliable control QP | 64 B | Generated `experiment.json` | `physical_bytes` |
| natural buffer drop | Switch MMU-admission or egress-queue rejection under offered load | Native calibration pending | `switch_admission_drop` or `switch_egress_queue_drop` in ns-3 | `data_natural_buffer_drop_count` and control counterpart |
| `decision_hash` | Stable selection hash over seed, run, operation, endpoints, and tag | Deterministic | `ExperimentConfig.hh` | `flow_events.csv` |
| `kDecisionScale` | Integer probability scale for deterministic selection | 1,000,000 | `ExperimentConfig.hh` | N/A |
| logical bytes | Original ASTRA payload size | 68,359,375 B sampled DP bucket | Trace/request | `logical_bytes` |
| physical bytes | Bytes of the QP modeled by ns-3 | 64 B if selected; full payload otherwise | `entry.h` | `physical_bytes` |

## Profile fields

Profiles are strict JSON input validated by `generate.py`; unknown fields fail.

| Field | Meaning | 70B example | Notes |
| --- | --- | --- | --- |
| `parallelism.tp`, `.pp`, `.dp` | Logical parallel dimensions | 8, 1, 2 | Ranks equal $TP\times PP\times DP$ |
| `steps` | Modeled optimizer steps in the bounded experiment window | 20 | Long enough to express the decaying CLR schedule; the incast fires at the profile's `microburst_trigger_step` |
| `compute_duration_us` | Simulated compute duration per emitted compute node | 5,376 | Workload abstraction, not measured framework time |
| `tp_all_reduce_bytes`, `pp_bytes`, `dp_all_reduce_bytes` | Logical collective payloads per emitted event | 64 MiB, 0, 68,359,375 B | Separate from physical transport overhead |
| `seed` | Selection and CLR-mask seed unless overridden | 314159265 | Pair treatment must retain it |
| `dp_all_reduce_implementation` | Native algorithm for DP communicator groups only | `ring` (default) or `direct`/`direct<window>` | Written as `all-reduce-implementation-per-group` in `system.json`; TP/PP always keep the global ring. `direct` gives every DP rank $DP-1$ concurrent inbound shards |
| `network` | Typed Clos/ring topology and packet settings | 400 Gb/s Clos | Network schema is validated in `topology.py` |
| `network.queue_monitor_interval_ns` | Positive periodic queue-sampling interval | 10,000 ns | Prevents observability work from scaling with every packet event |
| `network.data_loss` | Optional independent physical data-only receive impairment | Absent | Requires probability, time window, scope, and RNG stream; separate from logical selection thresholds and packet trimming |
| `network.transport_recovery` | Required bounded recovery budget when physical loss or trimming is enabled | Absent | Requires positive retransmission timeout and retry budget; terminal exhaustion is recorded as a failure |
| `network.transport_recovery.selective_repair` | Switches recovery from go-back-$N$ to range-based selective repair with out-of-order acceptance (`SELECTIVE_RETRANSMISSION` in ns-3); not a SACK bitmap | `false` (go-back-$N$) | `network.transport_recovery.selective_repair` | `transport_summary.csv` W, retransmitted-byte counts |
| `network.transport_recovery.no_progress_timeout_ns` | Forward-progress deadline: fail a queue pair whose cumulative acknowledgement has not advanced for this simulated interval | `5000000000` | Liveness bound for recovery loops sustained by budget-exempt signals (NACKs, trim notifications); failure reason `no_forward_progress` |
| `network.fabric` | Switch buffer and flow-control regime | Absent (32 MB, PFC on) | Requires `buffer_size_mb`, `pfc_enabled`, `data_queue_bytes`; optional `headroom_factor`, `trimmed_queue_bytes`. Under `network.switch.profile: "ue"` the two queue bounds come from the preset and must be absent. Mandatory when trimming is enabled, and must be identical across every arm of a comparison |
| `network.packet_trimming` | Optional UEC 1.0.3 section 4.1 packet trimming | Absent | Requires `mode: "ftd"` (UET-conformant) or `"bts"` (research-only); optional `trimmed_queue` (default 2), `trimmed_queue_weight` (default 25), `min_trim_size_bytes` (default 24), and `last_hop_codepoint` (default `true`). Only switch admission or egress queue rejection can trigger it |
| `network.congestion_control` | End-host sender reaction to congestion | Absent (`none`) | `mode: "none"` writes `CC_MODE 12`, where a queue pair blasts at link rate inside a static window and nothing slows it; `mode: "dcqcn"` writes `CC_MODE 1`, Mellanox DCQCN, a rate controller; `mode: "nscc"` writes `CC_MODE 11`, UEC 1.0.3 NSCC, one window per queue pair moved by every acknowledgement's ECN mark and round trip and cut by trims and declared losses, which requires `network.load_balancing.mode` `ev_hash` or `spray_uniform` and is refused under `ecmp` and on a ring. Optional `rate_ai_fraction` (default 1/2000), `rate_hai_fraction` and `min_rate_fraction` (default 1/1000) are fractions of `link_rate`, so a profile keeps its aggressiveness at any link speed. Optional `ecn_threshold_scale` (default 1, positive) multiplies every switch `KMIN_MAP` and `KMAX_MAP` threshold, rounded to whole KB, and leaves the rate keys and `PMAX_MAP` unchanged. Name the mode in every result |
| `network.load_balancing` | How a leaf spreads a flow's packets over its spines (Clos only) | Absent (`ecmp`) | `mode: "ecmp"` hashes the four-tuple, so a flow keeps one path, and writes no key; `ev_hash` adds a per-packet 16-bit entropy value to the hash; `spray_uniform` has the sender name a spine per packet. Both write `LOAD_BALANCING`, reorder a flow's packets, make the receiver acknowledge every packet, and require `transport_recovery.selective_repair` and `packet_trimming.mode: "ftd"`; both are refused with a recovery `selection_policy.domain` |
| `network.switch` | Switch ECN marking and queue bounds | Absent (`default`) | `profile: "default"` is the historical switch: the HPCC `KMIN_MAP`/`KMAX_MAP` table scaled by `congestion_control.ecn_threshold_scale`, the queue bounds of `network.fabric`, and a RED marking probability at `KMAX` of `ecn_marking_probability` (default 0.2, in $(0, 1]$, written to `PMAX_MAP` for every link speed). `profile: "ue"` is UEC 1.0.3 section 3.6.17 for a trimming fabric: from Plane_BDP, the link rate times the longest unloaded host-to-host round trip (the backend's `maxRtt`), it marks queue_low from 0.2 to 0.8 Plane_BDP with probability 1 at the top (rounded to whole KB), trims at 1 Plane_BDP (`DATA_QUEUE_BYTES`) and drops trimmed packets at 1 Plane_BDP (`TRIMMED_QUEUE_BYTES`). The switch never marks queue 0 or a trimmed packet. `ue` requires `network.packet_trimming` and refuses `ecn_marking_probability`, `congestion_control.ecn_threshold_scale` and `fabric.data_queue_bytes`/`trimmed_queue_bytes`, which it sets; the manifest's `switch` entry records the derived values. Queue 0 (TC_high) stays unbounded, not at the section's 1 Plane_BDP |
| `selection_policy.domain` | Where the phase-aware budget is spent | Absent (`admission`) | `admission` substitutes whole payloads before they are offered. `recovery` offers everything and lets a switch-trimmed packet's bytes go, inside the same budget; it requires `network.transport_recovery.selective_repair` and `network.packet_trimming.mode: ftd` and is refused without them. It writes `semantics: recovery_forgiveness`, and `evaluate_shedding` never sheds in it, so the two domains cover the same eligible population |
| `selection_policy` | Typed low/high logical-admission selection knobs | `p_low=0.005`, `p_high=0.1` | Profile, manifest, and `experiment.json` | Materialized selection probabilities |
| `microburst_enabled` | Enables synthetic background flows | `true` | `false` is the no-incast control |
| `microburst_bytes` | Per-flow offered background bytes | 128 MiB | Required even when disabled |
| `microburst_flow_count` | Background source count | 7 | Must leave a destination rank |
| `microburst_destination_rank` | Shared background destination | 8 | Creates an incast |
| `microburst_offset_spacing_ns` | Flow start staggering | 0 | Zero means simultaneous scheduling |
| `model.gradient_accumulation_steps` | Accumulation microbatches represented by the 70B sampled layer window | 2 | The generator emits the sampled TP pattern for each accumulation microbatch; it does not replay every model layer |
| `model` | Structural or bounded event-window metadata | 70B BF16/FP16 sample | Validated against trace shape |
| `workload.kind: "sequential_dp_all_reduce"` | Communication-only trace with one chained DP All-Reduce per step | 64-rank Phase-1 reference | Requires $TP=PP=1$, zero compute/TP/PP bytes, and no model metadata |

`selection_policy` is a strict profile object. `compare.py` holds the fixed-low
baseline at `p_low` for both phases, then compares it with a policy that uses
`p_low` in CLR and `p_high` outside CLR. The low value must be in $(0, 1\%]$.

An omitted `clr_schedule` uses the seeded decay-and-spike proxy. An explicit
schedule is the profile's exact phase input and takes precedence over any
decay/spike command-line values. It transfers phase labels only; it does not
transfer a gradient detector, packet-loss event, or DBLP residual-loss rule.

## Background microburst

The generator derives a source/destination list, assigns every flow
`size_bytes=microburst_bytes`, and starts flow $i$ at:

$$
\text{trigger time}+i\times\texttt{microburst\_offset\_spacing\_ns}.
$$

The trigger is the first eligible step-2 DP All-Reduce request. This defines
offered background bytes and alignment, not a fixed burst lifetime. Flow end
time is an ns-3 result affected by queueing, PFC, congestion control, and path
contention.

The Llama paired condition and both 100B structural topology conditions retain
their microbursts as explicit, reproducible congestion stressors. The
no-incast profile is the negative control. Do not characterize the stressor as
a naturally emitted framework burst or as packet loss.

## Naming rules

- Say **selection probability** for the current `selection_policy.p_low` and
  `selection_policy.p_high` fields.
- Reserve **packet loss** for transport-level data delivery failures.
- `network.data_loss` is physical data-plane impairment. It never changes
  `selection_policy.p_low` or `selection_policy.p_high`, which remain logical
  payload-selection inputs.
- `network.packet_trimming` is independent from `network.data_loss`. It turns
  a congestion-rejected RDMA data packet into explicit loss metadata, never
  placeholder bytes or partial payload delivery.
- Buffer depth decides *how* incast produces tail latency: a deep buffer makes
  it queueing delay with no loss, a shallow best-effort buffer makes it loss
  that trimming reports. The two are different physical claims, so
  `network.fabric` must be identical across every arm of a comparison.
- A trimmed packet rides TC_med (`network.packet_trimming.trimmed_queue`,
  default queue 2), not the TC_high control queue, and it obeys that queue's
  admission thresholds. TC_med is drained ahead of data but is limited to
  `trimmed_queue_weight` percent of the link while data is queued, so trimmed
  traffic cannot starve payload: the congestion-collapse guard of UEC 1.0.3
  section 4.1. Per UEC 1.0.3 section 4.1 there is no guarantee a
  trimmed packet is delivered; `switch_trimmed_queue_drop` records the cases
  where it is not, and the RTO remains the backstop.
- A trimmed packet is not a successful data packet, and completion still
  requires ACK-backed delivery after repair. The default repair is go-back-$N$;
  a profile can instead set `network.transport_recovery.selective_repair` for
  range-based repair with out-of-order acceptance (this is not a SACK bitmap).
  Neither mode implements packet spraying, reorder buffering beyond the
  accepted out-of-order ranges, or the optional `DSCP_TRIMMABLE_RTX` codepoint
  for retransmitted data (UEC 1.0.3 section 3.6.4.7.1 marks that codepoint
  OPTIONAL).
- Under `CC_MODE 1` every trim notification is a rate cut, last-hop trims
  included. UEC 1.0.3 p. 356 excludes DSCP_TRIMMED_LASTHOP from the congestion
  signal only where RCCC covers the last hop; this model has no RCCC, so
  keeping the exclusion would leave destination incast uncontrolled.
- `timeouts` and `cnp_received` are per-queue-pair cumulative counts:
  retransmission-timeout firings that actually rescheduled data, and rate cuts
  taken. `m_recovery_retries` resets on every acknowledgement advance and
  cannot answer either question. `first_trim_ns` and `first_repair_ns` are
  simulated times with zero meaning never; their difference, summarized as
  `first_trim_to_first_repair_ns`, separates a repair-driven tail from a
  congestion-control-driven one.
- `rto_fired` and `cnp_taken` in `transport_summary.csv` are host-transport
  reactions, not packets. They ride the control plane and carry zero bytes.
  `trim_forgiven` is the receiver's answer to a switch's trim, not a second
  conversion: it rides the data plane with the payload bytes it released, and
  it is excluded from the packet-trimming conversion counts.
- **W'** is `(trimmed - forgiven) / offered`: the trimmed bytes the transport
  still had to repair. It is comparable with W only inside one run. Across
  domains the denominators differ, because admission shedding takes whole
  payloads off the wire while forgiveness only releases packets a switch
  already trimmed.
- `forgiven_bytes` are offered and undelivered. They stay inside
  `physical_bytes`, which remains the offered figure that joins `fct.txt` and
  denominates W; `delivered_bytes` is the figure that excludes them. A
  forgiven byte is never described as delivered, and a forgiven range is never
  described as a packet loss the policy caused: the switch trimmed it, and the
  policy declined to ask for it again.
- The **ledger law** is `shed + forgiven <= p(step) * eligible` per receiving
  rank and step, with `p` the strict CLR threshold on a critical step. Both
  terms only grow. `summary.json`'s `forgiveness.ledger_law` re-derives it from
  the telemetry and the run's own CLR mask; `violated` invalidates the arm.
- Configured control-impaired loss is always zero, but controls can still be
  delayed or dropped by modeled queue/admission behavior; use
  `transport_summary.csv` to distinguish those cases.
- A provenance replacement QP is UDP data on priority group 1, not an ACK,
  NACK, PFC, CNP, or a queue-0 wire-control packet.
- Reserve **residual-loss tolerance** for a future DBLP-like stop condition.
- Say **incast** for the finite background RDMA stressor.
- Never call a 64-byte provenance control QP a “dropped packet.”
- Do not call the existing priority-group mapping a general loss-protected
  control plane; see the [loss-tolerant RDMA decision](../../docs/agents/loss-tolerant-rdma-decision.md).
