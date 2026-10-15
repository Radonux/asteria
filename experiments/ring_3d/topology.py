"""Typed physical-topology specifications for reproducible ns-3 experiments.

The bundled ns-3 topology format declares one *bidirectional* link per edge.
This module keeps construction, validation, serialization, and manifest metadata
in one place so profile parsing cannot drift from emitted topology files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Any, Callable

try:
    from .links import (
        ClosIndex,
        LinkFailure,
        LinkOverride,
        load_link_failures,
        load_link_overrides,
    )
    from .switch import (
        ECN_THRESHOLDS_KB,
        DefaultSwitch,
        SwitchProfile,
        SwitchRequest,
        load_switch_request,
    )
except ImportError:
    from links import (
        ClosIndex,
        LinkFailure,
        LinkOverride,
        load_link_failures,
        load_link_overrides,
    )
    from switch import (
        ECN_THRESHOLDS_KB,
        DefaultSwitch,
        SwitchProfile,
        SwitchRequest,
        load_switch_request,
    )

DEFAULT_PACKET_PAYLOAD_BYTES = 1_000
MAX_PACKET_PAYLOAD_BYTES = 9_000
# One-way propagation of a host's link to its switch and of a link between two
# switches. The defaults are what every profile ran before the knobs existed.
DEFAULT_HOST_LINK_DELAY_NS = 5_000
DEFAULT_SWITCH_LINK_DELAY_NS = 12_500
DATA_LOSS_SCOPES = {
    "all",
    "host_to_switch",
    "switch_to_host",
    "switch_to_switch",
}
PACKET_TRIM_MODES = {"ftd", "bts"}
LOAD_BALANCING_MODES = {"ecmp", "ev_hash", "spray_uniform", "spray_policy"}
# The modes whose identification names the spine a packet goes up.
SPINE_NAMING_MODES = {"spray_uniform", "spray_policy"}
# The most spines a spray_policy report grades.
MAX_REPORTED_SPINES = 32
# What chooses each data packet's entropy value under ev_hash.
PATH_SELECTORS = ("ops", "reps", "ue_oblivious", "ue_aware", "mrc")
# What chooses each data packet's spine under spray_uniform instead of the
# uniform draw.
SPINE_SELECTORS = (
    "ack_redirect",
    "reps_hold",
    "reps_flow_hold",
    "reps_oracle",
    "mrc_hold",
    "reps_spine",
)
# The spine selectors that keep a loss hold: per host, per destination leaf,
# per spine either of the two, or per destination host under hold_scope, for
# reps_hold and mrc_hold; per queue pair for reps_flow_hold.
LOSS_HOLD_SELECTORS = ("reps_hold", "reps_flow_hold", "mrc_hold")
# The loss hold selectors that read hold_scope and write the host's loss
# trace.
HOLD_SCOPE_SELECTORS = ("reps_hold", "mrc_hold")
# The most spines a receiver's hold names in an acknowledgement's bitmap.
MAX_RECEIVER_HOLD_SPINES = 8
# The most spines a closing packet's per-spine counts name under
# tail_recovery.
MAX_TAIL_RECOVERY_SPINES = 8
# The backend's tail recovery delay value for its default, the fabric's
# largest base RTT, which it refuses as a given value.
TAIL_RECOVERY_DELAY_DEFAULT_SENTINEL = 2**64 - 1
# The value an acknowledgement's redirect field names no spine with, so
# ack_redirect names at most this many spines.
MAX_REDIRECT_SPINES = 255
# The ns-3 CC_MODE number of each congestion_control.mode. 12 is no sender
# reaction at all: a queue pair is set to link rate at creation and nothing ever
# changes it. 1 is Mellanox DCQCN. 11 is UEC NSCC, a window per queue pair
# clocked by per-packet acknowledgements.
CONGESTION_CONTROL_MODES = {"none": 12, "dcqcn": 1, "nscc": 11}
# The HPCC-era literals RATE_AI 50Mb/s, RATE_HAI 100Mb/s, MIN_RATE 100Mb/s were
# written for 100 Gb/s links; as fractions of the link they are 1/2000, 1/1000,
# 1/1000, which is what carries them to any other link rate.
DEFAULT_RATE_AI_FRACTION = 1 / 2000
DEFAULT_RATE_HAI_FRACTION = 1 / 1000
DEFAULT_MIN_RATE_FRACTION = 1 / 1000
# Queue 0 carries DSCP_CONTROL (TC_high) and priority groups 1 and 3 carry UET
# data (TC_low), so TC_med for DSCP_TRIMMED must avoid all three. UEC 1.0.3
# section 4.1.4.1 requires trimmed packets to sit in their own traffic class.
DATA_PRIORITY_GROUPS = frozenset({1, 3})
CONTROL_PRIORITY_GROUP = 0
DEFAULT_PACKET_TRIM_QUEUE = 2
# UEC 1.0.3 Table 4-1: UET over UDP/IP needs 24 B to retain the UDP header and
# the PDS request header that identify the trimmed packet.
DEFAULT_MIN_TRIM_SIZE_BYTES = 24
MAX_PRIORITY_GROUP = 7
# UEC 1.0.3 section 4.1 recommends WDRR with 25% of the bandwidth allocated to
# trimmed packets, and caps fair-queueing at 50%, because an unrestricted
# trimmed class can drive congestion collapse.
DEFAULT_TRIMMED_QUEUE_WEIGHT = 25
DEFAULT_BUFFER_SIZE_MB = 32
DEFAULT_HEADROOM_FACTOR = 3


@dataclass(frozen=True)
class TransportRecovery:
    """Bounded sender recovery for silent loss.

    The retry budget counts consecutive retransmission timeouts without ACK
    progress — silence, the only signal consistent with a dead path. Trim
    notifications and NACKs are live feedback from the fabric and never
    consume the budget; congestion is recoverable by definition.
    """

    retransmission_timeout_ns: int
    max_retransmission_retries: int
    # UEC-style selective repair: retransmit exactly the reported trimmed or
    # missing ranges and accept out-of-order payload at the receiver. The
    # timeout/retry budget remains the silent-loss fallback. Off by default so
    # every existing arm keeps go-back-N semantics.
    selective_repair: bool = False
    # Forward-progress deadline: fail a queue pair whose cumulative
    # acknowledgement has not advanced for this simulated interval. The retry
    # budget cannot bound a recovery loop sustained by budget-exempt signals
    # (NACKs, trim notifications); this deadline bounds the transfer itself.
    # Five simulated seconds is orders beyond any legitimate no-progress span
    # (the engineered burst is ~19 ms) yet ends a livelocked arm in minutes
    # of wall clock instead of hours.
    no_progress_timeout_ns: int = 5_000_000_000

    def manifest(self) -> dict[str, int | bool]:
        return {
            "enabled": True,
            "retransmission_timeout_ns": self.retransmission_timeout_ns,
            "max_retransmission_retries": self.max_retransmission_retries,
            "selective_repair": self.selective_repair,
            "no_progress_timeout_ns": self.no_progress_timeout_ns,
        }


@dataclass(frozen=True)
class CongestionControl:
    """End-host sender reaction to congestion.

    ``none`` is what every arm before this knob ran: CC_MODE 12, where a queue
    pair blasts at link rate inside a static window and no signal slows it.
    ``dcqcn`` is Mellanox DCQCN (CC_MODE 1), a rate controller. ``nscc`` is
    UEC's own NSCC (CC_MODE 11): one window per queue pair, moved by the ECN
    mark and the round trip of every acknowledged packet and cut by trims and
    losses, so it needs a load-balancing mode that acknowledges every packet.
    The three rates are DCQCN's and are fractions of the link rate rather than
    absolute literals, so a profile keeps its intended aggressiveness at any
    link speed. ``ecn_threshold_scale`` multiplies every switch KMIN and KMAX
    threshold and leaves the marking probability alone.
    """

    mode: str
    rate_ai_fraction: float = DEFAULT_RATE_AI_FRACTION
    rate_hai_fraction: float = DEFAULT_RATE_HAI_FRACTION
    min_rate_fraction: float = DEFAULT_MIN_RATE_FRACTION
    ecn_threshold_scale: float = 1.0

    @property
    def cc_mode(self) -> int:
        return CONGESTION_CONTROL_MODES[self.mode]

    def rates_bps(self, link_rate: str) -> tuple[int, int, int]:
        """Resolve the three rate knobs against a link rate such as 400Gbps."""
        link_bps = link_rate_bits_per_second(link_rate)
        return (
            round(link_bps * self.rate_ai_fraction),
            round(link_bps * self.rate_hai_fraction),
            round(link_bps * self.min_rate_fraction),
        )

    def ecn_threshold_maps(self) -> str:
        """The KMAX_MAP and KMIN_MAP lines, thresholds scaled, rates not."""
        kmax = " ".join(
            f"{rate} {round(high * self.ecn_threshold_scale)}"
            for rate, _, high in ECN_THRESHOLDS_KB
        )
        kmin = " ".join(
            f"{rate} {round(low * self.ecn_threshold_scale)}"
            for rate, low, _ in ECN_THRESHOLDS_KB
        )
        count = len(ECN_THRESHOLDS_KB)
        return f"KMAX_MAP {count} {kmax}\nKMIN_MAP {count} {kmin}\n"

    def manifest(self, link_rate: str) -> dict[str, str | int | float]:
        rate_ai, rate_hai, min_rate = self.rates_bps(link_rate)
        return {
            "mode": self.mode,
            "cc_mode": self.cc_mode,
            "rate_ai_fraction": self.rate_ai_fraction,
            "rate_hai_fraction": self.rate_hai_fraction,
            "min_rate_fraction": self.min_rate_fraction,
            "ecn_threshold_scale": self.ecn_threshold_scale,
            "rate_ai_bps": rate_ai,
            "rate_hai_bps": rate_hai,
            "min_rate_bps": min_rate,
        }


@dataclass(frozen=True)
class LoadBalancing:
    """How a leaf spreads a flow's data packets over its spine uplinks.

    ``ecmp`` hashes the four-tuple, so a flow keeps one path. ``ev_hash`` adds
    a 16-bit entropy value per packet to that hash, which ``selector`` chooses:
    ``ops`` draws it afresh, ``reps`` reuses the values that came back on
    unmarked acknowledgements, ``ue_oblivious`` rotates through a set of
    values, ``ue_aware`` rotates skipping a value once after a congestion
    report, and ``mrc`` rotates over values it holds GOOD, skipped or assumed
    bad. ``spray_uniform`` has the sender name a spine per packet, uniformly at
    random, and the source leaf send the packet up that spine; with
    ``selector`` ``ack_redirect`` the sender names instead the spine the latest
    acknowledgement redirects it to, else the spine that carried the packet it
    answers; with ``reps_hold`` or ``reps_flow_hold`` it chooses by REPS over
    the spines, leaving out those its loss hold holds, with ``mrc_hold`` by
    MRC over the spines with the same loss hold, and with
    ``reps_oracle`` by REPS over the spines, leaving out those the switches'
    state shows a fault on, and with ``reps_spine`` by REPS over the spines,
    leaving out none. ``spray_policy`` names a spine per packet the same
    way, drawn from scores the sender builds out of the grades the receiver
    reports in every acknowledgement. Every mode but ``ecmp`` reorders a
    flow's packets on the way.
    """

    mode: str
    selector: str = "ops"
    # The mode's or the selector's parameters by profile key, defaults filled
    # in.
    parameters: tuple[tuple[str, ParameterValue], ...] = ()

    def _names_selector(self) -> bool:
        return self.mode == "ev_hash" or self.selector in SPINE_SELECTORS

    def traces_spine_loss(self) -> bool:
        """Whether the run traces each host's pooled loss intervals: only the
        host-wide pool counts over them, so spray_policy with loss_thresholds
        and sender_pool_weight above 0, and only outside the switch_queue
        cost source, under which the sender counts no loss; and the loss
        hold of reps_hold and mrc_hold, kept per host, per destination leaf
        or per spine either. reps_flow_hold's holds are per queue pair, and
        the backend refuses the trace there."""
        values = dict(self.parameters)
        if self.mode == "spray_uniform":
            return self.selector in HOLD_SCOPE_SELECTORS
        return (
            self.mode == "spray_policy"
            and "loss_thresholds" in values
            and float(values.get("sender_pool_weight", 0.0)) > 0
            and values.get("grade_cost_source") != "switch_queue"
        )

    def traces_receiver_hold(self) -> bool:
        """Whether the run traces the receiving hosts' holds: under a loss
        hold selector with receiver_hold, the only configuration in which
        the backend reads the trace."""
        return (
            self.selector in LOSS_HOLD_SELECTORS
            and dict(self.parameters).get("receiver_hold") is True
        )

    def recovers_tail(self) -> bool:
        """Whether a message's last packets report each spine's sent count
        under spray_uniform, which names at most 8 spines."""
        return dict(self.parameters).get("tail_recovery") is True

    def manifest(self) -> dict[str, ParameterValue]:
        if not self._names_selector():
            return {"mode": self.mode, **dict(self.parameters)}
        return {"mode": self.mode, "selector": self.selector, **dict(self.parameters)}

    def parameter_settings(self) -> str:
        """PATH_SELECTOR, unless the selector is ``ops``, the simulator's
        default, and the parameters, each as its simulator key."""
        selector = (
            f"PATH_SELECTOR {self.selector}\n"
            if self._names_selector() and self.selector != "ops"
            else ""
        )
        values = dict(self.parameters)
        return selector + "".join(
            f"{parameter.key} "
            f"{(parameter.write or _setting)(values[parameter.name])}\n"
            for parameter in _parameters_of(self.mode, self.selector)
            if parameter.name in values
        )


@dataclass(frozen=True)
class PacketTrimming:
    """UEC 1.0.3 section 4.1 switch packet-trimming policy.

    ``mode: "ftd"`` is the specified behavior: a switch that fails buffer
    admission truncates the packet to ``min_trim_size_bytes``, remarks it
    DSCP_TRIMMED, and forwards it to the destination on TC_med, where it is
    still subject to that queue's drop threshold. ``mode: "bts"`` returns the
    notification to the sender instead; UEC 1.0.3 section 4.1 explicitly places
    that outside the specification, so it is a research-only mode.
    """

    mode: str
    trimmed_queue: int
    trimmed_queue_weight: int
    min_trim_size_bytes: int
    last_hop_codepoint: bool

    @property
    def uec_conformant(self) -> bool:
        return self.mode == "ftd"

    def manifest(self) -> dict[str, str | int | bool]:
        return {
            "enabled": True,
            "mode": self.mode,
            "trigger": "switch_admission_or_egress_rejection",
            "trimmed_queue": self.trimmed_queue,
            "trimmed_queue_weight": self.trimmed_queue_weight,
            "min_trim_size_bytes": self.min_trim_size_bytes,
            "last_hop_codepoint": self.last_hop_codepoint,
            "uec_conformant": self.uec_conformant,
        }


@dataclass(frozen=True)
class SwitchFabric:
    """Switch buffer and flow-control regime.

    These knobs decide *how* congestion manifests, so they must be held
    constant across every arm of a comparison. A deep buffer turns incast into
    queueing delay and nothing is ever dropped; a shallow best-effort buffer
    turns it into loss that trimming reports. UEC 1.0.3 section 3.6.4.5
    excludes PFC from best-effort networks, and section 4.1 sizes data queues
    at roughly one bandwidth-delay product.
    """

    buffer_size_mb: int
    pfc_enabled: bool
    headroom_factor: int
    data_queue_bytes: int
    trimmed_queue_bytes: int

    def manifest(self) -> dict[str, int | bool]:
        return {
            "buffer_size_mb": self.buffer_size_mb,
            "pfc_enabled": self.pfc_enabled,
            "headroom_factor": self.headroom_factor,
            "data_queue_bytes": self.data_queue_bytes,
            "trimmed_queue_bytes": self.trimmed_queue_bytes,
        }


@dataclass(frozen=True)
class DataPlaneLoss:
    """Data-only receive impairment independent from transport recovery."""

    probability: float
    start_ns: int
    duration_ns: int
    scope: str
    source_host: int | None
    destination_host: int | None
    receiver_node: int | None
    rng_stream: int

    def manifest(self) -> dict[str, int | float | str | None]:
        return {
            "enabled": True,
            "probability": self.probability,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "scope": self.scope,
            "source_host": self.source_host,
            "destination_host": self.destination_host,
            "receiver_node": self.receiver_node,
            "rng_stream": self.rng_stream,
        }


@dataclass(frozen=True)
class ClosNetwork:
    """A two-stage leaf-spine fabric with host-attached leaf switches.

    ``spine_count`` is the designed spine tier; ``failed_spine_count`` models
    a degraded state — spines lost to failure or drained for maintenance —
    by leaving that many designed spines dark. A rail-optimized 1:1 fabric
    with failed spines is how production AI clusters actually experience
    sustained oversubscription; it is an operating condition, not a design.
    """

    link_rate: str
    packet_payload_bytes: int
    queue_monitor_start_ns: int
    queue_monitor_interval_ns: int
    hosts_per_leaf: int
    spine_count: int
    failed_spine_count: int = 0
    host_link_delay_ns: int = DEFAULT_HOST_LINK_DELAY_NS
    switch_link_delay_ns: int = DEFAULT_SWITCH_LINK_DELAY_NS
    link_overrides: tuple[LinkOverride, ...] = ()
    link_failures: tuple[LinkFailure, ...] = ()

    @property
    def live_spine_count(self) -> int:
        return self.spine_count - self.failed_spine_count

    def clos_index(self, host_count: int) -> ClosIndex:
        """Node ids of the built fabric's leaves and spines."""
        return ClosIndex(
            host_count=host_count,
            leaf_count=host_count // self.hosts_per_leaf,
            spine_count=self.live_spine_count,
        )

    def link_failure_config(self, host_count: int) -> str:
        """The LINK_FAILURE lines of every failure, in profile order."""
        index = self.clos_index(host_count)
        return "".join(
            line
            for failure in self.link_failures
            for line in failure.config_lines(index)
        )
    data_loss: DataPlaneLoss | None = None
    transport_recovery: TransportRecovery | None = None
    packet_trimming: PacketTrimming | None = None
    fabric: SwitchFabric | None = None
    congestion_control: CongestionControl = CongestionControl(mode="none")
    load_balancing: LoadBalancing = LoadBalancing(mode="ecmp")
    switch: SwitchProfile = DefaultSwitch()

    @property
    def kind(self) -> str:
        return "clos"


@dataclass(frozen=True)
class RingNetwork:
    """A host-attached bidirectional ring of switches.

    Each modeled accelerator owns one RDMA host and one attached switch. The
    switches form the ring so ns-3 can model switch queues, PFC, and equal-cost
    paths in both directions; a direct host-only ring cannot provide those
    switch-level signals in this backend.
    """

    link_rate: str
    packet_payload_bytes: int
    queue_monitor_start_ns: int
    queue_monitor_interval_ns: int
    host_link_delay_ns: int = DEFAULT_HOST_LINK_DELAY_NS
    switch_link_delay_ns: int = DEFAULT_SWITCH_LINK_DELAY_NS
    link_overrides: tuple[LinkOverride, ...] = ()
    data_loss: DataPlaneLoss | None = None
    transport_recovery: TransportRecovery | None = None
    packet_trimming: PacketTrimming | None = None
    fabric: SwitchFabric | None = None
    congestion_control: CongestionControl = CongestionControl(mode="none")
    switch: SwitchProfile = DefaultSwitch()

    @property
    def kind(self) -> str:
        return "ring"

    @property
    def load_balancing(self) -> LoadBalancing:
        """A ring has no spine tier, so its switches only hash."""
        return LoadBalancing(mode="ecmp")

    @property
    def link_failures(self) -> tuple[LinkFailure, ...]:
        """Failures name spines, which a ring has none of."""
        return ()

    def link_failure_config(self, host_count: int) -> str:
        del host_count
        return ""


PhysicalNetwork = ClosNetwork | RingNetwork


@dataclass(frozen=True)
class TopologyLink:
    """One bidirectional ns-3 link declaration."""

    source: int
    destination: int
    delay_ns: int
    rate: str
    error_rate: float = 0.0

    @property
    def rate_bps(self) -> int:
        return link_rate_bits_per_second(self.rate)


@dataclass(frozen=True)
class TopologyLayout:
    """Validated topology data independent of its text serialization."""

    kind: str
    description: str
    host_count: int
    node_count: int
    switch_ids: tuple[int, ...]
    links: tuple[TopologyLink, ...]
    link_rate: str
    details: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.host_count <= 0 or self.node_count <= 0:
            raise ValueError("topology must contain at least one host and node")
        if self.host_count + len(self.switch_ids) != self.node_count:
            raise ValueError("topology host and switch counts do not partition nodes")
        if len(set(self.switch_ids)) != len(self.switch_ids):
            raise ValueError("topology switch IDs must be unique")
        if any(node < 0 or node >= self.node_count for node in self.switch_ids):
            raise ValueError("topology switch ID is outside the node range")

        undirected_edges: set[tuple[int, int]] = set()
        for link in self.links:
            if not 0 <= link.source < self.node_count:
                raise ValueError("topology link source is outside the node range")
            if not 0 <= link.destination < self.node_count:
                raise ValueError("topology link destination is outside the node range")
            if link.source == link.destination:
                raise ValueError("topology links must not contain self-loops")
            edge = tuple(sorted((link.source, link.destination)))
            if edge in undirected_edges:
                raise ValueError(
                    "topology must not declare duplicate bidirectional links"
                )
            undirected_edges.add(edge)

    def write(self, path: Path) -> None:
        """Write the exact topology format consumed by the bundled ns-3 parser."""
        with path.open("w", encoding="utf-8") as topology:
            topology.write(
                f"{self.node_count} {len(self.switch_ids)} {len(self.links)}\n"
            )
            topology.write(" ".join(str(node) for node in self.switch_ids) + "\n")
            for link in self.links:
                topology.write(
                    f"{link.source} {link.destination} {link.rate} "
                    f"{_milliseconds(link.delay_ns)}ms "
                    f"{format(link.error_rate, '.17g')}\n"
                )

    def manifest(self) -> dict[str, int | str]:
        """Return auditable physical-network metadata for manifests and reports."""
        return {
            "kind": self.kind,
            "description": self.description,
            "host_count": self.host_count,
            "node_count": self.node_count,
            "switch_count": len(self.switch_ids),
            "link_count": len(self.links),
            "link_rate": self.link_rate,
            **dict(self.details),
        }


def _milliseconds(nanoseconds: int) -> str:
    """Nanoseconds as the plain decimal milliseconds ns-3 parses, 5000 as 0.005."""
    return format(Decimal(nanoseconds) / 1_000_000, "f")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _link_rate(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Gbps"):
        raise ValueError("network.link_rate must be a rate such as 200Gbps")
    return value


def link_rate_bits_per_second(link_rate: str) -> int:
    """Bits per second of a validated ``<n>Gbps`` link rate."""
    return round(float(link_rate[: -len("Gbps")]) * 1_000_000_000)


def _fraction(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in (0, 1]")
    fraction = float(value)
    if not isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(f"{field} must be a number in (0, 1]")
    return fraction


def _load_congestion_control(document: dict[str, Any]) -> CongestionControl:
    """Parse the optional knob, defaulting to the historical no-CC transport."""
    if "congestion_control" not in document:
        return CongestionControl(mode="none")
    control = document["congestion_control"]
    optional = {
        "rate_ai_fraction",
        "rate_hai_fraction",
        "min_rate_fraction",
        "ecn_threshold_scale",
    }
    if not isinstance(control, dict) or not {"mode"} <= set(control) <= (
        {"mode"} | optional
    ):
        raise ValueError(
            "network.congestion_control must contain 'mode' and may contain "
            f"{sorted(optional)}"
        )
    mode = control["mode"]
    if not isinstance(mode, str) or mode not in CONGESTION_CONTROL_MODES:
        raise ValueError(
            "network.congestion_control.mode must be one of "
            f"{sorted(CONGESTION_CONTROL_MODES)}"
        )
    rate_ai = _fraction(
        control.get("rate_ai_fraction", DEFAULT_RATE_AI_FRACTION),
        "network.congestion_control.rate_ai_fraction",
    )
    rate_hai = _fraction(
        control.get("rate_hai_fraction", DEFAULT_RATE_HAI_FRACTION),
        "network.congestion_control.rate_hai_fraction",
    )
    # Hyper-additive increase is the escalation additive increase escalates
    # into, so a profile with the two the wrong way round is physically
    # backwards: the recovery would slow down the longer it went unchallenged.
    # Each fraction alone is in range, so only the ordering catches it.
    if rate_ai > rate_hai:
        raise ValueError(
            "network.congestion_control.rate_ai_fraction must not exceed "
            "rate_hai_fraction"
        )
    scale = control.get("ecn_threshold_scale", 1.0)
    smallest = min(low for _, low, _ in ECN_THRESHOLDS_KB)
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not isfinite(scale)
        or scale <= 0
        or round(smallest * scale) < 1
    ):
        raise ValueError(
            "network.congestion_control.ecn_threshold_scale must be a positive "
            "number that keeps every threshold at least 1 KB"
        )
    return CongestionControl(
        mode=mode,
        rate_ai_fraction=rate_ai,
        rate_hai_fraction=rate_hai,
        min_rate_fraction=_fraction(
            control.get("min_rate_fraction", DEFAULT_MIN_RATE_FRACTION),
            "network.congestion_control.min_rate_fraction",
        ),
        ecn_threshold_scale=float(scale),
    )


# A parameter's value: a number, a switch, a name or a list of thresholds.
ParameterValue = int | float | bool | str | tuple[float, ...]


@dataclass(frozen=True)
class BalancingParameter:
    """One parameter of a load-balancing mode or a path selector: its profile
    key, its simulator key, the default its source gives, and the parser that
    validates a value; and, where the simulator reads it otherwise than
    _setting writes it, how it is written."""

    name: str
    key: str
    default: ParameterValue
    parse: Callable[[Any, str], ParameterValue]
    write: Callable[[ParameterValue], str] | None = None


def _setting(value: ParameterValue) -> str:
    """A parameter's value as the simulator reads it."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, tuple):
        return " ".join(str(item) for item in value)
    return str(value)


def _reps_buffer_size(value: Any, field: str) -> int:
    # REPS's Table 1 holds the buffer's position and its count in a byte each.
    if _positive_int(value, field) > 255:
        raise ValueError(f"{field} must be at most 255")
    return value


def _ev_set_size(value: Any, field: str) -> int:
    # The values of a set are 0 to its size less one, in 16 bits.
    if _positive_int(value, field) > 65536:
        raise ValueError(f"{field} must be at most 65536")
    return value


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be a positive number")
    return float(value)


# UEC 1.0.3 section 3.6.16.3's typical space.
_UE_EV_SET_SIZE = BalancingParameter("ev_set_size", "UE_EV_SET_SIZE", 256, _ev_set_size)
SELECTOR_PARAMETERS: dict[str, tuple[BalancingParameter, ...]] = {
    "ops": (),
    # REPS section 3.1 and its artifact's exit_freeze_after.
    "reps": (
        BalancingParameter("buffer_size", "REPS_BUFFER_SIZE", 8, _reps_buffer_size),
        BalancingParameter(
            "freezing_timeout_ns", "REPS_FREEZING_TIMEOUT_NS", 10_000_000, _positive_int
        ),
    ),
    "ue_oblivious": (_UE_EV_SET_SIZE,),
    # UEC 1.0.3 section 3.6.16.4's default saturation.
    "ue_aware": (
        _UE_EV_SET_SIZE,
        BalancingParameter(
            "saturation_fraction", "UE_SATURATION_FRACTION", 0.5, _fraction
        ),
    ),
    # A set within OCP MRC 1.0 section 11.2.1's one to two windows; a SKIP
    # value out of use for a round trip, as UEC 1.0.3 section 3.6.16.4 keeps a
    # marked one; an assumed-bad value probed once per retransmission timeout.
    "mrc": (
        BalancingParameter("ev_set_size", "MRC_EV_SET_SIZE", 128, _ev_set_size),
        BalancingParameter(
            "skip_base_rtts", "MRC_SKIP_BASE_RTTS", 1.0, _positive_number
        ),
        BalancingParameter(
            "probe_timeouts", "MRC_PROBE_TIMEOUTS", 1.0, _positive_number
        ),
    ),
}


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    probability = float(value)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a number in [0, 1]")
    return probability


def _nonnegative_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field} must be a nonnegative number")
    return float(value)


def _increasing_thresholds(
    parse: Callable[[Any, str], float],
) -> Callable[[Any, str], tuple[float, ...]]:
    """Three thresholds, each parsed as parse does, strictly increasing."""

    def thresholds(value: Any, field: str) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{field} must be a list of three thresholds")
        parsed = tuple(parse(item, field) for item in value)
        if not parsed[0] < parsed[1] < parsed[2]:
            raise ValueError(f"{field} must increase")
        return parsed

    return thresholds


def _flag(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _grade(value: Any, field: str) -> int:
    # A report grades a spine in two bits.
    if _nonnegative_int(value, field) > 3:
        raise ValueError(f"{field} must be an integer in [0, 3]")
    return value


def _candidate_draw(value: Any, field: str) -> str:
    if value not in ("proportional", "uniform"):
        raise ValueError(f"{field} must be 'proportional' or 'uniform'")
    return value


def _absence_fraction(value: Any, field: str) -> float:
    # Zero holds no spine down for its absence; at one every spine would be.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value < 1
    ):
        raise ValueError(f"{field} must be a number in [0, 1)")
    return float(value)


def _grade_reference(value: Any, field: str) -> str:
    if value not in ("absolute", "median"):
        raise ValueError(f"{field} must be 'absolute' or 'median'")
    return value


def _grade_scope(value: Any, field: str) -> str:
    if value not in ("per_receiver", "per_connection"):
        raise ValueError(f"{field} must be 'per_receiver' or 'per_connection'")
    return value


def _grade_cost_source(value: Any, field: str) -> str:
    if value not in ("estimate", "switch_queue"):
        raise ValueError(f"{field} must be 'estimate' or 'switch_queue'")
    return value


def _held_exploration(value: Any, field: str) -> str:
    if value not in ("epsilon", "sentinel", "none"):
        raise ValueError(f"{field} must be 'epsilon', 'sentinel' or 'none'")
    return value


def _hold_scope(value: Any, field: str) -> str:
    if value not in ("host", "leaf", "auto", "pair"):
        raise ValueError(f"{field} must be 'host', 'leaf', 'auto' or 'pair'")
    return value


def _hold_rule(value: Any, field: str) -> str:
    if value not in ("count", "llr"):
        raise ValueError(f"{field} must be 'count' or 'llr'")
    return value


def _hold_release(value: Any, field: str) -> str:
    if value not in ("sentinels", "trickle"):
        raise ValueError(f"{field} must be 'sentinels' or 'trickle'")
    return value


def _sentinel_mode(value: Any, field: str) -> str:
    if value not in ("data", "duplicate"):
        raise ValueError(f"{field} must be 'data' or 'duplicate'")
    return value


def _number_in(
    low: float, high: float, low_open: bool, high_open: bool
) -> Callable[[Any, str], float]:
    """A number within low and high, each end open or closed."""
    interval = f"{'(' if low_open else '['}{low:g}, {high:g}{')' if high_open else ']'}"

    def parse(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number in {interval}")
        number = float(value)
        if (
            not isfinite(number)
            or number < low
            or number > high
            or (low_open and number == low)
            or (high_open and number == high)
        ):
            raise ValueError(f"{field} must be a number in {interval}")
        return number

    return parse


def _on_off(value: ParameterValue) -> str:
    """A switch as the backend's on and off keys read it."""
    return "on" if value else "off"


def _hold_auto_leaves(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{field} must be an integer of at least 2")
    return value


def _trim_scoring(value: Any, field: str) -> str:
    if value not in ("mark", "ignore"):
        raise ValueError(f"{field} must be 'mark' or 'ignore'")
    return value


# ack_redirect, modelled on MPTR's ACK redirection (Wang et al., IEEE TII
# 2026): the weight of the delay term of an arrival's score, 0.5 as MPTR's
# default; a redirect once an arrival scores 10 percent below the best, MPTR's
# default; spines scored per connection, from its own arrivals; the best
# spine kept for two base RTTs, within MPTR's cycle of one to three round
# trips; and a header trimmed before the last hop scored as a marked arrival
# on its spine. The last two only per connection: per receiver, the host's
# grader scores the spines once per report interval and counts trims.
ACK_REDIRECT_PARAMETERS: tuple[BalancingParameter, ...] = (
    BalancingParameter("omega", "ACK_REDIRECT_OMEGA", 0.5, _probability),
    BalancingParameter("flag", "ACK_REDIRECT_FLAG", 0.1, _nonnegative_number),
    BalancingParameter("scope", "ACK_REDIRECT_SCOPE", "per_connection", _grade_scope),
    BalancingParameter(
        "validity_base_rtts", "ACK_REDIRECT_VALIDITY_BASE_RTTS", 2.0, _positive_number
    ),
    BalancingParameter(
        "trim_scoring", "ACK_REDIRECT_TRIM_SCORING", "mark", _trim_scoring
    ),
)


def _window_penalty(value: Any, field: str) -> int:
    # UEC 1.0.3 section 3.6.13.2's Rcv_Cwnd_Pend is seven bits.
    if _nonnegative_int(value, field) > 127:
        raise ValueError(f"{field} must be at most 127")
    return value


# spray_policy's receiver, then its sender. One report per loaded round trip,
# about two base RTTs; DCTCP's gain of 1/16 on the marked and the trimmed
# fraction, which an interval of 32 packets replaces; CUSUM slack of half a
# grade's width and a threshold of two; grades in four equal bands of the
# fractions' and, with one-way delay, the delay's excess over the median
# spine's, the delay's up to NSCC's target_qdelay of 0.75 base RTTs; a spine
# held for arriving below an eighth of the median spine's arrivals over a
# window whose median reaches 16, and released at half its floor's arrivals;
# gamma 0.25 and epsilon 0.02, and one candidate, as the design gives them;
# NSCC's Rcv_Cwnd_Pend example of 64.
POLICY_PARAMETERS: tuple[BalancingParameter, ...] = (
    BalancingParameter(
        "report_interval_base_rtts",
        "SPRAY_REPORT_INTERVAL_BASE_RTTS",
        2.0,
        _positive_number,
    ),
    BalancingParameter("estimator_gain", "SPRAY_ESTIMATOR_GAIN", 0.0625, _fraction),
    BalancingParameter(
        "estimator_interval_samples",
        "SPRAY_ESTIMATOR_INTERVAL_SAMPLES",
        32,
        _nonnegative_int,
    ),
    BalancingParameter(
        "fraction_cusum_slack",
        "SPRAY_FRACTION_CUSUM_SLACK",
        0.125,
        _nonnegative_number,
    ),
    BalancingParameter(
        "fraction_cusum_threshold",
        "SPRAY_FRACTION_CUSUM_THRESHOLD",
        0.5,
        _positive_number,
    ),
    BalancingParameter(
        "congestion_thresholds",
        "SPRAY_CONGESTION_THRESHOLDS",
        (0.25, 0.5, 0.75),
        _increasing_thresholds(_fraction),
    ),
    # Absent, the simulator grades the trimmed fraction through
    # congestion_thresholds.
    BalancingParameter(
        "trim_thresholds",
        "SPRAY_TRIM_THRESHOLDS",
        (0.25, 0.5, 0.75),
        _increasing_thresholds(_fraction),
    ),
    # Absent, 0, every grade as the thresholds give it. With it, the least
    # grade the receiver gives a spine for its marked and trimmed fractions
    # and its one-way delay; the loss grade and a held spine still grade 0.
    BalancingParameter(
        "congestion_grade_floor", "SPRAY_CONGESTION_GRADE_FLOOR", 0, _grade
    ),
    # Absent, the sender counts no loss. With them, the sender counts per
    # spine the sends drawn from its scores and the sends declared lost, and
    # grades each spine at no more than its lost fraction earns.
    BalancingParameter(
        "loss_thresholds",
        "SPRAY_LOSS_THRESHOLDS",
        (0.25, 0.5, 0.75),
        _increasing_thresholds(_fraction),
    ),
    # The host-wide pool's loss interval, 0 for the report interval; read only
    # with loss_thresholds.
    BalancingParameter(
        "loss_interval_ns", "SPRAY_LOSS_INTERVAL_NS", 0, _nonnegative_int
    ),
    # The clean sentinels a spine whose lost fraction graded 0 must deliver,
    # consecutively, before it leaves grade 0; 0 releases it at the first
    # interval with no loss. Read only with loss_thresholds.
    BalancingParameter(
        "loss_hold_clean_sentinels",
        "SPRAY_LOSS_HOLD_CLEAN_SENTINELS",
        0,
        _nonnegative_int,
    ),
    BalancingParameter(
        "absence_fraction_of_median",
        "SPRAY_ABSENCE_FRACTION_OF_MEDIAN",
        0.125,
        _absence_fraction,
    ),
    BalancingParameter(
        "absence_minimum_median", "SPRAY_ABSENCE_MINIMUM_MEDIAN", 16, _positive_int
    ),
    BalancingParameter(
        "release_fraction_of_floor",
        "SPRAY_RELEASE_FRACTION_OF_FLOOR",
        0.5,
        _nonnegative_number,
    ),
    BalancingParameter(
        "grade_reference", "SPRAY_GRADE_REFERENCE", "median", _grade_reference
    ),
    BalancingParameter(
        "grade_scope", "SPRAY_GRADE_SCOPE", "per_receiver", _grade_scope
    ),
    # Simulation only: switch_queue grades the paths from the switches' own
    # queue, marking, rate and loss state, which no receiver can read.
    BalancingParameter(
        "grade_cost_source", "SPRAY_GRADE_COST_SOURCE", "estimate", _grade_cost_source
    ),
    BalancingParameter("one_way_delay", "SPRAY_ONE_WAY_DELAY", False, _flag),
    BalancingParameter(
        "delay_cusum_slack_base_rtts",
        "SPRAY_DELAY_CUSUM_SLACK_BASE_RTTS",
        0.125,
        _nonnegative_number,
    ),
    BalancingParameter(
        "delay_cusum_threshold_base_rtts",
        "SPRAY_DELAY_CUSUM_THRESHOLD_BASE_RTTS",
        0.5,
        _positive_number,
    ),
    BalancingParameter(
        "delay_thresholds_base_rtts",
        "SPRAY_DELAY_THRESHOLDS_BASE_RTTS",
        (0.25, 0.5, 0.75),
        _increasing_thresholds(_positive_number),
    ),
    BalancingParameter("gamma", "SPRAY_GAMMA", 0.25, _fraction),
    BalancingParameter("epsilon", "SPRAY_EPSILON", 0.02, _probability),
    # How a held spine, one whose last grade was 0, is explored: epsilon
    # gives it its share of the epsilon draw, sentinel one data packet per
    # sentinel interval and no share of the draw, none nothing.
    BalancingParameter(
        "held_exploration", "SPRAY_HELD_EXPLORATION", "epsilon", _held_exploration
    ),
    # Read only with held_exploration sentinel.
    BalancingParameter(
        "held_sentinel_interval_ns",
        "SPRAY_HELD_SENTINEL_INTERVAL_NS",
        0,
        _nonnegative_int,
    ),
    BalancingParameter("candidates", "SPRAY_CANDIDATES", 1, _positive_int),
    BalancingParameter(
        "candidate_draw", "SPRAY_CANDIDATE_DRAW", "proportional", _candidate_draw
    ),
    BalancingParameter(
        "edge_window_penalty", "SPRAY_EDGE_WINDOW_PENALTY", 64, _window_penalty
    ),
    BalancingParameter("edge_freeze", "SPRAY_EDGE_FREEZE", True, _flag),
    BalancingParameter(
        "sender_pool_weight", "SPRAY_SENDER_POOL_WEIGHT", 0.0, _probability
    ),
    # Absent, the host-wide pool decays once per report it takes; with it,
    # once per interval, depositing the mean of the grades taken in it. Read
    # only with sender_pool_weight above 0.
    BalancingParameter(
        "pool_decay_interval_ns", "SPRAY_POOL_DECAY_INTERVAL_NS", 0, _nonnegative_int
    ),
    # With it, a spine the host-wide pool holds, by its grade or its loss
    # hold, gets no share in any of the host's draws, and its sentinels are
    # sent once per host per sentinel interval. Read only with
    # sender_pool_weight above 0.
    BalancingParameter(
        "pool_hold_propagates", "SPRAY_POOL_HOLD_PROPAGATES", False, _flag
    ),
)
# The parameters written, and recorded in the manifest, only when the profile
# sets them; the simulator's defaults are the defaults above.
WRITTEN_WHEN_SET_PARAMETERS = {
    "edge_freeze",
    "sender_pool_weight",
    "trim_thresholds",
    "congestion_grade_floor",
    "loss_thresholds",
    "loss_interval_ns",
    "loss_hold_clean_sentinels",
    "held_exploration",
    "held_sentinel_interval_ns",
    "pool_decay_interval_ns",
    "pool_hold_propagates",
}
# The parameters only one-way delay reads.
ONE_WAY_DELAY_PARAMETERS = {
    "delay_cusum_slack_base_rtts",
    "delay_cusum_threshold_base_rtts",
    "delay_thresholds_base_rtts",
}


# The parameters of spray_policy's receiver-side grader, which ack_redirect
# also takes at scope per_receiver, where the host's grader scores the
# spines.
GRADER_PARAMETERS = tuple(
    parameter
    for parameter in POLICY_PARAMETERS
    if parameter.name
    in {
        "report_interval_base_rtts",
        "estimator_gain",
        "estimator_interval_samples",
        "fraction_cusum_slack",
        "fraction_cusum_threshold",
        "absence_fraction_of_median",
        "absence_minimum_median",
        "release_fraction_of_floor",
        "one_way_delay",
        "delay_cusum_slack_base_rtts",
        "delay_cusum_threshold_base_rtts",
    }
)
# The parameters only ack_redirect's per_connection scope reads.
ACK_REDIRECT_CONNECTION_PARAMETERS = {"validity_base_rtts", "trim_scoring"}


def _policy_parameter(name: str) -> BalancingParameter:
    return next(p for p in POLICY_PARAMETERS if p.name == name)


def _tail_recovery_delay_ns(value: Any, field: str) -> int:
    delay = _nonnegative_int(value, field)
    if delay >= TAIL_RECOVERY_DELAY_DEFAULT_SENTINEL:
        raise ValueError(
            f"{field} must be below {TAIL_RECOVERY_DELAY_DEFAULT_SENTINEL}"
        )
    return delay


# The keys spray_uniform reads under every selector and under none, each
# written only when set and absent taking the backend's default: path_seq,
# every data packet's count of the packets its queue pair sent before it up
# its requested spine; tail_recovery, which requires path_seq true, a
# message's closing packets' per-spine sent counts and the receiver's repair
# requests from them; tail_recovery_delay_ns, read only with tail_recovery,
# the delay before such a request, absent the fabric's largest base RTT;
# tail_loss_rtts, read only with tail_recovery, the base RTTs a send a repair
# request declares lost waits before its loss is reported to the path
# selector, a trim of it or an acknowledgement of it along its path in that
# time cancelling the report, its repair chosen again off its path where the
# selector offers another, absent 1.
PATH_SEQ_KEY = BalancingParameter("path_seq", "SPRAY_PATH_SEQ", False, _flag, _on_off)
TAIL_RECOVERY_KEYS: tuple[BalancingParameter, ...] = (
    BalancingParameter(
        "tail_recovery", "SPRAY_TAIL_RECOVERY", False, _flag, _on_off
    ),
    BalancingParameter(
        "tail_recovery_delay_ns",
        "SPRAY_TAIL_RECOVERY_DELAY_NS",
        0,
        _tail_recovery_delay_ns,
    ),
    BalancingParameter(
        "tail_loss_rtts", "SPRAY_TAIL_LOSS_RTTS", 1.0, _nonnegative_number
    ),
)
# ack_loss_rtts, after tail recovery's: the base RTTs after a later
# cumulative acknowledgement covers a send whose own acknowledgement never
# arrived at which that send is declared an acknowledgement loss on its spine,
# to the path selector and the loss hold, with no repair; absent, none is.
ACK_LOSS_KEY = BalancingParameter(
    "ack_loss_rtts", "SPRAY_ACK_LOSS_RTTS", 0.0, _positive_number
)
SPRAY_UNIFORM_KEYS: tuple[BalancingParameter, ...] = (
    PATH_SEQ_KEY,
    *TAIL_RECOVERY_KEYS,
    ACK_LOSS_KEY,
)
# receiver_tail_missing, after ack_loss_rtts, read only by the loss hold
# selectors with receiver_hold and tail_recovery true: whether the receiver's
# hold also counts missing the sends a tail repair request names that are
# still missing tail_loss_rtts base RTTs after the request, which holds a
# spine on which nothing arrives; absent, false.
RECEIVER_TAIL_MISSING_KEY = BalancingParameter(
    "receiver_tail_missing", "SPRAY_RECEIVER_TAIL_MISSING", False, _flag, _on_off
)


# The loss hold of reps_hold, reps_flow_hold and mrc_hold, after the
# selector's own parameters: the loss thresholds, the loss interval and the
# held exploration the backend requires, and the clean sentinels it requires
# under hold_release sentinels; then the keys written only when set.
# hold_scope, read by reps_hold and mrc_hold, keys the hold by host, by
# destination leaf, per spine by either, auto, the backend's default, or by
# destination host, pair;
# hold_auto_leaves, read only under auto or with receiver_hold, is the
# destination leaves (the sender leaves at the receiver) with a loss on a
# spine from which the hold covers every one. hold_rule, the hold test's
# probabilities, hold_release with hold_trickle, sentinel_mode,
# held_loss_rtts, path_seq and receiver_hold with receiver_hold_clean take the
# backend's defaults when absent; the backend reads loss_p0, loss_p1 and
# hold_alpha only under hold_rule llr or hold_release trickle, hold_beta only
# under llr.
LOSS_HOLD_KEYS: tuple[BalancingParameter, ...] = (
    _policy_parameter("loss_thresholds"),
    BalancingParameter("loss_interval_ns", "SPRAY_LOSS_INTERVAL_NS", 0, _positive_int),
    BalancingParameter(
        "loss_hold_clean_sentinels",
        "SPRAY_LOSS_HOLD_CLEAN_SENTINELS",
        0,
        _positive_int,
    ),
    _policy_parameter("held_exploration"),
    _policy_parameter("held_sentinel_interval_ns"),
    BalancingParameter("hold_scope", "SPRAY_HOLD_SCOPE", "auto", _hold_scope),
    BalancingParameter(
        "hold_auto_leaves", "SPRAY_HOLD_AUTO_LEAVES", 2, _hold_auto_leaves
    ),
    BalancingParameter("hold_rule", "SPRAY_HOLD_RULE", "count", _hold_rule),
    BalancingParameter("loss_p0", "SPRAY_LOSS_P0", 0.0, _number_in(0, 1, False, True)),
    BalancingParameter("loss_p1", "SPRAY_LOSS_P1", 0.001, _number_in(0, 1, True, True)),
    BalancingParameter(
        "hold_alpha", "SPRAY_HOLD_ALPHA", 0.001, _number_in(0, 0.5, True, True)
    ),
    BalancingParameter(
        "hold_beta", "SPRAY_HOLD_BETA", 0.001, _number_in(0, 0.5, True, True)
    ),
    BalancingParameter(
        "hold_release", "SPRAY_HOLD_RELEASE", "sentinels", _hold_release
    ),
    BalancingParameter(
        "hold_trickle", "SPRAY_HOLD_TRICKLE", 0.01, _number_in(0, 0.5, True, False)
    ),
    BalancingParameter("sentinel_mode", "SPRAY_SENTINEL_MODE", "data", _sentinel_mode),
    BalancingParameter(
        "held_loss_rtts", "SPRAY_HELD_LOSS_RTTS", 2, _nonnegative_int
    ),
    PATH_SEQ_KEY,
    BalancingParameter("receiver_hold", "SPRAY_RECEIVER_HOLD", False, _flag, _on_off),
    BalancingParameter(
        "receiver_hold_clean", "SPRAY_RECEIVER_HOLD_CLEAN", 1024, _positive_int
    ),
)
# Each loss hold selector's parameters: REPS's, or MRC's but its set, which
# is the spines, defaults filled in; then the loss hold's, path_seq among
# them, tail recovery's, ack_loss_rtts and receiver_tail_missing.
LOSS_HOLD_PARAMETERS: dict[str, tuple[BalancingParameter, ...]] = {
    "reps_hold": (
        *SELECTOR_PARAMETERS["reps"],
        *LOSS_HOLD_KEYS,
        *TAIL_RECOVERY_KEYS,
        ACK_LOSS_KEY,
        RECEIVER_TAIL_MISSING_KEY,
    ),
    "reps_flow_hold": (
        *SELECTOR_PARAMETERS["reps"],
        *LOSS_HOLD_KEYS,
        *TAIL_RECOVERY_KEYS,
        ACK_LOSS_KEY,
        RECEIVER_TAIL_MISSING_KEY,
    ),
    "mrc_hold": (
        *(
            parameter
            for parameter in SELECTOR_PARAMETERS["mrc"]
            if parameter.name != "ev_set_size"
        ),
        *LOSS_HOLD_KEYS,
        *TAIL_RECOVERY_KEYS,
        ACK_LOSS_KEY,
        RECEIVER_TAIL_MISSING_KEY,
    ),
}
LOSS_HOLD_REQUIRED_PARAMETERS = (
    "loss_thresholds",
    "loss_interval_ns",
    "held_exploration",
)
# Under hold_release trickle the trickle sends release a hold, and the
# backend refuses the sentinels' keys.
SENTINEL_RELEASE_PARAMETERS = (
    "loss_hold_clean_sentinels",
    "held_sentinel_interval_ns",
)
LOSS_HOLD_WRITTEN_WHEN_SET_PARAMETERS = {
    "loss_hold_clean_sentinels",
    "held_sentinel_interval_ns",
    "hold_scope",
    "hold_auto_leaves",
    "hold_rule",
    "loss_p0",
    "loss_p1",
    "hold_alpha",
    "hold_beta",
    "hold_release",
    "hold_trickle",
    "sentinel_mode",
    "held_loss_rtts",
    "path_seq",
    "receiver_hold",
    "receiver_hold_clean",
    "tail_recovery",
    "tail_recovery_delay_ns",
    "tail_loss_rtts",
    "ack_loss_rtts",
    "receiver_tail_missing",
}
# reps_oracle: REPS's parameters, defaults filled in, and the cost source the
# backend requires, switch_queue, which builds the switches' state it reads.
REPS_ORACLE_PARAMETERS: tuple[BalancingParameter, ...] = (
    *SELECTOR_PARAMETERS["reps"],
    _policy_parameter("grade_cost_source"),
)


# reps_spine: REPS's parameters, defaults filled in; no loss hold, no
# sentinel, and none of the loss hold's keys.
REPS_SPINE_PARAMETERS: tuple[BalancingParameter, ...] = SELECTOR_PARAMETERS["reps"]


def _parameters_of(mode: str, selector: str) -> tuple[BalancingParameter, ...]:
    if mode == "ev_hash":
        return SELECTOR_PARAMETERS[selector]
    if mode == "spray_policy":
        return POLICY_PARAMETERS
    if selector == "ack_redirect":
        return ACK_REDIRECT_PARAMETERS + GRADER_PARAMETERS + SPRAY_UNIFORM_KEYS
    if selector in LOSS_HOLD_SELECTORS:
        return LOSS_HOLD_PARAMETERS[selector]
    if selector == "reps_oracle":
        return REPS_ORACLE_PARAMETERS + SPRAY_UNIFORM_KEYS
    if selector == "reps_spine":
        return REPS_SPINE_PARAMETERS + SPRAY_UNIFORM_KEYS
    if mode == "spray_uniform":
        return SPRAY_UNIFORM_KEYS
    return ()


def _load_load_balancing(
    document: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """Parse the optional knob, defaulting to the flow-pinned ECMP hash."""
    if "load_balancing" not in document:
        return LoadBalancing(mode="ecmp")
    balancing = document["load_balancing"]
    if not isinstance(balancing, dict) or "mode" not in balancing:
        raise ValueError("network.load_balancing must contain 'mode'")
    mode = balancing["mode"]
    if not isinstance(mode, str) or mode not in LOAD_BALANCING_MODES:
        raise ValueError(
            "network.load_balancing.mode must be one of "
            f"{sorted(LOAD_BALANCING_MODES)}"
        )
    if mode == "spray_policy":
        return _load_spray_policy(balancing, recovery, trimming)
    if mode == "spray_uniform" and balancing.get("selector") in LOSS_HOLD_SELECTORS:
        return _load_loss_hold_selector(balancing, recovery, trimming)
    if mode == "spray_uniform" and balancing.get("selector") == "reps_oracle":
        return _load_reps_oracle(balancing, recovery, trimming)
    if mode == "spray_uniform" and balancing.get("selector") == "reps_spine":
        return _load_reps_spine(balancing, recovery, trimming)
    if mode == "spray_uniform" and balancing.get("selector") in SPINE_SELECTORS:
        return _load_spine_selector(balancing, recovery, trimming)
    if mode == "spray_uniform" and "selector" not in balancing:
        return _load_uniform_draw(balancing, recovery, trimming)
    if mode != "ev_hash" and set(balancing) != {"mode"}:
        raise ValueError(
            f"network.load_balancing.mode '{mode}' takes no other key; a selector "
            "and its parameters require mode 'ev_hash', or mode 'spray_uniform' "
            f"for a selector in {list(SPINE_SELECTORS)}"
        )
    selector = balancing.get("selector", "ops")
    if selector not in PATH_SELECTORS:
        raise ValueError(
            f"network.load_balancing.selector must be one of {list(PATH_SELECTORS)}"
        )
    parameters = SELECTOR_PARAMETERS[selector]
    unknown = set(balancing) - {"mode", "selector"} - {p.name for p in parameters}
    if unknown:
        raise ValueError(
            f"network.load_balancing.selector '{selector}' does not take "
            f"{sorted(unknown)}"
        )
    _require_reorder_tolerance(mode, recovery, trimming)
    return LoadBalancing(
        mode=mode,
        selector=selector,
        parameters=_parsed(balancing, parameters),
    )


def _load_spine_selector(
    balancing: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """spray_uniform with a selector that chooses each spine, and its
    parameters."""
    selector = balancing["selector"]
    scope = _grade_scope(
        balancing.get("scope", "per_connection"), "network.load_balancing.scope"
    )
    one_way_delay = _flag(
        balancing.get("one_way_delay", False), "network.load_balancing.one_way_delay"
    )
    if scope == "per_connection":
        parameters = ACK_REDIRECT_PARAMETERS
    else:
        parameters = tuple(
            parameter
            for parameter in ACK_REDIRECT_PARAMETERS + GRADER_PARAMETERS
            if parameter.name not in ACK_REDIRECT_CONNECTION_PARAMETERS
            and (one_way_delay or parameter.name not in ONE_WAY_DELAY_PARAMETERS)
        )
    unknown = (
        set(balancing)
        - {"mode", "selector"}
        - {p.name for p in parameters + SPRAY_UNIFORM_KEYS}
    )
    if unknown:
        raise ValueError(
            f"network.load_balancing.selector '{selector}' with scope '{scope}' "
            f"does not take {sorted(unknown)}"
        )
    _require_reorder_tolerance("spray_uniform", recovery, trimming)
    return LoadBalancing(
        mode="spray_uniform",
        selector=selector,
        parameters=_parsed(balancing, parameters) + _spray_uniform_keys(balancing),
    )


def _load_uniform_draw(
    balancing: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """spray_uniform without a selector, the uniform draw, which reads only
    the keys spray_uniform reads under every selector."""
    unknown = set(balancing) - {"mode"} - {p.name for p in SPRAY_UNIFORM_KEYS}
    if unknown:
        raise ValueError(
            "network.load_balancing.mode 'spray_uniform' without a selector "
            "takes no other key than path_seq, tail_recovery, "
            "tail_recovery_delay_ns, tail_loss_rtts and ack_loss_rtts; a selector "
            "and its parameters require mode 'ev_hash', or mode 'spray_uniform' "
            f"for a selector in {list(SPINE_SELECTORS)}"
        )
    _require_reorder_tolerance("spray_uniform", recovery, trimming)
    return LoadBalancing(
        mode="spray_uniform", parameters=_spray_uniform_keys(balancing)
    )


def _spray_uniform_keys(
    balancing: dict[str, Any],
) -> tuple[tuple[str, ParameterValue], ...]:
    """path_seq, tail_recovery, tail_recovery_delay_ns, tail_loss_rtts and
    ack_loss_rtts as given, each only when set, refused where the backend
    refuses them."""
    parsed = _parsed(
        balancing,
        tuple(parameter for parameter in SPRAY_UNIFORM_KEYS if parameter.name in balancing),
    )
    _require_tail_recovery(balancing, dict(parsed))
    return parsed


def _require_tail_recovery(
    balancing: dict[str, Any], values: dict[str, ParameterValue]
) -> None:
    """tail_recovery requires path_seq true, and tail_recovery_delay_ns and
    tail_loss_rtts are read only with tail_recovery true, as the backend reads
    them."""
    tail_recovery = values.get("tail_recovery") is True
    if tail_recovery and values.get("path_seq") is not True:
        raise ValueError("network.load_balancing.tail_recovery requires path_seq true")
    for name in ("tail_recovery_delay_ns", "tail_loss_rtts"):
        if name in balancing and not tail_recovery:
            raise ValueError(
                f"network.load_balancing.{name} requires tail_recovery true"
            )


def _load_loss_hold_selector(
    balancing: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """spray_uniform with reps_hold, reps_flow_hold or mrc_hold, which
    require the loss hold's keys and held_exploration sentinel, and read the
    keys of its rule, its release, its exploration sends and the receiver's
    hold where the backend reads them."""
    selector = balancing["selector"]
    scoped = selector in HOLD_SCOPE_SELECTORS
    # Without hold_scope, hold_auto_leaves is read only by the receiver's
    # hold.
    parameters = tuple(
        parameter
        for parameter in LOSS_HOLD_PARAMETERS[selector]
        if scoped
        or (
            parameter.name != "hold_scope"
            and (
                parameter.name != "hold_auto_leaves"
                or balancing.get("receiver_hold") is True
            )
        )
    )
    unknown = set(balancing) - {"mode", "selector"} - {p.name for p in parameters}
    if unknown:
        raise ValueError(
            f"network.load_balancing.selector '{selector}' does not take "
            f"{sorted(unknown)}"
        )
    trickle = (
        _hold_release(
            balancing.get("hold_release", "sentinels"),
            "network.load_balancing.hold_release",
        )
        == "trickle"
    )
    required = LOSS_HOLD_REQUIRED_PARAMETERS + (
        () if trickle else ("loss_hold_clean_sentinels",)
    )
    if any(name not in balancing for name in required):
        raise ValueError(
            f"network.load_balancing.selector '{selector}' requires "
            + (
                "loss_thresholds, loss_interval_ns and held_exploration "
                "'sentinel' under hold_release 'trickle'"
                if trickle
                else "loss_thresholds, loss_interval_ns, "
                "loss_hold_clean_sentinels and held_exploration 'sentinel'"
            )
        )
    for name in SENTINEL_RELEASE_PARAMETERS:
        if trickle and name in balancing:
            raise ValueError(
                f"network.load_balancing.{name} is read only under "
                "hold_release 'sentinels'"
            )
    parsed = _parsed(
        balancing,
        tuple(
            parameter
            for parameter in parameters
            if parameter.name not in LOSS_HOLD_WRITTEN_WHEN_SET_PARAMETERS
            or parameter.name in balancing
        ),
    )
    values = dict(parsed)
    # A held spine is left out of every choice, so only its exploration
    # sends can end the hold.
    if values["held_exploration"] != "sentinel":
        raise ValueError(
            f"network.load_balancing.selector '{selector}' requires "
            "held_exploration 'sentinel'"
        )
    _require_hold_test(balancing, values, trickle)
    receiver_hold = values.get("receiver_hold") is True
    if receiver_hold and values.get("path_seq") is not True:
        raise ValueError("network.load_balancing.receiver_hold requires path_seq true")
    _require_tail_recovery(balancing, values)
    if "receiver_hold_clean" in balancing and not receiver_hold:
        raise ValueError(
            "network.load_balancing.receiver_hold_clean requires receiver_hold true"
        )
    if "receiver_tail_missing" in balancing and not (
        receiver_hold and values.get("tail_recovery") is True
    ):
        raise ValueError(
            "network.load_balancing.receiver_tail_missing requires receiver_hold "
            "true and tail_recovery true"
        )
    # As the backend reads it: under auto, given or by default, or by the
    # receiver's hold.
    if (
        "hold_auto_leaves" in balancing
        and values.get("hold_scope", "auto") != "auto"
        and not receiver_hold
    ):
        raise ValueError(
            "network.load_balancing.hold_auto_leaves requires hold_scope 'auto' "
            "or receiver_hold true"
        )
    _require_reorder_tolerance("spray_uniform", recovery, trimming)
    return LoadBalancing(mode="spray_uniform", selector=selector, parameters=parsed)


def _require_hold_test(
    balancing: dict[str, Any], values: dict[str, ParameterValue], trickle: bool
) -> None:
    """The hold test's keys only where the backend reads them: hold_beta under
    hold_rule llr; loss_p0, loss_p1 and hold_alpha under llr or hold_release
    trickle, with loss_p0 below loss_p1; hold_trickle under trickle."""
    llr = values.get("hold_rule", "count") == "llr"
    if "hold_beta" in balancing and not llr:
        raise ValueError("network.load_balancing.hold_beta requires hold_rule 'llr'")
    for name in ("loss_p0", "loss_p1", "hold_alpha"):
        if name in balancing and not (llr or trickle):
            raise ValueError(
                f"network.load_balancing.{name} requires hold_rule 'llr' or "
                "hold_release 'trickle'"
            )
    if "hold_trickle" in balancing and not trickle:
        raise ValueError(
            "network.load_balancing.hold_trickle requires hold_release 'trickle'"
        )
    if (llr or trickle) and not float(values.get("loss_p0", 0.0)) < float(
        values.get("loss_p1", 0.001)
    ):
        raise ValueError(
            "network.load_balancing.loss_p0 must be below "
            "network.load_balancing.loss_p1"
        )


def _load_reps_oracle(
    balancing: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """spray_uniform with reps_oracle, which reads the switches' state and so
    requires grade_cost_source switch_queue, as the backend does."""
    names = {parameter.name for parameter in REPS_ORACLE_PARAMETERS + SPRAY_UNIFORM_KEYS}
    unknown = set(balancing) - {"mode", "selector"} - names
    if unknown:
        raise ValueError(
            "network.load_balancing.selector 'reps_oracle' does not take "
            f"{sorted(unknown)}"
        )
    parsed = _parsed(balancing, REPS_ORACLE_PARAMETERS)
    if "grade_cost_source" not in balancing or (
        dict(parsed)["grade_cost_source"] != "switch_queue"
    ):
        raise ValueError(
            "network.load_balancing.selector 'reps_oracle' requires "
            "grade_cost_source 'switch_queue'"
        )
    _require_reorder_tolerance("spray_uniform", recovery, trimming)
    return LoadBalancing(
        mode="spray_uniform",
        selector="reps_oracle",
        parameters=parsed + _spray_uniform_keys(balancing),
    )


def _load_reps_spine(
    balancing: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """spray_uniform with reps_spine, which reads REPS's parameters and the
    keys spray_uniform reads under every selector, and refuses the loss
    hold's keys and every spray_policy key, as the backend does."""
    names = {parameter.name for parameter in REPS_SPINE_PARAMETERS + SPRAY_UNIFORM_KEYS}
    unknown = set(balancing) - {"mode", "selector"} - names
    if unknown:
        raise ValueError(
            "network.load_balancing.selector 'reps_spine' does not take "
            f"{sorted(unknown)}"
        )
    _require_reorder_tolerance("spray_uniform", recovery, trimming)
    return LoadBalancing(
        mode="spray_uniform",
        selector="reps_spine",
        parameters=_parsed(balancing, REPS_SPINE_PARAMETERS)
        + _spray_uniform_keys(balancing),
    )


def _load_spray_policy(
    balancing: dict[str, Any],
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> LoadBalancing:
    """spray_policy and its parameters; those of one-way delay only with it."""
    unknown = set(balancing) - {"mode"} - {p.name for p in POLICY_PARAMETERS}
    if unknown:
        raise ValueError(
            "network.load_balancing.mode 'spray_policy' does not take "
            f"{sorted(unknown)}"
        )
    one_way_delay = _flag(
        balancing.get("one_way_delay", False), "network.load_balancing.one_way_delay"
    )
    delay_only = sorted(set(balancing) & ONE_WAY_DELAY_PARAMETERS)
    if not one_way_delay and delay_only:
        raise ValueError(f"network.load_balancing {delay_only} require one_way_delay")
    if "loss_interval_ns" in balancing and "loss_thresholds" not in balancing:
        raise ValueError(
            "network.load_balancing.loss_interval_ns requires loss_thresholds"
        )
    if "loss_hold_clean_sentinels" in balancing and "loss_thresholds" not in balancing:
        raise ValueError(
            "network.load_balancing.loss_hold_clean_sentinels requires loss_thresholds"
        )
    if balancing.get("held_exploration") == "none" and (
        balancing.get("grade_cost_source") != "switch_queue"
    ):
        raise ValueError(
            "network.load_balancing.held_exploration 'none' requires "
            "grade_cost_source 'switch_queue'"
        )
    if (
        "held_sentinel_interval_ns" in balancing
        and balancing.get("held_exploration") != "sentinel"
    ):
        raise ValueError(
            "network.load_balancing.held_sentinel_interval_ns requires "
            "held_exploration 'sentinel'"
        )
    for key in ("pool_decay_interval_ns", "pool_hold_propagates"):
        if key in balancing and not (
            _probability(
                balancing.get("sender_pool_weight", 0.0),
                "network.load_balancing.sender_pool_weight",
            )
            > 0
        ):
            raise ValueError(
                f"network.load_balancing.{key} requires sender_pool_weight above 0"
            )
    if balancing.get("grade_cost_source") == "switch_queue" and balancing.get(
        "grade_scope", "per_receiver"
    ) != "per_receiver":
        raise ValueError(
            "network.load_balancing.grade_cost_source 'switch_queue' requires "
            "grade_scope 'per_receiver'"
        )
    _require_reorder_tolerance("spray_policy", recovery, trimming)
    return LoadBalancing(
        mode="spray_policy",
        parameters=_parsed(
            balancing,
            tuple(
                parameter
                for parameter in POLICY_PARAMETERS
                if (one_way_delay or parameter.name not in ONE_WAY_DELAY_PARAMETERS)
                and (
                    parameter.name not in WRITTEN_WHEN_SET_PARAMETERS
                    or parameter.name in balancing
                )
            ),
        ),
    )


def _parsed(
    balancing: dict[str, Any], parameters: tuple[BalancingParameter, ...]
) -> tuple[tuple[str, ParameterValue], ...]:
    """Each parameter's value as given or its default, validated."""
    return tuple(
        (
            parameter.name,
            parameter.parse(
                balancing.get(parameter.name, parameter.default),
                f"network.load_balancing.{parameter.name}",
            ),
        )
        for parameter in parameters
    )


def _require_reorder_tolerance(
    mode: str,
    recovery: TransportRecovery | None,
    trimming: PacketTrimming | None,
) -> None:
    # A reordered flow needs a receiver that holds out-of-order data and a
    # fabric that names the range it cuts, instead of a gap the receiver
    # cannot tell from reordering.
    if mode != "ecmp":
        if recovery is None or not recovery.selective_repair:
            raise ValueError(
                f"network.load_balancing.mode '{mode}' requires "
                "network.transport_recovery.selective_repair"
            )
        if trimming is None or trimming.mode != "ftd":
            raise ValueError(
                f"network.load_balancing.mode '{mode}' requires "
                "network.packet_trimming.mode 'ftd'"
            )


def _optional_host(value: Any, field: str, host_count: int) -> int | None:
    if value is None:
        return None
    host = _nonnegative_int(value, field)
    if host >= host_count:
        raise ValueError(f"{field} is outside the host range")
    return host


def _load_data_loss(document: dict[str, Any], host_count: int) -> DataPlaneLoss | None:
    if "data_loss" not in document:
        return None
    loss = document["data_loss"]
    if not isinstance(loss, dict):
        raise ValueError("network.data_loss must be an object")
    required = {
        "probability",
        "start_ns",
        "duration_ns",
        "scope",
        "rng_stream",
    }
    optional = {"source_host", "destination_host", "receiver_node"}
    unknown = set(loss) - required - optional
    if unknown:
        raise ValueError(f"unknown network.data_loss keys: {sorted(unknown)}")
    missing = required - set(loss)
    if missing:
        raise ValueError(f"missing network.data_loss keys: {sorted(missing)}")
    scope = loss["scope"]
    if not isinstance(scope, str) or scope not in DATA_LOSS_SCOPES:
        raise ValueError(
            f"network.data_loss.scope must be one of {sorted(DATA_LOSS_SCOPES)}"
        )
    duration_ns = _positive_int(loss["duration_ns"], "network.data_loss.duration_ns")
    rng_stream = _positive_int(loss["rng_stream"], "network.data_loss.rng_stream")
    if rng_stream > 2**63 - 1:
        raise ValueError("network.data_loss.rng_stream exceeds the ns-3 range")
    source_host = _optional_host(
        loss.get("source_host"), "network.data_loss.source_host", host_count
    )
    destination_host = _optional_host(
        loss.get("destination_host"), "network.data_loss.destination_host", host_count
    )
    if source_host is not None and source_host == destination_host:
        raise ValueError(
            "network.data_loss source_host and destination_host must differ"
        )
    receiver_node = loss.get("receiver_node")
    if receiver_node is not None:
        receiver_node = _nonnegative_int(
            receiver_node, "network.data_loss.receiver_node"
        )
    return DataPlaneLoss(
        probability=_probability(loss["probability"], "network.data_loss.probability"),
        start_ns=_nonnegative_int(loss["start_ns"], "network.data_loss.start_ns"),
        duration_ns=duration_ns,
        scope=scope,
        source_host=source_host,
        destination_host=destination_host,
        receiver_node=receiver_node,
        rng_stream=rng_stream,
    )


def _load_transport_recovery(document: dict[str, Any]) -> TransportRecovery | None:
    if "transport_recovery" not in document:
        return None
    recovery = document["transport_recovery"]
    if not isinstance(recovery, dict):
        raise ValueError("network.transport_recovery must be an object")
    required = {"retransmission_timeout_ns", "max_retransmission_retries"}
    optional = {"selective_repair", "no_progress_timeout_ns"}
    if not required <= set(recovery) or set(recovery) - required - optional:
        raise ValueError(
            f"network.transport_recovery must contain exactly {sorted(required)} "
            f"plus optional {sorted(optional)}"
        )
    selective_repair = recovery.get("selective_repair", False)
    if not isinstance(selective_repair, bool):
        raise ValueError(
            "network.transport_recovery.selective_repair must be a boolean"
        )
    no_progress_timeout_ns = recovery.get(
        "no_progress_timeout_ns", TransportRecovery.no_progress_timeout_ns
    )
    return TransportRecovery(
        retransmission_timeout_ns=_positive_int(
            recovery["retransmission_timeout_ns"],
            "network.transport_recovery.retransmission_timeout_ns",
        ),
        max_retransmission_retries=_positive_int(
            recovery["max_retransmission_retries"],
            "network.transport_recovery.max_retransmission_retries",
        ),
        selective_repair=selective_repair,
        no_progress_timeout_ns=_positive_int(
            no_progress_timeout_ns,
            "network.transport_recovery.no_progress_timeout_ns",
        ),
    )


def _load_packet_trimming(document: dict[str, Any]) -> PacketTrimming | None:
    if "packet_trimming" not in document:
        return None
    trimming = document["packet_trimming"]
    optional = {
        "trimmed_queue",
        "trimmed_queue_weight",
        "min_trim_size_bytes",
        "last_hop_codepoint",
    }
    if not isinstance(trimming, dict) or not {"mode"} <= set(trimming) <= (
        {"mode"} | optional
    ):
        raise ValueError(
            "network.packet_trimming must contain 'mode' and may contain "
            f"{sorted(optional)}"
        )
    mode = trimming["mode"]
    if not isinstance(mode, str) or mode not in PACKET_TRIM_MODES:
        raise ValueError(
            f"network.packet_trimming.mode must be one of {sorted(PACKET_TRIM_MODES)}"
        )
    trimmed_queue = _positive_int(
        trimming.get("trimmed_queue", DEFAULT_PACKET_TRIM_QUEUE),
        "network.packet_trimming.trimmed_queue",
    )
    if trimmed_queue > MAX_PRIORITY_GROUP:
        raise ValueError(
            "network.packet_trimming.trimmed_queue must not exceed "
            f"{MAX_PRIORITY_GROUP}"
        )
    # UEC 1.0.3 section 4.1.4.1: DSCP_TRIMMED MUST be distinct from both
    # DSCP_TRIMMABLE and DSCP_CONTROL, and switches MUST place trimmed packets
    # into a traffic class other than the one carrying untrimmed data.
    if trimmed_queue == CONTROL_PRIORITY_GROUP or trimmed_queue in (
        DATA_PRIORITY_GROUPS
    ):
        raise ValueError(
            "network.packet_trimming.trimmed_queue must differ from the control "
            f"queue {CONTROL_PRIORITY_GROUP} and the data priority groups "
            f"{sorted(DATA_PRIORITY_GROUPS)}"
        )
    trimmed_queue_weight = _positive_int(
        trimming.get("trimmed_queue_weight", DEFAULT_TRIMMED_QUEUE_WEIGHT),
        "network.packet_trimming.trimmed_queue_weight",
    )
    if trimmed_queue_weight > 100:
        raise ValueError(
            "network.packet_trimming.trimmed_queue_weight is a percentage of "
            "egress bandwidth and must be in [1,100]"
        )
    min_trim_size_bytes = _positive_int(
        trimming.get("min_trim_size_bytes", DEFAULT_MIN_TRIM_SIZE_BYTES),
        "network.packet_trimming.min_trim_size_bytes",
    )
    if min_trim_size_bytes < DEFAULT_MIN_TRIM_SIZE_BYTES:
        raise ValueError(
            "network.packet_trimming.min_trim_size_bytes must be at least "
            f"{DEFAULT_MIN_TRIM_SIZE_BYTES} so the UDP and PDS request headers "
            "survive trimming"
        )
    last_hop_codepoint = trimming.get("last_hop_codepoint", True)
    if not isinstance(last_hop_codepoint, bool):
        raise ValueError(
            "network.packet_trimming.last_hop_codepoint must be a boolean"
        )
    return PacketTrimming(
        mode=mode,
        trimmed_queue=trimmed_queue,
        trimmed_queue_weight=trimmed_queue_weight,
        min_trim_size_bytes=min_trim_size_bytes,
        last_hop_codepoint=last_hop_codepoint,
    )


def _load_fabric(
    document: dict[str, Any],
    trimming: PacketTrimming | None,
    preset_queue_bytes: tuple[int, int] | None,
) -> SwitchFabric | None:
    """Parse the switch buffer; a switch preset may own the two queue bounds."""
    trimming_enabled = trimming is not None
    if "fabric" not in document:
        if trimming_enabled:
            raise ValueError(
                "network.packet_trimming requires an explicit network.fabric; "
                "trimming only fires when the data queue can reject a packet"
            )
        return None
    fabric = document["fabric"]
    queue_bounds = {"data_queue_bytes", "trimmed_queue_bytes"}
    if preset_queue_bytes is None:
        required = {"buffer_size_mb", "pfc_enabled", "data_queue_bytes"}
        optional = {"headroom_factor", "trimmed_queue_bytes"}
    elif isinstance(fabric, dict) and queue_bounds & set(fabric):
        raise ValueError(
            "network.switch.profile 'ue' sets network.fabric.data_queue_bytes "
            "and trimmed_queue_bytes from Plane_BDP; remove them"
        )
    else:
        required = {"buffer_size_mb", "pfc_enabled"}
        optional = {"headroom_factor"}
    if not isinstance(fabric, dict) or not required <= set(fabric) <= (
        required | optional
    ):
        raise ValueError(
            f"network.fabric must contain {sorted(required)} and may contain "
            f"{sorted(optional)}"
        )
    pfc_enabled = fabric["pfc_enabled"]
    if not isinstance(pfc_enabled, bool):
        raise ValueError("network.fabric.pfc_enabled must be a boolean")
    buffer_size_mb = _positive_int(
        fabric["buffer_size_mb"], "network.fabric.buffer_size_mb"
    )
    if preset_queue_bytes is None:
        data_queue_bytes = _positive_int(
            fabric["data_queue_bytes"], "network.fabric.data_queue_bytes"
        )
        trimmed_queue_bytes = _positive_int(
            fabric.get("trimmed_queue_bytes", data_queue_bytes),
            "network.fabric.trimmed_queue_bytes",
        )
    else:
        data_queue_bytes, trimmed_queue_bytes = preset_queue_bytes
    default_headroom = DEFAULT_HEADROOM_FACTOR if pfc_enabled else 0
    headroom_factor = _nonnegative_int(
        fabric.get("headroom_factor", default_headroom),
        "network.fabric.headroom_factor",
    )
    # PFC headroom only absorbs packets in flight when a PAUSE is sent. Without
    # PFC nothing pauses, so nonzero headroom is buffer that no mechanism ever
    # drains but that still has to fill before anything can be dropped.
    if not pfc_enabled and headroom_factor != 0:
        raise ValueError(
            "network.fabric.headroom_factor must be 0 when pfc_enabled is false"
        )
    # UEC 1.0.3 section 3.6.4.5 excludes PFC from best-effort networks, which is
    # the only regime in which trimming is meaningful.
    if trimming_enabled and pfc_enabled:
        raise ValueError(
            "network.packet_trimming requires network.fabric.pfc_enabled false"
        )
    return SwitchFabric(
        buffer_size_mb=buffer_size_mb,
        pfc_enabled=pfc_enabled,
        headroom_factor=headroom_factor,
        data_queue_bytes=data_queue_bytes,
        trimmed_queue_bytes=trimmed_queue_bytes,
    )


def _common_network_fields(document: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        _link_rate(document.get("link_rate")),
        _positive_int(
            document.get("packet_payload_bytes", DEFAULT_PACKET_PAYLOAD_BYTES),
            "network.packet_payload_bytes",
        ),
        _nonnegative_int(
            document.get("queue_monitor_start_ns", 0),
            "network.queue_monitor_start_ns",
        ),
        _positive_int(
            document.get("queue_monitor_interval_ns", 10_000),
            "network.queue_monitor_interval_ns",
        ),
    )


def load_network(document: Any, host_count: int) -> PhysicalNetwork:
    """Parse a discriminated physical-network profile with strict shape checks.

    A missing ``topology`` remains a legacy spelling for ``clos``. New profiles
    should always declare the discriminator explicitly.
    """
    if not isinstance(document, dict):
        raise ValueError("network must be an object")
    topology = document.get("topology", "clos")
    if topology not in {"clos", "ring"}:
        raise ValueError("network.topology must be either clos or ring")

    common_keys = {
        "topology",
        "link_rate",
        "packet_payload_bytes",
        "queue_monitor_start_ns",
        "queue_monitor_interval_ns",
        "data_loss",
        "transport_recovery",
        "packet_trimming",
        "fabric",
        "congestion_control",
        "switch",
        "host_link_delay_ns",
        "switch_link_delay_ns",
        "link_overrides",
    }
    topology_keys = {"hosts_per_leaf", "spine_count"} if topology == "clos" else set()
    optional_topology_keys = (
        {"failed_spine_count", "load_balancing", "link_failures"}
        if topology == "clos"
        else set()
    )
    unknown_keys = set(document) - common_keys - topology_keys - optional_topology_keys
    if unknown_keys:
        raise ValueError(f"unknown network keys: {sorted(unknown_keys)}")
    missing_keys = ({"link_rate"} | topology_keys) - set(document)
    if missing_keys:
        raise ValueError(f"missing network keys: {sorted(missing_keys)}")

    (
        link_rate,
        packet_payload_bytes,
        queue_monitor_start_ns,
        queue_monitor_interval_ns,
    ) = _common_network_fields(document)
    if packet_payload_bytes > MAX_PACKET_PAYLOAD_BYTES:
        raise ValueError(
            f"network.packet_payload_bytes must not exceed {MAX_PACKET_PAYLOAD_BYTES}"
        )
    switch_request = load_switch_request(document)
    data_loss = _load_data_loss(document, host_count)
    transport_recovery = _load_transport_recovery(document)
    packet_trimming = _load_packet_trimming(document)
    # The bare fabric: what the switch preset needs to measure Plane_BDP.
    geometry: PhysicalNetwork = _load_geometry(
        document,
        topology,
        host_count,
        link_rate,
        packet_payload_bytes,
        queue_monitor_start_ns,
        queue_monitor_interval_ns,
    )
    switch = _resolve_switch(document, switch_request, geometry, host_count)
    if switch.profile == "ue" and packet_trimming is None:
        raise ValueError(
            "network.switch.profile 'ue' requires network.packet_trimming: "
            "UEC 1.0.3 section 3.6.17 fixes the thresholds of a trimming fabric"
        )
    fabric = _load_fabric(document, packet_trimming, switch.queue_bytes())
    congestion_control = _load_congestion_control(document)
    if (
        data_loss is not None or packet_trimming is not None
    ) and transport_recovery is None:
        raise ValueError(
            "network.transport_recovery is required when data_loss or packet_trimming is enabled"
        )
    # A best-effort fabric drops on buffer exhaustion whether or not trimming is
    # enabled, so the transport must be able to recover or a dropped packet
    # strands its QP forever.
    if (
        fabric is not None
        and not fabric.pfc_enabled
        and transport_recovery is None
    ):
        raise ValueError(
            "network.transport_recovery is required when network.fabric disables "
            "PFC, because a best-effort fabric can drop data packets"
        )
    network: PhysicalNetwork = replace(
        geometry,
        data_loss=data_loss,
        transport_recovery=transport_recovery,
        packet_trimming=packet_trimming,
        fabric=fabric,
        congestion_control=congestion_control,
        switch=switch,
    )
    if isinstance(network, ClosNetwork):
        network = replace(
            network,
            load_balancing=_load_load_balancing(
                document, transport_recovery, packet_trimming
            ),
        )
        _require_reported_spines(network)
    _require_acknowledged_packets(network)
    return network


def _load_geometry(
    document: dict[str, Any],
    topology: str,
    host_count: int,
    link_rate: str,
    packet_payload_bytes: int,
    queue_monitor_start_ns: int,
    queue_monitor_interval_ns: int,
) -> PhysicalNetwork:
    """The ring or Clos with its links and packets, and nothing configured."""
    host_link_delay_ns = _positive_int(
        document.get("host_link_delay_ns", DEFAULT_HOST_LINK_DELAY_NS),
        "network.host_link_delay_ns",
    )
    switch_link_delay_ns = _positive_int(
        document.get("switch_link_delay_ns", DEFAULT_SWITCH_LINK_DELAY_NS),
        "network.switch_link_delay_ns",
    )
    if topology == "ring":
        if host_count < 3:
            raise ValueError("network.ring requires at least three hosts")
        ring = RingNetwork(
            link_rate=link_rate,
            packet_payload_bytes=packet_payload_bytes,
            queue_monitor_start_ns=queue_monitor_start_ns,
            queue_monitor_interval_ns=queue_monitor_interval_ns,
            host_link_delay_ns=host_link_delay_ns,
            switch_link_delay_ns=switch_link_delay_ns,
        )
        return replace(
            ring, link_overrides=_load_link_overrides(document, ring, host_count, None)
        )

    hosts_per_leaf = _positive_int(document["hosts_per_leaf"], "network.hosts_per_leaf")
    spine_count = _positive_int(document["spine_count"], "network.spine_count")
    if host_count % hosts_per_leaf:
        raise ValueError(
            "parallelism product must be divisible by network.hosts_per_leaf"
        )
    if spine_count > 255:
        raise ValueError("network.spine_count must not exceed 255")
    failed_value = document.get("failed_spine_count", 0)
    if isinstance(failed_value, bool) or not isinstance(failed_value, int):
        raise ValueError("network.failed_spine_count must be an integer")
    if failed_value < 0:
        raise ValueError("network.failed_spine_count must not be negative")
    if failed_value >= spine_count:
        raise ValueError(
            "network.failed_spine_count must leave at least one live spine"
        )
    clos = ClosNetwork(
        link_rate=link_rate,
        packet_payload_bytes=packet_payload_bytes,
        queue_monitor_start_ns=queue_monitor_start_ns,
        queue_monitor_interval_ns=queue_monitor_interval_ns,
        hosts_per_leaf=hosts_per_leaf,
        spine_count=spine_count,
        failed_spine_count=failed_value,
        host_link_delay_ns=host_link_delay_ns,
        switch_link_delay_ns=switch_link_delay_ns,
    )
    index = clos.clos_index(host_count)
    return replace(
        clos,
        link_overrides=_load_link_overrides(document, clos, host_count, index),
        link_failures=load_link_failures(document, index),
    )


def _load_link_overrides(
    document: dict[str, Any],
    network: PhysicalNetwork,
    host_count: int,
    index: ClosIndex | None,
) -> tuple[LinkOverride, ...]:
    """The overrides, checked against the links the fabric builds."""
    links = build_topology(network, host_count).links
    return load_link_overrides(
        document,
        ((link.source, link.destination) for link in links),
        index,
        (rate for rate, _, _ in ECN_THRESHOLDS_KB),
    )


def _resolve_switch(
    document: dict[str, Any],
    request: SwitchRequest,
    geometry: PhysicalNetwork,
    host_count: int,
) -> SwitchProfile:
    """The switch profile, a preset's thresholds measured on the built fabric."""
    if request.profile == "default":
        return request.default_switch()
    control = document.get("congestion_control")
    if isinstance(control, dict) and "ecn_threshold_scale" in control:
        raise ValueError(
            "network.congestion_control.ecn_threshold_scale cannot be combined "
            f"with network.switch.profile '{request.profile}', which sets the "
            "ECN thresholds from Plane_BDP"
        )
    return request.ue_switch(
        build_topology(geometry, host_count),
        geometry.packet_payload_bytes,
        link_rate_bits_per_second(geometry.link_rate),
    )


def _require_acknowledged_packets(network: PhysicalNetwork) -> None:
    """Refuse NSCC where the receiver does not acknowledge every packet.

    NSCC clocks its window on the acknowledgement of each data packet and times
    each from the packet's send record; under ``ecmp`` the receiver
    acknowledges cumulatively and the sender keeps no records.
    """
    if (
        network.congestion_control.mode == "nscc"
        and network.load_balancing.mode == "ecmp"
    ):
        raise ValueError(
            "network.congestion_control.mode 'nscc' requires "
            "network.load_balancing.mode 'ev_hash', 'spray_uniform' or "
            "'spray_policy'"
        )


def _require_reported_spines(network: ClosNetwork) -> None:
    """Refuse a spray_policy report over more spines than it can grade, and an
    ack_redirect field over more spines than it can name."""
    if (
        network.load_balancing.mode == "spray_policy"
        and network.live_spine_count > MAX_REPORTED_SPINES
    ):
        raise ValueError(
            "network.load_balancing.mode 'spray_policy' grades at most "
            f"{MAX_REPORTED_SPINES} spines"
        )
    if (
        network.load_balancing.selector == "ack_redirect"
        and dict(network.load_balancing.parameters).get("scope") == "per_receiver"
        and network.live_spine_count > MAX_REPORTED_SPINES
    ):
        raise ValueError(
            "network.load_balancing.selector 'ack_redirect' with scope "
            f"'per_receiver' grades at most {MAX_REPORTED_SPINES} spines"
        )
    if (
        network.load_balancing.selector in LOSS_HOLD_SELECTORS
        and network.live_spine_count > MAX_REPORTED_SPINES
    ):
        raise ValueError(
            f"network.load_balancing.selector '{network.load_balancing.selector}' "
            f"holds at most {MAX_REPORTED_SPINES} spines"
        )
    if (
        network.load_balancing.traces_receiver_hold()
        and network.live_spine_count > MAX_RECEIVER_HOLD_SPINES
    ):
        raise ValueError(
            "network.load_balancing.receiver_hold holds at most "
            f"{MAX_RECEIVER_HOLD_SPINES} spines"
        )
    if (
        network.load_balancing.recovers_tail()
        and network.live_spine_count > MAX_TAIL_RECOVERY_SPINES
    ):
        raise ValueError(
            "network.load_balancing.tail_recovery counts at most "
            f"{MAX_TAIL_RECOVERY_SPINES} spines"
        )
    if (
        network.load_balancing.selector == "ack_redirect"
        and network.live_spine_count >= MAX_REDIRECT_SPINES
    ):
        raise ValueError(
            "network.load_balancing.selector 'ack_redirect' names at most "
            f"{MAX_REDIRECT_SPINES - 1} spines"
        )


def build_topology(network: PhysicalNetwork, host_count: int) -> TopologyLayout:
    """Build a validated physical layout for the requested network variant."""
    if isinstance(network, ClosNetwork):
        layout = _build_clos_topology(network, host_count)
    elif isinstance(network, RingNetwork):
        layout = _build_ring_topology(network, host_count)
    else:
        raise TypeError(f"unsupported physical network: {type(network).__name__}")
    if (
        network.data_loss is not None
        and network.data_loss.receiver_node is not None
        and network.data_loss.receiver_node >= layout.node_count
    ):
        raise ValueError("network.data_loss.receiver_node is outside the topology")
    return layout


def _build_clos_topology(network: ClosNetwork, host_count: int) -> TopologyLayout:
    leaf_count = host_count // network.hosts_per_leaf
    leaf_start = host_count
    spine_start = leaf_start + leaf_count
    # Failed spines are simply absent from the built fabric: a dark spine
    # carries no traffic and terminates no link, so only the live tier is
    # materialized. The manifest keeps the designed count so a degraded run
    # is legible as "a 1:1 fabric with spines down", not a smaller design.
    node_count = spine_start + network.live_spine_count
    links = [
        TopologyLink(
            source=host,
            destination=leaf_start + host // network.hosts_per_leaf,
            delay_ns=network.host_link_delay_ns,
            rate=network.link_rate,
        )
        for host in range(host_count)
    ]
    links.extend(
        TopologyLink(
            source=leaf,
            destination=spine,
            delay_ns=network.switch_link_delay_ns,
            rate=network.link_rate,
        )
        for leaf in range(leaf_start, spine_start)
        for spine in range(spine_start, node_count)
    )
    return TopologyLayout(
        kind=network.kind,
        description=(
            "Two-stage leaf-spine Clos"
            if network.failed_spine_count == 0
            else "Two-stage leaf-spine Clos, degraded by failed spines"
        ),
        host_count=host_count,
        node_count=node_count,
        switch_ids=tuple(range(leaf_start, node_count)),
        links=_overridden(links, network.link_overrides),
        link_rate=network.link_rate,
        details=(
            ("leaf_count", leaf_count),
            ("spine_count", network.spine_count),
            ("failed_spine_count", network.failed_spine_count),
            ("live_spine_count", network.live_spine_count),
            ("hosts_per_leaf", network.hosts_per_leaf),
            ("host_link_delay_ns", network.host_link_delay_ns),
            ("switch_link_delay_ns", network.switch_link_delay_ns),
        ),
    )


def _build_ring_topology(network: RingNetwork, host_count: int) -> TopologyLayout:
    switch_start = host_count
    switch_ids = tuple(range(switch_start, switch_start + host_count))
    links = [
        TopologyLink(
            source=host,
            destination=switch_start + host,
            delay_ns=network.host_link_delay_ns,
            rate=network.link_rate,
        )
        for host in range(host_count)
    ]
    links.extend(
        TopologyLink(
            source=switch_start + index,
            destination=switch_start + (index + 1) % host_count,
            delay_ns=network.switch_link_delay_ns,
            rate=network.link_rate,
        )
        for index in range(host_count)
    )
    return TopologyLayout(
        kind=network.kind,
        description="Host-attached bidirectional switch ring",
        host_count=host_count,
        node_count=host_count * 2,
        switch_ids=switch_ids,
        links=_overridden(links, network.link_overrides),
        link_rate=network.link_rate,
        details=(
            ("switch_ring_size", host_count),
            ("host_link_delay_ns", network.host_link_delay_ns),
            ("switch_link_delay_ns", network.switch_link_delay_ns),
        ),
    )


def _overridden(
    links: list[TopologyLink], overrides: tuple[LinkOverride, ...]
) -> tuple[TopologyLink, ...]:
    """The links with each override's settings in place of the fabric's."""
    by_link = {tuple(sorted(override.endpoints)): override for override in overrides}
    result = []
    for link in links:
        override = by_link.get(tuple(sorted((link.source, link.destination))))
        if override is None:
            result.append(link)
            continue
        result.append(
            replace(
                link,
                rate=override.rate or link.rate,
                delay_ns=override.delay_ns or link.delay_ns,
                error_rate=override.error_rate or link.error_rate,
            )
        )
    return tuple(result)
