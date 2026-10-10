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
LOAD_BALANCING_MODES = {"ecmp", "ev_hash", "spray_uniform"}
# What chooses each data packet's entropy value under ev_hash.
PATH_SELECTORS = ("ops", "reps", "ue_oblivious", "ue_aware")
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
    values, and ``ue_aware`` rotates skipping a value once after a congestion
    report. ``spray_uniform`` has the sender name a spine per packet, uniformly
    at random, and the source leaf send the packet up that spine. Every mode but
    ``ecmp`` reorders a flow's packets on the way.
    """

    mode: str
    selector: str = "ops"
    # The selector's parameters by profile key, defaults filled in.
    parameters: tuple[tuple[str, int | float], ...] = ()

    def manifest(self) -> dict[str, str | int | float]:
        if self.mode != "ev_hash":
            return {"mode": self.mode}
        return {"mode": self.mode, "selector": self.selector, **dict(self.parameters)}

    def selector_settings(self) -> str:
        """PATH_SELECTOR and the selector's parameters. OPS has none and is the
        simulator's default, so an ``ops`` profile writes nothing."""
        if self.selector == "ops":
            return ""
        values = dict(self.parameters)
        return f"PATH_SELECTOR {self.selector}\n" + "".join(
            f"{parameter.key} {values[parameter.name]}\n"
            for parameter in SELECTOR_PARAMETERS[self.selector]
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


@dataclass(frozen=True)
class SelectorParameter:
    """One parameter of a path selector: its profile key, its simulator key,
    the default its source gives, and the parser that validates a value."""

    name: str
    key: str
    default: int | float
    parse: Callable[[Any, str], int | float]


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


# UEC 1.0.3 section 3.6.16.3's typical space.
_UE_EV_SET_SIZE = SelectorParameter("ev_set_size", "UE_EV_SET_SIZE", 256, _ev_set_size)
SELECTOR_PARAMETERS: dict[str, tuple[SelectorParameter, ...]] = {
    "ops": (),
    # REPS section 3.1 and its artifact's exit_freeze_after.
    "reps": (
        SelectorParameter("buffer_size", "REPS_BUFFER_SIZE", 8, _reps_buffer_size),
        SelectorParameter(
            "freezing_timeout_ns", "REPS_FREEZING_TIMEOUT_NS", 10_000_000, _positive_int
        ),
    ),
    "ue_oblivious": (_UE_EV_SET_SIZE,),
    # UEC 1.0.3 section 3.6.16.4's default saturation.
    "ue_aware": (
        _UE_EV_SET_SIZE,
        SelectorParameter(
            "saturation_fraction", "UE_SATURATION_FRACTION", 0.5, _fraction
        ),
    ),
}


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
    if mode != "ev_hash" and set(balancing) != {"mode"}:
        raise ValueError(
            f"network.load_balancing.mode '{mode}' takes no other key; a selector "
            "and its parameters require mode 'ev_hash'"
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
    return LoadBalancing(
        mode=mode,
        selector=selector,
        parameters=tuple(
            (
                parameter.name,
                parameter.parse(
                    balancing.get(parameter.name, parameter.default),
                    f"network.load_balancing.{parameter.name}",
                ),
            )
            for parameter in parameters
        ),
    )


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    probability = float(value)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a number in [0, 1]")
    return probability


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
            "network.load_balancing.mode 'ev_hash' or 'spray_uniform'"
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
