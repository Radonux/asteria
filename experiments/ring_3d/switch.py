"""Switch ECN and queue-threshold configuration for ring_3d profiles.

``network.switch`` picks how the switches mark and bound their queues. The
``default`` profile is what every profile ran before the knob existed: the
HPCC-era ECN table, scaled by ``congestion_control.ecn_threshold_scale``, with
a RED marking probability that the profile may now set. The ``ue`` profile is
the configuration UEC 1.0.3 section 3.6.17 recommends for a UET-CC fabric
that trims, every threshold derived from the fabric's Plane_BDP; with
``port_rate_thresholds`` each port's thresholds are those bytes scaled by its
link speed over the fabric's link_rate, so that every port marks, trims and
drops at one queueing delay.

This module imports nothing else from the package, so ``topology`` can import
it without a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Any, Protocol

SWITCH_PROFILES = ("default", "ue")
# The HPCC switch ECN table: (link bps, KMIN KB, KMAX KB) per link speed. ns-3
# reads each threshold as an integer count of kilobytes, so a scaled value is
# rounded and must stay at least 1.
ECN_THRESHOLDS_KB = (
    (25_000_000_000, 100, 400),
    (40_000_000_000, 200, 800),
    (100_000_000_000, 400, 1600),
    (200_000_000_000, 600, 2400),
    (400_000_000_000, 800, 3200),
    (2_400_000_000_000, 800, 3200),
)
# Link speeds outside the HPCC table, each written into the maps, ahead of
# the table, only where a link of the fabric runs at it, so that every other
# fabric's maps stay as they were: 12.5 Gb/s, a leaf-spine link of a leaf's
# 3.2 Tb/s over 256 spines, at half the 25 Gb/s thresholds.
ECN_EXTRA_THRESHOLDS_KB = ((12_500_000_000, 50, 200),)
# Every link speed the maps can cover.
ECN_RATES_BPS = frozenset(
    rate for rate, _, _ in ECN_EXTRA_THRESHOLDS_KB + ECN_THRESHOLDS_KB
)
# The marking probability at KMAX that the generator wrote for every link speed
# before it could be set.
DEFAULT_ECN_MARKING_PROBABILITY = 0.2
# UEC 1.0.3 section 3.6.17, for queue_low under probabilistic marking and with
# trimming enabled, as fractions of Plane_BDP: ECN from 0.2 to 0.8, marking
# every packet at 0.8, and trimming at 1. queue_med, which holds the trimmed
# packets, drops at 1.
UE_ECN_MIN_FRACTION = Decimal("0.2")
UE_ECN_MAX_FRACTION = Decimal("0.8")
UE_ECN_MARKING_PROBABILITY = 1.0
UE_TRIM_THRESHOLD_FRACTION = 1
UE_TRIMMED_QUEUE_DROP_FRACTION = 1
# ns-3 reads DATA_QUEUE_BYTES and TRIMMED_QUEUE_BYTES as 32-bit integers.
MAX_QUEUE_BYTES = 2**32 - 1


class _Link(Protocol):
    source: int
    destination: int
    delay_ns: int

    @property
    def rate_bps(self) -> int: ...


class _Layout(Protocol):
    host_count: int
    links: Sequence[_Link]


def ecn_rows(rates_bps: Iterable[int] = ()) -> tuple[tuple[int, int, int], ...]:
    """The ECN table's rows the maps write for a fabric whose links run at
    rates_bps: ECN_THRESHOLDS_KB, after the ECN_EXTRA_THRESHOLDS_KB rows of
    the rates among them."""
    used = set(rates_bps)
    return (
        tuple(row for row in ECN_EXTRA_THRESHOLDS_KB if row[0] in used)
        + ECN_THRESHOLDS_KB
    )


def _probability_map(
    probability: float, rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB
) -> str:
    """The PMAX_MAP line, one probability for every link speed of rows."""
    entries = " ".join(f"{rate} {probability!r}" for rate, _, _ in rows)
    return f"PMAX_MAP {len(rows)} {entries}\n"


@dataclass(frozen=True)
class DefaultSwitch:
    """The generator's historical switch: HPCC ECN table, settable RED pmax."""

    ecn_marking_probability: float = DEFAULT_ECN_MARKING_PROBABILITY

    @property
    def profile(self) -> str:
        return "default"

    def ecn_threshold_maps(
        self,
        scaled_maps: str,
        rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB,
    ) -> str:
        """The KMAX_MAP and KMIN_MAP lines: the congestion-control knob's,
        scaled over the same rows."""
        del rows
        return scaled_maps

    def probability_map(
        self, rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB
    ) -> str:
        return _probability_map(self.ecn_marking_probability, rows)

    def queue_bytes(self) -> None:
        """The default profile leaves the queue bounds to network.fabric."""

    def queue_maps(
        self, rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB
    ) -> str:
        """No per-speed queue bounds."""
        del rows
        return ""

    def manifest(self) -> dict[str, str | float]:
        return {
            "profile": self.profile,
            "ecn_marking_probability": self.ecn_marking_probability,
        }


@dataclass(frozen=True)
class UeSwitch:
    """UEC 1.0.3 section 3.6.17, derived from the fabric's Plane_BDP.

    Plane_BDP is the link rate times config_base_rtt, the longest round trip
    between two hosts with no queueing. Both are computed as the ns-3 backend
    computes its maxRtt and maxBdp, so the thresholds and the transport's own
    BDP agree. ECN marks only queue_low: queue 0 (TC_high) is never marked and
    the switch never marks a trimmed packet, which is what queue_med holds.

    With ``port_rate_thresholds`` a port of link speed r takes Plane_BDP's
    thresholds times r / link_bps: its ECN thresholds in its KMAX_MAP and
    KMIN_MAP rows, and its trim threshold and trimmed queue drop in the
    DATA_QUEUE_BYTES_MAP and TRIMMED_QUEUE_BYTES_MAP rows the backend reads
    in place of DATA_QUEUE_BYTES and TRIMMED_QUEUE_BYTES. A port at link_bps
    keeps the bytes it has without it. ``port_rates_bps`` are the fabric's
    link speeds, whose thresholds the manifest records.
    """

    config_base_rtt_ns: int
    link_bps: int
    port_rate_thresholds: bool = False
    port_rates_bps: tuple[int, ...] = ()

    @property
    def profile(self) -> str:
        return "ue"

    @property
    def plane_bdp_bytes(self) -> int:
        # The backend's integer arithmetic, in its order.
        return self.config_base_rtt_ns * self.link_bps // 1_000_000_000 // 8

    @property
    def ecn_min_kb(self) -> int:
        return _whole_kb(self.plane_bdp_bytes * UE_ECN_MIN_FRACTION)

    @property
    def ecn_max_kb(self) -> int:
        return _whole_kb(self.plane_bdp_bytes * UE_ECN_MAX_FRACTION)

    def _port_kb(self, fraction: Decimal, rate: int) -> int:
        """Plane_BDP's fraction scaled to a port of link speed rate, in the
        whole kilobytes ns-3 reads."""
        if not self.port_rate_thresholds:
            return _whole_kb(self.plane_bdp_bytes * fraction)
        return _whole_kb(self.plane_bdp_bytes * fraction * rate / self.link_bps)

    def port_queue_bytes(self, rate: int) -> tuple[int, int]:
        """queue_bytes for a port of link speed rate."""
        trim_threshold, trimmed_drop = self.queue_bytes()
        if not self.port_rate_thresholds:
            return trim_threshold, trimmed_drop
        return (
            trim_threshold * rate // self.link_bps,
            trimmed_drop * rate // self.link_bps,
        )

    def ecn_threshold_maps(
        self,
        scaled_maps: str,
        rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB,
    ) -> str:
        del scaled_maps  # the preset replaces the table
        count = len(rows)
        kmax = " ".join(
            f"{rate} {self._port_kb(UE_ECN_MAX_FRACTION, rate)}" for rate, _, _ in rows
        )
        kmin = " ".join(
            f"{rate} {self._port_kb(UE_ECN_MIN_FRACTION, rate)}" for rate, _, _ in rows
        )
        return f"KMAX_MAP {count} {kmax}\nKMIN_MAP {count} {kmin}\n"

    def queue_maps(
        self, rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB
    ) -> str:
        """Under port_rate_thresholds, the DATA_QUEUE_BYTES_MAP and
        TRIMMED_QUEUE_BYTES_MAP lines over rows; nothing otherwise."""
        if not self.port_rate_thresholds:
            return ""
        count = len(rows)
        data = " ".join(
            f"{rate} {self.port_queue_bytes(rate)[0]}" for rate, _, _ in rows
        )
        trimmed = " ".join(
            f"{rate} {self.port_queue_bytes(rate)[1]}" for rate, _, _ in rows
        )
        return (
            f"DATA_QUEUE_BYTES_MAP {count} {data}\n"
            f"TRIMMED_QUEUE_BYTES_MAP {count} {trimmed}\n"
        )

    def probability_map(
        self, rows: Sequence[tuple[int, int, int]] = ECN_THRESHOLDS_KB
    ) -> str:
        return _probability_map(UE_ECN_MARKING_PROBABILITY, rows)

    def queue_bytes(self) -> tuple[int, int]:
        """DATA_QUEUE_BYTES (the trim threshold) and TRIMMED_QUEUE_BYTES."""
        bdp = self.plane_bdp_bytes
        return (
            bdp * UE_TRIM_THRESHOLD_FRACTION,
            bdp * UE_TRIMMED_QUEUE_DROP_FRACTION,
        )

    def manifest(self) -> dict[str, str | int | float]:
        trim_threshold, trimmed_drop = self.queue_bytes()
        return {
            "profile": self.profile,
            "config_base_rtt_ns": self.config_base_rtt_ns,
            "plane_bdp_bytes": self.plane_bdp_bytes,
            "ecn_min_bytes": self.ecn_min_kb * 1000,
            "ecn_max_bytes": self.ecn_max_kb * 1000,
            "ecn_marking_probability": UE_ECN_MARKING_PROBABILITY,
            "trim_threshold_bytes": trim_threshold,
            "trimmed_queue_drop_bytes": trimmed_drop,
            **(
                {
                    "port_rate_thresholds": {
                        str(rate): {
                            "ecn_min_bytes": self._port_kb(UE_ECN_MIN_FRACTION, rate)
                            * 1000,
                            "ecn_max_bytes": self._port_kb(UE_ECN_MAX_FRACTION, rate)
                            * 1000,
                            "trim_threshold_bytes": self.port_queue_bytes(rate)[0],
                            "trimmed_queue_drop_bytes": self.port_queue_bytes(rate)[1],
                        }
                        for rate in self.port_rates_bps
                    }
                }
                if self.port_rate_thresholds
                else {}
            ),
        }


SwitchProfile = DefaultSwitch | UeSwitch


def _whole_kb(value: Decimal) -> int:
    """Bytes to the whole kilobytes ns-3 reads, rounded half up."""
    return int(
        (value / 1000 + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR")
    )


def longest_round_trip_ns(layout: _Layout, payload_bytes: int) -> int:
    """The longest unloaded host-to-host round trip, as ns-3 computes maxRtt.

    The backend routes over shortest-hop paths through switches only. A pair's
    round trip is twice the least one-way propagation among those paths plus
    the least serialization of one data packet along one of them, each link
    serializing at its own rate. A host's own link adds twice its delay and one
    packet time at either end.
    """
    def cost(link: _Link) -> tuple[int, int]:
        return link.delay_ns, payload_bytes * 1_000_000_000 * 8 // link.rate_bps

    hosts = range(layout.host_count)
    neighbours: dict[int, list[tuple[int, int, int]]] = {}
    access: dict[int, list[int]] = {}  # switch -> each attached host's link cost
    attached: set[int] = set()
    for link in layout.links:
        delay, serialization = cost(link)
        if link.source in hosts or link.destination in hosts:
            host, switch = sorted((link.source, link.destination))
            if host in attached:
                raise ValueError("a host attaches to more than one switch")
            attached.add(host)
            access.setdefault(switch, []).append(2 * delay + serialization)
        else:
            neighbours.setdefault(link.source, []).append(
                (link.destination, delay, serialization)
            )
            neighbours.setdefault(link.destination, []).append(
                (link.source, delay, serialization)
            )
    if len(attached) != layout.host_count:
        raise ValueError("every host must attach to one switch")
    for costs in access.values():
        costs.sort(reverse=True)
    longest_rtt = 0
    for source, source_costs in access.items():
        # Breadth first over switches, keeping per switch its hop count and the
        # least delay and serialization of the paths of that many hops.
        reached = {source: (0, 0, 0)}
        frontier = [source]
        while frontier:
            following = []
            for switch in frontier:
                hops, delay, serialization = reached[switch]
                for neighbour, link_delay, link_serialization in neighbours.get(
                    switch, ()
                ):
                    candidate = (
                        hops + 1,
                        delay + link_delay,
                        serialization + link_serialization,
                    )
                    if neighbour not in reached:
                        reached[neighbour] = candidate
                        following.append(neighbour)
                    elif reached[neighbour][0] == hops + 1:
                        _, best_delay, best_serialization = reached[neighbour]
                        reached[neighbour] = (
                            hops + 1,
                            min(best_delay, candidate[1]),
                            min(best_serialization, candidate[2]),
                        )
            frontier = following
        for target, target_costs in access.items():
            if target == source:
                if len(source_costs) < 2:
                    continue
                longest_rtt = max(longest_rtt, source_costs[0] + source_costs[1])
                continue
            if target not in reached:
                raise ValueError("the fabric does not connect every pair of hosts")
            _, delay, serialization = reached[target]
            longest_rtt = max(
                longest_rtt,
                source_costs[0] + 2 * delay + serialization + target_costs[0],
            )
    return longest_rtt


@dataclass(frozen=True)
class SwitchRequest:
    """``network.switch`` as written, before the fabric is known."""

    profile: str
    ecn_marking_probability: float | None
    port_rate_thresholds: bool = False

    def default_switch(self) -> DefaultSwitch:
        """The ``default`` profile, which needs nothing from the fabric."""
        if self.profile != "default":
            raise ValueError(f"network.switch.profile '{self.profile}' is a preset")
        return DefaultSwitch(
            ecn_marking_probability=(
                DEFAULT_ECN_MARKING_PROBABILITY
                if self.ecn_marking_probability is None
                else self.ecn_marking_probability
            )
        )

    def ue_switch(self, layout: _Layout, payload_bytes: int, link_bps: int) -> UeSwitch:
        """The ``ue`` preset, its thresholds measured on the built fabric."""
        if self.profile != "ue":
            raise ValueError(f"network.switch.profile '{self.profile}' is not 'ue'")
        switch = UeSwitch(
            config_base_rtt_ns=longest_round_trip_ns(layout, payload_bytes),
            link_bps=link_bps,
            port_rate_thresholds=self.port_rate_thresholds,
            port_rates_bps=tuple(sorted({link.rate_bps for link in layout.links})),
        )
        if switch.ecn_min_kb < 1:
            raise ValueError(
                "network.switch.profile 'ue' needs a Plane_BDP of at least 2.5 KB "
                "so that the ECN minimum is a whole kilobyte"
            )
        if any(
            switch._port_kb(UE_ECN_MIN_FRACTION, rate) < 1
            for rate in switch.port_rates_bps
        ):
            raise ValueError(
                "network.switch.port_rate_thresholds needs every port's scaled ECN "
                "minimum to be a whole kilobyte"
            )
        if switch.plane_bdp_bytes > MAX_QUEUE_BYTES:
            raise ValueError(
                "network.switch.profile 'ue' derives a Plane_BDP above the 32-bit "
                "queue bound ns-3 reads"
            )
        return switch


def load_switch_request(document: dict[str, Any]) -> SwitchRequest:
    """Parse the optional knob, defaulting to the historical switch."""
    if "switch" not in document:
        return SwitchRequest(profile="default", ecn_marking_probability=None)
    switch = document["switch"]
    allowed = {"profile", "ecn_marking_probability", "port_rate_thresholds"}
    if not isinstance(switch, dict) or not set(switch) <= allowed:
        raise ValueError(f"network.switch may contain only {sorted(allowed)}")
    profile = switch.get("profile", "default")
    if not isinstance(profile, str) or profile not in SWITCH_PROFILES:
        raise ValueError(
            f"network.switch.profile must be one of {list(SWITCH_PROFILES)}"
        )
    probability = switch.get("ecn_marking_probability")
    if probability is not None:
        if profile == "ue":
            raise ValueError(
                "network.switch.ecn_marking_probability is fixed at 1 by "
                "network.switch.profile 'ue'"
            )
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not isfinite(probability)
            or not 0.0 < probability <= 1.0
        ):
            raise ValueError(
                "network.switch.ecn_marking_probability must be a number in (0, 1]"
            )
        probability = float(probability)
    port_rate = switch.get("port_rate_thresholds", False)
    if not isinstance(port_rate, bool):
        raise ValueError("network.switch.port_rate_thresholds must be a boolean")
    if "port_rate_thresholds" in switch and profile != "ue":
        raise ValueError(
            "network.switch.port_rate_thresholds is read only by profile 'ue', "
            "whose thresholds derive from Plane_BDP"
        )
    return SwitchRequest(
        profile=profile,
        ecn_marking_probability=probability,
        port_rate_thresholds=port_rate,
    )
