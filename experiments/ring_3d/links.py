"""Per-link settings and link failures of a generated fabric.

``network.link_overrides`` gives chosen links their own rate, delay or error
rate for the whole run, in the topology file. ``network.link_failures`` changes
a spine's links, or one leaf-spine link, at a time in the run, through
``LINK_FAILURE`` lines in the network configuration, and, for a failure with a
duration, restores them through ``LINK_RESTORE`` lines. A Clos link is named by
its leaf and spine index, the spine index counting the spines that are built;
any link may also be named by the node ids at its two ends.

This module imports nothing else from the package, so ``topology`` can import
it without a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import expm1, isfinite, log1p
from typing import Any

FAILURE_MODELS = ("graceful", "silent", "gray")
# What a gray loss failure's error_rate is the rate of: each packet, frame,
# whatever its size, or each byte, byte, so that a packet is lost by its
# size, the rate derived so that a full data frame keeps error_rate.
LOSS_UNITS = ("frame", "byte")
# A failure's direction on its leaf-spine link: "up" is what the leaf sends
# the spine, "down" what the spine sends the leaf, "both" the two.
FAILURE_DIRECTIONS = ("up", "down", "both")


@dataclass(frozen=True)
class ClosIndex:
    """Node ids of a Clos built with hosts first, then leaves, then spines."""

    host_count: int
    leaf_count: int
    spine_count: int

    def leaf(self, index: int) -> int:
        return self.host_count + index

    def spine(self, index: int) -> int:
        return self.host_count + self.leaf_count + index


@dataclass(frozen=True)
class LinkOverride:
    """One link's own settings; None keeps the fabric's."""

    endpoints: tuple[int, int]
    rate: str | None
    delay_ns: int | None
    error_rate: float | None

    def manifest(self) -> dict[str, Any]:
        return {
            "endpoints": list(self.endpoints),
            "rate": self.rate,
            "delay_ns": self.delay_ns,
            "error_rate": self.error_rate,
        }


@dataclass(frozen=True)
class LinkFailure:
    """A spine's links, or one leaf-spine link, failing at start_ns.

    ``graceful`` takes the links down, so the routing and every leaf see it.
    ``silent`` leaves them up while a switch drops every packet arriving over
    them: the spine what arrives from the leaf under ``direction`` "up", the
    leaf what arrives from the spine under "down", both under "both". ``gray``
    gives the links an error rate or a lower rate in the same direction or
    directions: a loss is dropped by the receiving end, and a lower rate is
    the sending end's. A loss's ``loss_unit`` ``frame``, the default, drops
    every packet with ``error_rate``; ``byte`` drops a packet of s bytes with
    1 - (1 - r)^s, r being the rate per byte at which a full data frame is
    lost with ``error_rate`` (per_byte_rate), so that an acknowledgement or
    a trimmed header is lost about as much less often as it is smaller, as
    corruption failing a frame's check would lose it.

    With ``duration_ns`` the links are restored that long after each failure,
    and the failure happens ``repeat`` times, ``period_ns`` apart; without it
    the failure lasts to the end of the run. ``repeat`` and ``period_ns`` are
    None when the profile leaves them out.
    """

    model: str
    start_ns: int
    spine: int
    leaf: int | None
    error_rate: float | None
    rate: str | None
    duration_ns: int | None = None
    repeat: int | None = None
    period_ns: int | None = None
    # As the profile gives it; None when it leaves it out.
    direction: str | None = None
    # As the profile gives it; None when it leaves it out.
    loss_unit: str | None = None

    def effective_direction(self) -> str:
        """The direction the failure applies in: "up" for an unspecified
        silent failure, as one has always been written, and "both" for the
        others, which have always applied to the link as a whole."""
        if self.direction is not None:
            return self.direction
        return "up" if self.model == "silent" else "both"

    def occurrences(self) -> list[tuple[int, int | None]]:
        """Each occurrence's (start, end) in ns; the end is None when the
        failure has no duration and so never ends."""
        period = self.period_ns or 0
        return [
            (
                self.start_ns + i * period,
                (
                    None
                    if self.duration_ns is None
                    else self.start_ns + i * period + self.duration_ns
                ),
            )
            for i in range(self.repeat or 1)
        ]

    def overlaps(self, other: LinkFailure) -> bool:
        """Whether an occurrence of this failure and one of other share an
        instant, both taken as closed intervals."""
        return any(
            (end is None or other_start <= end)
            and (other_end is None or start <= other_end)
            for start, end in self.occurrences()
            for other_start, other_end in other.occurrences()
        )

    def leaves(self, leaf_count: int) -> range:
        return (
            range(leaf_count) if self.leaf is None else range(self.leaf, self.leaf + 1)
        )

    def per_byte_rate(self, frame_bytes: int) -> float:
        """The rate r per byte that loses a full data frame of frame_bytes
        with error_rate: 1 - (1 - error_rate)^(1 / frame_bytes)."""
        assert self.error_rate is not None
        return -expm1(log1p(-self.error_rate) / frame_bytes)

    def _action(self, frame_bytes: int) -> tuple[str, str]:
        """The LINK_FAILURE kind, and the value written after the two nodes:
        under loss_unit byte the rate per byte at a full data frame of
        frame_bytes."""
        if self.model == "graceful":
            return "down", ""
        if self.model == "silent":
            return "blackhole", ""
        if self.loss_unit == "byte":
            return "loss", " " + format(self.per_byte_rate(frame_bytes), ".17g")
        if self.error_rate is not None:
            return "loss", " " + format(self.error_rate, ".17g")
        return "rate", f" {self.rate}"

    def _ends(self, leaf: int, spine: int) -> list[tuple[int, int]]:
        """The node pairs written for one link, in the order written. A
        blackhole, and a loss or rate fault marked forward, applies from its
        first node to its second, so the leaf is named first for "up" and the
        spine first for "down". Under "both" a silent failure writes both
        orders, and the others name the leaf first and apply both ways."""
        direction = self.effective_direction()
        if direction == "up":
            return [(leaf, spine)]
        if direction == "down":
            return [(spine, leaf)]
        if self.model == "silent":
            return [(leaf, spine), (spine, leaf)]
        return [(leaf, spine)]

    def _backend_direction_token(self) -> str:
        """The direction token after a gray failure's LINK_FAILURE line, the
        only place one is written: " forward" for "up" and "down", which
        _ends turns into the order of the two nodes, and none for "both", the
        backend's default, so a failure without a direction is written as
        before. LINK_RESTORE takes no token."""
        if self.model == "gray" and self.effective_direction() != "both":
            return " forward"
        return ""

    def config_lines(self, index: ClosIndex, frame_bytes: int) -> list[str]:
        """Per affected link, per node pair _ends gives and per occurrence,
        one LINK_FAILURE line and, with a duration, one LINK_RESTORE line
        naming the two nodes in the same order, which the backend restores
        that failure by. frame_bytes is a full data frame's, as the backend
        checks under loss_unit byte, written after the direction."""
        kind, value = self._action(frame_bytes)
        token = self._backend_direction_token()
        if self.loss_unit == "byte":
            token += f" byte {frame_bytes}"
        lines = []
        for leaf in self.leaves(index.leaf_count):
            for a, b in self._ends(index.leaf(leaf), index.spine(self.spine)):
                link = f"{a} {b}"
                for start, end in self.occurrences():
                    lines.append(f"LINK_FAILURE {start} {kind} {link}{value}{token}\n")
                    if end is not None:
                        lines.append(f"LINK_RESTORE {end} {link}\n")
        return lines

    def manifest(self, frame_bytes: int) -> dict[str, Any]:
        """The failure as given; direction and loss_unit only when the
        profile sets them, so a manifest without them is unchanged, and under
        loss_unit byte the full data frame of frame_bytes the rate per byte
        is derived at and that rate."""
        recorded: dict[str, Any] = {
            "model": self.model,
            "start_ns": self.start_ns,
            "spine": self.spine,
            "leaf": self.leaf,
            "error_rate": self.error_rate,
            "rate": self.rate,
            "duration_ns": self.duration_ns,
            "repeat": self.repeat,
            "period_ns": self.period_ns,
        }
        if self.direction is not None:
            recorded["direction"] = self.direction
        if self.loss_unit is not None:
            recorded["loss_unit"] = self.loss_unit
        if self.loss_unit == "byte":
            recorded["reference_frame_bytes"] = frame_bytes
            recorded["per_byte_error_rate"] = self.per_byte_rate(frame_bytes)
        return recorded


def _direction(entry: dict[str, Any], field: str, model: str) -> str | None:
    """The failure's direction as given, None when absent."""
    if "direction" not in entry:
        return None
    direction = entry["direction"]
    if direction not in FAILURE_DIRECTIONS:
        raise ValueError(f"{field}.direction must be one of {list(FAILURE_DIRECTIONS)}")
    if model == "graceful" and direction != "both":
        raise ValueError(
            f"{field}.direction must be 'both' for a graceful failure, which "
            "takes the links down in both directions"
        )
    return direction


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _index(value: Any, field: str, bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < bound:
        raise ValueError(f"{field} must be an integer in [0, {bound})")
    return value


def _error_rate(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 < value <= 1.0
    ):
        raise ValueError(f"{field} must be a number in (0, 1]")
    return float(value)


def _rate(value: Any, field: str, allowed_bps: frozenset[int] | None) -> str:
    if not isinstance(value, str) or not value.endswith("Gbps"):
        raise ValueError(f"{field} must be a rate such as 200Gbps")
    try:
        bps = round(float(value[: -len("Gbps")]) * 1_000_000_000)
    except ValueError as error:
        raise ValueError(f"{field} must be a rate such as 200Gbps") from error
    if bps <= 0:
        raise ValueError(f"{field} must be a positive rate")
    if allowed_bps is not None and bps not in allowed_bps:
        raise ValueError(
            f"{field} must be a link speed the switch ECN table covers: "
            f"{sorted(allowed_bps)} bps"
        )
    return value


def load_link_overrides(
    document: dict[str, Any],
    links: Iterable[tuple[int, int]],
    clos: ClosIndex | None,
    allowed_bps: Iterable[int],
) -> tuple[LinkOverride, ...]:
    """Parse ``network.link_overrides`` against the fabric's links."""
    value = document.get("link_overrides", [])
    if not isinstance(value, list):
        raise ValueError("network.link_overrides must be an array")
    existing = {tuple(sorted(link)) for link in links}
    speeds = frozenset(allowed_bps)
    overrides: list[LinkOverride] = []
    named: set[tuple[int, int]] = set()
    settings = {"rate", "delay_ns", "error_rate"}
    for position, entry in enumerate(value):
        field = f"network.link_overrides[{position}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{field} must be an object")
        address = set(entry) - settings
        if address == {"endpoints"}:
            ends = entry["endpoints"]
            if (
                not isinstance(ends, list)
                or len(ends) != 2
                or any(isinstance(e, bool) or not isinstance(e, int) for e in ends)
            ):
                raise ValueError(f"{field}.endpoints must be two node ids")
            endpoints = (ends[0], ends[1])
        elif address == {"leaf", "spine"}:
            if clos is None:
                raise ValueError(
                    f"{field} names a leaf and spine, which a ring has not"
                )
            endpoints = (
                clos.leaf(_index(entry["leaf"], f"{field}.leaf", clos.leaf_count)),
                clos.spine(_index(entry["spine"], f"{field}.spine", clos.spine_count)),
            )
        else:
            raise ValueError(
                f"{field} must name 'endpoints' or 'leaf' and 'spine', and may set "
                f"{sorted(settings)}"
            )
        link = tuple(sorted(endpoints))
        if link not in existing:
            raise ValueError(f"{field} names {list(endpoints)}, which no link joins")
        if link in named:
            raise ValueError(f"{field} names a link an earlier override names")
        named.add(link)
        if not settings & set(entry):
            raise ValueError(f"{field} must set one of {sorted(settings)}")
        delay_ns = entry.get("delay_ns")
        if delay_ns is not None and (
            isinstance(delay_ns, bool) or not isinstance(delay_ns, int) or delay_ns <= 0
        ):
            raise ValueError(f"{field}.delay_ns must be a positive integer")
        overrides.append(
            LinkOverride(
                endpoints=endpoints,
                rate=(
                    _rate(entry["rate"], f"{field}.rate", speeds)
                    if "rate" in entry
                    else None
                ),
                delay_ns=delay_ns,
                error_rate=(
                    _error_rate(entry["error_rate"], f"{field}.error_rate")
                    if "error_rate" in entry
                    else None
                ),
            )
        )
    return tuple(overrides)


def load_link_failures(
    document: dict[str, Any], clos: ClosIndex | None
) -> tuple[LinkFailure, ...]:
    """Parse ``network.link_failures`` against the spines and leaves built."""
    value = document.get("link_failures", [])
    if not isinstance(value, list):
        raise ValueError("network.link_failures must be an array")
    if not value:
        return ()
    if clos is None:
        raise ValueError("network.link_failures needs a Clos fabric")
    failures: list[LinkFailure] = []
    failed: dict[tuple[int, int], list[LinkFailure]] = {}
    down: dict[int, set[int]] = {}
    for position, entry in enumerate(value):
        field = f"network.link_failures[{position}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{field} must be an object")
        model = entry.get("model")
        if model not in FAILURE_MODELS:
            raise ValueError(f"{field}.model must be one of {list(FAILURE_MODELS)}")
        allowed = {"model", "start_ns", "spine", "leaf", "direction"}
        if model == "gray":
            allowed |= {"error_rate", "rate"}
            if len({"error_rate", "rate"} & set(entry)) != 1:
                raise ValueError(f"{field} is gray and sets one of error_rate and rate")
        if "loss_unit" in entry:
            if model != "gray" or "error_rate" not in entry:
                raise ValueError(
                    f"{field}.loss_unit is read only by a gray failure that sets "
                    "error_rate"
                )
            if entry["loss_unit"] not in LOSS_UNITS:
                raise ValueError(f"{field}.loss_unit must be one of {list(LOSS_UNITS)}")
            allowed.add("loss_unit")
        if model != "graceful":
            allowed.add("duration_ns")
            if "duration_ns" in entry:
                allowed |= {"repeat", "period_ns"}
        unknown = set(entry) - allowed
        if unknown or not {"model", "start_ns", "spine"} <= set(entry):
            raise ValueError(
                f"{field} must contain model, start_ns and spine and may contain "
                f"{sorted(allowed - {'model', 'start_ns', 'spine'})}"
            )
        start_ns = entry["start_ns"]
        if isinstance(start_ns, bool) or not isinstance(start_ns, int) or start_ns < 0:
            raise ValueError(f"{field}.start_ns must be a nonnegative integer")
        duration_ns = (
            _positive(entry["duration_ns"], f"{field}.duration_ns")
            if "duration_ns" in entry
            else None
        )
        repeat = (
            _positive(entry["repeat"], f"{field}.repeat") if "repeat" in entry else None
        )
        period_ns = (
            _positive(entry["period_ns"], f"{field}.period_ns")
            if "period_ns" in entry
            else None
        )
        if (period_ns is not None) != (repeat is not None and repeat > 1):
            raise ValueError(f"{field}.period_ns is required exactly when repeat > 1")
        if period_ns is not None and duration_ns is not None and (
            period_ns <= duration_ns
        ):
            raise ValueError(f"{field}.period_ns must be greater than duration_ns")
        failure = LinkFailure(
            model=model,
            start_ns=start_ns,
            spine=_index(entry["spine"], f"{field}.spine", clos.spine_count),
            leaf=(
                _index(entry["leaf"], f"{field}.leaf", clos.leaf_count)
                if "leaf" in entry
                else None
            ),
            error_rate=(
                _error_rate(entry["error_rate"], f"{field}.error_rate")
                if "error_rate" in entry
                else None
            ),
            rate=_rate(entry["rate"], f"{field}.rate", None)
            if "rate" in entry
            else None,
            duration_ns=duration_ns,
            repeat=repeat,
            period_ns=period_ns,
            direction=_direction(entry, field, model),
            loss_unit=entry.get("loss_unit"),
        )
        for leaf in failure.leaves(clos.leaf_count):
            earlier = failed.setdefault((leaf, failure.spine), [])
            if any(failure.overlaps(other) for other in earlier):
                raise ValueError(
                    f"{field} fails a link while an earlier failure of it lasts"
                )
            earlier.append(failure)
            if model == "graceful":
                down.setdefault(leaf, set()).add(failure.spine)
        failures.append(failure)
    # A leaf with every uplink down is cut off and no flow to it can finish.
    if any(len(spines) == clos.spine_count for spines in down.values()):
        raise ValueError(
            "network.link_failures must leave every leaf a link to a spine"
        )
    return tuple(failures)
