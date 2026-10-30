"""Per-link settings and link failures of a generated fabric.

``network.link_overrides`` gives chosen links their own rate, delay or error
rate for the whole run, in the topology file. ``network.link_failures`` changes
a spine's links, one leaf-spine link, or one host's link to its leaf, at a time
in the run, through ``LINK_FAILURE`` lines in the network configuration, and,
for a failure with a duration, restores them through ``LINK_RESTORE`` lines. A
Clos link is named by its leaf and spine index, the spine index counting the
spines that are built, or a host's link by the host's index; any link may also
be named by the node ids at its two ends. A failure with ``draw`` instead fails
a fraction of the leaf-spine link directions, drawn by its seed
(draw_link_directions).

This module imports nothing else from the package, so ``topology`` can import
it without a cycle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from math import expm1, isfinite, log1p
from typing import Any

FAILURE_MODELS = ("graceful", "silent", "gray")
# What a gray loss failure's error_rate is the rate of: each packet, frame,
# whatever its size, or each byte, byte, so that a packet is lost by its
# size, the rate derived so that a full data frame keeps error_rate.
LOSS_UNITS = ("frame", "byte")
# A failure's direction on its leaf-spine link: "up" is what the leaf sends
# the spine, "down" what the spine sends the leaf, "both" the two; on a host's
# link, "up" is what the host sends its leaf and "down" what the leaf sends the
# host.
FAILURE_DIRECTIONS = ("up", "down", "both")
# The models a drawn failure (draw) can take: a loss, a lower rate or a
# black hole in one direction of each drawn link leaves its routes up.
DRAWN_MODELS = ("gray", "silent")


@dataclass(frozen=True)
class ClosIndex:
    """Node ids of a Clos built with hosts first, then leaves, then spines."""

    host_count: int
    leaf_count: int
    spine_count: int

    def host(self, index: int) -> int:
        return index

    def leaf(self, index: int) -> int:
        return self.host_count + index

    def spine(self, index: int) -> int:
        return self.host_count + self.leaf_count + index

    def host_leaf(self, host: int) -> int:
        """The node id of the leaf host is attached to: hosts are attached
        to the leaves in order, the same number to each."""
        return self.leaf(host // (self.host_count // self.leaf_count))


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
    """A spine's links, one leaf-spine link, or one host's link to its leaf,
    failing at start_ns.

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

    With ``host`` the failure is of that host's link to its leaf, and
    ``spine`` and ``leaf`` are None: ``direction`` "up" is what the host
    sends the leaf and "down" what the leaf sends the host. Only the leaf
    can drop what arrives as ``silent``, so a silent host link fails "up"
    alone, and a ``graceful`` one, which would cut the host off, is refused.
    """

    model: str
    start_ns: int
    spine: int | None
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
    # The host whose link to its leaf fails; None for a leaf-spine failure.
    host: int | None = None
    # The draw (fraction, seed) that chose this link and direction; None for
    # a failure the profile names.
    draw: tuple[float, int] | None = None

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
        for a, b in self._node_pairs(index):
            link = f"{a} {b}"
            for start, end in self.occurrences():
                lines.append(f"LINK_FAILURE {start} {kind} {link}{value}{token}\n")
                if end is not None:
                    lines.append(f"LINK_RESTORE {end} {link}\n")
        return lines

    def _node_pairs(self, index: ClosIndex) -> list[tuple[int, int]]:
        """The node pairs _ends gives for every affected link: the host's
        link, its host's end taking a leaf's place and its leaf's a spine's,
        or each leaf-spine link."""
        if self.host is not None:
            return self._ends(index.host(self.host), index.host_leaf(self.host))
        assert self.spine is not None
        return [
            pair
            for leaf in self.leaves(index.leaf_count)
            for pair in self._ends(index.leaf(leaf), index.spine(self.spine))
        ]

    def manifest(self, frame_bytes: int) -> dict[str, Any]:
        """The failure as given; direction and loss_unit only when the
        profile sets them, so a manifest without them is unchanged, and under
        loss_unit byte the full data frame of frame_bytes the rate per byte
        is derived at and that rate."""
        recorded: dict[str, Any] = {
            "model": self.model,
            "start_ns": self.start_ns,
            **(
                {"spine": self.spine, "leaf": self.leaf}
                if self.host is None
                else {"host": self.host}
            ),
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
        if self.draw is not None:
            recorded["draw"] = {"fraction": self.draw[0], "seed": self.draw[1]}
        return recorded


def draw_link_directions(
    fraction: float, seed: int, leaf_count: int, spine_count: int
) -> list[tuple[int, int, str]]:
    """The leaf-spine link directions a draw fails, as (leaf, spine,
    direction) in leaf and spine order, "up" and "down" merged into "both".

    Of the 2 * leaf_count * spine_count directions, round(fraction * that)
    (half up) are drawn: those whose SHA-256 of "<seed> <leaf> <spine>
    <direction>" is least, so that the draw depends on the seed and the
    fabric alone and the same fraction fails at every spine count.
    """
    directions = [
        (leaf, spine, direction)
        for leaf in range(leaf_count)
        for spine in range(spine_count)
        for direction in ("up", "down")
    ]
    count = int(
        (Decimal(repr(fraction)) * len(directions)).to_integral_value(ROUND_HALF_UP)
    )

    def rank(entry: tuple[int, int, str]) -> bytes:
        leaf, spine, direction = entry
        return hashlib.sha256(f"{seed} {leaf} {spine} {direction}".encode()).digest()

    drawn = sorted(directions, key=rank)[:count]
    links: dict[tuple[int, int], set[str]] = {}
    for leaf, spine, direction in drawn:
        links.setdefault((leaf, spine), set()).add(direction)
    return [
        (leaf, spine, "both" if len(found) == 2 else next(iter(found)))
        for (leaf, spine), found in sorted(links.items())
    ]


def _drawn_failures(
    entry: dict[str, Any],
    field: str,
    clos: ClosIndex,
    start_ns: int,
    duration_ns: int | None,
    repeat: int | None,
    period_ns: int | None,
) -> list[LinkFailure]:
    """A drawn failure's links: one failure per link draw_link_directions
    gives, in its direction, at the entry's model and times."""
    draw = entry["draw"]
    if not isinstance(draw, dict) or set(draw) != {"fraction", "seed"}:
        raise ValueError(f"{field}.draw must contain exactly fraction and seed")
    fraction = draw["fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not isfinite(fraction)
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError(f"{field}.draw.fraction must be a number in (0, 1]")
    seed = draw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{field}.draw.seed must be a nonnegative integer")
    drawn = draw_link_directions(
        float(fraction), seed, clos.leaf_count, clos.spine_count
    )
    if not drawn:
        raise ValueError(
            f"{field}.draw.fraction draws no link direction of the "
            f"{2 * clos.leaf_count * clos.spine_count}"
        )
    return [
        LinkFailure(
            model=entry["model"],
            start_ns=start_ns,
            spine=spine,
            leaf=leaf,
            error_rate=(
                _error_rate(entry["error_rate"], f"{field}.error_rate")
                if "error_rate" in entry
                else None
            ),
            rate=(
                _rate(entry["rate"], f"{field}.rate", None) if "rate" in entry else None
            ),
            duration_ns=duration_ns,
            repeat=repeat,
            period_ns=period_ns,
            direction=direction,
            loss_unit=entry.get("loss_unit"),
            draw=(float(fraction), seed),
        )
        for leaf, spine, direction in drawn
    ]


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
    """Parse ``network.link_failures`` against the hosts, spines and leaves
    built."""
    value = document.get("link_failures", [])
    if not isinstance(value, list):
        raise ValueError("network.link_failures must be an array")
    if not value:
        return ()
    if clos is None:
        raise ValueError("network.link_failures needs a Clos fabric")
    failures: list[LinkFailure] = []
    # Per link, the failures of it: a leaf-spine link by (leaf, spine), a
    # host's link by ("host", host).
    failed: dict[tuple[Any, int], list[LinkFailure]] = {}
    down: dict[int, set[int]] = {}

    def record(failure: LinkFailure, field: str) -> None:
        """Take failure, refused where it fails a link while an earlier
        failure of the link lasts."""
        links: list[tuple[Any, int]] = (
            [("host", failure.host)]
            if failure.host is not None
            else [(leaf, failure.spine) for leaf in failure.leaves(clos.leaf_count)]
        )
        for leaf, spine in links:
            earlier = failed.setdefault((leaf, spine), [])
            if any(failure.overlaps(other) for other in earlier):
                raise ValueError(
                    f"{field} fails a link while an earlier failure of it lasts"
                )
            earlier.append(failure)
            if failure.model == "graceful":
                down.setdefault(leaf, set()).add(spine)
        failures.append(failure)
    for position, entry in enumerate(value):
        field = f"network.link_failures[{position}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{field} must be an object")
        model = entry.get("model")
        if model not in FAILURE_MODELS:
            raise ValueError(f"{field}.model must be one of {list(FAILURE_MODELS)}")
        # A host's link is named by the host alone, and drawn links by the
        # draw alone, which also chooses their directions.
        if "host" in entry:
            named: tuple[str, ...] = ("host",)
        elif "draw" in entry:
            named = ("draw",)
        else:
            named = ("spine",)
        allowed = {"model", "start_ns", *named}
        if "draw" not in entry:
            allowed.add("direction")
        if "host" not in entry and "draw" not in entry:
            allowed.add("leaf")
        if "draw" in entry and model not in DRAWN_MODELS:
            raise ValueError(
                f"{field} draws its links and must be one of {list(DRAWN_MODELS)}, "
                "which leave the drawn links' routes up"
            )
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
        if unknown or not {"model", "start_ns", *named} <= set(entry):
            raise ValueError(
                f"{field} must contain model, start_ns and spine, host or draw, "
                f"and may contain "
                f"{sorted(allowed - {'model', 'start_ns', *named})}"
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
        host = (
            _index(entry["host"], f"{field}.host", clos.host_count)
            if "host" in entry
            else None
        )
        direction = _direction(entry, field, model)
        if host is not None and model == "graceful":
            raise ValueError(
                f"{field} is graceful, which would cut host {host} off from "
                "its leaf; a host's link fails silent or gray"
            )
        if host is not None and model == "silent" and direction not in (None, "up"):
            raise ValueError(
                f"{field}.direction must be 'up' for a silent host link: only "
                "the leaf, a switch, drops what arrives over it"
            )
        if "draw" in entry:
            for drawn in _drawn_failures(
                entry, field, clos, start_ns, duration_ns, repeat, period_ns
            ):
                record(drawn, field)
            continue
        failure = LinkFailure(
            model=model,
            start_ns=start_ns,
            spine=(
                None
                if host is not None
                else _index(entry["spine"], f"{field}.spine", clos.spine_count)
            ),
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
            direction=direction,
            loss_unit=entry.get("loss_unit"),
            host=host,
        )
        record(failure, field)
    # A leaf with every uplink down is cut off and no flow to it can finish.
    if any(len(spines) == clos.spine_count for spines in down.values()):
        raise ValueError(
            "network.link_failures must leave every leaf a link to a spine"
        )
    return tuple(failures)
