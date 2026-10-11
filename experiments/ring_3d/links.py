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
from math import isfinite
from typing import Any

FAILURE_MODELS = ("graceful", "silent", "gray")


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
    ``silent`` leaves them up while the spine drops every packet arriving from
    the leaves. ``gray`` gives the links an error rate or a lower rate.

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

    def _action(self) -> tuple[str, str]:
        """The LINK_FAILURE kind, and the value written after the two nodes."""
        if self.model == "graceful":
            return "down", ""
        if self.model == "silent":
            return "blackhole", ""
        if self.error_rate is not None:
            return "loss", " " + format(self.error_rate, ".17g")
        return "rate", f" {self.rate}"

    def config_lines(self, index: ClosIndex) -> list[str]:
        """Per affected link and per occurrence, one LINK_FAILURE line and,
        with a duration, one LINK_RESTORE line; the leaf named before the
        spine, so that a silent spine drops what arrives from the leaf."""
        kind, value = self._action()
        lines = []
        for leaf in self.leaves(index.leaf_count):
            link = f"{index.leaf(leaf)} {index.spine(self.spine)}"
            for start, end in self.occurrences():
                lines.append(f"LINK_FAILURE {start} {kind} {link}{value}\n")
                if end is not None:
                    lines.append(f"LINK_RESTORE {end} {link}\n")
        return lines

    def manifest(self) -> dict[str, Any]:
        return {
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
        allowed = {"model", "start_ns", "spine", "leaf"}
        if model == "gray":
            allowed |= {"error_rate", "rate"}
            if len({"error_rate", "rate"} & set(entry)) != 1:
                raise ValueError(f"{field} is gray and sets one of error_rate and rate")
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
