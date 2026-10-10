"""Per-link settings of a generated fabric.

``network.link_overrides`` gives chosen links their own rate, delay or error
rate for the whole run, in the topology file. A Clos link is named by its leaf
and spine index, the spine index counting the spines that are built; any link
may also be named by the node ids at its two ends.

This module imports nothing else from the package, so ``topology`` can import
it without a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any


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
                raise ValueError(f"{field} names a leaf and spine, which a ring has not")
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
