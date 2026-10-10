#!/usr/bin/env python3
"""Validate and summarize telemetry emitted by the 3D Ring experiment."""

from __future__ import annotations

import argparse
import csv
import json
from array import array
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any, Final

import numpy

# A 64-rank run emits one flow row per queue pair, which reaches millions of
# rows and gigabytes of CSV. Nothing in this module may retain a parsed row:
# every reported figure is a counter, a group-by over a bounded key space
# (training step, parallelism domain, flow kind), or a duration column held as
# a packed integer array. Materializing the rows costs ~2.9 KB each, which
# exceeds the memory of the runner that has to analyze them.


def _as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid integer field {key!r} in telemetry") from error


def _as_bool(row: dict[str, str], key: str) -> bool:
    value = row.get(key)
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid boolean field {key!r} in telemetry")


def _terminal_outcome(row: dict[str, str]) -> str:
    """Read the explicit terminal outcome for an issued transport flow."""
    outcome = row.get("terminal_outcome")
    if outcome not in {"completed", "failed"}:
        raise ValueError(f"invalid terminal outcome {outcome!r} in telemetry")
    return outcome


def _optional_nonnegative_int(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value is None or value == "":
        return 0
    parsed = _as_int(row, key)
    if parsed < 0:
        raise ValueError(f"telemetry field {key!r} must be nonnegative")
    return parsed


def _optional_bool(row: dict[str, str], key: str) -> bool:
    """Absence is false: a column a run never wrote asserts nothing."""
    value = row.get(key)
    if value is None or value == "":
        return False
    return _as_bool(row, key)


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    """Yield telemetry rows without materializing the file.

    The file stays open for the life of the generator, so consume it within
    the scope that created it rather than storing it for later.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing telemetry file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Read a *bounded* telemetry file whole. Never use this for flow events."""
    return list(iter_csv(path))


def _iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield numbered lines without holding the whole file in memory."""
    with path.open(encoding="utf-8") as handle:
        yield from enumerate(handle, start=1)


def _training_step_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _timing_statistics(values: Sequence[int]) -> dict[str, int | None]:
    """Return nearest-rank latency quantiles in nanoseconds.

    Accepts a packed `array("q")` as well as a list, and sorts through numpy so
    a column of millions of durations never becomes a list of Python integers.
    """
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "min_ns": None,
            "p50_ns": None,
            "p95_ns": None,
            "p99_ns": None,
            "max_ns": None,
        }
    ordered = numpy.sort(numpy.asarray(values, dtype=numpy.int64))

    def percentile(percent: int) -> int:
        return int(ordered[ceil(count * percent / 100) - 1])

    return {
        "count": count,
        "min_ns": int(ordered[0]),
        "p50_ns": percentile(50),
        "p95_ns": percentile(95),
        "p99_ns": percentile(99),
        "max_ns": int(ordered[-1]),
    }


def _flow_duration_ns(row: dict[str, str]) -> int:
    duration = _as_int(row, "end_time_ns") - _as_int(row, "start_time_ns")
    if duration < 0:
        raise ValueError("flow completion time must not be negative")
    return duration


def _durations() -> defaultdict[Any, array]:
    """Free monoid of durations per group key: identity empty, op append."""
    return defaultdict(lambda: array("q"))


def _timing_by_group(durations: dict[Any, array]) -> dict[str, Any]:
    return {
        name: _timing_statistics(values)
        for name, values in sorted(durations.items(), key=lambda entry: entry[0])
    }


def _timing_by_domain_and_kind(durations: dict[Any, array]) -> dict[str, Any]:
    nested: dict[str, dict[str, array]] = defaultdict(dict)
    for (domain, kind), values in durations.items():
        nested[domain][kind] = values
    return {
        domain: _timing_by_group(kinds)
        for domain, kinds in sorted(nested.items(), key=lambda entry: entry[0])
    }


@dataclass(slots=True)
class _Traffic:
    """Componentwise monoid on (count, logical bytes, physical bytes)."""

    flow_count: int = 0
    logical_bytes: int = 0
    physical_bytes: int = 0

    def add(self, logical_bytes: int, physical_bytes: int) -> None:
        self.flow_count += 1
        self.logical_bytes += logical_bytes
        self.physical_bytes += physical_bytes

    def summary(self) -> dict[str, int]:
        return {
            "flow_count": self.flow_count,
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
        }


@dataclass(frozen=True, slots=True)
class _Window:
    """A closed interval. `join` is the semilattice operation; the identity is
    `None`, so absence of a window is a state rather than a pair of nulls."""

    start_ns: int
    end_ns: int

    def join(self, start_ns: int, end_ns: int) -> _Window:
        return _Window(min(self.start_ns, start_ns), max(self.end_ns, end_ns))

    @property
    def span_ns(self) -> int:
        return self.end_ns - self.start_ns


def _timing_by_step(durations: dict[Any, array]) -> dict[str, Any]:
    """Group timings by training step, ordered numerically where possible."""
    return {
        step: _timing_statistics(values)
        for step, values in sorted(
            durations.items(), key=lambda entry: _training_step_sort_key(entry[0])
        )
    }


@dataclass(slots=True)
class _Operation:
    """One logical collective: the join of its ranks' intervals and how many
    ranks reported it. The contributing rows are never retained."""

    window: _Window
    rank_count: int = 1

    def record(self, start_ns: int, end_ns: int) -> None:
        self.window = self.window.join(start_ns, end_ns)
        self.rank_count += 1



def _summarize_rank_completions(
    completion_rows: list[dict[str, str]], expected_rank_count: int | None
) -> dict[str, int | str]:
    seen_ranks: set[int] = set()
    for row in completion_rows:
        rank = _as_int(row, "rank")
        completion_time_ns = _as_int(row, "completion_time_ns")
        if rank < 0 or completion_time_ns < 0:
            raise ValueError("rank completion telemetry must be nonnegative")
        if rank in seen_ranks:
            raise ValueError("rank completion telemetry contains a duplicate rank")
        seen_ranks.add(rank)

    result: dict[str, int | str] = {
        "status": "not_checked" if expected_rank_count is None else "verified",
        "recorded_rank_count": len(seen_ranks),
    }
    if expected_rank_count is not None:
        if expected_rank_count <= 0:
            raise ValueError("expected rank count must be positive")
        if seen_ranks != set(range(expected_rank_count)):
            raise ValueError(
                "rank completion telemetry does not cover every expected rank"
            )
        result["expected_rank_count"] = expected_rank_count
    return result


def _collective_duration_ns(row: dict[str, str]) -> int:
    duration = _as_int(row, "end_time_ns") - _as_int(row, "start_time_ns")
    if duration < 0:
        raise ValueError("collective completion time must not be negative")
    return duration


def _summarize_collectives(rows: Iterator[dict[str, str]]) -> dict[str, Any]:
    """Fold per-rank collective events into per-rank and whole-operation timing.

    A logical collective is identified by (domain, type, step, workload node);
    its all-rank span is the join of every contributing rank's interval, so the
    rows themselves are not retained. Nesting by domain and type reuses the
    flow-side grouping rather than restating it.
    """
    rank_event_count = 0
    per_rank_durations = array("q")
    per_rank_by_domain = _durations()
    per_rank_by_domain_and_type = _durations()
    operations: dict[tuple[str, str, int, int], _Operation] = {}
    seen_rank_events: set[tuple[str, str, int, int, int]] = set()

    for row in rows:
        rank_event_count += 1
        domain = row.get("parallelism_domain") or "unknown"
        collective_type = row.get("collective_type") or "unknown"
        training_step = _as_int(row, "training_step")
        node_id = _as_int(row, "workload_node_id")
        rank = _as_int(row, "rank")
        if rank < 0 or _as_int(row, "logical_bytes") < 0:
            raise ValueError(
                "collective telemetry rank and logical bytes must be nonnegative"
            )
        duration = _collective_duration_ns(row)
        event_key = (domain, collective_type, training_step, node_id, rank)
        if event_key in seen_rank_events:
            raise ValueError(
                "collective telemetry contains a duplicate rank completion"
            )
        seen_rank_events.add(event_key)
        per_rank_durations.append(duration)
        per_rank_by_domain[domain].append(duration)
        per_rank_by_domain_and_type[(domain, collective_type)].append(duration)

        start_time_ns = _as_int(row, "start_time_ns")
        end_time_ns = _as_int(row, "end_time_ns")
        operation_key = (domain, collective_type, training_step, node_id)
        operation = operations.get(operation_key)
        if operation is None:
            operations[operation_key] = _Operation(_Window(start_time_ns, end_time_ns))
        else:
            operation.record(start_time_ns, end_time_ns)

    operation_spans = array("q")
    spans_by_domain = _durations()
    spans_by_domain_and_type = _durations()
    spans_by_step = _durations()
    dp_all_reduce_span_by_step: dict[str, int] = {}
    rank_counts = array("q")
    for (domain, collective_type, training_step, _node), operation in operations.items():
        span = operation.window.span_ns
        operation_spans.append(span)
        spans_by_domain[domain].append(span)
        spans_by_domain_and_type[(domain, collective_type)].append(span)
        spans_by_step[str(training_step)].append(span)
        if domain == "dp" and collective_type == "all_reduce":
            step = str(training_step)
            # A step issues one DP All-Reduce per workload node; the step is
            # held for as long as its worst one, so the step reduces by max.
            dp_all_reduce_span_by_step[step] = max(
                span, dp_all_reduce_span_by_step.get(step, 0)
            )
        rank_counts.append(operation.rank_count)

    return {
        "status": "available",
        "rank_event_count": rank_event_count,
        "logical_collective_count": len(operations),
        "operation_rank_count": _timing_statistics(rank_counts),
        "per_rank_completion_time_ns": {
            "all": _timing_statistics(per_rank_durations),
            "by_parallelism_domain": _timing_by_group(per_rank_by_domain),
            "by_parallelism_domain_and_collective_type": _timing_by_domain_and_kind(
                per_rank_by_domain_and_type
            ),
        },
        "all_rank_operation_span_ns": {
            "all": _timing_statistics(operation_spans),
            "by_parallelism_domain": _timing_by_group(spans_by_domain),
            "by_parallelism_domain_and_collective_type": _timing_by_domain_and_kind(
                spans_by_domain_and_type
            ),
            "by_training_step": _timing_by_step(spans_by_step),
        },
        "dp_all_reduce_span_ns_by_training_step": {
            step: dp_all_reduce_span_by_step[step]
            for step in sorted(
                dp_all_reduce_span_by_step, key=_training_step_sort_key
            )
        },
    }


def _clr_steps(manifest: dict[str, Any] | None) -> frozenset[str] | None:
    """Critical steps from the mask the run actually used, as strings.

    The mask, not the schedule metadata: an explicit profile schedule and a
    generated proxy both end up in `clr_mask.csv`, and that file is what the
    simulator read.
    """
    if manifest is None:
        return None
    mask_path = manifest.get("clr_mask")
    if not isinstance(mask_path, str) or not Path(mask_path).is_file():
        return None
    return frozenset(
        row["step_id"] for row in iter_csv(Path(mask_path)) if row["is_clr"] == "1"
    )


def _scaled_policy(policy: dict[str, Any]) -> tuple[int, int, int]:
    """The manifest's decision scale and its two scaled thresholds.

    Only the integers are read. Deriving them from the probabilities beside
    them would put the rounding back in two places, which is the divergence
    this removes; the simulator refuses a file whose integers disagree with
    llround of those probabilities, so one number reaches all three.
    """
    scale = policy["decision_scale"]
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("decision_scale must be a positive integer")
    thresholds = []
    for name in ("p_low_threshold", "p_high_threshold"):
        scaled = policy[name]
        if isinstance(scaled, bool) or not isinstance(scaled, int):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= scaled <= scale:
            raise ValueError(f"{name} must be in [0, {scale}]")
        thresholds.append(scaled)
    return scale, thresholds[0], thresholds[1]


def _forgiven_by_step(
    cells: dict[tuple[str, str], dict[str, int]],
) -> dict[str, int]:
    """Sum forgiven bytes over receiving ranks, one pass over the cells."""
    totals: dict[str, int] = defaultdict(int)
    for (_dst, step), cell in cells.items():
        if cell["forgiven_bytes"]:
            totals[step] += cell["forgiven_bytes"]
    return {
        step: totals[step]
        for step in sorted(totals, key=_training_step_sort_key)
    }


_FORGIVING_DOMAINS: Final = frozenset({"recovery", "recovery_exempt"})
"""The domains the budget law binds. Admission sheds by a per-flow hash draw
rather than a per-cell budget, so the law caps nothing there."""


def _shedding_domain(manifest: dict[str, Any] | None) -> Any:
    """The domain the run declared, or None when its manifest declares none.

    One reader for that field, because the law's own result and the refusal
    acting on it must agree about which arm is forgiving.
    """
    policy = (manifest or {}).get("selection_policy")
    return policy.get("domain") if isinstance(policy, dict) else None


def _delivered_share(cell: dict[str, int]) -> Fraction:
    """The share of one cell's eligible bytes its receiving rank was left.

    Exact rather than floating, because the simulator enforced this share in
    integers and a float ratio of two byte counts can put a boundary cell on
    the wrong side of it. The caller eliminates cells with no eligible bytes,
    which are owed nothing and so have no share.
    """
    return Fraction(
        cell["eligible_bytes"] - cell["shed_bytes"] - cell["forgiven_bytes"],
        cell["eligible_bytes"],
    )


def _check_ledger_law(
    cells: dict[tuple[str, str], dict[str, int]],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify shed + forgiven <= p(step) * eligible for every (rank, step).

    The recovery domain's safety property, checkable from the telemetry
    alone: the flow rows carry eligible, shed, and forgiven bytes, and the
    CLR mask decides which threshold each step is held to.

    The inequality is the simulator's, byte for byte: integer arithmetic
    over the scaled thresholds the manifest carries, never a float product,
    so a cell landing exactly on the boundary cannot be spent in ns-3 and
    reported violated here. The simulator refuses a manifest whose scaled
    integers disagree with the probabilities beside them.

    The law caps nothing in the admission domain, where shedding is a
    per-flow hash draw rather than a per-cell budget, so a small cell can
    exceed the rate by chance alone. The result is tagged with the domain
    and reported not_applicable outside recovery.
    """
    if not cells:
        return {"status": "no_eligible_traffic"}
    policy = (manifest or {}).get("selection_policy")
    clr_steps = _clr_steps(manifest)
    if not isinstance(policy, dict) or clr_steps is None:
        return {
            "status": "not_available",
            "reason": (
                "the run manifest carries no readable selection policy "
                "and CLR mask"
            ),
            "cell_count": len(cells),
        }
    domain = _shedding_domain(manifest)
    if domain not in _FORGIVING_DOMAINS:
        return {
            "status": "not_applicable",
            "domain": domain,
            "cell_count": len(cells),
        }
    try:
        scale, low, high = _scaled_policy(policy)
    except (KeyError, TypeError, ValueError):
        return {
            "status": "not_available",
            "reason": (
                "the selection policy carries no readable decision scale "
                "and thresholds"
            ),
            "domain": domain,
            "cell_count": len(cells),
        }
    # Recovery sheds nothing at admission: evaluate_shedding returns before
    # the shed decision for this domain, so the whole budget is forgiveness's.
    # A shed byte here means the arm ran a domain its manifest denies.
    shed_cells = [
        {"dst": dst, "training_step": step, **cell}
        for (dst, step), cell in sorted(cells.items())
        if cell["shed_bytes"]
    ]
    violations = [
        {
            "dst": dst,
            "training_step": step,
            "threshold": low if step in clr_steps else high,
            **cell,
        }
        for (dst, step), cell in sorted(cells.items())
        if (cell["shed_bytes"] + cell["forgiven_bytes"]) * scale
        > cell["eligible_bytes"] * (low if step in clr_steps else high)
    ]
    # The contract as one number, and the cell that decides it: every
    # receiving rank was left at least this share of every step it was owed.
    shares = [
        (_delivered_share(cell), dst, step, cell)
        for (dst, step), cell in sorted(cells.items())
        if cell["eligible_bytes"]
    ]
    worst = min(shares, key=lambda entry: entry[0], default=None)
    return {
        "status": "violated" if violations or shed_cells else "verified",
        "domain": domain,
        "decision_scale": scale,
        "cell_count": len(cells),
        "min_delivered_share": float(worst[0]) if worst is not None else None,
        "worst_cell": (
            {"dst": worst[1], "training_step": worst[2], **worst[3]}
            if worst is not None
            else None
        ),
        "forgiven_cell_count": sum(
            1 for cell in cells.values() if cell["forgiven_bytes"]
        ),
        # Bounded on purpose: a violated arm is rejected, and the first few
        # cells name the rank and step to look at.
        "violations": violations[:10],
        "violation_count": len(violations),
        "admission_shed_cells": shed_cells[:10],
        "admission_shed_cell_count": len(shed_cells),
    }


def _require_lawful_ledger(
    law: dict[str, Any], manifest: dict[str, Any] | None
) -> None:
    """Refuse a forgiving arm whose contract is broken or unverifiable.

    The contract is that every receiving rank keeps at least `1 - p(step)` of
    what its step owes it, so an arm that broke it is not a result, and an arm
    whose mask or policy cannot be read is not one either. Admission keeps its
    `not_applicable` and a run with no DP payload keeps its
    `no_eligible_traffic`, because neither one made the promise.
    """
    if _shedding_domain(manifest) not in _FORGIVING_DOMAINS:
        return
    status = law["status"]
    if status == "violated":
        # Either list is evidence of the same broken budget, and the law sets
        # this status only when one of them is nonempty.
        first = (law["violations"] or law["admission_shed_cells"])[0]
        raise ValueError(
            f"forgiving run broke the ledger law, status {status}, "
            f"first offending cell {first}"
        )
    if status == "not_available":
        raise ValueError(
            f"forgiving run cannot verify the ledger law, status {status}, "
            f"because {law['reason']}"
        )


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"invalid run manifest: {path}")
    return manifest


def _models_no_loss_mechanism(manifest: dict[str, Any] | None) -> bool:
    if manifest is None:
        return False
    if any(
        bool((manifest.get(key) or {}).get("enabled"))
        for key in ("data_plane_loss", "transport_recovery", "packet_trimming")
    ):
        return False
    # A best-effort fabric drops on buffer rejection, so only a lossless
    # flow-controlled one leaves nothing that can lose or reorder a packet.
    fabric = manifest.get("fabric")
    return isinstance(fabric, dict) and bool(fabric.get("pfc_enabled"))


def _flow_control_regime(manifest: dict[str, Any] | None) -> str:
    """Name how the fabric releases buffer pressure, for regime-aware gates.

    A ``lossless_pfc`` fabric pauses upstream ports, so congestion leaves
    completed PFC pause intervals. A ``best_effort`` fabric rejects buffer
    admission instead, so congestion leaves trimmed or dropped packets and
    can never leave a pause. A summary from a run without a manifest cannot
    name its regime.
    """
    if manifest is None:
        return "unknown"
    fabric = manifest.get("fabric")
    if not isinstance(fabric, dict):
        return "unknown"
    return "lossless_pfc" if fabric.get("pfc_enabled") else "best_effort"


def _verify_lossless_transport(
    flow_count: int,
    retransmitted_bytes: int,
    recovery_events: int,
    manifest: dict[str, Any] | None,
) -> dict[str, int | str]:
    """Require zero recovery from a run that models no loss mechanism.

    A lossless fabric with no timeout recovery and no trimming has nothing
    that can drop, reorder, or resend a packet, and per-flow ECMP keeps a
    single flow on a single path. Every recovery counter must therefore be
    zero. A nonzero one means a packet reached a queue pair it does not
    belong to, which is the observable signature of a source port reused
    while a straggler from its previous flow was still in the network. The
    check exists because that is otherwise silent: the receiver would fold
    the stray sequence numbers into a healthy flow.
    """
    if not _models_no_loss_mechanism(manifest):
        return {"status": "not_applicable"}
    if retransmitted_bytes or recovery_events:
        raise ValueError(
            "run models no loss mechanism but recorded transport recovery "
            f"(retransmitted_bytes={retransmitted_bytes}, "
            f"recovery_events={recovery_events})"
        )
    return {"status": "verified", "flow_count": flow_count}


_KEY_FIELD_BITS = 24
_KEY_TIME_BITS = 64


def _packed_flow_key(
    src: int, dst: int, source_port: int, start_time_ns: int
) -> int:
    """Pack a flow's identity into a single integer.

    Millions of these are live at once during the join, where one four-integer
    tuple per flow costs roughly three times what the packed form does.
    """
    for value, bits in (
        (src, _KEY_FIELD_BITS),
        (dst, _KEY_FIELD_BITS),
        (source_port, _KEY_FIELD_BITS),
        (start_time_ns, _KEY_TIME_BITS),
    ):
        if value < 0 or value >> bits:
            raise ValueError("flow identity is outside the joinable range")
    return (
        ((src << _KEY_FIELD_BITS | dst) << _KEY_FIELD_BITS | source_port)
        << _KEY_TIME_BITS
    ) | start_time_ns


def _flow_key(row: dict[str, str]) -> int:
    return _packed_flow_key(
        _as_int(row, "src"),
        _as_int(row, "dst"),
        _as_int(row, "source_port"),
        _as_int(row, "start_time_ns"),
    )


def _fct_node_id(encoded_address: str) -> int:
    try:
        address = int(encoded_address, 16)
    except ValueError as error:
        raise ValueError(f"invalid FCT address: {encoded_address!r}") from error
    return (address >> 8) & 0xFFFF


def _load_fct_records(path: Path) -> dict[int, tuple[int, int] | None]:
    """Index `fct.txt` by packed flow key.

    A source port names a live five-tuple, not a flow: the ns-3 bridge reuses
    one once its queue pair terminates, so only the port together with the
    flow's start time identifies a flow across a whole run. Values are
    `(physical_bytes, duration_ns)`; the join replaces each with `None` as it
    consumes it, which both detects a duplicate on the telemetry side and
    releases the record while the pass is still running.
    """
    records: dict[int, tuple[int, int] | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"invalid FCT record at {path}:{line_number}")
            (
                source,
                destination,
                source_port,
                _destination_port,
                size,
                start,
                duration,
                standalone,
            ) = fields
            try:
                value = (int(size), int(duration))
                int(standalone)
                key = _packed_flow_key(
                    _fct_node_id(source),
                    _fct_node_id(destination),
                    int(source_port),
                    int(start),
                )
            except ValueError as error:
                raise ValueError(
                    f"invalid numeric FCT record at {path}:{line_number}"
                ) from error
            if key in records:
                raise ValueError("FCT records contain a duplicate flow key")
            records[key] = value
    return records


_ABSENT: Final = object()
"""Distinguishes an unknown key from a consumed one in the FCT index."""


class _FctJoin:
    """Match completed flow telemetry against `fct.txt` one row at a time.

    The set-based join built a dict of every telemetry row *and* a dict of
    every FCT record, then compared the two key sets. Only the FCT side
    survives here, and each record passes through three states exactly once:
    unknown (absent from the index), pending (a byte/duration pair), consumed
    (`None`). Python cannot seal that sum, so the transition is confined to
    `consume` and the sentinel is compared by identity.
    """

    __slots__ = ("_matched", "_mismatch", "_missing", "_records", "available")

    def __init__(self, fct_path: Path) -> None:
        self.available = fct_path.is_file()
        self._records = _load_fct_records(fct_path) if self.available else {}
        self._matched = 0
        self._missing = 0
        self._mismatch: str | None = None

    def consume(self, row: dict[str, str]) -> None:
        if not self.available:
            return
        key = _flow_key(row)
        record = self._records.get(key, _ABSENT)
        if record is _ABSENT:
            self._missing += 1
            return
        if record is None:
            raise ValueError("flow telemetry contains a duplicate flow key")
        self._records[key] = None
        self._matched += 1
        # Report a missing or extra record ahead of a field mismatch, as the
        # set-based join did: an incomplete join explains the mismatches.
        if self._mismatch is not None:
            return
        physical_bytes, duration_ns = record
        if _as_int(row, "physical_bytes") != physical_bytes:
            self._mismatch = "telemetry/FCT physical-byte mismatch"
        elif _flow_duration_ns(row) != duration_ns:
            self._mismatch = "telemetry/FCT duration mismatch"

    def result(self, failed_flow_count: int) -> dict[str, int | str]:
        if not self.available:
            return {
                "status": "not_available",
                "failed_flow_count": failed_flow_count,
            }
        extra = len(self._records) - self._matched
        if self._missing or extra:
            raise ValueError(
                "telemetry/FCT join is incomplete "
                f"(missing={self._missing}, extra={extra})"
            )
        if self._mismatch is not None:
            raise ValueError(self._mismatch)
        return {
            "status": "verified" if failed_flow_count == 0 else "partial_verified",
            "telemetry_flow_count": self._matched,
            "fct_record_count": len(self._records),
            "failed_flow_count": failed_flow_count,
        }


_HOST_TRANSPORT_EVENTS: Final = frozenset(
    {
        "rto_fired",
        "cnp_taken",
        "clipped_trim",
        "cc_signal_withheld",
        "allowance_gone_reports",
        "cc_exempt_granted",
        "cc_transition",
    }
)

_FORGIVEN_PAYLOAD_EVENTS: Final = frozenset(
    {"trim_forgiven", "remainder_forgiven"}
)
"""The receiver's two forgivenesses. Both account for payload bytes no
receiver was ever given, so both ride the data plane and carry their bytes."""


def _summarize_transport_events(ns3_dir: Path) -> dict[str, Any]:
    path = ns3_dir / "transport_summary.csv"
    if not path.is_file():
        return {"status": "not_available"}
    event_count = 0
    events: dict[str, int] = defaultdict(int)
    bytes_by_event: dict[str, int] = defaultdict(int)
    plane_events: dict[str, int] = defaultdict(int)
    plane_bytes: dict[str, int] = defaultdict(int)
    event_plane_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    valid_events = {
        "data_arrival",
        "data_deliver",
        "data_injected_drop",
        "control_arrival",
        "control_deliver",
        "queue_enqueue",
        "queue_dequeue",
        "qbb_drop",
        "switch_route_drop",
        "switch_admission_drop",
        "switch_egress_queue_drop",
        # A trimmed packet stays subject to the TC_med drop threshold at the
        # trimming switch and every downstream hop (UEC 1.0.3 section 4.1).
        "switch_trimmed_queue_drop",
        # A packet of either plane that a link's own error rate dropped.
        "link_error_drop",
        # Data a switch dropped, telling no one, on a port that has stopped
        # forwarding it.
        "switch_blackhole_drop",
        "trim_ftd_admission",
        "trim_ftd_egress_queue",
        "trim_ftd_lasthop_admission",
        "trim_ftd_lasthop_egress_queue",
        "trim_bts_admission",
        "trim_bts_egress_queue",
        "trim_bts_lasthop_admission",
        "trim_bts_lasthop_egress_queue",
        # A trimmed range the receiver accepted without it: the bytes are
        # undelivered, so the event accounts for them on the data plane.
        "trim_forgiven",
        # A quiet flow's unsent remainder, taken as delivered. Those bytes
        # never reached the wire at all, so they are separate from
        # trim_forgiven and W does not count them.
        "remainder_forgiven",
        # Host-transport reactions. They carry no packet, so they contribute
        # counts and no bytes, and they answer to the control plane: a
        # retransmission timeout is a missing ACK, a rate cut is a CNP.
        "rto_fired",
        "cnp_taken",
        # The congestion-exempt domain's reactions: a congestion signal the
        # sender withheld from its controller while exempt, the receiver's
        # report that the cell has no allowance left, the acknowledgement that
        # granted the exemption, and every report that changed the bit. None
        # carries a packet, so all four are counts and no bytes on the control
        # plane.
        "cc_signal_withheld",
        "allowance_gone_reports",
        "cc_exempt_granted",
        "cc_transition",
        # A trim whose range the receiver already partly holds, so the verdict
        # was asked about fewer bytes than the packet carried. Those bytes were
        # delivered, so the event carries a count and no bytes.
        "clipped_trim",
    }
    # Switch conversions only: trim_forgiven is the receiver's answer to one,
    # not a second conversion, and adding it would double-count the payload.
    trim_events = {
        event
        for event in valid_events
        if event.startswith("trim_") and event not in _FORGIVEN_PAYLOAD_EVENTS
    }
    # The simulator aggregates in memory and emits one row per (event, plane)
    # pair at exit: a raw row per packet event grew past 100 GB per arm and
    # its per-packet flushes exhausted shared CI storage.
    for row in iter_csv(path):
        event = row.get("event")
        plane = row.get("plane")
        if event not in valid_events or plane not in {"data", "control"}:
            raise ValueError("invalid transport event plane or event")
        if event.startswith("data_") and plane != "data":
            raise ValueError("configured data impairment dropped a control packet")
        if event.startswith("control_") and plane != "control":
            raise ValueError("invalid control transport event plane")
        if event.startswith("trim_") and plane != "data":
            raise ValueError("trim conversion must account for undelivered data")
        if event in _HOST_TRANSPORT_EVENTS and plane != "control":
            raise ValueError("host transport reaction must ride the control plane")
        if event in _FORGIVEN_PAYLOAD_EVENTS and plane != "data":
            raise ValueError("forgiveness must account for undelivered data")
        row_events = _as_int(row, "event_count")
        row_bytes = _as_int(row, "total_bytes")
        if row_events < 0 or row_bytes < 0:
            raise ValueError("transport event totals must be nonnegative")
        event_count += row_events
        events[event] += row_events
        bytes_by_event[event] += row_bytes
        plane_events[plane] += row_events
        plane_bytes[plane] += row_bytes
        event_plane_counts[event][plane] += row_events

    return {
        "status": "available",
        "event_count": event_count,
        "event_counts": dict(sorted(events.items())),
        "event_bytes": dict(sorted(bytes_by_event.items())),
        "plane_event_counts": {
            plane: plane_events[plane] for plane in ("data", "control")
        },
        "plane_bytes": {plane: plane_bytes[plane] for plane in ("data", "control")},
        "data_injected_drop_count": events["data_injected_drop"],
        "control_injected_drop_count": 0,
        "data_switch_admission_drop_count": event_plane_counts["switch_admission_drop"][
            "data"
        ],
        "control_switch_admission_drop_count": event_plane_counts[
            "switch_admission_drop"
        ]["control"],
        "data_switch_egress_queue_drop_count": event_plane_counts[
            "switch_egress_queue_drop"
        ]["data"],
        "control_switch_egress_queue_drop_count": event_plane_counts[
            "switch_egress_queue_drop"
        ]["control"],
        "data_natural_buffer_drop_count": (
            event_plane_counts["switch_admission_drop"]["data"]
            + event_plane_counts["switch_egress_queue_drop"]["data"]
        ),
        "control_natural_buffer_drop_count": (
            event_plane_counts["switch_admission_drop"]["control"]
            + event_plane_counts["switch_egress_queue_drop"]["control"]
        ),
        "packet_trimming": {
            "conversion_count": sum(events[event] for event in trim_events),
            "trimmed_payload_bytes": sum(
                bytes_by_event[event] for event in trim_events
            ),
            "ftd_conversion_count": sum(
                events[event]
                for event in trim_events
                if event.startswith("trim_ftd_")
            ),
            "bts_conversion_count": sum(
                events[event]
                for event in trim_events
                if event.startswith("trim_bts_")
            ),
            "admission_conversion_count": sum(
                events[event]
                for event in trim_events
                if event.endswith("_admission")
            ),
            "egress_queue_conversion_count": sum(
                events[event]
                for event in trim_events
                if event.endswith("_egress_queue")
            ),
            # DSCP_TRIMMED_LAST_HOP conversions are reported separately because
            # the source must not treat them as a path or NSCC congestion signal.
            "lasthop_conversion_count": sum(
                events[event] for event in trim_events if "_lasthop_" in event
            ),
            "trimmed_queue_drop_count": events["switch_trimmed_queue_drop"],
        },
        "rto_fired_count": events["rto_fired"],
        "cnp_taken_count": events["cnp_taken"],
        "clipped_trim_count": events["clipped_trim"],
        "trim_forgiven_count": events["trim_forgiven"],
        "trim_forgiven_bytes": bytes_by_event["trim_forgiven"],
        "remainder_forgiven_count": events["remainder_forgiven"],
        "remainder_forgiven_bytes": bytes_by_event["remainder_forgiven"],
    }


# Trimming on admission is the switch rejecting an offered byte outright, so
# these two events carry the whole of W. The egress-queue variants trim a
# packet the switch already admitted and are reported separately.
_TRIM_ADMISSION_EVENTS: Final = ("trim_ftd_admission", "trim_ftd_lasthop_admission")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _network_health(
    transport: dict[str, Any],
    offered_physical_bytes: int,
    background_window: _Window | None,
) -> dict[str, Any]:
    """Fabric health per offered byte: the pre-registered W and its companions.

    W is trimmed-on-admission payload bytes per offered byte. wire_per_offered
    is the bytes receivers actually saw per offered byte, so repair traffic
    pushes it above one and the two together separate a fabric that loses
    bytes from one that re-carries them. burst_drain_ns is how long the
    background incast occupied the fabric; compare it against the incast's
    serialization floor, which this module cannot see.
    """
    drain_ns = background_window.span_ns if background_window is not None else None
    if transport.get("status") != "available":
        return {
            "status": "not_available",
            "offered_physical_bytes": offered_physical_bytes,
            "burst_drain_ns": drain_ns,
        }
    event_bytes = transport.get("event_bytes") or {}
    # A fabric with no trimming trimmed none of the bytes it was offered, so
    # W is zero there and not undefined.
    trimmed_bytes = sum(
        int(event_bytes.get(event, 0)) for event in _TRIM_ADMISSION_EVENTS
    )
    # The data-plane arrival trace is connected only when the profile enables
    # loss or trimming, so a missing event means unmeasured, not zero arrivals.
    arrival = event_bytes.get("data_arrival")
    arrival_bytes = None if arrival is None else int(arrival)
    # Forgiven bytes were trimmed and then never repaired, so W' is what the
    # fabric lost that the transport still had to carry again.
    forgiven_bytes = int(event_bytes.get("trim_forgiven", 0))
    repaired_bytes = trimmed_bytes - forgiven_bytes
    return {
        "status": "available",
        "offered_physical_bytes": offered_physical_bytes,
        "trimmed_admission_bytes": trimmed_bytes,
        "W": _ratio(trimmed_bytes, offered_physical_bytes),
        "forgiven_bytes": forgiven_bytes,
        "W_prime": _ratio(max(repaired_bytes, 0), offered_physical_bytes),
        "data_arrival_bytes": arrival_bytes,
        "wire_per_offered": (
            None
            if arrival_bytes is None
            else _ratio(arrival_bytes, offered_physical_bytes)
        ),
        "burst_drain_ns": drain_ns,
    }


def _summarize_ns3_observability(ns3_dir: Path) -> dict[str, Any]:
    queue_path = ns3_dir / "qlen.txt"
    if queue_path.is_file():
        queue_rows = 0
        queues: set[tuple[int, int]] = set()
        max_queue_bytes = 0
        peak_queue_locations: set[tuple[int, int]] = set()
        for line_number, line in _iter_lines(queue_path):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 6 or fields[0] != "time" or (len(fields) - 3) % 3:
                raise ValueError(
                    f"invalid queue telemetry at {queue_path}:{line_number}"
                )
            try:
                switch = int(fields[2])
                for index in range(3, len(fields), 3):
                    if fields[index] != "j":
                        raise ValueError
                    port = int(fields[index + 1])
                    queue_bytes = int(fields[index + 2])
                    if queue_bytes < 0:
                        raise ValueError
                    queues.add((switch, port))
                    if queue_bytes > max_queue_bytes:
                        max_queue_bytes = queue_bytes
                        peak_queue_locations = {(switch, port)}
                    elif queue_bytes == max_queue_bytes:
                        peak_queue_locations.add((switch, port))
            except ValueError as error:
                raise ValueError(
                    f"invalid queue telemetry at {queue_path}:{line_number}"
                ) from error
            queue_rows += 1
        queue = {
            "status": "available",
            "sample_count": queue_rows,
            "observed_queue_count": len(queues),
            "max_queue_bytes": max_queue_bytes,
            "peak_switch_ports": [
                {"switch": switch, "port": port}
                for switch, port in sorted(peak_queue_locations)
            ],
        }
    else:
        queue = {"status": "not_available"}

    pfc_path = ns3_dir / "pfc.txt"
    if pfc_path.is_file():
        active_pauses: dict[tuple[int, int, int | None], list[int]] = defaultdict(list)
        intervals: list[int] = []
        pause_count = 0
        resume_count = 0
        unmatched_resumes = 0
        affected: dict[tuple[int, int, int | None], int] = {}
        event_count = 0
        uses_queue_identity = True
        for line_number, line in _iter_lines(pfc_path):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) == 6:
                timestamp, node, node_type, port, queue_field, event_type = fields
                queue_id: int | None = int(queue_field)
            elif len(fields) == 5:
                # Artifacts emitted before queue-aware PFC telemetry cannot
                # attribute pauses to a priority queue, but remain analyzable.
                timestamp, node, node_type, port, event_type = fields
                queue_id = None
                uses_queue_identity = False
            else:
                raise ValueError(f"invalid PFC telemetry at {pfc_path}:{line_number}")
            try:
                timestamp_ns = int(timestamp)
                node_id = int(node)
                node_kind = int(node_type)
                port_id = int(port)
                event = int(event_type)
            except ValueError as error:
                raise ValueError(
                    f"invalid PFC telemetry at {pfc_path}:{line_number}"
                ) from error
            if min(timestamp_ns, node_id, node_kind, port_id) < 0 or event not in (
                0,
                1,
            ):
                raise ValueError(f"invalid PFC telemetry at {pfc_path}:{line_number}")
            key = (node_id, port_id, queue_id)
            affected[key] = node_kind
            event_count += 1
            if event == 1:
                pause_count += 1
                if queue_id is not None and active_pauses[key]:
                    raise ValueError(
                        "PFC pause was received before the previous resume"
                    )
                active_pauses[key].append(timestamp_ns)
            else:
                resume_count += 1
                if not active_pauses[key]:
                    unmatched_resumes += 1
                    continue
                # Queue-unattributed historical traces cannot distinguish
                # overlapping pauses on one port. Pairing is deterministic but
                # explicitly labeled as an estimate below.
                start_time_ns = active_pauses[key].pop()
                if timestamp_ns < start_time_ns:
                    raise ValueError("PFC resume precedes its pause")
                intervals.append(timestamp_ns - start_time_ns)
        affected_switch_port_queues = [
            {
                "switch": node_id,
                "port": port_id,
                "queue": queue_id if queue_id is not None else "unknown",
            }
            for (node_id, port_id, queue_id), node_kind in sorted(affected.items())
            if node_kind == 1
        ]
        pfc = {
            "status": "available",
            "event_count": event_count,
            "pause_count": pause_count,
            "resume_count": resume_count,
            "completed_pause_interval_count": len(intervals),
            "total_paused_ns": sum(intervals),
            "max_paused_ns": max(intervals, default=0),
            "unmatched_resume_count": unmatched_resumes,
            "active_pause_count_at_end": sum(
                len(starts) for starts in active_pauses.values()
            ),
            "queue_identity_status": "available"
            if uses_queue_identity
            else "not_available",
            "pause_duration_status": (
                "exact" if uses_queue_identity else "estimated_without_queue_identity"
            ),
            "affected_switch_port_queues": affected_switch_port_queues,
        }
    else:
        pfc = {"status": "not_available"}
    return {
        "queue": queue,
        "pfc": pfc,
        "transport": _summarize_transport_events(ns3_dir),
    }


_COUNTER_FIELDS: Final = (
    "data_attempted_bytes",
    "retransmitted_bytes",
    "recovery_events",
    "trimmed_payload_bytes",
    "trim_notifications",
    "trim_lasthop_notifications",
    "trim_recovery_events",
    "stale_trim_notifications",
    "timeouts",
    "cnp_received",
    "forgiven_bytes",
    "forgiven_ranges",
    "forgiven_remainder_bytes",
    "pacing_refusals",
    "late_forgiven_bytes",
    "cc_signal_withheld",
    "allowance_gone_reports",
    "cc_transitions",
    "cc_obeying_ns",
)
"""Telemetry columns summed verbatim. The column name is the only name they
have, so the totals stay keyed by it rather than restating each one."""

_FOREGROUND_LOGICAL_KINDS: Final = frozenset(
    {"foreground_payload", "provenance_control"}
)

_LEDGER_FIELDS: Final = ("eligible_bytes", "shed_bytes", "forgiven_bytes")
"""One receiving rank's budget for one step, as the flow rows record it."""

_STEP_BUCKET_FIELDS: Final = (
    "flows",
    "completed_flows",
    "failed_flows",
    "shed_flows",
    "logical_bytes",
    "physical_bytes",
)


def _is_dp_all_reduce(row: dict[str, str]) -> bool:
    """A flow whose logical origin is a DP All-Reduce payload."""
    return (
        row.get("parallelism_domain") == "dp"
        and row.get("origin_transport_role") == "collective_payload"
        and row.get("collective_type") == "all_reduce"
    )


def _is_valid_shed(row: dict[str, str]) -> bool:
    """The selection policy may only substitute a DP All-Reduce payload."""
    return (
        _as_bool(row, "admission_eligible")
        and _is_dp_all_reduce(row)
        and row.get("flow_kind") == "provenance_control"
        and row.get("transport_role") == "provenance_control"
    )




class _FlowStatistics:
    """Every figure the analyzer derives from `flow_events.csv`, in one pass.

    Validation, aggregation, and the FCT join all consume the same row and
    then drop it. Duration columns are the only unbounded state and they hold
    8 bytes per flow, so a run with millions of queue pairs costs megabytes
    here instead of the tens of gigabytes that retaining the rows would.
    """

    __slots__ = (
        "_join",
        "all_durations",
        "background",
        "background_window",
        "by_domain_and_kind",
        "by_flow_kind",
        "by_parallelism_domain",
        "by_step",
        "by_training_step",
        "completed_count",
        "counters",
        "dp_all_reduce_traffic",
        "failed_by_reason",
        "failed_count",
        "flow_count",
        "foreground_traffic",
        "cc_exempt_count",
        "cc_obeying_flow_count",
        "soft_refusal_bytes",
        "forgiven_remainder_unsent_bytes",
        "shed_count",
        "shed_logical_bytes",
        "total_logical_bytes",
        "total_physical_bytes",
        "total_traffic",
        "trim_to_repair",
        "ledger",
    )

    def __init__(self, join: _FctJoin) -> None:
        self._join = join
        self.flow_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.shed_count = 0
        # Flows the receiver granted an exemption, and the subset that spent
        # any time back under their controller afterwards. Neither is a byte
        # count, so neither belongs in the summed counters; the soft refusals
        # below are bytes, and they decompose the repairs with the coin's.
        self.cc_exempt_count = 0
        self.soft_refusal_bytes = 0
        self.cc_obeying_flow_count = 0
        # The part of the forgiven remainder no sender put on the wire. The
        # law below computes it per flow, so summing it here costs nothing and
        # keeps the two readings of the same subtraction in one place.
        self.forgiven_remainder_unsent_bytes = 0
        self.total_logical_bytes = 0
        self.total_physical_bytes = 0
        self.shed_logical_bytes = 0
        self.counters = dict.fromkeys(_COUNTER_FIELDS, 0)
        self.failed_by_reason: dict[str, int] = defaultdict(int)
        self.total_traffic = _Traffic()
        self.foreground_traffic = _Traffic()
        self.dp_all_reduce_traffic = _Traffic()
        self.background = _Traffic()
        self.background_window: _Window | None = None
        self.by_step: dict[str, dict[str, int]] = defaultdict(
            lambda: dict.fromkeys(_STEP_BUCKET_FIELDS, 0)
        )
        self.all_durations = array("q")
        self.by_training_step = _durations()
        self.by_parallelism_domain = _durations()
        self.by_flow_kind = _durations()
        self.by_domain_and_kind = _durations()
        self.trim_to_repair = array("q")
        self.ledger: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: dict.fromkeys(_LEDGER_FIELDS, 0)
        )

    def consume(self, row: dict[str, str]) -> None:
        """Fold one flow into every accumulator, then drop it.

        The checks run in the order the row-list implementation ran them, so a
        malformed row still reports the same first violation.
        """
        logical_bytes = _as_int(row, "logical_bytes")
        physical_bytes = _as_int(row, "physical_bytes")
        outcome = _terminal_outcome(row)
        failure_reason = row.get("failure_reason")
        if outcome == "completed" and failure_reason:
            raise ValueError("completed flow must not have a failure reason")
        if outcome == "failed" and not failure_reason:
            raise ValueError("failed flow must record a failure reason")

        trim_notifications = _optional_nonnegative_int(row, "trim_notifications")
        if trim_notifications and (
            _optional_nonnegative_int(row, "trimmed_payload_bytes") == 0
        ):
            raise ValueError("trim notification must identify undelivered payload bytes")
        # A completed flow accounts for every physical byte: the sender
        # attempted it, or the receiver forgave a remainder that covers it.
        # The shortfall is the remainder no sender put on the wire, which the
        # summary reports and nothing in the simulator counts.
        if outcome == "completed":
            unsent_bytes = max(
                0,
                physical_bytes - _optional_nonnegative_int(row, "data_attempted_bytes"),
            )
            if unsent_bytes > _optional_nonnegative_int(
                row, "forgiven_remainder_bytes"
            ):
                raise ValueError(
                    "completed flow must attempt or be forgiven every byte"
                )
            self.forgiven_remainder_unsent_bytes += unsent_bytes

        # Zero means the flow never saw a trim or never sent a repair. A
        # repair before the first trim is NACK-driven and answers a different
        # question, so it stays out of this distribution.
        first_trim_ns = _optional_nonnegative_int(row, "first_trim_ns")
        first_repair_ns = _optional_nonnegative_int(row, "first_repair_ns")
        if first_trim_ns and first_repair_ns >= first_trim_ns:
            self.trim_to_repair.append(first_repair_ns - first_trim_ns)

        kind = row.get("flow_kind")
        domain = row.get("parallelism_domain")
        shed = row.get("decision") == "shed"
        if shed:
            if not _is_valid_shed(row):
                raise ValueError(
                    "shedding policy affected a flow outside DP All-Reduce payloads"
                )
            self.shed_count += 1
            self.shed_logical_bytes += logical_bytes
        if kind == "provenance_control" and physical_bytes <= 0:
            raise ValueError("provenance control flow must carry nonzero physical bytes")

        self.flow_count += 1
        self.total_logical_bytes += logical_bytes
        self.total_physical_bytes += physical_bytes
        for field in _COUNTER_FIELDS:
            self.counters[field] += _optional_nonnegative_int(row, field)
        self.cc_exempt_count += _optional_bool(row, "cc_exempt")
        self.soft_refusal_bytes += _optional_nonnegative_int(
            row, "soft_refusals"
        )
        self.cc_obeying_flow_count += (
            _optional_nonnegative_int(row, "cc_obeying_ns") > 0
        )

        self.total_traffic.add(logical_bytes, physical_bytes)
        if kind in _FOREGROUND_LOGICAL_KINDS:
            self.foreground_traffic.add(logical_bytes, physical_bytes)
        if _is_dp_all_reduce(row):
            self.dp_all_reduce_traffic.add(logical_bytes, physical_bytes)

        # The budget is charged to the receiving rank for the step, which is
        # exactly the key the simulator's ledger uses.
        forgiven_bytes = _optional_nonnegative_int(row, "forgiven_bytes")
        if _as_bool(row, "admission_eligible") or forgiven_bytes:
            cell = self.ledger[
                (row.get("dst", "unknown"), row.get("training_step", "unknown"))
            ]
            if _as_bool(row, "admission_eligible"):
                cell["eligible_bytes"] += logical_bytes
                if shed:
                    cell["shed_bytes"] += logical_bytes
            cell["forgiven_bytes"] += forgiven_bytes

        bucket = self.by_step[row.get("training_step", "unknown")]
        bucket["flows"] += 1
        bucket[f"{outcome}_flows"] += 1
        bucket["shed_flows"] += shed
        bucket["logical_bytes"] += logical_bytes
        bucket["physical_bytes"] += physical_bytes

        if kind == "background_microburst":
            self.background.add(logical_bytes, physical_bytes)
            start_time_ns = _as_int(row, "start_time_ns")
            end_time_ns = _as_int(row, "end_time_ns")
            window = self.background_window
            self.background_window = (
                _Window(start_time_ns, end_time_ns)
                if window is None
                else window.join(start_time_ns, end_time_ns)
            )

        if outcome == "failed":
            self.failed_count += 1
            self.failed_by_reason[failure_reason or ""] += 1
            return

        self.completed_count += 1
        duration = _flow_duration_ns(row)
        step = row.get("training_step") or "unknown"
        domain_name = domain or "unknown"
        kind_name = kind or "unknown"
        self.all_durations.append(duration)
        self.by_training_step[step].append(duration)
        self.by_parallelism_domain[domain_name].append(duration)
        self.by_flow_kind[kind_name].append(duration)
        self.by_domain_and_kind[(domain_name, kind_name)].append(duration)
        self._join.consume(row)

    def background_timeline(self) -> dict[str, int | str]:
        window = self.background_window
        if window is None:
            return {"status": "no_background_microburst"}
        return {
            "status": "available",
            **self.background.summary(),
            "start_time_ns": window.start_ns,
            "end_time_ns": window.end_ns,
            "span_ns": window.span_ns,
        }


def summarize(
    telemetry_dir: Path,
    fct_path: Path | None = None,
    ns3_dir: Path | None = None,
    expected_rank_count: int | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    telemetry_dir = telemetry_dir.resolve()
    flow_events = telemetry_dir / "flow_events.csv"
    if not flow_events.is_file():
        raise FileNotFoundError(f"missing telemetry file: {flow_events}")
    completion_rows = load_csv(telemetry_dir / "rank_completion.csv")
    collective_events = telemetry_dir / "collective_events.csv"
    if fct_path is None:
        fct_path = telemetry_dir.parent / "ns3" / "fct.txt"
    if ns3_dir is None:
        ns3_dir = telemetry_dir.parent / "ns3"
    if manifest_path is None:
        manifest_path = telemetry_dir.parent / "manifest.json"
    manifest = _load_manifest(manifest_path)

    rank_completion_status = _summarize_rank_completions(
        completion_rows, expected_rank_count
    )

    join = _FctJoin(fct_path.resolve())
    statistics = _FlowStatistics(join)
    for row in iter_csv(flow_events):
        statistics.consume(row)

    completion_times = [_as_int(row, "completion_time_ns") for row in completion_rows]
    flow_completion = _timing_statistics(statistics.all_durations)
    rank_completion = _timing_statistics(completion_times)
    fct_join = join.result(statistics.failed_count)
    lossless_transport = _verify_lossless_transport(
        statistics.flow_count,
        statistics.counters["retransmitted_bytes"],
        statistics.counters["recovery_events"],
        manifest,
    )
    collective_completion = (
        _summarize_collectives(iter_csv(collective_events))
        if collective_events.is_file()
        else {"status": "not_available"}
    )
    ns3_observability = _summarize_ns3_observability(ns3_dir.resolve())
    transport_events = ns3_observability.get("transport")
    network_health = _network_health(
        transport_events if isinstance(transport_events, dict) else {},
        statistics.total_physical_bytes,
        statistics.background_window,
    )
    ledger_law = _check_ledger_law(statistics.ledger, manifest)
    _require_lawful_ledger(ledger_law, manifest)
    primary_eligible = (
        expected_rank_count is not None
        and not statistics.failed_count
        and rank_completion_status["status"] == "verified"
        and collective_completion["status"] == "available"
        and fct_join["status"] == "verified"
        # A forgiving arm promised every rank a share of every step, so its
        # numbers are readable only once the ledger law confirms it kept the
        # promise. Every other domain promised nothing to confirm.
        and (
            _shedding_domain(manifest) not in _FORGIVING_DOMAINS
            or ledger_law["status"] == "verified"
        )
    )
    return {
        "flow_count": statistics.flow_count,
        "completed_flow_count": statistics.completed_count,
        "failed_flow_count": statistics.failed_count,
        "shed_flow_count": statistics.shed_count,
        "total_logical_bytes": statistics.total_logical_bytes,
        "total_physical_bytes": statistics.total_physical_bytes,
        "shed_logical_bytes": statistics.shed_logical_bytes,
        "completion_rank_count": len(completion_rows),
        "completion_time_ns_max": rank_completion["max_ns"] or 0,
        "rank_completion_time_ns": rank_completion,
        "rank_completion_status": rank_completion_status,
        "flow_completion_time_ns": {
            "all": flow_completion,
            "by_training_step": _timing_by_group(statistics.by_training_step),
            "by_parallelism_domain": _timing_by_group(
                statistics.by_parallelism_domain
            ),
            "by_flow_kind": _timing_by_group(statistics.by_flow_kind),
            "by_parallelism_domain_and_flow_kind": _timing_by_domain_and_kind(
                statistics.by_domain_and_kind
            ),
        },
        "transport_recovery": {
            "data_attempted_bytes": statistics.counters["data_attempted_bytes"],
            "retransmitted_bytes": statistics.counters["retransmitted_bytes"],
            "recovery_event_count": statistics.counters["recovery_events"],
            "failed_by_reason": dict(sorted(statistics.failed_by_reason.items())),
            "trimmed_payload_bytes": statistics.counters["trimmed_payload_bytes"],
            "trim_notification_count": statistics.counters["trim_notifications"],
            "trim_lasthop_notification_count": statistics.counters[
                "trim_lasthop_notifications"
            ],
            "trim_recovery_event_count": statistics.counters["trim_recovery_events"],
            "stale_trim_notification_count": statistics.counters[
                "stale_trim_notifications"
            ],
            "timeout_count": statistics.counters["timeouts"],
            "cnp_received_count": statistics.counters["cnp_received"],
            # How long a trimmed flow waited before its first repair left the
            # sender. It separates a repair-driven tail from a congestion-
            # control-driven one, which no byte counter can.
            "first_trim_to_first_repair_ns": _timing_statistics(
                statistics.trim_to_repair
            ),
        },
        "collective_completion": collective_completion,
        "physical_traffic_bytes": {
            "total": statistics.total_traffic.summary(),
            "foreground_logical_operations": statistics.foreground_traffic.summary(),
            "dp_all_reduce": statistics.dp_all_reduce_traffic.summary(),
        },
        "background_microburst_timeline": statistics.background_timeline(),
        "network_health": network_health,
        "forgiveness": {
            "forgiven_bytes": statistics.counters["forgiven_bytes"],
            "forgiven_range_count": statistics.counters["forgiven_ranges"],
            # What the budget paid for and the sender delivered anyway: bytes
            # that arrived for a range the receiver had already given up. The
            # charge stands, so the loss the training side actually saw is the
            # difference.
            "late_forgiven_bytes": statistics.counters["late_forgiven_bytes"],
            "actual_loss_bytes": statistics.counters["forgiven_bytes"]
            - statistics.counters["late_forgiven_bytes"],
            # The two v2 receiver policies. Remainder bytes are what the step
            # stop forgave when it ended a sender's step, whether or not that
            # sender had already put them on the wire; pacing refusals are
            # forgivable trims the coin declined, which no other counter can
            # see.
            "forgiven_remainder_bytes": statistics.counters[
                "forgiven_remainder_bytes"
            ],
            # The share of that remainder no sender put on the wire, which the
            # two byte columns already determine: the rest was attempted,
            # trimmed, and then forgiven instead of repaired.
            "forgiven_remainder_unsent_bytes": (
                statistics.forgiven_remainder_unsent_bytes
            ),
            "pacing_refusal_count": statistics.counters["pacing_refusals"],
            # What the congestion exemption did: how many flows were granted
            # one, how many congestion signals they withheld, how many
            # allowance reports reached them, and how many exemptions those
            # reports ended.
            # Repairs decompose into three: the soft cap's refusals for want
            # of vested allowance, the coin's, and the hard cap's.
            "soft_refusal_bytes": statistics.soft_refusal_bytes,
            "cc_exempt_flow_count": statistics.cc_exempt_count,
            "cc_signal_withheld_count": statistics.counters[
                "cc_signal_withheld"
            ],
            "allowance_gone_report_count": statistics.counters[
                "allowance_gone_reports"
            ],
            # The exemption follows the report both ways, so what it costs is
            # a duration and a count of changes, not a single re-arm instant.
            "cc_transition_count": statistics.counters["cc_transitions"],
            "cc_obeying_ns": statistics.counters["cc_obeying_ns"],
            "cc_obeying_flow_count": statistics.cc_obeying_flow_count,
            "forgiven_bytes_by_training_step": _forgiven_by_step(
                statistics.ledger
            ),
            "ledger_law": ledger_law,
        },
        "fct_join": fct_join,
        "flow_control_regime": _flow_control_regime(manifest),
        "lossless_transport": lossless_transport,
        "ns3_observability": ns3_observability,
        "primary_analysis_eligibility": {
            "status": "eligible" if primary_eligible else "ineligible",
            "expected_rank_count": expected_rank_count,
            "failed_flow_count": statistics.failed_count,
            "rank_completion_status": rank_completion_status["status"],
            "collective_completion_status": collective_completion["status"],
            "fct_join_status": fct_join["status"],
            "ledger_law_status": ledger_law["status"],
        },
        "by_training_step": dict(
            sorted(
                statistics.by_step.items(),
                key=lambda entry: _training_step_sort_key(entry[0]),
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument(
        "--fct-file",
        type=Path,
        help="ns-3 FCT output to join and verify; defaults to ../ns3/fct.txt",
    )
    parser.add_argument(
        "--ns3-dir",
        type=Path,
        help="directory containing ns-3 queue and PFC output; defaults to ../ns3",
    )
    parser.add_argument(
        "--expected-rank-count",
        type=int,
        help="require exactly one rank-completion row for every rank in [0, count)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "run manifest naming the modeled loss mechanisms; "
            "defaults to ../manifest.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path, help="write the JSON summary to this path"
    )
    arguments = parser.parse_args()
    summary = summarize(
        arguments.telemetry_dir.resolve(),
        arguments.fct_file.resolve() if arguments.fct_file else None,
        arguments.ns3_dir.resolve() if arguments.ns3_dir else None,
        arguments.expected_rank_count,
        arguments.manifest.resolve() if arguments.manifest else None,
    )
    encoded = json.dumps(summary, indent=2) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
