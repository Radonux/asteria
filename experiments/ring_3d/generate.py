#!/usr/bin/env python3
"""Materialize reproducible Chakra traces and ns-3 inputs for 3D Ring runs.

Run only through the repository's locked environment:

    uv run --locked python experiments/ring_3d/generate.py --profile ... --output ...
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    AttributeProto,
    GlobalMetadata,
    Node,
)
from chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage,
)

try:
    from .generate_clr_schedule import (
        ClrSchedule,
        ClrScheduleParameters,
        generate_clr_schedule,
        generate_explicit_clr_schedule,
        schedule_metadata,
        write_clr_mask,
    )
    from .switch import SwitchProfile
    from .topology import (
        CongestionControl,
        DataPlaneLoss,
        SPINE_NAMING_MODES,
        LoadBalancing,
        PacketTrimming,
        PhysicalNetwork,
        SwitchFabric,
        TransportRecovery,
        build_topology,
        load_network,
    )
except ImportError:
    from generate_clr_schedule import (
        ClrSchedule,
        ClrScheduleParameters,
        generate_clr_schedule,
        generate_explicit_clr_schedule,
        schedule_metadata,
        write_clr_mask,
    )
    from switch import SwitchProfile
    from topology import (
        CongestionControl,
        DataPlaneLoss,
        SPINE_NAMING_MODES,
        LoadBalancing,
        PacketTrimming,
        PhysicalNetwork,
        SwitchFabric,
        TransportRecovery,
        build_topology,
        load_network,
    )


REQUIRED_PROFILE_KEYS = {
    "schema_version",
    "name",
    "parallelism",
    "steps",
    "compute_duration_us",
    "tp_all_reduce_bytes",
    "pp_bytes",
    "dp_all_reduce_bytes",
    "seed",
    "selection_policy",
    "network",
    "microburst_bytes",
}
OPTIONAL_PROFILE_KEYS = {
    "clr_schedule",
    "dp_all_reduce_implementation",
    "microburst_flow_count",
    "microburst_destination_rank",
    "microburst_offset_spacing_ns",
    "microburst_source_ranks",
    "microburst_enabled",
    "microburst_trigger_step",
    "model",
    "workload",
}
EXPECTED_PROFILE_KEYS = REQUIRED_PROFILE_KEYS | OPTIONAL_PROFILE_KEYS
EXPECTED_MODEL_TRACE_KEYS = {
    "parameter_count",
    "parameter_dtype_bytes",
    "transformer_layers",
    "gradient_bucket_count",
    "hidden_size",
    "sequence_length",
    "microbatch_size",
    "activation_dtype_bytes",
    "tensor_parallel_all_reduces_per_layer",
    "pipeline_microbatches",
}
GRADIENT_BUCKET_SAMPLE_MODEL_KEYS = {
    "parameter_count",
    "parameter_dtype_bytes",
    "gradient_bucket_count",
    "transformer_layers",
    "hidden_size",
    "sequence_length",
    "microbatch_size",
    "activation_dtype_bytes",
    "tensor_parallel_all_reduces_per_layer",
    "gradient_accumulation_steps",
}
MAX_P_LOW = 0.01
# The simulator's budget law is exact integer arithmetic over thresholds
# scaled by this factor (ExperimentConfig.hh kDecisionScale). The generator
# resolves the scaled value once and writes it beside the float so the law is
# computed from one number in every language that checks it.
DECISION_SCALE = 1_000_000
DEFAULT_MICROBURST_TRIGGER_STEP = 2
DEFAULT_WORKLOAD_KIND = "three_dimensional_overlap"
# "ring" walks DP peers with fan-in 1 and can never form an incast; "direct"
# (optionally windowed, e.g. "direct4") sends every DP peer's shard
# concurrently, so each rank receives dp-1 flows at once.
DP_ALL_REDUCE_IMPLEMENTATION_PATTERN = re.compile(
    r"^(ring|direct(?:[1-9][0-9]{0,4})?)$"
)
SEQUENTIAL_DP_ALL_REDUCE_WORKLOAD = "sequential_dp_all_reduce"
PERMUTATION_WORKLOAD = "permutation"
WORKLOAD_KINDS = frozenset(
    {DEFAULT_WORKLOAD_KIND, SEQUENTIAL_DP_ALL_REDUCE_WORKLOAD, PERMUTATION_WORKLOAD}
)
# The tag every permutation message carries; one pair of ranks exchanges one.
PERMUTATION_TAG = 1


@dataclass(frozen=True)
class ModelTrace:
    """Structural transformer workload quantities used to derive ET nodes."""

    parameter_count: int
    parameter_dtype_bytes: int
    transformer_layers: int
    gradient_bucket_count: int
    hidden_size: int
    sequence_length: int
    microbatch_size: int
    activation_dtype_bytes: int
    tensor_parallel_all_reduces_per_layer: int
    pipeline_microbatches: int


@dataclass(frozen=True)
class GradientBucketSample:
    """Production-shaped metadata for a bounded representative event window.

    The trace samples one transformer layer across the configured gradient
    accumulation window and one local DP gradient bucket. It intentionally
    does not replay every layer or bucket in an optimizer iteration.
    """

    parameter_count: int
    parameter_dtype_bytes: int
    gradient_bucket_count: int
    transformer_layers: int
    hidden_size: int
    sequence_length: int
    microbatch_size: int
    activation_dtype_bytes: int
    tensor_parallel_all_reduces_per_layer: int
    gradient_accumulation_steps: int


class SheddingDomain(StrEnum):
    """Where the phase-aware budget is spent.

    ``admission`` decides before a payload is offered and substitutes the whole
    message. ``recovery`` decides after a switch has already trimmed a packet
    and lets that packet's bytes go, which is the only point at which the
    policy can act during a congestion episode. ``recovery_exempt`` forgives
    the same way and additionally lets an eligible flow on a non-critical step
    ignore congestion signals until the receiver refuses to forgive one of its
    trims, so the budget is paid in bounded loss rather than in rate.
    """

    ADMISSION = "admission"
    RECOVERY = "recovery"
    RECOVERY_EXEMPT = "recovery_exempt"


FORGIVING_DOMAINS: frozenset[SheddingDomain] = frozenset(
    {SheddingDomain.RECOVERY, SheddingDomain.RECOVERY_EXEMPT}
)
"""The domains that decide after the trim, and so need the trimming transport."""

class CapBase(StrEnum):
    """Which cap decides whether a trim is affordable.

    ``accounted`` is the law of record: the soft, vesting cap against the
    bytes the rank has accounted for, kept or forgiven, which grows as the
    step arrives. ``owed`` is the ablation, affording against the step's plan
    so that the whole allowance is available from the first packet. Revocation
    reads the hard side either way, so every forgiving profile carries the
    plan.
    """

    ACCOUNTED = "accounted"
    OWED = "owed"


class PacingKind(StrEnum):
    """The receiver's rule for declining a forgivable trim.

    ``none`` forgives whatever the cap affords, first come first served.
    ``bernoulli`` forgives a fixed fraction of forgivable ranges, drawn fresh
    on every trimmed arrival, so allowance survives the first burst and a
    declined range is asked again on its next trim. The cap itself is the same
    under both rules, because the receiver measures it against the bytes it
    has accounted for.
    """

    NONE = "none"
    BERNOULLI = "bernoulli"


@dataclass(frozen=True)
class Pacing:
    """A pacing rule and, under Bernoulli alone, its probability.

    The invariant is the whole type: a probability is meaningful under
    Bernoulli and under nothing else, so it is present exactly there.
    """

    kind: PacingKind = PacingKind.NONE
    p: float | None = None

    def __post_init__(self) -> None:
        if (self.kind is PacingKind.BERNOULLI) != (self.p is not None):
            raise ValueError(
                "selection_policy.pacing.p is required by kind 'bernoulli' "
                "and rejected by every other kind"
            )
        if self.p is not None and not 0.0 <= self.p < 1.0:
            raise ValueError(
                "selection_policy.pacing.p must be at least 0 and below 1; "
                "0 forgives nothing and keeps the exemption, 1 is no pacing"
            )

    def document(self) -> dict[str, Any]:
        """The object the simulator's parser reads back."""
        if self.p is None:
            return {"kind": self.kind.value}
        return {"kind": self.kind.value, "p": self.p}


SELECTION_SEMANTICS: dict[SheddingDomain, str] = {
    SheddingDomain.ADMISSION: "logical_admission_selection",
    SheddingDomain.RECOVERY: "recovery_forgiveness",
    SheddingDomain.RECOVERY_EXEMPT: "recovery_forgiveness_cc_exempt",
}
"""The contract string the simulator checks the domain against. Total over the
enum: a domain missing here raises rather than defaulting to admission."""


@dataclass(frozen=True)
class SelectionPolicy:
    """Logical-admission selection parameters retained for paired runs.

    These values are intentionally not presented as a DBLP residual-delivery
    contract: the current Ring-3D policy still substitutes whole payloads.
    """

    p_low: float
    p_high: float
    domain: SheddingDomain = SheddingDomain.ADMISSION
    # The receiver policies of FORGIVE v2, all meaningful only where the
    # receiver decides. ``step_stop`` ends a sender's step once ``1 - p`` of
    # what it owes this rank has arrived; it reads the step's plan, which
    # every forgiving domain carries, so it composes with either cap base.
    pacing: Pacing = Pacing()
    cap_base: CapBase = CapBase.ACCOUNTED
    step_stop: bool = False
    # Whether the receiver's report that the step's budget is spent ends the
    # sender's exemption. False is the reference arm: the report is still
    # carried and counted, and the budget alone bounds the loss.
    reengage: bool = True

    @property
    def semantics(self) -> str:
        return SELECTION_SEMANTICS[self.domain]


@dataclass(frozen=True)
class Permutation:
    """Every rank sends one message to the rank ``shift`` places on."""

    shift: int
    message_bytes: int

    def destination(self, rank: int, ranks: int) -> int:
        return (rank + self.shift) % ranks

    def source(self, rank: int, ranks: int) -> int:
        return (rank - self.shift) % ranks


@dataclass(frozen=True)
class Workload:
    """Trace-shape selector retained in the profile provenance."""

    kind: str
    # Present exactly when the kind is the permutation.
    permutation: Permutation | None = None

    def __post_init__(self) -> None:
        if (self.kind == PERMUTATION_WORKLOAD) != (self.permutation is not None):
            raise ValueError(
                "a permutation workload, and no other, carries a permutation"
            )

    def manifest(self) -> dict[str, Any]:
        if self.permutation is None:
            return {"kind": self.kind}
        return {
            "kind": self.kind,
            "shift": self.permutation.shift,
            "message_bytes": self.permutation.message_bytes,
        }


@dataclass(frozen=True)
class ExplicitClrSchedule:
    """One-based CLR labels imported from an external phase schedule."""

    critical_steps: tuple[int, ...]


@dataclass(frozen=True)
class Profile:
    name: str
    tp: int
    pp: int
    dp: int
    steps: int
    compute_duration_us: int
    tp_all_reduce_bytes: int
    pp_bytes: int
    dp_all_reduce_bytes: int
    seed: int
    selection_policy: SelectionPolicy
    workload: Workload
    explicit_clr_schedule: ExplicitClrSchedule | None
    network: PhysicalNetwork
    microburst_bytes: int
    microburst_flow_count: int
    microburst_destination_rank: int | None
    microburst_offset_spacing_ns: int
    microburst_source_ranks: tuple[int, ...] | None
    microburst_enabled: bool
    microburst_trigger_step: int
    dp_all_reduce_implementation: str
    model: ModelTrace | GradientBucketSample | None

    @property
    def ranks(self) -> int:
        return self.tp * self.pp * self.dp


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return probability


def scaled_threshold(probability: float) -> int:
    """Scale a probability the way C++ std::llround(p * kDecisionScale) does.

    Round half away from zero over the exact double, not Python's round, whose
    ties go to even, and not int(x + 0.5), which rounds the addition first. A
    probability landing on a half-integer scaled value is the only case where
    the three disagree, and it is the case the boundary tests pin.
    """
    scaled = Decimal(probability * DECISION_SCALE)
    return int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _load_pacing(policy: dict[str, Any]) -> Pacing:
    """The one smart constructor for the pacing rule.

    The kind is a closed sum, so an unknown one is refused rather than
    defaulting to none; ``Pacing`` itself owns the probability invariant.
    """
    value = policy.get("pacing")
    if value is None:
        return Pacing()
    if not isinstance(value, dict) or not set(value) <= {"kind", "p"}:
        raise ValueError(
            "selection_policy.pacing must be an object with kind and, under "
            "'bernoulli', p"
        )
    kind_value = value.get("kind")
    if kind_value not in tuple(kind.value for kind in PacingKind):
        raise ValueError(
            "selection_policy.pacing.kind must be one of "
            f"{sorted(kind.value for kind in PacingKind)}"
        )
    if "p" not in value:
        return Pacing(kind=PacingKind(kind_value))
    return Pacing(
        kind=PacingKind(kind_value),
        p=_probability(value["p"], "selection_policy.pacing.p"),
    )


def _load_cap_base(policy: dict[str, Any]) -> CapBase:
    value = policy.get("cap_base", CapBase.ACCOUNTED.value)
    if value not in tuple(base.value for base in CapBase):
        raise ValueError(
            "selection_policy.cap_base must be one of "
            f"{sorted(base.value for base in CapBase)}"
        )
    return CapBase(value)


def _load_step_stop(policy: dict[str, Any]) -> bool:
    value = policy.get("step_stop", False)
    if not isinstance(value, bool):
        raise ValueError("selection_policy.step_stop must be a boolean")
    return value


def _load_reengage(policy: dict[str, Any]) -> bool:
    value = policy.get("reengage", True)
    if not isinstance(value, bool):
        raise ValueError("selection_policy.reengage must be a boolean")
    return value


def _load_selection_policy(document: dict[str, Any]) -> SelectionPolicy:
    policy = document.get("selection_policy")
    if not isinstance(policy, dict):
        raise ValueError("selection_policy must be an object")
    required_keys = {"p_low", "p_high"}
    optional_keys = {"domain", "pacing", "cap_base", "step_stop", "reengage"}
    if not required_keys <= set(policy) <= required_keys | optional_keys:
        raise ValueError(
            f"selection_policy must contain exactly {sorted(required_keys)} "
            f"plus optional {sorted(optional_keys)}"
        )
    p_low = _probability(policy["p_low"], "selection_policy.p_low")
    p_high = _probability(policy["p_high"], "selection_policy.p_high")
    if p_low > MAX_P_LOW:
        raise ValueError("selection_policy.p_low must be at most 0.01")
    if p_high < p_low:
        raise ValueError("selection_policy.p_high must be at least p_low")
    domain_value = policy.get("domain", SheddingDomain.ADMISSION.value)
    if domain_value not in tuple(domain.value for domain in SheddingDomain):
        raise ValueError(
            "selection_policy.domain must be one of "
            f"{sorted(domain.value for domain in SheddingDomain)}"
        )
    domain = SheddingDomain(domain_value)
    pacing = _load_pacing(policy)
    cap_base = _load_cap_base(policy)
    step_stop = _load_step_stop(policy)
    reengage = _load_reengage(policy)
    # Every v2 policy belongs to the receiver, and only a forgiving domain has
    # a receiver that decides anything. Accepting them elsewhere would write a
    # profile whose knob moves nothing.
    if domain not in FORGIVING_DOMAINS:
        for key in ("pacing", "cap_base", "step_stop", "reengage"):
            if key in policy:
                raise ValueError(
                    f"selection_policy.{key} requires a forgiving domain"
                )
    return SelectionPolicy(
        p_low=p_low,
        p_high=p_high,
        domain=domain,
        pacing=pacing,
        cap_base=cap_base,
        step_stop=step_stop,
        reengage=reengage,
    )


def _load_workload(document: dict[str, Any]) -> Workload:
    """Load an explicit trace shape while preserving the 3D default."""
    value = document.get("workload")
    if value is None:
        return Workload(DEFAULT_WORKLOAD_KIND)
    if not isinstance(value, dict) or "kind" not in value:
        raise ValueError("workload must contain 'kind'")
    kind = value["kind"]
    if kind not in WORKLOAD_KINDS:
        raise ValueError(f"workload.kind must be one of {sorted(WORKLOAD_KINDS)}")
    if kind != PERMUTATION_WORKLOAD:
        if set(value) != {"kind"}:
            raise ValueError(f"workload.kind '{kind}' takes no other key")
        return Workload(kind)
    if set(value) != {"kind", "shift", "message_bytes"}:
        raise ValueError(
            "workload.kind 'permutation' must contain exactly "
            "['kind', 'message_bytes', 'shift']"
        )
    return Workload(
        kind,
        Permutation(
            shift=_require_positive_int(value["shift"], "workload.shift"),
            message_bytes=_require_positive_int(
                value["message_bytes"], "workload.message_bytes"
            ),
        ),
    )


def _load_explicit_clr_schedule(
    document: dict[str, Any], steps: int
) -> ExplicitClrSchedule | None:
    """Load a strict externally derived phase mask when one is supplied."""
    value = document.get("clr_schedule")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"kind", "critical_steps"}:
        raise ValueError("clr_schedule must contain exactly ['critical_steps', 'kind']")
    if value["kind"] != "explicit_critical_steps":
        raise ValueError("clr_schedule.kind must be 'explicit_critical_steps'")
    critical_steps = value["critical_steps"]
    if not isinstance(critical_steps, list):
        raise ValueError("clr_schedule.critical_steps must be an array")
    validated_steps = tuple(
        _require_positive_int(step, "clr_schedule.critical_steps entry")
        for step in critical_steps
    )
    if len(set(validated_steps)) != len(validated_steps):
        raise ValueError("clr_schedule.critical_steps must not contain duplicates")
    if any(step > steps for step in validated_steps):
        raise ValueError("clr_schedule.critical_steps entry is outside the profile")
    return ExplicitClrSchedule(critical_steps=validated_steps)


def _load_model_workload(
    document: dict[str, Any], tp: int, pp: int
) -> ModelTrace | GradientBucketSample | None:
    value = document.get("model")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("model must be an object")
    if set(value) == GRADIENT_BUCKET_SAMPLE_MODEL_KEYS:
        model = GradientBucketSample(
            parameter_count=_require_positive_int(
                value["parameter_count"], "model.parameter_count"
            ),
            parameter_dtype_bytes=_require_positive_int(
                value["parameter_dtype_bytes"], "model.parameter_dtype_bytes"
            ),
            gradient_bucket_count=_require_positive_int(
                value["gradient_bucket_count"], "model.gradient_bucket_count"
            ),
            transformer_layers=_require_positive_int(
                value["transformer_layers"], "model.transformer_layers"
            ),
            hidden_size=_require_positive_int(
                value["hidden_size"], "model.hidden_size"
            ),
            sequence_length=_require_positive_int(
                value["sequence_length"], "model.sequence_length"
            ),
            microbatch_size=_require_positive_int(
                value["microbatch_size"], "model.microbatch_size"
            ),
            activation_dtype_bytes=_require_positive_int(
                value["activation_dtype_bytes"], "model.activation_dtype_bytes"
            ),
            tensor_parallel_all_reduces_per_layer=_require_positive_int(
                value["tensor_parallel_all_reduces_per_layer"],
                "model.tensor_parallel_all_reduces_per_layer",
            ),
            gradient_accumulation_steps=_require_positive_int(
                value["gradient_accumulation_steps"],
                "model.gradient_accumulation_steps",
            ),
        )
        total_gradient_bytes = model.parameter_count * model.parameter_dtype_bytes
        shard_count = tp * pp
        if total_gradient_bytes % shard_count:
            raise ValueError("model gradient bytes must divide evenly across TP × PP")
        if (total_gradient_bytes // shard_count) % model.gradient_bucket_count:
            raise ValueError(
                "per-rank gradient bytes must divide evenly across sample buckets"
            )
        return model
    if set(value) != EXPECTED_MODEL_TRACE_KEYS:
        raise ValueError(
            "model must contain exactly one of "
            f"{sorted(GRADIENT_BUCKET_SAMPLE_MODEL_KEYS)} or "
            f"{sorted(EXPECTED_MODEL_TRACE_KEYS)}"
        )
    model = ModelTrace(
        parameter_count=_require_positive_int(
            value["parameter_count"], "model.parameter_count"
        ),
        parameter_dtype_bytes=_require_positive_int(
            value["parameter_dtype_bytes"], "model.parameter_dtype_bytes"
        ),
        transformer_layers=_require_positive_int(
            value["transformer_layers"], "model.transformer_layers"
        ),
        gradient_bucket_count=_require_positive_int(
            value["gradient_bucket_count"], "model.gradient_bucket_count"
        ),
        hidden_size=_require_positive_int(value["hidden_size"], "model.hidden_size"),
        sequence_length=_require_positive_int(
            value["sequence_length"], "model.sequence_length"
        ),
        microbatch_size=_require_positive_int(
            value["microbatch_size"], "model.microbatch_size"
        ),
        activation_dtype_bytes=_require_positive_int(
            value["activation_dtype_bytes"], "model.activation_dtype_bytes"
        ),
        tensor_parallel_all_reduces_per_layer=_require_positive_int(
            value["tensor_parallel_all_reduces_per_layer"],
            "model.tensor_parallel_all_reduces_per_layer",
        ),
        pipeline_microbatches=_require_positive_int(
            value["pipeline_microbatches"], "model.pipeline_microbatches"
        ),
    )
    if model.transformer_layers % pp:
        raise ValueError("model.transformer_layers must be divisible by PP")
    total_gradient_bytes = model.parameter_count * model.parameter_dtype_bytes
    shard_count = tp * pp
    if total_gradient_bytes % shard_count:
        raise ValueError("model gradient bytes must divide evenly across TP × PP")
    return model


def load_profile(profile_path: Path) -> Profile:
    with profile_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return parse_profile_document(document)


def parse_profile_document(document: Any) -> Profile:
    """Decode one in-memory Ring-3D profile through the canonical validator."""
    if not isinstance(document, dict):
        raise ValueError("profile must be a JSON object")
    unknown_keys = set(document) - EXPECTED_PROFILE_KEYS
    if unknown_keys:
        raise ValueError(f"unknown profile keys: {sorted(unknown_keys)}")
    workload = _load_workload(document)
    # A permutation's message size is its own, so it takes no DP bucket size.
    required_keys = (
        REQUIRED_PROFILE_KEYS - {"dp_all_reduce_bytes"}
        if workload.permutation is not None
        else REQUIRED_PROFILE_KEYS
    )
    missing_keys = required_keys - set(document)
    if missing_keys:
        raise ValueError(f"missing profile keys: {sorted(missing_keys)}")
    if workload.permutation is not None and "dp_all_reduce_bytes" in document:
        raise ValueError(
            "workload.kind 'permutation' sends workload.message_bytes and takes "
            "no dp_all_reduce_bytes"
        )
    if document.get("schema_version") != 1:
        raise ValueError("profile schema_version must be 1")

    parallelism = document.get("parallelism")
    if not isinstance(parallelism, dict) or set(parallelism) != {"tp", "pp", "dp"}:
        raise ValueError("parallelism must contain exactly tp, pp, and dp")
    name = document.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    steps = _require_positive_int(document.get("steps"), "steps")
    tp = _require_positive_int(parallelism["tp"], "parallelism.tp")
    pp = _require_positive_int(parallelism["pp"], "parallelism.pp")
    dp = _require_positive_int(parallelism["dp"], "parallelism.dp")
    ranks = tp * pp * dp
    selection_policy = _load_selection_policy(document)
    explicit_clr_schedule = _load_explicit_clr_schedule(document, steps)
    network = load_network(document.get("network"), ranks)
    if selection_policy.domain in FORGIVING_DOMAINS:
        domain_value = selection_policy.domain.value
        recovery = network.transport_recovery
        trimming = network.packet_trimming
        if recovery is None or not recovery.selective_repair:
            raise ValueError(
                f"selection_policy.domain '{domain_value}' requires "
                "network.transport_recovery.selective_repair"
            )
        if trimming is None or trimming.mode != "ftd":
            raise ValueError(
                f"selection_policy.domain '{domain_value}' requires "
                "network.packet_trimming.mode 'ftd'"
            )
        # Under 'none' the sender takes no rate cut to begin with, so an
        # exemption from rate cuts is vacuous. A profile asking for it has a
        # mistake in it, not a no-op, and a silent no-op would report as a
        # null result from a mechanism that never ran.
        if (
            selection_policy.domain is SheddingDomain.RECOVERY_EXEMPT
            and network.congestion_control.mode != "dcqcn"
        ):
            raise ValueError(
                "selection_policy.domain 'recovery_exempt' requires "
                "network.congestion_control.mode 'dcqcn'"
            )
        if network.load_balancing.mode != "ecmp":
            raise ValueError(
                f"selection_policy.domain '{domain_value}' and "
                f"network.load_balancing.mode '{network.load_balancing.mode}' "
                "are mutually exclusive"
            )
    microburst_flow_count = _require_positive_int(
        document.get("microburst_flow_count", min(2, ranks // 2)),
        "microburst_flow_count",
    )
    if microburst_flow_count >= ranks:
        raise ValueError(
            "microburst_flow_count must leave at least one destination rank"
        )
    destination_value = document.get("microburst_destination_rank")
    if destination_value is None:
        microburst_destination_rank = None
    else:
        microburst_destination_rank = _require_nonnegative_int(
            destination_value, "microburst_destination_rank"
        )
        if microburst_destination_rank >= ranks:
            raise ValueError("microburst_destination_rank is outside the profile")
    source_values = document.get("microburst_source_ranks")
    if source_values is None:
        microburst_source_ranks = None
    else:
        if not isinstance(source_values, list) or not source_values:
            raise ValueError("microburst_source_ranks must be a nonempty array")
        microburst_source_ranks = tuple(
            _require_nonnegative_int(source, "microburst_source_ranks entry")
            for source in source_values
        )
        if len(set(microburst_source_ranks)) != len(microburst_source_ranks):
            raise ValueError("microburst_source_ranks must not contain duplicates")
        if any(source >= ranks for source in microburst_source_ranks):
            raise ValueError(
                "microburst_source_ranks contains a rank outside the profile"
            )
        if microburst_destination_rank is None:
            raise ValueError(
                "microburst_source_ranks requires microburst_destination_rank"
            )
        if microburst_destination_rank in microburst_source_ranks:
            raise ValueError("microburst source and destination ranks must differ")
        microburst_flow_count = len(microburst_source_ranks)
    microburst_enabled = document.get("microburst_enabled", True)
    if not isinstance(microburst_enabled, bool):
        raise ValueError("microburst_enabled must be a boolean")
    microburst_trigger_step = _require_positive_int(
        document.get("microburst_trigger_step", DEFAULT_MICROBURST_TRIGGER_STEP),
        "microburst_trigger_step",
    )
    if microburst_enabled and microburst_trigger_step > steps:
        raise ValueError(
            "microburst_trigger_step must land within the profile's training steps"
        )
    dp_all_reduce_implementation = document.get(
        "dp_all_reduce_implementation", "ring"
    )
    if not isinstance(
        dp_all_reduce_implementation, str
    ) or not DP_ALL_REDUCE_IMPLEMENTATION_PATTERN.fullmatch(
        dp_all_reduce_implementation
    ):
        raise ValueError(
            "dp_all_reduce_implementation must be 'ring', 'direct', or "
            "'direct<window>'"
        )
    model = _load_model_workload(document, tp, pp)

    profile = Profile(
        name=name,
        tp=tp,
        pp=pp,
        dp=dp,
        steps=steps,
        compute_duration_us=_require_nonnegative_int(
            document.get("compute_duration_us"), "compute_duration_us"
        ),
        tp_all_reduce_bytes=_require_nonnegative_int(
            document.get("tp_all_reduce_bytes"), "tp_all_reduce_bytes"
        ),
        pp_bytes=_require_nonnegative_int(document.get("pp_bytes"), "pp_bytes"),
        dp_all_reduce_bytes=(
            0
            if workload.permutation is not None
            else _require_positive_int(
                document.get("dp_all_reduce_bytes"), "dp_all_reduce_bytes"
            )
        ),
        seed=_require_positive_int(document.get("seed"), "seed"),
        selection_policy=selection_policy,
        workload=workload,
        explicit_clr_schedule=explicit_clr_schedule,
        network=network,
        microburst_bytes=_require_positive_int(
            document.get("microburst_bytes"), "microburst_bytes"
        ),
        microburst_flow_count=microburst_flow_count,
        microburst_destination_rank=microburst_destination_rank,
        microburst_offset_spacing_ns=_require_nonnegative_int(
            document.get("microburst_offset_spacing_ns", 1_000),
            "microburst_offset_spacing_ns",
        ),
        microburst_source_ranks=microburst_source_ranks,
        microburst_enabled=microburst_enabled,
        microburst_trigger_step=microburst_trigger_step,
        dp_all_reduce_implementation=dp_all_reduce_implementation,
        model=model,
    )
    if profile.workload.permutation is not None:
        if profile.workload.permutation.shift >= profile.ranks:
            raise ValueError("workload.shift must be below the rank count")
        if profile.steps != 1:
            raise ValueError("workload.kind 'permutation' sends once and needs steps 1")
        if (
            profile.compute_duration_us != 0
            or profile.tp_all_reduce_bytes != 0
            or profile.pp_bytes != 0
        ):
            raise ValueError(
                "workload.kind 'permutation' requires compute_duration_us, "
                "tp_all_reduce_bytes and pp_bytes to be zero"
            )
        if profile.model is not None:
            raise ValueError(
                "workload.kind 'permutation' cannot include model metadata"
            )
        # The background incast fires on a DP All-Reduce, of which there is none.
        if profile.microburst_enabled:
            raise ValueError(
                "workload.kind 'permutation' requires microburst_enabled false"
            )
    elif profile.workload.kind == SEQUENTIAL_DP_ALL_REDUCE_WORKLOAD:
        if profile.tp != 1 or profile.pp != 1:
            raise ValueError("sequential_dp_all_reduce requires TP=1 and PP=1")
        if profile.compute_duration_us != 0:
            raise ValueError(
                "sequential_dp_all_reduce requires compute_duration_us to be zero"
            )
        if profile.tp_all_reduce_bytes != 0 or profile.pp_bytes != 0:
            raise ValueError(
                "sequential_dp_all_reduce requires TP and PP payload bytes to be zero"
            )
        if profile.model is not None:
            raise ValueError("sequential_dp_all_reduce cannot include model metadata")
    elif profile.compute_duration_us == 0 or profile.tp_all_reduce_bytes == 0:
        raise ValueError(
            "three-dimensional overlap workloads require positive compute_duration_us "
            "and tp_all_reduce_bytes"
        )
    if profile.pp > 1 and profile.pp_bytes == 0:
        raise ValueError("pp_bytes must be positive when PP is greater than one")
    if isinstance(profile.model, (ModelTrace, GradientBucketSample)):
        activation_bytes = (
            profile.model.microbatch_size
            * profile.model.sequence_length
            * profile.model.hidden_size
            * profile.model.activation_dtype_bytes
        )
        if profile.tp_all_reduce_bytes != activation_bytes:
            raise ValueError(
                "tp_all_reduce_bytes must equal the model activation tensor bytes"
            )
        if (
            isinstance(profile.model, ModelTrace)
            and profile.pp_bytes != activation_bytes
        ):
            raise ValueError("pp_bytes must equal the model activation tensor bytes")
        gradient_bytes_per_rank = (
            profile.model.parameter_count
            * profile.model.parameter_dtype_bytes
            // (profile.tp * profile.pp)
        )
        expected_bucket_bytes = (
            gradient_bytes_per_rank // profile.model.gradient_bucket_count
            if isinstance(profile.model, GradientBucketSample)
            else (gradient_bytes_per_rank + profile.model.gradient_bucket_count - 1)
            // profile.model.gradient_bucket_count
        )
        if profile.dp_all_reduce_bytes != expected_bucket_bytes:
            raise ValueError(
                "dp_all_reduce_bytes must equal the model-derived maximum "
                "per-rank gradient bucket bytes"
            )
    return profile


def rank_for(tp_rank: int, pp_rank: int, dp_rank: int, profile: Profile) -> int:
    """Return a TP-fastest global rank."""
    if not (
        0 <= tp_rank < profile.tp
        and 0 <= pp_rank < profile.pp
        and 0 <= dp_rank < profile.dp
    ):
        raise ValueError("parallelism coordinate is outside the profile")
    return ((dp_rank * profile.pp) + pp_rank) * profile.tp + tp_rank


def _gradient_bucket_sizes(
    total_gradient_bytes: int,
    shard_count: int,
    bucket_count: int,
) -> tuple[int, ...]:
    """Partition one local gradient shard into exact, near-equal buckets."""
    if total_gradient_bytes % shard_count:
        raise ValueError("model gradient bytes must divide evenly across TP × PP")
    gradient_bytes_per_rank = total_gradient_bytes // shard_count
    base, remainder = divmod(gradient_bytes_per_rank, bucket_count)
    return tuple(
        base + (1 if bucket_index < remainder else 0)
        for bucket_index in range(bucket_count)
    )


def coordinates_for(rank: int, profile: Profile) -> tuple[int, int, int]:
    """Invert the TP-fastest rank mapping as (tp_rank, pp_rank, dp_rank)."""
    if not 0 <= rank < profile.ranks:
        raise ValueError("rank is outside the profile")
    tp_rank = rank % profile.tp
    pp_rank = (rank // profile.tp) % profile.pp
    dp_rank = rank // (profile.tp * profile.pp)
    return tp_rank, pp_rank, dp_rank


def generate_groups(
    profile: Profile,
) -> tuple[dict[str, list[int]], dict[str, int], dict[str, int], dict[str, int]]:
    """Generate explicit TP, PP, and DP communicator membership and lookup IDs."""
    groups: dict[str, list[int]] = {}
    tp_group_for_rank: dict[str, int] = {}
    pp_group_for_rank: dict[str, int] = {}
    dp_group_for_rank: dict[str, int] = {}
    next_group_id = 1

    for dp_rank in range(profile.dp):
        for pp_rank in range(profile.pp):
            members = [
                rank_for(tp_rank, pp_rank, dp_rank, profile)
                for tp_rank in range(profile.tp)
            ]
            groups[str(next_group_id)] = members
            for rank in members:
                tp_group_for_rank[str(rank)] = next_group_id
            next_group_id += 1

    for dp_rank in range(profile.dp):
        for tp_rank in range(profile.tp):
            members = [
                rank_for(tp_rank, pp_rank, dp_rank, profile)
                for pp_rank in range(profile.pp)
            ]
            groups[str(next_group_id)] = members
            for rank in members:
                pp_group_for_rank[str(rank)] = next_group_id
            next_group_id += 1

    for pp_rank in range(profile.pp):
        for tp_rank in range(profile.tp):
            members = [
                rank_for(tp_rank, pp_rank, dp_rank, profile)
                for dp_rank in range(profile.dp)
            ]
            groups[str(next_group_id)] = members
            for rank in members:
                dp_group_for_rank[str(rank)] = next_group_id
            next_group_id += 1

    expected_group_count = (
        profile.dp * profile.pp + profile.dp * profile.tp + profile.pp * profile.tp
    )
    if len(groups) != expected_group_count:
        raise AssertionError("unexpected communicator group count")
    return groups, tp_group_for_rank, pp_group_for_rank, dp_group_for_rank


def _attribute(name: str, **value: Any) -> AttributeProto:
    return AttributeProto(name=name, **value)


class TraceWriter:
    def __init__(
        self,
        path: Path,
        rank: int,
        profile: Profile,
        groups: tuple[
            dict[str, list[int]], dict[str, int], dict[str, int], dict[str, int]
        ],
    ) -> None:
        self.path = path
        self.rank = rank
        self.profile = profile
        self.coords = coordinates_for(rank, profile)
        self.groups = groups
        self.node_id = 1
        self.nodes: list[Node] = []

    def _new_node(self, name: str, node_type: int, dependencies: Iterable[int]) -> Node:
        node = Node(id=self.node_id, name=name, type=node_type)
        node.ctrl_deps.extend(dependencies)
        self.node_id += 1
        self.nodes.append(node)
        return node

    def compute(self, name: str, dependencies: Iterable[int]) -> int:
        node = self._new_node(name, COMP_NODE, dependencies)
        node.duration_micros = self.profile.compute_duration_us
        node.attr.append(_attribute("is_cpu_op", bool_val=False))
        return node.id

    def all_reduce(
        self,
        name: str,
        dependencies: Iterable[int],
        domain: str,
        group_id: int,
        size_bytes: int,
        step: int,
    ) -> int:
        node = self._new_node(name, COMM_COLL_NODE, dependencies)
        node.attr.extend(
            [
                _attribute("is_cpu_op", bool_val=False),
                _attribute("comm_type", int64_val=ALL_REDUCE),
                _attribute("comm_size", uint64_val=size_bytes),
                _attribute("pg_name", string_val=str(group_id)),
                _attribute("parallelism_domain", string_val=domain),
                _attribute("training_step", uint64_val=step),
            ]
        )
        return node.id

    def point_to_point_node(
        self,
        name: str,
        node_type: int,
        dependencies: Iterable[int],
        src: int,
        dst: int,
        tag: int,
        size_bytes: int,
        step: int,
        domain: str | None,
    ) -> int:
        node = self._new_node(name, node_type, dependencies)
        node.attr.extend(
            [
                _attribute("is_cpu_op", bool_val=False),
                _attribute("comm_src", uint32_val=src),
                _attribute("comm_dst", uint32_val=dst),
                _attribute("comm_tag", uint32_val=tag),
                _attribute("comm_size", uint64_val=size_bytes),
            ]
        )
        if domain is not None:
            node.attr.append(_attribute("parallelism_domain", string_val=domain))
        node.attr.append(_attribute("training_step", uint64_val=step))
        return node.id

    def pipeline_node(
        self,
        name: str,
        node_type: int,
        dependencies: Iterable[int],
        src: int,
        dst: int,
        tag: int,
        step: int,
    ) -> int:
        return self.point_to_point_node(
            name,
            node_type,
            dependencies,
            src,
            dst,
            tag,
            self.profile.pp_bytes,
            step,
            "pp",
        )

    def _build_smoke_trace(self) -> None:
        _, tp_groups, _, dp_groups = self.groups
        tp_rank, pp_rank, dp_rank = self.coords
        predecessor: int | None = None

        for step in range(1, self.profile.steps + 1):
            forward = self.compute(
                f"step_{step}_forward_compute",
                [] if predecessor is None else [predecessor],
            )
            tp_reduce = self.all_reduce(
                f"step_{step}_tp_all_reduce",
                [forward],
                "tp",
                tp_groups[str(self.rank)],
                self.profile.tp_all_reduce_bytes,
                step,
            )

            pipeline_dependencies = [tp_reduce]
            pipeline_events: list[int] = []
            pipeline_tag = step * 100_000 + dp_rank * 1_000 + tp_rank
            if pp_rank > 0:
                source = rank_for(tp_rank, pp_rank - 1, dp_rank, self.profile)
                pipeline_events.append(
                    self.pipeline_node(
                        f"step_{step}_pp_recv_from_{source}",
                        COMM_RECV_NODE,
                        pipeline_dependencies,
                        source,
                        self.rank,
                        pipeline_tag,
                        step,
                    )
                )
            if pp_rank < self.profile.pp - 1:
                destination = rank_for(tp_rank, pp_rank + 1, dp_rank, self.profile)
                pipeline_events.append(
                    self.pipeline_node(
                        f"step_{step}_pp_send_to_{destination}",
                        COMM_SEND_NODE,
                        pipeline_dependencies,
                        self.rank,
                        destination,
                        pipeline_tag,
                        step,
                    )
                )

            bucket_zero = self.compute(
                f"step_{step}_backward_bucket_0",
                [tp_reduce, *pipeline_events],
            )
            dp_reduce_zero = self.all_reduce(
                f"step_{step}_dp_all_reduce_bucket_0",
                [bucket_zero],
                "dp",
                dp_groups[str(self.rank)],
                self.profile.dp_all_reduce_bytes,
                step,
            )
            bucket_one = self.compute(
                f"step_{step}_backward_bucket_1",
                [bucket_zero],
            )
            dp_reduce_one = self.all_reduce(
                f"step_{step}_dp_all_reduce_bucket_1",
                [bucket_one],
                "dp",
                dp_groups[str(self.rank)],
                self.profile.dp_all_reduce_bytes,
                step,
            )
            predecessor = self.compute(
                f"step_{step}_optimizer",
                [dp_reduce_zero, dp_reduce_one],
            )

    def _build_sequential_dp_all_reduce_trace(self) -> None:
        """Emit the historical Phase-1 shape: one chained DP All-Reduce per step."""
        _, _, _, dp_groups = self.groups
        predecessor: int | None = None
        for step in range(1, self.profile.steps + 1):
            predecessor = self.all_reduce(
                f"step_{step}_dp_all_reduce",
                [] if predecessor is None else [predecessor],
                "dp",
                dp_groups[str(self.rank)],
                self.profile.dp_all_reduce_bytes,
                step,
            )

    def _build_gradient_bucket_sample_trace(self) -> None:
        """Emit a bounded, production-shaped DP/TP event window.

        This samples one transformer's forward/backward collective pattern for
        every accumulation microbatch, followed by one local DP bucket. It is
        deliberately smaller than a full-model replay.
        """
        assert isinstance(self.profile.model, GradientBucketSample)
        model = self.profile.model
        _, tp_groups, _, dp_groups = self.groups
        predecessor: int | None = None
        forward_tp_collectives = max(
            1, model.tensor_parallel_all_reduces_per_layer // 2
        )
        backward_tp_collectives = (
            model.tensor_parallel_all_reduces_per_layer - forward_tp_collectives
        )

        for step in range(1, self.profile.steps + 1):
            microbatch_tails: list[int] = []
            microbatch_predecessor = predecessor
            for microbatch in range(model.gradient_accumulation_steps):
                forward = self.compute(
                    f"step_{step}_microbatch_{microbatch}_layer_sample_forward_compute",
                    [] if microbatch_predecessor is None else [microbatch_predecessor],
                )
                forward_tail = forward
                for collective in range(forward_tp_collectives):
                    forward_tail = self.all_reduce(
                        f"step_{step}_microbatch_{microbatch}_layer_sample_"
                        f"tp_forward_{collective}",
                        [forward_tail],
                        "tp",
                        tp_groups[str(self.rank)],
                        self.profile.tp_all_reduce_bytes,
                        step,
                    )
                backward_tail = self.compute(
                    f"step_{step}_microbatch_{microbatch}_layer_sample_backward_compute",
                    [forward_tail],
                )
                for collective in range(backward_tp_collectives):
                    backward_tail = self.all_reduce(
                        f"step_{step}_microbatch_{microbatch}_layer_sample_"
                        f"tp_backward_{collective}",
                        [backward_tail],
                        "tp",
                        tp_groups[str(self.rank)],
                        self.profile.tp_all_reduce_bytes,
                        step,
                    )
                microbatch_tails.append(backward_tail)
                microbatch_predecessor = backward_tail
            dp_reduce = self.all_reduce(
                f"step_{step}_dp_all_reduce_bucket_sample",
                microbatch_tails,
                "dp",
                dp_groups[str(self.rank)],
                self.profile.dp_all_reduce_bytes,
                step,
            )
            predecessor = self.compute(f"step_{step}_optimizer", [dp_reduce])

    def _build_model_trace(self, model: ModelTrace) -> None:
        """Emit a structural transformer step with exact model-derived DP bytes.

        The profile is a workload specification rather than a framework replay:
        it partitions parameters evenly across TP and PP, then emits one DP
        All-Reduce bucket per specified gradient bucket. The sum of bucket
        sizes on every rank is therefore the local model-gradient shard.
        """
        _, tp_groups, _, dp_groups = self.groups
        tp_rank, pp_rank, dp_rank = self.coords
        local_layers = model.transformer_layers // self.profile.pp
        gradient_bucket_sizes = _gradient_bucket_sizes(
            model.parameter_count * model.parameter_dtype_bytes,
            self.profile.tp * self.profile.pp,
            model.gradient_bucket_count,
        )
        forward_tp_collectives = max(
            1, model.tensor_parallel_all_reduces_per_layer // 2
        )
        backward_tp_collectives = (
            model.tensor_parallel_all_reduces_per_layer - forward_tp_collectives
        )
        predecessor: int | None = None

        for step in range(1, self.profile.steps + 1):
            forward_tail = predecessor
            for layer in range(local_layers):
                forward = self.compute(
                    f"step_{step}_layer_{layer}_forward_compute",
                    [] if forward_tail is None else [forward_tail],
                )
                forward_tail = forward
                for collective in range(forward_tp_collectives):
                    forward_tail = self.all_reduce(
                        f"step_{step}_layer_{layer}_tp_forward_{collective}",
                        [forward_tail],
                        "tp",
                        tp_groups[str(self.rank)],
                        self.profile.tp_all_reduce_bytes,
                        step,
                    )

            forward_pipeline: list[int] = []
            for microbatch in range(model.pipeline_microbatches):
                pipeline_tag = (
                    step * 10_000_000 + microbatch * 100_000 + dp_rank * 1_000 + tp_rank
                )
                dependencies = [forward_tail] if forward_tail is not None else []
                if pp_rank > 0:
                    source = rank_for(tp_rank, pp_rank - 1, dp_rank, self.profile)
                    forward_pipeline.append(
                        self.pipeline_node(
                            f"step_{step}_microbatch_{microbatch}_pp_forward_recv",
                            COMM_RECV_NODE,
                            dependencies,
                            source,
                            self.rank,
                            pipeline_tag,
                            step,
                        )
                    )
                if pp_rank < self.profile.pp - 1:
                    destination = rank_for(tp_rank, pp_rank + 1, dp_rank, self.profile)
                    forward_pipeline.append(
                        self.pipeline_node(
                            f"step_{step}_microbatch_{microbatch}_pp_forward_send",
                            COMM_SEND_NODE,
                            dependencies,
                            self.rank,
                            destination,
                            pipeline_tag,
                            step,
                        )
                    )

            backward_tail = self.compute(
                f"step_{step}_backward_start",
                [forward_tail, *forward_pipeline] if forward_tail else forward_pipeline,
            )
            dp_reductions: list[int] = []
            emitted_buckets = 0
            for layer_offset, layer in enumerate(reversed(range(local_layers))):
                backward_tail = self.compute(
                    f"step_{step}_layer_{layer}_backward_compute",
                    [backward_tail],
                )
                for collective in range(backward_tp_collectives):
                    backward_tail = self.all_reduce(
                        f"step_{step}_layer_{layer}_tp_backward_{collective}",
                        [backward_tail],
                        "tp",
                        tp_groups[str(self.rank)],
                        self.profile.tp_all_reduce_bytes,
                        step,
                    )
                target_buckets = (
                    (layer_offset + 1) * model.gradient_bucket_count // local_layers
                )
                while emitted_buckets < target_buckets:
                    dp_reductions.append(
                        self.all_reduce(
                            f"step_{step}_dp_all_reduce_bucket_{emitted_buckets}",
                            [backward_tail],
                            "dp",
                            dp_groups[str(self.rank)],
                            gradient_bucket_sizes[emitted_buckets],
                            step,
                        )
                    )
                    emitted_buckets += 1

            backward_pipeline: list[int] = []
            for microbatch in range(model.pipeline_microbatches):
                pipeline_tag = (
                    50_000_000
                    + step * 10_000_000
                    + microbatch * 100_000
                    + dp_rank * 1_000
                    + tp_rank
                )
                if pp_rank < self.profile.pp - 1:
                    source = rank_for(tp_rank, pp_rank + 1, dp_rank, self.profile)
                    backward_pipeline.append(
                        self.pipeline_node(
                            f"step_{step}_microbatch_{microbatch}_pp_backward_recv",
                            COMM_RECV_NODE,
                            [backward_tail],
                            source,
                            self.rank,
                            pipeline_tag,
                            step,
                        )
                    )
                if pp_rank > 0:
                    destination = rank_for(tp_rank, pp_rank - 1, dp_rank, self.profile)
                    backward_pipeline.append(
                        self.pipeline_node(
                            f"step_{step}_microbatch_{microbatch}_pp_backward_send",
                            COMM_SEND_NODE,
                            [backward_tail],
                            self.rank,
                            destination,
                            pipeline_tag,
                            step,
                        )
                    )
            predecessor = self.compute(
                f"step_{step}_optimizer",
                [*dp_reductions, *backward_pipeline],
            )

    def _build_permutation_trace(self, permutation: Permutation) -> None:
        """One message to the rank shift places on and one from the rank as
        far back, neither waiting for anything. No parallel dimension owns
        them, so they carry no domain."""
        ranks = self.profile.ranks
        destination = permutation.destination(self.rank, ranks)
        source = permutation.source(self.rank, ranks)
        for name, node_type, src, dst in (
            (
                f"permutation_send_to_{destination}",
                COMM_SEND_NODE,
                self.rank,
                destination,
            ),
            (f"permutation_recv_from_{source}", COMM_RECV_NODE, source, self.rank),
        ):
            self.point_to_point_node(
                name,
                node_type,
                [],
                src,
                dst,
                PERMUTATION_TAG,
                permutation.message_bytes,
                1,
                None,
            )

    def build(self) -> None:
        if self.profile.workload.permutation is not None:
            self._build_permutation_trace(self.profile.workload.permutation)
        elif self.profile.workload.kind == SEQUENTIAL_DP_ALL_REDUCE_WORKLOAD:
            self._build_sequential_dp_all_reduce_trace()
        elif self.profile.model is None:
            self._build_smoke_trace()
        elif isinstance(self.profile.model, GradientBucketSample):
            self._build_gradient_bucket_sample_trace()
        else:
            self._build_model_trace(self.profile.model)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as trace:
            encodeMessage(trace, GlobalMetadata(version="0.0.4"))
            for node in self.nodes:
                encodeMessage(trace, node)


def write_network_config(
    path: Path,
    topology: Path,
    output_dir: Path,
    packet_payload_bytes: int,
    queue_monitor_start_ns: int,
    queue_monitor_interval_ns: int,
    data_loss: DataPlaneLoss | None,
    transport_recovery: TransportRecovery | None,
    packet_trimming: PacketTrimming | None,
    fabric: SwitchFabric | None,
    congestion_control: CongestionControl,
    link_rate: str,
    load_balancing: LoadBalancing,
    switch: SwitchProfile,
    link_failure_settings: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The bundled ns-3 setup unconditionally opens these legacy input files,
    # even though ASTRA-sim dynamically creates the RDMA QPs after setup.
    # A zero count is therefore the correct empty static-flow/trace workload.
    (output_dir / "flow.txt").write_text("0\n", encoding="utf-8")
    (output_dir / "trace.txt").write_text("0\n", encoding="utf-8")
    entries = {
        "TOPOLOGY_FILE": topology,
        "FLOW_FILE": output_dir / "flow.txt",
        "TRACE_FILE": output_dir / "trace.txt",
        "TRACE_OUTPUT_FILE": output_dir / "mix.tr",
        "FCT_OUTPUT_FILE": output_dir / "fct.txt",
        "PFC_OUTPUT_FILE": output_dir / "pfc.txt",
        "QLEN_MON_FILE": output_dir / "qlen.txt",
        # The raw per-packet research artifact, in its historic ns-3 CSV
        # syntax, zstd-compressed by the simulator on a worker thread.
        "TRANSPORT_EVENT_OUTPUT_FILE": output_dir / "transport_events.csv.zst",
        # Rotation size for the raw stream's segments. Sized for storage,
        # not shipping: segments are hashed and deleted in flight by the
        # CI sidecar (segment_hasher.py), never uploaded, so the old
        # 1.8 GB asset-cap sizing would only raise the in-flight disk
        # bound (two segments) on shared cluster scratch.
        "TRANSPORT_EVENT_SEGMENT_BYTES": 268435456,
        # Aggregated per-(event, plane) totals; what analysis consumes.
        "TRANSPORT_EVENT_SUMMARY_OUTPUT_FILE": output_dir
        / "transport_summary.csv",
    }
    if data_loss is None:
        data_loss_settings = (
            "DATA_LOSS_PROBABILITY 0.0\nDATA_LOSS_START_NS 0\n"
            "DATA_LOSS_DURATION_NS 0\nDATA_LOSS_SCOPE all\n"
            "DATA_LOSS_SOURCE_HOST -1\nDATA_LOSS_DESTINATION_HOST -1\n"
            "DATA_LOSS_RECEIVER_NODE -1\nDATA_LOSS_RNG_STREAM 51\n"
        )
    else:
        data_loss_settings = (
            f"DATA_LOSS_PROBABILITY {data_loss.probability:.17g}\n"
            f"DATA_LOSS_START_NS {data_loss.start_ns}\n"
            f"DATA_LOSS_DURATION_NS {data_loss.duration_ns}\n"
            f"DATA_LOSS_SCOPE {data_loss.scope}\n"
            f"DATA_LOSS_SOURCE_HOST {data_loss.source_host if data_loss.source_host is not None else -1}\n"
            "DATA_LOSS_DESTINATION_HOST "
            f"{data_loss.destination_host if data_loss.destination_host is not None else -1}\n"
            f"DATA_LOSS_RECEIVER_NODE {data_loss.receiver_node if data_loss.receiver_node is not None else -1}\n"
            f"DATA_LOSS_RNG_STREAM {data_loss.rng_stream}\n"
        )
    if transport_recovery is None:
        recovery_settings = (
            "RETRANSMISSION_TIMEOUT_NS 0\nMAX_RETRANSMISSION_RETRIES 0\n"
            "NO_PROGRESS_TIMEOUT_NS 0\nSELECTIVE_RETRANSMISSION 0\n"
        )
    else:
        recovery_settings = (
            "RETRANSMISSION_TIMEOUT_NS "
            f"{transport_recovery.retransmission_timeout_ns}\n"
            "MAX_RETRANSMISSION_RETRIES "
            f"{transport_recovery.max_retransmission_retries}\n"
            "NO_PROGRESS_TIMEOUT_NS "
            f"{transport_recovery.no_progress_timeout_ns}\n"
            "SELECTIVE_RETRANSMISSION "
            f"{int(transport_recovery.selective_repair)}\n"
        )
    if fabric is None:
        fabric_settings = (
            "BUFFER_SIZE 32\nENABLE_PFC 1\nHEADROOM_FACTOR 3\n"
            "DATA_QUEUE_BYTES 0\nTRIMMED_QUEUE_BYTES 0\n"
        )
    else:
        fabric_settings = (
            f"BUFFER_SIZE {fabric.buffer_size_mb}\n"
            f"ENABLE_PFC {int(fabric.pfc_enabled)}\n"
            f"HEADROOM_FACTOR {fabric.headroom_factor}\n"
            f"DATA_QUEUE_BYTES {fabric.data_queue_bytes}\n"
            f"TRIMMED_QUEUE_BYTES {fabric.trimmed_queue_bytes}\n"
        )
    rate_ai_bps, rate_hai_bps, min_rate_bps = congestion_control.rates_bps(link_rate)
    congestion_settings = (
        f"CC_MODE {congestion_control.cc_mode}\n"
        f"RATE_AI {rate_ai_bps}bps\n"
        f"RATE_HAI {rate_hai_bps}bps\n"
        f"MIN_RATE {min_rate_bps}bps\n"
    )
    if packet_trimming is None:
        trim_settings = "PACKET_TRIM_MODE disabled\n"
    else:
        trim_settings = (
            f"PACKET_TRIM_MODE {packet_trimming.mode}\n"
            f"PACKET_TRIM_QUEUE {packet_trimming.trimmed_queue}\n"
            f"PACKET_TRIM_QUEUE_WEIGHT {packet_trimming.trimmed_queue_weight}\n"
            f"MIN_TRIM_SIZE {packet_trimming.min_trim_size_bytes}\n"
            f"PACKET_TRIM_LASTHOP {int(packet_trimming.last_hop_codepoint)}\n"
        )
    # Absent means ECMP to the simulator, so an ECMP profile keeps the exact
    # configuration it had before the knob existed. Under per-packet paths the
    # run also writes what every switch port sent and, where the identification
    # names a spine, what every host received over each.
    load_balancing_settings = (
        ""
        if load_balancing.mode == "ecmp"
        else f"LOAD_BALANCING {load_balancing.mode}\n"
        + load_balancing.parameter_settings()
    )
    for key, counter_file in counter_files(load_balancing, output_dir).items():
        load_balancing_settings += f"{key} {counter_file}\n"
    with path.open("w", encoding="utf-8") as config:
        config.write(
            "ENABLE_QCN 1\nUSE_DYNAMIC_PFC_THRESHOLD 1\n\n"
            f"PACKET_PAYLOAD_SIZE {packet_payload_bytes}\n\n"
        )
        for key, value in entries.items():
            config.write(f"{key} {value}\n")
        config.write(
            f"QLEN_MON_START {queue_monitor_start_ns}\n"
            f"QLEN_MON_INTERVAL {queue_monitor_interval_ns}\n"
            "QLEN_MON_END 20000\n\n"
            "SIMULATOR_STOP_TIME 40000000000000.00\n\n"
            f"{congestion_settings}"
            "ALPHA_RESUME_INTERVAL 1\nRATE_DECREASE_INTERVAL 4\n"
            "CLAMP_TARGET_RATE 0\nRP_TIMER 900\nEWMA_GAIN 0.00390625\n"
            "FAST_RECOVERY_TIMES 1\nDCTCP_RATE_AI 1000Mb/s\n\n"
            "ERROR_RATE_PER_LINK 0.0000\nL2_CHUNK_SIZE 4000\nL2_ACK_INTERVAL 1\n"
            "L2_BACK_TO_ZERO 0\n"
            f"{data_loss_settings}{recovery_settings}{trim_settings}"
            f"{load_balancing_settings}\n"
            "HAS_WIN 1\nGLOBAL_T 0\nVAR_WIN 1\n"
            "FAST_REACT 1\nU_TARGET 0.95\nMI_THRESH 0\nINT_MULTI 1\nMULTI_RATE 0\n"
            "SAMPLE_FEEDBACK 0\nPINT_LOG_BASE 1.05\nPINT_PROB 1.0\n"
            "NIC_TOTAL_PAUSE_TIME 0\n\nRATE_BOUND 1\nACK_HIGH_PRIO 1\n"
            "LINK_DOWN 0 0 0\nENABLE_TRACE 1\n\n"
            f"{switch.ecn_threshold_maps(congestion_control.ecn_threshold_maps())}"
            f"{switch.probability_map()}"
            f"{fabric_settings}"
            f"{link_failure_settings}"
        )


def counter_files(load_balancing: LoadBalancing, output_dir: Path) -> dict[str, Path]:
    """The network configuration keys of the counter files a run writes."""
    files: dict[str, Path] = {}
    if load_balancing.mode != "ecmp":
        files["PORT_COUNTER_OUTPUT_FILE"] = output_dir / "port_counters.csv"
    if load_balancing.mode in SPINE_NAMING_MODES:
        files["SPINE_ARRIVAL_OUTPUT_FILE"] = output_dir / "spine_arrivals.csv"
    if load_balancing.mode == "spray_policy":
        files["SPINE_REPORT_OUTPUT_FILE"] = output_dir / "spine_reports.csv"
    return files


def _microburst_flows(profile: Profile) -> list[dict[str, int]]:
    if not profile.microburst_enabled:
        return []
    if profile.microburst_destination_rank is None:
        cross_rack_base = profile.ranks // 2
        destinations = [
            cross_rack_base + index for index in range(profile.microburst_flow_count)
        ]
        if any(destination >= profile.ranks for destination in destinations):
            raise ValueError(
                "microburst_flow_count requires microburst_destination_rank "
                "when one-to-one cross-rack destinations exceed the profile"
            )
        endpoints = list(zip(range(profile.microburst_flow_count), destinations))
    else:
        destination = profile.microburst_destination_rank
        sources = (
            list(profile.microburst_source_ranks)
            if profile.microburst_source_ranks is not None
            else [rank for rank in range(profile.ranks) if rank != destination]
        )
        endpoints = [
            (source, destination) for source in sources[: profile.microburst_flow_count]
        ]
    return [
        {
            "src": source,
            "dst": destination,
            "size_bytes": profile.microburst_bytes,
            "offset_ns": index * profile.microburst_offset_spacing_ns,
            "priority_group": 3,
        }
        for index, (source, destination) in enumerate(endpoints)
    ]


def dp_fan_in(dp: int, dp_all_reduce_implementation: str) -> int:
    """Peak concurrent inbound DP shard flows per receiving rank.

    Ring walks peers one at a time, so a rank never receives from more than
    one sender. Direct keeps ``min(window, dp - 1)`` transfers in flight per
    rank (AllToAll sets ``parallel_reduce`` exactly this way), and the
    unwindowed form saturates at every peer at once.
    """
    if dp <= 1:
        return 0
    if dp_all_reduce_implementation == "ring":
        return 1
    window = dp_all_reduce_implementation[len("direct"):]
    peers = dp - 1
    return min(int(window), peers) if window else peers


def resolve_selection_policy(
    profile: Profile,
    *,
    p_low: float | None = None,
    p_high: float | None = None,
    allow_clr_exposure: bool = False,
    domain: SheddingDomain | None = None,
) -> SelectionPolicy:
    """Resolve optional command-line overrides against the typed profile policy.

    ``allow_clr_exposure`` lifts the strict-CLR ceiling on ``p_low``. It
    exists for exactly one caller: the fixed-high comparison arm, which
    deliberately sheds at the permissive rate during critical steps to
    measure the headroom an unbounded policy would take. A policy run must
    never pass it.
    """
    resolved_low = (
        profile.selection_policy.p_low
        if p_low is None
        else _probability(p_low, "p_low")
    )
    resolved_high = (
        profile.selection_policy.p_high
        if p_high is None
        else _probability(p_high, "p_high")
    )
    if resolved_low > MAX_P_LOW and not allow_clr_exposure:
        raise ValueError("p_low must be at most 0.01")
    if resolved_high < resolved_low:
        raise ValueError("p_high must be at least p_low")
    return SelectionPolicy(
        p_low=resolved_low,
        p_high=resolved_high,
        domain=(
            profile.selection_policy.domain if domain is None else domain
        ),
        pacing=profile.selection_policy.pacing,
        cap_base=profile.selection_policy.cap_base,
        step_stop=profile.selection_policy.step_stop,
        reengage=profile.selection_policy.reengage,
    )


def _selection_probabilities_for_schedule(
    schedule: ClrSchedule,
    policy: SelectionPolicy,
) -> dict[str, float]:
    return {
        step: (policy.p_low if is_clr else policy.p_high)
        for step, is_clr in schedule.is_clr_by_step().items()
    }


def write_experiment_config(
    path: Path,
    profile: Profile,
    clr_schedule: ClrSchedule,
    selection_policy: SelectionPolicy,
) -> None:
    microburst_flows = _microburst_flows(profile)
    policy_document: dict[str, Any] = {
        "semantics": selection_policy.semantics,
        "p_low": selection_policy.p_low,
        "p_high": selection_policy.p_high,
        # The scaled integers the budget law is actually computed from. The
        # simulator refuses the file if either disagrees with llround of the
        # float beside it, so the two representations cannot drift.
        "p_low_threshold": scaled_threshold(selection_policy.p_low),
        "p_high_threshold": scaled_threshold(selection_policy.p_high),
    }
    if selection_policy.domain in FORGIVING_DOMAINS:
        policy_document["domain"] = selection_policy.domain.value
        # The simulator's frontend cannot read network_config.txt, so the
        # transport contract the recovery domain depends on is asserted here
        # and checked against the built transport at setup.
        policy_document["transport"] = {
            "selective_repair": True,
            "packet_trimming_ftd": True,
        }
        # The receiver's two v2 policies. An arm that overrides the domain to
        # admission drops both, because the simulator refuses them there and
        # neither would move anything if it did not.
        policy_document["pacing"] = selection_policy.pacing.document()
        # The base is written out whatever it is: it is the law the arm ran
        # under, and an artifact that names it can be read years later without
        # the generator. The other two are written where they move something.
        policy_document["cap_base"] = selection_policy.cap_base.value
        if selection_policy.step_stop:
            policy_document["step_stop"] = True
        if not selection_policy.reengage:
            policy_document["reengage"] = False
    policy = {
        "schema_version": 1,
        "enabled": True,
        "seed": profile.seed,
        "run_id": profile.name,
        "eligibility": "dp_all_reduce_only",
        "selection_probability_by_step": _selection_probabilities_for_schedule(
            clr_schedule, selection_policy
        ),
        "selection_policy": policy_document,
        # The receiving ranks and steps the forgiveness ledger is sized for.
        "scale": {"ranks": profile.ranks, "steps": profile.steps},
        "default_priority_group": 3,
        "provenance": {"control_bytes": 64, "priority_group": 1},
        "vnet_to_priority_group": {"0": 3},
        "microburst": {
            "enabled": profile.microburst_enabled,
            "trigger_step": profile.microburst_trigger_step,
            "flows": microburst_flows,
        },
    }
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def write_system_config(
    path: Path,
    dp_group_ids: Iterable[int] = (),
    dp_all_reduce_implementation: str = "ring",
) -> None:
    configuration: dict[str, Any] = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ["ring"],
        "all-gather-implementation": ["ring"],
        "reduce-scatter-implementation": ["ring"],
        "all-to-all-implementation": ["ring"],
        "collective-optimization": "localBWAware",
        "local-mem-bw": 1600,
        "boost-mode": 0,
        "roofline-enabled": 0,
        "peak-perf": 900,
    }
    if dp_all_reduce_implementation != "ring":
        # Override only the DP communicator groups: TP and PP keep the
        # global ring while every DP all-reduce runs the incast-forming
        # direct algorithm.
        configuration["all-reduce-implementation-per-group"] = {
            str(group_id): dp_all_reduce_implementation
            for group_id in sorted(dp_group_ids)
        }
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")


def model_trace_metadata(profile: Profile) -> dict[str, int | float | str] | None:
    if profile.model is None:
        return None
    model = profile.model
    total_gradient_bytes = model.parameter_count * model.parameter_dtype_bytes
    if isinstance(model, GradientBucketSample):
        gradient_bytes_per_rank = total_gradient_bytes // (profile.tp * profile.pp)
        return {
            "workload_kind": "representative_production_event_window",
            "sampling_contract": (
                "one transformer-layer collective pattern across every "
                "accumulation microbatch plus one local DP bucket per step"
            ),
            "parameter_count": model.parameter_count,
            "parameter_dtype_bytes": model.parameter_dtype_bytes,
            "total_gradient_bytes_per_data_parallel_replica": total_gradient_bytes,
            "gradient_bucket_count": model.gradient_bucket_count,
            "gradient_bytes_per_rank": gradient_bytes_per_rank,
            "simulated_gradient_bucket_bytes": profile.dp_all_reduce_bytes,
            "simulated_gradient_buckets_per_step": 1,
            "transformer_layers": model.transformer_layers,
            "activation_bytes_per_microbatch": profile.tp_all_reduce_bytes,
            "tensor_parallel_all_reduces_per_layer": (
                model.tensor_parallel_all_reduces_per_layer
            ),
            "gradient_accumulation_steps": model.gradient_accumulation_steps,
            "sampled_tp_all_reduces_per_step": (
                model.gradient_accumulation_steps
                * model.tensor_parallel_all_reduces_per_layer
            ),
            "packet_payload_bytes": profile.network.packet_payload_bytes,
            "queue_monitor_start_ns": profile.network.queue_monitor_start_ns,
        }
    gradient_bytes_per_rank = total_gradient_bytes // (profile.tp * profile.pp)
    gradient_bucket_sizes = _gradient_bucket_sizes(
        total_gradient_bytes,
        profile.tp * profile.pp,
        model.gradient_bucket_count,
    )
    return {
        "workload_kind": "structural_transformer_trace",
        "parameter_count": model.parameter_count,
        "parameter_dtype_bytes": model.parameter_dtype_bytes,
        "total_gradient_bytes_per_data_parallel_replica": total_gradient_bytes,
        "gradient_bytes_per_rank": gradient_bytes_per_rank,
        "gradient_bucket_count": model.gradient_bucket_count,
        "gradient_bucket_min_bytes": min(gradient_bucket_sizes),
        "gradient_bucket_max_bytes": max(gradient_bucket_sizes),
        "gradient_bucket_total_bytes": sum(gradient_bucket_sizes),
        "transformer_layers": model.transformer_layers,
        "transformer_layers_per_pipeline_stage": model.transformer_layers // profile.pp,
        "activation_bytes_per_microbatch": profile.pp_bytes,
        "pipeline_microbatches": model.pipeline_microbatches,
        "tensor_parallel_all_reduces_per_layer": model.tensor_parallel_all_reduces_per_layer,
    }


def materialize(
    profile_path: Path,
    output_dir: Path,
    clean: bool = False,
    *,
    seed_override: int | None = None,
    p_low: float | None = None,
    p_high: float | None = None,
    allow_clr_exposure: bool = False,
    domain: SheddingDomain | None = None,
    clr_schedule_parameters: ClrScheduleParameters | None = None,
) -> dict[str, Any]:
    profile = load_profile(profile_path)
    if seed_override is not None:
        profile = replace(
            profile, seed=_require_positive_int(seed_override, "seed_override")
        )
    selection_policy = resolve_selection_policy(
        profile,
        p_low=p_low,
        p_high=p_high,
        allow_clr_exposure=allow_clr_exposure,
        domain=domain,
    )
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_config = output_dir / "profile.json"
    shutil.copyfile(profile_path, profile_config)
    clr_mask = output_dir / "clr_mask.csv"
    clr_schedule = (
        generate_explicit_clr_schedule(
            profile.steps,
            profile.seed,
            profile.explicit_clr_schedule.critical_steps,
        )
        if profile.explicit_clr_schedule is not None
        else generate_clr_schedule(
            profile.steps,
            profile.seed,
            clr_schedule_parameters or ClrScheduleParameters(),
        )
    )
    write_clr_mask(clr_mask, clr_schedule)

    groups, tp_groups, pp_groups, dp_groups = generate_groups(profile)
    workload_dir = output_dir / "workload"
    for rank in range(profile.ranks):
        writer = TraceWriter(
            workload_dir / f"ring_3d.{rank}.et",
            rank,
            profile,
            (groups, tp_groups, pp_groups, dp_groups),
        )
        writer.build()
        writer.write()

    topology = output_dir / "topology.txt"
    physical_topology = build_topology(profile.network, profile.ranks)
    physical_topology.write(topology)
    network_config = output_dir / "network_config.txt"
    write_network_config(
        network_config,
        topology.resolve(),
        output_dir / "ns3",
        profile.network.packet_payload_bytes,
        profile.network.queue_monitor_start_ns,
        profile.network.queue_monitor_interval_ns,
        profile.network.data_loss,
        profile.network.transport_recovery,
        profile.network.packet_trimming,
        profile.network.fabric,
        profile.network.congestion_control,
        profile.network.link_rate,
        profile.network.load_balancing,
        profile.network.switch,
        profile.network.link_failure_config(profile.ranks),
    )
    experiment_config = output_dir / "experiment.json"
    write_experiment_config(
        experiment_config, profile, clr_schedule, selection_policy
    )
    system_config = output_dir / "system.json"
    write_system_config(
        system_config,
        dp_group_ids=set(dp_groups.values()),
        dp_all_reduce_implementation=profile.dp_all_reduce_implementation,
    )
    (output_dir / "communicator_groups.json").write_text(
        json.dumps(groups, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "logical_topology.json").write_text(
        json.dumps({"logical-dims": [str(profile.ranks)]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "remote_memory.json").write_text(
        json.dumps({"memory-type": "NO_MEMORY_EXPANSION"}, indent=2) + "\n",
        encoding="utf-8",
    )
    model_metadata = model_trace_metadata(profile)
    if model_metadata is not None:
        (output_dir / "model_trace.json").write_text(
            json.dumps(model_metadata, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "profile": profile.name,
        "seed": profile.seed,
        "ranks": profile.ranks,
        "parallelism": {"tp": profile.tp, "pp": profile.pp, "dp": profile.dp},
        "selection_policy": {
            "semantics": selection_policy.semantics,
            "domain": selection_policy.domain.value,
            "p_low": selection_policy.p_low,
            "p_high": selection_policy.p_high,
            # The analyzer checks the budget law from these, not from the
            # floats, so it computes the inequality the simulator enforced.
            "decision_scale": DECISION_SCALE,
            "p_low_threshold": scaled_threshold(selection_policy.p_low),
            "p_high_threshold": scaled_threshold(selection_policy.p_high),
        },
        "workload": profile.workload.manifest(),
        "collective_implementations": {
            "default_all_reduce": "ring",
            "dp_all_reduce": profile.dp_all_reduce_implementation,
            "dp_fan_in": dp_fan_in(
                profile.dp, profile.dp_all_reduce_implementation
            ),
        },
        "clr_schedule_source": (
            {
                "kind": "explicit_critical_steps",
                "critical_steps": list(profile.explicit_clr_schedule.critical_steps),
            }
            if profile.explicit_clr_schedule is not None
            else {"kind": "decay_and_spike_proxy"}
        ),
        "physical_topology": physical_topology.manifest(),
        "data_plane_loss": (
            profile.network.data_loss.manifest()
            if profile.network.data_loss is not None
            else {"enabled": False}
        ),
        "transport_recovery": (
            profile.network.transport_recovery.manifest()
            if profile.network.transport_recovery is not None
            else {"enabled": False}
        ),
        "packet_trimming": (
            profile.network.packet_trimming.manifest()
            if profile.network.packet_trimming is not None
            else {"enabled": False}
        ),
        "congestion_control": profile.network.congestion_control.manifest(
            profile.network.link_rate
        ),
        "load_balancing": profile.network.load_balancing.manifest(),
        "link_overrides": [
            override.manifest() for override in profile.network.link_overrides
        ],
        "link_failures": [
            failure.manifest() for failure in profile.network.link_failures
        ],
        "switch": profile.network.switch.manifest(),
        "fabric": (
            profile.network.fabric.manifest()
            if profile.network.fabric is not None
            else {
                "buffer_size_mb": 32,
                "pfc_enabled": True,
                "headroom_factor": 3,
                "data_queue_bytes": 0,
                "trimmed_queue_bytes": 0,
            }
        ),
        "profile_config": str(profile_config.resolve()),
        "workload_prefix": str((workload_dir / "ring_3d").resolve()),
        "system_config": str(system_config.resolve()),
        "network_config": str(network_config.resolve()),
        "remote_memory_config": str((output_dir / "remote_memory.json").resolve()),
        "communicator_groups": str((output_dir / "communicator_groups.json").resolve()),
        "logical_topology": str((output_dir / "logical_topology.json").resolve()),
        "experiment_config": str(experiment_config.resolve()),
        "clr_mask": str(clr_mask.resolve()),
        "clr_schedule": schedule_metadata(clr_schedule),
        "telemetry_dir": str((output_dir / "telemetry").resolve()),
        # The raw stream is written as numbered zstd segments, rotated at
        # TRANSPORT_EVENT_SEGMENT_BYTES above; segment .000 always exists,
        # and `cat <base>.* | zstd -d` reconstructs the exact single-file
        # CSV while segments last. In CI they do not last: the sidecar
        # hasher records each segment's uncompressed sha256 and the whole
        # stream's, then deletes it. The recorded path is the family base.
        "transport_event_file": str(
            (output_dir / "ns3" / "transport_events.csv.zst").resolve()
        ),
        "transport_event_first_segment": str(
            (output_dir / "ns3" / "transport_events.csv.zst.000").resolve()
        ),
        "transport_event_summary_file": str(
            (output_dir / "ns3" / "transport_summary.csv").resolve()
        ),
        "counter_files": {
            name: str(path.resolve())
            for name, path in counter_files(
                profile.network.load_balancing, output_dir / "ns3"
            ).items()
        },
    }
    if model_metadata is not None:
        manifest["model_trace"] = model_metadata
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clean", action="store_true", help="replace an existing output directory"
    )
    parser.add_argument("--seed", type=int, help="override the profile seed")
    parser.add_argument(
        "--p-low",
        type=float,
        help="override the low logical-admission selection probability (0, 0.01]",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        help="override the high logical-admission selection probability",
    )
    parser.add_argument(
        "--clr-decay-rate",
        type=float,
        help="CLR probability decay over the full normalized training run",
    )
    parser.add_argument(
        "--clr-epoch-steps",
        type=int,
        help="steps between Gaussian CLR epoch-boundary spikes",
    )
    parser.add_argument(
        "--clr-spike-stddev-steps",
        type=float,
        help="Gaussian CLR epoch-boundary spike width in steps",
    )
    parser.add_argument(
        "--clr-spike-amplitude",
        type=float,
        help="Gaussian CLR epoch-boundary spike amplitude",
    )
    arguments = parser.parse_args()
    clr_schedule_parameters = ClrScheduleParameters(
        **{
            field: value
            for field, value in {
                "decay_rate": arguments.clr_decay_rate,
                "epoch_steps": arguments.clr_epoch_steps,
                "spike_stddev_steps": arguments.clr_spike_stddev_steps,
                "spike_amplitude": arguments.clr_spike_amplitude,
            }.items()
            if value is not None
        }
    )
    manifest = materialize(
        arguments.profile.resolve(),
        arguments.output.resolve(),
        arguments.clean,
        seed_override=arguments.seed,
        p_low=arguments.p_low,
        p_high=arguments.p_high,
        clr_schedule_parameters=clr_schedule_parameters,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
