"""Switch ECN and queue-threshold configuration for ring_3d profiles.

``network.switch`` picks how the switches mark and bound their queues. The
``default`` profile is what every profile ran before the knob existed: the
HPCC-era ECN table, scaled by ``congestion_control.ecn_threshold_scale``, with
a RED marking probability that the profile may now set.

This module imports nothing else from the package, so ``topology`` can import
it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

SWITCH_PROFILES = ("default",)
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
# The marking probability at KMAX that the generator wrote for every link speed
# before it could be set.
DEFAULT_ECN_MARKING_PROBABILITY = 0.2


def _probability_map(probability: float) -> str:
    """The PMAX_MAP line, one probability for every link speed."""
    entries = " ".join(f"{rate} {probability!r}" for rate, _, _ in ECN_THRESHOLDS_KB)
    return f"PMAX_MAP {len(ECN_THRESHOLDS_KB)} {entries}\n"


@dataclass(frozen=True)
class DefaultSwitch:
    """The generator's historical switch: HPCC ECN table, settable RED pmax."""

    ecn_marking_probability: float = DEFAULT_ECN_MARKING_PROBABILITY

    @property
    def profile(self) -> str:
        return "default"

    def ecn_threshold_maps(self, scaled_maps: str) -> str:
        """The KMAX_MAP and KMIN_MAP lines: the congestion-control knob's."""
        return scaled_maps

    def probability_map(self) -> str:
        return _probability_map(self.ecn_marking_probability)

    def manifest(self) -> dict[str, str | float]:
        return {
            "profile": self.profile,
            "ecn_marking_probability": self.ecn_marking_probability,
        }


SwitchProfile = DefaultSwitch


@dataclass(frozen=True)
class SwitchRequest:
    """``network.switch`` as written, before the fabric is known."""

    profile: str
    ecn_marking_probability: float | None

    def default_switch(self) -> DefaultSwitch:
        """The ``default`` profile, which needs nothing from the fabric."""
        return DefaultSwitch(
            ecn_marking_probability=(
                DEFAULT_ECN_MARKING_PROBABILITY
                if self.ecn_marking_probability is None
                else self.ecn_marking_probability
            )
        )


def load_switch_request(document: dict[str, Any]) -> SwitchRequest:
    """Parse the optional knob, defaulting to the historical switch."""
    if "switch" not in document:
        return SwitchRequest(profile="default", ecn_marking_probability=None)
    switch = document["switch"]
    allowed = {"profile", "ecn_marking_probability"}
    if not isinstance(switch, dict) or not set(switch) <= allowed:
        raise ValueError(f"network.switch may contain only {sorted(allowed)}")
    profile = switch.get("profile", "default")
    if not isinstance(profile, str) or profile not in SWITCH_PROFILES:
        raise ValueError(
            f"network.switch.profile must be one of {list(SWITCH_PROFILES)}"
        )
    probability = switch.get("ecn_marking_probability")
    if probability is not None:
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
    return SwitchRequest(profile=profile, ecn_marking_probability=probability)
