"""The network.switch knob: the RED marking probability."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.generate import load_profile, materialize

PROFILES = REPOSITORY_ROOT / "experiments/ring_3d/profiles"
RATES = (
    25_000_000_000,
    40_000_000_000,
    100_000_000_000,
    200_000_000_000,
    400_000_000_000,
    2_400_000_000_000,
)
HISTORICAL_PMAX = (
    "PMAX_MAP 6 25000000000 0.2 40000000000 0.2 100000000000 0.2 "
    "200000000000 0.2 400000000000 0.2 2400000000000 0.2\n"
)


def _map(key: str, value: object) -> str:
    return f"{key} 6 " + " ".join(f"{rate} {value}" for rate in RATES) + "\n"


def _trimming_clos() -> dict[str, Any]:
    """Eight hosts, two per leaf, four spines, 200 Gb/s, FTD trimming."""
    return json.loads((PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8"))


class SwitchKnobTests(unittest.TestCase):
    def _materialize(self, document: dict[str, Any]) -> tuple[dict[str, Any], str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            manifest = materialize(path, output)
            config = (output / "network_config.txt").read_text(encoding="utf-8")
        # Paths name the temporary directory; the settings are what is compared.
        return manifest, config.replace(temporary_directory, "<tmp>")

    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_absent_knob_writes_the_historical_switch(self) -> None:
        absent = _trimming_clos()
        explicit = _trimming_clos()
        explicit["network"]["switch"] = {
            "profile": "default",
            "ecn_marking_probability": 0.2,
        }
        manifest, config = self._materialize(absent)
        _, explicit_config = self._materialize(explicit)
        self.assertIn(HISTORICAL_PMAX, config)
        self.assertEqual(
            manifest["switch"], {"profile": "default", "ecn_marking_probability": 0.2}
        )
        self.assertEqual(config, explicit_config)

    def test_marking_probability_reaches_every_link_speed(self) -> None:
        for probability in (1, 0.5, 0.05):
            document = _trimming_clos()
            document["network"]["switch"] = {"ecn_marking_probability": probability}
            _, baseline = self._materialize(_trimming_clos())
            manifest, config = self._materialize(document)
            with self.subTest(probability=probability):
                self.assertIn(_map("PMAX_MAP", float(probability)), config)
                self.assertNotIn(HISTORICAL_PMAX, config)
                self.assertEqual(
                    manifest["switch"]["ecn_marking_probability"], float(probability)
                )
                # Only the probability moves: thresholds and queues stay.
                for key in ("KMAX_MAP", "KMIN_MAP", "DATA_QUEUE_BYTES"):
                    line = next(l for l in baseline.splitlines() if l.startswith(key))
                    self.assertIn(line + "\n", config)

    def test_marking_probability_refuses_values_outside_zero_to_one(self) -> None:
        for value in (0, -0.1, 1.0001, True, "0.5", float("nan")):
            document = _trimming_clos()
            document["network"]["switch"] = {"ecn_marking_probability": value}
            with self.subTest(value=value):
                self._refused(
                    document, r"ecn_marking_probability must be a number in \(0, 1\]"
                )

    def test_switch_refuses_unknown_keys_and_profiles(self) -> None:
        document = _trimming_clos()
        document["network"]["switch"] = {"profile": "default", "kmin": 1}
        self._refused(document, "network.switch may contain only")
        document["network"]["switch"] = {"profile": "uec"}
        self._refused(document, "network.switch.profile must be one of")
        document["network"]["switch"] = "default"
        self._refused(document, "network.switch may contain only")


if __name__ == "__main__":
    unittest.main()
