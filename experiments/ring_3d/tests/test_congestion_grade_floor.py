from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.generate import load_profile, materialize

PROFILE = REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"


class CongestionGradeFloorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def write(self, balancing: dict[str, object]) -> None:
        self.document["network"]["load_balancing"] = balancing
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str]:
        self.write(balancing)
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        config = (output / "network_config.txt").read_text(encoding="utf-8")
        return manifest["load_balancing"], config

    def test_writes_floor_before_loss_thresholds(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "trim_thresholds": [0.1, 0.2, 0.3],
                "congestion_grade_floor": 2,
                "loss_thresholds": [0.01, 0.02, 0.03],
            }
        )
        self.assertIn(
            "\nSPRAY_TRIM_THRESHOLDS 0.1 0.2 0.3\n"
            "SPRAY_CONGESTION_GRADE_FLOOR 2\n"
            "SPRAY_LOSS_THRESHOLDS 0.01 0.02 0.03\n",
            config,
        )
        self.assertEqual(manifest["congestion_grade_floor"], 2)

    def test_every_grade_is_accepted(self) -> None:
        for grade in range(4):
            with self.subTest(grade=grade):
                manifest, config = self.generated(
                    {"mode": "spray_policy", "congestion_grade_floor": grade}
                )
                self.assertIn(f"\nSPRAY_CONGESTION_GRADE_FLOOR {grade}\n", config)
                self.assertEqual(manifest["congestion_grade_floor"], grade)

    def test_unset_writes_nothing(self) -> None:
        manifest, config = self.generated({"mode": "spray_policy"})
        self.assertNotIn("SPRAY_CONGESTION_GRADE_FLOOR", config)
        self.assertNotIn("congestion_grade_floor", manifest)

    def test_refusals(self) -> None:
        policy = {"mode": "spray_policy"}
        refusals = [
            (
                {**policy, "congestion_grade_floor": value},
                "congestion_grade_floor must be",
            )
            for value in (-1, 4, 1.5, 2.0, True, "2")
        ]
        refusals += [
            ({"mode": "spray_uniform", "congestion_grade_floor": 1}, "takes no other"),
            ({"mode": "ev_hash", "congestion_grade_floor": 1}, "'ops' does not take"),
            ({"mode": "ecmp", "congestion_grade_floor": 1}, "takes no other key"),
        ]
        refusals += [
            (
                {
                    "mode": "spray_uniform",
                    "selector": "ack_redirect",
                    "scope": scope,
                    "congestion_grade_floor": 1,
                },
                "does not take",
            )
            for scope in ("per_receiver", "per_connection")
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
