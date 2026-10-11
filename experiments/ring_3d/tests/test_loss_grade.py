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


class LossGradeTest(unittest.TestCase):
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

    def test_writes_loss_thresholds_and_interval_when_set(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "loss_thresholds": [0.01, 0.05, 1],
                "loss_interval_ns": 20000,
            }
        )
        self.assertIn(
            "\nSPRAY_CONGESTION_THRESHOLDS 0.25 0.5 0.75\n"
            "SPRAY_LOSS_THRESHOLDS 0.01 0.05 1.0\n"
            "SPRAY_LOSS_INTERVAL_NS 20000\n"
            "SPRAY_ABSENCE_FRACTION_OF_MEDIAN ",
            config,
        )
        self.assertEqual(manifest["loss_thresholds"], (0.01, 0.05, 1.0))
        self.assertEqual(manifest["loss_interval_ns"], 20000)

    def test_writes_thresholds_alone_and_after_trim_thresholds(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "trim_thresholds": [0.1, 0.2, 0.3],
                "loss_thresholds": [0.1, 0.2, 0.3],
            }
        )
        self.assertIn(
            "SPRAY_TRIM_THRESHOLDS 0.1 0.2 0.3\n"
            "SPRAY_LOSS_THRESHOLDS 0.1 0.2 0.3\n"
            "SPRAY_ABSENCE_FRACTION_OF_MEDIAN ",
            config,
        )
        self.assertNotIn("SPRAY_LOSS_INTERVAL_NS", config)
        self.assertNotIn("loss_interval_ns", manifest)

    def test_zero_interval_is_written_when_set(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "loss_thresholds": [0.1, 0.2, 0.3],
                "loss_interval_ns": 0,
            }
        )
        self.assertIn("\nSPRAY_LOSS_INTERVAL_NS 0\n", config)
        self.assertEqual(manifest["loss_interval_ns"], 0)

    def test_switch_queue_takes_loss_thresholds(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "grade_cost_source": "switch_queue",
                "loss_thresholds": [0.1, 0.2, 0.3],
            }
        )
        self.assertIn("\nSPRAY_LOSS_THRESHOLDS 0.1 0.2 0.3\n", config)
        self.assertEqual(manifest["loss_thresholds"], (0.1, 0.2, 0.3))

    def test_unset_writes_nothing(self) -> None:
        manifest, config = self.generated({"mode": "spray_policy"})
        self.assertNotIn("SPRAY_LOSS_", config)
        self.assertNotIn("loss_thresholds", manifest)
        self.assertNotIn("loss_interval_ns", manifest)

    def test_refusals(self) -> None:
        policy = {"mode": "spray_policy"}
        thresholds = [0.1, 0.2, 0.3]
        refusals = [
            (
                {**policy, "loss_interval_ns": 1000},
                "loss_interval_ns requires loss_thresholds",
            ),
            (
                {**policy, "loss_thresholds": [0.2, 0.1, 0.3]},
                "loss_thresholds must increase",
            ),
            (
                {**policy, "loss_thresholds": [0.1, 0.1, 0.3]},
                "loss_thresholds must increase",
            ),
            (
                {**policy, "loss_thresholds": [0.1, 0.2]},
                "loss_thresholds must be a list",
            ),
            (
                {**policy, "loss_thresholds": [0, 0.2, 0.3]},
                "loss_thresholds must be a number",
            ),
            (
                {**policy, "loss_thresholds": [0.1, 0.2, 1.5]},
                "loss_thresholds must be a number",
            ),
            (
                {**policy, "loss_thresholds": [0.1, 0.2, True]},
                "loss_thresholds must be a number",
            ),
            (
                {**policy, "loss_thresholds": thresholds, "loss_interval_ns": -1},
                "loss_interval_ns must be a nonnegative integer",
            ),
            (
                {**policy, "loss_thresholds": thresholds, "loss_interval_ns": 1.5},
                "loss_interval_ns must be a nonnegative integer",
            ),
            (
                {**policy, "loss_thresholds": thresholds, "loss_interval_ns": True},
                "loss_interval_ns must be a nonnegative integer",
            ),
            (
                {"mode": "spray_uniform", "loss_thresholds": thresholds},
                "takes no other key",
            ),
            ({"mode": "spray_uniform", "loss_interval_ns": 0}, "takes no other key"),
            ({"mode": "ev_hash", "loss_thresholds": thresholds}, "'ops' does not take"),
            ({"mode": "ecmp", "loss_thresholds": thresholds}, "takes no other key"),
        ]
        refusals += [
            (
                {
                    "mode": "spray_uniform",
                    "selector": "ack_redirect",
                    "scope": scope,
                    knob: value,
                },
                "does not take",
            )
            for scope in ("per_receiver", "per_connection")
            for knob, value in (
                ("loss_thresholds", thresholds),
                ("loss_interval_ns", 0),
            )
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
