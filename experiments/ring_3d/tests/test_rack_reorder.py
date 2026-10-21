"""rack_reorder_ns under spray_uniform with any selector."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.generate import materialize

PROFILE = REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
}


class RackReorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str]:
        self.document["network"]["load_balancing"] = balancing
        self.document["network"]["spine_count"] = 4
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        return manifest, (output / "network_config.txt").read_text(encoding="utf-8")

    def test_written_under_every_selector(self) -> None:
        for balancing in (
            HOLD,
            {"mode": "spray_uniform", "path_seq": True},
            {"mode": "spray_uniform", "selector": "reps_spine"},
        ):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated({**balancing, "rack_reorder_ns": 3081})
                self.assertIn("\nSPRAY_RACK_REORDER_NS 3081\n", config)
                self.assertEqual(manifest["load_balancing"]["rack_reorder_ns"], 3081)

    def test_absent_is_not_written(self) -> None:
        manifest, config = self.generated(HOLD)
        self.assertNotIn("SPRAY_RACK_REORDER_NS", config)
        self.assertNotIn("rack_reorder_ns", manifest["load_balancing"])

    def test_rejects_a_window_that_is_not_positive(self) -> None:
        for value in (0, -1, 1.5, "3081"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "rack_reorder_ns"):
                self.generated({**HOLD, "rack_reorder_ns": value})


if __name__ == "__main__":
    unittest.main()
