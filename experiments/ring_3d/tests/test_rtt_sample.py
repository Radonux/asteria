"""congestion_control.rtt_sample under nscc, and load_balancing
duplicate_report, true or retract, with coverage_first."""

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
HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
    "path_seq": True,
    "tail_recovery": True,
}


class RttSampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def write(self, control: dict[str, object], balancing: dict[str, object]) -> None:
        self.document["network"]["congestion_control"] = control
        self.document["network"]["load_balancing"] = balancing
        self.document["network"]["spine_count"] = 4
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")

    def generated(self, control: dict[str, object], balancing: dict[str, object]) -> tuple[dict, str]:
        self.write(control, balancing)
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        return manifest, (output / "network_config.txt").read_text(encoding="utf-8")

    def test_rtt_sample(self) -> None:
        manifest, config = self.generated({"mode": "nscc", "window_state": "connection", "rtt_sample": "karn"}, HOLD)
        self.assertIn("\nNSCC_WINDOW_STATE connection\nNSCC_RTT_SAMPLE karn\n", config)
        self.assertEqual(manifest["congestion_control"]["rtt_sample"], "karn")
        _, config = self.generated({"mode": "nscc"}, HOLD)
        self.assertNotIn("NSCC_RTT_SAMPLE", config)
        for control, message in (({"mode": "nscc", "rtt_sample": "first"}, "'any' or 'karn'"),
                                 ({"mode": "dcqcn", "rtt_sample": "karn"}, "rtt_sample requires mode 'nscc'")):
            self.write(control, HOLD)
            with self.subTest(control=control):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)

    def test_duplicate_report(self) -> None:
        for balancing in (HOLD, {"mode": "spray_uniform", "path_seq": True}):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated(
                    {"mode": "nscc"}, {**balancing, "coverage_first": True, "duplicate_report": True})
                self.assertIn("\nSPRAY_COVERAGE_FIRST on\nSPRAY_DUPLICATE_REPORT on\n", config)
                self.assertTrue(manifest["load_balancing"]["duplicate_report"])
        _, config = self.generated({"mode": "nscc"}, {**HOLD, "coverage_first": True})
        self.assertNotIn("SPRAY_DUPLICATE_REPORT", config)
        for balancing in ({**HOLD, "duplicate_report": True},
                          {**HOLD, "coverage_first": False, "duplicate_report": True}):
            self.write({"mode": "nscc"}, balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, "duplicate_report requires coverage_first true"):
                    load_profile(self.profile_path)

    def test_duplicate_report_retract(self) -> None:
        manifest, config = self.generated(
            {"mode": "nscc"}, {**HOLD, "coverage_first": True, "duplicate_report": "retract"})
        self.assertIn("\nSPRAY_COVERAGE_FIRST on\nSPRAY_DUPLICATE_REPORT retract\n", config)
        self.assertEqual(manifest["load_balancing"]["duplicate_report"], "retract")
        for balancing, message in (
            ({**HOLD, "coverage_first": True, "duplicate_report": "later"}, "a boolean or 'retract'"),
            ({**HOLD, "duplicate_report": "retract"}, "duplicate_report requires coverage_first true"),
            ({**HOLD, "coverage_first": True, "duplicate_report": "retract", "hold_scope": "leaf"},
             "'retract' requires selector 'reps_hold' or 'mrc_hold' with hold_scope 'auto'"),
            ({**HOLD, "coverage_first": True, "duplicate_report": "retract", "hold_rule": "llr",
              "loss_p0": 0.0},
             "'retract' requires selector"),
            ({"mode": "spray_uniform", "path_seq": True, "coverage_first": True, "duplicate_report": "retract"},
             "'retract' requires selector"),
        ):
            self.write({"mode": "nscc"}, balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
