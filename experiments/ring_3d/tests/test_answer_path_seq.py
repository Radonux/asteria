"""coverage_first under spray_uniform, and answer_path_seq with its keys
under the loss hold selectors."""

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
}
SPRAY = {"path_seq": True, "tail_recovery": True, "ack_route": "spray"}
ORACLE = {"mode": "spray_uniform", "selector": "reps_oracle", "grade_cost_source": "switch_queue"}


class AnswerPathSeqTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def write(self, balancing: dict[str, object]) -> None:
        self.document["network"]["load_balancing"] = balancing
        self.document["network"]["spine_count"] = 4
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str]:
        self.write(balancing)
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        config = (output / "network_config.txt").read_text(encoding="utf-8")
        return manifest, config

    def test_coverage_first_under_every_selector(self) -> None:
        for balancing in (
            {**HOLD, **SPRAY, "sentinel_mode": "duplicate", "copy_expiry": "deadline"},
            {"mode": "spray_uniform", "path_seq": True},
            {**ORACLE, "path_seq": True},
            {"mode": "spray_uniform", "selector": "reps_spine"},
        ):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated({**balancing, "coverage_first": True})
                self.assertIn("\nSPRAY_COVERAGE_FIRST on\n", config)
                self.assertTrue(manifest["load_balancing"]["coverage_first"])
        _, config = self.generated(
            {**HOLD, **SPRAY, "sentinel_mode": "duplicate", "copy_expiry": "deadline",
             "coverage_first": True}
        )
        self.assertIn("\nSPRAY_COPY_EXPIRY deadline\nSPRAY_COVERAGE_FIRST on\n", config)

    def test_answer_path_seq_written_last_with_its_trace(self) -> None:
        manifest, config = self.generated(
            {**HOLD, **SPRAY, "coverage_first": True, "answer_path_seq": True,
             "answer_probe_interval_ns": 500000, "answer_hold_clean": 4}
        )
        self.assertIn(
            "\nSPRAY_COVERAGE_FIRST on\nSPRAY_ANSWER_PATH_SEQ on\n"
            "SPRAY_ANSWER_PROBE_INTERVAL_NS 500000\nSPRAY_ANSWER_HOLD_CLEAN 4\n",
            config,
        )
        self.assertIn("ANSWER_HOLD_OUTPUT_FILE ", config)
        self.assertTrue(
            any(str(path).endswith("answer_holds.csv") for path in manifest["counter_files"].values())
            if isinstance(manifest.get("counter_files"), dict)
            else "answer_holds.csv" in json.dumps(manifest)
        )

    def test_unset_is_not_written(self) -> None:
        manifest, config = self.generated({**HOLD, **SPRAY})
        for key in ("SPRAY_COVERAGE_FIRST", "SPRAY_ANSWER_PATH_SEQ", "ANSWER_HOLD_OUTPUT_FILE"):
            self.assertNotIn(key, config)
        for key in ("coverage_first", "answer_path_seq"):
            self.assertNotIn(key, manifest["load_balancing"])
        self.assertNotIn("answer_holds.csv", json.dumps(manifest))

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**HOLD, **SPRAY, "coverage_first": "yes"}, "coverage_first"),
            ({**HOLD, "path_seq": True, "answer_path_seq": True},
             "answer_path_seq requires ack_route 'spray' and path_seq true"),
            ({**HOLD, "ack_route": "spray", "answer_path_seq": True},
             "answer_path_seq requires ack_route 'spray' and path_seq true"),
            ({**HOLD, **SPRAY, "answer_probe_interval_ns": 1000},
             "answer_probe_interval_ns requires answer_path_seq true"),
            ({**HOLD, **SPRAY, "answer_path_seq": False, "answer_hold_clean": 2},
             "answer_hold_clean requires answer_path_seq true"),
            ({**HOLD, **SPRAY, "answer_path_seq": True, "answer_hold_clean": 0}, "answer_hold_clean"),
            ({"mode": "spray_uniform", **SPRAY, "answer_path_seq": True}, "takes no other key"),
            ({**ORACLE, **SPRAY, "answer_path_seq": True}, "does not take"),
            ({"mode": "spray_policy", "coverage_first": True}, "does not take"),
            ({"mode": "ev_hash", "selector": "reps", "coverage_first": True}, "does not take"),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
