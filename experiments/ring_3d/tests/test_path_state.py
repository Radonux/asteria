"""path_state under every spray_uniform selector and none."""

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
UNIFORM = {"mode": "spray_uniform"}
HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
}
FLOW_HOLD = {**HOLD, "selector": "reps_flow_hold"}
MRC_HOLD = {**HOLD, "selector": "mrc_hold"}
ORACLE = {
    "mode": "spray_uniform",
    "selector": "reps_oracle",
    "grade_cost_source": "switch_queue",
}
REDIRECT = {"mode": "spray_uniform", "selector": "ack_redirect"}
REPS_SPINE = {"mode": "spray_uniform", "selector": "reps_spine"}


class PathStateTest(unittest.TestCase):
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

    def test_every_spray_uniform_selector_writes_it(self) -> None:
        selectors = (UNIFORM, HOLD, FLOW_HOLD, MRC_HOLD, ORACLE, REDIRECT, REPS_SPINE)
        for balancing in selectors:
            for state in ("connection", "message"):
                with self.subTest(balancing=balancing, state=state):
                    manifest, config = self.generated({**balancing, "path_state": state})
                    self.assertIn(f"\nSPRAY_PATH_STATE {state}\n", config)
                    self.assertEqual(manifest["load_balancing"]["path_state"], state)

    def test_it_is_written_last(self) -> None:
        _, config = self.generated(
            {
                **FLOW_HOLD,
                "path_seq": True,
                "tail_recovery": True,
                "ack_loss_rtts": 1,
                "path_state": "connection",
            }
        )
        self.assertIn(
            "\nSPRAY_TAIL_RECOVERY on\n"
            "SPRAY_ACK_LOSS_RTTS 1.0\n"
            "SPRAY_PATH_STATE connection\n",
            config,
        )
        _, config = self.generated(
            {**UNIFORM, "ack_loss_rtts": 2, "path_state": "connection"}
        )
        self.assertIn(
            "\nSPRAY_ACK_LOSS_RTTS 2.0\nSPRAY_PATH_STATE connection\n", config
        )

    def test_unset_is_not_written(self) -> None:
        for balancing in (UNIFORM, HOLD, REPS_SPINE):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated(balancing)
                self.assertNotIn("SPRAY_PATH_STATE", config)
                self.assertNotIn("path_state", manifest["load_balancing"])

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            (
                {**HOLD, "path_state": value},
                "path_state must be 'message' or 'connection'",
            )
            for value in ("flow", "", None, True, 1)
        ]
        refusals += [
            ({"mode": "ecmp", "path_state": "connection"}, "takes no other key"),
            ({"mode": "spray_policy", "path_state": "connection"}, "does not take"),
            ({"mode": "ev_hash", "path_state": "connection"}, "does not take"),
            (
                {"mode": "ev_hash", "selector": "reps", "path_state": "connection"},
                "does not take",
            ),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
