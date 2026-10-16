"""ack_route reverse_set and repath, with ack_route_set_size and
ack_route_set_probe_rtts and ack_route_set_reply, under spray_uniform."""

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


class AckRouteModesTest(unittest.TestCase):
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
        return manifest, (output / "network_config.txt").read_text(encoding="utf-8")

    def test_written(self) -> None:
        for balancing in (HOLD, {"mode": "spray_uniform", "path_seq": True, "tail_recovery": True}):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated({**balancing, "ack_route": "reverse_set",
                                                   "ack_route_set_size": 3})
                self.assertIn("\nACK_ROUTE reverse_set\nACK_ROUTE_SET_SIZE 3\n", config)
                self.assertEqual(manifest["load_balancing"]["ack_route_set_size"], 3)
                self.assertNotIn("ACK_ROUTE_SET_PROBE_RTTS", config)
                manifest, config = self.generated({**balancing, "ack_route": "reverse_set",
                                                   "ack_route_set_size": 4, "ack_route_set_probe_rtts": 16})
                self.assertIn("\nACK_ROUTE reverse_set\nACK_ROUTE_SET_SIZE 4\nACK_ROUTE_SET_PROBE_RTTS 16.0\n",
                              config)
                self.assertEqual(manifest["load_balancing"]["ack_route_set_probe_rtts"], 16.0)
                self.assertNotIn("ACK_ROUTE_SET_REPLY", config)
                manifest, config = self.generated({**balancing, "ack_route": "reverse_set",
                                                   "ack_route_set_probe_rtts": 4, "ack_route_set_reply": "answer"})
                self.assertIn("\nACK_ROUTE reverse_set\nACK_ROUTE_SET_PROBE_RTTS 4.0\nACK_ROUTE_SET_REPLY answer\n",
                              config)
                self.assertEqual(manifest["load_balancing"]["ack_route_set_reply"], "answer")
                manifest, config = self.generated({**balancing, "ack_route": "repath",
                                                   "coverage_first": True})
                self.assertIn("\nACK_ROUTE repath\n", config)
                self.assertNotIn("ACK_ROUTE_SET_SIZE", config)
                self.assertEqual(manifest["load_balancing"]["ack_route"], "repath")
        _, config = self.generated({**HOLD, "ack_route": "reverse_set"})
        self.assertIn("\nACK_ROUTE reverse_set\n", config)
        self.assertNotIn("ACK_ROUTE_SET_SIZE", config)

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**HOLD, "ack_route": "mirror"}, "'data_spine', 'spray', 'reverse_set' or 'repath'"),
            ({**HOLD, "ack_route": "spray", "ack_route_set_size": 2},
             "ack_route_set_size requires ack_route 'reverse_set'"),
            ({**HOLD, "ack_route_set_size": 2}, "ack_route_set_size requires ack_route 'reverse_set'"),
            ({**HOLD, "ack_route": "reverse_set", "ack_route_set_size": 0}, "ack_route_set_size"),
            ({**HOLD, "ack_route": "spray", "ack_route_set_probe_rtts": 4},
             "ack_route_set_probe_rtts requires ack_route 'reverse_set'"),
            ({**HOLD, "ack_route": "reverse_set", "ack_route_set_probe_rtts": 0}, "ack_route_set_probe_rtts"),
            ({**HOLD, "ack_route": "spray", "ack_route_set_reply": "answer"},
             "ack_route_set_reply requires ack_route 'reverse_set'"),
            ({**HOLD, "ack_route": "reverse_set", "ack_route_set_reply": "control"},
             "ack_route_set_reply must be 'data' or 'answer'"),
            ({**HOLD, "tail_recovery": False, "ack_route": "repath"}, "'repath' requires tail_recovery true"),
            ({"mode": "spray_uniform", "path_seq": True, "ack_route": "repath"},
             "'repath' requires tail_recovery true"),
            ({**HOLD, "ack_route": "reverse_set", "ack_route_oracle": True},
             "ack_route_oracle requires ack_route 'spray'"),
            ({**HOLD, "ack_route": "repath", "answer_path_seq": True},
             "answer_path_seq requires ack_route 'spray'"),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
