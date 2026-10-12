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
KEYS = (
    "held_exploration",
    "held_sentinel_interval_ns",
    "pool_decay_interval_ns",
    "pool_hold_propagates",
)


class HeldExplorationTest(unittest.TestCase):
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

    def test_writes_sentinel_and_interval_after_epsilon(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "held_exploration": "sentinel",
                "held_sentinel_interval_ns": 50000,
            }
        )
        self.assertIn(
            "\nSPRAY_EPSILON 0.02\n"
            "SPRAY_HELD_EXPLORATION sentinel\n"
            "SPRAY_HELD_SENTINEL_INTERVAL_NS 50000\n"
            "SPRAY_CANDIDATES ",
            config,
        )
        self.assertEqual(manifest["held_exploration"], "sentinel")
        self.assertEqual(manifest["held_sentinel_interval_ns"], 50000)
        self.assertNotIn("pool_decay_interval_ns", manifest)

    def test_sentinel_without_interval(self) -> None:
        manifest, config = self.generated(
            {"mode": "spray_policy", "held_exploration": "sentinel"}
        )
        self.assertIn("\nSPRAY_HELD_EXPLORATION sentinel\nSPRAY_CANDIDATES ", config)
        self.assertNotIn("SPRAY_HELD_SENTINEL_INTERVAL_NS", config)
        self.assertNotIn("held_sentinel_interval_ns", manifest)

    def test_epsilon_is_written_when_set(self) -> None:
        manifest, config = self.generated(
            {"mode": "spray_policy", "held_exploration": "epsilon"}
        )
        self.assertIn("\nSPRAY_HELD_EXPLORATION epsilon\n", config)
        self.assertEqual(manifest["held_exploration"], "epsilon")

    def test_none_under_switch_queue(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "grade_cost_source": "switch_queue",
                "held_exploration": "none",
            }
        )
        self.assertIn("\nSPRAY_HELD_EXPLORATION none\n", config)
        self.assertEqual(manifest["held_exploration"], "none")

    def test_writes_pool_decay_interval_after_pool_weight(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "sender_pool_weight": 0.5,
                "pool_decay_interval_ns": 10000,
            }
        )
        self.assertIn(
            "\nSPRAY_SENDER_POOL_WEIGHT 0.5\nSPRAY_POOL_DECAY_INTERVAL_NS 10000\n",
            config,
        )
        self.assertEqual(manifest["pool_decay_interval_ns"], 10000)

    def test_zero_intervals_are_written_when_set(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "held_exploration": "sentinel",
                "held_sentinel_interval_ns": 0,
                "sender_pool_weight": 1,
                "pool_decay_interval_ns": 0,
            }
        )
        self.assertIn("\nSPRAY_HELD_SENTINEL_INTERVAL_NS 0\n", config)
        self.assertIn("\nSPRAY_POOL_DECAY_INTERVAL_NS 0\n", config)
        self.assertEqual(manifest["held_sentinel_interval_ns"], 0)
        self.assertEqual(manifest["pool_decay_interval_ns"], 0)

    def test_writes_pool_hold_propagates_after_decay_interval(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "sender_pool_weight": 0.5,
                "pool_decay_interval_ns": 10000,
                "pool_hold_propagates": True,
            }
        )
        self.assertIn(
            "\nSPRAY_SENDER_POOL_WEIGHT 0.5\n"
            "SPRAY_POOL_DECAY_INTERVAL_NS 10000\n"
            "SPRAY_POOL_HOLD_PROPAGATES 1\n",
            config,
        )
        self.assertIs(manifest["pool_hold_propagates"], True)

    def test_pool_hold_propagates_false_without_decay_interval(self) -> None:
        manifest, config = self.generated(
            {
                "mode": "spray_policy",
                "sender_pool_weight": 1,
                "pool_hold_propagates": False,
            }
        )
        self.assertIn(
            "\nSPRAY_SENDER_POOL_WEIGHT 1.0\nSPRAY_POOL_HOLD_PROPAGATES 0\n", config
        )
        self.assertNotIn("SPRAY_POOL_DECAY_INTERVAL_NS", config)
        self.assertIs(manifest["pool_hold_propagates"], False)
        self.assertNotIn("pool_decay_interval_ns", manifest)

    def test_unset_writes_nothing(self) -> None:
        manifest, config = self.generated(
            {"mode": "spray_policy", "sender_pool_weight": 0.5}
        )
        self.assertNotIn("SPRAY_HELD_", config)
        self.assertNotIn("SPRAY_POOL_DECAY_", config)
        self.assertNotIn("SPRAY_POOL_HOLD_", config)
        for key in KEYS:
            self.assertNotIn(key, manifest)

    def test_refusals(self) -> None:
        policy = {"mode": "spray_policy"}
        sentinel = {**policy, "held_exploration": "sentinel"}
        pooled = {**policy, "sender_pool_weight": 0.5}
        refusals = [
            (
                {**policy, "held_exploration": "none"},
                "held_exploration 'none' requires grade_cost_source 'switch_queue'",
            ),
            (
                {
                    **policy,
                    "grade_cost_source": "estimate",
                    "held_exploration": "none",
                },
                "held_exploration 'none' requires grade_cost_source 'switch_queue'",
            ),
            (
                {**policy, "held_exploration": "uniform"},
                "held_exploration must be 'epsilon', 'sentinel' or 'none'",
            ),
            (
                {**policy, "held_exploration": True},
                "held_exploration must be 'epsilon', 'sentinel' or 'none'",
            ),
            (
                {**policy, "held_sentinel_interval_ns": 1000},
                "held_sentinel_interval_ns requires held_exploration 'sentinel'",
            ),
            (
                {
                    **policy,
                    "held_exploration": "epsilon",
                    "held_sentinel_interval_ns": 1000,
                },
                "held_sentinel_interval_ns requires held_exploration 'sentinel'",
            ),
            (
                {**sentinel, "held_sentinel_interval_ns": -1},
                "held_sentinel_interval_ns must be a nonnegative integer",
            ),
            (
                {**sentinel, "held_sentinel_interval_ns": 1.5},
                "held_sentinel_interval_ns must be a nonnegative integer",
            ),
            (
                {**sentinel, "held_sentinel_interval_ns": True},
                "held_sentinel_interval_ns must be a nonnegative integer",
            ),
            (
                {**policy, "pool_decay_interval_ns": 1000},
                "pool_decay_interval_ns requires sender_pool_weight above 0",
            ),
            (
                {**policy, "sender_pool_weight": 0, "pool_decay_interval_ns": 1000},
                "pool_decay_interval_ns requires sender_pool_weight above 0",
            ),
            (
                {**policy, "sender_pool_weight": 2, "pool_decay_interval_ns": 1000},
                "sender_pool_weight must be a number in",
            ),
            (
                {**pooled, "pool_decay_interval_ns": -1},
                "pool_decay_interval_ns must be a nonnegative integer",
            ),
            (
                {**pooled, "pool_decay_interval_ns": 1.5},
                "pool_decay_interval_ns must be a nonnegative integer",
            ),
            (
                {**pooled, "pool_decay_interval_ns": True},
                "pool_decay_interval_ns must be a nonnegative integer",
            ),
            (
                {**policy, "pool_hold_propagates": True},
                "pool_hold_propagates requires sender_pool_weight above 0",
            ),
            (
                {**policy, "sender_pool_weight": 0, "pool_hold_propagates": False},
                "pool_hold_propagates requires sender_pool_weight above 0",
            ),
            (
                {**policy, "sender_pool_weight": -1, "pool_hold_propagates": True},
                "sender_pool_weight must be a number in",
            ),
            (
                {**pooled, "pool_hold_propagates": 1},
                "pool_hold_propagates must be a boolean",
            ),
            (
                {**pooled, "pool_hold_propagates": "true"},
                "pool_hold_propagates must be a boolean",
            ),
        ]
        knobs = (
            ("held_exploration", "sentinel"),
            ("held_sentinel_interval_ns", 0),
            ("pool_decay_interval_ns", 0),
            ("pool_hold_propagates", True),
        )
        for knob, value in knobs:
            refusals += [
                ({"mode": "spray_uniform", knob: value}, "takes no other key"),
                ({"mode": "ev_hash", knob: value}, "'ops' does not take"),
                ({"mode": "ecmp", knob: value}, "takes no other key"),
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
            ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
