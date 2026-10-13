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
FLOW_HOLD = {**HOLD, "selector": "reps_flow_hold"}
ORACLE = {
    "mode": "spray_uniform",
    "selector": "reps_oracle",
    "grade_cost_source": "switch_queue",
}
HOLD_SETTINGS = (
    "\nLOAD_BALANCING spray_uniform\n"
    "PATH_SELECTOR {selector}\n"
    "REPS_BUFFER_SIZE 8\n"
    "REPS_FREEZING_TIMEOUT_NS 10000000\n"
    "SPRAY_LOSS_THRESHOLDS 0.01 0.02 0.05\n"
    "SPRAY_LOSS_INTERVAL_NS 100000\n"
    "SPRAY_LOSS_HOLD_CLEAN_SENTINELS 4\n"
    "SPRAY_HELD_EXPLORATION sentinel\n"
)


class RepsSpineSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def write(self, balancing: dict[str, object], spines: int = 4) -> None:
        self.document["network"]["load_balancing"] = balancing
        self.document["network"]["spine_count"] = spines
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str, Path]:
        self.write(balancing)
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        config = (output / "network_config.txt").read_text(encoding="utf-8")
        return manifest, config, output

    def test_reps_hold_writes_the_loss_hold_and_the_trace(self) -> None:
        manifest, config, output = self.generated(HOLD)
        self.assertIn(HOLD_SETTINGS.format(selector="reps_hold"), config)
        self.assertNotIn("SPRAY_HELD_SENTINEL_INTERVAL_NS", config)
        self.assertNotIn("SPRAY_HOLD_SCOPE", config)
        self.assertNotIn("SPRAY_HOLD_AUTO_LEAVES", config)
        self.assertEqual(
            manifest["load_balancing"],
            {
                "mode": "spray_uniform",
                "selector": "reps_hold",
                "buffer_size": 8,
                "freezing_timeout_ns": 10000000,
                "loss_thresholds": (0.01, 0.02, 0.05),
                "loss_interval_ns": 100000,
                "loss_hold_clean_sentinels": 4,
                "held_exploration": "sentinel",
            },
        )
        trace = str((output / "ns3").resolve() / "spine_losses.csv")
        self.assertIn(f"\nSPINE_LOSS_OUTPUT_FILE {trace}\n", config)
        self.assertEqual(manifest["counter_files"]["SPINE_LOSS_OUTPUT_FILE"], trace)
        self.assertNotIn("SPINE_REPORT_OUTPUT_FILE", config)
        self.assertNotIn("SPRAY_GRADE_COST_SOURCE", config)

    def test_reps_hold_writes_set_keys_in_order(self) -> None:
        manifest, config, _ = self.generated(
            {
                **HOLD,
                "buffer_size": 16,
                "freezing_timeout_ns": 5000,
                "held_sentinel_interval_ns": 20000,
                "hold_scope": "host",
            }
        )
        self.assertIn(
            "\nPATH_SELECTOR reps_hold\n"
            "REPS_BUFFER_SIZE 16\n"
            "REPS_FREEZING_TIMEOUT_NS 5000\n"
            "SPRAY_LOSS_THRESHOLDS 0.01 0.02 0.05\n"
            "SPRAY_LOSS_INTERVAL_NS 100000\n"
            "SPRAY_LOSS_HOLD_CLEAN_SENTINELS 4\n"
            "SPRAY_HELD_EXPLORATION sentinel\n"
            "SPRAY_HELD_SENTINEL_INTERVAL_NS 20000\n"
            "SPRAY_HOLD_SCOPE host\n",
            config,
        )
        balancing = manifest["load_balancing"]
        self.assertEqual(balancing["held_sentinel_interval_ns"], 20000)
        self.assertEqual(balancing["hold_scope"], "host")
        self.assertIn("SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"])

    def test_reps_hold_leaf_scope_is_written_when_set(self) -> None:
        manifest, config, _ = self.generated({**HOLD, "hold_scope": "leaf"})
        self.assertIn("\nSPRAY_HOLD_SCOPE leaf\n", config)
        self.assertEqual(manifest["load_balancing"]["hold_scope"], "leaf")
        self.assertIn("SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"])

    def test_reps_hold_auto_scope_is_written_when_set(self) -> None:
        manifest, config, _ = self.generated(
            {**HOLD, "hold_scope": "auto", "hold_auto_leaves": 3}
        )
        self.assertIn("\nSPRAY_HOLD_SCOPE auto\nSPRAY_HOLD_AUTO_LEAVES 3\n", config)
        balancing = manifest["load_balancing"]
        self.assertEqual(balancing["hold_scope"], "auto")
        self.assertEqual(balancing["hold_auto_leaves"], 3)
        self.assertIn("SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"])

    def test_reps_hold_auto_leaves_under_the_default_scope(self) -> None:
        manifest, config, _ = self.generated({**HOLD, "hold_auto_leaves": 2})
        self.assertNotIn("SPRAY_HOLD_SCOPE", config)
        self.assertIn(
            "\nSPRAY_HELD_EXPLORATION sentinel\nSPRAY_HOLD_AUTO_LEAVES 2\n", config
        )
        balancing = manifest["load_balancing"]
        self.assertNotIn("hold_scope", balancing)
        self.assertEqual(balancing["hold_auto_leaves"], 2)

    def test_reps_flow_hold_writes_no_trace(self) -> None:
        manifest, config, _ = self.generated(
            {**FLOW_HOLD, "held_sentinel_interval_ns": 0, "buffer_size": 2}
        )
        self.assertIn("\nPATH_SELECTOR reps_flow_hold\nREPS_BUFFER_SIZE 2\n", config)
        self.assertIn(
            "\nSPRAY_HELD_EXPLORATION sentinel\nSPRAY_HELD_SENTINEL_INTERVAL_NS 0\n",
            config,
        )
        self.assertNotIn("SPRAY_HOLD_SCOPE", config)
        self.assertNotIn("SPINE_LOSS_OUTPUT_FILE", config)
        self.assertNotIn("SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"])
        self.assertEqual(manifest["load_balancing"]["selector"], "reps_flow_hold")
        self.assertNotIn("hold_scope", manifest["load_balancing"])
        self.assertNotIn("SPRAY_HOLD_AUTO_LEAVES", config)

    def test_reps_oracle_writes_the_cost_source(self) -> None:
        manifest, config, _ = self.generated(
            {**ORACLE, "buffer_size": 4, "freezing_timeout_ns": 1000}
        )
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "PATH_SELECTOR reps_oracle\n"
            "REPS_BUFFER_SIZE 4\n"
            "REPS_FREEZING_TIMEOUT_NS 1000\n"
            "SPRAY_GRADE_COST_SOURCE switch_queue\n",
            config,
        )
        self.assertEqual(
            manifest["load_balancing"],
            {
                "mode": "spray_uniform",
                "selector": "reps_oracle",
                "buffer_size": 4,
                "freezing_timeout_ns": 1000,
                "grade_cost_source": "switch_queue",
            },
        )
        self.assertNotIn("SPRAY_LOSS", config)
        self.assertNotIn("SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"])

    def test_refusals(self) -> None:
        required = (
            "requires loss_thresholds, loss_interval_ns, "
            "loss_hold_clean_sentinels and held_exploration 'sentinel'"
        )
        refusals: list[tuple[dict[str, object], str]] = []
        for hold in (HOLD, FLOW_HOLD):
            selector = hold["selector"]
            for key in (
                "loss_thresholds",
                "loss_interval_ns",
                "loss_hold_clean_sentinels",
                "held_exploration",
            ):
                missing = {k: v for k, v in hold.items() if k != key}
                refusals.append((missing, f"'{selector}' {required}"))
            refusals += [
                (
                    {**hold, "held_exploration": exploration},
                    f"'{selector}' requires held_exploration 'sentinel'",
                )
                for exploration in ("epsilon", "none")
            ]
            refusals += [
                ({**hold, "held_exploration": "random"}, "held_exploration must be"),
                (
                    {**hold, "loss_interval_ns": 0},
                    "loss_interval_ns must be a positive",
                ),
                (
                    {**hold, "loss_hold_clean_sentinels": 0},
                    "loss_hold_clean_sentinels must be a positive",
                ),
                (
                    {**hold, "loss_thresholds": [0.05, 0.02, 0.01]},
                    "loss_thresholds must increase",
                ),
                (
                    {**hold, "loss_thresholds": [0.1, 0.5, 1.5]},
                    r"loss_thresholds must be a number in \(0, 1\]",
                ),
                (
                    {**hold, "held_sentinel_interval_ns": -1},
                    "held_sentinel_interval_ns must be a nonnegative integer",
                ),
                ({**hold, "buffer_size": 256}, "buffer_size must be at most 255"),
                (
                    {**hold, "freezing_timeout_ns": 0},
                    "freezing_timeout_ns must be a positive integer",
                ),
            ]
            for key, value in (
                ("grade_cost_source", "switch_queue"),
                ("sender_pool_weight", 0.5),
                ("congestion_grade_floor", 1),
                ("epsilon", 0.1),
                ("scope", "per_receiver"),
            ):
                refusals.append(({**hold, key: value}, f"'{selector}' does not take"))
        refusals += [
            (
                {**HOLD, "hold_scope": "rack"},
                "hold_scope must be 'host', 'leaf' or 'auto'",
            ),
            ({**FLOW_HOLD, "hold_scope": "host"}, "'reps_flow_hold' does not take"),
            ({**FLOW_HOLD, "hold_scope": "leaf"}, "'reps_flow_hold' does not take"),
            ({**FLOW_HOLD, "hold_scope": "auto"}, "'reps_flow_hold' does not take"),
            (
                {**FLOW_HOLD, "hold_auto_leaves": 2},
                "'reps_flow_hold' does not take",
            ),
            *(
                (
                    {**HOLD, "hold_scope": scope, "hold_auto_leaves": 2},
                    "hold_auto_leaves requires hold_scope 'auto'",
                )
                for scope in ("host", "leaf")
            ),
            *(
                (
                    {**HOLD, "hold_auto_leaves": leaves},
                    "hold_auto_leaves must be an integer of at least 2",
                )
                for leaves in (1, 0, -2, True, 2.0, "2")
            ),
            (
                {k: v for k, v in ORACLE.items() if k != "grade_cost_source"},
                "'reps_oracle' requires grade_cost_source 'switch_queue'",
            ),
            (
                {**ORACLE, "grade_cost_source": "estimate"},
                "'reps_oracle' requires grade_cost_source 'switch_queue'",
            ),
            (
                {**ORACLE, "grade_cost_source": "queue"},
                "grade_cost_source must be 'estimate' or 'switch_queue'",
            ),
            ({**ORACLE, "buffer_size": 0}, "buffer_size must be a positive integer"),
        ]
        for key, value in (
            ("loss_thresholds", [0.01, 0.02, 0.05]),
            ("loss_interval_ns", 100000),
            ("loss_hold_clean_sentinels", 4),
            ("held_exploration", "sentinel"),
            ("held_sentinel_interval_ns", 0),
            ("hold_scope", "leaf"),
            ("hold_auto_leaves", 2),
            ("congestion_grade_floor", 1),
        ):
            refusals.append(({**ORACLE, key: value}, "'reps_oracle' does not take"))
        # Each selector names spines, so only spray_uniform reads it.
        for balancing in (HOLD, FLOW_HOLD, ORACLE):
            refusals += [
                ({**balancing, "mode": "ev_hash"}, "selector must be one of"),
                ({**balancing, "mode": "spray_policy"}, "does not take"),
                ({**balancing, "mode": "ecmp"}, "takes no other key"),
            ]
        # And no other selector reads the keys they add.
        for key, value in (("hold_scope", "leaf"), ("hold_auto_leaves", 2)):
            refusals += [
                ({"mode": "spray_policy", key: value}, "does not take"),
                ({"mode": "spray_uniform", key: value}, "takes no other key"),
                ({"mode": "ev_hash", "selector": "reps", key: value}, "does not take"),
                (
                    {"mode": "spray_uniform", "selector": "ack_redirect", key: value},
                    "does not take",
                ),
            ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)

    def test_loss_hold_holds_at_most_32_spines(self) -> None:
        for balancing in (HOLD, FLOW_HOLD):
            for spines, refused in ((32, False), (33, True)):
                self.write(balancing, spines=spines)
                with self.subTest(selector=balancing["selector"], spines=spines):
                    if refused:
                        with self.assertRaisesRegex(
                            ValueError,
                            f"'{balancing['selector']}' holds at most 32 spines",
                        ):
                            load_profile(self.profile_path)
                    else:
                        load_profile(self.profile_path)
        # reps_oracle keeps no per-spine report or hold.
        self.write(ORACLE, spines=33)
        load_profile(self.profile_path)

    def test_reorder_tolerance_is_required(self) -> None:
        for balancing in (HOLD, FLOW_HOLD, ORACLE):
            self.document["network"]["transport_recovery"]["selective_repair"] = False
            self.write(balancing)
            with self.subTest(selector=balancing["selector"]):
                with self.assertRaisesRegex(ValueError, "requires network.transport"):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
