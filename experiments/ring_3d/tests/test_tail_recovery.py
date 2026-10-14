"""path_seq under every spray_uniform selector, tail_recovery, its delay and
its loss report guard."""

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
ORACLE = {
    "mode": "spray_uniform",
    "selector": "reps_oracle",
    "grade_cost_source": "switch_queue",
}
REDIRECT = {"mode": "spray_uniform", "selector": "ack_redirect"}
REPS_SPINE = {"mode": "spray_uniform", "selector": "reps_spine"}
TAIL = {"path_seq": True, "tail_recovery": True}


class TailRecoveryTest(unittest.TestCase):
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

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str]:
        self.write(balancing)
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        config = (output / "network_config.txt").read_text(encoding="utf-8")
        return manifest, config

    def test_path_seq_under_every_spray_uniform_selector(self) -> None:
        for balancing in (UNIFORM, HOLD, ORACLE, REDIRECT, REPS_SPINE):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated({**balancing, "path_seq": True})
                self.assertIn("\nSPRAY_PATH_SEQ on\n", config)
                self.assertIs(manifest["load_balancing"]["path_seq"], True)
                self.assertNotIn("SPRAY_TAIL_RECOVERY", config)
                _, config = self.generated({**balancing, "path_seq": False})
                self.assertIn("\nSPRAY_PATH_SEQ off\n", config)

    def test_uniform_draw_writes_tail_recovery(self) -> None:
        manifest, config = self.generated({**UNIFORM, **TAIL})
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "SPRAY_PATH_SEQ on\n"
            "SPRAY_TAIL_RECOVERY on\n",
            config,
        )
        self.assertNotIn("PATH_SELECTOR", config)
        self.assertNotIn("SPRAY_TAIL_RECOVERY_DELAY_NS", config)
        self.assertEqual(
            manifest["load_balancing"],
            {"mode": "spray_uniform", "path_seq": True, "tail_recovery": True},
        )
        manifest, config = self.generated(
            {**UNIFORM, **TAIL, "tail_recovery_delay_ns": 0}
        )
        self.assertIn(
            "\nSPRAY_TAIL_RECOVERY on\nSPRAY_TAIL_RECOVERY_DELAY_NS 0\n", config
        )
        self.assertEqual(manifest["load_balancing"]["tail_recovery_delay_ns"], 0)
        _, config = self.generated({**UNIFORM, "tail_recovery": False})
        self.assertIn("\nSPRAY_TAIL_RECOVERY off\n", config)
        self.assertNotIn("SPRAY_PATH_SEQ", config)

    def test_selectors_write_tail_recovery(self) -> None:
        for balancing in (HOLD, ORACLE, REDIRECT):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated(
                    {**balancing, **TAIL, "tail_recovery_delay_ns": 25000}
                )
                self.assertIn(
                    "SPRAY_TAIL_RECOVERY on\nSPRAY_TAIL_RECOVERY_DELAY_NS 25000\n",
                    config,
                )
                self.assertIn(f"PATH_SELECTOR {balancing['selector']}\n", config)
                self.assertEqual(
                    manifest["load_balancing"]["tail_recovery_delay_ns"], 25000
                )
        _, config = self.generated(
            {**HOLD, **TAIL, "receiver_hold": True, "sentinel_mode": "duplicate"}
        )
        self.assertIn(
            "\nSPRAY_SENTINEL_MODE duplicate\n"
            "SPRAY_PATH_SEQ on\n"
            "SPRAY_RECEIVER_HOLD on\n"
            "SPRAY_TAIL_RECOVERY on\n",
            config,
        )

    def test_tail_recovery_counts_at_most_8_spines(self) -> None:
        for balancing in (UNIFORM, HOLD):
            for spines, refused in ((8, False), (9, True)):
                self.write({**balancing, **TAIL}, spines=spines)
                with self.subTest(balancing=balancing, spines=spines):
                    if refused:
                        with self.assertRaisesRegex(
                            ValueError, "tail_recovery counts at most 8 spines"
                        ):
                            load_profile(self.profile_path)
                    else:
                        load_profile(self.profile_path)
        for balancing in (
            {**UNIFORM, "path_seq": True},
            {**UNIFORM, "path_seq": True, "tail_recovery": False},
        ):
            self.write(balancing, spines=9)
            load_profile(self.profile_path)

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = []
        for balancing in (UNIFORM, HOLD, ORACLE, REDIRECT):
            refusals += [
                ({**balancing, "tail_recovery": True}, "tail_recovery requires path_seq true"),
                (
                    {**balancing, "path_seq": False, "tail_recovery": True},
                    "tail_recovery requires path_seq true",
                ),
                (
                    {**balancing, "path_seq": True, "tail_recovery_delay_ns": 0},
                    "tail_recovery_delay_ns requires tail_recovery true",
                ),
                (
                    {**balancing, "path_seq": True, "tail_recovery": False,
                     "tail_recovery_delay_ns": 1000},
                    "tail_recovery_delay_ns requires tail_recovery true",
                ),
                ({**balancing, **TAIL, "tail_recovery": "on"}, "tail_recovery must be a boolean"),
                (
                    {**balancing, **TAIL, "tail_recovery_delay_ns": -1},
                    "tail_recovery_delay_ns must be a nonnegative integer",
                ),
                (
                    {**balancing, **TAIL, "tail_recovery_delay_ns": 1.5},
                    "tail_recovery_delay_ns must be a nonnegative integer",
                ),
                (
                    {**balancing, **TAIL, "tail_recovery_delay_ns": True},
                    "tail_recovery_delay_ns must be a nonnegative integer",
                ),
                (
                    {**balancing, **TAIL, "tail_recovery_delay_ns": 2**64 - 1},
                    "tail_recovery_delay_ns must be below",
                ),
            ]
        refusals += [
            ({**UNIFORM, "path_seq": "on"}, "path_seq must be a boolean"),
            ({**UNIFORM, "hold_scope": "leaf"}, "takes no other key"),
            ({**UNIFORM, "selector": "ops", "path_seq": True}, "takes no other key"),
        ]
        # Only spray_uniform reads the keys.
        for key, value in (
            ("path_seq", True),
            ("tail_recovery", True),
            ("tail_recovery_delay_ns", 0),
            ("tail_loss_rtts", 1),
        ):
            refusals += [
                ({"mode": "ecmp", key: value}, "takes no other key"),
                ({"mode": "spray_policy", key: value}, "does not take"),
                ({"mode": "ev_hash", key: value}, "does not take"),
                ({"mode": "ev_hash", "selector": "reps", key: value}, "does not take"),
            ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)

    def test_tail_loss_rtts_is_written_after_the_delay(self) -> None:
        manifest, config = self.generated({**UNIFORM, **TAIL, "tail_loss_rtts": 2})
        self.assertIn(
            "\nSPRAY_TAIL_RECOVERY on\nSPRAY_TAIL_LOSS_RTTS 2.0\n", config
        )
        self.assertNotIn("PATH_SELECTOR", config)
        self.assertEqual(manifest["load_balancing"]["tail_loss_rtts"], 2.0)
        manifest, config = self.generated(
            {**UNIFORM, **TAIL, "tail_loss_rtts": 0, "ack_loss_rtts": 1}
        )
        self.assertIn(
            "\nSPRAY_TAIL_RECOVERY on\n"
            "SPRAY_TAIL_LOSS_RTTS 0.0\n"
            "SPRAY_ACK_LOSS_RTTS 1.0\n",
            config,
        )
        self.assertEqual(manifest["load_balancing"]["tail_loss_rtts"], 0.0)
        for balancing in (HOLD, REPS_SPINE):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated(
                    {
                        **balancing,
                        **TAIL,
                        "tail_recovery_delay_ns": 0,
                        "tail_loss_rtts": 0.5,
                    }
                )
                self.assertIn(f"PATH_SELECTOR {balancing['selector']}\n", config)
                self.assertIn(
                    "\nSPRAY_TAIL_RECOVERY on\n"
                    "SPRAY_TAIL_RECOVERY_DELAY_NS 0\n"
                    "SPRAY_TAIL_LOSS_RTTS 0.5\n",
                    config,
                )
                self.assertEqual(manifest["load_balancing"]["tail_loss_rtts"], 0.5)

    def test_tail_loss_rtts_unset_is_not_written(self) -> None:
        for balancing in (UNIFORM, HOLD, REPS_SPINE):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated({**balancing, **TAIL})
                self.assertNotIn("SPRAY_TAIL_LOSS_RTTS", config)
                self.assertNotIn("tail_loss_rtts", manifest["load_balancing"])

    def test_tail_loss_rtts_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = []
        for balancing in (UNIFORM, HOLD, REPS_SPINE):
            refusals += [
                (
                    {**balancing, "tail_loss_rtts": 1},
                    "tail_loss_rtts requires tail_recovery true",
                ),
                (
                    {**balancing, "path_seq": True, "tail_loss_rtts": 0},
                    "tail_loss_rtts requires tail_recovery true",
                ),
                (
                    {**balancing, "path_seq": True, "tail_recovery": False,
                     "tail_loss_rtts": 1},
                    "tail_loss_rtts requires tail_recovery true",
                ),
            ]
            for value in (-1, -0.5, True, "1", None, float("inf"), float("nan")):
                refusals.append(
                    (
                        {**balancing, **TAIL, "tail_loss_rtts": value},
                        "tail_loss_rtts must be a nonnegative number",
                    )
                )
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)

    def test_spray_uniform_requires_selective_repair(self) -> None:
        self.document["network"]["transport_recovery"]["selective_repair"] = False
        self.write({**UNIFORM, **TAIL})
        with self.assertRaisesRegex(ValueError, "requires network.transport_recovery"):
            load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
