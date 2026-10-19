"""reps_spine under spray_uniform: REPS over the spines, no loss hold."""

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
SPINE = {"mode": "spray_uniform", "selector": "reps_spine"}


class RepsSpineTest(unittest.TestCase):
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

    def test_writes_reps_defaults(self) -> None:
        manifest, config = self.generated(SPINE)
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "PATH_SELECTOR reps_spine\n"
            "REPS_BUFFER_SIZE 8\n"
            "REPS_FREEZING_TIMEOUT_NS 10000000\n",
            config,
        )
        self.assertEqual(
            manifest["load_balancing"],
            {
                "mode": "spray_uniform",
                "selector": "reps_spine",
                "buffer_size": 8,
                "freezing_timeout_ns": 10_000_000,
            },
        )
        self.assertNotIn("SPRAY_LOSS", config)
        self.assertNotIn("SPRAY_HELD", config)
        self.assertNotIn("SPRAY_PATH_SEQ", config)
        self.assertNotIn("SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"])
        self.assertNotIn("RECEIVER_HOLD_OUTPUT_FILE", config)

    def test_writes_given_keys_in_order(self) -> None:
        manifest, config = self.generated(
            {
                **SPINE,
                "buffer_size": 4,
                "freezing_timeout_ns": 1000,
                "path_seq": True,
                "tail_recovery": True,
            }
        )
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "PATH_SELECTOR reps_spine\n"
            "REPS_BUFFER_SIZE 4\n"
            "REPS_FREEZING_TIMEOUT_NS 1000\n"
            "SPRAY_PATH_SEQ on\n"
            "SPRAY_TAIL_RECOVERY on\n",
            config,
        )
        self.assertEqual(
            manifest["load_balancing"],
            {
                "mode": "spray_uniform",
                "selector": "reps_spine",
                "buffer_size": 4,
                "freezing_timeout_ns": 1000,
                "path_seq": True,
                "tail_recovery": True,
            },
        )

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**SPINE, "tail_recovery": True}, "tail_recovery requires path_seq true"),
            (
                {**SPINE, "path_seq": True, "tail_recovery_delay_ns": 0},
                "tail_recovery_delay_ns requires tail_recovery true",
            ),
            ({**SPINE, "buffer_size": 0}, "buffer_size"),
            ({"mode": "ev_hash", "selector": "reps_spine"}, "selector must be one of"),
            ({"mode": "spray_policy", "selector": "reps_spine"}, "does not take"),
        ]
        for key, value in (
            ("loss_thresholds", [0.01, 0.02, 0.05]),
            ("loss_interval_ns", 100000),
            ("loss_hold_clean_sentinels", 4),
            ("held_exploration", "sentinel"),
            ("held_sentinel_interval_ns", 1000),
            ("held_sentinel_spread", 64),
            ("hold_scope", "host"),
            ("hold_auto_leaves", 2),
            ("hold_rule", "llr"),
            ("hold_release", "trickle"),
            ("sentinel_mode", "duplicate"),
            ("held_loss_rtts", 2),
            ("receiver_hold", True),
            ("receiver_hold_clean", 8),
            ("receiver_hold_slots", 8),
            ("grade_cost_source", "switch_queue"),
            ("ev_set_size", 8),
        ):
            refusals.append(({**SPINE, key: value}, "'reps_spine' does not take"))
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)

    def test_tail_recovery_counts_every_spine(self) -> None:
        balancing = {**SPINE, "path_seq": True, "tail_recovery": True}
        self.write(balancing, spines=256)
        load_profile(self.profile_path)
        self.write(balancing, spines=257)
        with self.assertRaisesRegex(ValueError, "spine_count must not exceed 256"):
            load_profile(self.profile_path)

    def test_reorder_tolerance_is_required(self) -> None:
        self.document["network"]["transport_recovery"]["selective_repair"] = False
        self.write(SPINE)
        with self.assertRaisesRegex(ValueError, "requires network.transport"):
            load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
