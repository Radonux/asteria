"""stray_path_seq under spray_uniform with path_seq true."""

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
RECEIVER = {
    "path_seq": True,
    "tail_recovery": True,
    "tail_loss_rtts": 4,
    "receiver_hold": True,
    "receiver_tail_missing": True,
}
ORACLE = {
    "mode": "spray_uniform",
    "selector": "reps_oracle",
    "grade_cost_source": "switch_queue",
}
PATH_SEQ = {"path_seq": True}


class StrayPathSeqTest(unittest.TestCase):
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

    def test_written_where_read(self) -> None:
        for stray in ("fresh", "closed"):
            with self.subTest(stray=stray):
                manifest, config = self.generated(
                    {**HOLD, **RECEIVER, "sentinel_mode": "duplicate", "copy_expiry": "deadline",
                     "stray_path_seq": stray}
                )
                self.assertIn(
                    f"\nSPRAY_COPY_EXPIRY deadline\nSPRAY_STRAY_PATH_SEQ {stray}\n", config
                )
                self.assertEqual(manifest["load_balancing"]["stray_path_seq"], stray)
        for balancing in (
            {"mode": "spray_uniform", **PATH_SEQ},
            {"mode": "spray_uniform", "selector": "reps_spine", **PATH_SEQ},
            {"mode": "spray_uniform", "selector": "ack_redirect", **PATH_SEQ},
            {"mode": "spray_uniform", "selector": "reps_flow_hold", **{k: v for k, v in HOLD.items()
                                                                       if k != "selector"}, **PATH_SEQ},
        ):
            with self.subTest(balancing=balancing):
                _, config = self.generated({**balancing, "stray_path_seq": "closed"})
                self.assertIn("\nSPRAY_PATH_SEQ on\n", config)
                self.assertIn("\nSPRAY_STRAY_PATH_SEQ closed\n", config)
        _, config = self.generated(
            {**ORACLE, **PATH_SEQ, "stray_path_seq": "closed", "oracle_hops": "forward"}
        )
        self.assertIn("\nSPRAY_STRAY_PATH_SEQ closed\nREPS_ORACLE_HOPS forward\n", config)

    def test_unset_is_not_written(self) -> None:
        manifest, config = self.generated({**HOLD, **RECEIVER})
        self.assertNotIn("SPRAY_STRAY_PATH_SEQ", config)
        self.assertNotIn("stray_path_seq", manifest["load_balancing"])

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**HOLD, **RECEIVER, "stray_path_seq": value}, "stray_path_seq must be 'fresh' or 'closed'")
            for value in ("open", "", None, True, 1)
        ]
        refusals += [
            ({**HOLD, "stray_path_seq": "closed"}, "stray_path_seq requires path_seq true"),
            ({**HOLD, "path_seq": False, "stray_path_seq": "fresh"}, "stray_path_seq requires path_seq true"),
            ({"mode": "spray_uniform", "stray_path_seq": "closed"}, "stray_path_seq requires path_seq true"),
            ({**ORACLE, "stray_path_seq": "closed"}, "stray_path_seq requires path_seq true"),
            ({"mode": "spray_policy", "stray_path_seq": "closed"}, "does not take"),
            ({"mode": "ev_hash", "selector": "reps", "stray_path_seq": "closed"}, "does not take"),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
