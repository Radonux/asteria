"""oracle_hops under reps_oracle."""

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
ORACLE = {
    "mode": "spray_uniform",
    "selector": "reps_oracle",
    "grade_cost_source": "switch_queue",
}
HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
}
TAIL = {"path_seq": True, "tail_recovery": True, "tail_loss_rtts": 4}


class OracleHopsTest(unittest.TestCase):
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

    def test_written_last(self) -> None:
        for hops in ("round_trip", "forward"):
            with self.subTest(hops=hops):
                manifest, config = self.generated(
                    {
                        **ORACLE,
                        **TAIL,
                        "oracle_hops": hops,
                        "ack_route": "spray",
                        "ack_route_oracle": True,
                    }
                )
                self.assertIn(
                    "\nACK_ROUTE spray\nACK_ROUTE_ORACLE on\n"
                    f"REPS_ORACLE_HOPS {hops}\n",
                    config,
                )
                self.assertEqual(manifest["load_balancing"]["oracle_hops"], hops)
                self.assertEqual(
                    list(manifest["load_balancing"])[-1], "oracle_hops"
                )
        _, config = self.generated({**ORACLE, "oracle_hops": "forward"})
        self.assertIn(
            "\nSPRAY_GRADE_COST_SOURCE switch_queue\nREPS_ORACLE_HOPS forward\n",
            config,
        )

    def test_unset_is_not_written(self) -> None:
        manifest, config = self.generated({**ORACLE, **TAIL, "ack_route": "spray"})
        self.assertNotIn("REPS_ORACLE_HOPS", config)
        self.assertNotIn("oracle_hops", manifest["load_balancing"])

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**ORACLE, "oracle_hops": value}, "oracle_hops must be 'round_trip' or 'forward'")
            for value in ("reverse", "", None, True, 1)
        ]
        refusals += [
            ({"mode": "spray_uniform", "oracle_hops": "forward"}, "takes no other key"),
            ({**HOLD, "oracle_hops": "forward"}, "does not take"),
            (
                {"mode": "spray_uniform", "selector": "reps_spine", "oracle_hops": "forward"},
                "does not take",
            ),
            (
                {"mode": "spray_uniform", "selector": "ack_redirect", "oracle_hops": "forward"},
                "does not take",
            ),
            ({"mode": "spray_policy", "oracle_hops": "forward"}, "does not take"),
            (
                {"mode": "ev_hash", "selector": "reps", "oracle_hops": "forward"},
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
