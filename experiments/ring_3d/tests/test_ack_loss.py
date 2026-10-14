"""ack_loss_rtts under every spray_uniform selector, and the
ack_loss_declared count."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.analyze import _summarize_transport_events
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
REPS_SPINE = {"mode": "spray_uniform", "selector": "reps_spine"}
TAIL = {"path_seq": True, "tail_recovery": True}


class AckLossTest(unittest.TestCase):
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

    def test_uniform_draw_writes_ack_loss_rtts(self) -> None:
        manifest, config = self.generated({**UNIFORM, "ack_loss_rtts": 2})
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\nSPRAY_ACK_LOSS_RTTS 2.0\n", config
        )
        self.assertNotIn("PATH_SELECTOR", config)
        self.assertEqual(
            manifest["load_balancing"],
            {"mode": "spray_uniform", "ack_loss_rtts": 2.0},
        )
        _, config = self.generated({**UNIFORM, **TAIL, "ack_loss_rtts": 0.5})
        self.assertIn(
            "\nSPRAY_PATH_SEQ on\nSPRAY_TAIL_RECOVERY on\nSPRAY_ACK_LOSS_RTTS 0.5\n",
            config,
        )

    def test_selectors_write_ack_loss_rtts_last(self) -> None:
        manifest, config = self.generated(
            {**HOLD, **TAIL, "tail_recovery_delay_ns": 0, "ack_loss_rtts": 1.5}
        )
        self.assertIn("PATH_SELECTOR reps_hold\n", config)
        self.assertIn(
            "\nSPRAY_TAIL_RECOVERY_DELAY_NS 0\nSPRAY_ACK_LOSS_RTTS 1.5\n", config
        )
        self.assertEqual(manifest["load_balancing"]["ack_loss_rtts"], 1.5)
        manifest, config = self.generated({**REPS_SPINE, "ack_loss_rtts": 3})
        self.assertIn(
            "\nPATH_SELECTOR reps_spine\n"
            "REPS_BUFFER_SIZE 8\n"
            "REPS_FREEZING_TIMEOUT_NS 10000000\n"
            "SPRAY_ACK_LOSS_RTTS 3.0\n",
            config,
        )
        self.assertEqual(manifest["load_balancing"]["ack_loss_rtts"], 3.0)

    def test_unset_is_not_written(self) -> None:
        for balancing in (UNIFORM, HOLD, REPS_SPINE):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated(balancing)
                self.assertNotIn("SPRAY_ACK_LOSS_RTTS", config)
                self.assertNotIn("ack_loss_rtts", manifest["load_balancing"])

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = []
        for balancing in (UNIFORM, HOLD, REPS_SPINE):
            for value in (0, 0.0, -1, -0.5, True, "2", float("inf")):
                refusals.append(
                    (
                        {**balancing, "ack_loss_rtts": value},
                        "ack_loss_rtts must be a positive number",
                    )
                )
        refusals += [
            ({"mode": "spray_policy", "ack_loss_rtts": 2}, "does not take"),
            ({"mode": "ecmp", "ack_loss_rtts": 2}, "takes no other key"),
            ({"mode": "ev_hash", "ack_loss_rtts": 2}, "does not take"),
            ({"mode": "ev_hash", "selector": "reps", "ack_loss_rtts": 2}, "does not take"),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)

    def test_ack_loss_declared_is_counted_on_the_control_plane(self) -> None:
        ns3 = self.directory / "ns3"
        ns3.mkdir()
        summary = ns3 / "transport_summary.csv"
        summary.write_text(
            "event,plane,event_count,total_bytes\nrto_fired,control,3,0\n",
            encoding="utf-8",
        )
        self.assertEqual(_summarize_transport_events(ns3)["ack_loss_declared_count"], 0)
        summary.write_text(
            "event,plane,event_count,total_bytes\n"
            "rto_fired,control,3,0\n"
            "ack_loss_declared,control,6,0\n",
            encoding="utf-8",
        )
        transport = _summarize_transport_events(ns3)
        self.assertEqual(transport["ack_loss_declared_count"], 6)
        self.assertEqual(transport["rto_fired_count"], 3)


if __name__ == "__main__":
    unittest.main()
