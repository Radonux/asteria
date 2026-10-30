"""ack_route and ack_route_oracle under every spray_uniform selector and none,
and the answer_drop counts with their breakdown by spine."""

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
FLOW_HOLD = {**HOLD, "selector": "reps_flow_hold"}
MRC_HOLD = {**HOLD, "selector": "mrc_hold"}
ORACLE = {
    "mode": "spray_uniform",
    "selector": "reps_oracle",
    "grade_cost_source": "switch_queue",
}
REDIRECT = {"mode": "spray_uniform", "selector": "ack_redirect"}
REPS_SPINE = {"mode": "spray_uniform", "selector": "reps_spine"}


class AckRouteTest(unittest.TestCase):
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

    def test_every_spray_uniform_selector_writes_them(self) -> None:
        selectors = (UNIFORM, HOLD, FLOW_HOLD, MRC_HOLD, ORACLE, REDIRECT, REPS_SPINE)
        for balancing in selectors:
            for route in ("data_spine", "spray"):
                with self.subTest(balancing=balancing, route=route):
                    manifest, config = self.generated({**balancing, "ack_route": route})
                    self.assertIn(f"\nACK_ROUTE {route}\n", config)
                    self.assertNotIn("ACK_ROUTE_ORACLE", config)
                    self.assertEqual(manifest["load_balancing"]["ack_route"], route)
            for oracle in (True, False):
                with self.subTest(balancing=balancing, oracle=oracle):
                    manifest, config = self.generated(
                        {**balancing, "ack_route": "spray", "ack_route_oracle": oracle}
                    )
                    self.assertIn(
                        "\nACK_ROUTE spray\n"
                        f"ACK_ROUTE_ORACLE {'on' if oracle else 'off'}\n",
                        config,
                    )
                    self.assertEqual(
                        manifest["load_balancing"]["ack_route_oracle"], oracle
                    )

    def test_they_are_written_last(self) -> None:
        _, config = self.generated(
            {
                **FLOW_HOLD,
                "path_seq": True,
                "tail_recovery": True,
                "ack_loss_rtts": 1,
                "receiver_hold": True,
                "receiver_tail_missing": True,
                "path_state": "connection",
                "ack_route": "spray",
                "ack_route_oracle": True,
            }
        )
        self.assertIn(
            "\nSPRAY_ACK_LOSS_RTTS 1.0\n"
            "SPRAY_RECEIVER_TAIL_MISSING on\n"
            "SPRAY_PATH_STATE connection\n"
            "ACK_ROUTE spray\n"
            "ACK_ROUTE_ORACLE on\n",
            config,
        )
        _, config = self.generated(
            {**ORACLE, "path_state": "connection", "ack_route": "spray"}
        )
        self.assertIn("\nSPRAY_PATH_STATE connection\nACK_ROUTE spray\n", config)

    def test_unset_is_not_written(self) -> None:
        for balancing in (UNIFORM, HOLD, ORACLE, REPS_SPINE):
            with self.subTest(balancing=balancing):
                manifest, config = self.generated(balancing)
                self.assertNotIn("ACK_ROUTE", config)
                self.assertNotIn("ack_route", manifest["load_balancing"])
                self.assertNotIn("ack_route_oracle", manifest["load_balancing"])

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            (
                {**HOLD, "ack_route": value},
                "ack_route must be 'data_spine', 'spray', 'reverse_set' or 'repath'",
            )
            for value in ("data", "", None, True, 1)
        ]
        refusals += [
            (
                {**UNIFORM, "ack_route": "spray", "ack_route_oracle": value},
                "ack_route_oracle must be a boolean",
            )
            for value in ("on", 1, None)
        ]
        refusals += [
            (
                {**balancing, **route, "ack_route_oracle": True},
                "ack_route_oracle requires ack_route 'spray'",
            )
            for balancing in (UNIFORM, HOLD, ORACLE, REPS_SPINE)
            for route in ({}, {"ack_route": "data_spine"})
        ]
        refusals += [
            ({"mode": "ecmp", "ack_route": "spray"}, "takes no other key"),
            ({"mode": "spray_policy", "ack_route": "spray"}, "does not take"),
            ({"mode": "ev_hash", "ack_route": "spray"}, "does not take"),
            (
                {"mode": "ev_hash", "selector": "reps", "ack_route_oracle": True},
                "does not take",
            ),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


class AnswerDropTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.ns3 = Path(directory.name) / "ns3"
        self.ns3.mkdir()
        self.summary = self.ns3 / "transport_summary.csv"

    def test_zero_rows_read_zero(self) -> None:
        self.summary.write_text(
            "event,plane,event_count,total_bytes\n"
            "answer_drop_down,control,0,0\n"
            "answer_drop_up,control,0,0\n",
            encoding="utf-8",
        )
        transport = _summarize_transport_events(self.ns3)
        self.assertEqual(transport["answer_drop_up_count"], 0)
        self.assertEqual(transport["answer_drop_down_count"], 0)
        self.assertEqual(transport["answer_drop_up_by_spine"], {})
        self.assertEqual(transport["answer_drop_down_by_spine"], {})
        self.assertEqual(transport["event_count"], 0)

    def test_counts_break_down_by_spine(self) -> None:
        self.summary.write_text(
            "event,plane,event_count,total_bytes\n"
            "answer_drop_down,control,5,330\n"
            "answer_drop_down_spine3,control,5,330\n"
            "answer_drop_up,control,9,594\n"
            "answer_drop_up_spine1,control,2,132\n"
            "answer_drop_up_spine3,control,7,462\n"
            "link_error_drop,control,9,594\n"
            "switch_blackhole_drop,control,5,330\n",
            encoding="utf-8",
        )
        transport = _summarize_transport_events(self.ns3)
        self.assertEqual(transport["answer_drop_up_count"], 9)
        self.assertEqual(transport["answer_drop_down_count"], 5)
        self.assertEqual(transport["answer_drop_up_by_spine"], {1: 2, 3: 7})
        self.assertEqual(transport["answer_drop_down_by_spine"], {3: 5})
        # Every dropped answer is a link error or a black-hole drop already.
        self.assertEqual(transport["event_count"], 14)
        self.assertNotIn("answer_drop_up", transport["event_counts"])
        self.assertNotIn("answer_drop_up_spine3", transport["event_counts"])

    def test_refusals(self) -> None:
        for event, plane, message in (
            ("answer_drop_up", "data", "control plane"),
            ("answer_drop_down_spine3", "data", "control plane"),
            ("answer_drop_sideways", "control", "invalid transport event"),
            ("answer_drop_up_spinex", "control", "invalid transport event"),
            ("answer_drop_spine3", "control", "invalid transport event"),
        ):
            self.summary.write_text(
                f"event,plane,event_count,total_bytes\n{event},{plane},1,0\n",
                encoding="utf-8",
            )
            with self.subTest(event=event, plane=plane):
                with self.assertRaisesRegex(ValueError, message):
                    _summarize_transport_events(self.ns3)


if __name__ == "__main__":
    unittest.main()
