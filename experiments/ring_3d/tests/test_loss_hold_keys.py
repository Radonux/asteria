"""The loss hold's rule, release, exploration and receiver keys, and mrc_hold."""

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
MRC_HOLD = {**HOLD, "selector": "mrc_hold"}
TRICKLE = {
    k: v for k, v in HOLD.items() if k != "loss_hold_clean_sentinels"
} | {"hold_release": "trickle"}
RECEIVER = {**HOLD, "path_seq": True, "receiver_hold": True}


class LossHoldKeysTest(unittest.TestCase):
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
        return manifest, config, (output / "ns3").resolve()

    def test_mrc_hold_writes_mrc_and_the_loss_hold(self) -> None:
        manifest, config, ns3 = self.generated({**MRC_HOLD, "skip_base_rtts": 2})
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "PATH_SELECTOR mrc_hold\n"
            "MRC_SKIP_BASE_RTTS 2.0\n"
            "MRC_PROBE_TIMEOUTS 1.0\n"
            "SPRAY_LOSS_THRESHOLDS 0.01 0.02 0.05\n"
            "SPRAY_LOSS_INTERVAL_NS 100000\n"
            "SPRAY_LOSS_HOLD_CLEAN_SENTINELS 4\n"
            "SPRAY_HELD_EXPLORATION sentinel\n",
            config,
        )
        for key in ("REPS_", "MRC_EV_SET_SIZE", "SPRAY_HOLD_SCOPE"):
            self.assertNotIn(key, config)
        self.assertEqual(
            manifest["load_balancing"],
            {
                "mode": "spray_uniform",
                "selector": "mrc_hold",
                "skip_base_rtts": 2.0,
                "probe_timeouts": 1.0,
                "loss_thresholds": (0.01, 0.02, 0.05),
                "loss_interval_ns": 100000,
                "loss_hold_clean_sentinels": 4,
                "held_exploration": "sentinel",
            },
        )
        self.assertEqual(
            manifest["counter_files"]["SPINE_LOSS_OUTPUT_FILE"],
            str(ns3 / "spine_losses.csv"),
        )
        _, config, _ = self.generated(
            {**MRC_HOLD, "hold_scope": "auto", "hold_auto_leaves": 3}
        )
        self.assertIn("\nSPRAY_HOLD_SCOPE auto\nSPRAY_HOLD_AUTO_LEAVES 3\n", config)

    def test_llr_rule_writes_its_test(self) -> None:
        manifest, config, _ = self.generated(
            {
                **HOLD,
                "hold_rule": "llr",
                "loss_p0": 0.0001,
                "loss_p1": 0.01,
                "hold_alpha": 0.002,
                "hold_beta": 0.003,
            }
        )
        self.assertIn(
            "\nSPRAY_HELD_EXPLORATION sentinel\n"
            "SPRAY_HOLD_RULE llr\n"
            "SPRAY_LOSS_P0 0.0001\n"
            "SPRAY_LOSS_P1 0.01\n"
            "SPRAY_HOLD_ALPHA 0.002\n"
            "SPRAY_HOLD_BETA 0.003\n",
            config,
        )
        balancing = manifest["load_balancing"]
        self.assertEqual(balancing["hold_rule"], "llr")
        self.assertEqual(balancing["hold_beta"], 0.003)
        _, config, _ = self.generated({**HOLD, "hold_rule": "count"})
        self.assertIn("\nSPRAY_HOLD_RULE count\n", config)
        self.assertNotIn("SPRAY_LOSS_P", config)

    def test_trickle_release_drops_the_sentinels_keys(self) -> None:
        for selector in ("reps_hold", "reps_flow_hold", "mrc_hold"):
            with self.subTest(selector=selector):
                manifest, config, _ = self.generated(
                    {**TRICKLE, "selector": selector, "hold_trickle": 0.02}
                )
                self.assertIn(
                    "\nSPRAY_HELD_EXPLORATION sentinel\n"
                    "SPRAY_HOLD_RELEASE trickle\n"
                    "SPRAY_HOLD_TRICKLE 0.02\n",
                    config,
                )
                self.assertNotIn("SPRAY_LOSS_HOLD_CLEAN_SENTINELS", config)
                self.assertNotIn("loss_hold_clean_sentinels", manifest["load_balancing"])
        _, config, _ = self.generated({**TRICKLE, "loss_p1": 0.05})
        self.assertIn("\nSPRAY_LOSS_P1 0.05\n", config)

    def test_duplicate_sentinels_are_written(self) -> None:
        manifest, config, _ = self.generated({**MRC_HOLD, "sentinel_mode": "duplicate"})
        self.assertIn("\nSPRAY_SENTINEL_MODE duplicate\n", config)
        self.assertEqual(manifest["load_balancing"]["sentinel_mode"], "duplicate")

    def test_held_loss_rtts_is_written_only_when_set(self) -> None:
        for selector in ("reps_hold", "reps_flow_hold", "mrc_hold"):
            for rtts in (0, 3):
                with self.subTest(selector=selector, rtts=rtts):
                    manifest, config, _ = self.generated(
                        {**HOLD, "selector": selector, "held_loss_rtts": rtts}
                    )
                    self.assertIn(f"\nSPRAY_HELD_LOSS_RTTS {rtts}\n", config)
                    self.assertEqual(
                        manifest["load_balancing"]["held_loss_rtts"], rtts
                    )
        manifest, config, _ = self.generated(
            {**MRC_HOLD, "sentinel_mode": "duplicate", "held_loss_rtts": 1}
        )
        self.assertIn(
            "\nSPRAY_SENTINEL_MODE duplicate\nSPRAY_HELD_LOSS_RTTS 1\n", config
        )
        manifest, config, _ = self.generated(HOLD)
        self.assertNotIn("SPRAY_HELD_LOSS_RTTS", config)
        self.assertNotIn("held_loss_rtts", manifest["load_balancing"])

    def test_overdue_repair_is_written_only_when_set(self) -> None:
        for selector in ("reps_hold", "reps_flow_hold", "mrc_hold"):
            for value, written in (("evidence", "evidence"), ("probe", "probe"), ("reprobe", "reprobe"),
                                   (False, "off")):
                with self.subTest(selector=selector, value=value):
                    manifest, config, _ = self.generated(
                        {**HOLD, "selector": selector, "held_loss_rtts": 1, "overdue_repair": value}
                    )
                    self.assertIn(
                        f"\nSPRAY_HELD_LOSS_RTTS 1\nSPRAY_OVERDUE_REPAIR {written}\n",
                        config,
                    )
                    self.assertEqual(manifest["load_balancing"]["overdue_repair"], value)
        manifest, config, _ = self.generated(HOLD)
        self.assertNotIn("SPRAY_OVERDUE_REPAIR", config)
        self.assertNotIn("overdue_repair", manifest["load_balancing"])
        with self.assertRaisesRegex(ValueError, "overdue_repair requires held_loss_rtts above 0"):
            self.generated({**HOLD, "held_loss_rtts": 0, "overdue_repair": "probe"})
        with self.assertRaisesRegex(ValueError, "must be 'evidence', 'probe', 'reprobe' or false"):
            self.generated({**HOLD, "overdue_repair": True})
        # Under reprobe the repairs per send, written after it when set.
        manifest, config, _ = self.generated(
            {**HOLD, "held_loss_rtts": 1, "overdue_repair": "reprobe", "overdue_repair_limit": 3}
        )
        self.assertIn(
            "\nSPRAY_HELD_LOSS_RTTS 1\nSPRAY_OVERDUE_REPAIR reprobe\nSPRAY_OVERDUE_REPAIR_LIMIT 3\n", config
        )
        self.assertEqual(manifest["load_balancing"]["overdue_repair_limit"], 3)
        manifest, config, _ = self.generated({**HOLD, "held_loss_rtts": 1, "overdue_repair": "reprobe"})
        self.assertNotIn("SPRAY_OVERDUE_REPAIR_LIMIT", config)
        for value in ("probe", False):
            with self.assertRaisesRegex(ValueError, "overdue_repair_limit requires overdue_repair 'reprobe'"):
                self.generated({**HOLD, "overdue_repair": value, "overdue_repair_limit": 2})
        with self.assertRaisesRegex(ValueError, "overdue_repair_limit"):
            self.generated({**HOLD, "overdue_repair": "reprobe", "overdue_repair_limit": 0})
        with self.assertRaisesRegex(ValueError, "must be 'evidence', 'probe', 'reprobe' or false"):
            self.generated({"mode": "spray_uniform", "overdue_repair": True})

    def test_overdue_reprobe_without_a_loss_hold(self) -> None:
        # reps_spine, reps_oracle and the uniform draw read reprobe and its
        # limit, written last and only when set, after the backend's wait.
        for balancing in (
            {"mode": "spray_uniform", "selector": "reps_spine"},
            {"mode": "spray_uniform", "selector": "reps_oracle", "grade_cost_source": "switch_queue"},
            {"mode": "spray_uniform"},
        ):
            with self.subTest(balancing=balancing):
                manifest, config, _ = self.generated(
                    {**balancing, "path_seq": True, "overdue_repair": "reprobe", "overdue_repair_limit": 3}
                )
                written = config.index("\nSPRAY_OVERDUE_REPAIR reprobe\nSPRAY_OVERDUE_REPAIR_LIMIT 3\n")
                self.assertLess(config.index("\nSPRAY_PATH_SEQ on\n"), written)
                self.assertNotIn("SPRAY_HELD_LOSS_RTTS", config)
                self.assertEqual(
                    (manifest["load_balancing"]["overdue_repair"], manifest["load_balancing"]["overdue_repair_limit"]),
                    ("reprobe", 3),
                )
                self.assertEqual(list(manifest["load_balancing"])[-2:], ["overdue_repair", "overdue_repair_limit"])
                manifest, config, _ = self.generated(balancing)
                self.assertNotIn("SPRAY_OVERDUE_REPAIR", config)
                self.assertNotIn("overdue_repair", manifest["load_balancing"])
                manifest, config, _ = self.generated({**balancing, "overdue_repair": False})
                self.assertIn("\nSPRAY_OVERDUE_REPAIR off\n", config)
                for value in ("evidence", "probe"):
                    with self.assertRaisesRegex(ValueError, "must be 'reprobe' or false without a loss hold selector"):
                        self.generated({**balancing, "overdue_repair": value})
                with self.assertRaisesRegex(ValueError, "overdue_repair_limit requires overdue_repair 'reprobe'"):
                    self.generated({**balancing, "overdue_repair_limit": 3})
        with self.assertRaisesRegex(ValueError, "does not take"):
            self.generated({"mode": "spray_uniform", "selector": "ack_redirect", "overdue_repair": "reprobe"})

    def test_receiver_hold_writes_its_trace(self) -> None:
        for selector in ("reps_hold", "reps_flow_hold", "mrc_hold"):
            with self.subTest(selector=selector):
                manifest, config, ns3 = self.generated(
                    {
                        **RECEIVER,
                        "selector": selector,
                        "receiver_hold_clean": 512,
                        "hold_auto_leaves": 3,
                    }
                )
                self.assertIn(
                    "\nSPRAY_HOLD_AUTO_LEAVES 3\n"
                    "SPRAY_PATH_SEQ on\n"
                    "SPRAY_RECEIVER_HOLD on\n"
                    "SPRAY_RECEIVER_HOLD_CLEAN 512\n",
                    config,
                )
                trace = str(ns3 / "receiver_holds.csv")
                self.assertIn(f"\nRECEIVER_HOLD_OUTPUT_FILE {trace}\n", config)
                self.assertEqual(
                    manifest["counter_files"]["RECEIVER_HOLD_OUTPUT_FILE"], trace
                )
                balancing = manifest["load_balancing"]
                self.assertIs(balancing["path_seq"], True)
                self.assertIs(balancing["receiver_hold"], True)
        manifest, config, _ = self.generated({**HOLD, "path_seq": True})
        self.assertIn("\nSPRAY_PATH_SEQ on\n", config)
        self.assertNotIn("RECEIVER_HOLD", config)
        self.assertNotIn("RECEIVER_HOLD_OUTPUT_FILE", manifest["counter_files"])
        _, config, _ = self.generated({**HOLD, "path_seq": False, "receiver_hold": False})
        self.assertIn("\nSPRAY_PATH_SEQ off\nSPRAY_RECEIVER_HOLD off\n", config)
        self.assertNotIn("RECEIVER_HOLD_OUTPUT_FILE", config)

    def test_receiver_hold_widen_is_written_only_when_set(self) -> None:
        manifest, config, _ = self.generated({**RECEIVER, "receiver_hold_widen": 2})
        self.assertIn("\nSPRAY_RECEIVER_HOLD on\nSPRAY_RECEIVER_HOLD_WIDEN 2\n", config)
        self.assertEqual(manifest["load_balancing"]["receiver_hold_widen"], 2)
        manifest, config, _ = self.generated(RECEIVER)
        self.assertNotIn("SPRAY_RECEIVER_HOLD_WIDEN", config)
        self.assertNotIn("receiver_hold_widen", manifest["load_balancing"])

    def test_receiver_hold_slots_is_written_only_when_set(self) -> None:
        manifest, config, _ = self.generated({**RECEIVER, "receiver_hold_slots": 8})
        self.assertIn("\nSPRAY_RECEIVER_HOLD on\nSPRAY_RECEIVER_HOLD_SLOTS 8\n", config)
        self.assertEqual(manifest["load_balancing"]["receiver_hold_slots"], 8)
        manifest, config, _ = self.generated(RECEIVER)
        self.assertNotIn("SPRAY_RECEIVER_HOLD_SLOTS", config)
        self.assertNotIn("receiver_hold_slots", manifest["load_balancing"])

    def test_every_run_writes_the_rto_trace(self) -> None:
        for balancing in ({"mode": "ecmp"}, {"mode": "spray_uniform"}, HOLD, MRC_HOLD):
            with self.subTest(balancing=balancing):
                manifest, config, ns3 = self.generated(balancing)
                trace = str(ns3 / "rto_events.csv")
                self.assertIn(f"\nRTO_EVENT_OUTPUT_FILE {trace}\n", config)
                self.assertEqual(manifest["counter_files"]["RTO_EVENT_OUTPUT_FILE"], trace)

    def test_receiver_hold_names_every_spine(self) -> None:
        for spines, refused in ((256, False), (257, True)):
            self.write(RECEIVER, spines=spines)
            with self.subTest(spines=spines):
                if refused:
                    with self.assertRaisesRegex(
                        ValueError, "spine_count must not exceed 256"
                    ):
                        load_profile(self.profile_path)
                else:
                    load_profile(self.profile_path)

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**MRC_HOLD, "ev_set_size": 8}, "'mrc_hold' does not take"),
            ({**MRC_HOLD, "buffer_size": 8}, "'mrc_hold' does not take"),
            ({**MRC_HOLD, "skip_base_rtts": 0}, "skip_base_rtts must be a positive"),
            ({**MRC_HOLD, "held_exploration": "epsilon"}, "requires held_exploration"),
            (
                {k: v for k, v in MRC_HOLD.items() if k != "loss_hold_clean_sentinels"},
                "'mrc_hold' requires loss_thresholds, loss_interval_ns, "
                "loss_hold_clean_sentinels",
            ),
            ({**MRC_HOLD, "mode": "ev_hash"}, "selector must be one of"),
            ({**HOLD, "hold_rule": "sprt"}, "hold_rule must be 'count' or 'llr'"),
            ({**HOLD, "hold_release": "drip"}, "hold_release must be 'sentinels'"),
            ({**HOLD, "sentinel_mode": "copy"}, "sentinel_mode must be 'data'"),
            ({**HOLD, "held_loss_rtts": -1}, "held_loss_rtts must be a nonnegative integer"),
            ({**HOLD, "held_loss_rtts": 1.5}, "held_loss_rtts must be a nonnegative integer"),
            ({**HOLD, "held_loss_rtts": True}, "held_loss_rtts must be a nonnegative integer"),
            ({**HOLD, "hold_beta": 0.01}, "hold_beta requires hold_rule 'llr'"),
            (
                {**HOLD, "hold_rule": "llr", "hold_trickle": 0.1},
                "hold_trickle requires hold_release 'trickle'",
            ),
            (
                {**HOLD, "hold_release": "sentinels", "hold_trickle": 0.1},
                "hold_trickle requires hold_release 'trickle'",
            ),
            ({**TRICKLE, "hold_beta": 0.01}, "hold_beta requires hold_rule 'llr'"),
            (
                {**HOLD, "hold_rule": "llr", "loss_p0": 0.01, "loss_p1": 0.01},
                "loss_p0 must be below",
            ),
            ({**TRICKLE, "loss_p0": 0.002}, "loss_p0 must be below"),
            ({**HOLD, "hold_rule": "llr", "loss_p0": -0.1}, r"loss_p0 must be a number in \[0, 1\)"),
            ({**HOLD, "hold_rule": "llr", "loss_p1": 1}, r"loss_p1 must be a number in \(0, 1\)"),
            ({**HOLD, "hold_rule": "llr", "hold_alpha": 0.5}, r"hold_alpha must be a number in \(0, 0.5\)"),
            ({**HOLD, "hold_rule": "llr", "hold_beta": 0}, r"hold_beta must be a number in \(0, 0.5\)"),
            ({**TRICKLE, "hold_trickle": 0.6}, r"hold_trickle must be a number in \(0, 0.5\]"),
            ({**TRICKLE, "hold_trickle": 0}, r"hold_trickle must be a number in \(0, 0.5\]"),
            (
                {**TRICKLE, "loss_hold_clean_sentinels": 4},
                "loss_hold_clean_sentinels is read only under hold_release 'sentinels'",
            ),
            (
                {**TRICKLE, "held_sentinel_interval_ns": 1000},
                "held_sentinel_interval_ns is read only under hold_release 'sentinels'",
            ),
            (
                {k: v for k, v in TRICKLE.items() if k != "loss_interval_ns"},
                "requires loss_thresholds, loss_interval_ns and held_exploration "
                "'sentinel' under hold_release 'trickle'",
            ),
            ({**HOLD, "path_seq": "on"}, "path_seq must be a boolean"),
            ({**HOLD, "receiver_hold": True}, "receiver_hold requires path_seq true"),
            (
                {**HOLD, "path_seq": False, "receiver_hold": True},
                "receiver_hold requires path_seq true",
            ),
            (
                {**HOLD, "path_seq": True, "receiver_hold_clean": 8},
                "receiver_hold_clean requires receiver_hold true",
            ),
            ({**RECEIVER, "receiver_hold_clean": 0}, "receiver_hold_clean must be a positive"),
            (
                {**HOLD, "path_seq": True, "receiver_hold_slots": 8},
                "receiver_hold_slots requires receiver_hold true",
            ),
            ({**RECEIVER, "receiver_hold_slots": 0}, "receiver_hold_slots must be an integer from 1"),
            ({**RECEIVER, "receiver_hold_slots": 128}, "receiver_hold_slots must be an integer from 1"),
            ({**RECEIVER, "receiver_hold_slots": True}, "receiver_hold_slots must be an integer from 1"),
            (
                {**HOLD, "path_seq": True, "receiver_hold_widen": 2},
                "receiver_hold_widen requires receiver_hold true",
            ),
            ({**RECEIVER, "receiver_hold_widen": 1}, "receiver_hold_widen must be an integer of at least 2"),
            ({**RECEIVER, "receiver_hold_widen": True}, "receiver_hold_widen must be an integer of at least 2"),
            (
                {**FLOW_HOLD, "path_seq": True, "receiver_hold": True, "receiver_hold_widen": 2},
                "'reps_flow_hold' does not take",
            ),
            (
                {**HOLD, "hold_scope": "host", "hold_auto_leaves": 2},
                "hold_auto_leaves requires hold_scope 'auto' or receiver_hold true",
            ),
            ({**FLOW_HOLD, "hold_auto_leaves": 2}, "'reps_flow_hold' does not take"),
            ({**FLOW_HOLD, "path_seq": True, "hold_scope": "host"}, "'reps_flow_hold' does not take"),
        ]
        # No other selector or mode reads the new keys.
        for key, value in (
            ("hold_rule", "llr"),
            ("hold_release", "trickle"),
            ("sentinel_mode", "duplicate"),
            ("held_loss_rtts", 2),
            ("receiver_hold", True),
            ("receiver_hold_clean", 8),
            ("receiver_hold_slots", 8),
            ("receiver_hold_widen", 2),
            ("loss_p0", 0.0),
        ):
            refusals += [
                ({"mode": "spray_policy", key: value}, "does not take"),
                ({"mode": "spray_uniform", key: value}, "takes no other key"),
                ({"mode": "ev_hash", "selector": "mrc", key: value}, "does not take"),
                (
                    {"mode": "spray_uniform", "selector": "reps_oracle",
                     "grade_cost_source": "switch_queue", key: value},
                    "does not take",
                ),
            ]
        # path_seq is read under spray_uniform only.
        refusals += [
            ({"mode": "spray_policy", "path_seq": True}, "does not take"),
            ({"mode": "ev_hash", "selector": "mrc", "path_seq": True}, "does not take"),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
