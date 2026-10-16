"""The all-to-all group of the first hosts of each leaf, and the background
load the other hosts send."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.generate import AllToAll, load_profile, materialize
from experiments.ring_3d.tests.test_all_to_all import _all_to_all

HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
}
EVERY_LEAF = list(range(8))
BACKGROUND = {"pattern": "leaf_shift", "rate": "350Gbps"}


def _job(hosts: int = 2, background: dict[str, Any] | None = BACKGROUND,
         **group: Any) -> dict[str, Any]:
    """64 ranks on eight leaves of eight hosts, the first hosts of each leaf
    the all-to-all's group, the rest the background's."""
    document = _all_to_all(group={"leaves": EVERY_LEAF, "hosts": hosts, **group})
    if background is not None:
        document["workload"]["background"] = background
    document["network"]["spine_count"] = 4
    document["network"]["link_rate"] = "400Gbps"
    document["network"]["load_balancing"] = HOLD
    return document


def _generated(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "profile.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        output = Path(temporary_directory) / "out"
        manifest = materialize(path, output)
        experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        config = (output / "network_config.txt").read_text(encoding="utf-8")
    return manifest, experiment, config


class GroupHostsTests(unittest.TestCase):
    def test_hosts_are_the_first_of_each_leaf(self) -> None:
        group = AllToAll(4, 1, 0, leaves=(0, 1), hosts=2)
        self.assertEqual(group.members(16, 8), (0, 1, 8, 9))
        self.assertEqual(group.manifest_group(), {"leaves": [0, 1], "hosts": 2})

    def test_the_group_writes_its_hosts_and_traces_them_alone(self) -> None:
        manifest, experiment, config = _generated(_job(background=None))
        self.assertEqual(manifest["workload"]["group"], {"leaves": EVERY_LEAF, "hosts": 2})
        self.assertEqual(manifest["workload"]["group_size"], 16)
        self.assertNotIn("background", experiment)
        members = [leaf * 8 + host for leaf in range(8) for host in range(2)]
        self.assertIn(
            f"\nHOLD_TRACE_HOSTS 16 {' '.join(map(str, members))}\n", config
        )

    def test_a_group_of_whole_leaves_traces_every_host(self) -> None:
        document = _all_to_all(group={"leaves": [0, 1]})
        document["network"]["spine_count"] = 4
        document["network"]["load_balancing"] = HOLD
        _, _, config = _generated(document)
        self.assertNotIn("HOLD_TRACE_HOSTS", config)


class BackgroundTests(unittest.TestCase):
    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_each_other_host_sends_one_flow_and_receives_one(self) -> None:
        manifest, experiment, _ = _generated(_job())
        flows = experiment["background"]["flows"]
        members = {leaf * 8 + host for leaf in range(8) for host in range(2)}
        self.assertEqual(len(flows), 64 - 16)
        self.assertEqual({flow["src"] for flow in flows}, set(range(64)) - members)
        self.assertEqual({flow["dst"] for flow in flows}, set(range(64)) - members)
        for flow in flows:
            leaf, index = divmod(flow["src"], 8)
            with self.subTest(src=flow["src"]):
                self.assertEqual(flow["dst"], ((leaf + 1 + index % 7) % 8) * 8 + index)
                # What a member sends to the 14 members on other leaves.
                self.assertEqual(flow["size_bytes"], 65_536 * 14)
                self.assertEqual(flow["rate_bps"], 350_000_000_000)
                self.assertEqual(flow["priority_group"], 3)
        # Each leaf's six send to six of the other leaves, one each.
        for leaf in range(8):
            targets = [flow["dst"] // 8 for flow in flows if flow["src"] // 8 == leaf]
            self.assertNotIn(leaf, targets)
            self.assertEqual(len(set(targets)), 6)
        self.assertEqual(
            manifest["workload"]["background"],
            {"pattern": "leaf_shift", "rate_bps": 350_000_000_000},
        )

    def test_background_refusals(self) -> None:
        for document, message in (
            (_job(background={"pattern": "ring", "rate": "350Gbps"}), "pattern must be 'leaf_shift'"),
            (_job(background={"pattern": "leaf_shift", "rate": "350"}), "rate must be a rate"),
            (_job(background={"pattern": "leaf_shift"}), "exactly \\['pattern', 'rate'\\]"),
            (_job(background={"pattern": "leaf_shift", "rate": "500Gbps"}), "at most network.link_rate"),
            (_job(hosts=8), "group.hosts below network.hosts_per_leaf"),
        ):
            with self.subTest(message=message):
                self._refused(document, message)
        whole = _all_to_all(group={"leaves": EVERY_LEAF})
        whole["workload"]["background"] = BACKGROUND
        self._refused(whole, "group.hosts below network.hosts_per_leaf")
        part = _all_to_all(group={"leaves": [0, 1], "hosts": 2})
        part["workload"]["background"] = BACKGROUND
        self._refused(part, "group.leaves to be every leaf")


class PortCounterIntervalTests(unittest.TestCase):
    def test_the_interval_is_written_only_when_set(self) -> None:
        document = _job(background=None)
        document["network"]["port_counter_interval_ns"] = 50_000
        manifest, _, config = _generated(document)
        self.assertIn("\nPORT_COUNTER_INTERVAL_NS 50000\nSWITCH_QUEUE_OUTPUT_FILE ", config)
        self.assertIn("/ns3/switch_queues.csv\n", config)
        self.assertEqual(manifest["physical_topology"]["port_counter_interval_ns"], 50_000)
        _, _, config = _generated(_job(background=None))
        self.assertNotIn("PORT_COUNTER_INTERVAL_NS", config)
        self.assertNotIn("SWITCH_QUEUE_OUTPUT_FILE", config)

    def test_the_interval_needs_the_port_counters(self) -> None:
        document = _all_to_all()
        document["network"]["load_balancing"] = {"mode": "ecmp"}
        document["network"]["port_counter_interval_ns"] = 50_000
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "port_counter_interval_ns requires"):
                load_profile(path)


class SparseHoldTraceTests(unittest.TestCase):
    def test_the_sparse_trace_is_written_only_when_set(self) -> None:
        document = _job(background=None)
        document["network"]["sparse_hold_trace"] = True
        manifest, _, config = _generated(document)
        self.assertIn("\nSPINE_LOSS_SPARSE on\n", config)
        self.assertIs(manifest["physical_topology"]["sparse_hold_trace"], True)
        manifest, _, config = _generated(_job(background=None))
        self.assertNotIn("SPINE_LOSS_SPARSE", config)
        self.assertNotIn("sparse_hold_trace", manifest["physical_topology"])

    def test_the_sparse_trace_needs_the_trace(self) -> None:
        document = _all_to_all()
        document["network"]["load_balancing"] = {"mode": "spray_uniform"}
        document["network"]["sparse_hold_trace"] = True
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sparse_hold_trace requires"):
                load_profile(path)


if __name__ == "__main__":
    unittest.main()
