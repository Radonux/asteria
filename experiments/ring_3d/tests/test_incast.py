"""The incast workload: hosts on chosen leaves receive fan_in messages each."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chakra.schema.protobuf.et_def_pb2 import (
    COMP_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    GlobalMetadata,
    Node,
)
from chakra.src.third_party.utils.protolib import decodeMessage

from experiments.ring_3d.generate import Incast, load_profile, materialize

PROFILES = REPOSITORY_ROOT / "experiments/ring_3d/profiles"


def _incast(
    fan_in: int = 32, receiver_leaf: int = 0, message_bytes: int = 2_097_152
) -> dict[str, Any]:
    """64 ranks on eight leaves of eight hosts."""
    document = json.loads(
        (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
    )
    document.pop("dp_all_reduce_bytes")
    document.update(
        {
            "parallelism": {"tp": 1, "pp": 1, "dp": 64},
            "steps": 1,
            "compute_duration_us": 0,
            "tp_all_reduce_bytes": 0,
            "pp_bytes": 0,
            "microburst_enabled": False,
            "workload": {
                "kind": "incast",
                "receiver_leaf": receiver_leaf,
                "fan_in": fan_in,
                "message_bytes": message_bytes,
            },
        }
    )
    document["network"]["hosts_per_leaf"] = 8
    return document


def _nodes(trace: Path) -> list[Node]:
    nodes = []
    with trace.open("rb") as handle:
        decodeMessage(handle, GlobalMetadata())
        while True:
            node = Node()
            if not decodeMessage(handle, node):
                return nodes
            nodes.append(node)


def _ends(node: Node) -> tuple[int, int, int, int]:
    attributes = {attribute.name: attribute for attribute in node.attr}
    return (
        attributes["comm_src"].uint32_val,
        attributes["comm_dst"].uint32_val,
        attributes["comm_tag"].uint32_val,
        attributes["comm_size"].uint64_val,
    )


class IncastMessageTests(unittest.TestCase):
    def test_senders_span_every_other_leaf_round_robin(self) -> None:
        messages = Incast(receiver_leaves=(0,), fan_in=14, message_bytes=1).messages(
            8, 8
        )
        self.assertEqual(len(messages), 8 * 14)
        for receiver in range(8):
            with self.subTest(receiver=receiver):
                mine = [m for m in messages if m[1] == receiver]
                self.assertEqual([tag for _, _, tag in mine], list(range(1, 15)))
                self.assertEqual(
                    [source // 8 for source, _, _ in mine], [1, 2, 3, 4, 5, 6, 7] * 2
                )
                # The second round takes the next host on each leaf.
                self.assertEqual(
                    [source % 8 for source, _, _ in mine],
                    [receiver] * 7 + [(receiver + 1) % 8] * 7,
                )

    def test_fan_in_above_the_senders_wraps_into_second_messages(self) -> None:
        messages = Incast(receiver_leaves=(0,), fan_in=64, message_bytes=1).messages(
            8, 8
        )
        for receiver in range(8):
            with self.subTest(receiver=receiver):
                pairs = Counter(
                    source
                    for source, destination, _ in messages
                    if destination == receiver
                )
                self.assertEqual(len(pairs), 56)
                self.assertEqual(sorted(pairs.values()), [1] * 48 + [2] * 8)
        # Every sending host carries the same number of messages.
        load = Counter(source for source, _, _ in messages)
        self.assertEqual(set(load.values()), {64 * 8 // 56, 64 * 8 // 56 + 1})
        self.assertEqual(sum(load.values()), 512)

    def test_receiver_leaf_is_never_a_source(self) -> None:
        messages = Incast(receiver_leaves=(3,), fan_in=21, message_bytes=1).messages(
            8, 8
        )
        self.assertEqual({destination // 8 for _, destination, _ in messages}, {3})
        self.assertNotIn(3, {source // 8 for source, _, _ in messages})

    def test_each_receiving_leaf_takes_its_first_receiver_count_hosts(self) -> None:
        incast = Incast(
            receiver_leaves=(5, 2), fan_in=9, message_bytes=1, receiver_count=3
        )
        messages = incast.messages(8, 8)
        self.assertEqual(
            list(dict.fromkeys(destination for _, destination, _ in messages)),
            [40, 41, 42, 16, 17, 18],
        )
        for receiver in (40, 41, 42, 16, 17, 18):
            leaf, position = divmod(receiver, 8)
            others = [other for other in range(8) if other != leaf]
            with self.subTest(receiver=receiver):
                self.assertEqual(
                    [(s, t) for s, d, t in messages if d == receiver],
                    [
                        (
                            others[j % 7] * 8 + (position + j // 7) % 8,
                            j + 1,
                        )
                        for j in range(9)
                    ],
                )
        # The other receiving leaf sends too: host 16 receives and sends.
        self.assertIn(16, {source for source, _, _ in messages})
        self.assertIn(40, {source for source, _, _ in messages})

    def test_one_receiver_leaf_is_a_list_of_one(self) -> None:
        for count in (None, 8):
            with self.subTest(receiver_count=count):
                self.assertEqual(
                    Incast((3,), 21, 1, count).messages(8, 8),
                    Incast((3,), 21, 1).messages(8, 8),
                )


class IncastTraceTests(unittest.TestCase):
    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_every_rank_sends_and_receives_exactly_its_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(_incast()), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            manifest = materialize(path, output)
            traces = {
                rank: _nodes(output / f"workload/ring_3d.{rank}.et")
                for rank in range(64)
            }
            system = json.loads((output / "system.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["workload"],
            {
                "kind": "incast",
                "receiver_leaves": [0],
                "receiver_count": 8,
                "fan_in": 32,
                "message_bytes": 2_097_152,
            },
        )
        # 256 messages over 56 sending hosts: every message starts at once only
        # if a rank may hold its most, five, in flight.
        self.assertEqual(system["comm-ops-in-flight"], 5)
        expected = Incast((0,), 32, 2_097_152).messages(8, 8)
        sends = sorted(
            _ends(node)[:3]
            for nodes in traces.values()
            for node in nodes
            if node.type == COMM_SEND_NODE
        )
        receives = sorted(
            _ends(node)[:3]
            for nodes in traces.values()
            for node in nodes
            if node.type == COMM_RECV_NODE
        )
        self.assertEqual(sends, sorted(expected))
        self.assertEqual(receives, sorted(expected))
        for rank, nodes in traces.items():
            with self.subTest(rank=rank):
                for node in nodes:
                    source, destination, _, size = _ends(node)
                    self.assertEqual(list(node.ctrl_deps), [])
                    self.assertEqual(size, 2_097_152)
                    self.assertIn(rank, (source, destination))
                    self.assertEqual(
                        node.type,
                        COMM_SEND_NODE if rank == source else COMM_RECV_NODE,
                    )
                if rank < 8:
                    self.assertEqual(len(nodes), 32)

    def test_incast_refuses_what_it_cannot_send(self) -> None:
        cases = (
            (_incast(fan_in=0), "workload.fan_in must be a positive integer"),
            (_incast(receiver_leaf=8), "workload.receiver_leaf must be below"),
            (_incast(message_bytes=0), "workload.message_bytes must be a positive"),
        )
        for document, message in cases:
            with self.subTest(message=message):
                self._refused(document, message)
        changes = (
            ({"steps": 2}, "'incast' sends once and needs steps 1"),
            ({"pp_bytes": 1}, "'incast' requires compute_duration_us"),
            ({"dp_all_reduce_bytes": 1_024}, "takes no dp_all_reduce_bytes"),
        )
        for change, message in changes:
            document = _incast()
            document.update(change)
            with self.subTest(change=change):
                self._refused(document, message)
        document = _incast()
        del document["workload"]["fan_in"]
        self._refused(document, "'incast' must contain exactly")
        document = _incast()
        document["network"]["hosts_per_leaf"] = 64
        self._refused(document, "needs two leaves or more")
        receivers = (
            ({"receiver_leaves": [1]}, "'incast' must contain exactly"),
            ({"receiver_count": 9}, "receiver_count must be at most"),
            ({"receiver_count": 0}, "receiver_count must be a positive integer"),
            ({"receiver_count": True}, "receiver_count must be a positive integer"),
            ({"receiver_extra": 1}, "'incast' must contain exactly"),
        )
        for change, message in receivers:
            document = _incast()
            document["workload"].update(change)
            with self.subTest(change=change):
                self._refused(document, message)
        lists = (
            ([], "receiver_leaves must be a nonempty array"),
            (2, "receiver_leaves must be a nonempty array"),
            ([1, 1], "receiver_leaves must be distinct"),
            ([0, 8], "each of workload.receiver_leaves"),
            ([0, -1], "receiver_leaves must be a nonnegative integer"),
        )
        for leaves, message in lists:
            document = _incast()
            del document["workload"]["receiver_leaf"]
            document["workload"]["receiver_leaves"] = leaves
            with self.subTest(receiver_leaves=leaves):
                self._refused(document, message)
        document = _incast()
        del document["workload"]["receiver_leaf"]
        self._refused(document, "'incast' must contain exactly")

    def test_receiver_leaf_writes_what_receiver_leaves_of_one_writes(self) -> None:
        single = _incast(fan_in=14, receiver_leaf=3)
        listed = _incast(fan_in=14, receiver_leaf=3)
        del listed["workload"]["receiver_leaf"]
        listed["workload"].update({"receiver_leaves": [3], "receiver_count": 8})
        outputs = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            for name, document in (("single", single), ("listed", listed)):
                path = Path(temporary_directory) / f"{name}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / name
                manifest = materialize(path, output)
                outputs.append(
                    (
                        manifest["workload"],
                        {
                            rank: (output / f"workload/ring_3d.{rank}.et").read_bytes()
                            for rank in range(64)
                        },
                        (output / "system.json").read_bytes(),
                    )
                )
        self.assertEqual(outputs[0], outputs[1])

    def test_two_receiving_leaves_trace_exactly_their_messages(self) -> None:
        document = _incast(fan_in=10)
        del document["workload"]["receiver_leaf"]
        document["workload"].update({"receiver_leaves": [6, 1], "receiver_count": 6})
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            manifest = materialize(path, output)
            traces = {
                rank: _nodes(output / f"workload/ring_3d.{rank}.et")
                for rank in range(64)
            }
            system = json.loads((output / "system.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["workload"]["receiver_leaves"], [6, 1])
        self.assertEqual(manifest["workload"]["receiver_count"], 6)
        expected = Incast((6, 1), 10, 2_097_152, 6).messages(8, 8)
        self.assertEqual(len(expected), 2 * 6 * 10)
        for node_type in (COMM_SEND_NODE, COMM_RECV_NODE):
            with self.subTest(node_type=node_type):
                self.assertEqual(
                    sorted(
                        _ends(node)[:3]
                        for nodes in traces.values()
                        for node in nodes
                        if node.type == node_type
                    ),
                    sorted(expected),
                )
        self.assertEqual(
            system["comm-ops-in-flight"],
            max(Counter(source for source, _, _ in expected).values()),
        )
        # Hosts 6 and 7 of a receiving leaf receive nothing.
        self.assertFalse([node for node in traces[14] if node.type == COMM_RECV_NODE])

    def test_a_rank_with_no_message_still_has_a_node(self) -> None:
        document = _incast(fan_in=8)
        del document["workload"]["receiver_leaf"]
        document["workload"].update({"receiver_leaves": [0, 1], "receiver_count": 6})
        messages = Incast((0, 1), 8, 2_097_152, 6).messages(8, 8)
        busy = {rank for message in messages for rank in message[:2]}
        self.assertIn(7, set(range(64)) - busy)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            materialize(path, output)
            traces = {
                rank: _nodes(output / f"workload/ring_3d.{rank}.et")
                for rank in range(64)
            }
        for rank, nodes in traces.items():
            with self.subTest(rank=rank):
                self.assertTrue(nodes)
                if rank not in busy:
                    self.assertEqual(
                        [(node.type, node.duration_micros) for node in nodes],
                        [(COMP_NODE, 0)],
                    )
                    self.assertEqual(list(nodes[0].ctrl_deps), [])


if __name__ == "__main__":
    unittest.main()
