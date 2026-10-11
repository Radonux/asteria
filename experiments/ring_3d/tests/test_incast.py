"""The incast workload: every host on one leaf receives fan_in messages."""

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
        messages = Incast(receiver_leaf=0, fan_in=14, message_bytes=1).messages(8, 8)
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
        messages = Incast(receiver_leaf=0, fan_in=64, message_bytes=1).messages(8, 8)
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
        messages = Incast(receiver_leaf=3, fan_in=21, message_bytes=1).messages(8, 8)
        self.assertEqual({destination // 8 for _, destination, _ in messages}, {3})
        self.assertNotIn(3, {source // 8 for source, _, _ in messages})


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
                "receiver_leaf": 0,
                "fan_in": 32,
                "message_bytes": 2_097_152,
            },
        )
        # 256 messages over 56 sending hosts: every message starts at once only
        # if a rank may hold its most, five, in flight.
        self.assertEqual(system["comm-ops-in-flight"], 5)
        expected = Incast(0, 32, 2_097_152).messages(8, 8)
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


if __name__ == "__main__":
    unittest.main()
