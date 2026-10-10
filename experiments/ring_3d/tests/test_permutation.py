"""The permutation workload: every rank sends one message to a shifted rank."""

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

from chakra.schema.protobuf.et_def_pb2 import (
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    GlobalMetadata,
    Node,
)
from chakra.src.third_party.utils.protolib import decodeMessage

from experiments.ring_3d.generate import load_profile, materialize

PROFILES = REPOSITORY_ROOT / "experiments/ring_3d/profiles"


def _permutation(shift: int = 8, message_bytes: int = 2_097_152) -> dict[str, Any]:
    """64 ranks on eight leaves, each sending to the rank one leaf on."""
    document = json.loads(
        (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
    )
    document.pop("dp_all_reduce_bytes")
    document.update(
        {
            "parallelism": {"tp": 8, "pp": 1, "dp": 8},
            "steps": 1,
            "compute_duration_us": 0,
            "tp_all_reduce_bytes": 0,
            "pp_bytes": 0,
            "microburst_enabled": False,
            "workload": {
                "kind": "permutation",
                "shift": shift,
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


class PermutationTests(unittest.TestCase):
    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_each_rank_sends_to_its_shifted_rank_and_receives_from_the_other(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(_permutation()), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            manifest = materialize(path, output)
            traces = {
                rank: _nodes(output / f"workload/ring_3d.{rank}.et")
                for rank in range(64)
            }
            policy = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["workload"],
            {"kind": "permutation", "shift": 8, "message_bytes": 2_097_152},
        )
        self.assertEqual(policy["microburst"]["flows"], [])
        for rank, nodes in traces.items():
            with self.subTest(rank=rank):
                self.assertEqual([node.type for node in nodes], [COMM_SEND_NODE, COMM_RECV_NODE])
                ends = []
                for node in nodes:
                    attributes = {attribute.name: attribute for attribute in node.attr}
                    self.assertEqual(list(node.ctrl_deps), [])
                    self.assertNotIn("parallelism_domain", attributes)
                    self.assertEqual(attributes["comm_size"].uint64_val, 2_097_152)
                    self.assertEqual(attributes["training_step"].uint64_val, 1)
                    ends.append(
                        (attributes["comm_src"].uint32_val, attributes["comm_dst"].uint32_val)
                    )
                self.assertEqual(ends, [(rank, (rank + 8) % 64), ((rank - 8) % 64, rank)])

    def test_permutation_refuses_what_it_cannot_send(self) -> None:
        cases = (
            (_permutation(shift=0), "workload.shift must be a positive integer"),
            (_permutation(shift=64), "workload.shift must be below the rank count"),
            (_permutation(message_bytes=0), "workload.message_bytes must be a positive"),
        )
        for document, message in cases:
            with self.subTest(message=message):
                self._refused(document, message)
        changes = (
            ({"steps": 2}, "sends once and needs steps 1"),
            ({"compute_duration_us": 1}, "requires compute_duration_us"),
            ({"tp_all_reduce_bytes": 65_536}, "requires compute_duration_us"),
            (
                {"microburst_enabled": True, "microburst_trigger_step": 1},
                "requires microburst_enabled false",
            ),
            ({"dp_all_reduce_bytes": 1_024}, "takes no dp_all_reduce_bytes"),
        )
        for change, message in changes:
            document = _permutation()
            document.update(change)
            with self.subTest(change=change):
                self._refused(document, message)
        document = _permutation()
        del document["workload"]["shift"]
        self._refused(document, "must contain exactly")
        document = _permutation()
        document["workload"] = {"kind": "sequential_dp_all_reduce", "shift": 8}
        self._refused(document, "takes no other key")
        document = _permutation()
        document["workload"] = {"kind": "three_dimensional_overlap"}
        self._refused(document, r"missing profile keys: \['dp_all_reduce_bytes'\]")


if __name__ == "__main__":
    unittest.main()
