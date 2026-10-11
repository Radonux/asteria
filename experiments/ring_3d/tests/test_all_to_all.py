"""The all-to-all workload: a group runs chained native all-to-all collectives."""

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
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMP_NODE,
    GlobalMetadata,
    Node,
)
from chakra.src.third_party.utils.protolib import decodeMessage

from experiments.ring_3d.generate import AllToAll, load_profile, materialize

PROFILES = REPOSITORY_ROOT / "experiments/ring_3d/profiles"


def _all_to_all(
    group: Any = "all",
    message_bytes: int = 65_536,
    iterations: int = 3,
    compute_ns: int = 2_000,
) -> dict[str, Any]:
    """64 ranks on eight leaves of eight hosts."""
    document = json.loads(
        (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
    )
    document.pop("dp_all_reduce_bytes")
    document.update(
        {
            "parallelism": {"tp": 1, "pp": 1, "dp": 64},
            "steps": iterations,
            "compute_duration_us": 0,
            "tp_all_reduce_bytes": 0,
            "pp_bytes": 0,
            "microburst_enabled": False,
            "workload": {
                "kind": "all_to_all",
                "group": group,
                "message_bytes": message_bytes,
                "iterations": iterations,
                "compute_ns": compute_ns,
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


def _attributes(node: Node) -> dict[str, Any]:
    return {attribute.name: attribute for attribute in node.attr}


def _materialize(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, list[Node]], dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "profile.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        output = Path(temporary_directory) / "out"
        manifest = materialize(path, output)
        traces = {
            rank: _nodes(output / f"workload/ring_3d.{rank}.et")
            for rank in range(manifest["ranks"])
        }
        system = json.loads((output / "system.json").read_text(encoding="utf-8"))
        groups = json.loads(
            (output / "communicator_groups.json").read_text(encoding="utf-8")
        )
    return manifest, traces, system, groups


class AllToAllGroupTests(unittest.TestCase):
    def test_all_is_every_rank(self) -> None:
        self.assertEqual(AllToAll(4, 1, 0).members(16, 8), tuple(range(16)))
        self.assertEqual(AllToAll(4, 1, 0).members(16, None), tuple(range(16)))

    def test_leaves_are_their_hosts_in_rank_order(self) -> None:
        group = AllToAll(4, 1, 0, leaves=(3, 1))
        self.assertEqual(group.members(32, 8), (*range(8, 16), *range(24, 32)))

    def test_the_collective_carries_one_message_per_member(self) -> None:
        self.assertEqual(AllToAll(4_096, 1, 0).collective_bytes(16), 65_536)


class AllToAllTraceTests(unittest.TestCase):
    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_every_rank_chains_its_iterations_through_compute(self) -> None:
        manifest, traces, system, groups = _materialize(_all_to_all())
        self.assertEqual(
            manifest["workload"],
            {
                "kind": "all_to_all",
                "group": "all",
                "message_bytes": 65_536,
                "iterations": 3,
                "compute_ns": 2_000,
                "group_size": 64,
                "communicator_group": 130,
            },
        )
        self.assertEqual(manifest["collective_implementations"]["all_to_all"], "direct")
        self.assertEqual(groups["130"], list(range(64)))
        self.assertEqual(len(groups), 130)
        self.assertEqual(
            system["all-to-all-implementation-per-group"], {"130": "direct"}
        )
        self.assertNotIn("comm-ops-in-flight", system)
        for rank, nodes in traces.items():
            with self.subTest(rank=rank):
                self.assertEqual(
                    [node.name for node in nodes],
                    [
                        "iteration_1_all_to_all",
                        "iteration_1_compute",
                        "iteration_2_all_to_all",
                        "iteration_2_compute",
                        "iteration_3_all_to_all",
                    ],
                )
                self.assertEqual(list(nodes[0].ctrl_deps), [])
                for previous, node in zip(nodes, nodes[1:]):
                    self.assertEqual(list(node.ctrl_deps), [previous.id])
                for step, node in enumerate(nodes[::2], start=1):
                    attributes = _attributes(node)
                    self.assertEqual(node.type, COMM_COLL_NODE)
                    self.assertEqual(attributes["comm_type"].int64_val, ALL_TO_ALL)
                    self.assertEqual(attributes["comm_size"].uint64_val, 65_536 * 64)
                    self.assertEqual(attributes["pg_name"].string_val, "130")
                    self.assertEqual(attributes["training_step"].uint64_val, step)
                    self.assertNotIn("parallelism_domain", attributes)
                for node in nodes[1::2]:
                    self.assertEqual(
                        (node.type, node.duration_micros), (COMP_NODE, 2)
                    )

    def test_zero_compute_chains_the_collectives_directly(self) -> None:
        _, traces, _, _ = _materialize(_all_to_all(compute_ns=0))
        nodes = traces[5]
        self.assertEqual(
            [node.name for node in nodes],
            [f"iteration_{step}_all_to_all" for step in (1, 2, 3)],
        )
        for previous, node in zip(nodes, nodes[1:]):
            self.assertEqual(list(node.ctrl_deps), [previous.id])

    def test_ranks_outside_the_leaves_are_idle(self) -> None:
        manifest, traces, system, groups = _materialize(
            _all_to_all(group={"leaves": [6, 2]})
        )
        members = [*range(16, 24), *range(48, 56)]
        self.assertEqual(manifest["workload"]["group"], {"leaves": [6, 2]})
        self.assertEqual(manifest["workload"]["group_size"], 16)
        self.assertEqual(groups["130"], members)
        for rank, nodes in traces.items():
            with self.subTest(rank=rank):
                if rank in members:
                    self.assertEqual(
                        _attributes(nodes[0])["comm_size"].uint64_val, 65_536 * 16
                    )
                else:
                    self.assertEqual(
                        [(node.type, node.duration_micros) for node in nodes],
                        [(COMP_NODE, 0)],
                    )
                    self.assertEqual(list(nodes[0].ctrl_deps), [])

    def test_other_workloads_write_no_all_to_all_group(self) -> None:
        document = json.loads(
            (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
        )
        _, _, system, groups = _materialize(document)
        self.assertNotIn("130", groups)
        self.assertNotIn("all-to-all-implementation-per-group", system)

    def test_all_to_all_refuses_what_it_cannot_send(self) -> None:
        cases = (
            (_all_to_all(message_bytes=0), "message_bytes must be a positive"),
            (_all_to_all(message_bytes=6), "must be a multiple of 4"),
            (
                _all_to_all(group={"leaves": [1]}, message_bytes=28),
                "at least 4 times the rank count",
            ),
            (_all_to_all(compute_ns=-1000), "compute_ns must be a nonnegative"),
            (_all_to_all(compute_ns=1500), "compute_ns must be a multiple of 1000"),
            (_all_to_all(group="some"), "group must be 'all' or an object"),
            (_all_to_all(group={"leaves": []}), "leaves must be a nonempty array"),
            (_all_to_all(group={"leaves": [1, 1]}), "leaves must be distinct"),
            (_all_to_all(group={"leaves": [8]}), "must be below the leaf count"),
            (
                _all_to_all(group={"leaves": [1], "hosts": 2}),
                "group must be 'all' or an object",
            ),
        )
        for document, message in cases:
            with self.subTest(message=message):
                self._refused(document, message)
        document = _all_to_all()
        document["network"]["hosts_per_leaf"] = 1
        document["network"]["spine_count"] = 1
        document["workload"]["group"] = {"leaves": [3]}
        self._refused(document, "needs a group of two ranks or more")
        changes = (
            ({"steps": 1}, "needs steps equal to workload.iterations"),
            ({"pp_bytes": 1}, "'all_to_all' requires compute_duration_us"),
            ({"dp_all_reduce_bytes": 1_024}, "takes no dp_all_reduce_bytes"),
            ({"microburst_enabled": True}, "requires microburst_enabled false"),
        )
        for change, message in changes:
            document = _all_to_all()
            document.update(change)
            with self.subTest(change=change):
                self._refused(document, message)
        document = _all_to_all()
        del document["workload"]["compute_ns"]
        self._refused(document, "'all_to_all' must contain exactly")


if __name__ == "__main__":
    unittest.main()
