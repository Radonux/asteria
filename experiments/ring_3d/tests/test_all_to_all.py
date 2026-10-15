"""The all-to-all workload: a group runs chained native all-to-all collectives."""

from __future__ import annotations

import hashlib
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
    **combine: int,
) -> dict[str, Any]:
    """64 ranks on eight leaves of eight hosts; combine holds the optional
    combine_bytes and expert_compute_ns."""
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
                **combine,
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


class AllToAllPinnedOutputTests(unittest.TestCase):
    """A profile without a combine writes the traces, system.json and
    communicator groups it wrote before the combine existed."""

    PINNED = (
        (
            {"message_bytes": 262_144, "compute_ns": 259_000},
            "38b40b1c574b84d0db58e3eb3c173434bf720363861a4ca2a423588ce82219ac",
        ),
        (
            {
                "group": {"leaves": [0, 1]},
                "message_bytes": 262_144,
                "compute_ns": 62_000,
            },
            "4f68be44b8640d38fa3c54b962fa35aed3027bc807aafd4f77e0db0db0982752",
        ),
    )

    def test_outputs_are_byte_identical(self) -> None:
        for change, expected in self.PINNED:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "profile.json"
                path.write_text(json.dumps(_all_to_all(**change)), encoding="utf-8")
                output = Path(directory) / "out"
                manifest = materialize(path, output)
                # None of these files names the output directory, so their
                # bytes do not depend on where the run is written.
                digest = hashlib.sha256()
                for written in [
                    *sorted(output.glob("workload/*.et")),
                    output / "system.json",
                    output / "communicator_groups.json",
                ]:
                    digest.update(written.relative_to(output).as_posix().encode())
                    digest.update(written.read_bytes())
                self.assertEqual(digest.hexdigest(), expected)
                self.assertNotIn("combine_bytes", manifest["workload"])
                self.assertNotIn("expert_compute_ns", manifest["workload"])
                self.assertNotIn("dataset_splits", manifest["workload"])


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

    def test_a_combine_follows_each_all_to_all_in_the_same_step(self) -> None:
        manifest, traces, _, _ = _materialize(
            _all_to_all(combine_bytes=131_072, expert_compute_ns=5_000)
        )
        self.assertEqual(
            manifest["workload"],
            {
                "kind": "all_to_all",
                "group": "all",
                "message_bytes": 65_536,
                "iterations": 3,
                "compute_ns": 2_000,
                "combine_bytes": 131_072,
                "expert_compute_ns": 5_000,
                "group_size": 64,
                "communicator_group": 130,
            },
        )
        for rank, nodes in traces.items():
            with self.subTest(rank=rank):
                self.assertEqual(
                    [node.name for node in nodes],
                    [
                        "iteration_1_all_to_all",
                        "iteration_1_expert_compute",
                        "iteration_1_combine_all_to_all",
                        "iteration_1_compute",
                        "iteration_2_all_to_all",
                        "iteration_2_expert_compute",
                        "iteration_2_combine_all_to_all",
                        "iteration_2_compute",
                        "iteration_3_all_to_all",
                        "iteration_3_expert_compute",
                        "iteration_3_combine_all_to_all",
                    ],
                )
                self.assertEqual(list(nodes[0].ctrl_deps), [])
                for previous, node in zip(nodes, nodes[1:]):
                    self.assertEqual(list(node.ctrl_deps), [previous.id])
                for step, index in enumerate((0, 4, 8), start=1):
                    dispatch, expert, combine = nodes[index : index + 3]
                    for node, size in ((dispatch, 65_536), (combine, 131_072)):
                        attributes = _attributes(node)
                        self.assertEqual(node.type, COMM_COLL_NODE)
                        self.assertEqual(attributes["comm_type"].int64_val, ALL_TO_ALL)
                        self.assertEqual(attributes["comm_size"].uint64_val, size * 64)
                        self.assertEqual(attributes["pg_name"].string_val, "130")
                        self.assertEqual(attributes["training_step"].uint64_val, step)
                    self.assertEqual(
                        (expert.type, expert.duration_micros), (COMP_NODE, 5)
                    )
                for node in nodes[3::4]:
                    self.assertEqual((node.type, node.duration_micros), (COMP_NODE, 2))

    def test_zero_expert_compute_chains_the_combine_directly(self) -> None:
        _, traces, _, _ = _materialize(
            _all_to_all(compute_ns=0, combine_bytes=4_096, expert_compute_ns=0)
        )
        nodes = traces[5]
        self.assertEqual(
            [node.name for node in nodes],
            [
                "iteration_1_all_to_all",
                "iteration_1_combine_all_to_all",
                "iteration_2_all_to_all",
                "iteration_2_combine_all_to_all",
                "iteration_3_all_to_all",
                "iteration_3_combine_all_to_all",
            ],
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

    def test_dataset_splits_sets_the_system_layers_chunks(self) -> None:
        _, _, default, _ = _materialize(_all_to_all())
        self.assertEqual(default["preferred-dataset-splits"], 4)
        manifest, traces, system, _ = _materialize(
            _all_to_all(message_bytes=65_538, dataset_splits=1)
        )
        # One chunk: the system layer sends each pair its message_bytes as
        # one message per collective, and nothing else in system.json moves.
        self.assertEqual(system, {**default, "preferred-dataset-splits": 1})
        self.assertEqual(manifest["workload"]["dataset_splits"], 1)
        self.assertEqual(
            _attributes(traces[0][0])["comm_size"].uint64_val, 65_538 * 64
        )
        _, _, two, _ = _materialize(_all_to_all(dataset_splits=2))
        self.assertEqual(two["preferred-dataset-splits"], 2)

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
            (
                _all_to_all(combine_bytes=4_096, expert_compute_ns=-1000),
                "expert_compute_ns must be a nonnegative",
            ),
            (
                _all_to_all(combine_bytes=4_096, expert_compute_ns=1500),
                "expert_compute_ns must be a multiple of 1000",
            ),
            (
                _all_to_all(combine_bytes=0, expert_compute_ns=1000),
                "combine_bytes must be a positive",
            ),
            (
                _all_to_all(combine_bytes=6, expert_compute_ns=1000),
                "combine_bytes must be a multiple of 4",
            ),
            (
                _all_to_all(
                    group={"leaves": [1]}, combine_bytes=28, expert_compute_ns=1000
                ),
                "combine_bytes times the group size must be at least 4 times",
            ),
            (
                _all_to_all(combine_bytes=4_096),
                "may contain \\['combine_bytes', 'expert_compute_ns'\\] together",
            ),
            (
                _all_to_all(expert_compute_ns=1000),
                "may contain \\['combine_bytes', 'expert_compute_ns'\\] together",
            ),
            (_all_to_all(dataset_splits=0), "dataset_splits must be a positive"),
            (_all_to_all(dataset_splits=True), "dataset_splits must be a positive"),
            (
                _all_to_all(message_bytes=65_540, dataset_splits=8),
                "must be a multiple of 8",
            ),
            (
                _all_to_all(group={"leaves": [1]}, message_bytes=56, dataset_splits=8),
                "at least 8 times the rank count",
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
