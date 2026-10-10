"""Per-link settings of the generated fabric: propagation delays, per-link
overrides and link failures."""

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

from experiments.ring_3d.generate import load_profile, materialize

PROFILES = REPOSITORY_ROOT / "experiments/ring_3d/profiles"


def _sprayed_clos() -> dict[str, Any]:
    """64 hosts, eight per leaf, eight spines, 400 Gb/s, 4096 B packets,
    sprayed under NSCC on the ue switch."""
    document = json.loads(
        (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
    )
    document["parallelism"] = {"tp": 8, "pp": 1, "dp": 8}
    document["network"].update(
        {
            "hosts_per_leaf": 8,
            "spine_count": 8,
            "link_rate": "400Gbps",
            "packet_payload_bytes": 4096,
            "congestion_control": {"mode": "nscc"},
            "load_balancing": {"mode": "spray_uniform"},
            "switch": {"profile": "ue"},
        }
    )
    for key in ("data_queue_bytes", "trimmed_queue_bytes"):
        document["network"]["fabric"].pop(key)
    return document


class LinkTests(unittest.TestCase):
    def _materialize(self, document: dict[str, Any]) -> tuple[dict[str, Any], str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            manifest = materialize(path, output)
            topology = (output / "topology.txt").read_text(encoding="utf-8")
        return manifest, topology

    def _config(self, document: dict[str, Any]) -> str:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            materialize(path, output)
            config = (output / "network_config.txt").read_text(encoding="utf-8")
        return config.replace(temporary_directory, "<tmp>")

    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)


class LinkDelayTests(LinkTests):
    def test_absent_delays_write_the_historical_links(self) -> None:
        absent = _sprayed_clos()
        explicit = _sprayed_clos()
        explicit["network"]["host_link_delay_ns"] = 5_000
        explicit["network"]["switch_link_delay_ns"] = 12_500
        _, absent_topology = self._materialize(absent)
        _, explicit_topology = self._materialize(explicit)
        self.assertEqual(absent_topology, explicit_topology)
        links = absent_topology.splitlines()[2:]
        self.assertEqual(links[0], "0 64 400Gbps 0.005ms 0")
        self.assertEqual(links[-1], "71 79 400Gbps 0.0125ms 0")

    def test_delays_reach_every_link_and_the_manifest(self) -> None:
        document = _sprayed_clos()
        document["network"]["host_link_delay_ns"] = 1_000
        document["network"]["switch_link_delay_ns"] = 2_000
        manifest, topology = self._materialize(document)
        delays = {line.split()[3] for line in topology.splitlines()[2:66]}
        self.assertEqual(delays, {"0.001ms"})
        delays = {line.split()[3] for line in topology.splitlines()[66:]}
        self.assertEqual(delays, {"0.002ms"})
        self.assertEqual(manifest["physical_topology"]["host_link_delay_ns"], 1_000)
        self.assertEqual(manifest["physical_topology"]["switch_link_delay_ns"], 2_000)

    def test_a_ring_takes_the_same_two_delays(self) -> None:
        document = json.loads(
            (PROFILES / "model_100b_256_ring.json").read_text(encoding="utf-8")
        )
        document["network"]["host_link_delay_ns"] = 1_500
        document["network"]["switch_link_delay_ns"] = 333
        _, topology = self._materialize(document)
        links = topology.splitlines()[2:]
        self.assertEqual(links[0].split()[3], "0.0015ms")
        self.assertEqual(links[-1].split()[3], "0.000333ms")

    def test_ue_preset_follows_the_delays(self) -> None:
        """Four links each way at 81 ns of serialization per hop: the base RTT
        is four times the two delays plus four packet times."""
        for host, switch in ((5_000, 12_500), (1_000, 2_000), (250, 500)):
            document = _sprayed_clos()
            document["network"]["host_link_delay_ns"] = host
            document["network"]["switch_link_delay_ns"] = switch
            manifest, _ = self._materialize(document)
            rtt = 4 * (host + switch) + 4 * 81
            with self.subTest(host=host, switch=switch):
                self.assertEqual(manifest["switch"]["config_base_rtt_ns"], rtt)
                self.assertEqual(
                    manifest["switch"]["plane_bdp_bytes"],
                    rtt * 400_000_000_000 // 1_000_000_000 // 8,
                )

    def test_delays_must_be_positive_whole_nanoseconds(self) -> None:
        for key in ("host_link_delay_ns", "switch_link_delay_ns"):
            for value in (0, -1, 1.5, True, "1000"):
                document = _sprayed_clos()
                document["network"][key] = value
                with self.subTest(key=key, value=value):
                    self._refused(document, f"network.{key} must be a positive integer")


# Node ids of the 64-host fabric: hosts 0-63, leaves 64-71, spines 72-79.
LEAF = 64
SPINE = 72


def _line(topology: str, a: int, b: int) -> str:
    return next(
        line
        for line in topology.splitlines()[2:]
        if sorted(map(int, line.split()[:2])) == sorted((a, b))
    )


class LinkOverrideTests(LinkTests):
    def test_an_override_changes_its_link_alone(self) -> None:
        document = _sprayed_clos()
        document["network"]["host_link_delay_ns"] = 1_000
        document["network"]["switch_link_delay_ns"] = 2_000
        _, plain = self._materialize(document)
        document["network"]["link_overrides"] = [
            {
                "leaf": 2,
                "spine": 5,
                "rate": "200Gbps",
                "delay_ns": 4_000,
                "error_rate": 0.01,
            },
            {"endpoints": [3, LEAF], "delay_ns": 1_500},
        ]
        manifest, topology = self._materialize(document)
        self.assertEqual(
            _line(topology, LEAF + 2, SPINE + 5), "66 77 200Gbps 0.004ms 0.01"
        )
        self.assertEqual(_line(topology, 3, LEAF), "3 64 400Gbps 0.0015ms 0")
        changed = set(topology.splitlines()) - set(plain.splitlines())
        self.assertEqual(len(changed), 2)
        self.assertEqual(
            manifest["link_overrides"],
            [
                {
                    "endpoints": [66, 77],
                    "rate": "200Gbps",
                    "delay_ns": 4_000,
                    "error_rate": 0.01,
                },
                {
                    "endpoints": [3, 64],
                    "rate": None,
                    "delay_ns": 1_500,
                    "error_rate": None,
                },
            ],
        )

    def test_leaf_and_spine_name_the_link_endpoints_name(self) -> None:
        by_index = _sprayed_clos()
        by_index["network"]["link_overrides"] = [
            {"leaf": 7, "spine": 0, "rate": "200Gbps"}
        ]
        by_nodes = _sprayed_clos()
        by_nodes["network"]["link_overrides"] = [
            {"endpoints": [SPINE, LEAF + 7], "rate": "200Gbps"}
        ]
        self.assertEqual(self._materialize(by_index)[1], self._materialize(by_nodes)[1])

    def test_a_slow_spine_leaves_the_base_rtt_and_a_slow_host_does_not(self) -> None:
        """The base RTT takes the best shortest-hop path, as ns-3 now does."""
        document = _sprayed_clos()
        document["network"]["host_link_delay_ns"] = 1_000
        document["network"]["switch_link_delay_ns"] = 2_000
        document["network"]["link_overrides"] = [
            {"leaf": leaf, "spine": 7, "rate": "200Gbps", "delay_ns": 4_000}
            for leaf in range(8)
        ]
        manifest, _ = self._materialize(document)
        self.assertEqual(manifest["switch"]["config_base_rtt_ns"], 12_324)
        # Host 0's link at 200 Gb/s and 3000 ns: 163 ns a packet on it.
        document["network"]["link_overrides"] = [
            {"endpoints": [0, LEAF], "rate": "200Gbps", "delay_ns": 3_000}
        ]
        manifest, _ = self._materialize(document)
        self.assertEqual(
            manifest["switch"]["config_base_rtt_ns"],
            2 * (3_000 + 2_000 + 2_000 + 1_000) + 163 + 3 * 81,
        )

    def test_a_ring_link_is_named_by_its_endpoints(self) -> None:
        document = json.loads(
            (PROFILES / "model_100b_256_ring.json").read_text(encoding="utf-8")
        )
        hosts = 256
        document["network"]["link_overrides"] = [
            {"endpoints": [hosts, hosts + 1], "error_rate": 0.5}
        ]
        _, topology = self._materialize(document)
        self.assertTrue(_line(topology, hosts, hosts + 1).endswith(" 0.5"))
        document["network"]["link_overrides"] = [
            {"leaf": 0, "spine": 0, "error_rate": 0.5}
        ]
        self._refused(document, "names a leaf and spine, which a ring has not")

    def test_overrides_refuse_what_they_cannot_mean(self) -> None:
        cases = (
            ({"leaf": 0}, "must name 'endpoints' or 'leaf' and 'spine'"),
            (
                {"leaf": 0, "spine": 0, "endpoints": [0, LEAF], "rate": "200Gbps"},
                "must name 'endpoints' or 'leaf' and 'spine'",
            ),
            (
                {"leaf": 0, "spine": 0, "speed": "200Gbps"},
                "must name 'endpoints' or 'leaf' and 'spine'",
            ),
            ({"leaf": 0, "spine": 0}, "must set one of"),
            (
                {"leaf": 8, "spine": 0, "rate": "200Gbps"},
                r"leaf must be an integer in \[0, 8\)",
            ),
            (
                {"leaf": 0, "spine": 8, "rate": "200Gbps"},
                r"spine must be an integer in \[0, 8\)",
            ),
            ({"endpoints": [0, 1], "rate": "200Gbps"}, "which no link joins"),
            ({"endpoints": [0], "rate": "200Gbps"}, "endpoints must be two node ids"),
            ({"leaf": 0, "spine": 0, "rate": "300Gbps"}, "the switch ECN table covers"),
            ({"leaf": 0, "spine": 0, "rate": "200G"}, "must be a rate such as 200Gbps"),
            (
                {"leaf": 0, "spine": 0, "delay_ns": 0},
                "delay_ns must be a positive integer",
            ),
            (
                {"leaf": 0, "spine": 0, "error_rate": 0},
                r"error_rate must be a number in \(0, 1\]",
            ),
            (
                {"leaf": 0, "spine": 0, "error_rate": 1.5},
                r"error_rate must be a number in \(0, 1\]",
            ),
        )
        for override, message in cases:
            document = _sprayed_clos()
            document["network"]["link_overrides"] = [override]
            with self.subTest(override=override):
                self._refused(document, message)
        document = _sprayed_clos()
        document["network"]["link_overrides"] = [
            {"leaf": 0, "spine": 0, "rate": "200Gbps"},
            {"endpoints": [SPINE, LEAF], "delay_ns": 1},
        ]
        self._refused(document, "a link an earlier override names")


def _failure_lines(config: str) -> list[str]:
    return [line for line in config.splitlines() if line.startswith("LINK_FAILURE")]


class LinkFailureTests(LinkTests):
    def test_absent_failures_write_no_line(self) -> None:
        document = _sprayed_clos()
        document["network"]["link_failures"] = []
        self.assertEqual(self._config(document), self._config(_sprayed_clos()))
        self.assertEqual(_failure_lines(self._config(document)), [])

    def test_each_model_writes_its_lines(self) -> None:
        document = _sprayed_clos()
        document["network"]["link_failures"] = [
            {"model": "graceful", "spine": 3, "start_ns": 200_000},
            {"model": "silent", "spine": 1, "leaf": 4, "start_ns": 0},
            {"model": "gray", "spine": 2, "leaf": 0, "start_ns": 5, "error_rate": 0.01},
            {"model": "gray", "spine": 2, "leaf": 1, "start_ns": 5, "rate": "200Gbps"},
        ]
        lines = _failure_lines(self._config(document))
        self.assertEqual(
            lines,
            [f"LINK_FAILURE 200000 down {LEAF + leaf} {SPINE + 3}" for leaf in range(8)]
            + [
                f"LINK_FAILURE 0 blackhole {LEAF + 4} {SPINE + 1}",
                f"LINK_FAILURE 5 loss {LEAF} {SPINE + 2} 0.01",
                f"LINK_FAILURE 5 rate {LEAF + 1} {SPINE + 2} 200Gbps",
            ],
        )
        manifest, _ = self._materialize(document)
        self.assertEqual(
            manifest["link_failures"][2],
            {
                "model": "gray",
                "start_ns": 5,
                "spine": 2,
                "leaf": 0,
                "error_rate": 0.01,
                "rate": None,
            },
        )

    def test_failures_refuse_what_they_cannot_mean(self) -> None:
        cases = (
            ({"model": "crash", "spine": 0, "start_ns": 0}, "model must be one of"),
            (
                {"model": "graceful", "spine": 0},
                "must contain model, start_ns and spine",
            ),
            (
                {"model": "graceful", "spine": 0, "start_ns": 0, "error_rate": 0.1},
                "must contain model, start_ns and spine",
            ),
            (
                {"model": "gray", "spine": 0, "start_ns": 0},
                "sets one of error_rate and rate",
            ),
            (
                {
                    "model": "gray",
                    "spine": 0,
                    "start_ns": 0,
                    "error_rate": 0.1,
                    "rate": "200Gbps",
                },
                "sets one of error_rate and rate",
            ),
            (
                {"model": "gray", "spine": 0, "start_ns": 0, "error_rate": 2},
                r"error_rate must be a number in \(0, 1\]",
            ),
            (
                {"model": "gray", "spine": 0, "start_ns": 0, "rate": "fast"},
                "must be a rate such as 200Gbps",
            ),
            (
                {"model": "silent", "spine": 8, "start_ns": 0},
                r"spine must be an integer in \[0, 8\)",
            ),
            (
                {"model": "silent", "spine": 0, "leaf": -1, "start_ns": 0},
                r"leaf must be an integer in \[0, 8\)",
            ),
            (
                {"model": "silent", "spine": 0, "start_ns": -1},
                "start_ns must be a nonnegative",
            ),
        )
        for failure, message in cases:
            document = _sprayed_clos()
            document["network"]["link_failures"] = [failure]
            with self.subTest(failure=failure):
                self._refused(document, message)
        document = _sprayed_clos()
        document["network"]["link_failures"] = [
            {"model": "silent", "spine": 0, "start_ns": 0},
            {"model": "gray", "spine": 0, "leaf": 3, "start_ns": 9, "error_rate": 0.1},
        ]
        self._refused(document, "a link an earlier failure names")
        document["network"]["link_failures"] = [
            {"model": "graceful", "spine": spine, "leaf": 2, "start_ns": 0}
            for spine in range(8)
        ]
        self._refused(document, "must leave every leaf a link to a spine")

    def test_a_ring_has_no_spine_to_fail(self) -> None:
        document = json.loads(
            (PROFILES / "model_100b_256_ring.json").read_text(encoding="utf-8")
        )
        document["network"]["link_failures"] = [
            {"model": "graceful", "spine": 0, "start_ns": 0}
        ]
        self._refused(document, "unknown network keys")


if __name__ == "__main__":
    unittest.main()
