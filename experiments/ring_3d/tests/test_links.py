"""Per-link settings of the generated fabric: propagation delays."""

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


if __name__ == "__main__":
    unittest.main()
