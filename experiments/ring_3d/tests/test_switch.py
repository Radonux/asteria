"""The network.switch knob: the RED marking probability and the ``ue`` preset."""

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
RATES = (
    25_000_000_000,
    40_000_000_000,
    100_000_000_000,
    200_000_000_000,
    400_000_000_000,
    2_400_000_000_000,
)
HISTORICAL_PMAX = (
    "PMAX_MAP 6 25000000000 0.2 40000000000 0.2 100000000000 0.2 "
    "200000000000 0.2 400000000000 0.2 2400000000000 0.2\n"
)


def _map(key: str, value: object) -> str:
    return f"{key} 6 " + " ".join(f"{rate} {value}" for rate in RATES) + "\n"


def _trimming_clos() -> dict[str, Any]:
    """Eight hosts, two per leaf, four spines, 200 Gb/s, FTD trimming."""
    return json.loads((PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8"))


def _fixture_fabric() -> dict[str, Any]:
    """The Phase-3 NSCC fixture fabric: 64 hosts, eight per leaf, eight
    spines, 400 Gb/s and 4096 B packets, sprayed under NSCC. ns-3 prints
    maxRtt=70324 maxBdp=3516200 for it."""
    document = _trimming_clos()
    document["parallelism"] = {"tp": 8, "pp": 1, "dp": 8}
    document["network"].update(
        {
            "hosts_per_leaf": 8,
            "spine_count": 8,
            "link_rate": "400Gbps",
            "packet_payload_bytes": 4096,
            "congestion_control": {"mode": "nscc"},
            "load_balancing": {"mode": "spray_uniform"},
        }
    )
    return document


def _ue(document: dict[str, Any]) -> dict[str, Any]:
    document["network"]["switch"] = {"profile": "ue"}
    for key in ("data_queue_bytes", "trimmed_queue_bytes"):
        document["network"]["fabric"].pop(key, None)
    return document


class SwitchKnobTests(unittest.TestCase):
    def _materialize(self, document: dict[str, Any]) -> tuple[dict[str, Any], str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "out"
            manifest = materialize(path, output)
            config = (output / "network_config.txt").read_text(encoding="utf-8")
        # Paths name the temporary directory; the settings are what is compared.
        return manifest, config.replace(temporary_directory, "<tmp>")

    def _refused(self, document: dict[str, Any], message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_absent_knob_writes_the_historical_switch(self) -> None:
        absent = _trimming_clos()
        explicit = _trimming_clos()
        explicit["network"]["switch"] = {
            "profile": "default",
            "ecn_marking_probability": 0.2,
        }
        manifest, config = self._materialize(absent)
        _, explicit_config = self._materialize(explicit)
        self.assertIn(HISTORICAL_PMAX, config)
        self.assertEqual(
            manifest["switch"], {"profile": "default", "ecn_marking_probability": 0.2}
        )
        self.assertEqual(config, explicit_config)

    def test_marking_probability_reaches_every_link_speed(self) -> None:
        for probability in (1, 0.5, 0.05):
            document = _trimming_clos()
            document["network"]["switch"] = {"ecn_marking_probability": probability}
            _, baseline = self._materialize(_trimming_clos())
            manifest, config = self._materialize(document)
            with self.subTest(probability=probability):
                self.assertIn(_map("PMAX_MAP", float(probability)), config)
                self.assertNotIn(HISTORICAL_PMAX, config)
                self.assertEqual(
                    manifest["switch"]["ecn_marking_probability"], float(probability)
                )
                # Only the probability moves: thresholds and queues stay.
                for key in ("KMAX_MAP", "KMIN_MAP", "DATA_QUEUE_BYTES"):
                    line = next(l for l in baseline.splitlines() if l.startswith(key))
                    self.assertIn(line + "\n", config)

    def test_marking_probability_refuses_values_outside_zero_to_one(self) -> None:
        for value in (0, -0.1, 1.0001, True, "0.5", float("nan")):
            document = _trimming_clos()
            document["network"]["switch"] = {"ecn_marking_probability": value}
            with self.subTest(value=value):
                self._refused(
                    document, r"ecn_marking_probability must be a number in \(0, 1\]"
                )

    def test_switch_refuses_unknown_keys_and_profiles(self) -> None:
        document = _trimming_clos()
        document["network"]["switch"] = {"profile": "ue", "kmin": 1}
        self._refused(document, "network.switch may contain only")
        document["network"]["switch"] = {"profile": "uec"}
        self._refused(document, "network.switch.profile must be one of")
        document["network"]["switch"] = "ue"
        self._refused(document, "network.switch may contain only")

    def test_ue_preset_on_the_fixture_fabric(self) -> None:
        manifest, config = self._materialize(_ue(_fixture_fabric()))
        self.assertEqual(
            manifest["switch"],
            {
                "profile": "ue",
                "config_base_rtt_ns": 70_324,
                "plane_bdp_bytes": 3_516_200,
                "ecn_min_bytes": 703_000,
                "ecn_max_bytes": 2_813_000,
                "ecn_marking_probability": 1.0,
                "trim_threshold_bytes": 3_516_200,
                "trimmed_queue_drop_bytes": 3_516_200,
            },
        )
        self.assertIn(_map("KMAX_MAP", 2813) + _map("KMIN_MAP", 703), config)
        self.assertIn(_map("PMAX_MAP", 1.0), config)
        self.assertIn(
            "\nDATA_QUEUE_BYTES 3516200\nTRIMMED_QUEUE_BYTES 3516200\n", config
        )
        self.assertEqual(manifest["fabric"]["data_queue_bytes"], 3_516_200)
        self.assertEqual(manifest["fabric"]["trimmed_queue_bytes"], 3_516_200)

    def test_ue_preset_follows_the_fabric(self) -> None:
        """Plane_BDP is the link rate times the longest unloaded round trip:
        twice the one-way propagation plus one packet's serialization per hop."""
        clos = _ue(_trimming_clos())  # 200 Gb/s, 1000 B packets: 40 ns a hop
        one_leaf = _ue(_trimming_clos())
        one_leaf["network"]["hosts_per_leaf"] = 8
        ring = _ue(_trimming_clos())
        ring["network"]["topology"] = "ring"
        for key in ("hosts_per_leaf", "spine_count"):
            ring["network"].pop(key)
        cases = (
            # host, leaf, spine, leaf, host: 5 + 12.5 + 12.5 + 5 us, 4 hops.
            ("clos", clos, 2 * 35_000 + 4 * 40),
            # One leaf: host, leaf, host.
            ("one leaf", one_leaf, 2 * 10_000 + 2 * 40),
            # Eight switches in a ring: four switch hops to the far side.
            ("ring", ring, 2 * (10_000 + 4 * 12_500) + 6 * 40),
        )
        for name, document, rtt in cases:
            manifest, config = self._materialize(document)
            bdp = rtt * 200_000_000_000 // 1_000_000_000 // 8
            with self.subTest(fabric=name):
                self.assertEqual(manifest["switch"]["config_base_rtt_ns"], rtt)
                self.assertEqual(manifest["switch"]["plane_bdp_bytes"], bdp)
                self.assertIn(f"\nDATA_QUEUE_BYTES {bdp}\n", config)
                self.assertIn(_map("KMIN_MAP", round(bdp * 0.2 / 1000)), config)

    def test_ue_preset_refuses_what_it_sets(self) -> None:
        document = _ue(_trimming_clos())
        document["network"]["fabric"]["data_queue_bytes"] = 262144
        self._refused(document, "sets network.fabric.data_queue_bytes")
        document = _ue(_trimming_clos())
        document["network"]["fabric"]["trimmed_queue_bytes"] = 65536
        self._refused(document, "sets network.fabric.data_queue_bytes")
        document = _ue(_trimming_clos())
        document["network"]["congestion_control"] = {
            "mode": "dcqcn",
            "ecn_threshold_scale": 1.0,
        }
        self._refused(document, "ecn_threshold_scale cannot be combined")
        document = _ue(_trimming_clos())
        document["network"]["switch"]["ecn_marking_probability"] = 1.0
        self._refused(document, "fixed at 1 by network.switch.profile 'ue'")

    def test_ue_preset_requires_trimming(self) -> None:
        document = _ue(_trimming_clos())
        document["network"].pop("packet_trimming")
        self._refused(document, "'ue' requires network.packet_trimming")

    def test_default_profile_keeps_fabric_queue_bounds_required(self) -> None:
        document = _trimming_clos()
        document["network"]["switch"] = {"profile": "default"}
        document["network"]["fabric"].pop("data_queue_bytes")
        self._refused(document, "network.fabric must contain")


if __name__ == "__main__":
    unittest.main()
