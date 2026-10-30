"""Fabrics of up to 256 spines: their leaf-spine link rate, the data frame the
backend sizes its headers at, the ECN maps of their link speeds, the pair
bandwidth a sprayed pair's window is sized by, and a fraction of their link
directions failed by a seeded draw."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.links import draw_link_directions
from experiments.ring_3d.tests.test_links import (
    LEAF,
    SPINE,
    LinkTests,
    _change_lines,
    _sprayed_clos,
)

HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
    "path_seq": True,
    "tail_recovery": True,
    "ack_route": "spray",
    "answer_path_seq": True,
}
LOSS = {"model": "gray", "start_ns": 5, "error_rate": 0.01, "loss_unit": "byte"}


def _fabric(spines: int, rate: str | None = None) -> dict[str, Any]:
    """_sprayed_clos over spines spines at rate, with the CI fabric's link
    delays."""
    document = _sprayed_clos()
    document["network"]["spine_count"] = spines
    document["network"]["host_link_delay_ns"] = 1_000
    document["network"]["switch_link_delay_ns"] = 2_000
    if rate is not None:
        document["network"]["spine_link_rate"] = rate
    return document


class SpineCountTests(LinkTests):
    def test_spine_link_rate_sets_the_leaf_spine_links(self) -> None:
        manifest, topology = self._materialize(_fabric(32, "100Gbps"))
        links = topology.splitlines()[2:]
        self.assertEqual({line.split()[2] for line in links[:64]}, {"400Gbps"})
        self.assertEqual({line.split()[2] for line in links[64:]}, {"100Gbps"})
        self.assertEqual(len(links), 64 + 8 * 32)
        self.assertEqual(manifest["physical_topology"]["spine_link_rate"], "100Gbps")
        # Absent, every link runs at link_rate and the manifest names none.
        manifest, absent = self._materialize(_fabric(8))
        self.assertNotIn("spine_link_rate", manifest["physical_topology"])
        same, explicit = self._materialize(_fabric(8, "400Gbps"))
        self.assertEqual(absent, explicit)

    def test_spine_link_rate_refusals(self) -> None:
        for rate, message in (
            ("13Gbps", "spine_link_rate must be a link speed the switch ECN table"),
            ("fast", "spine_link_rate must be a rate such as 100Gbps"),
            (400, "spine_link_rate must be a rate such as 100Gbps"),
        ):
            with self.subTest(rate=rate):
                self._refused(_fabric(8, rate), message)
        ring = _fabric(8, "100Gbps")
        ring["network"]["topology"] = "ring"
        for key in ("hosts_per_leaf", "spine_count", "load_balancing"):
            ring["network"].pop(key)
        self._refused(ring, "unknown network keys: \\['spine_link_rate'\\]")

    def test_the_ecn_maps_cover_the_fabric_link_speeds(self) -> None:
        def maps(document: dict[str, Any]) -> list[str]:
            return [
                line
                for line in self._config(document).splitlines()
                if line.split()[:1] in (["KMAX_MAP"], ["KMIN_MAP"], ["PMAX_MAP"])
            ]

        def rates(lines: list[str]) -> list[list[str]]:
            return [line.split()[2::2] for line in lines]

        eight = maps(_fabric(8))
        self.assertTrue(all(line.split()[1] == "6" for line in eight))
        # The table's rows alone where every link speed is in it.
        self.assertEqual(rates(maps(_fabric(32, "100Gbps"))), rates(eight))
        wide = maps(_fabric(256, "12.5Gbps"))
        for line, before in zip(wide, eight):
            words = line.split()
            self.assertEqual(words[1], "7")
            self.assertEqual(words[2], "12500000000")
            # The other rows are the table's, at the fabric's thresholds.
            self.assertEqual(len(words[4:]), len(before.split()[2:]))
        default = _fabric(256, "12.5Gbps")
        default["network"].pop("switch")
        default["network"]["fabric"]["data_queue_bytes"] = 1_000_000
        kmax, kmin, pmax = maps(default)
        self.assertEqual(kmax.split()[:4], ["KMAX_MAP", "7", "12500000000", "200"])
        self.assertEqual(kmin.split()[:4], ["KMIN_MAP", "7", "12500000000", "50"])
        self.assertEqual(pmax.split()[:4], ["PMAX_MAP", "7", "12500000000", "0.2"])

    def test_the_base_rtt_follows_the_spine_links(self) -> None:
        # Per hop a 4096 B packet takes 81 ns at 400 Gb/s and 2,621 ns at
        # 12.5 Gb/s, over two host links and two leaf-spine links.
        for spines, rate, rtt in ((8, None, 12_324), (256, "12.5Gbps", 17_404)):
            with self.subTest(spines=spines):
                manifest, _ = self._materialize(_fabric(spines, rate))
                self.assertEqual(manifest["switch"]["config_base_rtt_ns"], rtt)
                self.assertEqual(manifest["switch"]["plane_bdp_bytes"], rtt * 50)

    def test_the_data_frame_counts_the_headers_over_256_spines(self) -> None:
        # The payload, the 36 B of headers, the path sequence, a bit per spine
        # of answer_hold, and the tail fields: a count per spine over at most
        # 8 spines, the flags alone over more, which only a closing header
        # follows with its counts.
        for spines, header in ((8, 36 + 2 + 1 + 1 + 16), (9, 36 + 2 + 2 + 1), (256, 36 + 2 + 32 + 1)):
            with self.subTest(spines=spines):
                document = _fabric(spines)
                document["network"]["load_balancing"] = HOLD
                document["network"]["link_failures"] = [{**LOSS, "spine": 0, "leaf": 0}]
                lines = _change_lines(self._config(document))
                self.assertEqual(lines[0].split()[-1], str(4096 + header))

    def test_pair_bandwidth(self) -> None:
        document = _fabric(256, "12.5Gbps")
        document["network"]["pair_bandwidth"] = "spines"
        manifest, _ = self._materialize(document)
        self.assertEqual(manifest["physical_topology"]["pair_bandwidth"], "spines")
        self.assertIn("\nPAIR_BANDWIDTH spines\n", self._config(document))
        document["network"]["pair_bandwidth"] = "path"
        self.assertIn("\nPAIR_BANDWIDTH path\n", self._config(document))
        self.assertNotIn("PAIR_BANDWIDTH", self._config(_fabric(8)))
        refused = _fabric(8)
        refused["network"]["pair_bandwidth"] = "spray"
        self._refused(refused, "pair_bandwidth must be one of \\['path', 'spines'\\]")
        refused["network"]["pair_bandwidth"] = "spines"
        refused["network"]["load_balancing"] = {"mode": "ev_hash", "selector": "reps"}
        self._refused(refused, "pair_bandwidth 'spines' requires a load_balancing mode")


class PortRateThresholdTests(LinkTests):
    def _thresholds(self, document: dict[str, Any]) -> dict[str, dict[int, int]]:
        rows = {}
        for line in self._config(document).splitlines():
            words = line.split()
            if words and words[0] in (
                "KMAX_MAP", "KMIN_MAP", "DATA_QUEUE_BYTES_MAP", "TRIMMED_QUEUE_BYTES_MAP"
            ):
                rows[words[0]] = {
                    int(rate): int(value) for rate, value in zip(words[2::2], words[3::2])
                }
        return rows

    def test_each_port_takes_its_speed_share(self) -> None:
        document = _fabric(256, "12.5Gbps")
        document["network"]["switch"] = {"profile": "ue", "port_rate_thresholds": True}
        rows = self._thresholds(document)
        # Plane_BDP 17,404 ns at 400 Gb/s: 870,200 B, a 12.5 Gb/s port's share
        # a 32nd of it.
        self.assertEqual(rows["KMAX_MAP"][400_000_000_000], 696)
        self.assertEqual(rows["KMAX_MAP"][12_500_000_000], 22)
        self.assertEqual(rows["KMIN_MAP"][400_000_000_000], 174)
        self.assertEqual(rows["KMIN_MAP"][12_500_000_000], 5)
        for key in ("DATA_QUEUE_BYTES_MAP", "TRIMMED_QUEUE_BYTES_MAP"):
            self.assertEqual(rows[key][400_000_000_000], 870_200)
            self.assertEqual(rows[key][12_500_000_000], 27_193)
            self.assertEqual(rows[key][25_000_000_000], 54_387)
        manifest, _ = self._materialize(document)
        self.assertEqual(
            manifest["switch"]["port_rate_thresholds"]["12500000000"],
            {"ecn_min_bytes": 5000, "ecn_max_bytes": 22000,
             "trim_threshold_bytes": 27_193, "trimmed_queue_drop_bytes": 27_193},
        )
        self.assertEqual(sorted(manifest["switch"]["port_rate_thresholds"]),
                         ["12500000000", "400000000000"])

    def test_a_port_at_link_rate_keeps_its_bytes(self) -> None:
        plain = self._thresholds(_fabric(8))
        document = _fabric(8)
        document["network"]["switch"] = {"profile": "ue", "port_rate_thresholds": True}
        scaled = self._thresholds(document)
        for key in ("KMAX_MAP", "KMIN_MAP"):
            self.assertEqual(scaled[key][400_000_000_000], plain[key][400_000_000_000])
        self.assertNotIn("DATA_QUEUE_BYTES_MAP", plain)
        self.assertEqual(scaled["DATA_QUEUE_BYTES_MAP"][400_000_000_000], 616_200)
        self.assertIn("\nDATA_QUEUE_BYTES 616200\n", self._config(document))
        manifest, _ = self._materialize(_fabric(8))
        self.assertNotIn("port_rate_thresholds", manifest["switch"])

    def test_refusals(self) -> None:
        document = _fabric(8)
        document["network"]["switch"] = {"profile": "default", "port_rate_thresholds": True}
        document["network"]["fabric"]["data_queue_bytes"] = 1_000_000
        self._refused(document, "port_rate_thresholds is read only by profile 'ue'")
        document["network"]["switch"] = {"profile": "ue", "port_rate_thresholds": 1}
        self._refused(document, "port_rate_thresholds must be a boolean")


class LinkDrawTests(LinkTests):
    def test_the_draw_fails_the_same_fraction_at_every_spine_count(self) -> None:
        for spines, count in ((8, 3), (32, 10), (128, 41), (256, 82)):
            with self.subTest(spines=spines):
                drawn = draw_link_directions(0.02, 7, 8, spines)
                directions = sum(2 if d == "both" else 1 for _, _, d in drawn)
                self.assertEqual(directions, count)
                self.assertEqual(drawn, sorted(drawn))
                self.assertEqual(drawn, draw_link_directions(0.02, 7, 8, spines))
        self.assertNotEqual(
            draw_link_directions(0.02, 7, 8, 256), draw_link_directions(0.02, 8, 8, 256)
        )
        # Every direction of a link drawn is one failure of it in both.
        self.assertEqual(
            draw_link_directions(1.0, 7, 2, 2),
            [(0, 0, "both"), (0, 1, "both"), (1, 0, "both"), (1, 1, "both")],
        )

    def test_a_drawn_failure_writes_each_drawn_link_in_its_direction(self) -> None:
        for model, extra in (("gray", {"error_rate": 0.01, "loss_unit": "byte"}),
                             ("silent", {})):
            with self.subTest(model=model):
                document = _fabric(32, "100Gbps")
                document["network"]["link_failures"] = [
                    {"model": model, "start_ns": 5, "draw": {"fraction": 0.05, "seed": 3},
                     **extra}
                ]
                drawn = draw_link_directions(0.05, 3, 8, 32)
                lines = _change_lines(self._config(document))
                expected = 0
                for leaf, spine, direction in drawn:
                    leaf_node, spine_node = LEAF + leaf, SPINE + spine
                    pairs = {
                        "up": [(leaf_node, spine_node)],
                        "down": [(spine_node, leaf_node)],
                        "both": [(leaf_node, spine_node)]
                        + ([(spine_node, leaf_node)] if model == "silent" else []),
                    }[direction]
                    for a, b in pairs:
                        expected += 1
                        self.assertTrue(
                            any(line.split()[3:5] == [str(a), str(b)] for line in lines),
                            (leaf, spine, direction),
                        )
                self.assertEqual(len(lines), expected)
                manifest, _ = self._materialize(document)
                recorded = manifest["link_failures"]
                self.assertEqual(len(recorded), len(drawn))
                self.assertEqual(
                    [(f["leaf"], f["spine"], f["direction"]) for f in recorded], drawn
                )
                self.assertEqual(
                    {str(f["draw"]) for f in recorded}, {"{'fraction': 0.05, 'seed': 3}"}
                )

    def test_draw_refusals(self) -> None:
        for failure, message in (
            ({"model": "graceful", "start_ns": 5, "draw": {"fraction": 0.1, "seed": 1}},
             "draws its links and must be one of \\['gray', 'silent'\\]"),
            ({**LOSS, "spine": 2, "draw": {"fraction": 0.1, "seed": 1}},
             "must contain model, start_ns and spine, host or draw"),
            ({**LOSS, "direction": "up", "draw": {"fraction": 0.1, "seed": 1}},
             "must contain model, start_ns and spine, host or draw"),
            ({**LOSS, "draw": {"fraction": 0.1}}, "draw must contain exactly fraction and seed"),
            ({**LOSS, "draw": {"fraction": 0, "seed": 1}}, "draw.fraction must be a number"),
            ({**LOSS, "draw": {"fraction": 0.1, "seed": -1}}, "draw.seed must be a nonnegative"),
            ({**LOSS, "draw": {"fraction": 0.001, "seed": 1}}, "draws no link direction of the 128"),
        ):
            with self.subTest(failure=failure):
                document = _fabric(8)
                document["network"]["link_failures"] = [failure]
                self._refused(document, message)
        # A drawn link is refused where a named failure of it lasts.
        document = _fabric(8)
        leaf, spine, _ = draw_link_directions(1.0, 1, 8, 8)[0]
        document["network"]["link_failures"] = [
            {**LOSS, "spine": spine, "leaf": leaf},
            {**LOSS, "draw": {"fraction": 1.0, "seed": 1}},
        ]
        self._refused(document, "fails a link while an earlier failure of it lasts")


if __name__ == "__main__":
    import unittest

    unittest.main()
