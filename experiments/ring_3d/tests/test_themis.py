"""The themis selector under spray_uniform."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.generate import materialize

PROFILE = REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
THEMIS = {"mode": "spray_uniform", "selector": "themis"}


class ThemisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str]:
        self.document["network"]["load_balancing"] = balancing
        self.document["network"]["spine_count"] = 4
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        return manifest, (output / "network_config.txt").read_text(encoding="utf-8")

    def test_defaults_are_themis(self) -> None:
        manifest, config = self.generated(THEMIS)
        self.assertIn("\nPATH_SELECTOR themis\n", config)
        for line in (
            "SPRAY_THEMIS_PATHS 0",
            "SPRAY_THEMIS_OOO_THRESHOLD 448",
            "SPRAY_THEMIS_STASH_MIN_NS 0",
            "SPRAY_THEMIS_AVOIDANCE_WINDOW 2000000",
            "SPRAY_THEMIS_RETX_REROUTE on",
            "SPRAY_THEMIS_LEAF_AVOIDANCE off",
        ):
            self.assertIn(f"\n{line}\n", config)
        self.assertEqual(manifest["load_balancing"]["selector"], "themis")
        self.assertEqual(manifest["load_balancing"]["themis_ooo_threshold"], 448)

    def test_written_as_given_with_the_common_transport(self) -> None:
        manifest, config = self.generated(
            {
                **THEMIS,
                "themis_paths": 2,
                "themis_ooo_threshold": 5,
                "themis_stash_min_ns": 24648,
                "themis_avoidance_window": 0,
                "themis_retx_reroute": False,
                "themis_leaf_avoidance": True,
                "path_seq": True,
                "rack_reorder_ns": 3081,
                "overdue_repair": "reprobe",
            }
        )
        for line in (
            "SPRAY_THEMIS_PATHS 2",
            "SPRAY_THEMIS_OOO_THRESHOLD 5",
            "SPRAY_THEMIS_STASH_MIN_NS 24648",
            "SPRAY_THEMIS_AVOIDANCE_WINDOW 0",
            "SPRAY_THEMIS_RETX_REROUTE off",
            "SPRAY_THEMIS_LEAF_AVOIDANCE on",
            "SPRAY_PATH_SEQ on",
            "SPRAY_RACK_REORDER_NS 3081",
            "SPRAY_OVERDUE_REPAIR reprobe",
        ):
            self.assertIn(f"\n{line}\n", config)
        self.assertFalse(manifest["load_balancing"]["themis_retx_reroute"])
        self.assertTrue(manifest["load_balancing"]["themis_leaf_avoidance"])

    def test_refuses_loss_hold_keys(self) -> None:
        for key, value in (("loss_interval_ns", 100000), ("receiver_hold", True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "'themis' does not take"):
                self.generated({**THEMIS, key: value})

    def test_refuses_a_negative_threshold(self) -> None:
        for key in ("themis_ooo_threshold", "themis_stash_min_ns"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.generated({**THEMIS, key: -1})


if __name__ == "__main__":
    unittest.main()
