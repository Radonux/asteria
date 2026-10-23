"""mrc_spine and ue_aware_spine under spray_uniform: MRC and UEC path-aware
spraying over the spines, no loss hold."""

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


class SpineBaselineTest(unittest.TestCase):
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

    def test_mrc_spine_writes_mrc_defaults_without_a_set_size(self) -> None:
        manifest, config = self.generated({"mode": "spray_uniform", "selector": "mrc_spine"})
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "PATH_SELECTOR mrc_spine\n"
            "MRC_SKIP_BASE_RTTS 1.0\n"
            "MRC_PROBE_TIMEOUTS 1.0\n",
            config,
        )
        self.assertNotIn("MRC_EV_SET_SIZE", config)
        self.assertNotIn("SPRAY_LOSS", config)
        self.assertEqual(
            manifest["load_balancing"],
            {"mode": "spray_uniform", "selector": "mrc_spine", "skip_base_rtts": 1.0,
             "probe_timeouts": 1.0},
        )

    def test_ue_aware_spine_writes_its_keys_and_the_transport(self) -> None:
        manifest, config = self.generated(
            {"mode": "spray_uniform", "selector": "ue_aware_spine",
             "path_seq": True, "rack_reorder_ns": 3081, "overdue_repair": "reprobe"}
        )
        self.assertIn(
            "\nLOAD_BALANCING spray_uniform\n"
            "PATH_SELECTOR ue_aware_spine\n"
            "UE_SATURATION_FRACTION 0.5\n",
            config,
        )
        for line in ("SPRAY_PATH_SEQ on", "SPRAY_RACK_REORDER_NS 3081", "SPRAY_OVERDUE_REPAIR reprobe"):
            self.assertIn(f"\n{line}\n", config)
        self.assertNotIn("UE_EV_SET_SIZE", config)
        self.assertEqual(manifest["load_balancing"]["saturation_fraction"], 0.5)

    def test_refuses_loss_hold_keys_and_the_set_size(self) -> None:
        for selector, key, value in (("mrc_spine", "ev_set_size", 8), ("mrc_spine", "loss_interval_ns", 1000),
                                     ("ue_aware_spine", "ev_set_size", 8),
                                     ("ue_aware_spine", "receiver_hold", True),
                                     ("ue_aware_spine", "loss_avoid_ns", 1000)):
            with self.subTest(selector=selector, key=key), self.assertRaisesRegex(
                    ValueError, f"'{selector}' does not take"):
                self.generated({"mode": "spray_uniform", "selector": selector, key: value})


if __name__ == "__main__":
    unittest.main()
