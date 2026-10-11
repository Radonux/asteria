"""The counter files a run writes under per-packet paths."""

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

PROFILES = REPOSITORY_ROOT / "experiments/ring_3d/profiles"
KEYS = (
    "PORT_COUNTER_OUTPUT_FILE",
    "SPINE_ARRIVAL_OUTPUT_FILE",
    "SPINE_REPORT_OUTPUT_FILE",
)


class FabricCounterTests(unittest.TestCase):
    def test_each_mode_names_the_counters_it_can_fill(self) -> None:
        document = json.loads(
            (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
        )
        written = {
            "ecmp": (),
            "ev_hash": ("PORT_COUNTER_OUTPUT_FILE",),
            "spray_uniform": KEYS[:2],
            "spray_policy": KEYS,
        }
        files = {
            "PORT_COUNTER_OUTPUT_FILE": "port_counters.csv",
            "SPINE_ARRIVAL_OUTPUT_FILE": "spine_arrivals.csv",
            "SPINE_REPORT_OUTPUT_FILE": "spine_reports.csv",
        }
        for mode, keys in written.items():
            document["network"]["load_balancing"] = {"mode": mode}
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "profile.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / "out"
                manifest = materialize(path, output)
                config = (output / "network_config.txt").read_text(encoding="utf-8")
                ns3 = (output / "ns3").resolve()
            with self.subTest(mode=mode):
                for key in KEYS:
                    line = f"\n{key} {ns3 / files[key]}\n"
                    if key in keys:
                        self.assertIn(line, config)
                    else:
                        self.assertNotIn(key, config)
                self.assertEqual(
                    manifest["counter_files"],
                    {key: str(ns3 / files[key]) for key in keys},
                )

    def test_spine_loss_trace_follows_the_pooled_loss_count(self) -> None:
        document = json.loads(
            (PROFILES / "no_incast_8_zero.json").read_text(encoding="utf-8")
        )
        counted = {
            "mode": "spray_policy",
            "loss_thresholds": [0.1, 0.2, 0.3],
            "sender_pool_weight": 0.5,
        }
        cases = (
            (counted, True),
            ({**counted, "loss_interval_ns": 5000}, True),
            ({**counted, "grade_cost_source": "estimate"}, True),
            ({**counted, "sender_pool_weight": 0}, False),
            ({**counted, "grade_cost_source": "switch_queue"}, False),
            ({"mode": "spray_policy", "sender_pool_weight": 0.5}, False),
            ({"mode": "spray_policy", "loss_thresholds": [0.1, 0.2, 0.3]}, False),
            ({"mode": "spray_uniform"}, False),
        )
        for balancing, traced in cases:
            document["network"]["load_balancing"] = balancing
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "profile.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / "out"
                manifest = materialize(path, output)
                config = (output / "network_config.txt").read_text(encoding="utf-8")
                trace = str((output / "ns3").resolve() / "spine_losses.csv")
            with self.subTest(balancing=balancing):
                if traced:
                    self.assertIn(f"\nSPINE_LOSS_OUTPUT_FILE {trace}\n", config)
                    self.assertEqual(
                        manifest["counter_files"]["SPINE_LOSS_OUTPUT_FILE"], trace
                    )
                else:
                    self.assertNotIn("SPINE_LOSS_OUTPUT_FILE", config)
                    self.assertNotIn(
                        "SPINE_LOSS_OUTPUT_FILE", manifest["counter_files"]
                    )


if __name__ == "__main__":
    unittest.main()
