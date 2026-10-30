"""copy_expiry under the loss hold selectors with sentinel_mode duplicate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.generate import load_profile, materialize

PROFILE = REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
HOLD = {
    "mode": "spray_uniform",
    "selector": "reps_hold",
    "loss_thresholds": [0.01, 0.02, 0.05],
    "loss_interval_ns": 100000,
    "loss_hold_clean_sentinels": 4,
    "held_exploration": "sentinel",
    "sentinel_mode": "duplicate",
}
FLOW_HOLD = {**HOLD, "selector": "reps_flow_hold"}
MRC_HOLD = {**HOLD, "selector": "mrc_hold"}
TAIL = {"path_seq": True, "tail_recovery": True, "tail_loss_rtts": 4}


class CopyExpiryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROFILE.read_text(encoding="utf-8"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile_path = self.directory / "profile.json"
        self.runs = 0

    def write(self, balancing: dict[str, object]) -> None:
        self.document["network"]["load_balancing"] = balancing
        self.document["network"]["spine_count"] = 4
        self.profile_path.write_text(json.dumps(self.document), encoding="utf-8")

    def generated(self, balancing: dict[str, object]) -> tuple[dict, str]:
        self.write(balancing)
        self.runs += 1
        output = self.directory / f"run{self.runs}"
        manifest = materialize(self.profile_path, output)
        config = (output / "network_config.txt").read_text(encoding="utf-8")
        return manifest, config

    def test_every_loss_hold_selector_writes_it_last(self) -> None:
        for balancing in (HOLD, FLOW_HOLD, MRC_HOLD):
            for expiry in ("data_ack", "deadline"):
                with self.subTest(balancing=balancing, expiry=expiry):
                    manifest, config = self.generated(
                        {**balancing, **TAIL, "ack_route": "spray", "copy_expiry": expiry}
                    )
                    self.assertIn(
                        f"\nACK_ROUTE spray\nSPRAY_COPY_EXPIRY {expiry}\n", config
                    )
                    self.assertEqual(manifest["load_balancing"]["copy_expiry"], expiry)
        # Without tail recovery the guard is the backend's default of 1.
        _, config = self.generated({**HOLD, "copy_expiry": "deadline"})
        self.assertIn("\nSPRAY_COPY_EXPIRY deadline\n", config)
        self.assertNotIn("SPRAY_TAIL_LOSS_RTTS", config)

    def test_unset_is_not_written(self) -> None:
        manifest, config = self.generated({**HOLD, **TAIL})
        self.assertNotIn("SPRAY_COPY_EXPIRY", config)
        self.assertNotIn("copy_expiry", manifest["load_balancing"])

    def test_refusals(self) -> None:
        refusals: list[tuple[dict[str, object], str]] = [
            ({**HOLD, "copy_expiry": value}, "copy_expiry must be 'data_ack' or 'deadline'")
            for value in ("own_ack", "", None, True, 1)
        ]
        refusals += [
            (
                {**HOLD, "sentinel_mode": "data", "copy_expiry": "deadline"},
                "copy_expiry requires sentinel_mode 'duplicate'",
            ),
            (
                {key: value for key, value in HOLD.items() if key != "sentinel_mode"}
                | {"copy_expiry": "data_ack"},
                "copy_expiry requires sentinel_mode 'duplicate'",
            ),
            (
                {**HOLD, **TAIL, "tail_loss_rtts": 0, "copy_expiry": "deadline"},
                "copy_expiry 'deadline' requires tail_loss_rtts above 0",
            ),
            ({"mode": "spray_uniform", "copy_expiry": "deadline"}, "takes no other key"),
            (
                {"mode": "spray_uniform", "selector": "reps_spine", "copy_expiry": "deadline"},
                "does not take",
            ),
            ({"mode": "spray_policy", "copy_expiry": "deadline"}, "does not take"),
        ]
        for balancing, message in refusals:
            self.write(balancing)
            with self.subTest(balancing=balancing):
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(self.profile_path)


if __name__ == "__main__":
    unittest.main()
