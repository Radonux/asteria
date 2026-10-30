"""The unit of a gray loss failure's error_rate: each packet, the default, or
each byte, derived so that a full data frame keeps the rate."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.tests.test_links import (
    LEAF,
    SPINE,
    LinkTests,
    _change_lines,
    _sprayed_clos,
)

LOSS = {"model": "gray", "spine": 2, "leaf": 0, "start_ns": 5, "error_rate": 0.01}
# 4096 B of payload and the 36 B of PPP, IPv4, UDP and data headers.
FRAME = 4096 + 36


class LossUnitTests(LinkTests):
    def _lines(self, *failures: dict, balancing: dict | None = None) -> list[str]:
        document = _sprayed_clos()
        document["network"]["link_failures"] = list(failures)
        if balancing is not None:
            document["network"]["load_balancing"] = balancing
        return _change_lines(self._config(document))

    def _per_byte(self, line: str) -> float:
        return float(line.split()[5])

    def test_frame_is_the_default_and_writes_the_same_lines(self) -> None:
        for failure in (LOSS, {**LOSS, "direction": "down"}):
            with self.subTest(failure=failure):
                self.assertEqual(
                    self._lines({**failure, "loss_unit": "frame"}), self._lines(failure)
                )
        self.assertEqual(
            self._lines(LOSS), [f"LINK_FAILURE 5 loss {LEAF} {SPINE + 2} 0.01"]
        )

    def test_byte_writes_the_rate_per_byte_and_the_frame(self) -> None:
        lines = self._lines({**LOSS, "loss_unit": "byte", "direction": "up"})
        self.assertEqual(len(lines), 1)
        words = lines[0].split()
        self.assertEqual(words[:5], ["LINK_FAILURE", "5", "loss", str(LEAF), str(SPINE + 2)])
        self.assertEqual(words[6:], ["forward", "byte", str(FRAME)])
        per_byte = self._per_byte(lines[0])
        # A full data frame keeps the registered rate; a 48 B packet is lost
        # about FRAME / 48 times less often.
        self.assertAlmostEqual(1 - (1 - per_byte) ** FRAME, 0.01, places=12)
        self.assertAlmostEqual(
            (1 - (1 - per_byte) ** 48) / 0.01, 48 / FRAME, delta=0.01 * 48 / FRAME
        )
        both = self._lines({**LOSS, "loss_unit": "byte"})
        self.assertEqual(both[0].split()[6:], ["byte", str(FRAME)])

    def test_the_frame_counts_the_headers_the_run_sets(self) -> None:
        for balancing, frame in (
            ({"mode": "spray_uniform", "path_seq": True}, FRAME + 2),
            (
                {"mode": "spray_uniform", "path_seq": True, "tail_recovery": True},
                FRAME + 2 + 1 + 2 * 8,
            ),
            ({"mode": "spray_uniform", "selector": "ack_redirect"}, FRAME + 8),
            (
                {
                    "mode": "spray_uniform",
                    "selector": "reps_hold",
                    "loss_thresholds": [0.01, 0.02, 0.05],
                    "loss_interval_ns": 100000,
                    "loss_hold_clean_sentinels": 4,
                    "held_exploration": "sentinel",
                    "path_seq": True,
                    "ack_route": "spray",
                    "answer_path_seq": True,
                },
                FRAME + 2 + 1,
            ),
        ):
            with self.subTest(balancing=balancing):
                lines = self._lines({**LOSS, "loss_unit": "byte"}, balancing=balancing)
                self.assertEqual(lines[0].split()[-1], str(frame))
                per_byte = self._per_byte(lines[0])
                self.assertAlmostEqual(1 - (1 - per_byte) ** frame, 0.01, places=12)

    def test_manifest_records_the_unit_frame_and_rate(self) -> None:
        document = _sprayed_clos()
        document["network"]["link_failures"] = [
            LOSS,
            {**LOSS, "spine": 3, "loss_unit": "frame"},
            {**LOSS, "spine": 4, "loss_unit": "byte"},
        ]
        manifest, _ = self._materialize(document)
        config = self._config(document)
        recorded = manifest["link_failures"]
        self.assertNotIn("loss_unit", recorded[0])
        self.assertEqual(recorded[1]["loss_unit"], "frame")
        self.assertNotIn("reference_frame_bytes", recorded[1])
        self.assertEqual(recorded[2]["loss_unit"], "byte")
        self.assertEqual(recorded[2]["reference_frame_bytes"], FRAME)
        self.assertEqual(recorded[2]["error_rate"], 0.01)
        per_byte = recorded[2]["per_byte_error_rate"]
        self.assertIn(f" {format(per_byte, '.17g')} byte {FRAME}\n", config)

    def test_refusals(self) -> None:
        cases = [
            ({**LOSS, "loss_unit": "bit"}, r"loss_unit must be one of"),
            ({**LOSS, "loss_unit": None}, r"loss_unit must be one of"),
        ]
        for failure in (
            {"model": "silent", "spine": 1, "leaf": 4, "start_ns": 0},
            {"model": "graceful", "spine": 3, "start_ns": 0},
            {"model": "gray", "spine": 2, "leaf": 1, "start_ns": 5, "rate": "200Gbps"},
        ):
            cases.append(
                (
                    {**failure, "loss_unit": "byte"},
                    "loss_unit is read only by a gray failure that sets error_rate",
                )
            )
        for failure, message in cases:
            document = _sprayed_clos()
            document["network"]["link_failures"] = [failure]
            with self.subTest(failure=failure):
                self._refused(document, message)


if __name__ == "__main__":
    import unittest

    unittest.main()
