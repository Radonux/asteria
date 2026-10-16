"""A link failure of one host's link to its leaf, named by the host: the
order its LINK_FAILURE lines name the two nodes in, its models and loss
unit, its manifest, and its refusals."""

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

# Host 9 is the second host of leaf 1.
HOST = 9
HOST_LEAF = LEAF + 1
LOSS = {"model": "gray", "host": HOST, "start_ns": 5, "error_rate": 0.01}
RATE = {"model": "gray", "host": HOST, "start_ns": 5, "rate": "200Gbps"}
SILENT = {"model": "silent", "host": HOST, "start_ns": 0}


class HostLinkTests(LinkTests):
    def _document(self, *failures: dict) -> dict:
        document = _sprayed_clos()
        document["network"]["link_failures"] = list(failures)
        return document

    def _lines(self, *failures: dict) -> list[str]:
        return _change_lines(self._config(self._document(*failures)))

    def test_up_names_the_host_first_and_down_its_leaf(self) -> None:
        self.assertEqual(
            self._lines({**LOSS, "direction": "up"}),
            [f"LINK_FAILURE 5 loss {HOST} {HOST_LEAF} 0.01 forward"],
        )
        self.assertEqual(
            self._lines({**LOSS, "direction": "down"}),
            [f"LINK_FAILURE 5 loss {HOST_LEAF} {HOST} 0.01 forward"],
        )
        self.assertEqual(
            self._lines(LOSS), [f"LINK_FAILURE 5 loss {HOST} {HOST_LEAF} 0.01"]
        )
        self.assertEqual(
            self._lines({**RATE, "direction": "down"}),
            [f"LINK_FAILURE 5 rate {HOST_LEAF} {HOST} 200Gbps forward"],
        )
        self.assertEqual(
            self._lines(SILENT), [f"LINK_FAILURE 0 blackhole {HOST} {HOST_LEAF}"]
        )
        self.assertEqual(
            self._lines({**SILENT, "direction": "up", "duration_ns": 10}),
            [
                f"LINK_FAILURE 0 blackhole {HOST} {HOST_LEAF}",
                f"LINK_RESTORE 10 {HOST} {HOST_LEAF}",
            ],
        )
        # The last host of the last leaf.
        self.assertEqual(
            self._lines({**LOSS, "host": 63, "direction": "up"}),
            [f"LINK_FAILURE 5 loss 63 {LEAF + 7} 0.01 forward"],
        )

    def test_byte_unit(self) -> None:
        lines = self._lines({**LOSS, "direction": "down", "loss_unit": "byte"})
        words = lines[0].split()
        self.assertEqual(words[:5], ["LINK_FAILURE", "5", "loss", str(HOST_LEAF), str(HOST)])
        self.assertEqual(words[6:8], ["forward", "byte"])
        frame = int(words[8])
        self.assertAlmostEqual(1 - (1 - float(words[5])) ** frame, 0.01, places=12)

    def test_manifest(self) -> None:
        manifest, _ = self._materialize(
            self._document(
                {**LOSS, "direction": "up", "loss_unit": "byte"},
                {"model": "gray", "spine": 3, "leaf": 0, "start_ns": 5, "error_rate": 0.01},
            )
        )
        host, spine = manifest["link_failures"]
        self.assertEqual(host["host"], HOST)
        self.assertNotIn("spine", host)
        self.assertNotIn("leaf", host)
        self.assertEqual((host["direction"], host["loss_unit"]), ("up", "byte"))
        self.assertNotIn("host", spine)
        self.assertEqual((spine["spine"], spine["leaf"]), (3, 0))

    def test_a_host_link_and_a_spine_link_fail_together(self) -> None:
        self.assertEqual(
            self._lines(
                {**LOSS, "direction": "up"},
                {**LOSS, "host": HOST + 1, "direction": "down", "start_ns": 7},
                {"model": "silent", "spine": 1, "leaf": 1, "start_ns": 5},
            ),
            [
                f"LINK_FAILURE 5 loss {HOST} {HOST_LEAF} 0.01 forward",
                f"LINK_FAILURE 7 loss {HOST_LEAF} {HOST + 1} 0.01 forward",
                f"LINK_FAILURE 5 blackhole {HOST_LEAF} {SPINE + 1}",
            ],
        )

    def test_refusals(self) -> None:
        for failure, message in (
            ({**LOSS, "host": 64}, r"host must be an integer in \[0, 64\)"),
            ({**LOSS, "host": True}, r"host must be an integer"),
            ({**LOSS, "spine": 3}, "must contain model, start_ns and spine, or model, start_ns and host"),
            ({**LOSS, "leaf": 1}, "may contain"),
            ({"model": "graceful", "host": HOST, "start_ns": 0}, "would cut host 9 off"),
            ({**SILENT, "direction": "down"}, "must be 'up' for a silent host link"),
            ({**SILENT, "direction": "both"}, "must be 'up' for a silent host link"),
            ({**SILENT, "loss_unit": "byte"}, "loss_unit is read only by a gray failure"),
        ):
            with self.subTest(failure=failure):
                self._refused(self._document(failure), message)
        # A host's link fails once at a time, whatever the directions, as a
        # leaf-spine link does.
        self._refused(
            self._document({**LOSS, "direction": "up"}, {**RATE, "direction": "down", "start_ns": 10}),
            "fails a link while an earlier failure of it lasts",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
