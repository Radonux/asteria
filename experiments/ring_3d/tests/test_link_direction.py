"""The direction of a link failure: which way a silent failure drops, and
the refusal of any direction but "both" for the failures the backend applies
to a whole link."""

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

SILENT = {"model": "silent", "spine": 1, "leaf": 4, "start_ns": 0}
LOSS = {"model": "gray", "spine": 2, "leaf": 0, "start_ns": 5, "error_rate": 0.01}
RATE = {"model": "gray", "spine": 2, "leaf": 1, "start_ns": 5, "rate": "200Gbps"}
GRACEFUL = {"model": "graceful", "spine": 3, "start_ns": 200_000}


class LinkDirectionTests(LinkTests):
    def _lines(self, *failures: dict) -> list[str]:
        document = _sprayed_clos()
        document["network"]["link_failures"] = list(failures)
        return _change_lines(self._config(document))

    def test_up_is_the_default_and_writes_the_same_lines(self) -> None:
        for failure in (SILENT, {**SILENT, "duration_ns": 10}, LOSS, RATE, GRACEFUL):
            with self.subTest(failure=failure):
                explicit = {
                    **failure,
                    "direction": "up" if failure["model"] == "silent" else "both",
                }
                absent = _sprayed_clos()
                absent["network"]["link_failures"] = [failure]
                given = _sprayed_clos()
                given["network"]["link_failures"] = [explicit]
                self.assertEqual(self._config(absent), self._config(given))
        self.assertEqual(
            self._lines(SILENT), [f"LINK_FAILURE 0 blackhole {LEAF + 4} {SPINE + 1}"]
        )

    def test_down_names_the_spine_first(self) -> None:
        self.assertEqual(
            self._lines({**SILENT, "direction": "down"}),
            [f"LINK_FAILURE 0 blackhole {SPINE + 1} {LEAF + 4}"],
        )
        self.assertEqual(
            self._lines(
                {
                    **SILENT,
                    "direction": "down",
                    "duration_ns": 10,
                    "repeat": 2,
                    "period_ns": 100,
                }
            ),
            [
                f"LINK_FAILURE 0 blackhole {SPINE + 1} {LEAF + 4}",
                f"LINK_RESTORE 10 {SPINE + 1} {LEAF + 4}",
                f"LINK_FAILURE 100 blackhole {SPINE + 1} {LEAF + 4}",
                f"LINK_RESTORE 110 {SPINE + 1} {LEAF + 4}",
            ],
        )

    def test_both_writes_two_lines_per_link(self) -> None:
        self.assertEqual(
            self._lines({**SILENT, "direction": "both"}),
            [
                f"LINK_FAILURE 0 blackhole {LEAF + 4} {SPINE + 1}",
                f"LINK_FAILURE 0 blackhole {SPINE + 1} {LEAF + 4}",
            ],
        )
        whole_spine = {"model": "silent", "spine": 5, "start_ns": 7}
        self.assertEqual(
            self._lines({**whole_spine, "direction": "both"}),
            [
                line
                for leaf in range(8)
                for line in (
                    f"LINK_FAILURE 7 blackhole {LEAF + leaf} {SPINE + 5}",
                    f"LINK_FAILURE 7 blackhole {SPINE + 5} {LEAF + leaf}",
                )
            ],
        )

    def test_two_sided_failures_take_both(self) -> None:
        self.assertEqual(
            self._lines({**LOSS, "direction": "both"}, {**RATE, "direction": "both"}),
            [
                f"LINK_FAILURE 5 loss {LEAF} {SPINE + 2} 0.01",
                f"LINK_FAILURE 5 rate {LEAF + 1} {SPINE + 2} 200Gbps",
            ],
        )

    def test_manifest_records_direction_only_when_set(self) -> None:
        document = _sprayed_clos()
        document["network"]["link_failures"] = [
            SILENT,
            {**SILENT, "spine": 2, "direction": "down"},
            {**LOSS, "direction": "both"},
        ]
        manifest, _ = self._materialize(document)
        recorded = manifest["link_failures"]
        self.assertNotIn("direction", recorded[0])
        self.assertEqual(recorded[1]["direction"], "down")
        self.assertEqual(recorded[2]["direction"], "both")

    def test_refusals(self) -> None:
        cases = [
            ({**SILENT, "direction": "sideways"}, r"direction must be one of"),
            ({**SILENT, "direction": None}, r"direction must be one of"),
            (
                {**SILENT, "direction": "both", "duration_ns": 10},
                "takes no duration_ns",
            ),
            ({**GRACEFUL, "direction": "up"}, "'both' for a graceful failure"),
        ]
        cases += [
            ({**failure, "direction": direction}, "'both' for a gray failure")
            for failure in (LOSS, RATE)
            for direction in ("up", "down")
        ]
        for failure, message in cases:
            document = _sprayed_clos()
            document["network"]["link_failures"] = [failure]
            with self.subTest(failure=failure):
                self._refused(document, message)

    def test_directions_of_one_link_may_not_overlap(self) -> None:
        document = _sprayed_clos()
        document["network"]["link_failures"] = [
            {**SILENT, "duration_ns": 10},
            {**SILENT, "start_ns": 5, "direction": "down"},
        ]
        self._refused(document, "while an earlier failure of it lasts")


if __name__ == "__main__":
    import unittest

    unittest.main()
