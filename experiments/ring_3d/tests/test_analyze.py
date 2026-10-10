from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.ring_3d.analyze import summarize
from experiments.ring_3d.generate import DECISION_SCALE, scaled_threshold

FLOW_FIELDS = [
    "flow_kind",
    "decision",
    "admission_eligible",
    "parallelism_domain",
    "origin_transport_role",
    "transport_role",
    "collective_type",
    "training_step",
    "workload_node_id",
    "message_sequence",
    "src",
    "dst",
    "tag",
    "source_port",
    "priority_group",
    "logical_bytes",
    "physical_bytes",
    "data_attempted_bytes",
    "retransmitted_bytes",
    "trimmed_payload_bytes",
    "recovery_events",
    "trim_notifications",
    "trim_lasthop_notifications",
    "trim_recovery_events",
    "stale_trim_notifications",
    "terminal_outcome",
    "failure_reason",
    "decision_hash",
    "start_time_ns",
    "end_time_ns",
    "timeouts",
    "cnp_received",
    "first_trim_ns",
    "first_repair_ns",
    "forgiven_bytes",
    "forgiven_ranges",
    "forgiven_remainder_bytes",
    "pacing_refusals",
    "late_forgiven_bytes",
    "delivered_bytes",
    "cc_exempt",
    "cc_signal_withheld",
    "allowance_gone_reports",
    "cc_transitions",
    "cc_obeying_ns",
]

COLLECTIVE_FIELDS = [
    "rank",
    "parallelism_domain",
    "collective_type",
    "training_step",
    "workload_node_id",
    "logical_bytes",
    "start_time_ns",
    "end_time_ns",
]


class Ring3DAnalysisTests(unittest.TestCase):
    def write_telemetry(
        self,
        directory: Path,
        flows: dict[str, str] | list[dict[str, str]],
        completions: list[tuple[int, int]] | None = None,
        collectives: list[dict[str, str]] | None = None,
    ) -> None:
        directory.mkdir(parents=True)
        if isinstance(flows, dict):
            flows = [flows]
        with (directory / "flow_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FLOW_FIELDS)
            writer.writeheader()
            writer.writerows(flows)
        (directory / "rank_completion.csv").write_text(
            "rank,completion_time_ns\n"
            + "".join(
                f"{rank},{completion_time_ns}\n"
                for rank, completion_time_ns in (completions or [(0, 100), (1, 120)])
            ),
            encoding="utf-8",
        )
        if collectives is not None:
            with (directory / "collective_events.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=COLLECTIVE_FIELDS)
                writer.writeheader()
                writer.writerows(collectives)

    def write_fct(self, directory: Path, flows: list[dict[str, str]]) -> Path:
        directory.mkdir(parents=True)
        lines = []
        for flow in flows:
            source_address = 0x0B000001 | (int(flow["src"]) << 8)
            destination_address = 0x0B000001 | (int(flow["dst"]) << 8)
            duration = int(flow["end_time_ns"]) - int(flow["start_time_ns"])
            lines.append(
                f"{source_address:08x} {destination_address:08x} "
                f"{flow['source_port']} 100 {flow['physical_bytes']} "
                f"{flow['start_time_ns']} {duration} {duration}\n"
            )
        path = directory / "fct.txt"
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def valid_shed_flow(self) -> dict[str, str]:
        return {
            "flow_kind": "provenance_control",
            "decision": "shed",
            "admission_eligible": "true",
            "parallelism_domain": "dp",
            "origin_transport_role": "collective_payload",
            "transport_role": "provenance_control",
            "collective_type": "all_reduce",
            "training_step": "2",
            "workload_node_id": "7",
            "message_sequence": "3",
            "src": "0",
            "dst": "4",
            "tag": "5",
            "source_port": "10000",
            "priority_group": "1",
            "logical_bytes": "1048576",
            "physical_bytes": "64",
            "data_attempted_bytes": "64",
            "retransmitted_bytes": "0",
            "trimmed_payload_bytes": "0",
            "recovery_events": "0",
            "trim_notifications": "0",
            "trim_lasthop_notifications": "0",
            "trim_recovery_events": "0",
            "stale_trim_notifications": "0",
            "terminal_outcome": "completed",
            "failure_reason": "",
            "decision_hash": "42",
            "start_time_ns": "10",
            "end_time_ns": "20",
            "timeouts": "0",
            "cnp_received": "0",
            "first_trim_ns": "0",
            "first_repair_ns": "0",
            "forgiven_bytes": "0",
            "forgiven_ranges": "0",
            "forgiven_remainder_bytes": "0",
            "pacing_refusals": "0",
            "delivered_bytes": "64",
        }

    def test_summary_accepts_dp_all_reduce_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            summary = summarize(telemetry)
            self.assertEqual(summary["shed_flow_count"], 1)
            self.assertEqual(summary["total_physical_bytes"], 64)
            self.assertEqual(summary["completion_time_ns_max"], 120)

    def test_summary_rejects_missing_terminal_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            missing_outcome = self.valid_shed_flow()
            missing_outcome.pop("terminal_outcome")
            self.write_telemetry(telemetry, missing_outcome)

            with self.assertRaisesRegex(ValueError, "terminal outcome"):
                summarize(telemetry)

    def test_summary_rejects_non_dp_shedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            invalid = self.valid_shed_flow()
            invalid["parallelism_domain"] = "tp"
            self.write_telemetry(telemetry, invalid)
            with self.assertRaisesRegex(ValueError, "outside DP All-Reduce"):
                summarize(telemetry)

    def test_summary_reports_verified_fct_latency_and_ns3_observability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flows = []
            for index, duration in enumerate((10, 20, 30, 40, 50)):
                flow = self.valid_shed_flow()
                flow["source_port"] = str(10_000 + index)
                flow["workload_node_id"] = str(index)
                flow["message_sequence"] = str(index)
                flow["start_time_ns"] = str(100 + index * 100)
                flow["end_time_ns"] = str(100 + index * 100 + duration)
                if index:
                    flow.update(
                        {
                            "flow_kind": "foreground_payload",
                            "decision": "admitted",
                            "admission_eligible": "false",
                            "transport_role": "collective_payload",
                            "logical_bytes": "1024",
                            "physical_bytes": "1024",
                            "data_attempted_bytes": "1024",
                        }
                    )
                flows.append(flow)
            self.write_telemetry(telemetry, flows, [(0, 100), (1, 120), (2, 140)])
            ns3 = root / "ns3"
            fct = self.write_fct(ns3, flows)
            (ns3 / "qlen.txt").write_text(
                "time 10 8 j 3 120 j 4 240\ntime 20 8 j 3 180\n",
                encoding="utf-8",
            )
            (ns3 / "pfc.txt").write_text(
                "10 8 1 3 3 1\n20 8 1 3 3 0\n", encoding="utf-8"
            )

            summary = summarize(telemetry, fct, ns3)

            self.assertEqual(summary["fct_join"]["status"], "verified")
            self.assertEqual(summary["fct_join"]["fct_record_count"], 5)
            self.assertEqual(summary["flow_completion_time_ns"]["all"]["p50_ns"], 30)
            self.assertEqual(summary["flow_completion_time_ns"]["all"]["p95_ns"], 50)
            self.assertEqual(summary["rank_completion_time_ns"]["p50_ns"], 120)
            self.assertEqual(summary["rank_completion_time_ns"]["max_ns"], 140)
            self.assertEqual(
                summary["ns3_observability"]["queue"]["max_queue_bytes"], 240
            )
            self.assertEqual(summary["ns3_observability"]["pfc"]["event_count"], 2)
            self.assertEqual(summary["ns3_observability"]["pfc"]["total_paused_ns"], 10)
            self.assertEqual(
                summary["ns3_observability"]["pfc"]["affected_switch_port_queues"],
                [{"switch": 8, "port": 3, "queue": 3}],
            )

    def test_network_health_normalizes_trims_and_wire_by_offered_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            background = self.valid_shed_flow()
            background.update(
                {
                    "flow_kind": "background_microburst",
                    "decision": "admitted",
                    "admission_eligible": "false",
                    "parallelism_domain": "unknown",
                    "origin_transport_role": "background_traffic",
                    "transport_role": "background_traffic",
                    "collective_type": "none",
                    "source_port": "10001",
                    "logical_bytes": "936",
                    "physical_bytes": "936",
                    "data_attempted_bytes": "936",
                    "start_time_ns": "40",
                    "end_time_ns": "540",
                }
            )
            self.write_telemetry(telemetry, [self.valid_shed_flow(), background])
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "data_arrival,data,3,1500\n"
                "trim_ftd_admission,data,2,150\n"
                "trim_ftd_lasthop_admission,data,1,50\n"
                "trim_ftd_egress_queue,data,1,1000\n",
                encoding="utf-8",
            )

            health = summarize(telemetry, ns3_dir=ns3)["network_health"]

        # 64 B of provenance plus 936 B of incast.
        self.assertEqual(health["offered_physical_bytes"], 1_000)
        self.assertEqual(health["trimmed_admission_bytes"], 200)
        self.assertAlmostEqual(health["W"], 0.2)
        self.assertAlmostEqual(health["wire_per_offered"], 1.5)
        self.assertEqual(health["burst_drain_ns"], 500)

    def test_network_health_marks_an_unconnected_arrival_trace_unmeasured(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\nqbb_drop,data,1,0\n",
                encoding="utf-8",
            )

            health = summarize(telemetry, ns3_dir=ns3)["network_health"]

        self.assertEqual(health["status"], "available")
        self.assertEqual(health["W"], 0.0)
        self.assertIsNone(health["data_arrival_bytes"])
        self.assertIsNone(health["wire_per_offered"])

    def test_network_health_without_transport_events_keeps_the_burst_drain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())

            health = summarize(telemetry)["network_health"]

        self.assertEqual(health["status"], "not_available")
        self.assertIsNone(health["burst_drain_ns"])
        self.assertNotIn("W", health)

    def test_dp_all_reduce_span_reduces_each_step_to_its_worst_collective(
        self,
    ) -> None:
        def collective(
            rank: int, step: int, node: int, start: int, end: int, domain: str = "dp"
        ) -> dict[str, str]:
            return {
                "rank": str(rank),
                "parallelism_domain": domain,
                "collective_type": "all_reduce",
                "training_step": str(step),
                "workload_node_id": str(node),
                "logical_bytes": "1024",
                "start_time_ns": str(start),
                "end_time_ns": str(end),
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(
                telemetry,
                self.valid_shed_flow(),
                collectives=[
                    collective(0, 18, 1, 100, 400),
                    collective(1, 18, 1, 150, 900),
                    collective(0, 18, 2, 100, 300),
                    collective(0, 19, 3, 1_000, 1_400),
                    collective(0, 19, 4, 1_000, 1_100, domain="tp"),
                ],
            )

            spans = summarize(telemetry)["collective_completion"][
                "dp_all_reduce_span_ns_by_training_step"
            ]

        # Step 18 keeps its worst node (800 ns), not node 2's 200 ns; the TP
        # collective at step 19 never enters the DP table.
        self.assertEqual(spans, {"18": 800, "19": 400})

    def test_host_transport_counters_reach_the_recovery_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            trimmed = self.valid_shed_flow()
            trimmed.update(
                {
                    "flow_kind": "foreground_payload",
                    "decision": "admitted",
                    "admission_eligible": "false",
                    "transport_role": "collective_payload",
                    "logical_bytes": "4096",
                    "physical_bytes": "4096",
                    "data_attempted_bytes": "8192",
                    "trimmed_payload_bytes": "1024",
                    "trim_notifications": "2",
                    "timeouts": "3",
                    "cnp_received": "5",
                    "first_trim_ns": "1000",
                    "first_repair_ns": "1600",
                }
            )
            # A repair that precedes the first trim answers a NACK, not a
            # trim, so it must not enter the distribution.
            nack_repaired = self.valid_shed_flow()
            nack_repaired.update(
                {
                    "source_port": "10002",
                    "first_trim_ns": "5000",
                    "first_repair_ns": "4000",
                }
            )
            self.write_telemetry(telemetry, [trimmed, nack_repaired])
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "rto_fired,control,3,0\n"
                "cnp_taken,control,5,0\n"
                "clipped_trim,control,2,0\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

        recovery = summary["transport_recovery"]
        self.assertEqual(recovery["timeout_count"], 3)
        self.assertEqual(recovery["cnp_received_count"], 5)
        self.assertEqual(
            recovery["first_trim_to_first_repair_ns"]["count"], 1
        )
        self.assertEqual(recovery["first_trim_to_first_repair_ns"]["p50_ns"], 600)
        transport = summary["ns3_observability"]["transport"]
        self.assertEqual(transport["rto_fired_count"], 3)
        self.assertEqual(transport["cnp_taken_count"], 5)
        self.assertEqual(transport["clipped_trim_count"], 2)

    def test_host_transport_event_must_ride_the_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\ncnp_taken,data,1,0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "control plane"):
                summarize(telemetry, ns3_dir=ns3)

    def test_a_forgiven_remainder_must_account_for_undelivered_data(
        self,
    ) -> None:
        """Its bytes were never put on the wire, so they are data-plane bytes.

        On the control plane they would be counted beside acknowledgements and
        rate cuts, where a byte count means something else entirely.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "remainder_forgiven,control,1,4096\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "undelivered data"):
                summarize(telemetry, ns3_dir=ns3)

    def test_forgiven_remainder_bytes_stay_out_of_the_trimmed_load(self) -> None:
        """W counts what the fabric trimmed, and a remainder was never sent."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "trim_ftd_admission,data,2,8192\n"
                "trim_forgiven,data,1,4096\n"
                "remainder_forgiven,data,3,12288\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

        transport = summary["ns3_observability"]["transport"]
        self.assertEqual(transport["remainder_forgiven_count"], 3)
        self.assertEqual(transport["remainder_forgiven_bytes"], 12_288)
        self.assertEqual(transport["packet_trimming"]["conversion_count"], 2)
        self.assertEqual(
            transport["packet_trimming"]["trimmed_payload_bytes"], 8_192
        )
        self.assertEqual(summary["network_health"]["forgiven_bytes"], 4_096)

    def eligible_flow(
        self,
        dst: str,
        step: str,
        logical_bytes: int,
        forgiven_bytes: int = 0,
        port: str = "10000",
    ) -> dict[str, str]:
        """An admitted DP All-Reduce payload, the population the budget covers."""
        flow = self.valid_shed_flow()
        flow.update(
            {
                "flow_kind": "foreground_payload",
                "decision": "admitted",
                "admission_eligible": "true",
                "transport_role": "collective_payload",
                "dst": dst,
                "training_step": step,
                "source_port": port,
                "logical_bytes": str(logical_bytes),
                "physical_bytes": str(logical_bytes),
                "data_attempted_bytes": str(logical_bytes),
                "forgiven_bytes": str(forgiven_bytes),
                "forgiven_ranges": "1" if forgiven_bytes else "0",
            }
        )
        return flow

    def write_recovery_manifest(
        self,
        root: Path,
        clr_steps: tuple[str, ...],
        p_low: float,
        p_high: float,
        domain: str = "recovery",
    ) -> Path:
        mask = root / "clr_mask.csv"
        mask.write_text(
            "step_id,is_clr,probability\n"
            + "".join(
                f"{step},{1 if str(step) in clr_steps else 0},"
                f"{p_low if str(step) in clr_steps else p_high}\n"
                for step in range(1, 4)
            ),
            encoding="utf-8",
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "clr_mask": str(mask),
                    "selection_policy": {
                        "semantics": {
                            "admission": "logical_admission_selection",
                            "recovery": "recovery_forgiveness",
                            "recovery_exempt": "recovery_forgiveness_cc_exempt",
                        }[domain],
                        "domain": domain,
                        "p_low": p_low,
                        "p_high": p_high,
                        "decision_scale": DECISION_SCALE,
                        "p_low_threshold": scaled_threshold(p_low),
                        "p_high_threshold": scaled_threshold(p_high),
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_ledger_law_verifies_a_budget_respecting_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            # Step 2 is non-critical, so the budget is 10% of 1000 B.
            self.write_telemetry(
                telemetry,
                [
                    self.eligible_flow("4", "2", 1_000, 100, "10001"),
                    self.eligible_flow("5", "2", 1_000, 0, "10002"),
                ],
            )
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            summary = summarize(telemetry, manifest_path=manifest)

        forgiveness = summary["forgiveness"]
        self.assertEqual(forgiveness["forgiven_bytes"], 100)
        self.assertEqual(forgiveness["forgiven_bytes_by_training_step"], {"2": 100})
        self.assertEqual(forgiveness["ledger_law"]["status"], "verified")
        self.assertEqual(forgiveness["ledger_law"]["forgiven_cell_count"], 1)

    def test_exemption_counters_reach_the_summary(self) -> None:
        """The three mechanism counters, and the law still binding the domain.

        The exemption spends no budget, so the ledger law reads exactly as it
        does in the CC-neutral domain; what the exempt arm adds is the count
        of flows granted an exemption, the congestion signals they withheld,
        the allowance reports they were sent, and the time they spent back
        under their controller when a report said the budget was gone.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            exempt = self.eligible_flow("4", "2", 1_000, 100, "10001")
            exempt.update(
                {
                    "cc_exempt": "true",
                    "cc_signal_withheld": "7",
                    "allowance_gone_reports": "1",
                    "cc_transitions": "2",
                    "cc_obeying_ns": "500",
                }
            )
            still_exempt = self.eligible_flow("5", "2", 1_000, 0, "10002")
            still_exempt.update(
                {
                    "cc_exempt": "true",
                    "cc_signal_withheld": "5",
                    "allowance_gone_reports": "0",
                    "cc_transitions": "0",
                    "cc_obeying_ns": "0",
                }
            )
            obeying = self.eligible_flow("6", "2", 1_000, 0, "10003")
            obeying.update(
                {
                    "cc_exempt": "false",
                    "cc_signal_withheld": "0",
                    "allowance_gone_reports": "0",
                    "cc_transitions": "0",
                    "cc_obeying_ns": "0",
                }
            )
            self.write_telemetry(telemetry, [exempt, still_exempt, obeying])
            manifest = self.write_recovery_manifest(
                root, ("1",), 0.005, 0.1, domain="recovery_exempt"
            )

            summary = summarize(telemetry, manifest_path=manifest)

        forgiveness = summary["forgiveness"]
        self.assertEqual(forgiveness["cc_exempt_flow_count"], 2)
        self.assertEqual(forgiveness["cc_signal_withheld_count"], 12)
        self.assertEqual(forgiveness["allowance_gone_report_count"], 1)
        self.assertEqual(forgiveness["cc_transition_count"], 2)
        self.assertEqual(forgiveness["cc_obeying_ns"], 500)
        self.assertEqual(forgiveness["cc_obeying_flow_count"], 1)
        self.assertEqual(forgiveness["ledger_law"]["status"], "verified")
        self.assertEqual(forgiveness["ledger_law"]["domain"], "recovery_exempt")

    def test_v2_receiver_counters_reach_the_summary(self) -> None:
        """The remainder split and the coin refusals, both summed by column.

        A forgiven remainder is a subset of the forgiven bytes, so the total
        does not move and the split is the only new information; a coin
        refusal is visible nowhere else, because it leaves the cap untouched.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            stopped_early = self.eligible_flow("4", "2", 1_000, 100, "10001")
            stopped_early.update(
                {"forgiven_remainder_bytes": "60", "pacing_refusals": "3"}
            )
            paced = self.eligible_flow("5", "2", 1_000, 0, "10002")
            paced.update({"pacing_refusals": "4"})
            self.write_telemetry(telemetry, [stopped_early, paced])
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            summary = summarize(telemetry, manifest_path=manifest)

        forgiveness = summary["forgiveness"]
        self.assertEqual(forgiveness["forgiven_bytes"], 100)
        self.assertEqual(forgiveness["forgiven_remainder_bytes"], 60)
        self.assertEqual(forgiveness["pacing_refusal_count"], 7)
        self.assertEqual(forgiveness["ledger_law"]["status"], "verified")

    def test_a_forgiven_remainder_covers_the_bytes_no_sender_attempted(self) -> None:
        """The step stop completes a flow whose sender was stopped early.

        The receiver takes the unsent tail as delivered, so the sender never
        attempts it and the flow still completes. The gap is the part of the
        remainder that was never on the wire, and a flow that attempted
        everything adds nothing to it.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            stopped_early = self.eligible_flow("4", "2", 1_000, 100, "10001")
            stopped_early.update(
                {
                    "data_attempted_bytes": "940",
                    "forgiven_remainder_bytes": "60",
                }
            )
            attempted_in_full = self.eligible_flow("5", "2", 1_000, 0, "10002")
            self.write_telemetry(telemetry, [stopped_early, attempted_in_full])
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            summary = summarize(telemetry, manifest_path=manifest)

        forgiveness = summary["forgiveness"]
        self.assertEqual(forgiveness["forgiven_remainder_bytes"], 60)
        self.assertEqual(forgiveness["forgiven_remainder_unsent_bytes"], 60)

    def test_a_forgiven_remainder_may_cover_bytes_already_attempted(self) -> None:
        """Most of the remainder was attempted, trimmed and awaiting repair.

        The law binds the shortfall alone, so a remainder larger than the
        shortfall is lawful and only the shortfall reaches the derived key.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            stopped_early = self.eligible_flow("4", "2", 1_000, 100, "10001")
            stopped_early.update(
                {
                    "data_attempted_bytes": "990",
                    "forgiven_remainder_bytes": "60",
                }
            )
            self.write_telemetry(telemetry, stopped_early)
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            summary = summarize(telemetry, manifest_path=manifest)

        forgiveness = summary["forgiveness"]
        self.assertEqual(forgiveness["forgiven_remainder_bytes"], 60)
        self.assertEqual(forgiveness["forgiven_remainder_unsent_bytes"], 10)

    def test_summary_rejects_a_completed_flow_whose_remainder_falls_short(
        self,
    ) -> None:
        """No forgiveness, no excuse: the sender owed those bytes."""
        for remainder in ("0", "59"):
            with self.subTest(forgiven_remainder_bytes=remainder):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    telemetry = root / "telemetry"
                    short = self.eligible_flow("4", "2", 1_000, 100, "10001")
                    short.update(
                        {
                            "data_attempted_bytes": "940",
                            "forgiven_remainder_bytes": remainder,
                        }
                    )
                    self.write_telemetry(telemetry, short)
                    manifest = self.write_recovery_manifest(
                        root, ("1",), 0.005, 0.1
                    )

                    with self.assertRaisesRegex(
                        ValueError, "attempt or be forgiven every byte"
                    ):
                        summarize(telemetry, manifest_path=manifest)

    def test_ledger_law_rejects_a_cell_over_its_budget(self) -> None:
        """A broken contract refuses the summary rather than annotating it."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            # Step 1 is critical, so the budget is 0.5% of 1000 B, and 100 B
            # forgiven is twenty times over.
            self.write_telemetry(
                telemetry, [self.eligible_flow("4", "1", 1_000, 100, "10001")]
            )
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            with self.assertRaisesRegex(
                ValueError, "broke the ledger law, status violated"
            ) as refusal:
                summarize(telemetry, manifest_path=manifest)

        message = str(refusal.exception)
        self.assertIn("'dst': '4'", message)
        self.assertIn("'training_step': '1'", message)

    def test_a_forgiving_run_without_its_mask_is_refused(self) -> None:
        """An arm whose mask cannot be read has verified nothing.

        The domain still promised every rank a share of every step, and a
        summary that cannot check the promise is not a result.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(
                telemetry, [self.eligible_flow("4", "2", 1_000, 100, "10001")]
            )
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)
            (root / "clr_mask.csv").unlink()

            with self.assertRaisesRegex(
                ValueError, "cannot verify the ledger law, status not_available"
            ):
                summarize(telemetry, manifest_path=manifest)

    def test_the_ledger_law_names_the_worst_cell_and_its_share(self) -> None:
        """Two cells at step 2, where the budget is 10% of 1000 B.

        The first cell spends its cap to the byte and leaves its rank 90% of
        what the step owed it, the second spends nothing, and a third rank is
        owed no bytes at all, so it has no share to report and no division to
        perform. The reported share is the smallest of the two the law
        measures, and the worst cell is the one that set it.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(
                telemetry,
                [
                    self.eligible_flow("4", "2", 1_000, 100, "10001"),
                    self.eligible_flow("5", "2", 1_000, 0, "10002"),
                    self.eligible_flow("6", "2", 0, 0, "10003"),
                ],
            )
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            law = summarize(telemetry, manifest_path=manifest)["forgiveness"][
                "ledger_law"
            ]

        self.assertEqual(law["status"], "verified")
        self.assertEqual(law["cell_count"], 3)
        self.assertEqual(law["min_delivered_share"], 0.9)
        self.assertEqual(
            law["worst_cell"],
            {
                "dst": "4",
                "training_step": "2",
                "eligible_bytes": 1_000,
                "shed_bytes": 0,
                "forgiven_bytes": 100,
            },
        )

    def test_ledger_law_uses_the_integer_law_the_simulator_used(self) -> None:
        """Two cells the float law misjudges, one in each direction.

        Step 1 is critical at p_low 5e-7, which scales to threshold 1, so its
        budget is one byte per million eligible: 1 B forgiven of 1000000 B
        eligible is exactly on the boundary and lawful, while the float
        product 0.5 B calls it a violation. Step 2 is non-critical at p_high
        1.2e-6, which also scales to 1, so 11 B forgiven of 10000000 B
        eligible is one byte over the integer budget of 10 while the float
        product 12 B calls it lawful.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(
                telemetry,
                [self.eligible_flow("4", "1", 1_000_000, 1, "10001")],
            )
            manifest = self.write_recovery_manifest(root, ("1",), 5e-7, 1.2e-6)

            law = summarize(telemetry, manifest_path=manifest)["forgiveness"][
                "ledger_law"
            ]

        self.assertEqual(law["status"], "verified")
        self.assertEqual(law["decision_scale"], DECISION_SCALE)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(
                telemetry,
                [self.eligible_flow("4", "2", 10_000_000, 11, "10001")],
            )
            manifest = self.write_recovery_manifest(root, ("1",), 5e-7, 1.2e-6)

            with self.assertRaisesRegex(
                ValueError, "broke the ledger law, status violated"
            ) as refusal:
                summarize(telemetry, manifest_path=manifest)

        self.assertIn("'threshold': 1", str(refusal.exception))

    def test_ledger_law_does_not_apply_to_the_admission_domain(self) -> None:
        """Admission shedding is a per-flow hash draw, not a per-cell cap.

        A small cell can exceed the rate by chance, so reporting the budget
        law there would call a sound arm violated.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            # Twenty times over the critical-step rate, and lawful anyway.
            self.write_telemetry(
                telemetry, [self.eligible_flow("4", "1", 1_000, 100, "10001")]
            )
            manifest = self.write_recovery_manifest(
                root, ("1",), 0.005, 0.1, domain="admission"
            )

            law = summarize(telemetry, manifest_path=manifest)["forgiveness"][
                "ledger_law"
            ]

        self.assertEqual(law["status"], "not_applicable")
        self.assertEqual(law["domain"], "admission")

    def test_ledger_law_rejects_a_recovery_arm_that_shed_at_admission(self) -> None:
        """The recovery arm spends its budget after the trim, never before."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            # A shed row is a substituted payload, so it is the provenance
            # control flow the policy put on the wire in its place.
            shed = self.valid_shed_flow()
            shed.update({"dst": "4", "training_step": "2", "source_port": "10001"})
            self.write_telemetry(telemetry, [shed])
            manifest = self.write_recovery_manifest(root, ("1",), 0.005, 0.1)

            with self.assertRaisesRegex(
                ValueError, "broke the ledger law, status violated"
            ) as refusal:
                summarize(telemetry, manifest_path=manifest)

        self.assertIn("'shed_bytes': 1048576", str(refusal.exception))

    def test_ledger_law_is_unavailable_without_the_mask_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(
                telemetry, [self.eligible_flow("4", "2", 1_000, 100, "10001")]
            )

            law = summarize(telemetry)["forgiveness"]["ledger_law"]

        self.assertEqual(law["status"], "not_available")

    def test_summary_reports_data_loss_without_control_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "data_arrival,data,1,1024\n"
                "data_injected_drop,data,1,1024\n"
                "control_arrival,control,1,60\n"
                "control_deliver,control,1,60\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

            transport = summary["ns3_observability"]["transport"]
            self.assertEqual(transport["status"], "available")
            self.assertEqual(transport["data_injected_drop_count"], 1)
            self.assertEqual(transport["control_injected_drop_count"], 0)
            self.assertEqual(transport["plane_event_counts"], {"data": 2, "control": 2})

    def test_summary_distinguishes_natural_data_and_control_buffer_drops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "switch_admission_drop,data,1,1024\n"
                "switch_egress_queue_drop,data,1,1024\n"
                "switch_egress_queue_drop,control,1,60\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

            transport = summary["ns3_observability"]["transport"]
            self.assertEqual(transport["data_natural_buffer_drop_count"], 2)
            self.assertEqual(transport["control_natural_buffer_drop_count"], 1)
            self.assertEqual(transport["data_injected_drop_count"], 0)

    def test_summary_reconciles_ftd_trim_with_required_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            trimmed = self.valid_shed_flow()
            trimmed.update(
                {
                    "flow_kind": "foreground_payload",
                    "decision": "admitted",
                    "admission_eligible": "false",
                    "transport_role": "collective_payload",
                    "logical_bytes": "1024",
                    "physical_bytes": "1024",
                    "data_attempted_bytes": "2048",
                    "retransmitted_bytes": "1024",
                    "trimmed_payload_bytes": "1000",
                    "trim_notifications": "1",
                    "trim_recovery_events": "1",
                }
            )
            self.write_telemetry(telemetry, trimmed)
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "trim_ftd_admission,data,1,1000\n"
                "control_arrival,control,1,60\n"
                "control_deliver,control,1,60\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

            trim = summary["ns3_observability"]["transport"]["packet_trimming"]
            self.assertEqual(trim["conversion_count"], 1)
            self.assertEqual(trim["trimmed_payload_bytes"], 1000)
            self.assertEqual(trim["ftd_conversion_count"], 1)
            self.assertEqual(trim["lasthop_conversion_count"], 0)
            self.assertEqual(trim["trimmed_queue_drop_count"], 0)
            self.assertEqual(
                summary["transport_recovery"]["trim_notification_count"], 1
            )

    def test_summary_counts_lasthop_trims_and_trimmed_queue_drops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            trimmed = self.valid_shed_flow()
            trimmed.update(
                {
                    "flow_kind": "foreground_payload",
                    "decision": "admitted",
                    "admission_eligible": "false",
                    "transport_role": "collective_payload",
                    "logical_bytes": "1024",
                    "physical_bytes": "1024",
                    "data_attempted_bytes": "2048",
                    "retransmitted_bytes": "1024",
                    "trimmed_payload_bytes": "1000",
                    "trim_notifications": "1",
                    "trim_lasthop_notifications": "1",
                    "trim_recovery_events": "1",
                }
            )
            self.write_telemetry(telemetry, trimmed)
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "trim_ftd_lasthop_admission,data,1,1000\n"
                "switch_trimmed_queue_drop,data,1,60\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

            trim = summary["ns3_observability"]["transport"]["packet_trimming"]
            self.assertEqual(trim["conversion_count"], 1)
            self.assertEqual(trim["ftd_conversion_count"], 1)
            self.assertEqual(trim["lasthop_conversion_count"], 1)
            self.assertEqual(trim["admission_conversion_count"], 1)
            self.assertEqual(trim["trimmed_queue_drop_count"], 1)
            self.assertEqual(
                summary["transport_recovery"]["trim_lasthop_notification_count"], 1
            )

    def test_summary_counts_drops_on_a_lossy_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "link_error_drop,control,3,144\n"
                "link_error_drop,data,2,8292\n",
                encoding="utf-8",
            )

            transport = summarize(telemetry, ns3_dir=ns3)["ns3_observability"][
                "transport"
            ]

            self.assertEqual(transport["event_counts"]["link_error_drop"], 5)
            self.assertEqual(transport["event_bytes"]["link_error_drop"], 8436)
            self.assertEqual(transport["plane_event_counts"], {"data": 2, "control": 3})

    def test_summary_requires_exact_expected_rank_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(
                telemetry, self.valid_shed_flow(), [(0, 100), (1, 120)]
            )

            summary = summarize(telemetry, expected_rank_count=2)

            self.assertEqual(
                summary["rank_completion_status"],
                {
                    "status": "verified",
                    "recorded_rank_count": 2,
                    "expected_rank_count": 2,
                },
            )
            self.assertEqual(
                summary["primary_analysis_eligibility"]["status"], "ineligible"
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(
                telemetry, self.valid_shed_flow(), [(0, 100), (2, 120)]
            )
            with self.assertRaisesRegex(
                ValueError, "does not cover every expected rank"
            ):
                summarize(telemetry, expected_rank_count=2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(
                telemetry, self.valid_shed_flow(), [(0, 100), (0, 120)]
            )
            with self.assertRaisesRegex(ValueError, "duplicate rank"):
                summarize(telemetry, expected_rank_count=2)

    def test_summary_rejects_control_configured_loss_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "transport_summary.csv").write_text(
                "event,plane,event_count,total_bytes\n"
                "data_injected_drop,control,1,60\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "configured data impairment"):
                summarize(telemetry, ns3_dir=ns3)

    def test_summary_reports_explicit_failed_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            failed = self.valid_shed_flow()
            failed.update(
                {
                    "flow_kind": "foreground_payload",
                    "decision": "admitted",
                    "admission_eligible": "false",
                    "transport_role": "collective_payload",
                    "logical_bytes": "1024",
                    "physical_bytes": "1024",
                    "data_attempted_bytes": "4096",
                    "retransmitted_bytes": "3072",
                    "recovery_events": "3",
                    "terminal_outcome": "failed",
                    "failure_reason": "retry_exhausted",
                }
            )
            self.write_telemetry(telemetry, failed, completions=[])

            summary = summarize(telemetry)

            self.assertEqual(summary["completed_flow_count"], 0)
            self.assertEqual(summary["failed_flow_count"], 1)
            self.assertEqual(
                summary["transport_recovery"],
                {
                    "data_attempted_bytes": 4096,
                    "retransmitted_bytes": 3072,
                    "recovery_event_count": 3,
                    "failed_by_reason": {"retry_exhausted": 1},
                    "trimmed_payload_bytes": 0,
                    "trim_notification_count": 0,
                    "trim_lasthop_notification_count": 0,
                    "trim_recovery_event_count": 0,
                    "stale_trim_notification_count": 0,
                    "timeout_count": 0,
                    "cnp_received_count": 0,
                    "first_trim_to_first_repair_ns": {
                        "count": 0,
                        "min_ns": None,
                        "p50_ns": None,
                        "p95_ns": None,
                        "p99_ns": None,
                        "max_ns": None,
                    },
                },
            )

    def test_summary_uses_native_collective_events_for_logical_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            collectives = [
                {
                    "rank": "0",
                    "parallelism_domain": "dp",
                    "collective_type": "all_reduce",
                    "training_step": "2",
                    "workload_node_id": "7",
                    "logical_bytes": "1048576",
                    "start_time_ns": "100",
                    "end_time_ns": "160",
                },
                {
                    "rank": "1",
                    "parallelism_domain": "dp",
                    "collective_type": "all_reduce",
                    "training_step": "2",
                    "workload_node_id": "7",
                    "logical_bytes": "1048576",
                    "start_time_ns": "110",
                    "end_time_ns": "180",
                },
            ]
            self.write_telemetry(
                telemetry, self.valid_shed_flow(), collectives=collectives
            )

            summary = summarize(telemetry)

            collective = summary["collective_completion"]
            self.assertEqual(collective["status"], "available")
            self.assertEqual(collective["rank_event_count"], 2)
            self.assertEqual(collective["logical_collective_count"], 1)
            per_rank = collective["per_rank_completion_time_ns"]
            self.assertEqual(per_rank["all"]["p50_ns"], 60)
            operation_span = collective["all_rank_operation_span_ns"]
            self.assertEqual(operation_span["all"]["p99_ns"], 80)
            self.assertEqual(operation_span["by_training_step"]["2"]["max_ns"], 80)

    def test_summary_rejects_duplicate_or_reversed_collective_events(self) -> None:
        collective = {
            "rank": "0",
            "parallelism_domain": "dp",
            "collective_type": "all_reduce",
            "training_step": "2",
            "workload_node_id": "7",
            "logical_bytes": "1048576",
            "start_time_ns": "100",
            "end_time_ns": "160",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "duplicate"
            self.write_telemetry(
                telemetry,
                self.valid_shed_flow(),
                collectives=[collective, dict(collective)],
            )
            with self.assertRaisesRegex(ValueError, "duplicate rank completion"):
                summarize(telemetry)

        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "reversed"
            reversed_collective = dict(
                collective, start_time_ns="161", end_time_ns="160"
            )
            self.write_telemetry(
                telemetry, self.valid_shed_flow(), collectives=[reversed_collective]
            )
            with self.assertRaisesRegex(ValueError, "must not be negative"):
                summarize(telemetry)

    def test_summary_labels_queue_unattributed_pfc_durations_as_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            ns3 = root / "ns3"
            ns3.mkdir()
            (ns3 / "pfc.txt").write_text(
                "10 8 1 3 1\n12 8 1 3 1\n20 8 1 3 0\n24 8 1 3 0\n",
                encoding="utf-8",
            )

            summary = summarize(telemetry, ns3_dir=ns3)

            pfc = summary["ns3_observability"]["pfc"]
            self.assertEqual(pfc["queue_identity_status"], "not_available")
            self.assertEqual(
                pfc["pause_duration_status"], "estimated_without_queue_identity"
            )
            self.assertEqual(pfc["completed_pause_interval_count"], 2)

    def test_summary_rejects_fct_physical_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flow = self.valid_shed_flow()
            self.write_telemetry(telemetry, flow)
            fct = self.write_fct(root / "ns3", [flow])
            fct.write_text(
                fct.read_text(encoding="utf-8").replace(" 64 10 10 10", " 65 10 10 10"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "physical-byte mismatch"):
                summarize(telemetry, fct)

    def write_manifest(
        self,
        directory: Path,
        loss: bool,
        recovery: bool,
        trimming: bool,
        pfc: bool = True,
    ) -> Path:
        path = directory / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "data_plane_loss": {"enabled": loss},
                    "transport_recovery": {"enabled": recovery},
                    "packet_trimming": {"enabled": trimming},
                    "fabric": {"pfc_enabled": pfc},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_summary_verifies_lossless_run_recorded_no_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            self.write_telemetry(telemetry, self.valid_shed_flow())
            manifest = self.write_manifest(root, False, False, False)

            summary = summarize(telemetry, manifest_path=manifest)

            self.assertEqual(summary["lossless_transport"]["status"], "verified")

    def test_summary_rejects_recovery_without_a_loss_mechanism(self) -> None:
        # A stray packet reaching a reused source port surfaces exactly here.
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flow = self.valid_shed_flow()
            flow["recovery_events"] = "1"
            self.write_telemetry(telemetry, flow)
            manifest = self.write_manifest(root, False, False, False)

            with self.assertRaisesRegex(ValueError, "models no loss mechanism"):
                summarize(telemetry, manifest_path=manifest)

    def test_summary_allows_recovery_when_the_run_models_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flow = self.valid_shed_flow()
            flow["recovery_events"] = "2"
            flow["retransmitted_bytes"] = "64"
            self.write_telemetry(telemetry, flow)
            manifest = self.write_manifest(root, False, True, False)

            summary = summarize(telemetry, manifest_path=manifest)

            self.assertEqual(summary["lossless_transport"]["status"], "not_applicable")

    def test_summary_allows_recovery_on_a_best_effort_fabric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flow = self.valid_shed_flow()
            flow["recovery_events"] = "1"
            self.write_telemetry(telemetry, flow)
            manifest = self.write_manifest(root, False, False, False, pfc=False)

            summary = summarize(telemetry, manifest_path=manifest)

            self.assertEqual(summary["lossless_transport"]["status"], "not_applicable")

    def test_summary_skips_the_lossless_check_without_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flow = self.valid_shed_flow()
            flow["recovery_events"] = "3"
            self.write_telemetry(telemetry, flow)

            summary = summarize(telemetry)

            self.assertEqual(summary["lossless_transport"]["status"], "not_applicable")

    def test_summary_accepts_telemetry_with_no_flows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            self.write_telemetry(telemetry, [])

            summary = summarize(telemetry)

            self.assertEqual(summary["flow_count"], 0)
            self.assertIsNone(summary["flow_completion_time_ns"]["all"]["p50_ns"])
            self.assertEqual(summary["physical_traffic_bytes"]["total"]["flow_count"], 0)
            self.assertEqual(
                summary["background_microburst_timeline"]["status"],
                "no_background_microburst",
            )

    def test_summary_spans_every_background_microburst_flow(self) -> None:
        # The timeline is the join of each burst flow's interval, so it must
        # widen to the earliest start and the latest end, not the last row's.
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "telemetry"
            flows = []
            for index, (start, end) in enumerate(((700, 900), (300, 500), (600, 1200))):
                flow = self.valid_shed_flow()
                flow.update(
                    {
                        "flow_kind": "background_microburst",
                        "decision": "admitted",
                        "admission_eligible": "false",
                        "transport_role": "background_traffic",
                        "message_sequence": str(index),
                        "source_port": str(20000 + index),
                        "logical_bytes": "2048",
                        "physical_bytes": "2048",
                        "data_attempted_bytes": "2048",
                        "start_time_ns": str(start),
                        "end_time_ns": str(end),
                    }
                )
                flows.append(flow)
            self.write_telemetry(telemetry, flows)

            timeline = summarize(telemetry)["background_microburst_timeline"]

            self.assertEqual(timeline["status"], "available")
            self.assertEqual(timeline["flow_count"], 3)
            self.assertEqual(timeline["start_time_ns"], 300)
            self.assertEqual(timeline["end_time_ns"], 1200)
            self.assertEqual(timeline["span_ns"], 900)
            self.assertEqual(timeline["physical_bytes"], 6144)

    def test_summary_joins_reused_source_ports(self) -> None:
        # The ns-3 bridge returns a source port to its host pair once the
        # queue pair terminates, so one port names several flows in a run.
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry = root / "telemetry"
            flows = []
            for index, start in enumerate((100, 400, 700)):
                flow = self.valid_shed_flow()
                flow["source_port"] = "10000"
                flow["message_sequence"] = str(index)
                flow["start_time_ns"] = str(start)
                flow["end_time_ns"] = str(start + 10 * (index + 1))
                flows.append(flow)
            self.write_telemetry(telemetry, flows)
            fct = self.write_fct(root / "ns3", flows)

            summary = summarize(telemetry, fct)

            self.assertEqual(summary["fct_join"]["status"], "verified")
            self.assertEqual(summary["fct_join"]["fct_record_count"], 3)
            self.assertEqual(summary["fct_join"]["telemetry_flow_count"], 3)


if __name__ == "__main__":
    unittest.main()
