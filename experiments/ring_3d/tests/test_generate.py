from __future__ import annotations

import csv
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.third_party.utils.protolib import decodeMessage

from experiments.ring_3d.generate import (
    DECISION_SCALE,
    REPOSITORY_ROOT,
    coordinates_for,
    generate_groups,
    load_profile,
    materialize,
    rank_for,
    scaled_threshold,
)
from experiments.ring_3d.topology import build_topology


class Ring3DGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/smoke_8.json"
        )
        self.profile = load_profile(self.profile_path)

    def test_rank_mapping_is_bijective(self) -> None:
        ranks = set()
        for dp_rank in range(self.profile.dp):
            for pp_rank in range(self.profile.pp):
                for tp_rank in range(self.profile.tp):
                    rank = rank_for(tp_rank, pp_rank, dp_rank, self.profile)
                    ranks.add(rank)
                    self.assertEqual(
                        coordinates_for(rank, self.profile),
                        (tp_rank, pp_rank, dp_rank),
                    )
        self.assertEqual(ranks, set(range(self.profile.ranks)))

    def test_groups_have_expected_membership(self) -> None:
        groups, tp_groups, pp_groups, dp_groups = generate_groups(self.profile)
        self.assertEqual(len(groups), 12)
        for rank in range(self.profile.ranks):
            self.assertEqual(len(groups[str(tp_groups[str(rank)])]), self.profile.tp)
            self.assertEqual(len(groups[str(pp_groups[str(rank)])]), self.profile.pp)
            self.assertEqual(len(groups[str(dp_groups[str(rank)])]), self.profile.dp)

    def test_materialized_inputs_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(self.profile_path, output)

            self.assertEqual(len(list((output / "workload").glob("ring_3d.*.et"))), 8)
            topology_lines = (
                (output / "topology.txt").read_text(encoding="utf-8").splitlines()
            )
            node_count, switch_count, edge_count = map(int, topology_lines[0].split())
            self.assertEqual((node_count, switch_count, edge_count), (16, 8, 24))
            self.assertEqual(len(topology_lines), edge_count + 2)
            self.assertEqual(
                (output / "ns3/flow.txt").read_text(encoding="utf-8"), "0\n"
            )
            self.assertEqual(
                (output / "ns3/trace.txt").read_text(encoding="utf-8"), "0\n"
            )
            self.assertEqual(
                json.loads((output / "profile.json").read_text(encoding="utf-8")),
                json.loads(self.profile_path.read_text(encoding="utf-8")),
            )

            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(policy["eligibility"], "dp_all_reduce_only")
            self.assertEqual(
                policy["selection_probability_by_step"],
                {"1": 0.005, "2": 0.1, "3": 0.1},
            )
            self.assertEqual(
                policy["selection_policy"],
                {
                    "semantics": "logical_admission_selection",
                    "p_low": 0.005,
                    "p_high": 0.1,
                    "p_low_threshold": 5_000,
                    "p_high_threshold": 100_000,
                },
            )
            with (output / "clr_mask.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    list(csv.DictReader(handle)),
                    [
                        {"step_id": "1", "is_clr": "1", "probability": "1"},
                        {
                            "step_id": "2",
                            "is_clr": "0",
                            "probability": "0.22986810714751529",
                        },
                        {
                            "step_id": "3",
                            "is_clr": "0",
                            "probability": "0.099574136735727889",
                        },
                    ],
                )
            self.assertEqual(policy["provenance"]["priority_group"], 1)
            self.assertEqual(manifest["ranks"], 8)
            self.assertEqual(Path(manifest["profile_config"]), output / "profile.json")
            self.assertEqual(Path(manifest["clr_mask"]), output / "clr_mask.csv")
            self.assertEqual(manifest["clr_schedule"]["clr_step_count"], 1)
            self.assertIn(
                "ACK_HIGH_PRIO 1",
                (output / "network_config.txt").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "QLEN_MON_INTERVAL 10000",
                (output / "network_config.txt").read_text(encoding="utf-8"),
            )

    def test_llama3_profile_materializes_production_event_window_and_incast(
        self,
    ) -> None:
        profile_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/llama3_70b_16.json"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(profile_path, output)

            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            flows = policy["microburst"]["flows"]
            self.assertTrue(policy["microburst"]["enabled"])
            # The burst probes the converged tail of the CLR schedule instead
            # of inheriting the hardcoded step-2 trigger.
            self.assertEqual(policy["microburst"]["trigger_step"], 18)
            self.assertEqual(len(flows), 7)
            self.assertEqual({flow["dst"] for flow in flows}, {8})
            self.assertEqual({flow["src"] for flow in flows}, set(range(7)))
            self.assertEqual(
                {flow["size_bytes"] for flow in flows}, {128 * 1024 * 1024}
            )
            self.assertEqual({flow["offset_ns"] for flow in flows}, {0})
            self.assertIn(
                "PACKET_PAYLOAD_SIZE 4096",
                (output / "network_config.txt").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "QLEN_MON_START 0",
                (output / "network_config.txt").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "QLEN_MON_INTERVAL 10000",
                (output / "network_config.txt").read_text(encoding="utf-8"),
            )

    def test_direct_dp_profile_overrides_only_the_dp_groups(self) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/llama3_70b_32_direct.json"
        )
        profile = load_profile(profile_path)
        self.assertEqual(profile.dp_all_reduce_implementation, "direct2")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)

            system = json.loads((output / "system.json").read_text("utf-8"))
            groups = json.loads(
                (output / "communicator_groups.json").read_text("utf-8")
            )
            overrides = system["all-reduce-implementation-per-group"]
            # Global native algorithms are untouched; only DP groups differ.
            self.assertEqual(system["all-reduce-implementation"], ["ring"])
            self.assertEqual(set(overrides.values()), {"direct2"})
            # Every overridden group is exactly one DP group: dp-many members,
            # one per tensor-parallel position, spanning distinct leaves.
            self.assertEqual(len(overrides), profile.tp * profile.pp)
            for group_id in overrides:
                members = groups[group_id]
                self.assertEqual(len(members), profile.dp)
                self.assertEqual(
                    len({rank % profile.tp for rank in members}), 1
                )
            self.assertEqual(
                manifest["collective_implementations"],
                {
                    "default_all_reduce": "ring",
                    "dp_all_reduce": "direct2",
                    "dp_fan_in": 2,
                },
            )

    def test_ring_dp_profile_writes_no_group_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(self.profile_path, output)
            system = json.loads((output / "system.json").read_text("utf-8"))
            self.assertNotIn("all-reduce-implementation-per-group", system)
            self.assertEqual(
                manifest["collective_implementations"],
                {
                    "default_all_reduce": "ring",
                    "dp_all_reduce": "ring",
                    "dp_fan_in": 1,
                },
            )

    def test_dp_all_reduce_implementation_accepts_windowed_direct(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["dp_all_reduce_implementation"] = "direct4"
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            profile = load_profile(profile_path)
            self.assertEqual(profile.dp_all_reduce_implementation, "direct4")

    def test_dp_all_reduce_implementation_rejects_unknown_algorithms(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        for invalid in ("halvingDoubling", "direct0", "Direct", ""):
            document["dp_all_reduce_implementation"] = invalid
            with tempfile.TemporaryDirectory() as temporary_directory:
                profile_path = Path(temporary_directory) / "profile.json"
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "dp_all_reduce_implementation"
                ):
                    load_profile(profile_path)

    def test_microburst_trigger_step_override_reaches_the_policy(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["microburst_trigger_step"] = 3
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "experiment"
            materialize(profile_path, output)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(policy["microburst"]["trigger_step"], 3)

    def test_profile_name_does_not_change_the_selection_inputs(self) -> None:
        """Two profiles differing only by name must stay a matched pair.

        ``run_id`` is provenance; the simulator draws its selection stream
        from the remaining fields, so renaming a profile must move nothing
        that ``evaluate_shedding`` reads.
        """
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        renamed = dict(document, name=f"{document['name']}_renamed")
        policies = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, variant in enumerate((document, renamed)):
                profile_path = root / f"profile_{index}.json"
                profile_path.write_text(json.dumps(variant), encoding="utf-8")
                output = root / f"experiment_{index}"
                materialize(profile_path, output)
                policies.append(
                    json.loads(
                        (output / "experiment.json").read_text(encoding="utf-8")
                    )
                )

        self.assertEqual(policies[0]["run_id"], document["name"])
        self.assertEqual(policies[1]["run_id"], renamed["name"])
        for policy in policies:
            del policy["run_id"]
        self.assertEqual(policies[0], policies[1])

    def test_congestion_control_defaults_to_no_sender_reaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(self.profile_path, output)
            config = Path(manifest["network_config"]).read_text(encoding="utf-8")

        self.assertEqual(manifest["congestion_control"]["mode"], "none")
        self.assertEqual(manifest["congestion_control"]["cc_mode"], 12)
        self.assertIn("CC_MODE 12\n", config)
        # smoke_8 runs 200 Gb/s links, so the HPCC 100 G literals scale x2.
        self.assertIn("RATE_AI 100000000bps\n", config)
        self.assertIn("RATE_HAI 200000000bps\n", config)
        self.assertIn("MIN_RATE 200000000bps\n", config)

    def test_congestion_control_dcqcn_writes_mode_one_and_scaled_rates(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["congestion_control"] = {
            "mode": "dcqcn",
            "rate_ai_fraction": 0.001,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)
            config = Path(manifest["network_config"]).read_text(encoding="utf-8")

        self.assertEqual(manifest["congestion_control"]["cc_mode"], 1)
        self.assertEqual(manifest["congestion_control"]["rate_ai_bps"], 200_000_000)
        self.assertIn("CC_MODE 1\n", config)
        self.assertIn("RATE_AI 200000000bps\n", config)

    def test_congestion_control_rejects_unknown_keys_and_modes(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["congestion_control"] = {"mode": "dcqcn", "gain": 0.5}
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "congestion_control"):
                load_profile(profile_path)

            document["network"]["congestion_control"] = {"mode": "swift"}
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "congestion_control.mode"):
                load_profile(profile_path)

            document["network"]["congestion_control"] = {
                "mode": "dcqcn",
                "min_rate_fraction": 0,
            }
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "min_rate_fraction"):
                load_profile(profile_path)

    def test_congestion_control_nscc_writes_mode_eleven(self) -> None:
        document = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
            ).read_text(encoding="utf-8")
        )
        document["network"]["congestion_control"] = {"mode": "nscc"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for mode in ("ev_hash", "spray_uniform"):
                document["network"]["load_balancing"] = {"mode": mode}
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / mode
                manifest = materialize(profile_path, output)
                config = (output / "network_config.txt").read_text(encoding="utf-8")
                with self.subTest(mode=mode):
                    self.assertEqual(manifest["congestion_control"]["mode"], "nscc")
                    self.assertEqual(manifest["congestion_control"]["cc_mode"], 11)
                    self.assertIn("\nCC_MODE 11\n", config)
                    self.assertIn(f"\nLOAD_BALANCING {mode}\n", config)
                    self.assertIn("\nHAS_WIN 1\n", config)

    def test_congestion_control_nscc_refuses_ecmp(self) -> None:
        """NSCC needs an acknowledgement per packet, which ECMP does not send."""
        ring_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/model_100b_256_ring.json"
        )
        clos_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
        )
        absent = json.loads(clos_path.read_text(encoding="utf-8"))
        explicit = json.loads(json.dumps(absent))
        explicit["network"]["load_balancing"] = {"mode": "ecmp"}
        ring = json.loads(ring_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for name, document in (
                ("clos without load_balancing", absent),
                ("clos with ecmp", explicit),
                ("ring", ring),
            ):
                document["network"]["congestion_control"] = {"mode": "nscc"}
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(network=name):
                    with self.assertRaisesRegex(
                        ValueError, "'nscc' requires network.load_balancing.mode"
                    ):
                        load_profile(profile_path)

    def test_congestion_control_bounds_each_fraction_at_one(self) -> None:
        """A fraction of the link rate: exactly the link rate is the ceiling."""
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for field in (
                "rate_ai_fraction",
                "rate_hai_fraction",
                "min_rate_fraction",
            ):
                document["network"]["congestion_control"] = {
                    "mode": "dcqcn",
                    "rate_ai_fraction": 1.0,
                    "rate_hai_fraction": 1.0,
                    "min_rate_fraction": 1.0,
                }
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                profile = load_profile(profile_path)
                self.assertEqual(
                    profile.network.congestion_control.rates_bps(
                        profile.network.link_rate
                    ),
                    (200_000_000_000, 200_000_000_000, 200_000_000_000),
                )

                document["network"]["congestion_control"][field] = 1.0000001
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        load_profile(profile_path)

    def test_congestion_control_refuses_a_backwards_increase(self) -> None:
        """Hyper-additive increase is what additive increase escalates into.

        Both fractions are individually in range, so only their order says the
        recovery would slow down the longer it went unchallenged.
        """
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["congestion_control"] = {
            "mode": "dcqcn",
            "rate_ai_fraction": 0.002,
            "rate_hai_fraction": 0.001,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rate_ai_fraction"):
                load_profile(profile_path)

            # Equal is allowed: the defaults already run hai at min_rate.
            document["network"]["congestion_control"]["rate_ai_fraction"] = 0.001
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            profile = load_profile(profile_path)

        self.assertEqual(
            profile.network.congestion_control.rate_ai_fraction,
            profile.network.congestion_control.rate_hai_fraction,
        )

    def test_ecn_threshold_scale_moves_thresholds_and_no_rate_key(self) -> None:
        """The default writes today's literals; a scale moves KMIN and KMAX only."""
        rates = ("25000000000", "40000000000", "100000000000",
                 "200000000000", "400000000000", "2400000000000")
        kmax = (400, 800, 1600, 2400, 3200, 3200)
        kmin = (100, 200, 400, 600, 800, 800)
        pmax = "PMAX_MAP 6 " + " ".join(f"{rate} 0.2" for rate in rates) + "\n"

        def ecn_map(name: str, values: tuple[float, ...]) -> str:
            pairs = " ".join(f"{r} {round(v)}" for r, v in zip(rates, values))
            return f"{name} 6 {pairs}\n"

        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for scale in (None, 2.0, 0.5):
                control = {"mode": "dcqcn"}
                if scale is not None:
                    control["ecn_threshold_scale"] = scale
                document["network"]["congestion_control"] = control
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / f"experiment_{scale}"
                manifest = materialize(profile_path, output)
                config = Path(manifest["network_config"]).read_text(
                    encoding="utf-8"
                )
                factor = 1.0 if scale is None else scale
                with self.subTest(scale=scale):
                    self.assertEqual(
                        manifest["congestion_control"]["ecn_threshold_scale"],
                        factor,
                    )
                    self.assertIn(
                        ecn_map("KMAX_MAP", tuple(v * factor for v in kmax))
                        + ecn_map("KMIN_MAP", tuple(v * factor for v in kmin))
                        + pmax,
                        config,
                    )

            for scale in (0, -1.0, 0.001, True, "2"):
                document["network"]["congestion_control"] = {
                    "mode": "dcqcn",
                    "ecn_threshold_scale": scale,
                }
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(rejected=scale):
                    with self.assertRaisesRegex(ValueError, "ecn_threshold_scale"):
                        load_profile(profile_path)

    def test_only_new_families_turn_congestion_control_on(self) -> None:
        """Every arm measured to date ran with no sender reaction at all.

        The regime map sweeps it, the forgiveness DCQCN fixture needs it to
        have rate cuts to be neutral about, and the congestion-exempt family
        is refused without it. Anything else turning it on would silently
        rewrite what the earlier waves measured.
        """
        profiles = sorted(
            (REPOSITORY_ROOT / "experiments/ring_3d/profiles").glob("*.json")
        )
        self.assertTrue(profiles)
        swept = set()
        for path in profiles:
            network = json.loads(path.read_text(encoding="utf-8"))["network"]
            mode = network.get("congestion_control", {"mode": "none"})["mode"]
            if mode != "none":
                swept.add(path.name)
                continue
            with self.subTest(profile=path.name):
                self.assertEqual(mode, "none")
        self.assertEqual(
            swept,
            {"forgiveness_dcqcn_8.json", "exempt_smoke_8.json"}
            | {
                f"regime_64_dcqcn_{algorithm}_{ratio}.json"
                for algorithm in ("direct2", "direct7")
                for ratio in ("2to1", "4to1")
            }
            | {
                "regime_64_dcqcn_direct7_4to1_exempt.json",
                "regime_64_dcqcn_direct2_2to1_exempt.json",
            }
            | {
                f"regime_64_dcqcn_direct7_4to1_exempt_{tag}.json"
                for tag in ("p005", "p01", "p02", "p06", "allsteps")
            }
            # The FORGIVE v2 receiver policies, each the p01 cell with one
            # knob moved and congestion control untouched.
            | {
                f"regime_64_dcqcn_direct7_4to1_exempt_p01_{tag}.json"
                for tag in ("b50", "b25", "b10", "b05", "owed",
                            "owed_b25", "stepstop", "b25_stepstop",
                            "noreengage")
            }
            # Round 2 pairs the coin with a budget other than 0.1.
            | {
                f"regime_64_dcqcn_direct7_4to1_exempt_{tag}_b25.json"
                for tag in ("p005", "p02")
            }
            # The zero-tolerance and congestion-neutral references keep their
            # cell's fabric and move the selection policy alone, so a join
            # against that cell's published arms holds.
            | {
                "regime_64_dcqcn_direct7_4to1_zero.json",
                "regime_64_dcqcn_direct2_2to1_zero.json",
                "regime_64_dcqcn_direct7_4to1_recovery_p01.json",
                # The forgiveness-only ablation row: recovery under the
                # whole-step cap, the controller obeyed.
                "regime_64_dcqcn_direct7_4to1_recovery_p01_owed.json",
            }
            # The 63-source incast on the 4:1 cell, each the partner of its
            # no-burst 4:1 profile with the burst alone moved.
            | {
                "regime_64_dcqcn_direct7_4to1_zero_burst63.json",
                "regime_64_dcqcn_direct7_4to1_exempt_p01_burst63.json",
            }
            # The healthy 1:1 cell, which differs from the 4:1 cell in the
            # spine count alone, so congestion control must stay on for the
            # oversubscription to be the only thing that moved.
            | {
                "regime_64_dcqcn_direct7_1to1_exempt_p01.json",
                "regime_64_dcqcn_direct7_1to1_exempt_p01_b25.json",
                "regime_64_dcqcn_direct7_1to1_zero.json",
                "regime_64_dcqcn_direct7_1to1_zero_burst63.json",
                "regime_64_dcqcn_direct7_1to1_exempt_p01_burst63.json",
                "regime_64_dcqcn_direct7_4to1_exempt_p01_p0.json",
                "regime_64_dcqcn_direct2_2to1_exempt_p01.json",
            }
            # Wave 2, the DCQCN sensitivity check: the 4:1 p01 cell with the
            # controller's increase rates or ECN thresholds scaled.
            | {
                f"regime_64_dcqcn_direct7_4to1_exempt_p01_{tag}.json"
                for tag in ("rateai4x", "rateai025x", "ecn2x", "ecn05x")
            },
        )

    def test_recovery_domain_reaches_the_experiment_configuration(self) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(policy["selection_policy"]["domain"], "recovery")
        self.assertEqual(
            policy["selection_policy"]["semantics"], "recovery_forgiveness"
        )
        self.assertEqual(
            policy["selection_policy"]["transport"],
            {"selective_repair": True, "packet_trimming_ftd": True},
        )
        self.assertEqual(policy["scale"], {"ranks": 8, "steps": 3})
        self.assertEqual(manifest["selection_policy"]["domain"], "recovery")

    def test_congestion_exempt_domain_reaches_the_experiment_configuration(
        self,
    ) -> None:
        profile_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/exempt_smoke_8.json"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(policy["selection_policy"]["domain"], "recovery_exempt")
        self.assertEqual(
            policy["selection_policy"]["semantics"],
            "recovery_forgiveness_cc_exempt",
        )
        # The exemption forgives, so it owes the same transport contract the
        # CC-neutral domain owes.
        self.assertEqual(
            policy["selection_policy"]["transport"],
            {"selective_repair": True, "packet_trimming_ftd": True},
        )
        self.assertEqual(
            manifest["selection_policy"]["domain"], "recovery_exempt"
        )

    def test_congestion_exempt_domain_refuses_a_fabric_without_dcqcn(self) -> None:
        """Under CC_MODE 12 no rate cut exists to be exempt from.

        The refusal names the field, because a silent no-op would report as a
        null result from a mechanism that never ran.
        """
        document = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/exempt_smoke_8.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"

            without_dcqcn = json.loads(json.dumps(document))
            without_dcqcn["network"]["congestion_control"]["mode"] = "none"
            profile_path.write_text(json.dumps(without_dcqcn), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "network.congestion_control.mode"
            ):
                load_profile(profile_path)

            without_the_key = json.loads(json.dumps(document))
            del without_the_key["network"]["congestion_control"]
            profile_path.write_text(json.dumps(without_the_key), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "network.congestion_control.mode"
            ):
                load_profile(profile_path)

            # Everything the CC-neutral domain owes is still owed.
            without_repair = json.loads(json.dumps(document))
            del without_repair["network"]["transport_recovery"]["selective_repair"]
            profile_path.write_text(json.dumps(without_repair), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selective_repair"):
                load_profile(profile_path)

    def test_the_v2_receiver_policies_reach_the_experiment_configuration(
        self,
    ) -> None:
        """Both knobs cross into the simulator exactly as the profile names.

        The frontend cannot read the profile, so a knob that does not reach
        experiment.json is a variant that ran the arm beside it.
        """
        document = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
            ).read_text(encoding="utf-8")
        )
        document["selection_policy"]["pacing"] = {"kind": "bernoulli", "p": 0.25}
        document["selection_policy"]["cap_base"] = "owed"
        document["selection_policy"]["step_stop"] = True
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            output = Path(temporary_directory) / "experiment"
            materialize(profile_path, output)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            policy["selection_policy"]["pacing"],
            {"kind": "bernoulli", "p": 0.25},
        )
        self.assertEqual(policy["selection_policy"]["cap_base"], "owed")
        self.assertTrue(policy["selection_policy"]["step_stop"])

    def test_an_unpaced_profile_still_names_its_rule(self) -> None:
        """The default is a rule, not an absence, so it is written down."""
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(profile_path, output)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(policy["selection_policy"]["pacing"], {"kind": "none"})
        self.assertEqual(
            policy["selection_policy"]["cap_base"], "accounted"
        )
        self.assertNotIn("step_stop", policy["selection_policy"])
        self.assertNotIn("reengage", policy["selection_policy"])

    def test_the_v2_receiver_policies_refuse_a_malformed_profile(self) -> None:
        """Each refusal names its field.

        A probability is meaningful under Bernoulli and under nothing else, a
        probability outside the open interval is one of the other two rules
        spelled badly, and both policies belong to a domain whose receiver
        decides.
        """
        document = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
            ).read_text(encoding="utf-8")
        )
        broken = {
            "pacing.p": {"pacing": {"kind": "none", "p": 0.5}},
            "pacing.p ": {"pacing": {"kind": "bernoulli"}},
            "pacing.p   ": {"pacing": {"kind": "bernoulli", "p": 1.0}},
            "pacing.kind": {"pacing": {"kind": "poisson"}},
            # An unknown kind, because the receiver measures every cap
            # against what it has accounted for.
            "pacing.kind ": {"pacing": {"kind": "vesting"}},
            "cap_base": {"cap_base": "launched"},
            "step_stop": {"step_stop": "yes"},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for field, policy in broken.items():
                with self.subTest(field=field.strip(), policy=policy):
                    candidate = json.loads(json.dumps(document))
                    candidate["selection_policy"].update(policy)
                    profile_path.write_text(
                        json.dumps(candidate), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, field.strip()):
                        load_profile(profile_path)

            # Neither policy means anything where nothing is forgiven.
            for key, value in (
                ("pacing", {"kind": "bernoulli", "p": 0.5}),
                ("cap_base", "owed"),
                ("step_stop", True),
            ):
                with self.subTest(admission=key):
                    candidate = json.loads(json.dumps(document))
                    candidate["selection_policy"]["domain"] = "admission"
                    candidate["selection_policy"][key] = value
                    profile_path.write_text(
                        json.dumps(candidate), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, key):
                        load_profile(profile_path)

    def test_admission_domain_writes_no_recovery_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(self.profile_path, output)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            policy["selection_policy"]["semantics"], "logical_admission_selection"
        )
        self.assertNotIn("domain", policy["selection_policy"])
        self.assertNotIn("transport", policy["selection_policy"])

    def test_recovery_domain_refuses_a_transport_that_cannot_carry_it(self) -> None:
        document = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"

            without_repair = json.loads(json.dumps(document))
            del without_repair["network"]["transport_recovery"]["selective_repair"]
            profile_path.write_text(json.dumps(without_repair), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selective_repair"):
                load_profile(profile_path)

            without_ftd = json.loads(json.dumps(document))
            without_ftd["network"]["packet_trimming"]["mode"] = "bts"
            profile_path.write_text(json.dumps(without_ftd), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ftd"):
                load_profile(profile_path)

            unknown_domain = json.loads(json.dumps(document))
            unknown_domain["selection_policy"]["domain"] = "switch"
            profile_path.write_text(json.dumps(unknown_domain), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "domain"):
                load_profile(profile_path)

    def test_domain_override_forces_the_admission_arm(self) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
        )
        from experiments.ring_3d.generate import SheddingDomain

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(profile_path, output, domain=SheddingDomain.ADMISSION)
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            policy["selection_policy"]["semantics"], "logical_admission_selection"
        )
        self.assertNotIn("domain", policy["selection_policy"])

    def test_microburst_trigger_step_must_land_inside_the_run(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["microburst_trigger_step"] = document["steps"] + 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "microburst_trigger_step"):
                load_profile(profile_path)

    def test_no_incast_profile_disables_synthetic_background_traffic(self) -> None:
        profile_path = REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8.json"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(profile_path, output)

            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertFalse(policy["microburst"]["enabled"])
            self.assertEqual(policy["microburst"]["flows"], [])
            self.assertEqual(
                policy["selection_probability_by_step"],
                {"1": 0.005, "2": 0.1, "3": 0.1},
            )

    def test_fixed_p_low_override_preserves_enabled_microburst(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(
                self.profile_path,
                output,
                p_low=0.005,
                p_high=0.005,
            )

            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertTrue(policy["enabled"])
            self.assertTrue(policy["microburst"]["enabled"])
            self.assertEqual(
                policy["selection_probability_by_step"],
                {"1": 0.005, "2": 0.005, "3": 0.005},
            )

    def test_clr_exposure_flag_lifts_only_the_low_ceiling(self) -> None:
        from experiments.ring_3d.generate import resolve_selection_policy

        profile = load_profile(self.profile_path)
        with self.assertRaisesRegex(ValueError, "p_low"):
            resolve_selection_policy(profile, p_low=0.1, p_high=0.1)
        policy = resolve_selection_policy(
            profile, p_low=0.1, p_high=0.1, allow_clr_exposure=True
        )
        self.assertEqual((policy.p_low, policy.p_high), (0.1, 0.1))
        # The escape hatch lifts the ceiling only; ordering still holds.
        with self.assertRaisesRegex(ValueError, "p_high"):
            resolve_selection_policy(
                profile, p_low=0.1, p_high=0.05, allow_clr_exposure=True
            )

    def test_selection_policy_accepts_zero_and_keeps_its_other_bounds(
        self,
    ) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["selection_policy"]["p_low"] = 0.0
        document["selection_policy"]["p_high"] = 0.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "zero.json"
            output = Path(temporary_directory) / "experiment"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            materialize(profile_path, output)

            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                policy["selection_probability_by_step"],
                {"1": 0.0, "2": 0.0, "3": 0.0},
            )
            self.assertEqual(
                policy["selection_policy"],
                {
                    "semantics": "logical_admission_selection",
                    "p_low": 0.0,
                    "p_high": 0.0,
                    "p_low_threshold": 0,
                    "p_high_threshold": 0,
                },
            )
            # The zero is the only bound that moved.
            document["selection_policy"]["p_high"] = 0.005
            document["selection_policy"]["p_low"] = 0.0101
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "p_low"):
                load_profile(profile_path)
            document["selection_policy"]["p_low"] = 0.005
            document["selection_policy"]["p_high"] = 0.0
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "p_high"):
                load_profile(profile_path)

    def test_selection_policy_rejects_low_value_above_one_percent(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["selection_policy"]["p_low"] = 0.0101
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "p_low"):
                load_profile(profile_path)

    def test_data_loss_materializes_data_only_contract(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["data_loss"] = {
            "probability": 0.25,
            "start_ns": 100,
            "duration_ns": 5_000,
            "scope": "host_to_switch",
            "source_host": 0,
            "destination_host": 4,
            "receiver_node": 8,
            "rng_stream": 71,
        }
        document["network"]["transport_recovery"] = {
            "retransmission_timeout_ns": 500,
            "max_retransmission_retries": 3,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "lossy.json"
            output = Path(temporary_directory) / "experiment"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            manifest = materialize(profile_path, output)

            self.assertEqual(
                manifest["data_plane_loss"],
                {
                    "enabled": True,
                    "probability": 0.25,
                    "start_ns": 100,
                    "duration_ns": 5_000,
                    "scope": "host_to_switch",
                    "source_host": 0,
                    "destination_host": 4,
                    "receiver_node": 8,
                    "rng_stream": 71,
                },
            )
            self.assertEqual(
                manifest["transport_recovery"],
                {
                    "enabled": True,
                    "retransmission_timeout_ns": 500,
                    "max_retransmission_retries": 3,
                    "selective_repair": False,
                    "no_progress_timeout_ns": 5_000_000_000,
                },
            )
            network_config = (output / "network_config.txt").read_text(encoding="utf-8")
            self.assertIn("DATA_LOSS_PROBABILITY 0.25", network_config)
            self.assertIn("DATA_LOSS_SCOPE host_to_switch", network_config)
            self.assertIn("DATA_LOSS_RECEIVER_NODE 8", network_config)
            self.assertIn("RETRANSMISSION_TIMEOUT_NS 500", network_config)
            self.assertIn("MAX_RETRANSMISSION_RETRIES 3", network_config)
            self.assertIn("NO_PROGRESS_TIMEOUT_NS 5000000000", network_config)
            self.assertIn(
                "TRANSPORT_EVENT_OUTPUT_FILE "
                f"{output / 'ns3' / 'transport_events.csv.zst'}",
                network_config,
            )
            self.assertIn(
                "TRANSPORT_EVENT_SUMMARY_OUTPUT_FILE "
                f"{output / 'ns3' / 'transport_summary.csv'}",
                network_config,
            )
            # Rotation is sized for storage (segments are hashed and
            # deleted in flight), not for the release-asset cap.
            self.assertIn("TRANSPORT_EVENT_SEGMENT_BYTES 268435456", network_config)
            self.assertIn("ACK_HIGH_PRIO 1", network_config)
            self.assertEqual(
                Path(manifest["transport_event_file"]),
                output / "ns3" / "transport_events.csv.zst",
            )

    def test_retry_exhaustion_profile_is_a_data_only_failure_fixture(self) -> None:
        profile_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/retry_exhaustion_8.json"
        )
        profile = load_profile(profile_path)

        self.assertFalse(profile.microburst_enabled)
        self.assertIsNotNone(profile.network.data_loss)
        self.assertEqual(profile.network.data_loss.probability, 1.0)
        self.assertIsNotNone(profile.network.transport_recovery)
        self.assertEqual(
            profile.network.transport_recovery.max_retransmission_retries, 1
        )

    def test_data_loss_requires_bounded_recovery(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["data_loss"] = {
            "probability": 0.25,
            "start_ns": 0,
            "duration_ns": 1,
            "scope": "all",
            "rng_stream": 51,
        }
        document["network"]["transport_recovery"] = {
            "retransmission_timeout_ns": 0,
            "max_retransmission_retries": 3,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "transport_recovery.retransmission_timeout_ns"
            ):
                load_profile(profile_path)

    def test_data_loss_rejects_non_string_scope(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["data_loss"] = {
            "probability": 0.25,
            "start_ns": 0,
            "duration_ns": 1,
            "scope": ["all"],
            "rng_stream": 51,
        }
        document["network"]["transport_recovery"] = {
            "retransmission_timeout_ns": 100,
            "max_retransmission_retries": 1,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data_loss.scope"):
                load_profile(profile_path)

    def test_packet_trimming_materializes_both_standard_notification_modes(
        self,
    ) -> None:
        for mode in ("ftd", "bts"):
            document = json.loads(self.profile_path.read_text(encoding="utf-8"))
            document["network"]["packet_trimming"] = {"mode": mode}
            document["network"]["fabric"] = {
                "buffer_size_mb": 2,
                "pfc_enabled": False,
                "headroom_factor": 0,
                "data_queue_bytes": 262144,
                "trimmed_queue_bytes": 65536,
            }
            document["network"]["transport_recovery"] = {
                "retransmission_timeout_ns": 500,
                "max_retransmission_retries": 3,
            }
            with tempfile.TemporaryDirectory() as temporary_directory:
                profile_path = Path(temporary_directory) / f"trim-{mode}.json"
                output = Path(temporary_directory) / "experiment"
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                manifest = materialize(profile_path, output)

                self.assertEqual(
                    manifest["packet_trimming"],
                    {
                        "enabled": True,
                        "mode": mode,
                        "trigger": "switch_admission_or_egress_rejection",
                        "trimmed_queue": 2,
                        "trimmed_queue_weight": 25,
                        "min_trim_size_bytes": 24,
                        "last_hop_codepoint": True,
                        "uec_conformant": mode == "ftd",
                    },
                )
                network_config = (output / "network_config.txt").read_text(
                    encoding="utf-8"
                )
                self.assertIn(f"PACKET_TRIM_MODE {mode}", network_config)
                self.assertIn("PACKET_TRIM_QUEUE 2", network_config)
                self.assertIn("PACKET_TRIM_QUEUE_WEIGHT 25", network_config)
                self.assertIn("MIN_TRIM_SIZE 24", network_config)
                self.assertIn("PACKET_TRIM_LASTHOP 1", network_config)
                self.assertIn("RETRANSMISSION_TIMEOUT_NS 500", network_config)

    def test_trimmed_queue_must_not_collide_with_control_or_data(self) -> None:
        for queue in (0, 1, 3):
            document = json.loads(self.profile_path.read_text(encoding="utf-8"))
            document["network"]["packet_trimming"] = {
                "mode": "ftd",
                "trimmed_queue": queue,
            }
            document["network"]["fabric"] = {
                "buffer_size_mb": 2,
                "pfc_enabled": False,
                "headroom_factor": 0,
                "data_queue_bytes": 262144,
                "trimmed_queue_bytes": 65536,
            }
            document["network"]["transport_recovery"] = {
                "retransmission_timeout_ns": 500,
                "max_retransmission_retries": 3,
            }
            with tempfile.TemporaryDirectory() as temporary_directory:
                profile_path = Path(temporary_directory) / "invalid.json"
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "trimmed_queue"):
                    load_profile(profile_path)

    def test_trimmed_queue_weight_must_be_a_percentage(self) -> None:
        for weight in (0, 101):
            document = json.loads(self.profile_path.read_text(encoding="utf-8"))
            document["network"]["packet_trimming"] = {
                "mode": "ftd",
                "trimmed_queue_weight": weight,
            }
            document["network"]["fabric"] = {
                "buffer_size_mb": 2,
                "pfc_enabled": False,
                "headroom_factor": 0,
                "data_queue_bytes": 262144,
                "trimmed_queue_bytes": 65536,
            }
            document["network"]["transport_recovery"] = {
                "retransmission_timeout_ns": 500,
                "max_retransmission_retries": 3,
            }
            with tempfile.TemporaryDirectory() as temporary_directory:
                profile_path = Path(temporary_directory) / "invalid.json"
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "trimmed_queue_weight"):
                    load_profile(profile_path)

    def test_min_trim_size_must_retain_transport_headers(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["packet_trimming"] = {
            "mode": "ftd",
            "min_trim_size_bytes": 16,
        }
        document["network"]["fabric"] = {
            "buffer_size_mb": 2,
            "pfc_enabled": False,
            "headroom_factor": 0,
            "data_queue_bytes": 262144,
            "trimmed_queue_bytes": 65536,
        }
        document["network"]["transport_recovery"] = {
            "retransmission_timeout_ns": 500,
            "max_retransmission_retries": 3,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "min_trim_size_bytes"):
                load_profile(profile_path)

    def test_packet_trimming_requires_shared_recovery(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["packet_trimming"] = {"mode": "ftd"}
        document["network"]["fabric"] = {
            "buffer_size_mb": 2,
            "pfc_enabled": False,
            "headroom_factor": 0,
            "data_queue_bytes": 262144,
            "trimmed_queue_bytes": 65536,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transport_recovery"):
                load_profile(profile_path)

    def test_trimming_requires_an_explicit_best_effort_fabric(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["packet_trimming"] = {"mode": "ftd"}
        document["network"]["transport_recovery"] = {
            "retransmission_timeout_ns": 500,
            "max_retransmission_retries": 3,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "network.fabric"):
                load_profile(profile_path)

    def test_trimming_rejects_a_fabric_that_keeps_pfc(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["packet_trimming"] = {"mode": "ftd"}
        document["network"]["fabric"] = {
            "buffer_size_mb": 2,
            "pfc_enabled": True,
            "data_queue_bytes": 262144,
        }
        document["network"]["transport_recovery"] = {
            "retransmission_timeout_ns": 500,
            "max_retransmission_retries": 3,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pfc_enabled"):
                load_profile(profile_path)

    def test_disabled_pfc_requires_zero_headroom(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["fabric"] = {
            "buffer_size_mb": 2,
            "pfc_enabled": False,
            "headroom_factor": 3,
            "data_queue_bytes": 262144,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "headroom_factor"):
                load_profile(profile_path)

    def test_best_effort_fabric_requires_transport_recovery(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["fabric"] = {
            "buffer_size_mb": 2,
            "pfc_enabled": False,
            "headroom_factor": 0,
            "data_queue_bytes": 262144,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transport_recovery"):
                load_profile(profile_path)

    def test_comparison_arms_share_one_fabric(self) -> None:
        # Buffer depth decides whether incast becomes queueing delay or loss, so
        # it must be identical across every arm or it confounds the result.
        fabrics = []
        for filename in (
            "uec_trim_ftd_8.json",
            "bts_trim_8.json",
            "besteffort_baseline_8.json",
        ):
            profile = load_profile(
                REPOSITORY_ROOT / "experiments/ring_3d/profiles" / filename
            )
            self.assertIsNotNone(profile.network.fabric)
            fabrics.append(profile.network.fabric)
        self.assertEqual(len(set(fabrics)), 1)

    def test_checked_in_trim_profiles_select_expected_modes(self) -> None:
        # Only "ftd" is UEC 1.0.3 trimming, so only that profile is named "uec".
        for mode, filename in (
            ("ftd", "uec_trim_ftd_8.json"),
            ("bts", "bts_trim_8.json"),
        ):
            profile = load_profile(
                REPOSITORY_ROOT / "experiments/ring_3d/profiles" / filename
            )
            self.assertIsNotNone(profile.network.packet_trimming)
            self.assertEqual(profile.network.packet_trimming.mode, mode)
            self.assertEqual(
                profile.network.packet_trimming.uec_conformant, mode == "ftd"
            )
            self.assertIsNotNone(profile.network.transport_recovery)

    def test_queue_monitor_interval_must_be_positive(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["queue_monitor_interval_ns"] = 0
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "invalid.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "queue_monitor_interval_ns"):
                load_profile(profile_path)

    def test_materialization_covers_every_profile_training_step(self) -> None:
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["steps"] = 6
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "six_steps.json"
            output = Path(temporary_directory) / "experiment"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            manifest = materialize(profile_path, output)

            with (output / "clr_mask.csv").open(newline="", encoding="utf-8") as handle:
                mask_rows = list(csv.DictReader(handle))
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["step_id"] for row in mask_rows],
                [str(step) for step in range(1, 7)],
            )
            self.assertEqual(
                set(policy["selection_probability_by_step"]),
                {str(step) for step in range(1, 7)},
            )
            self.assertEqual(manifest["clr_schedule"]["steps"], 6)

    def test_llama3_profile_samples_production_sized_dp_and_tp_events(self) -> None:
        profile_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/llama3_70b_16.json"
        )
        profile = load_profile(profile_path)
        self.assertIsNotNone(profile.model)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)

            self.assertEqual(manifest["ranks"], 16)
            self.assertEqual(manifest["model_trace"]["parameter_count"], 70_000_000_000)
            self.assertEqual(
                manifest["model_trace"]["gradient_bytes_per_rank"], 17_500_000_000
            )
            self.assertEqual(
                manifest["model_trace"]["simulated_gradient_bucket_bytes"],
                68_359_375,
            )
            self.assertEqual(
                manifest["model_trace"]["gradient_accumulation_steps"],
                2,
            )
            self.assertEqual(
                manifest["model_trace"]["sampled_tp_all_reduces_per_step"], 4
            )
            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertTrue(policy["microburst"]["enabled"])
            self.assertEqual(len(policy["microburst"]["flows"]), 7)

            trace_path = output / "workload/ring_3d.8.et"
            dp_bytes_by_step: dict[int, int] = {}
            dp_buckets_by_step: dict[int, int] = {}
            tp_collectives_by_step: dict[int, int] = {}
            with trace_path.open("rb") as trace:
                metadata = GlobalMetadata()
                self.assertTrue(decodeMessage(trace, metadata))
                while True:
                    node = Node()
                    if not decodeMessage(trace, node):
                        break
                    attributes = {attribute.name: attribute for attribute in node.attr}
                    if (
                        attributes.get("parallelism_domain")
                        and attributes["parallelism_domain"].string_val == "dp"
                    ):
                        step = attributes["training_step"].uint64_val
                        dp_bytes_by_step[step] = (
                            dp_bytes_by_step.get(step, 0)
                            + attributes["comm_size"].uint64_val
                        )
                        dp_buckets_by_step[step] = dp_buckets_by_step.get(step, 0) + 1
                    if (
                        attributes.get("parallelism_domain")
                        and attributes["parallelism_domain"].string_val == "tp"
                    ):
                        step = attributes["training_step"].uint64_val
                        tp_collectives_by_step[step] = (
                            tp_collectives_by_step.get(step, 0) + 1
                        )
            steps = range(1, profile.steps + 1)
            self.assertEqual(
                dp_bytes_by_step, {step: 68_359_375 for step in steps}
            )
            self.assertEqual(dp_buckets_by_step, {step: 1 for step in steps})
            self.assertEqual(tp_collectives_by_step, {step: 4 for step in steps})

    def test_phase1_reference_profile_materializes_exact_sequential_dp_trace(
        self,
    ) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/dblp_phase1_effnet_64dp.json"
        )
        profile = load_profile(profile_path)
        self.assertEqual((profile.tp, profile.pp, profile.dp), (1, 1, 64))
        self.assertEqual(profile.steps, 186)
        self.assertEqual(profile.workload.kind, "sequential_dp_all_reduce")
        self.assertEqual(profile.dp_all_reduce_bytes, 21_200_000)
        self.assertEqual(
            profile.explicit_clr_schedule.critical_steps,
            (1, 2, 153, 166),
        )
        self.assertFalse(profile.microburst_enabled)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)
            self.assertEqual(manifest["workload"], {"kind": "sequential_dp_all_reduce"})
            self.assertEqual(
                manifest["clr_schedule_source"],
                {
                    "kind": "explicit_critical_steps",
                    "critical_steps": [1, 2, 153, 166],
                },
            )
            self.assertEqual(
                manifest["clr_schedule"],
                {
                    "model": "explicit_critical_steps",
                    "seed": 314159265,
                    "steps": 186,
                    "clr_step_count": 4,
                    "critical_steps": [1, 2, 153, 166],
                },
            )
            self.assertEqual(
                (output / "topology.txt").read_text(encoding="utf-8").splitlines()[0],
                "128 64 128",
            )

            policy = json.loads(
                (output / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(policy["selection_probability_by_step"]["1"], 0.008)
            self.assertEqual(policy["selection_probability_by_step"]["3"], 0.408)
            self.assertEqual(policy["selection_probability_by_step"]["153"], 0.008)
            self.assertFalse(policy["microburst"]["enabled"])
            self.assertEqual(policy["microburst"]["flows"], [])

            nodes: list[Node] = []
            with (output / "workload/ring_3d.0.et").open("rb") as trace:
                metadata = GlobalMetadata()
                self.assertTrue(decodeMessage(trace, metadata))
                while True:
                    node = Node()
                    if not decodeMessage(trace, node):
                        break
                    nodes.append(node)
            self.assertEqual(len(nodes), 186)
            for index, node in enumerate(nodes, start=1):
                attributes = {attribute.name: attribute for attribute in node.attr}
                self.assertEqual(node.name, f"step_{index}_dp_all_reduce")
                self.assertEqual(
                    attributes["parallelism_domain"].string_val,
                    "dp",
                )
                self.assertEqual(attributes["training_step"].uint64_val, index)
                self.assertEqual(attributes["comm_size"].uint64_val, 21_200_000)
                self.assertEqual(
                    list(node.ctrl_deps), [] if index == 1 else [index - 1]
                )

    def test_100b_profiles_materialize_topologies_with_exact_nonuniform_buckets(
        self,
    ) -> None:
        expected_topologies = {
            "model_100b_256_clos.json": {
                "kind": "clos",
                "header": (288, 32, 512),
                "description": "Two-stage leaf-spine Clos",
            },
            "model_100b_256_ring.json": {
                "kind": "ring",
                "header": (512, 256, 512),
                "description": "Host-attached bidirectional switch ring",
            },
        }
        for profile_name, expected in expected_topologies.items():
            with self.subTest(profile=profile_name):
                profile_path = (
                    REPOSITORY_ROOT / "experiments/ring_3d/profiles" / profile_name
                )
                profile = load_profile(profile_path)
                layout = build_topology(profile.network, profile.ranks)
                self.assertEqual(profile.ranks, 256)
                self.assertEqual(layout.kind, expected["kind"])
                self.assertEqual(
                    (layout.node_count, len(layout.switch_ids), len(layout.links)),
                    expected["header"],
                )
                if expected["kind"] == "ring":
                    self.assertEqual(layout.links[0].source, 0)
                    self.assertEqual(layout.links[0].destination, 256)
                    self.assertEqual(layout.links[255].source, 255)
                    self.assertEqual(layout.links[255].destination, 511)
                    self.assertEqual(layout.links[256].source, 256)
                    self.assertEqual(layout.links[256].destination, 257)
                    self.assertEqual(layout.links[-1].source, 511)
                    self.assertEqual(layout.links[-1].destination, 256)

                with tempfile.TemporaryDirectory() as temporary_directory:
                    output = Path(temporary_directory) / "experiment"
                    manifest = materialize(profile_path, output)
                    topology_lines = (
                        (output / "topology.txt")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    )
                    self.assertEqual(
                        tuple(map(int, topology_lines[0].split())), expected["header"]
                    )
                    self.assertEqual(len(topology_lines), expected["header"][2] + 2)
                    self.assertEqual(
                        manifest["physical_topology"]["kind"], expected["kind"]
                    )
                    self.assertEqual(
                        manifest["physical_topology"]["description"],
                        expected["description"],
                    )
                    self.assertIn(
                        "QLEN_MON_INTERVAL 10000",
                        (output / "network_config.txt").read_text(encoding="utf-8"),
                    )
                    self.assertEqual(
                        manifest["model_trace"]["parameter_count"], 100_000_000_000
                    )
                    self.assertEqual(
                        manifest["model_trace"]["gradient_bytes_per_rank"],
                        6_250_000_000,
                    )
                    self.assertEqual(
                        manifest["model_trace"]["gradient_bucket_count"], 96
                    )
                    self.assertEqual(
                        manifest["model_trace"]["gradient_bucket_min_bytes"], 65_104_166
                    )
                    self.assertEqual(
                        manifest["model_trace"]["gradient_bucket_max_bytes"], 65_104_167
                    )
                    self.assertEqual(
                        manifest["model_trace"]["gradient_bucket_total_bytes"],
                        6_250_000_000,
                    )
                    policy = json.loads(
                        (output / "experiment.json").read_text(encoding="utf-8")
                    )
                    self.assertTrue(policy["microburst"]["enabled"])
                    self.assertEqual(policy["microburst"]["trigger_step"], 2)
                    self.assertEqual(len(policy["microburst"]["flows"]), 7)
                    self.assertEqual(
                        {flow["src"] for flow in policy["microburst"]["flows"]},
                        {36, 68, 100, 132, 164, 196, 228},
                    )
                    self.assertEqual(
                        {flow["dst"] for flow in policy["microburst"]["flows"]},
                        {4},
                    )
                    self.assertEqual(
                        len(list((output / "workload").glob("ring_3d.*.et"))), 256
                    )
                    dp_bucket_count: dict[int, int] = {}
                    dp_bytes: dict[int, int] = {}
                    dp_bucket_sizes: set[int] = set()
                    with (output / "workload/ring_3d.0.et").open("rb") as trace:
                        metadata = GlobalMetadata()
                        self.assertTrue(decodeMessage(trace, metadata))
                        while True:
                            node = Node()
                            if not decodeMessage(trace, node):
                                break
                            attributes = {
                                attribute.name: attribute for attribute in node.attr
                            }
                            if attributes.get("parallelism_domain") and (
                                attributes["parallelism_domain"].string_val == "dp"
                            ):
                                step = attributes["training_step"].uint64_val
                                dp_bucket_count[step] = dp_bucket_count.get(step, 0) + 1
                                dp_bytes[step] = (
                                    dp_bytes.get(step, 0)
                                    + attributes["comm_size"].uint64_val
                                )
                                dp_bucket_sizes.add(attributes["comm_size"].uint64_val)
                    self.assertEqual(dp_bucket_count, {1: 96, 2: 96})
                    self.assertEqual(
                        dp_bytes,
                        {1: 6_250_000_000, 2: 6_250_000_000},
                    )
                    self.assertEqual(dp_bucket_sizes, {65_104_166, 65_104_167})

    def test_degraded_clos_builds_only_the_live_spine_tier(self) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/llama3_70b_64_direct2.json"
        )
        profile = load_profile(profile_path)
        self.assertEqual(profile.network.spine_count, 8)
        self.assertEqual(profile.network.failed_spine_count, 2)
        layout = build_topology(profile.network, profile.ranks)
        # 64 hosts + 8 leaves + 6 live spines; dark spines carry no links.
        self.assertEqual(layout.node_count, 78)
        self.assertEqual(len(layout.links), 64 + 8 * 6)
        manifest = layout.manifest()
        self.assertEqual(manifest["spine_count"], 8)
        self.assertEqual(manifest["failed_spine_count"], 2)
        self.assertEqual(manifest["live_spine_count"], 6)
        self.assertIn("degraded", manifest["description"])

    def test_selective_repair_reaches_config_and_manifest(self) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/llama3_70b_64_sr2x.json"
        )
        profile = load_profile(profile_path)
        self.assertTrue(profile.network.transport_recovery.selective_repair)
        # The canary runs the designed-2:1 fabric that collapsed go-back-N.
        self.assertEqual(profile.network.spine_count, 4)
        self.assertEqual(profile.network.failed_spine_count, 0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = materialize(profile_path, output)
            self.assertIn(
                "SELECTIVE_RETRANSMISSION 1",
                (output / "network_config.txt").read_text(encoding="utf-8"),
            )
            self.assertTrue(manifest["transport_recovery"]["selective_repair"])

    def test_selective_repair_defaults_off_and_rejects_non_boolean(self) -> None:
        profile = load_profile(
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/llama3_70b_64_direct2.json"
        )
        self.assertFalse(profile.network.transport_recovery.selective_repair)
        document = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments/ring_3d/profiles/llama3_70b_64_sr2x.json"
            ).read_text(encoding="utf-8")
        )
        document["network"]["transport_recovery"]["selective_repair"] = 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_profile = Path(temporary_directory) / "invalid.json"
            invalid_profile.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selective_repair"):
                load_profile(invalid_profile)

    def test_no_progress_deadline_defaults_on_and_rejects_zero(self) -> None:
        # The deadline is the liveness bound for budget-exempt recovery
        # signals, so every recovery-enabled profile must carry one.
        profile = load_profile(
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/llama3_70b_64_direct2.json"
        )
        self.assertEqual(
            profile.network.transport_recovery.no_progress_timeout_ns,
            5_000_000_000,
        )
        document = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments/ring_3d/profiles/llama3_70b_64_sr2x.json"
            ).read_text(encoding="utf-8")
        )
        document["network"]["transport_recovery"]["no_progress_timeout_ns"] = 0
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_profile = Path(temporary_directory) / "invalid.json"
            invalid_profile.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no_progress_timeout_ns"):
                load_profile(invalid_profile)

    def test_degraded_clos_requires_a_live_spine(self) -> None:
        profile_path = (
            REPOSITORY_ROOT
            / "experiments/ring_3d/profiles/llama3_70b_64_direct2.json"
        )
        document = json.loads(profile_path.read_text(encoding="utf-8"))
        document["network"]["failed_spine_count"] = 8
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_profile = Path(temporary_directory) / "invalid.json"
            invalid_profile.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "live spine"):
                load_profile(invalid_profile)

    def test_ring_network_rejects_clos_only_fields(self) -> None:
        profile_path = (
            REPOSITORY_ROOT / "experiments/ring_3d/profiles/model_100b_256_ring.json"
        )
        document = json.loads(profile_path.read_text(encoding="utf-8"))
        document["network"]["spine_count"] = 16
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_profile = Path(temporary_directory) / "invalid.json"
            invalid_profile.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown network keys"):
                load_profile(invalid_profile)

    def test_load_balancing_defaults_to_ecmp_and_writes_no_key(self) -> None:
        """An explicit ``ecmp`` generates exactly what the absent knob does."""
        document = json.loads(self.profile_path.read_text(encoding="utf-8"))
        document["network"]["load_balancing"] = {"mode": "ecmp"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            outputs = {}
            for arm, path in (("absent", self.profile_path), ("ecmp", profile_path)):
                output = Path(temporary_directory) / arm
                manifest = materialize(path, output)
                self.assertEqual(manifest["load_balancing"], {"mode": "ecmp"})
                outputs[arm] = {
                    name: (output / name).read_bytes().replace(
                        str(output).encode(), b"<output>"
                    )
                    for name in (
                        "network_config.txt",
                        "topology.txt",
                        "experiment.json",
                    )
                }

        self.assertEqual(outputs["absent"], outputs["ecmp"])
        self.assertNotIn(b"LOAD_BALANCING", outputs["ecmp"]["network_config.txt"])

    def test_load_balancing_modes_reach_the_configuration(self) -> None:
        document = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for mode in ("ev_hash", "spray_uniform"):
                document["network"]["load_balancing"] = {"mode": mode}
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / mode
                manifest = materialize(profile_path, output)
                config = (output / "network_config.txt").read_text(encoding="utf-8")
                expected = {"mode": mode}
                if mode == "ev_hash":
                    expected["selector"] = "ops"
                with self.subTest(mode=mode):
                    self.assertEqual(manifest["load_balancing"], expected)
                    self.assertIn(f"\nLOAD_BALANCING {mode}\n", config)
                    self.assertNotIn("PATH_SELECTOR", config)

    def test_path_selector_and_its_parameters_reach_the_configuration(self) -> None:
        """A selector writes its key and every parameter, defaults filled in;
        ``ops``, the simulator's default, writes nothing."""
        document = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            runs = itertools.count()

            def generated(balancing: dict[str, object]) -> tuple[dict, str]:
                document["network"]["load_balancing"] = balancing
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                output = Path(temporary_directory) / f"run{next(runs)}"
                manifest = materialize(profile_path, output)
                config = (output / "network_config.txt").read_text(encoding="utf-8")
                return manifest["load_balancing"], config.replace(str(output), "<o>")

            absent = generated({"mode": "ev_hash"})
            ops = generated({"mode": "ev_hash", "selector": "ops"})
            self.assertEqual(absent, ops)
            self.assertEqual(ops[0], {"mode": "ev_hash", "selector": "ops"})

            manifest, config = generated({"mode": "ev_hash", "selector": "reps"})
            self.assertEqual(
                manifest,
                {
                    "mode": "ev_hash",
                    "selector": "reps",
                    "buffer_size": 8,
                    "freezing_timeout_ns": 10_000_000,
                },
            )
            self.assertIn(
                "\nLOAD_BALANCING ev_hash\nPATH_SELECTOR reps\nREPS_BUFFER_SIZE 8\n"
                "REPS_FREEZING_TIMEOUT_NS 10000000\n",
                config,
            )
            manifest, config = generated(
                {
                    "mode": "ev_hash",
                    "selector": "reps",
                    "buffer_size": 4,
                    "freezing_timeout_ns": 500_000,
                }
            )
            self.assertEqual(
                (manifest["buffer_size"], manifest["freezing_timeout_ns"]), (4, 500_000)
            )
            self.assertIn(
                "\nREPS_BUFFER_SIZE 4\nREPS_FREEZING_TIMEOUT_NS 500000\n", config
            )

    def test_path_selector_refuses_what_nothing_would_read(self) -> None:
        document = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            reps = {"mode": "ev_hash", "selector": "reps"}
            refusals = [
                (
                    {"mode": "spray_uniform", "selector": "ops"},
                    "require mode 'ev_hash'",
                ),
                ({"mode": "ecmp", "selector": "reps"}, "require mode 'ev_hash'"),
                ({"mode": "ev_hash", "selector": "random"}, "selector must be one of"),
                ({"mode": "ev_hash", "buffer_size": 8}, "'ops' does not take"),
                ({**reps, "explore": 1}, "'reps' does not take"),
                ({**reps, "freezing_timeout_ns": 0}, "freezing_timeout_ns must be"),
            ]
            refusals += [
                ({**reps, "buffer_size": size}, "buffer_size must be")
                for size in (0, 256, True, 1.5)
            ]
            for balancing, message in refusals:
                document["network"]["load_balancing"] = balancing
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(balancing=balancing):
                    with self.assertRaisesRegex(ValueError, message):
                        load_profile(profile_path)

    def test_load_balancing_refuses_a_transport_that_cannot_carry_it(self) -> None:
        document = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/no_incast_8_zero.json"
            ).read_text(encoding="utf-8")
        )
        document["network"]["load_balancing"] = {"mode": "spray_uniform"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"

            def refused(variant: dict[str, object], message: str) -> None:
                profile_path.write_text(json.dumps(variant), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_profile(profile_path)

            without_repair = json.loads(json.dumps(document))
            del without_repair["network"]["transport_recovery"]["selective_repair"]
            refused(without_repair, "selective_repair")

            without_ftd = json.loads(json.dumps(document))
            without_ftd["network"]["packet_trimming"]["mode"] = "bts"
            refused(without_ftd, "ftd")

            without_trimming = json.loads(json.dumps(document))
            del without_trimming["network"]["packet_trimming"]
            refused(without_trimming, "ftd")

            for balancing in ({"mode": "spray_policy"}, {"mode": 2}):
                unknown_mode = json.loads(json.dumps(document))
                unknown_mode["network"]["load_balancing"] = balancing
                with self.subTest(balancing=balancing):
                    refused(unknown_mode, "load_balancing.mode")

            extra_key = json.loads(json.dumps(document))
            extra_key["network"]["load_balancing"]["selector"] = "ops"
            refused(extra_key, "takes no other key")

        ring = json.loads(
            (
                REPOSITORY_ROOT / "experiments/ring_3d/profiles/model_100b_256_ring.json"
            ).read_text(encoding="utf-8")
        )
        ring["network"]["load_balancing"] = {"mode": "ecmp"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(json.dumps(ring), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown network keys"):
                load_profile(profile_path)

    def test_load_balancing_and_recovery_domain_are_mutually_exclusive(self) -> None:
        document = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments/ring_3d/profiles/forgiveness_smoke_8.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            for mode in ("ev_hash", "spray_uniform"):
                document["network"]["load_balancing"] = {"mode": mode}
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                        load_profile(profile_path)

            document["network"]["load_balancing"] = {"mode": "ecmp"}
            profile_path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_profile(profile_path).network.load_balancing.mode, "ecmp"
            )

    def test_trace_has_explicit_domains_and_overlap_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            materialize(self.profile_path, output)
            trace_path = output / "workload/ring_3d.0.et"
            nodes: list[Node] = []
            with trace_path.open("rb") as trace:
                metadata = GlobalMetadata()
                self.assertTrue(decodeMessage(trace, metadata))
                while True:
                    node = Node()
                    if not decodeMessage(trace, node):
                        break
                    nodes.append(node)

            attributes = {
                node.name: {attribute.name: attribute for attribute in node.attr}
                for node in nodes
            }
            self.assertEqual(
                attributes["step_1_tp_all_reduce"]["parallelism_domain"].string_val,
                "tp",
            )
            self.assertEqual(
                attributes["step_1_pp_send_to_2"]["parallelism_domain"].string_val,
                "pp",
            )
            self.assertEqual(
                attributes["step_1_dp_all_reduce_bucket_0"][
                    "parallelism_domain"
                ].string_val,
                "dp",
            )

            ids = {node.name: node.id for node in nodes}
            optimizer = next(node for node in nodes if node.name == "step_1_optimizer")
            bucket_one = next(
                node for node in nodes if node.name == "step_1_backward_bucket_1"
            )
            self.assertIn(ids["step_1_dp_all_reduce_bucket_0"], optimizer.ctrl_deps)
            self.assertIn(ids["step_1_dp_all_reduce_bucket_1"], optimizer.ctrl_deps)
            self.assertNotIn(ids["step_1_dp_all_reduce_bucket_0"], bucket_one.ctrl_deps)


class Ring3DDecisionThresholdTests(unittest.TestCase):
    """The scaled threshold must be what C++ llround(p * kDecisionScale) is.

    A tie is the only case where the candidate roundings differ, and it is
    reachable: p_low 5e-7 scales to exactly 0.5.
    """

    def test_a_tie_rounds_away_from_zero_not_to_even(self) -> None:
        self.assertEqual(scaled_threshold(5e-7), 1)
        self.assertEqual(scaled_threshold(2.5e-6), 3)
        self.assertEqual(scaled_threshold(1.5e-6), 2)

    def test_a_non_tie_rounds_to_nearest(self) -> None:
        self.assertEqual(scaled_threshold(1.2e-6), 1)
        self.assertEqual(scaled_threshold(1.9e-6), 2)

    def test_the_documented_probabilities_scale_exactly(self) -> None:
        self.assertEqual(scaled_threshold(0.005), 5_000)
        self.assertEqual(scaled_threshold(0.1), 100_000)
        self.assertEqual(scaled_threshold(1.0), DECISION_SCALE)
        self.assertEqual(scaled_threshold(0.0), 0)


if __name__ == "__main__":
    unittest.main()
