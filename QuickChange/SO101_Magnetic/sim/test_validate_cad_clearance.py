#!/usr/bin/env python3
"""Focused regressions for the core exact CAD-clearance authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cadquery as cq

import validate_cad_clearance as clearance


def _load_matcha_cad_source():
    path = (
        clearance.QUICK_CHANGE_DIR
        / "matcha_tools"
        / "generate_matcha_tool_cad.py"
    )
    spec = importlib.util.spec_from_file_location("matcha_cad_stop_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATCHA_CAD = _load_matcha_cad_source()


def _published_report_matches_current_validator() -> bool:
    if not clearance.REPORT_PATH.is_file():
        return False
    try:
        report = json.loads(clearance.REPORT_PATH.read_text())
        return report["authorities"]["validator"]["sha256"] == clearance._sha256(
            Path(clearance.__file__)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


class CoreCadClearanceUnitTests(unittest.TestCase):
    def test_sweep_grid_closes_exactly_from_zero_to_eighty(self) -> None:
        positions = clearance._sweep_positions(1.0)
        self.assertEqual(positions[0], 0.0)
        self.assertEqual(positions[-1], 80.0)
        self.assertEqual(len(positions), 81)
        self.assertEqual(
            clearance._canonical_sha256(positions),
            clearance._canonical_sha256(clearance._sweep_positions(1.0)),
        )
        with self.assertRaises(RuntimeError):
            clearance._sweep_positions(3.0)

    def test_intended_keeper_contacts_are_an_exact_named_set(self) -> None:
        self.assertEqual(
            clearance.INTENDED_ZERO_VOLUME_CONTACT_PAIRS,
            frozenset(
                {
                    ("stock_tool_plate", "left_lower_rail"),
                    ("stock_tool_plate", "right_lower_rail"),
                    ("stock_tool_plate", "left_upper_rail"),
                    ("stock_tool_plate", "right_upper_rail"),
                    ("robot_plate", "left_lower_rail"),
                }
            ),
        )
        self.assertNotIn(
            ("robot_plate", "positive_lock_cam"),
            clearance.INTENDED_ZERO_VOLUME_CONTACT_PAIRS,
        )

    def test_zero_volume_tangency_is_rejected_without_explicit_policy(self) -> None:
        moving = clearance.BRepComponent(
            "synthetic_moving",
            cq.Workplane("XY").box(1.0, 1.0, 1.0),
            "test",
        )
        fixed = (
            cq.Workplane("XY")
            .box(1.0, 1.0, 1.0)
            .translate((1.0, 0.0, 0.0))
            .val()
        )
        forbidden = clearance._brep_sweep_record(
            moving,
            fixed,
            [0.0, 1.0],
            dock_component="synthetic_fixed",
            intended_zero_volume_contact=False,
        )
        intended = clearance._brep_sweep_record(
            moving,
            fixed,
            [0.0, 1.0],
            dock_component="synthetic_fixed",
            intended_zero_volume_contact=True,
        )
        self.assertFalse(forbidden["passed"])
        self.assertTrue(intended["passed"])
        self.assertEqual(forbidden["maximum_sampled_overlap_volume_mm3"], 0.0)
        self.assertEqual(intended["initial_distance_mm"], 0.0)

    def test_mount_contract_is_composed_from_published_source(self) -> None:
        contract = clearance._source_mount_contract()
        for observed, expected in zip(
            contract["tool_local_fixed_step_pos_mm"], [0.4875, 0.218, 9.551]
        ):
            self.assertTrue(math.isclose(observed, expected, abs_tol=1.0e-12))
        self.assertEqual(contract["tool_local_fixed_step_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(contract["source_function_sha256"]), 64)
        self.assertTrue(math.isclose(
            contract["robot_assembly_fixed_step_pos_mm"][2]
            - contract["robot_assembly_stock_plate_pos_mm"][2],
            9.551,
            abs_tol=1.0e-12,
        ))

        sim_transform = clearance._required_stock_sim_body_transform()
        self.assertLessEqual(sim_transform["position_residual_m"], 1.0e-12)
        self.assertLessEqual(
            sim_transform["rotation_residual_frobenius"], 1.0e-12
        )
        self.assertTrue(
            math.isclose(
                sim_transform["wrapper_body_pos_m"][0],
                0.0004875,
                abs_tol=1.0e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                sim_transform["wrapper_body_pos_m"][2],
                0.010500706,
                abs_tol=1.0e-15,
            )
        )

    def test_mating_hardware_nominal_fit_and_authority_blockers_are_explicit(
        self,
    ) -> None:
        cad = clearance.CAD
        contract = cad.interface_hardware_fit_contract()
        robot_native = cad.robot_plate().val()
        tool_native = cad.tool_plate(stock_gripper=True).val()
        self.assertTrue(robot_native.isValid())
        self.assertTrue(tool_native.isValid())
        self.assertEqual(len(robot_native.Solids()), 1)
        self.assertEqual(len(tool_native.Solids()), 1)

        robot = cad.robot_plate().translate(
            (0.0, 0.0, -cad.PLATE_THICKNESS)
        ).val()
        tool = tool_native
        motion_bound = contract["unqualified_local_motion_allowance_mm"]
        arithmetic_residual = contract[
            "unqualified_arithmetic_residual_mm"
        ]
        required = contract["required_clearance_mm"]
        self.assertFalse(contract["release_authority"]["release_ready"])
        self.assertFalse(
            contract["release_authority"][
                "fabrication_process_tolerance_qualified"
            ]
        )
        self.assertIsNone(
            contract["release_authority"]["qualified_combined_error_limit_mm"]
        )

        pad_distances = []
        for x_value, y_value in cad.pogo_points():
            pad = cad.contact_pad().translate(
                (x_value, y_value, -cad.CONTACT_PAD_THICKNESS)
            ).val()
            self.assertEqual(clearance._intersection_volume_mm3(robot, pad), 0.0)
            pad_distances.append(float(robot.distance(pad)))
        self.assertTrue(
            all(math.isclose(value, 0.30, abs_tol=1.0e-12) for value in pad_distances)
        )
        self.assertGreaterEqual(
            min(pad_distances) - motion_bound - required,
            arithmetic_residual - 1.0e-12,
        )

        cross_distances = []
        magnet_target_distances = []
        for x_value, y_value in cad.magnet_points():
            magnet = cad.screw_on_magnet().translate(
                (
                    x_value,
                    y_value,
                    -cad.MAGNET_HEIGHT - cad.MAGNETIC_HARDWARE_FACE_RECESS,
                )
            ).val()
            target = cad.steel_target().translate(
                (x_value, y_value, cad.MAGNETIC_HARDWARE_FACE_RECESS)
            ).val()
            self.assertEqual(
                clearance._intersection_volume_mm3(magnet, robot), 0.0
            )
            self.assertEqual(
                clearance._intersection_volume_mm3(target, tool), 0.0
            )
            self.assertTrue(math.isclose(magnet.distance(robot), 0.0, abs_tol=1.0e-12))
            self.assertTrue(math.isclose(target.distance(tool), 0.0, abs_tol=1.0e-12))
            self.assertEqual(clearance._intersection_volume_mm3(magnet, tool), 0.0)
            self.assertEqual(clearance._intersection_volume_mm3(target, robot), 0.0)
            cross_distances.extend(
                [float(magnet.distance(tool)), float(target.distance(robot))]
            )
            magnet_target_distances.append(float(magnet.distance(target)))

        expected_cross_distance = math.hypot(0.30, 0.05)
        self.assertTrue(
            all(
                math.isclose(value, expected_cross_distance, abs_tol=1.0e-12)
                for value in cross_distances
            )
        )
        self.assertGreaterEqual(
            min(cross_distances) - motion_bound - required,
            arithmetic_residual - 1.0e-12,
        )
        self.assertTrue(
            all(
                math.isclose(value, 0.10, abs_tol=1.0e-12)
                for value in magnet_target_distances
            )
        )

        studs = [
            cad.shoulder_lock_stud().translate((x_value, 0.0, 0.0)).val()
            for x_value in (-cad.LOCK_STUD_X, cad.LOCK_STUD_X)
        ]
        minimum_stud_distance = min(float(robot.distance(stud)) for stud in studs)
        self.assertTrue(
            math.isclose(minimum_stud_distance, 0.40, abs_tol=1.0e-12)
        )
        self.assertGreaterEqual(
            minimum_stud_distance - motion_bound - required,
            arithmetic_residual - 1.0e-12,
        )

        pad_contract = contract["pogo_target_pad_relief"]
        self.assertTrue(
            math.isclose(
                pad_contract["minimum_adjacent_surface_web_mm"],
                0.40,
                abs_tol=1.0e-12,
            )
        )
        pogo_contract = pad_contract["pogo_interface_authority"]
        selected_mount = pad_contract["selected_sectional_mount"]
        self.assertEqual(selected_mount["mode"], "knurl_solder_cup_first")
        self.assertTrue(
            math.isclose(
                selected_mount["retention_land_diameter_mm"],
                1.58,
                abs_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                selected_mount["body_counterbore_diameter_mm"],
                2.31,
                abs_tol=1.0e-12,
            )
        )
        self.assertFalse(
            pogo_contract["official_sources"]["redistribution"]
            ["manufacturer_file_license_confirmed"]
        )
        self.assertFalse(
            pogo_contract["official_sources"]["redistribution"]
            ["manufacturer_files_vendored"]
        )
        self.assertEqual(
            pogo_contract["official_sources"]["dimension_drawing_svg"]
            ["sha256"],
            cad.POGO_DIMENSION_DRAWING_SHA256,
        )
        self.assertEqual(
            pogo_contract["official_sources"]
            ["press_fit_application_note_pdf"]["sha256"],
            cad.POGO_PRESS_FIT_NOTE_SHA256,
        )
        self.assertTrue(
            math.isclose(
                cad.POGO_PLUNGER_DIAMETER,
                0.042 * 25.4,
                abs_tol=1.0e-12,
            )
        )
        fixed_pin_bounds = cad.pogo_official_fixed_shell().val().BoundingBox()
        self.assertTrue(
            math.isclose(
                fixed_pin_bounds.zmax,
                cad.POGO_FIXED_SHELL_LENGTH,
                abs_tol=1.0e-12,
            )
        )
        for signal in cad.CONTACT_SIGNALS:
            datum = cad.pogo_installed_datum(signal)
            mated_plunger = cad.pogo_official_plunger(
                datum["mated_compression_mm"]
            ).translate((0.0, 0.0, datum["base_z_mm"]))
            self.assertTrue(
                math.isclose(
                    mated_plunger.val().BoundingBox().zmax,
                    cad.PLATE_THICKNESS - cad.CONTACT_PAD_THICKNESS,
                    abs_tol=1.0e-12,
                )
            )
        for bad_compression in (
            -0.01,
            cad.POGO_MAXIMUM_FULL_STROKE + 0.01,
            math.nan,
        ):
            with self.assertRaises(ValueError):
                cad.pogo_official_plunger(bad_compression)
        well_contract = contract["fixed_stud_head_wells"]
        self.assertGreaterEqual(well_contract["cam_relief_ligament_mm"], 7.95)
        self.assertGreaterEqual(
            well_contract["nearest_horn_counterbore_ligament_mm"], 1.90
        )
        self.assertGreaterEqual(
            well_contract["slider_lobe_bearing_annulus_mm"], 0.80
        )
        evidence = clearance._interface_hardware_fit_record()
        self.assertTrue(evidence["geometry_passed"], evidence)
        self.assertFalse(evidence["authority_passed"])
        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["release_ready"])
        self.assertTrue(evidence["pogo_sectional_mount"]["geometry_passed"])
        self.assertEqual(
            evidence["authority_blockers"],
            contract["release_authority"]["blockers"],
        )
        forged_authority = copy.deepcopy(contract["release_authority"])
        forged_authority["release_ready"] = True
        forged_verdict = clearance._interface_authority_verdict(
            forged_authority
        )
        self.assertFalse(forged_verdict["computed_release_ready"])
        self.assertFalse(forged_verdict["declaration_consistent"])
        missing_limit = copy.deepcopy(contract["release_authority"])
        for flag, _blocker in clearance.INTERFACE_AUTHORITY_REQUIREMENTS:
            missing_limit[flag] = True
        missing_limit["blockers"] = []
        missing_limit["release_ready"] = True
        missing_limit_verdict = clearance._interface_authority_verdict(
            missing_limit
        )
        self.assertFalse(missing_limit_verdict["computed_release_ready"])
        self.assertIn(
            "qualified_combined_error_limit_missing",
            missing_limit_verdict["expected_blockers"],
        )

    def test_official_pogo_sectional_mount_and_adversarials(self) -> None:
        cad = clearance.CAD
        record = clearance._pogo_sectional_mount_record()
        self.assertTrue(record["geometry_passed"], record)
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertFalse(record["release_ready"])
        self.assertEqual(
            set(record["blockers"]),
            {
                "ground_first_mate_tolerance_stack_unqualified",
                "knurl_press_fit_process_and_pullout_unqualified",
                "installed_electrical_cycle_reliability_unqualified",
            },
        )
        authority_dir_entries = sorted(
            path.name for path in cad.POGO_SOURCE_AUTHORITY_DIR.iterdir()
        )
        self.assertEqual(authority_dir_entries, ["authority_ledger.json"])
        self.assertEqual(len(record["segment_records"]), 5)
        self.assertEqual(len(record["pin_records"]), 4)
        for pin in record["pin_records"]:
            self.assertGreater(
                pin["fixed_feature_records"]["knurl"]
                ["overlap_with_printed_plate_mm3"],
                0.0,
            )
            self.assertEqual(
                pin["fixed_feature_records"]["shoulder"]
                ["overlap_with_printed_plate_mm3"],
                0.0,
            )
            self.assertEqual(
                pin["mated_plunger_to_target_pad_overlap_mm3"],
                0.0,
            )
            self.assertTrue(
                math.isclose(
                    pin["mated_plunger_tip_z_mm"],
                    cad.PLATE_THICKNESS - cad.CONTACT_PAD_THICKNESS,
                    abs_tol=1.0e-12,
                )
            )

        first_mate = record["source_contract"]["first_mate_tolerance_stack"]
        self.assertTrue(
            clearance._pogo_first_mate_stack_is_consistent(first_mate)
        )
        self.assertTrue(
            math.isclose(
                first_mate["independent_pin_pair_error_bound_mm"],
                0.6096,
                abs_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                first_mate["guaranteed_worst_case_ground_lead_mm"],
                -0.4096,
                abs_tol=1.0e-12,
            )
        )
        for key, replacement in (
            ("independent_standard_length_tolerance_term_count", 2),
            ("independent_pin_pair_error_bound_mm", 0.3048),
            ("guaranteed_worst_case_ground_lead_mm", 0.01),
            ("passed", True),
        ):
            adversarial = copy.deepcopy(first_mate)
            adversarial[key] = replacement
            self.assertFalse(
                clearance._pogo_first_mate_stack_is_consistent(adversarial),
                key,
            )

        interface_authority = clearance.CAD.interface_hardware_fit_contract()[
            "release_authority"
        ]
        self_attested = copy.deepcopy(interface_authority)
        for flag, _blocker in clearance.INTERFACE_AUTHORITY_REQUIREMENTS:
            self_attested[flag] = True
        self_attested["qualified_combined_error_limit_mm"] = 0.01
        self_attested["blockers"] = []
        self_attested["release_ready"] = True
        verdict = clearance._interface_authority_verdict(
            self_attested,
            record,
        )
        self.assertFalse(verdict["computed_release_ready"])
        self.assertFalse(verdict["declared_flags_match_evidence"])
        self.assertFalse(verdict["declaration_consistent"])

    def test_robot_plate_cam_relief_is_continuously_clear(self) -> None:
        robot_plate = next(
            component
            for component in clearance._robot_side_components()
            if component.name == "robot_plate"
        )
        result = clearance._brep_sweep_record(
            robot_plate,
            clearance.CAD.positive_lock_cam().val(),
            clearance._sweep_positions(clearance.CAM_CLEARANCE_SWEEP_STEP_MM),
            dock_component="positive_lock_cam",
            intended_zero_volume_contact=False,
        )
        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(
            result["minimum_sampled_distance_mm"] + 1.0e-12,
            clearance.CAD.ROBOT_CAM_CLEARANCE_MM,
        )
        self.assertGreaterEqual(
            result["continuous_certified_clearance_mm"],
            clearance.MANUFACTURING_CLEARANCE_MM,
        )
        self.assertEqual(result["maximum_sampled_overlap_volume_mm3"], 0.0)
        self.assertEqual(result["sample_step_mm"], 0.1)
        contract = clearance.CAD.robot_cam_relief_contract()
        self.assertTrue(contract["through_full_printed_plate_thickness"])
        self.assertEqual(contract["bounds_native_mm"]["z"], [0.0, 9.5])

    def test_guided_axial_capture_has_continuous_cam_clearance(self) -> None:
        robot_plate = next(
            component
            for component in clearance._robot_side_components()
            if component.name == "robot_plate"
        )
        result = clearance._axial_capture_cam_record(
            robot_plate, clearance.CAD.positive_lock_cam().val()
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["sample_count"], 151)
        self.assertEqual(result["lateral_offset_start_mm"], 0.20)
        self.assertEqual(result["lateral_offset_end_mm"], 0.0)
        self.assertEqual(result["recenter_start_preseat_mm"], 6.4)
        self.assertEqual(result["recenter_end_preseat_mm"], 3.2)
        self.assertTrue(
            math.isclose(result["minimum_sampled_distance_mm"], 0.30, abs_tol=1.0e-12)
        )
        self.assertTrue(
            math.isclose(
                result["continuous_certified_clearance_mm"],
                0.24990243893162106,
                abs_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                result["maximum_between_sample_motion_bound_mm"],
                0.050097561068379634,
                abs_tol=1.0e-12,
            )
        )
        self.assertEqual(result["maximum_sampled_overlap_volume_mm3"], 0.0)

    def test_cam_relief_preserves_slider_and_keyhole_mechanism(self) -> None:
        cad = clearance.CAD
        slider = cad.locking_slider()
        self.assertTrue(
            math.isclose(
                slider.val().Volume(),
                clearance.POSITIVE_LOCK_RELEASE_SLIDER_VOLUME_MM3,
                abs_tol=1.0e-9,
            )
        )
        bounds = slider.val().BoundingBox()
        self.assertEqual(
            [bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, bounds.zmin, bounds.zmax],
            [-15.974062500000002, 24.0, -4.4, 4.4, 0.0, 1.6],
        )
        cam = cad.positive_lock_cam().val()
        unlocked = slider.translate(
            (0.0, 0.0, cad.SLIDER_Z - cad.PLATE_THICKNESS)
        ).val()
        locked = slider.translate(
            (cad.SLIDER_TRAVEL, 0.0, cad.SLIDER_Z - cad.PLATE_THICKNESS)
        ).val()
        self.assertTrue(math.isclose(unlocked.distance(cam), 0.05, abs_tol=1.0e-12))
        self.assertTrue(
            math.isclose(
                clearance._intersection_volume_mm3(locked, cam),
                8.970937500000003,
                abs_tol=1.0e-9,
            )
        )
        self.assertTrue(
            math.isclose(
                cad.SLIDER_TAB_END_X + cad.SLIDER_TRAVEL - cad.DOCK_CAM_X_INNER,
                2.95,
                abs_tol=1.0e-12,
            )
        )
        preservation = clearance._mechanism_preservation_record()
        self.assertTrue(preservation["passed"], preservation)
        self.assertTrue(all(preservation["checks"].values()))
        self.assertTrue(
            math.isclose(
                preservation["robot_plate_volume_mm3"],
                clearance.ROBOT_PLATE_EXPECTED_SOURCE_VOLUME_MM3,
                abs_tol=1.0e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                preservation["removed_volume_mm3"],
                291.9951063340377,
                abs_tol=1.0e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                preservation["relief_contract"][
                    "minimum_relief_to_stud_well_ligament_mm"
                ],
                7.950000000000001,
                abs_tol=1.0e-12,
            )
        )
        self.assertEqual(
            preservation["gross_volume_guard_role"],
            "secondary_sanity_only_local_web_ligament_and_clearance_checks_are_authoritative",
        )
        self.assertTrue(
            math.isclose(
                preservation["relief_contract"][
                    "minimum_relief_to_slider_lobe_ligament_mm"
                ],
                7.15,
                abs_tol=1.0e-12,
            )
        )

    def test_integral_cam_lead_passive_capture_and_release_are_exact(self) -> None:
        cad = clearance.CAD
        contract = cad.positive_lock_cam_contract()
        self.assertEqual(
            contract["construction"],
            "union_main_xy_wedge_ruled_axial_lead_hold_finger",
        )
        self.assertEqual(
            contract["axial_lead"]["lower_rectangle_mm"],
            {"x": [27.25, 29.0], "y": [0.0, 2.0], "z": -9.6},
        )
        self.assertEqual(
            contract["axial_lead"]["upper_rectangle_mm"],
            {"x": [24.05, 29.0], "y": [0.0, 2.0], "z": -6.4},
        )
        self.assertEqual(
            contract["outer_root_bridge"]["bounds_mm"],
            {
                "x": [28.0, 29.0],
                "y": [-1.0, 1.0],
                "z": [-4.65, -3.65],
            },
        )
        self.assertTrue(contract["outer_root_bridge"]["outside_locked_tab_swept_x"])
        self.assertAlmostEqual(contract["expected_geometry"]["total_volume_mm3"], 325.435)
        self.assertAlmostEqual(contract["manufacturability"]["minimum_feature_mm"], 1.0)
        self.assertAlmostEqual(
            contract["quasistatic_load_envelope"]["maximum_spring_force_n"],
            3.528,
        )
        self.assertAlmostEqual(
            contract["quasistatic_load_envelope"]["maximum_cam_normal_force_n"],
            4.989345448052276,
        )

        self.assertAlmostEqual(cad.positive_lock_cam_capture_lateral_offset_mm(6.4), 0.20)
        self.assertAlmostEqual(cad.positive_lock_cam_capture_lateral_offset_mm(3.2), 0.0)
        self.assertAlmostEqual(cad.positive_lock_cam_capture_q_max_mm(6.4), 3.0)
        self.assertAlmostEqual(cad.positive_lock_cam_capture_q_max_mm(3.2), 0.05)
        self.assertAlmostEqual(cad.positive_lock_cam_capture_q_max_mm(3.1), 0.05)
        self.assertAlmostEqual(cad.positive_lock_cam_release_q_max_mm(2.0), 0.05)
        self.assertAlmostEqual(cad.positive_lock_cam_release_q_max_mm(15.0), 3.0)
        for helper in (
            cad.positive_lock_cam_capture_lateral_offset_mm,
            cad.positive_lock_cam_capture_q_max_mm,
            cad.positive_lock_cam_release_q_max_mm,
        ):
            with self.assertRaises(ValueError):
                helper(-0.01)
            with self.assertRaises(ValueError):
                helper(math.nan)

        result = clearance._passive_positive_lock_cam_record()
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()), result)
        self.assertTrue(result["geometry"]["cam_is_single_valid_solid"])
        self.assertTrue(
            math.isclose(
                result["geometry"]["cam_volume_mm3"],
                325.435,
                abs_tol=1.0e-9,
            )
        )
        head = result["capture"]["head_entry_sample"]
        self.assertEqual(head["preseat_mm"], 3.1)
        self.assertEqual(head["lateral_mm"], 0.0)
        self.assertLessEqual(head["q_mm"], 0.05 + 1.0e-12)
        self.assertGreaterEqual(head["minimum_slider_stud_distance_mm"], 0.20)
        self.assertEqual(result["capture"]["maximum_slider_cam_overlap_mm3"], 0.0)
        self.assertEqual(result["capture"]["maximum_slider_stud_overlap_mm3"], 0.0)
        self.assertGreaterEqual(
            result["capture"]["tight_stud_clearance"][
                "continuous_certified_clearance_mm"
            ],
            0.20,
        )
        self.assertGreaterEqual(
            result["capture"]["robot_plate_cam"][
                "continuous_certified_clearance_mm"
            ],
            0.20,
        )
        self.assertTrue(
            all(
                record["maximum_overlap_volume_mm3"] == 0.0
                for record in result["capture"]["component_cam_records"]
            )
        )
        exit_sample = result["release"]["nominal_exit_sample"]
        self.assertEqual(exit_sample["withdrawal_mm"], 15.0)
        self.assertEqual(exit_sample["q_mm"], 3.0)
        self.assertEqual(exit_sample["slider_cam_overlap_mm3"], 0.0)
        self.assertTrue(
            math.isclose(
                exit_sample["slider_cam_distance_mm"],
                0.25181477893752996,
                abs_tol=1.0e-9,
            )
        )

    def test_positive_lock_capsule_clears_shoulders_and_retains_heads(self) -> None:
        cad = clearance.CAD
        contract = cad.positive_lock_keyhole_contract()
        self.assertEqual(contract["shoulder_path_kind"], "capsule")
        self.assertEqual(
            contract["shoulder_path_centerline_x_offsets_mm"], [-3.0, 0.0]
        )
        self.assertEqual(
            contract["shoulder_path_overall_x_offsets_mm"], [-5.125, 2.125]
        )
        self.assertTrue(
            math.isclose(
                contract["minimum_radial_shoulder_clearance_mm"],
                0.125,
                abs_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                contract["minimum_radial_head_retention_overlap_mm"],
                0.875,
                abs_tol=1.0e-12,
            )
        )
        result = clearance._positive_lock_travel_record()
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()), result)
        self.assertEqual(result["sample_count"], 61)
        self.assertEqual(result["sample_step_mm"], 0.05)
        self.assertLessEqual(
            result["maximum_sampled_shoulder_overlap_volume_mm3"],
            clearance.OVERLAP_VOLUME_TOLERANCE_MM3,
        )
        self.assertTrue(
            math.isclose(
                result["minimum_sampled_shoulder_clearance_mm"],
                0.125,
                abs_tol=1.0e-12,
            )
        )
        self.assertEqual(
            result["unlocked_projected_head_overlap_volume_mm3"],
            {"left": 0.0, "right": 0.0},
        )
        self.assertTrue(
            math.isclose(
                result["locked_projected_head_retention_volume_mm3"]["left"],
                4.149293812932416,
                abs_tol=1.0e-9,
            )
        )
        self.assertTrue(
            math.isclose(
                result["locked_projected_head_retention_volume_mm3"]["right"],
                12.190289700424891,
                abs_tol=1.0e-9,
            )
        )
    def test_per_dock_stop_contracts_match_exact_source_solids(self) -> None:
        core_spec = clearance.CAD.core_dock_stop_spec()
        core_bounds = clearance._bbox_record(
            clearance._bbox_tuple(clearance.CAD.core_dock_stop().val())
        )
        self.assertEqual(
            core_bounds,
            {
                "x_mm": core_spec["bounds_mm"]["x"],
                "y_mm": core_spec["bounds_mm"]["y"],
                "z_mm": core_spec["bounds_mm"]["z"],
            },
        )
        local_matcha = MATCHA_CAD.matcha_dock_stop_spec()
        matcha_bounds = clearance._bbox_record(
            clearance._bbox_tuple(MATCHA_CAD._dock_parts()["seating_stop"].val())
        )
        self.assertEqual(
            matcha_bounds,
            {
                "x_mm": local_matcha["bounds_mm"]["x"],
                "y_mm": local_matcha["bounds_mm"]["y"],
                "z_mm": local_matcha["bounds_mm"]["z"],
            },
        )
        self.assertNotEqual(core_spec["bounds_mm"], local_matcha["bounds_mm"])
        for bay in MATCHA_CAD.RACK_BAY_NAMES:
            bay_spec = MATCHA_CAD.matcha_dock_stop_spec(bay)
            self.assertEqual(
                bay_spec["center_mm"][0],
                local_matcha["center_mm"][0] + MATCHA_CAD.rack_bay_x(bay),
            )

    def test_stud_cam_inequalities_are_recomputed(self) -> None:
        result = clearance._stud_cam_inequality()
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["inequalities"].values()))
        self.assertTrue(math.isclose(result["stud_to_cam_x_margin_mm"], 9.05))
        self.assertTrue(math.isclose(result["unlocked_tab_to_cam_gap_mm"], 0.05))
        self.assertTrue(math.isclose(result["locked_tab_cam_engagement_mm"], 2.95))

    def test_authority_records_match_current_bytes(self) -> None:
        authorities = clearance._authorities_record()
        records = [
            value
            for key, value in authorities.items()
            if key != "core_exports"
        ] + authorities["core_exports"]
        self.assertGreaterEqual(len(records), 10)
        for record in records:
            path = clearance.REPO_ROOT / record["path"]
            self.assertEqual(record["bytes"], path.stat().st_size)
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_core_manifest_is_current_after_source_semantic_update(self) -> None:
        self.assertEqual(clearance.validate_core_manifest(), [])
        manifest = json.loads(clearance.CORE_MANIFEST_PATH.read_text())
        self.assertEqual(
            manifest["contracts"], clearance._expected_core_manifest_contracts()
        )
        inventory_payload = [
            {
                key: record[key]
                for key in ("path", "role", "bytes", "sha256")
            }
            for record in manifest["files"]
        ]
        self.assertEqual(
            manifest["inventory_sha256"],
            clearance._canonical_sha256(inventory_payload),
        )


class CoreDockFloorSupportSourceCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = clearance._core_dock_support_record()
        cls.contract = cls.record["contract"]

    def test_rolled_frame_and_release_roster_are_exact_and_hash_bound(self) -> None:
        frame = self.contract["frame"]
        self.assertEqual(
            frame["quat_wxyz"],
            [
                0.6440855284765126,
                -0.6440855284765125,
                0.2918112952014223,
                -0.2918112952014225,
            ],
        )
        self.assertTrue(frame["source_negative_y_is_world_up"])
        release = self.record["release"]
        self.assertEqual(release["row_count"], 31)
        self.assertEqual(
            release["roster_canonical_sha256"],
            "f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293",
        )
        self.assertTrue(
            math.isclose(
                release["maximum_joint_step_deg"],
                0.2360031832899985,
                abs_tol=1.0e-12,
            )
        )
        self.assertEqual(release["slider_q_mm"][-1], 3.0)

    def test_runtime_geometric_projection_is_exact_acyclic_and_closed(self) -> None:
        self.assertEqual(
            self.contract["schema_version"],
            "1.1-source-runtime-geometric-closure",
        )
        self.assertEqual(
            self.contract["blockers"],
            [
                "vendor_or_normative_source_missing_for_selected_M4_and_M6_fasteners",
                "floor_fixture_substrate_and_M6_thread_authority_missing",
                "PA12_modulus_strength_creep_and_process_allowables_unqualified",
                "printed_dimensional_tolerance_and_anchor_strength_unqualified",
                "cam_contact_friction_reverse_insertion_and_capture_dynamics_unvalidated",
            ],
        )
        self.assertNotIn(
            "runtime_placements_and_matcha_base_authority_are_stale",
            self.contract["blockers"],
        )
        self.assertNotIn(
            "full_compiled_arm_collision_screen_not_yet_regenerated_from_this_source",
            self.contract["blockers"],
        )

        binding = {
            "path": (
                "QuickChange/SO101_Magnetic/sim/"
                "rolled_core_dock_runtime_report.json"
            ),
            "required_schema_version": (
                "1.0-rolled-core-dock-runtime-authority"
            ),
            "method": (
                "actual_compiled_mesh_outer_AABB_to_actual_compiled_static_"
                "dock_geom_outer_AABB_with_topology_joint_motion_bound"
            ),
        }
        solver = self.contract["release_roster"]["solver_audit"]
        self.assertEqual(solver["runtime_report"], binding)
        self.assertEqual(
            solver["maximum_row_fk_position_error_mm"],
            1.3645432973688894e-13,
        )
        self.assertEqual(
            solver["maximum_row_fk_orientation_error_rad"],
            6.990106082579211e-16,
        )
        self.assertIs(solver["runtime_recomputation_pending"], False)
        self.assertIs(solver["passed"], True)

        screen = self.contract["geometry_audit"]["full_arm_screen"]
        self.assertEqual(screen["runtime_report"], binding)
        self.assertEqual(
            {
                key: screen[key]
                for key in clearance.ROLLED_CORE_DOCK_RUNTIME_EXPECTED[
                    "compiled_collision_sweep"
                ]
            },
            clearance.ROLLED_CORE_DOCK_RUNTIME_EXPECTED[
                "compiled_collision_sweep"
            ],
        )
        self.assertIs(screen["runtime_recomputation_pending"], False)
        self.assertIs(screen["geometric_clearance_authority"], True)
        self.assertIs(screen["physical_release_authority"], False)
        self.assertIs(screen["passed"], True)
        self.assertIn("robot_self_collision", screen["explicit_exclusions"])
        self.assertIn(
            "matcha_physical_base_or_floor_fixture_authority",
            screen["explicit_exclusions"],
        )
        self.assertFalse(
            any("sha" in key.lower() or "hash" in key.lower() for key in binding)
        )

        evidence = clearance._rolled_core_dock_runtime_geometric_evidence()
        self.assertTrue(evidence["passed"], evidence)
        self.assertTrue(all(evidence["checks"].values()), evidence)
        self.assertFalse(evidence["release_ready"])
        self.assertTrue(
            self.record["checks"][
                "rolled_runtime_geometric_evidence_closes"
            ]
        )
        self.assertTrue(self.record["engineering_checks_passed"])
        self.assertFalse(self.record["release_ready"])

    def test_runtime_geometric_projection_mutations_fail_closed(self) -> None:
        published = json.loads(
            clearance.ROLLED_CORE_DOCK_RUNTIME_REPORT_PATH.read_text()
        )

        def validate_payload(payload: object) -> dict[str, object]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rolled-runtime.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with mock.patch.object(
                    clearance,
                    "ROLLED_CORE_DOCK_RUNTIME_REPORT_PATH",
                    path,
                ):
                    return (
                        clearance._rolled_core_dock_runtime_geometric_evidence()
                    )

        mutations = {}
        mutations["coherently_shifted_clearance"] = copy.deepcopy(published)
        shifted = mutations["coherently_shifted_clearance"][
            "continuous_clearance"
        ]
        shifted["minimum_sampled_outer_aabb_lower_bound_mm"] += 1.0
        shifted["continuous_clearance_lower_bound_mm"] += 1.0

        mutations["resealed_state_and_evaluation_counts"] = copy.deepcopy(
            published
        )
        resealed = mutations["resealed_state_and_evaluation_counts"]
        resealed["sampling"]["unique_state_count"] = 302
        resealed["sampling"]["expected_unique_state_count"] = 302
        resealed["sampling"]["distance_evaluation_count"] = 380_520
        resealed_families = resealed["continuous_clearance"][
            "target_family_clearance"
        ]
        resealed_families["core_dock_fixture"][
            "distance_evaluation_count"
        ] = 334_012
        resealed_families["floor_support_proxy"][
            "distance_evaluation_count"
        ] = 46_508

        mutations["physical_authority_promotion"] = copy.deepcopy(published)
        mutations["physical_authority_promotion"]["release_ready"] = True
        mutations["physical_authority_promotion"]["authority_scope"][
            "physical_release_authority"
        ] = True

        mutations["malformed_family_count"] = copy.deepcopy(published)
        mutations["malformed_family_count"]["continuous_clearance"][
            "target_family_clearance"
        ]["core_dock_fixture"]["target_geom_count"] = "79"

        mutations["malformed_nested_mapping"] = copy.deepcopy(published)
        mutations["malformed_nested_mapping"]["continuous_clearance"][
            "target_family_clearance"
        ] = []

        mutations["row_count_float"] = copy.deepcopy(published)
        mutations["row_count_float"]["release_route"]["row_count"] = 31.0

        mutations["penetration_count_bool"] = copy.deepcopy(published)
        mutations["penetration_count_bool"]["startup"][
            "penetration_count"
        ] = False

        mutations["support_geom_count_float"] = copy.deepcopy(published)
        mutations["support_geom_count_float"]["continuous_clearance"][
            "target_family_clearance"
        ]["floor_support_proxy"]["target_geom_count"] = 11.0

        mutations["tangency_distance_string"] = copy.deepcopy(published)
        mutations["tangency_distance_string"]["support_topology"][
            "tangencies"
        ][0]["distance_m"] = "0.0"

        mutations["zero_distance_int"] = copy.deepcopy(published)
        mutations["zero_distance_int"]["startup"][
            "max_penetration_m"
        ] = 0

        mutations["contact_distance_string"] = copy.deepcopy(published)
        mutations["contact_distance_string"]["startup"][
            "contact_records"
        ][0]["signed_distance_m"] = "-1.4268967951647227e-13"

        mutations["startup_contact_distance_not_tangent"] = copy.deepcopy(
            published
        )
        mutations["startup_contact_distance_not_tangent"]["startup"][
            "contact_records"
        ][0]["signed_distance_m"] = 2.0e-9

        mutations["startup_contact_name_changed"] = copy.deepcopy(published)
        mutations["startup_contact_name_changed"]["startup"][
            "contact_records"
        ][0]["geom_a"] = "invented_startup_contact_geom"

        mutations["support_tangency_duplicated"] = copy.deepcopy(published)
        duplicated_tangencies = mutations["support_tangency_duplicated"][
            "support_topology"
        ]["tangencies"]
        duplicated_tangencies[1] = copy.deepcopy(duplicated_tangencies[0])

        mutations["support_tangency_fake_pair"] = copy.deepcopy(published)
        mutations["support_tangency_fake_pair"]["support_topology"][
            "tangencies"
        ][0]["first"] = "invented_support_tangency_geom"

        mutations["support_tangency_distance_not_zero"] = copy.deepcopy(
            published
        )
        mutations["support_tangency_distance_not_zero"][
            "support_topology"
        ]["tangencies"][0]["distance_m"] = 2.0e-12

        mutations["core_family_clearance_resealed"] = copy.deepcopy(published)
        core_family = mutations["core_family_clearance_resealed"][
            "continuous_clearance"
        ]["target_family_clearance"]["core_dock_fixture"]
        core_family["minimum_sampled_outer_aabb_lower_bound_mm"] = -100.0
        core_family["continuous_clearance_lower_bound_mm"] = -100.0
        core_family["required_clearance_mm"] = 100.0
        core_family["passed"] = True

        mutations["support_required_clearance_resealed"] = copy.deepcopy(
            published
        )
        mutations["support_required_clearance_resealed"][
            "continuous_clearance"
        ]["target_family_clearance"]["floor_support_proxy"][
            "required_clearance_mm"
        ] = -100.0

        mutations["top_level_bool_as_int"] = copy.deepcopy(published)
        mutations["top_level_bool_as_int"]["geometry_passed"] = 1

        for field, expected in (
            clearance.ROLLED_CORE_DOCK_RUNTIME_AUTHORITY_FLAGS.items()
        ):
            mutated = copy.deepcopy(published)
            mutated["authority_scope"][field] = not expected
            mutations[f"authority_scope_{field}"] = mutated

        mutations["unknown_authority_key"] = copy.deepcopy(published)
        mutations["unknown_authority_key"]["authority_scope"][
            "fabrication_authority"
        ] = True

        mutations["missing_authority_key"] = copy.deepcopy(published)
        mutations["missing_authority_key"]["authority_scope"].pop(
            "mass_authority"
        )

        for field in (
            clearance.ROLLED_CORE_DOCK_RUNTIME_SUPPORT_PROXY_FALSE_AUTHORITIES
        ):
            mutated = copy.deepcopy(published)
            mutated["support_proxy"][field] = True
            mutations[f"support_proxy_{field}"] = mutated

        mutations["support_proxy_unknown_authority_key"] = copy.deepcopy(
            published
        )
        mutations["support_proxy_unknown_authority_key"]["support_proxy"][
            "fabrication_authority"
        ] = True

        mutations["passage_witness_inside_proxy"] = copy.deepcopy(published)
        mutations["passage_witness_inside_proxy"]["support_proxy"][
            "passage_witness_inside_proxy"
        ] = True

        mutations["release_in_default_controller_actions"] = copy.deepcopy(
            published
        )
        mutations["release_in_default_controller_actions"]["release_route"][
            "included_in_default_controller_actions"
        ] = True

        mutations["static_release_in_default_actions"] = copy.deepcopy(
            published
        )
        mutations["static_release_in_default_actions"]["default_actions"][
            "static_release_continuation_included"
        ] = True

        mutations["rack_exit_rejection_disabled"] = copy.deepcopy(published)
        mutations["rack_exit_rejection_disabled"]["default_actions"][
            "rack_exit_flag_rejected"
        ] = False

        mutations["gravity_feedforward_dynamics_authority"] = copy.deepcopy(
            published
        )
        mutations["gravity_feedforward_dynamics_authority"][
            "gravity_feedforward"
        ]["dynamics_authority"] = True

        for name, mutated in mutations.items():
            with self.subTest(name=name):
                result = validate_payload(mutated)
                self.assertFalse(result["passed"], result)
                self.assertFalse(result["release_ready"])

        source_projection_mutations = {}
        source_projection_mutations["source_row_count_float"] = copy.deepcopy(
            clearance.ROLLED_CORE_DOCK_RUNTIME_EXPECTED
        )
        source_projection_mutations["source_row_count_float"][
            "release_roster"
        ]["row_count"] = 31.0
        source_projection_mutations["source_penetration_count_bool"] = (
            copy.deepcopy(clearance.ROLLED_CORE_DOCK_RUNTIME_EXPECTED)
        )
        source_projection_mutations["source_penetration_count_bool"][
            "compiled_collision_sweep"
        ]["startup_penetration_count"] = False
        for name, source_projection in source_projection_mutations.items():
            with self.subTest(name=name), mock.patch.object(
                clearance,
                "_rolled_core_dock_source_projection",
                return_value=source_projection,
            ):
                result = validate_payload(published)
                self.assertFalse(result["passed"], result)
                self.assertFalse(result["release_ready"])

        digest_free = copy.deepcopy(published)
        digest_free.pop("source_binding")
        for key in list(digest_free["release_route"]):
            if "sha256" in key:
                digest_free["release_route"].pop(key)
        digest_free_result = validate_payload(digest_free)
        self.assertTrue(digest_free_result["passed"], digest_free_result)
        self.assertNotIn(
            "sha256",
            json.dumps(digest_free_result["observed_projection"]),
        )

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with mock.patch.object(
                clearance,
                "ROLLED_CORE_DOCK_RUNTIME_REPORT_PATH",
                missing,
            ):
                result = (
                    clearance._rolled_core_dock_runtime_geometric_evidence()
                )
        self.assertFalse(result["available"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["release_ready"])

        deep_json = '{"nested":' * 10_000 + "null" + "}" * 10_000
        with tempfile.TemporaryDirectory() as directory:
            deep = Path(directory) / "deep.json"
            deep.write_text(deep_json, encoding="utf-8")
            with mock.patch.object(
                clearance,
                "ROLLED_CORE_DOCK_RUNTIME_REPORT_PATH",
                deep,
            ):
                result = (
                    clearance._rolled_core_dock_runtime_geometric_evidence()
                )
        self.assertFalse(result["available"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["release_ready"])

    def test_exact_brep_has_positive_internal_unions_and_fastener_holes(self) -> None:
        brep = self.record["brep"]
        self.assertEqual(brep["support"]["solid_count"], 1)
        self.assertTrue(brep["support"]["valid"])
        self.assertTrue(
            math.isclose(
                brep["support"]["volume_mm3"],
                162415.4180526403,
                abs_tol=1.0e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                brep["dock"]["volume_mm3"],
                21743.904784962568,
                abs_tol=1.0e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                brep["installed_printed_volume_mm3"],
                184159.32283760287,
                abs_tol=1.0e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                brep["head_post_overlap_after_pockets_mm3"],
                716.8886804667261,
                abs_tol=1.0e-9,
            )
        )
        self.assertTrue(
            math.isclose(brep["post_base_overlap_mm3"], 1248.0, abs_tol=1.0e-9)
        )
        for records in self.record["hole_provenance"].values():
            self.assertTrue(records)
            removal_keys = [key for key in records[0] if key.startswith("removed_from_")]
            self.assertEqual(len(removal_keys), 1)
            self.assertTrue(all(record[removal_keys[0]] > 0.2 for record in records))

    def test_stop_is_bolted_not_allowlisted_and_other_dock_features_clear(self) -> None:
        distances = self.record["brep"]["fixed_dock_distances_mm"]
        overlaps = self.record["brep"]["fixed_dock_overlap_volumes_mm3"]
        self.assertEqual(distances["seating_stop"], 0.0)
        self.assertEqual(overlaps["seating_stop"], 0.0)
        self.assertEqual(distances["right_upper_rail"], 1.0)
        for name, overlap in overlaps.items():
            if name != "seating_stop":
                self.assertLessEqual(
                    overlap, clearance.OVERLAP_VOLUME_TOLERANCE_MM3, name
                )
        self.assertTrue(
            self.record["checks"][
                "stop_is_explicit_fastened_zero_distance_not_allowlist"
            ]
        )

    def test_floor_closure_tolerance_and_load_proxy_are_bound_but_not_released(self) -> None:
        floor = self.record["floor_closure"]
        self.assertLessEqual(abs(floor["base_world_bounds"]["z_m"][0]), 1.0e-12)
        self.assertEqual(len(floor["anchor_centres_world_m"]), 4)
        self.assertTrue(
            all(point[2] == 0.0 for point in floor["anchor_centres_world_m"])
        )
        self.assertTrue(self.record["engineering_checks_passed"])
        self.assertFalse(self.record["release_ready"])
        self.assertIn(
            "floor_fixture_substrate_and_M6_thread_authority_missing",
            self.record["blockers"],
        )
        self.assertIn(
            "cam_contact_friction_reverse_insertion_and_capture_dynamics_unvalidated",
            self.record["blockers"],
        )
        self.assertFalse(self.contract["tolerance_budget"]["dimensionally_qualified"])
        self.assertTrue(
            math.isclose(
                self.contract["load_proxy"]["combined_moment_Nm"],
                4.25271213611,
                abs_tol=1.0e-12,
            )
        )
        self.assertIsNone(self.contract["printed_brep"]["mass_claim"])


@unittest.skipUnless(
    _published_report_matches_current_validator(),
    "published CAD-clearance report is absent or intentionally stale in this source checkpoint",
)
class PublishedCoreCadClearanceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(clearance.REPORT_PATH.read_text())

    def test_published_report_revalidates_against_current_sources(self) -> None:
        self.assertEqual(clearance.validate_report(self.report), [])

    def test_report_verdict_is_derived_even_when_release_is_red(self) -> None:
        self.assertEqual(self.report["release_ready"], not self.report["blockers"])
        self.assertEqual(self.report["passed"], self.report["release_ready"])

    def test_tampered_cam_verdict_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.report)
        cam = next(
            result
            for result in tampered["withdrawal_sweep"]["results"]
            if result["component"] == "robot_plate"
            and result["dock_component"] == "positive_lock_cam"
        )
        cam["passed"] = not cam["passed"]
        tampered["validation"][
            "machine_json_canonical_sha256_without_this_field"
        ] = None
        errors = clearance.validate_report(tampered)
        self.assertTrue(
            any(error.startswith("path_result_verdict_mismatch:") for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
