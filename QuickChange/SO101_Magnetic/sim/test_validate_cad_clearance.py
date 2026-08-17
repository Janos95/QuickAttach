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
import unittest

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
        self.assertEqual(result["guided_open_side_offset_mm"], 0.20)
        self.assertTrue(
            math.isclose(result["minimum_sampled_distance_mm"], 0.30, abs_tol=1.0e-12)
        )
        self.assertTrue(
            math.isclose(
                result["continuous_certified_clearance_mm"],
                0.25,
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
                5.135937500000003,
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
        self.assertGreaterEqual(
            preservation["retained_volume_fraction"],
            clearance.ROBOT_PLATE_MINIMUM_RETAINED_VOLUME_FRACTION,
        )
        self.assertTrue(
            math.isclose(
                preservation["removed_volume_mm3"],
                106.30309519747607,
                abs_tol=1.0e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                preservation["relief_contract"][
                    "minimum_relief_to_stud_well_ligament_mm"
                ],
                8.225,
                abs_tol=1.0e-12,
            )
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

    def test_core_manifest_closes_exports_and_physical_contracts(self) -> None:
        self.assertEqual(clearance.validate_core_manifest(), [])
        manifest = json.loads(clearance.CORE_MANIFEST_PATH.read_text())
        expected_paths = [
            f"QuickChange/SO101_Magnetic/exports/{name}"
            for name in sorted(clearance.CAD.CORE_OUTPUT_NAMES)
        ]
        self.assertEqual(
            [record["path"] for record in manifest["files"]],
            expected_paths,
        )
        self.assertEqual(manifest["file_count"], len(expected_paths))
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
