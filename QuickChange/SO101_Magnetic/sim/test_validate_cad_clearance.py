#!/usr/bin/env python3
"""Focused regressions for the core exact CAD-clearance authority."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

import cadquery as cq

import validate_cad_clearance as clearance


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
        self.assertEqual(contract["tool_local_fixed_step_pos_mm"], [0.4875, 0.218, 9.551])
        self.assertEqual(contract["tool_local_fixed_step_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(contract["source_function_sha256"]), 64)
        self.assertTrue(math.isclose(
            contract["robot_assembly_fixed_step_pos_mm"][2]
            - contract["robot_assembly_stock_plate_pos_mm"][2],
            9.551,
            abs_tol=1.0e-12,
        ))

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


@unittest.skipUnless(
    clearance.REPORT_PATH.is_file(),
    "published CAD-clearance report not present in this source-only checkpoint",
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
        cam["passed"] = True
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
