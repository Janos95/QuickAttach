#!/usr/bin/env python3
"""Focused source-level tests for the exact matcha rack report generator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_matcha_rack as rack_validation  # noqa: E402


class MatchaRackValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = rack_validation.build_report(require_exports=False)

    def test_complete_pair_inventory_and_explicit_gripper_scope_boundary(self) -> None:
        report = self.report
        self.assertTrue(report["tool_validation_passed"])
        self.assertEqual(
            report["release_ready"], report["input_artifacts"]["closure_passed"]
        )
        self.assertEqual(report["passed"], report["release_ready"])
        self.assertEqual(report["collision_pair_closure"]["expected_count"], 1026)
        self.assertEqual(report["collision_pair_closure"]["evaluated_count"], 1026)
        self.assertEqual(report["collision_pair_closure"]["failed_count"], 0)
        self.assertTrue(report["collision_pair_closure"]["passed"])
        self.assertNotIn(
            "stock_gripper_keeper_geometry_unresolved",
            [blocker["code"] for blocker in report["blockers"]],
        )
        scope = report["scope_boundary"]
        self.assertEqual(scope["matcha_rack_bays"], ["spoon", "whisk"])
        self.assertFalse(scope["normal_gripper_included_in_matcha_rack"])
        self.assertEqual(len(scope["normal_gripper_authority_files"]), 5)
        self.assertEqual(rack_validation.validate_report_structure(report), [])

    def test_only_six_exact_named_intended_contacts_exist(self) -> None:
        intended = [
            record
            for record in self.report["pair_results"]
            if record["semantic_evaluation"].startswith("intended_printed_plate_")
        ]
        self.assertEqual(len(intended), 6)
        expected = {
            ("spoon", "common_tool_plate", "dock_spoon_left_lower_ledge"),
            ("spoon", "common_tool_plate", "dock_spoon_right_lower_ledge"),
            ("spoon", "common_tool_plate", "dock_spoon_seating_stop"),
            ("whisk", "common_tool_plate", "dock_whisk_left_lower_ledge"),
            ("whisk", "common_tool_plate", "dock_whisk_right_lower_ledge"),
            ("whisk", "common_tool_plate", "dock_whisk_seating_stop"),
        }
        self.assertEqual(
            {
                (record["tool"], record["tool_component"], record["rack_component"])
                for record in intended
            },
            expected,
        )
        for record in intended:
            self.assertFalse(record["fcpw_screen"]["clearance_authority"])
            self.assertTrue(record["occt_critical_diagnostic"]["clearance_authority"])
            self.assertLessEqual(
                record["maximum_overlap_volume_mm3"],
                rack_validation.OVERLAP_VOLUME_TOLERANCE_MM3,
            )
            self.assertGreaterEqual(
                record["outside_window_clearance"]["minimum_exact_clearance_mm"],
                rack_validation.MANUFACTURING_CLEARANCE_MM,
            )

    def test_all_other_pairs_have_continuous_clearance(self) -> None:
        forbidden = [
            record
            for record in self.report["pair_results"]
            if record["semantic_evaluation"] == "forbidden_pair_continuous_clearance"
        ]
        self.assertEqual(len(forbidden), 1020)
        self.assertFalse(
            any(
                record["semantic_evaluation"] == "unresolved_forbidden_pair"
                for record in self.report["pair_results"]
            )
        )
        self.assertTrue(
            all(
                record["minimum_signed_clearance_mm"]
                + rack_validation.NUMERIC_TOLERANCE_MM
                >= rack_validation.MANUFACTURING_CLEARANCE_MM
                for record in forbidden
            )
        )

    def test_whisk_extrema_and_reverse_path_are_bound(self) -> None:
        whisk = self.report["tool_path_results"]["matcha_whisk"]
        states = {
            (state["eccentric_x_mm"], state["compliance_z_mm"])
            for state in whisk["mechanism_states"]
        }
        for x in (-4.0, 0.0, 4.0):
            for z in (-5.05, 0.05):
                self.assertIn((x, z), states)
        path = whisk["straight_y_path"]
        self.assertEqual(path["sample_count_per_mechanism_state"], 801)
        self.assertEqual(path["maximum_between_sample_motion_bound_mm"], 0.05)
        self.assertTrue(path["reverse_path_set_equivalent"])

    def test_validator_rejects_mutated_pair_evidence(self) -> None:
        mutated = deepcopy(self.report)
        record = next(
            item
            for item in mutated["pair_results"]
            if item["semantic_evaluation"] == "forbidden_pair_continuous_clearance"
        )
        record["minimum_signed_clearance_mm"] = -1.0
        errors = rack_validation.validate_report_structure(mutated)
        self.assertIn("forbidden_pair_below_manufacturing_clearance", errors)

        missing = deepcopy(self.report)
        missing["pair_results"].pop()
        errors = rack_validation.validate_report_structure(missing)
        self.assertIn("pair_evaluated_count_not_recomputed", errors)
        self.assertIn("pair_inventory_digest_not_recomputed", errors)


if __name__ == "__main__":
    unittest.main()
