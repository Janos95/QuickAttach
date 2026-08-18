#!/usr/bin/env python3
"""Focused adversarials for the positive-lock cam load/timing authority."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest

import validate_positive_lock_cam_load_authority as authority


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _common_qualification(method: str) -> dict[str, object]:
    return {
        "qualified": True,
        "source_sha256": authority._sha256(authority.CAD_GENERATOR_PATH),
        "qualification_report_sha256": _digest(method),
        "method": method,
    }


def _complete_evidence(
    *, mu_min: float = 0.10, mu_max: float = 0.15
) -> dict[str, object]:
    """Return synthetic complete evidence used only to exercise all gates."""

    evidence: dict[str, object] = {
        "slope_tolerance_qualification": {
            **_common_qualification("measured_dimensional_bounds"),
            "capture": {
                "run_absolute_tolerance_mm": 0.001,
                "rise_absolute_tolerance_mm": 0.001,
            },
            "return": {
                "run_absolute_tolerance_mm": 0.001,
                "rise_absolute_tolerance_mm": 0.001,
            },
        },
        "friction_qualification": {
            **_common_qualification("dry_material_pair_tribometer_envelope"),
            "mu_min": mu_min,
            "mu_max": mu_max,
            "material_pair": "synthetic_test_pair",
            "environment_envelope": "synthetic_test_temperature_humidity_load",
        },
        "timing_tolerance_qualification": {
            **_common_qualification("capture_stack_interval_analysis"),
            "fully_open_preseat_absolute_tolerance_mm": 0.01,
            "head_entry_absolute_tolerance_mm": 0.01,
            "includes_slope_interval_propagation": True,
        },
    }
    preliminary = authority.build_report(evidence=evidence)
    load_case_sha = _digest("synthetic_adverse_load_case")
    evidence["finite_contact_patch_evidence"] = {
        **_common_qualification("finite_patch_nonlinear_analysis"),
        "contact_patch_method": "finite_patch_lower_bound_from_nonlinear_fea",
        "uses_full_source_face_area": False,
        "finite_patch_lower_bound": True,
        "minimum_contact_area_mm2": 1.0,
        "allowable_contact_pressure_mpa": 100.0,
        "required_safety_factor": 2.0,
        "load_case_sha256": load_case_sha,
    }
    evidence["root_strength_evidence"] = {
        **_common_qualification("exact_root_section_analysis"),
        "geometry_method": "hash_pinned_exact_solid_section",
        "minimum_root_ligament_width_mm": 1.0,
        "root_thickness_mm": 1.0,
        "minimum_section_modulus_mm3": 1.0,
        "maximum_bending_stress_mpa": 10.0,
        "allowable_bending_stress_mpa": 100.0,
        "required_safety_factor": 2.0,
        "load_case_sha256": load_case_sha,
    }
    direction_records: dict[str, object] = {}
    for name, force in preliminary["mechanics"][
        "route_force_requirements_n"
    ].items():
        if force is None:
            raise AssertionError("synthetic evidence requires a finite force case")
        direction_records[name] = {
            "swept_every_route_state": True,
            "uses_actual_slider_tab_contact_point": True,
            "includes_payload_gravity": True,
            "includes_commanded_acceleration": True,
            "route_samples_sha256": _digest(name),
            "applied_cam_force_bound_n": float(force) + 1.0,
            "joint_torque_demand_nm": [0.5] * 6,
            "joint_torque_limit_nm": [3.35] * 6,
        }
    evidence["route_torque_evidence"] = {
        **_common_qualification("full_route_static_dynamic_torque_sweep"),
        "model_sha256": _digest("model"),
        "payload_sha256": _digest("payload"),
        "load_case_sha256": load_case_sha,
        "directions": direction_records,
    }
    return evidence


class PositiveLockCamLoadAuthorityTests(unittest.TestCase):
    def test_current_source_is_explicitly_red_and_quantifies_self_lock(self) -> None:
        report = authority.build_report()
        self.assertFalse(report["release_ready"])
        self.assertFalse(report["passed"])
        self.assertIn("friction_interval_unqualified", report["blockers"])
        self.assertIn("passive_return_self_lock_risk", report["blockers"])
        self.assertIn("reverse_insertion_jam_risk", report["blockers"])
        self.assertIn(
            "finite_contact_patch_evidence_missing_or_invalid",
            report["blockers"],
        )
        self.assertIn(
            "route_torque_provenance_missing_or_invalid", report["blockers"]
        )
        slopes = report["source_nominal_slopes"]
        self.assertAlmostEqual(slopes["capture_nominal"], 1.0)
        self.assertAlmostEqual(slopes["return_nominal"], 0.246875)
        return_case = report["mechanics"]["passive_return_negative_y"]
        self.assertAlmostEqual(
            return_case["self_lock_margin_k_minus_mu"], -0.003125
        )
        self.assertFalse(return_case["jam_free"])
        self.assertIsNone(
            report["mechanics"]["reverse_insertion_positive_y"][
                "required_drive_force_n"
            ]
        )

    def test_nominal_opening_is_before_p31_but_tolerance_authority_is_missing(
        self,
    ) -> None:
        timing = authority.build_report()["opening_timing"]
        self.assertAlmostEqual(timing["fully_open_preseat_mm"], 3.2)
        self.assertAlmostEqual(timing["head_entry_tangent_preseat_mm"], 3.1)
        self.assertAlmostEqual(
            timing["nominal_open_before_head_margin_mm"], 0.1
        )
        self.assertTrue(timing["nominal_ordering_passed"])
        self.assertFalse(timing["qualified"])
        self.assertFalse(timing["passed"])

    def test_complete_synthetic_evidence_exercises_a_possible_green_path(self) -> None:
        report = authority.build_report(evidence=_complete_evidence())
        self.assertTrue(report["release_ready"], report)
        self.assertEqual(report["blockers"], [])
        self.assertTrue(all(report["checks"].values()))

    def test_adverse_friction_uses_mu_max_and_resisting_signs(self) -> None:
        evidence = _complete_evidence(mu_min=0.10, mu_max=0.20)
        report = authority.build_report(evidence=evidence)
        mechanics = report["mechanics"]
        capture = mechanics["capture"]
        k_capture = report["slope_intervals"]["capture"]["upper_slope"]
        spring = mechanics["spring"]["maximum_force_n"]
        expected_capture = spring * (k_capture + 0.20) / (
            1.0 - 0.20 * k_capture
        )
        self.assertAlmostEqual(capture["axial_force_n"], expected_capture)
        self.assertEqual(capture["formula_axial"], "Fs*(k+mu)/(1-mu*k)")

        passive = mechanics["passive_return_negative_y"]
        k_return = report["slope_intervals"]["return"]["lower_slope"]
        self.assertAlmostEqual(
            passive["self_lock_margin_k_minus_mu"], k_return - 0.20
        )
        self.assertEqual(passive["formula"], "Fs*(k-mu)/(1+mu*k)")
        self.assertNotAlmostEqual(
            passive["self_lock_margin_k_minus_mu"], k_return - 0.10
        )

    def test_reversed_friction_interval_is_rejected_not_silently_swapped(self) -> None:
        evidence = _complete_evidence()
        friction = evidence["friction_qualification"]
        assert isinstance(friction, dict)
        friction["mu_min"] = 0.25
        friction["mu_max"] = 0.20
        report = authority.build_report(evidence=evidence)
        self.assertFalse(report["friction"]["qualified"])
        self.assertIn(
            "mu_interval_not_strictly_increasing", report["friction"]["errors"]
        )
        self.assertEqual(
            report["friction"]["analysis_interval"],
            list(authority.SCREENING_MU_INTERVAL),
        )
        self.assertFalse(report["release_ready"])

    def test_nominal_only_slope_claim_is_not_a_tolerance_interval(self) -> None:
        evidence = _complete_evidence()
        slope = evidence["slope_tolerance_qualification"]
        assert isinstance(slope, dict)
        for name in ("capture", "return"):
            item = slope[name]
            assert isinstance(item, dict)
            item["run_absolute_tolerance_mm"] = 0.0
            item["rise_absolute_tolerance_mm"] = 0.0
        report = authority.build_report(evidence=evidence)
        self.assertFalse(report["slope_intervals"]["capture"]["qualified"])
        self.assertFalse(report["slope_intervals"]["return"]["qualified"])
        self.assertTrue(report["slope_intervals"]["capture"]["nominal_only"])
        self.assertIn(
            "capture_slope_tolerance_unqualified", report["blockers"]
        )
        self.assertIn("return_slope_tolerance_unqualified", report["blockers"])

    def test_nonpositive_return_denominator_cannot_be_omitted_or_overridden(
        self,
    ) -> None:
        evidence = _complete_evidence()
        friction = evidence["friction_qualification"]
        assert isinstance(friction, dict)
        friction["mu_min"] = 0.24
        friction["mu_max"] = 0.25
        friction["claimed_reverse_denominator"] = 1.0
        friction["claimed_reverse_force_n"] = 1.0
        report = authority.build_report(evidence=evidence)
        reverse = report["mechanics"]["reverse_insertion_positive_y"]
        self.assertLess(reverse["denominator_margin_k_minus_mu"], 0.0)
        self.assertIsNone(reverse["required_drive_force_n"])
        self.assertIsNone(reverse["normal_force_n"])
        self.assertIn("reverse_insertion_jam_risk", report["blockers"])
        self.assertFalse(report["release_ready"])

    def test_omitted_reverse_route_direction_fails_closed(self) -> None:
        evidence = _complete_evidence()
        route = evidence["route_torque_evidence"]
        assert isinstance(route, dict)
        directions = route["directions"]
        assert isinstance(directions, dict)
        del directions["reverse_insertion_positive_y"]
        report = authority.build_report(evidence=evidence)
        self.assertFalse(report["route_torque"]["passed"])
        self.assertTrue(
            any(
                error.startswith("missing_route_directions:")
                and "reverse_insertion_positive_y" in error
                for error in report["route_torque"]["errors"]
            )
        )
        self.assertIn(
            "route_torque_provenance_missing_or_invalid", report["blockers"]
        )

    def test_line_contact_and_full_face_mean_pressure_are_both_rejected(self) -> None:
        for method, area, uses_full_face in (
            ("line_contact", 0.0, False),
            (
                "full_source_face_mean_pressure",
                authority.CAD.positive_lock_cam_contract()[
                    "quasistatic_load_envelope"
                ]["contact_face_area_mm2"],
                True,
            ),
        ):
            with self.subTest(method=method):
                evidence = _complete_evidence()
                contact = evidence["finite_contact_patch_evidence"]
                assert isinstance(contact, dict)
                contact["contact_patch_method"] = method
                contact["minimum_contact_area_mm2"] = area
                contact["uses_full_source_face_area"] = uses_full_face
                report = authority.build_report(evidence=evidence)
                self.assertFalse(report["finite_contact_patch"]["passed"])
                self.assertIn(
                    "finite_contact_patch_evidence_missing_or_invalid",
                    report["blockers"],
                )

    def test_missing_strength_and_route_provenance_remain_separate_blockers(
        self,
    ) -> None:
        evidence = _complete_evidence()
        root = evidence["root_strength_evidence"]
        route = evidence["route_torque_evidence"]
        assert isinstance(root, dict) and isinstance(route, dict)
        root.pop("qualification_report_sha256")
        root.pop("load_case_sha256")
        route.pop("model_sha256")
        route.pop("payload_sha256")
        report = authority.build_report(evidence=evidence)
        self.assertFalse(report["root_strength"]["passed"])
        self.assertFalse(report["route_torque"]["passed"])
        self.assertIn(
            "root_strength_evidence_missing_or_invalid", report["blockers"]
        )
        self.assertIn(
            "route_torque_provenance_missing_or_invalid", report["blockers"]
        )

    def test_report_mutation_is_rejected_by_fresh_recomputation(self) -> None:
        evidence = _complete_evidence()
        report = authority.build_report(evidence=evidence)
        self.assertEqual(authority.validate_report(report, evidence=evidence), [])
        mutated = copy.deepcopy(report)
        mutated["mechanics"]["capture"]["axial_force_n"] = 0.0
        self.assertEqual(
            authority.validate_report(mutated, evidence=evidence),
            ["report_recomputation_mismatch"],
        )


if __name__ == "__main__":
    unittest.main()
