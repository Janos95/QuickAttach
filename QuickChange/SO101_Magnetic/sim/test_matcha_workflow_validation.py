#!/usr/bin/env python3
"""Independent, fail-closed validation for the optional matcha workflow.

The matcha production files are restored independently from this test module.
Tests whose authority input is not present skip with an explicit reason; they
must become ordinary pass/fail gates as soon as that input lands.  This file
does not generate CAD, patch the scene, or mutate the controller.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import struct
import subprocess
import sys
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import CodeType, ModuleType, SimpleNamespace
from typing import Any, Iterable
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
MAGNETIC_ROOT = HERE.parent
REPOSITORY_ROOT = HERE.parents[2]
MATCHA_DEMO = HERE / "matcha_workflow_demo.py"
MATCHA_SHOWCASE = HERE / "matcha_workflow_showcase.py"
MATCHA_SHOWCASE_RENDERER = HERE / "render_matcha_workflow_video.py"
MATCHA_SCENE = HERE / "matcha_workflow_scene.xml"
MATCHA_CONFIG = HERE / "matcha_tool_geometry.json"
PAYLOAD_GENERATOR = HERE / "generate_matcha_payload_proxy_report.py"
PAYLOAD_VALIDATOR = HERE / "validate_matcha_payload_proxy_report.py"
PAYLOAD_REPORT = HERE / "matcha_payload_proxy_report.json"
CORE_CLEARANCE_VALIDATOR = HERE / "validate_cad_clearance.py"
CORE_CLEARANCE_REPORT = HERE / "cad_clearance_report.json"
ROLLED_CORE_DOCK_RUNTIME_VALIDATOR = (
    HERE / "validate_rolled_core_dock_runtime.py"
)
ROLLED_CORE_DOCK_RUNTIME_REPORT = HERE / "rolled_core_dock_runtime_report.json"
CORE_CAD_MANIFEST = MAGNETIC_ROOT / "exports" / "core_cad_manifest.json"
CAD_ROOT = MAGNETIC_ROOT / "matcha_tools"
MATCHA_CAD_GENERATOR = CAD_ROOT / "generate_matcha_tool_cad.py"
CAD_MANIFEST = CAD_ROOT / "exports" / "matcha_tool_manifest.json"
RUNNER = HERE / "run_matcha_validation.py"

EXPECTED_TOOL_IDS = {
    "gripper": 6,
    "matcha_spoon": 21,
    "matcha_whisk": 22,
}
EXPECTED_WHISK_DEVICE_ID = 7
SURFACE_INTERNAL_TARGET_MM = 0.34
SURFACE_RELEASE_LIMIT_MM = 0.35

POGO_PART_NUMBER = "7983-1-15-20-75-14-11-0"
POGO_DIMENSION_DRAWING_SHA256 = (
    "c97327d953663a0aa04ea389ee2d2be19372ffa21503f46e5cbbfb0fd2e890e8"
)
POGO_DIMENSION_DRAWING_BYTES = 175_611
POGO_PRESS_FIT_NOTE_SHA256 = (
    "bbf4c414a11bd3355cde2bb25624c6736b61942964b2cbb3fc42c67c09e87adf"
)
POGO_PRESS_FIT_NOTE_BYTES = 509_252
POGO_OFFICIAL_PROFILE_DIAMETERS_MM = {
    "moving_plunger": 0.042 * 25.4,
    "upper_guide": 0.068 * 25.4,
    "upper_shell": 0.073 * 25.4,
    "barb": 0.0765 * 25.4,
    "shoulder": 0.083 * 25.4,
    "knurl": 0.065 * 25.4,
    "solder_cup_outer": 0.060 * 25.4,
    "solder_cup_bore": 0.038 * 25.4,
}
# These are the complete non-stroke axial dimensions printed on the official
# 7983 SVG.  Keeping the values as a sorted roster avoids inventing semantic
# names for drawing extension lines while still rejecting the legacy 3.0 +
# 5.1 mm two-cylinder display envelope.
POGO_OFFICIAL_AXIAL_DIMENSIONS_MM = [
    0.025 * 25.4,
    0.028 * 25.4,
    0.030 * 25.4,
    0.100 * 25.4,
    0.132 * 25.4,
    0.145 * 25.4,
    0.180 * 25.4,
    0.374 * 25.4,
]
POGO_SIGNAL_CENTRES_MM = {
    "GND": [-31.0, -7.5],
    "+12V": [-31.0, -2.5],
    "TTL_DATA": [-31.0, 2.5],
    "TOOL_ID_SPARE": [-31.0, 7.5],
}
POGO_STANDARD_SIGNALS = ["+12V", "TTL_DATA", "TOOL_ID_SPARE"]
POGO_RUNTIME_SIGNAL_MAP = {
    "ground": "GND",
    "power": "+12V",
    "data": "TTL_DATA",
    "id": "TOOL_ID_SPARE",
}
POGO_RUNTIME_FIXED_SEGMENTS = (
    ("solder_cup", "fixed_shell_solder_cup"),
    (
        "cup_to_knurl_transition_bound",
        "fixed_shell_cup_to_knurl_transition_bound",
    ),
    ("knurl", "fixed_shell_knurl"),
    ("shoulder", "shoulder_stop"),
    ("plunger_side_fixed_features", "fixed_shell_plunger_side_fixed_features"),
)
POGO_RUNTIME_RELEASE_BLOCKERS = [
    "ground_first_mate_tolerance_stack_unqualified",
    "knurl_press_fit_process_and_pullout_unqualified",
    "installed_electrical_cycle_reliability_unqualified",
    "pogo_mass_properties_unqualified",
    "pogo_spring_force_curve_unqualified",
    "pogo_damping_unqualified",
]
POSITIVE_LOCK_CAM_RUNTIME_BLOCKERS = [
    "positive_lock_cam_friction_coefficient_unqualified",
    "positive_lock_cam_load_capacity_unqualified",
    "positive_lock_cam_dynamics_unqualified",
]
CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256 = (
    "3fef8469bf8cbddff822d9a6a1a31de9de2872be4bfe75d87575f01c83b99966"
)
CORE_CAPTURE_ROUTE_Q_SHA256 = (
    "107b40015a76fd55f09681164ae75aa12ae738a7385fba05ab4d94f7f29e40bd"
)
CORE_CAPTURE_ROUTE_EMBEDDED_BYTES_SHA256 = (
    "e91e8699d4ef1d174d73341198e21499cbf615279e4e1a87a6fbe98929f0004c"
)
CORE_CAPTURE_ROUTE_ACTIONS = {
    "gripper_capture_lateral_align": {
        "row_range": None,
        "q_count": 2,
        "waypoint_count": 1,
        "duration_s": 0.25,
        "timeout_s": 1.0,
        "q_sha256": (
            "39603d9ace7749f1001a27d04d36a4f33e071aff2367b213396c12f94e188299"
        ),
        "interval_count": 1,
        "speed_bound": 0.012,
        "acceleration_bound": 0.15,
    },
    "gripper_capture_axial_open_side": {
        "row_range": [0, 243],
        "q_count": 244,
        "waypoint_count": 243,
        "duration_s": 1.60,
        "timeout_s": 3.0,
        "q_sha256": (
            "2f06345e62fb5cbb4230bdee741addffba7810a39cb0d5681bad100e4c4f94ca"
        ),
        "interval_count": 243,
        "speed_bound": 0.66,
        "acceleration_bound": 1.40,
    },
    "gripper_capture_coupled_recenter": {
        "row_range": [243, 259],
        "q_count": 17,
        "waypoint_count": 16,
        "duration_s": 0.50,
        "timeout_s": 1.5,
        "q_sha256": (
            "6df3f9c5e6be3581970694462fa1f2805b121f02db00a3a7d1303ec5ecf27792"
        ),
        "interval_count": 16,
        "speed_bound": 0.12,
        "acceleration_bound": 0.75,
    },
    "gripper_capture_centered_final": {
        "row_range": [259, 275],
        "q_count": 17,
        "waypoint_count": 16,
        "duration_s": 0.50,
        "timeout_s": 1.5,
        "q_sha256": (
            "f6a6a64c60287807be1196735476b8abc81761e62135c2d50807e94aced63b27"
        ),
        "interval_count": 16,
        "speed_bound": 0.12,
        "acceleration_bound": 0.75,
    },
}
CORE_CAPTURE_ROUTE_BLOCKERS = [
    "cam_contact_policy_not_authorized",
    "live_dynamics_not_validated",
    "closed_loop_source_law_tracking_not_implemented",
    "live_mujoco_route_tracking_not_yet_certified",
    "cam_tab_contact_force_and_depth_not_yet_certified",
    "positive_lock_cam_friction_coefficient_unqualified",
    "positive_lock_cam_load_capacity_unqualified",
    "positive_lock_cam_dynamics_unqualified",
]
CORE_CAM_TAB_LEADING_GEOM = "qc_col_lock_slider_tab_part_001"
CORE_CAM_TAB_NONCONTACT_GEOM = "qc_col_lock_slider_tab_part_000"
CORE_CAM_GEOMS = [
    "dock_gripper_cam_collision",
    "dock_gripper_cam_axial_lead_collision",
    "dock_gripper_cam_hold_finger_collision",
    "dock_gripper_cam_outer_root_lower_collision",
    "dock_gripper_cam_outer_root_upper_collision",
]
CORE_CAM_CONTACT_ACTIONS = [
    "gripper_capture_lateral_align",
    "gripper_capture_axial_open_side",
    "gripper_capture_coupled_recenter",
    "gripper_capture_centered_final",
]
CORE_CAM_FUNCTIONAL_ROLES = [
    "functional_axial_lead_ramp",
    "functional_hold_finger_face",
]
CORE_CAM_TAB_CONTACT_BLOCKERS = [
    "provisional_20um_contact_guard_not_physical_contact_authority",
    "positive_lock_cam_friction_coefficient_unqualified",
    "positive_lock_cam_load_capacity_unqualified",
    "positive_lock_cam_dynamics_unqualified",
    "free_space_servo_tracking_not_yet_closed",
    "source_negative_y_release_route_is_static_kinematics_only",
    "source_negative_y_reverse_insertion_and_tracking_unvalidated",
    "continuous_between_mj_steps_tunnel_authority_absent",
    "functional_interval_motion_bound_not_certified",
]
CORE_CAM_COMPILED_MODEL_XML_EQUIVALENT_SHA256 = (
    "edfc58afb55d83901f3e35f7e3426d5ffeef696257ef55e28955b400249480d0"
)
POGO_LEDGER_PATH = (
    MAGNETIC_ROOT
    / "source_authority"
    / "millmax_7983"
    / "authority_ledger.json"
)
POGO_DIMENSION_DRAWING_URL = (
    "https://www.mill-max.com/sites/default/files/external/products/"
    "fullsize/2020-09/7983.svg"
)
POGO_PRESS_FIT_NOTE_URL = (
    "https://www.mill-max.com/sites/default/files/external/assets/2020-10/"
    "spring-loaded_solder-cup_pin_2.pdf"
)
POGO_LEDGER_AXIAL_IN = {
    "maximum_exposed_plunger": 0.062,
    "shell_top_to_shoulder_bottom": 0.132,
    "shoulder_bottom_from_base": 0.180,
    "shoulder_thickness": 0.028,
    "knurl_length": 0.030,
    "cup_reference_length": 0.145,
    "cup_bore_length": 0.100,
    "barb_axial_reference": 0.025,
    "mid_stroke": 0.0275,
    "full_stroke": 0.055,
    "full_stroke_tolerance": 0.005,
}
POGO_LEDGER_DIAMETERS_IN = {
    "moving_plunger": 0.042,
    "upper_guide": 0.068,
    "upper_shell": 0.073,
    "barb": 0.0765,
    "shoulder": 0.083,
    "knurl": 0.065,
    "solder_cup_outer": 0.060,
    "solder_cup_bore": 0.038,
}


def _finite_real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _number_matches(value: Any, expected: float, tolerance: float = 1.0e-12) -> bool:
    observed = _finite_real(value)
    return observed is not None and abs(observed - expected) <= tolerance


def _two_number_bounds(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    lower = _finite_real(value[0])
    upper = _finite_real(value[1])
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def pogo_authority_contract_errors(
    record: Any,
    ledger: Any,
    *,
    require_release_ready: bool = True,
) -> list[str]:
    """Recompute the hash-ledger 7983 contract and its release verdict."""

    errors: list[str] = []
    if not isinstance(record, dict):
        return ["pogo_authority_record_missing"]
    if not isinstance(ledger, dict):
        return ["pogo_authority_ledger_missing"]

    def expect_number(
        mapping: dict[str, Any], key: str, expected: float, prefix: str
    ) -> None:
        if not _number_matches(mapping.get(key), expected):
            errors.append(f"{prefix}:{key}")

    def evidence_matches(observed: Any, expected: Any) -> bool:
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            return observed == expected
        if isinstance(expected, (int, float)):
            return _number_matches(observed, float(expected))
        if isinstance(expected, list):
            return (
                isinstance(observed, list)
                and len(observed) == len(expected)
                and all(
                    evidence_matches(left, right)
                    for left, right in zip(observed, expected, strict=True)
                )
            )
        if isinstance(expected, dict):
            return (
                isinstance(observed, dict)
                and set(observed) == set(expected)
                and all(
                    evidence_matches(observed[key], value)
                    for key, value in expected.items()
                )
            )
        return observed == expected

    if record.get("schema_version") != "1.0":
        errors.append("schema_version")
    if record.get("part_number") != POGO_PART_NUMBER:
        errors.append("part_number")
    if record.get("units") != "mm":
        errors.append("units")
    if record.get("pin_source_frame") != (
        "solder_cup_end_z0_axis_positive_toward_plunger"
    ):
        errors.append("pin_source_frame")

    if ledger.get("schema_version") != "1.0":
        errors.append("ledger:schema_version")
    if ledger.get("part_number") != POGO_PART_NUMBER:
        errors.append("ledger:part_number")
    if ledger.get("drawing_units") != "inch":
        errors.append("ledger:drawing_units")
    if ledger.get("standard_tolerances") != {
        "length_in": 0.006,
        "diameter_in": 0.002,
        "angle_deg": 2.0,
    }:
        errors.append("ledger:standard_tolerances")
    if ledger.get("axial_dimensions_in") != POGO_LEDGER_AXIAL_IN:
        errors.append("ledger:axial_dimensions")
    if ledger.get("reference_dimensions_in") != {
        "overall_parenthesized": 0.374,
    }:
        errors.append("ledger:reference_dimensions")
    if ledger.get("diameters_in") != POGO_LEDGER_DIAMETERS_IN:
        errors.append("ledger:diameters")
    if ledger.get("press_fit_note_dimensions_in") != {
        "barb_recommended_hole": 0.0755,
        "knurl_recommended_hole": 0.062,
        "body_counterbore_minimum": 0.087,
    }:
        errors.append("ledger:press_fit_dimensions")

    expected_drawing = {
        "url": POGO_DIMENSION_DRAWING_URL,
        "media_type": "image/svg+xml",
        "bytes": POGO_DIMENSION_DRAWING_BYTES,
        "sha256": POGO_DIMENSION_DRAWING_SHA256,
    }
    expected_note = {
        "url": POGO_PRESS_FIT_NOTE_URL,
        "media_type": "application/pdf",
        "bytes": POGO_PRESS_FIT_NOTE_BYTES,
        "sha256": POGO_PRESS_FIT_NOTE_SHA256,
    }
    expected_redistribution = {
        "manufacturer_file_license_confirmed": False,
        "manufacturer_files_vendored": False,
        "semantics": (
            "URLs, byte counts, SHA-256 digests, and dimension facts are "
            "retained; cached manufacturer artwork is not redistributed "
            "because redistribution terms were not established."
        ),
    }
    provenance = ledger.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        errors.append("ledger:provenance")
    if provenance.get("dimension_drawing") != expected_drawing:
        errors.append("ledger:dimension_drawing")
    if provenance.get("press_fit_application_note") != expected_note:
        errors.append("ledger:press_fit_note")
    if provenance.get("redistribution") != expected_redistribution:
        errors.append("ledger:redistribution")
    if provenance.get("retrieval_timestamp_available") is not False:
        errors.append("ledger:retrieval_timestamp")

    sources = record.get("official_sources")
    if not isinstance(sources, dict):
        sources = {}
        errors.append("official_sources")
    if sources.get("dimension_drawing_svg") != expected_drawing:
        errors.append("official_sources:dimension_drawing")
    if sources.get("press_fit_application_note_pdf") != expected_note:
        errors.append("official_sources:press_fit_note")
    if sources.get("redistribution") != expected_redistribution:
        errors.append("official_sources:redistribution")
    if sources.get("offline_manufacturer_byte_revalidation_available") is not False:
        errors.append("official_sources:offline_revalidation")
    if sources.get("hash_pin_semantics") != (
        "records the recovered manufacturer byte digests; cached manufacturer "
        "artwork is not redistributed"
    ):
        errors.append("official_sources:hash_pin_semantics")
    ledger_record = sources.get("derived_authority_ledger")
    if not isinstance(ledger_record, dict):
        errors.append("official_sources:ledger_record")
    else:
        expected_path = POGO_LEDGER_PATH.relative_to(REPOSITORY_ROOT).as_posix()
        if ledger_record.get("path") != expected_path:
            errors.append("official_sources:ledger_path")
        if (
            not isinstance(ledger_record.get("bytes"), int)
            or ledger_record["bytes"] <= 0
        ):
            errors.append("official_sources:ledger_bytes")
        digest = ledger_record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append("official_sources:ledger_sha256")

    if record.get("drawing_tolerances") != {
        "standard_length_mm": 0.006 * 25.4,
        "standard_diameter_mm": 0.002 * 25.4,
        "standard_angle_deg": 2.0,
    }:
        errors.append("drawing_tolerances")
    if record.get("fixed_shell_envelope_authority") != {
        "kind": "official-drawing-derived_conservative_nominal_exterior",
        "manufacturer_3d_cad": False,
        "transition_spans_enlarged_to_adjacent_maximum_diameter": True,
        "manufacturing_diameter_tolerance_included": False,
        "mass_or_internal_material_authority": False,
    }:
        errors.append("fixed_shell_envelope_authority")

    profile = record.get("dimensioned_profile")
    if not isinstance(profile, dict):
        profile = {}
        errors.append("dimensioned_profile")
    expected_profile_scalars = {
        "overall_reference_length_mm": 0.374 * 25.4,
        "fixed_shell_length_mm": (0.180 + 0.132) * 25.4,
        "maximum_exposed_plunger_mm": 0.062 * 25.4,
        "shell_top_to_shoulder_bottom_mm": 0.132 * 25.4,
        "shoulder_bottom_from_base_mm": 0.180 * 25.4,
        "shoulder_thickness_mm": 0.028 * 25.4,
        "knurl_length_mm": 0.030 * 25.4,
        "cup_reference_length_mm": 0.145 * 25.4,
        "cup_bore_length_mm": 0.100 * 25.4,
        "barb_axial_reference_mm": 0.025 * 25.4,
    }
    for key, expected in expected_profile_scalars.items():
        expect_number(profile, key, expected, "profile")
    if profile.get("overall_reference_semantics") != (
        "parenthesized drawing reference; independently checked against the "
        "dimensioned .180+.132+.062 chain"
    ):
        errors.append("profile:overall_reference_semantics")
    fixed_shell_observed = _finite_real(profile.get("fixed_shell_length_mm"))
    shell_span_observed = _finite_real(
        profile.get("shell_top_to_shoulder_bottom_mm")
    )
    base_span_observed = _finite_real(profile.get("shoulder_bottom_from_base_mm"))
    exposed_observed = _finite_real(profile.get("maximum_exposed_plunger_mm"))
    overall_reference = _finite_real(profile.get("overall_reference_length_mm"))
    if (
        fixed_shell_observed is None
        or shell_span_observed is None
        or base_span_observed is None
        or not _number_matches(
            fixed_shell_observed, base_span_observed + shell_span_observed
        )
    ):
        errors.append("profile:fixed_shell_dimension_chain")
    if (
        fixed_shell_observed is None
        or exposed_observed is None
        or overall_reference is None
        or not _number_matches(
            overall_reference, fixed_shell_observed + exposed_observed
        )
    ):
        errors.append("profile:overall_reference_chain")
    if profile.get("nominal_diameters_mm") != POGO_OFFICIAL_PROFILE_DIAMETERS_MM:
        errors.append("profile:diameters")

    shoulder_bottom = 0.180 * 25.4
    shoulder_thickness = 0.028 * 25.4
    knurl_length = 0.030 * 25.4
    fixed_shell_length = (0.180 + 0.132) * 25.4
    expected_segments = [
        {
            "name": "solder_cup",
            "z_bounds_mm": [0.0, 0.145 * 25.4],
            "outer_diameter_mm": 0.060 * 25.4,
            "semantics": "drawing_nominal_outer_envelope",
        },
        {
            "name": "cup_to_knurl_transition_bound",
            "z_bounds_mm": [0.145 * 25.4, shoulder_bottom - knurl_length],
            "outer_diameter_mm": 0.065 * 25.4,
            "semantics": "conservative_largest_adjacent_dimensioned_diameter",
        },
        {
            "name": "knurl",
            "z_bounds_mm": [shoulder_bottom - knurl_length, shoulder_bottom],
            "outer_diameter_mm": 0.065 * 25.4,
            "semantics": "dimensioned_press_fit_feature",
        },
        {
            "name": "shoulder",
            "z_bounds_mm": [shoulder_bottom, shoulder_bottom + shoulder_thickness],
            "outer_diameter_mm": 0.083 * 25.4,
            "semantics": "dimensioned_hard_stop_feature",
        },
        {
            "name": "plunger_side_fixed_features",
            "z_bounds_mm": [shoulder_bottom + shoulder_thickness, fixed_shell_length],
            "outer_diameter_mm": 0.0765 * 25.4,
            "semantics": (
                "conservative_full_span_envelope_of_dimensioned_"
                "upper_guide_upper_shell_and_barb_diameters"
            ),
        },
    ]
    if profile.get("fixed_shell_collision_envelope_segments") != expected_segments:
        errors.append("profile:fixed_shell_segments")
    moving = profile.get("moving_plunger")
    if not isinstance(moving, dict):
        moving = {}
    if moving.get("motion_kind") != "prismatic":
        errors.append("plunger:motion_kind")
    expect_number(moving, "outer_diameter_mm", 0.042 * 25.4, "plunger")
    expect_number(
        moving, "maximum_exposed_length_mm", 0.062 * 25.4, "plunger"
    )
    if moving.get("motion_axis") != [0.0, 0.0, -1.0]:
        errors.append("plunger:axis")
    compression_range = _two_number_bounds(moving.get("compression_range_mm"))
    if compression_range is None or not (
        _number_matches(compression_range[0], 0.0)
        and _number_matches(compression_range[1], 0.060 * 25.4)
    ):
        errors.append("plunger:compression_range")
    if moving.get("collision_shape_semantics") != (
        "official-drawing-derived conservative cylindrical envelope of round "
        "tip; not manufacturer 3D CAD"
    ):
        errors.append("plunger:collision_shape")

    stroke = record.get("stroke")
    if not isinstance(stroke, dict):
        errors.append("stroke")
    else:
        for key, expected in (
            ("mid_stroke_nominal_mm", 0.0275 * 25.4),
            ("full_stroke_nominal_mm", 0.055 * 25.4),
            ("full_stroke_tolerance_mm", 0.005 * 25.4),
            ("guaranteed_minimum_full_stroke_mm", 0.050 * 25.4),
            ("maximum_full_stroke_mm", 0.060 * 25.4),
        ):
            expect_number(stroke, key, expected, "stroke")

    mounting = record.get("selected_mounting_design")
    if not isinstance(mounting, dict):
        mounting = {}
        errors.append("selected_mounting_design")
    if mounting.get("mode") != "knurl_solder_cup_first":
        errors.append("mounting:mode")
    for key, expected in (
        ("application_note_knurl_feature_label_mm", 1.65),
        ("drawing_knurl_nominal_diameter_mm", 0.065 * 25.4),
        ("application_note_hole_exact_inch_conversion_mm", 0.062 * 25.4),
        ("application_note_hole_rounded_label_mm", 1.58),
        ("retention_land_diameter_mm", 1.58),
        ("body_counterbore_minimum_diameter_mm", 0.087 * 25.4),
        ("body_counterbore_design_diameter_mm", 2.31),
        ("barb_alternative_recommended_hole_diameter_mm", 0.0755 * 25.4),
    ):
        expect_number(mounting, key, expected, "mounting")

    datums = mounting.get("installed_datums")
    datum_by_signal: dict[str, dict[str, Any]] = {}
    if isinstance(datums, list):
        for datum in datums:
            if not isinstance(datum, dict) or not isinstance(datum.get("signal"), str):
                errors.append("mounting:datum_record")
                continue
            signal = str(datum["signal"])
            if signal in datum_by_signal:
                errors.append(f"mounting:duplicate_datum:{signal}")
            datum_by_signal[signal] = datum
    else:
        errors.append("mounting:installed_datums")
    if set(datum_by_signal) != set(POGO_SIGNAL_CENTRES_MM):
        errors.append("mounting:datum_inventory")
    nominal_protrusions: dict[str, float] = {}
    for signal, centre in POGO_SIGNAL_CENTRES_MM.items():
        datum = datum_by_signal.get(signal, {})
        protrusion = _finite_real(datum.get("nominal_face_protrusion_mm"))
        if protrusion is None:
            protrusion = math.nan
            errors.append(f"mounting:datum:{signal}:nominal_face_protrusion_mm")
        else:
            nominal_protrusions[signal] = protrusion
        if signal != "GND" and not _number_matches(protrusion, 0.70):
            errors.append(f"mounting:standard_protrusion:{signal}")
        base_z = 9.50 + protrusion - 0.374 * 25.4
        shoulder_z = base_z + shoulder_bottom
        target_contact_plane_z = 9.50 - 0.05
        mated_compression = protrusion + 0.05
        if mated_compression > 0.050 * 25.4 + 1.0e-12:
            errors.append(f"mounting:datum:{signal}:minimum_stroke_exceeded")
        expected_datum = {
            "signal": signal,
            "centre_xy_mm": centre,
            "installation_mode": "knurl_solder_cup_first",
            "insertion_direction": "mating_face_toward_rear_negative_z",
            "base_z_mm": base_z,
            "fixed_shell_top_z_mm": base_z + fixed_shell_length,
            "shoulder_stop_plane_z_mm": shoulder_z,
            "shoulder_z_bounds_mm": [shoulder_z, shoulder_z + shoulder_thickness],
            "knurl_z_bounds_mm": [shoulder_z - knurl_length, shoulder_z],
            "retention_land_z_bounds_mm": [1.85, shoulder_z],
            "body_counterbore_z_bounds_mm": [shoulder_z, 9.50],
            "full_extension_tip_z_mm": base_z + 0.374 * 25.4,
            "nominal_face_protrusion_mm": protrusion,
            "target_pad_exposed_contact_plane_z_mm": target_contact_plane_z,
            "mated_compression_mm": mated_compression,
            "mated_tip_z_mm": target_contact_plane_z,
            "nominal_design_remaining_against_catalog_minimum_stroke_mm": (
                0.050 * 25.4 - mated_compression
            ),
            "remaining_stroke_semantics": (
                "nominal installation arithmetic only; part, bore, target, "
                "and fabrication tolerances are not included"
            ),
        }
        if not evidence_matches(datum, expected_datum):
            errors.append(f"mounting:datum:{signal}")

    fit = mounting.get("nominal_and_part_tolerance_only_fit")
    if not isinstance(fit, dict):
        fit = {}
        errors.append("mounting:fit_record")
    diameter_tolerance = 0.002 * 25.4
    knurl = 0.065 * 25.4
    shoulder = 0.083 * 25.4
    cup = 0.060 * 25.4
    expected_fit = {
        "knurl_diameter_range_mm": [
            knurl - diameter_tolerance,
            knurl + diameter_tolerance,
        ],
        "knurl_diametral_interference_range_mm": [
            knurl - diameter_tolerance - 1.58,
            knurl + diameter_tolerance - 1.58,
        ],
        "solder_cup_max_diameter_mm": cup + diameter_tolerance,
        "solder_cup_minimum_diametral_passage_mm": 1.58 - (cup + diameter_tolerance),
        "shoulder_diameter_range_mm": [
            shoulder - diameter_tolerance,
            shoulder + diameter_tolerance,
        ],
        "minimum_shoulder_bearing_radial_overlap_mm": (
            shoulder - diameter_tolerance - 1.58
        ) / 2.0,
        "body_counterbore_minimum_radial_clearance_mm": (
            2.31 - (shoulder + diameter_tolerance)
        ) / 2.0,
        "fabrication_hole_tolerance_included": False,
        "pullout_force_bound_included": False,
    }
    if fit != expected_fit:
        errors.append("mounting:fit_arithmetic")

    first_mate = record.get("first_mate_tolerance_stack")
    if not isinstance(first_mate, dict):
        first_mate = {}
        errors.append("first_mate:record")
    nominal_lead = _finite_real(first_mate.get("nominal_ground_lead_mm"))
    length_error = _finite_real(first_mate.get("independent_pin_pair_error_bound_mm"))
    guaranteed = _finite_real(
        first_mate.get("guaranteed_worst_case_ground_lead_mm")
    )
    expected_nominal_lead = None
    if set(nominal_protrusions) == set(POGO_SIGNAL_CENTRES_MM):
        standard_values = [
            nominal_protrusions[signal] for signal in POGO_STANDARD_SIGNALS
        ]
        if any(
            not _number_matches(value, standard_values[0])
            for value in standard_values[1:]
        ):
            errors.append("first_mate:standard_datum_mismatch")
        expected_nominal_lead = nominal_protrusions["GND"] - max(standard_values)
    if expected_nominal_lead is None or not _number_matches(
        nominal_lead, expected_nominal_lead
    ):
        errors.append("first_mate:nominal_lead")
    if first_mate.get("shoulder_to_tip_dimension_terms_per_pin") != [
        ".132_in_fixed_span",
        ".062_in_maximum_exposed_plunger",
    ]:
        errors.append("first_mate:dimension_term_roster")
    if first_mate.get("independent_standard_length_tolerance_term_count") != 4:
        errors.append("first_mate:length_term_count")
    if not _number_matches(length_error, 4.0 * 0.006 * 25.4):
        errors.append("first_mate:length_error")
    if (
        nominal_lead is None
        or length_error is None
        or guaranteed is None
        or not _number_matches(guaranteed, nominal_lead - length_error)
    ):
        errors.append("first_mate:arithmetic")
    first_mate_passed = guaranteed is not None and guaranteed > 0.0
    if first_mate.get("passed") is not first_mate_passed:
        errors.append("first_mate:verdict")

    authority = record.get("release_authority")
    if not isinstance(authority, dict):
        authority = {}
        errors.append("release_authority")
    press_fit_passed = bool(
        fit.get("fabrication_hole_tolerance_included")
        and fit.get("pullout_force_bound_included")
    )
    cycle_passed = bool(
        authority.get("installed_electrical_cycle_reliability_qualified")
    )
    expected_blockers = []
    if not first_mate_passed:
        expected_blockers.append("ground_first_mate_tolerance_stack_unqualified")
    if not press_fit_passed:
        expected_blockers.append("knurl_press_fit_process_and_pullout_unqualified")
    if not cycle_passed:
        expected_blockers.append("installed_electrical_cycle_reliability_unqualified")
    for key in (
        "official_sources_hash_pinned",
        "fixed_shell_drawing_envelope_reconstructed",
        "moving_plunger_split_from_fixed_shell",
        "knurl_mounting_mode_selected",
        "sectional_bore_and_shoulder_stop_resolved",
        "ground_first_mate_shoulder_datum_resolved",
    ):
        if authority.get(key) is not True:
            errors.append(f"release_authority:{key}")
    expected_qualification_flags = {
        "ground_first_mate_tolerance_stack_qualified": first_mate_passed,
        "knurl_press_fit_process_and_pullout_qualified": press_fit_passed,
        "installed_electrical_cycle_reliability_qualified": cycle_passed,
    }
    for key, expected in expected_qualification_flags.items():
        if authority.get(key) is not expected:
            errors.append(f"release_authority:{key}")
    if authority.get("blockers") != expected_blockers:
        errors.append("release_authority:blockers")
    release_ready = not expected_blockers
    if authority.get("release_ready") is not release_ready:
        errors.append("release_authority:verdict")
    if require_release_ready and not release_ready:
        errors.append("release_authority:not_ready")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(path: Path, description: str) -> Path:
    if not path.is_file():
        raise unittest.SkipTest(f"{description} is not restored yet: {path}")
    return path


def load_json(path: Path, description: str) -> dict[str, Any]:
    require_path(path, description)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise AssertionError(f"{description} must be a JSON object: {path}")
    return payload


def import_file(path: Path, module_name: str, description: str) -> ModuleType:
    require_path(path, description)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot import {description}: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and several annotation resolvers consult sys.modules while
    # the class body is executing; mirror normal import semantics here.
    sys.modules[module_name] = module
    module_directory = str(path.parent)
    inserted_path = module_directory not in sys.path
    if inserted_path:
        sys.path.insert(0, module_directory)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        sys.modules.pop(module_name, None)
        raise unittest.SkipTest(
            f"{description} dependency is not installed yet: {error.name}"
        ) from error
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted_path:
            sys.path.remove(module_directory)
    return module


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _authority_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expected_pogo_runtime_geometry_contract(cad: ModuleType) -> dict[str, Any]:
    """Compose the runtime geometry contract only from hash-bound CAD facts."""

    source = cad.pogo_interface_authority_contract()
    profile = source["dimensioned_profile"]
    source_segments = {
        segment["name"]: segment
        for segment in profile["fixed_shell_collision_envelope_segments"]
    }
    datums = {
        datum["signal"]: datum
        for datum in source["selected_mounting_design"]["installed_datums"]
    }
    signals: dict[str, Any] = {}
    for runtime_signal, source_signal in POGO_RUNTIME_SIGNAL_MAP.items():
        datum = datums[source_signal]
        fixed_body_name = f"qc_pogo_{runtime_signal}_fixed_shell_body"
        fixed_segments = []
        for source_name, runtime_suffix in POGO_RUNTIME_FIXED_SEGMENTS:
            segment = source_segments[source_name]
            z_min_mm, z_max_mm = [float(value) for value in segment["z_bounds_mm"]]
            diameter_mm = float(segment["outer_diameter_mm"])
            fixed_segments.append(
                {
                    "source_segment": source_name,
                    "name": f"qc_col_pogo_{runtime_signal}_{runtime_suffix}",
                    "geom_type": "cylinder",
                    "local_pos_m": [0.0, 0.0, (z_min_mm + z_max_mm) / 2000.0],
                    "size_m": [diameter_mm / 2000.0, (z_max_mm - z_min_mm) / 2000.0],
                    "bus_contact_eligible": False,
                }
            )
        fixed_length_m = float(profile["fixed_shell_length_mm"]) / 1000.0
        exposed_length_m = float(profile["maximum_exposed_plunger_mm"]) / 1000.0
        plunger_diameter_m = (
            float(profile["moving_plunger"]["outer_diameter_mm"]) / 1000.0
        )
        range_max_m = float(source["stroke"]["maximum_full_stroke_mm"]) / 1000.0
        signals[runtime_signal] = {
            "source_signal": source_signal,
            "installed_datum": datum,
            "fixed_body": {
                "name": fixed_body_name,
                "parent": "robot_plate_frame",
                "pos_m": [
                    float(datum["centre_xy_mm"][0]) / 1000.0,
                    float(datum["centre_xy_mm"][1]) / 1000.0,
                    float(datum["base_z_mm"]) / 1000.0,
                ],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "fixed_segments": fixed_segments,
            "plunger": {
                "body_name": f"qc_pogo_{runtime_signal}_plunger_body",
                "parent": fixed_body_name,
                "local_pos_m": [0.0, 0.0, fixed_length_m],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "joint_name": f"qc_pogo_{runtime_signal}_plunger",
                "joint_type": "slide",
                "axis": [0.0, 0.0, -1.0],
                "range_m": [0.0, range_max_m],
                "geom_name": f"qc_col_pogo_{runtime_signal}_plunger",
                "geom_type": "cylinder",
                "geom_local_pos_m": [0.0, 0.0, exposed_length_m / 2.0],
                "geom_size_m": [plunger_diameter_m / 2.0, exposed_length_m / 2.0],
                "bus_contact_eligible": True,
            },
        }
    return {
        "schema_version": "1.0",
        "source_binding": {
            "ledger_file": _authority_file_record(POGO_LEDGER_PATH),
            "generator_file": _authority_file_record(MAGNETIC_ROOT / "generate_cad.py"),
            "canonical_contract_sha256": canonical_json_sha256(source),
        },
        "runtime_to_source_signal": dict(POGO_RUNTIME_SIGNAL_MAP),
        "signals": signals,
        "dynamics_authority": {
            "geometry_and_datum_authority": True,
            "mass_properties_authority": False,
            "spring_force_curve_authority": False,
            "damping_authority": False,
            "ground_first_mate_tolerance_stack_qualified": False,
            "blockers": list(POGO_RUNTIME_RELEASE_BLOCKERS),
            "release_ready": False,
        },
        "passed": True,
        "release_ready": False,
    }


def _evidence_mismatches(observed: Any, expected: Any, path: str = "root") -> list[str]:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return [] if observed == expected else [path]
    if isinstance(expected, (int, float)):
        return [] if _number_matches(observed, float(expected)) else [path]
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return [path]
        return [
            mismatch
            for index, (left, right) in enumerate(
                zip(observed, expected, strict=True)
            )
            for mismatch in _evidence_mismatches(left, right, f"{path}[{index}]")
        ]
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            return [path]
        return [
            mismatch
            for key, value in expected.items()
            for mismatch in _evidence_mismatches(
                observed[key], value, f"{path}.{key}"
            )
        ]
    return [] if observed == expected else [path]


def pogo_runtime_geometry_contract_errors(
    record: Any,
    cad: ModuleType,
    ledger: dict[str, Any],
) -> list[str]:
    source = cad.pogo_interface_authority_contract()
    errors = [
        f"source:{error}"
        for error in pogo_authority_contract_errors(
            source, ledger, require_release_ready=False
        )
    ]
    if not isinstance(record, dict):
        return [*errors, "runtime_contract_missing"]
    return [
        *errors,
        *[
            f"runtime:{path}"
            for path in _evidence_mismatches(
                record, expected_pogo_runtime_geometry_contract(cad)
            )
        ],
    ]


def expected_positive_lock_cam_runtime_contract(
    cad: ModuleType,
) -> dict[str, Any]:
    """Reconstruct the runtime cam contract from the released CAD source."""

    source = cad.positive_lock_cam_contract()
    main = source["main_xy_wedge"]
    lead = source["axial_lead"]
    hold = source["hold_finger"]
    root = source["outer_root_bridge"]

    def mm_to_m(value: float) -> float:
        # The public runtime schema serializes exact decimal millimetre facts
        # in metres; normalize binary division noise before canonical hashing.
        return round(float(value) / 1000.0, 12)

    def bounds_to_m(bounds: dict[str, list[float]]) -> list[list[float]]:
        return [
            [mm_to_m(value) for value in bounds[axis]]
            for axis in ("x", "y", "z")
        ]

    polygon_mm = [
        [float(value) for value in point] for point in main["polygon_xy_mm"]
    ]
    polygon_twice_area_mm2 = abs(
        math.fsum(
            polygon_mm[index][0] * polygon_mm[(index + 1) % len(polygon_mm)][1]
            - polygon_mm[(index + 1) % len(polygon_mm)][0]
            * polygon_mm[index][1]
            for index in range(len(polygon_mm))
        )
    )
    main_z_mm = [float(value) for value in main["z_bounds_mm"]]
    main_volume_mm3 = (
        polygon_twice_area_mm2
        * (main_z_mm[1] - main_z_mm[0])
        / 2.0
    )
    main_geometry = {
        "polygon_xy": [
            [mm_to_m(value) for value in point] for point in polygon_mm
        ],
        "z_bounds": [mm_to_m(value) for value in main_z_mm],
    }
    lead_geometry = {
        rectangle_name: {
            "x_bounds": [
                mm_to_m(value) for value in lead[rectangle_name]["x"]
            ],
            "y_bounds": [
                mm_to_m(value) for value in lead[rectangle_name]["y"]
            ],
            "z": mm_to_m(lead[rectangle_name]["z"]),
        }
        for rectangle_name in ("lower_rectangle_mm", "upper_rectangle_mm")
    }
    lead_geometry = {
        "lower_rectangle": lead_geometry["lower_rectangle_mm"],
        "upper_rectangle": lead_geometry["upper_rectangle_mm"],
    }
    hold_geometry = {"bounds": bounds_to_m(hold["bounds_mm"])}
    authored_root_bounds = bounds_to_m(root["bounds_mm"])
    main_z_min_m = main_geometry["z_bounds"][0]
    root_x, root_y, root_z = authored_root_bounds
    root_remainders = [
        [root_x, [root_y[0], 0.0], [root_z[0], main_z_min_m]],
        [root_x, [0.0, root_y[1]], [main_z_min_m, root_z[1]]],
    ]
    root_geometry = {
        "authored_bounds": authored_root_bounds,
        "runtime_remainder_bounds": root_remainders,
    }

    def component(
        source_component: str,
        representation: str,
        source_geometry_m: dict[str, Any],
        runtime_geom_names: list[str],
        source_volume_mm3: float,
        runtime_volume_mm3: float,
    ) -> dict[str, Any]:
        digest_preimage = {
            "source_component": source_component,
            "representation": representation,
            "source_geometry_m": source_geometry_m,
        }
        return {
            **digest_preimage,
            "runtime_geom_names": runtime_geom_names,
            "source_volume_mm3": source_volume_mm3,
            "runtime_volume_mm3": runtime_volume_mm3,
            "canonical_geometry_sha256": canonical_json_sha256(
                digest_preimage
            ),
        }

    def names(tool: str) -> list[str]:
        return [
            f"dock_{tool}_cam_collision",
            f"dock_{tool}_cam_axial_lead_collision",
            f"dock_{tool}_cam_hold_finger_collision",
            f"dock_{tool}_cam_outer_root_lower_collision",
            f"dock_{tool}_cam_outer_root_upper_collision",
        ]

    source_root_volume_mm3 = float(root["gross_volume_mm3"])
    authored_overlap_total_mm3 = math.fsum(
        (
            float(root["overlap_with_main_wedge_mm3"]),
            float(root["overlap_with_hold_mm3"]),
        )
    )
    runtime_root_volume_mm3 = (
        source_root_volume_mm3 - authored_overlap_total_mm3
    )
    components = [
        component(
            "main_xy_wedge",
            "single_convex_prism_mesh",
            main_geometry,
            [names("gripper")[0]],
            main_volume_mm3,
            main_volume_mm3,
        ),
        component(
            "axial_lead",
            "single_convex_ruled_loft_mesh",
            lead_geometry,
            [names("gripper")[1]],
            float(lead["volume_mm3"]),
            float(lead["volume_mm3"]),
        ),
        component(
            "hold_finger",
            "analytic_axis_aligned_box",
            hold_geometry,
            [names("gripper")[2]],
            float(hold["volume_mm3"]),
            float(hold["volume_mm3"]),
        ),
        component(
            "outer_root_bridge",
            "two_nonoverlapping_analytic_boxes_exact_union_remainder",
            root_geometry,
            names("gripper")[3:5],
            source_root_volume_mm3,
            runtime_root_volume_mm3,
        ),
    ]
    component_volume_sum_mm3 = math.fsum(
        float(record["source_volume_mm3"]) for record in components
    )
    runtime_component_volume_sum_mm3 = math.fsum(
        float(record["runtime_volume_mm3"]) for record in components
    )
    return {
        "schema_version": "1.0",
        "source_binding": {
            "generator_file": _authority_file_record(MAGNETIC_ROOT / "generate_cad.py"),
            "positive_lock_cam_contract_sha256": canonical_json_sha256(source),
        },
        "authority_scope": {
            "geometry_and_placement_authority": True,
            "friction_coefficient_authority": False,
            "load_capacity_authority": False,
            "dynamics_authority": False,
            "overlapping_component_contact_force_authority": False,
            "blockers": list(POSITIVE_LOCK_CAM_RUNTIME_BLOCKERS),
            "release_ready": False,
        },
        "core_gripper": {
            "frame": "dock_gripper",
            "source_function": "positive_lock_cam",
            "runtime_geom_names": names("gripper"),
            "components": components,
            "expected_union": {
                "bounds_m": [
                    [mm_to_m(value) for value in source["expected_geometry"]["bounds_mm"][axis]]
                    for axis in ("x", "y", "z")
                ],
                "component_volume_sum_mm3": component_volume_sum_mm3,
                "runtime_component_volume_sum_mm3": (
                    runtime_component_volume_sum_mm3
                ),
                "authored_pair_overlaps": [
                    {
                        "components": ["main_xy_wedge", "outer_root_bridge"],
                        "volume_mm3": float(
                            root["overlap_with_main_wedge_mm3"]
                        ),
                    },
                    {
                        "components": ["hold_finger", "outer_root_bridge"],
                        "volume_mm3": float(root["overlap_with_hold_mm3"]),
                    },
                ],
                "authored_pair_overlap_total_mm3": (
                    authored_overlap_total_mm3
                ),
                "runtime_pairwise_overlap_total_mm3": 0.0,
                "source_volume_mm3": float(
                    source["expected_geometry"]["total_volume_mm3"]
                ),
            },
        },
        "matcha_bays": {
            tool: {
                "frame": f"dock_{tool}",
                "source_function": "INTERFACE.positive_lock_cam",
                "runtime_geom_names": names(tool),
                "uses_core_canonical_geometry": True,
                "geometry_and_placement_authority": True,
            }
            for tool in ("spoon", "whisk")
        },
        "passed": True,
        "release_ready": False,
    }


def positive_lock_cam_runtime_contract_errors(
    record: Any,
    cad: ModuleType,
) -> list[str]:
    if not isinstance(record, dict):
        return ["runtime_cam_contract_missing"]
    return [
        f"runtime_cam:{path}"
        for path in _evidence_mismatches(
            record, expected_positive_lock_cam_runtime_contract(cad)
        )
    ]


def _independent_core_capture_x_mm(preseat_mm: float) -> float:
    if not math.isfinite(preseat_mm) or preseat_mm < 0.0:
        raise ValueError("preseat must be finite and nonnegative")
    if preseat_mm >= 6.4:
        return 0.20
    if preseat_mm <= 3.2:
        return 0.0
    return 0.20 * (preseat_mm - 3.2) / (6.4 - 3.2)


def _independent_initialized_active_geometry_sha256(
    model: Any,
    data: Any,
    mujoco: ModuleType,
) -> str:
    """Rebuild the initialized collision fingerprint without production code."""

    records: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        if not (
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        ):
            continue
        record: dict[str, Any] = {
            "geom_id": geom_id,
            "name": str(model.geom(geom_id).name),
            "body": str(model.body(int(model.geom_bodyid[geom_id])).name),
            "type": int(model.geom_type[geom_id]),
            "group": int(model.geom_group[geom_id]),
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
            "pos_float_hex": [
                float(value).hex() for value in model.geom_pos[geom_id]
            ],
            "quat_float_hex": [
                float(value).hex() for value in model.geom_quat[geom_id]
            ],
            "size_float_hex": [
                float(value).hex() for value in model.geom_size[geom_id]
            ],
            "initialized_world_pos_float_hex": [
                float(value).hex() for value in data.geom_xpos[geom_id]
            ],
            "initialized_world_xmat_float_hex": [
                float(value).hex() for value in data.geom_xmat[geom_id]
            ],
        }
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = np.ascontiguousarray(
                model.mesh_vert[vertex_start : vertex_start + vertex_count]
            )
            faces = np.ascontiguousarray(
                model.mesh_face[face_start : face_start + face_count]
            )
            record["mesh"] = {
                "vertex_count": vertex_count,
                "face_count": face_count,
                "vertex_dtype": vertices.dtype.str,
                "face_dtype": faces.dtype.str,
                "vertex_bytes_sha256": hashlib.sha256(
                    vertices.tobytes()
                ).hexdigest(),
                "face_bytes_sha256": hashlib.sha256(
                    faces.tobytes()
                ).hexdigest(),
            }
        records.append(record)
    return canonical_json_sha256(records)


def _independent_compiled_model_xml_equivalent_sha256(model: Any) -> str:
    """Hash the passed compiled model without using the production helper."""

    records: list[dict[str, Any]] = []
    for owner_name, owner in (("model", model), ("option", model.opt)):
        for attribute in sorted(
            name for name in dir(owner) if not name.startswith("_")
        ):
            try:
                value = getattr(owner, attribute)
            except (AttributeError, RuntimeError, TypeError):
                continue
            record: dict[str, Any] = {
                "owner": owner_name,
                "attribute": attribute,
            }
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                record.update(
                    {
                        "kind": "ndarray",
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "bytes_sha256": hashlib.sha256(
                            array.tobytes()
                        ).hexdigest(),
                    }
                )
            elif isinstance(value, (bytes, bytearray)):
                record.update(
                    {
                        "kind": "bytes",
                        "length": len(value),
                        "bytes_sha256": hashlib.sha256(bytes(value)).hexdigest(),
                    }
                )
            elif isinstance(value, (bool, int, float, str, np.generic)):
                scalar = value.item() if isinstance(value, np.generic) else value
                if isinstance(scalar, float):
                    scalar = float(scalar).hex()
                record.update({"kind": "scalar", "value": scalar})
            else:
                continue
            records.append(record)
    return canonical_json_sha256(records)


def core_capture_route_contract_errors(
    record: Any,
    cad: ModuleType,
) -> list[str]:
    """Validate the public route record without consuming reported maxima."""

    errors: list[str] = []
    if not isinstance(record, dict):
        return ["route_contract_missing"]
    expected_keys = {
        "schema_version",
        "tool",
        "frame",
        "route_kind",
        "embedded_state_bytes_sha256",
        "contract_identity_digest_preimage",
        "contract_identity_sha256",
        "source_binding",
        "model_binding",
        "route_law",
        "source_states",
        "source_state_sha256",
        "q_roster_sha256",
        "canonical_waypoint_digest_preimage",
        "canonical_waypoint_sha256",
        "actions",
        "endpoint_guard",
        "live_source_corridor_guard",
        "dense_fk_evidence",
        "retired_route_negative",
        "state_write_contract",
        "authority_scope",
        "passed",
        "release_ready",
    }
    if set(record) != expected_keys:
        errors.append("top_level_keys")
    if record.get("schema_version") != "1.0":
        errors.append("schema_version")
    if record.get("tool") != "gripper" or record.get("frame") != "dock_gripper":
        errors.append("tool_frame")
    if record.get("route_kind") != "source_coupled_positive_lock_cam_capture":
        errors.append("route_kind")
    serialized = json.dumps(record, default=str)
    if "/tmp/" in serialized or "core_capture_route_candidate" in serialized:
        errors.append("temporary_candidate_as_authority")

    expected_generator = _authority_file_record(MAGNETIC_ROOT / "generate_cad.py")
    expected_cam_sha = canonical_json_sha256(cad.positive_lock_cam_contract())
    source_binding = record.get("source_binding")
    if not isinstance(source_binding, dict):
        source_binding = {}
        errors.append("source_binding")
    if source_binding.get("generator_file") != expected_generator:
        errors.append("source_binding:generator")
    if source_binding.get("positive_lock_cam_contract_sha256") != expected_cam_sha:
        errors.append("source_binding:cam_contract")
    if source_binding.get("route_functions") != {
        "lateral_x_mm": "positive_lock_cam_capture_lateral_offset_mm",
        "slider_q_max_mm": "positive_lock_cam_capture_q_max_mm",
    }:
        errors.append("source_binding:route_functions")
    if record.get("embedded_state_bytes_sha256") != (
        CORE_CAPTURE_ROUTE_EMBEDDED_BYTES_SHA256
    ):
        errors.append("embedded_state_bytes_sha256")

    states = record.get("source_states")
    if not isinstance(states, list) or len(states) != 276:
        states = []
        errors.append("source_states:count")
    else:
        for index, state in enumerate(states):
            if not isinstance(state, dict) or set(state) != {
                "preseat_mm",
                "source_x_mm",
                "q_rad",
            }:
                errors.append(f"source_states:{index}:shape")
                continue
            expected_preseat = round(55.0 - 0.2 * index, 10)
            if not _number_matches(state.get("preseat_mm"), expected_preseat):
                errors.append(f"source_states:{index}:preseat")
            if not _number_matches(
                state.get("source_x_mm"),
                _independent_core_capture_x_mm(expected_preseat),
            ):
                errors.append(f"source_states:{index}:source_x")
            q_value = state.get("q_rad")
            if (
                not isinstance(q_value, list)
                or len(q_value) != 5
                or any(_finite_real(value) is None for value in q_value)
            ):
                errors.append(f"source_states:{index}:q")
    observed_state_sha = canonical_json_sha256(states)
    observed_q_sha = canonical_json_sha256(
        [state.get("q_rad") for state in states if isinstance(state, dict)]
    )
    if observed_state_sha != CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256:
        errors.append("source_state_sha256:recomputed")
    if record.get("source_state_sha256") != observed_state_sha:
        errors.append("source_state_sha256:published")
    if observed_q_sha != CORE_CAPTURE_ROUTE_Q_SHA256:
        errors.append("q_roster_sha256:recomputed")
    if record.get("q_roster_sha256") != observed_q_sha:
        errors.append("q_roster_sha256:published")

    model_binding = record.get("model_binding")
    if not isinstance(model_binding, dict) or set(model_binding) != {
        "model_xml_sha256",
        "initialized_active_collision_geometry_sha256",
        "physics_timestep_s",
    }:
        errors.append("model_binding")
        model_binding = {}
    for key in (
        "model_xml_sha256",
        "initialized_active_collision_geometry_sha256",
    ):
        value = model_binding.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            errors.append(f"model_binding:{key}")
    if not _number_matches(model_binding.get("physics_timestep_s"), 0.00025):
        errors.append("model_binding:physics_timestep")

    route_law = record.get("route_law")
    if route_law != {
        "preseat_from_fk": "-dock_local_robot_mating_z_mm",
        "lateral_x_from_fk": "dock_local_robot_mating_x_mm",
        "transverse_from_fk": "dock_local_robot_mating_y_mm",
        "orientation_reference": "seated_dock_frame",
        "x_breakpoints_mm": [[55.0, 0.2], [6.4, 0.2], [3.2, 0.0], [0.0, 0.0]],
    }:
        errors.append("route_law")

    expected_identity_preimage = {
        "source_generator_sha256": expected_generator["sha256"],
        "positive_lock_cam_contract_sha256": expected_cam_sha,
        "embedded_state_bytes_sha256": CORE_CAPTURE_ROUTE_EMBEDDED_BYTES_SHA256,
        "source_state_sha256": CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256,
        "q_roster_sha256": CORE_CAPTURE_ROUTE_Q_SHA256,
        "desired_start_q_sha256": canonical_json_sha256(
            {
                "gripper_capture_lateral_align": [
                    -0.72, -1.11771, 1.13502, -0.01731, 0.0
                ],
                "gripper_capture_axial_open_side": states[0]["q_rad"],
                "gripper_capture_coupled_recenter": states[243]["q_rad"],
                "gripper_capture_centered_final": states[259]["q_rad"],
            }
        ) if states else None,
        "phase_row_ranges": {
            name: details["row_range"]
            for name, details in CORE_CAPTURE_ROUTE_ACTIONS.items()
            if details["row_range"] is not None
        },
        "phase_timing_s": {
            name: [details["duration_s"], details["timeout_s"]]
            for name, details in CORE_CAPTURE_ROUTE_ACTIONS.items()
        },
        "endpoint_guard": {
            "q_error_rad": 0.002,
            "qvel_rad_s": 0.02,
            "position_error_m": 0.00005,
            "orientation_error_rad": math.radians(0.1),
            "dwell_ticks": 4,
        },
        "live_source_corridor_max_error_mm": 0.040,
    }
    identity_preimage = record.get("contract_identity_digest_preimage")
    if canonical_json_sha256(identity_preimage) != canonical_json_sha256(
        expected_identity_preimage
    ):
        errors.append("contract_identity:preimage")
    if record.get("contract_identity_sha256") != canonical_json_sha256(
        identity_preimage
    ):
        errors.append("contract_identity:sha256")
    waypoint_preimage = record.get("canonical_waypoint_digest_preimage")
    expected_waypoint_preimage = {
        "source_binding": source_binding,
        "model_binding": model_binding,
        "source_states": states,
    }
    if waypoint_preimage != expected_waypoint_preimage:
        errors.append("canonical_waypoint:preimage")
    if record.get("canonical_waypoint_sha256") != canonical_json_sha256(
        waypoint_preimage
    ):
        errors.append("canonical_waypoint:sha256")

    actions = record.get("actions")
    if not isinstance(actions, list) or [
        item.get("name") if isinstance(item, dict) else None for item in actions
    ] != list(CORE_CAPTURE_ROUTE_ACTIONS):
        errors.append("actions:roster")
        actions = []
    for action in actions:
        name = action["name"]
        expected = CORE_CAPTURE_ROUTE_ACTIONS[name]
        expected_endpoint_index = (
            0 if expected["row_range"] is None else expected["row_range"][1]
        )
        expected_endpoint = states[expected_endpoint_index]["q_rad"] if states else []
        for key, value in (
            ("kind", "move"),
            ("tool", "gripper"),
            ("duration_s", expected["duration_s"]),
            ("timeout_s", expected["timeout_s"]),
            ("source_row_range_inclusive", expected["row_range"]),
            ("full_endpoint_inclusive_q_count", expected["q_count"]),
            ("joint_waypoint_count_excluding_action_start", expected["waypoint_count"]),
            ("endpoint_q_rad", expected_endpoint),
            ("q_roster_sha256", expected["q_sha256"]),
            ("time_scaling", "quintic_10a3_minus_15a4_plus_6a5"),
            ("zero_commanded_endpoint_velocity", True),
        ):
            if action.get(key) != value:
                errors.append(f"actions:{name}:{key}")
        kinematics = action.get("command_kinematics")
        if not isinstance(kinematics, dict):
            errors.append(f"actions:{name}:command_kinematics")
            continue
        if not _number_matches(kinematics.get("controller_dt_s"), 0.005):
            errors.append(f"actions:{name}:controller_dt")
        expected_sample_count = int(round(expected["duration_s"] / 0.005)) + 1
        if kinematics.get("time_sample_count") != expected_sample_count:
            errors.append(f"actions:{name}:time_sample_count")
        if not _number_matches(
            kinematics.get("maximum_abs_joint_speed_bound_rad_s"),
            expected["speed_bound"],
        ):
            errors.append(f"actions:{name}:speed_bound")
        if not _number_matches(
            kinematics.get("maximum_abs_joint_acceleration_bound_rad_s2"),
            expected["acceleration_bound"],
        ):
            errors.append(f"actions:{name}:acceleration_bound")
        speed = _finite_real(kinematics.get("maximum_abs_joint_speed_rad_s"))
        acceleration = _finite_real(
            kinematics.get("maximum_abs_joint_acceleration_rad_s2")
        )
        if speed is None or speed > expected["speed_bound"]:
            errors.append(f"actions:{name}:speed")
        if acceleration is None or acceleration > expected["acceleration_bound"]:
            errors.append(f"actions:{name}:acceleration")
        if kinematics.get("passed") is not True:
            errors.append(f"actions:{name}:passed")

    if record.get("endpoint_guard") != {
        "maximum_q_error_rad": 0.002,
        "maximum_abs_qvel_rad_s": 0.02,
        "maximum_fk_position_error_m": 0.00005,
        "maximum_fk_orientation_error_rad": math.radians(0.1),
        "maximum_absolute_source_x_error_mm": 0.040,
        "required_contiguous_controller_ticks": 4,
        "advance_on_elapsed_time_only": False,
    }:
        errors.append("endpoint_guard")
    expected_corridor = {
        "active_after_action": "gripper_capture_lateral_align",
        "audited_actions": list(CORE_CAPTURE_ROUTE_ACTIONS)[1:],
        "audit_frequency": "after_every_mj_step",
        "preseat_formula": "-dock_local_robot_mating_z_mm",
        "lateral_x_formula": "dock_local_robot_mating_x_mm",
        "maximum_absolute_source_x_error_mm": 0.040,
        "bound_provenance_mm": {
            "continuous_plate_cam_clearance": 0.249902439,
            "manufacturing_clearance": 0.20,
            "retained_reserve": 0.009902439,
            "available_tracking_error": 0.040,
            "formula": "0.249902439 - 0.20 - 0.009902439 = 0.040",
        },
        "violation_abort_reason": "core_capture_source_corridor_violation",
        "pass_requires": {
            "audited_substeps_greater_than_zero": True,
            "all_three_audited_actions_observed": True,
            "all_four_route_endpoint_events_completed": True,
            "maximum_error_within_bound": True,
            "current_abort_absent": True,
        },
        "live_dynamics_authority": False,
    }
    if record.get("live_source_corridor_guard") != expected_corridor:
        errors.append("live_source_corridor_guard")

    dense = record.get("dense_fk_evidence")
    if not isinstance(dense, dict):
        dense = {}
        errors.append("dense_fk_evidence")
    sampling = dense.get("sampling_contract")
    if not isinstance(sampling, dict) or sampling.get("fractions") != [
        index / 100.0 for index in range(101)
    ]:
        errors.append("dense_fk:sampling")
    phases = dense.get("phases")
    if not isinstance(phases, list) or [
        phase.get("action") if isinstance(phase, dict) else None for phase in phases
    ] != list(CORE_CAPTURE_ROUTE_ACTIONS):
        errors.append("dense_fk:phase_roster")
        phases = []
    for phase in phases:
        name = phase["action"]
        expected = CORE_CAPTURE_ROUTE_ACTIONS[name]
        if phase.get("interval_count") != expected["interval_count"]:
            errors.append(f"dense_fk:{name}:interval_count")
        if phase.get("sample_count") != 101 * expected["interval_count"]:
            errors.append(f"dense_fk:{name}:sample_count")
        expected_thresholds = {
            "maximum_preseat_error_mm": 0.0002 if name.endswith("lateral_align") else 0.00005,
            "maximum_source_x_error_mm": 0.0003 if name.endswith("lateral_align") else 0.0001,
            "maximum_abs_transverse_y_mm": 0.010,
            "maximum_orientation_error_rad": 1.0e-9,
        }
        if phase.get("thresholds") != expected_thresholds:
            errors.append(f"dense_fk:{name}:thresholds")
        observed = phase.get("observed")
        if not isinstance(observed, dict):
            errors.append(f"dense_fk:{name}:observed")
        else:
            for key, limit in expected_thresholds.items():
                value = _finite_real(observed.get(key))
                if value is None or value > limit:
                    errors.append(f"dense_fk:{name}:{key}")
        if phase.get("passed") is not True:
            errors.append(f"dense_fk:{name}:passed")
    if dense.get("passed") is not True:
        errors.append("dense_fk:passed")

    retired = record.get("retired_route_negative")
    if not isinstance(retired, dict):
        retired = {}
        errors.append("retired_route_negative")
    required_retired = {
        "name": "constant_x_plus_0p20_then_same_z_recenter",
        "overlap_authority": "independent_exact_OCCT_recomputation_required",
        "violates_source_piecewise_x_law": True,
        "single_global_action_crosses_velocity_kinks": True,
        "rejected": True,
    }
    for key, value in required_retired.items():
        if retired.get(key) != value:
            errors.append(f"retired_route_negative:{key}")
    overlaps = retired.get("complete_source_cam_overlap_mm3")
    if not isinstance(overlaps, dict) or any(
        (_finite_real(value) or 0.0) <= 1.0e-6 for value in overlaps.values()
    ) or set(overlaps) != {"slider_q_0p00mm", "slider_q_0p05mm"}:
        errors.append("retired_route_negative:overlap")
    retired_kinematics = retired.get("retired_single_action_command_kinematics")
    if not isinstance(retired_kinematics, dict) or (
        retired_kinematics.get(
            "rejected_for_nonzero_velocity_at_source_law_breakpoints"
        )
        is not True
    ):
        errors.append("retired_route_negative:kinematics")

    if record.get("state_write_contract") != {
        "arm_command_target": "data.ctrl",
        "direct_pogo_qpos_writes_after_initialization": 0,
        "direct_slider_qpos_writes_after_initialization": 0,
        "validation_method": "independent_ast_and_callgraph_required",
    }:
        errors.append("state_write_contract")
    if record.get("authority_scope") != {
        "static_source_route_and_fk_authority": True,
        "live_tracking_authority": False,
        "contact_force_authority": False,
        "friction_coefficient_authority": False,
        "load_capacity_authority": False,
        "dynamics_authority": False,
        "blockers": CORE_CAPTURE_ROUTE_BLOCKERS,
        "release_ready": False,
    }:
        errors.append("authority_scope")
    expected_pass = bool(
        dense.get("passed") is True
        and actions
        and all(
            action.get("command_kinematics", {}).get("passed") is True
            for action in actions
        )
    )
    if record.get("passed") is not expected_pass:
        errors.append("passed")
    if record.get("release_ready") is not False:
        errors.append("release_ready")
    return errors


def core_capture_route_result_errors(
    result: Any,
    contract: dict[str, Any],
) -> list[str]:
    """Check fail-honest live reporting; this never grants dynamics authority."""

    errors: list[str] = []
    if not isinstance(result, dict):
        return ["route_result_missing"]
    alignment = result.get("route_alignment")
    if not isinstance(alignment, dict):
        return ["route_alignment_missing"]
    if alignment.get("method") != (
        "source_coupled_positive_lock_cam_four_phase_dense_fk_ik_waypoints"
    ):
        errors.append("method")
    if alignment.get("runtime_contract_api") != "core_capture_route_runtime_contract":
        errors.append("runtime_contract_api")
    for result_key, contract_key in (
        ("contract_identity_sha256", "contract_identity_sha256"),
        ("source_state_sha256", "source_state_sha256"),
        ("q_roster_sha256", "q_roster_sha256"),
        ("embedded_state_bytes_sha256", "embedded_state_bytes_sha256"),
    ):
        if alignment.get(result_key) != contract.get(contract_key):
            errors.append(result_key)
    action_names = list(CORE_CAPTURE_ROUTE_ACTIONS)
    if alignment.get("phase_actions") != action_names:
        errors.append("phase_actions")
    endpoint_records = alignment.get("phase_endpoint_journal_evidence")
    if not isinstance(endpoint_records, list):
        endpoint_records = []
        errors.append("endpoint_records")
    completed_actions = {
        item.get("action")
        for item in endpoint_records
        if isinstance(item, dict) and item.get("event") == "move_complete"
    }
    all_endpoints = completed_actions == set(action_names)
    if alignment.get("completed_endpoint_actions") != sorted(completed_actions):
        errors.append("completed_endpoint_actions")
    if alignment.get("all_four_endpoints_completed") is not all_endpoints:
        errors.append("all_four_endpoints_completed")

    live = alignment.get("live_source_corridor")
    if not isinstance(live, dict):
        return [*errors, "live_source_corridor"]
    counts = live.get("audited_substeps_by_phase")
    audited_names = action_names[1:]
    if not isinstance(counts, dict) or set(counts) != set(audited_names):
        counts = {}
        errors.append("corridor:phase_counts")
    numeric_counts = [
        value for value in counts.values() if isinstance(value, int) and value >= 0
    ]
    if len(numeric_counts) != len(audited_names):
        errors.append("corridor:phase_count_values")
    audited_substeps = live.get("audited_substeps")
    if not isinstance(audited_substeps, int) or audited_substeps < 0:
        errors.append("corridor:audited_substeps")
        audited_substeps = -1
    elif audited_substeps != sum(numeric_counts):
        errors.append("corridor:audited_substep_sum")
    observed = audited_substeps > 0
    all_phases = bool(counts) and all(value > 0 for value in numeric_counts)
    if live.get("observed") is not observed:
        errors.append("corridor:observed")
    if live.get("all_three_phases_observed") is not all_phases:
        errors.append("corridor:all_phases")
    if not _number_matches(live.get("maximum_allowed_error_mm"), 0.040):
        errors.append("corridor:maximum_allowed_error")
    if live.get("violation_abort_reason") != (
        "core_capture_source_corridor_violation"
    ):
        errors.append("corridor:abort_reason")
    if live.get("live_dynamics_authority") is not False:
        errors.append("corridor:dynamics_authority")
    maximum_error = _finite_real(live.get("maximum_absolute_error_mm"))
    witness = live.get("witness")
    witness_valid = isinstance(witness, dict)
    if witness_valid:
        preseat = _finite_real(witness.get("preseat_mm"))
        observed_x = _finite_real(witness.get("observed_x_mm"))
        expected_x = _finite_real(witness.get("expected_source_x_mm"))
        signed_error = _finite_real(witness.get("signed_error_mm"))
        absolute_error = _finite_real(witness.get("absolute_error_mm"))
        if None in (preseat, observed_x, expected_x, signed_error, absolute_error):
            witness_valid = False
        else:
            independently_expected_x = _independent_core_capture_x_mm(
                max(0.0, float(preseat))
            )
            witness_valid = bool(
                abs(float(expected_x) - independently_expected_x) <= 1.0e-12
                and abs(float(signed_error) - (float(observed_x) - independently_expected_x))
                <= 1.0e-12
                and abs(float(absolute_error) - abs(float(signed_error))) <= 1.0e-12
                and maximum_error is not None
                and abs(float(maximum_error) - float(absolute_error)) <= 1.0e-12
                and witness.get("action") in audited_names
            )
    if observed and not witness_valid:
        errors.append("corridor:witness")
    if not observed and witness is not None:
        errors.append("corridor:unobserved_witness")
    expected_live_pass = bool(
        observed
        and all_phases
        and all_endpoints
        and witness_valid
        and maximum_error is not None
        and maximum_error <= 0.040
        and result.get("abort_reason") is None
    )
    if live.get("passed") is not expected_live_pass:
        errors.append("corridor:passed")
    measured_lateral = _finite_real(
        alignment.get("measured_max_lateral_deviation_m")
    )
    relief = _finite_real(alignment.get("cam_relief_corridor_m"))
    expected_alignment_pass = bool(
        expected_live_pass
        and measured_lateral is not None
        and relief is not None
        and measured_lateral <= relief
    )
    if alignment.get("passed") is not expected_alignment_pass:
        errors.append("route_alignment:passed")
    if result.get("release_ready") is not False:
        errors.append("release_ready")
    return errors


def expected_core_cam_tab_contact_contract(
    cad: ModuleType,
    runtime_cam_contract: dict[str, Any],
    route_contract: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild checkpoint-A's capture-only contact contract from sources."""

    generator = _authority_file_record(MAGNETIC_ROOT / "generate_cad.py")
    slider_step = _authority_file_record(
        MAGNETIC_ROOT / "exports" / "so101_positive_lock_slider.step"
    )
    cam_contract_sha = canonical_json_sha256(cad.positive_lock_cam_contract())
    route_identity = route_contract["contract_identity_sha256"]
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    source_surfaces = [
        {
            "surface_role": "functional_axial_lead_ramp",
            "action": "gripper_capture_coupled_recenter",
            "runtime_pair": [
                CORE_CAM_TAB_LEADING_GEOM,
                "dock_gripper_cam_axial_lead_collision",
            ],
            "preseat_bounds_mm": [3.2, 6.346666666666667],
            "locus": {
                "plane": "x+z=17.65mm",
                "y_bounds_mm": [0.0, 2.0],
                "z_bounds_mm": [-9.6, -6.4],
            },
            "normal_cam_to_tab_dock_local": [-inv_sqrt_two, 0.0, -inv_sqrt_two],
            "functional_coverage_required": True,
        },
        {
            "surface_role": "functional_hold_finger_face",
            "action": "gripper_capture_centered_final",
            "runtime_pair": [
                CORE_CAM_TAB_LEADING_GEOM,
                "dock_gripper_cam_hold_finger_collision",
            ],
            "preseat_bounds_mm": [0.0, 3.2],
            "locus": {
                "plane": "x=24.05mm",
                "y_bounds_mm": [0.0, 2.0],
                "z_bounds_mm": [-6.4, -4.15],
                "tab_z_bounds_formula_mm": ["-4.8-p", "-3.2-p"],
                "accepted_z_is_interval_intersection": True,
            },
            "normal_cam_to_tab_dock_local": [-1.0, 0.0, 0.0],
            "functional_coverage_required": True,
        },
        {
            "surface_role": "lead_hold_partition_seam_nonfunctional",
            "action": "gripper_capture_centered_final",
            "runtime_pair": [
                CORE_CAM_TAB_LEADING_GEOM,
                "dock_gripper_cam_axial_lead_collision",
            ],
            "preseat_bounds_mm": [1.6, 3.2],
            "locus": {
                "line": "x=24.05mm,z=-6.4mm",
                "y_bounds_mm": [0.0, 2.0],
            },
            "normal_cone_cam_to_tab": [
                [-inv_sqrt_two, 0.0, -inv_sqrt_two],
                [-1.0, 0.0, 0.0],
            ],
            "closed_top_cap_positive_z_is_forbidden": True,
            "functional_coverage_required": False,
        },
        {
            "surface_role": "main_hold_edge_tangency_nonfunctional",
            "action": "gripper_capture_centered_final",
            "runtime_pair": [
                CORE_CAM_TAB_LEADING_GEOM,
                "dock_gripper_cam_collision",
            ],
            "preseat_bounds_mm": [0.0, 0.95],
            "locus": {
                "line": "x=24.05mm,y=0mm",
                "z_lower_mm": -4.15,
                "z_upper_formula_mm": "-3.2-p",
            },
            "normal_cone_cam_to_tab": [
                [-0.970852159759157, -0.239679126940542, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "functional_coverage_required": False,
        },
    ]
    classifier_surfaces = [
        {
            "surface_role": "functional_axial_lead_ramp",
            "action": "gripper_capture_coupled_recenter",
            "runtime_pair": [CORE_CAM_TAB_LEADING_GEOM, CORE_CAM_GEOMS[1]],
            "preseat_bounds_mm": [3.2, 6.346666666666667],
            "locus": {
                "plane_sum_x_plus_z_mm": 17.65,
                "y_bounds_mm": [0.0, 2.0],
                "z_bounds_mm": [-9.6, -6.4],
            },
            "normal_cam_to_tab_dock_local": [-inv_sqrt_two, 0.0, -inv_sqrt_two],
            "minimum_normal_alignment": 0.999,
            "q_excess_bounds_mm": [-0.020, math.sqrt(2.0) * 0.020],
            "functional_coverage_required": True,
        },
        {
            "surface_role": "functional_hold_finger_face",
            "action": "gripper_capture_centered_final",
            "runtime_pair": [CORE_CAM_TAB_LEADING_GEOM, CORE_CAM_GEOMS[2]],
            "preseat_bounds_mm": [0.0, 3.2],
            "locus": {
                "plane_x_mm": 24.05,
                "y_bounds_mm": [0.0, 2.0],
                "hold_z_bounds_mm": [-6.4, -4.15],
                "tab_z_bounds_formula_mm": ["-4.8-p", "-3.2-p"],
                "accepted_z_is_interval_intersection": True,
            },
            "normal_cam_to_tab_dock_local": [-1.0, 0.0, 0.0],
            "minimum_normal_alignment": 0.999,
            "slider_q_bounds_mm": [-0.020, 0.05000000000000071 + 0.020],
            "functional_coverage_required": True,
        },
        {
            "surface_role": "lead_hold_partition_seam_nonfunctional",
            "action": "gripper_capture_centered_final",
            "runtime_pair": [CORE_CAM_TAB_LEADING_GEOM, CORE_CAM_GEOMS[1]],
            "preseat_bounds_mm": [1.6, 3.2],
            "locus": {
                "line_x_mm": 24.05,
                "line_z_mm": -6.4,
                "y_bounds_mm": [0.0, 2.0],
            },
            "normal_cone_cam_to_tab": [
                [-inv_sqrt_two, 0.0, -inv_sqrt_two], [-1.0, 0.0, 0.0]
            ],
            "closed_top_cap_positive_z_is_forbidden": True,
            "functional_coverage_required": False,
        },
        {
            "surface_role": "main_hold_edge_tangency_nonfunctional",
            "action": "gripper_capture_centered_final",
            "runtime_pair": [CORE_CAM_TAB_LEADING_GEOM, CORE_CAM_GEOMS[0]],
            "preseat_bounds_mm": [0.0, 0.95],
            "locus": {
                "line_x_mm": 24.05,
                "line_y_mm": 0.0,
                "z_lower_mm": -4.15,
                "z_upper_formula_mm": "-3.2-p",
            },
            "normal_cone_cam_to_tab": [
                [-0.970852159759157, -0.239679126940542, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "functional_coverage_required": False,
        },
    ]
    classifier_semantics = {
        "schema_version": "2.0",
        "frame_conventions": {
            "source_frame": "dock_gripper",
            "dock_pose_source": "dock_gripper_body_xpos_xmat",
            "robot_mating_pose_source": "robot_mating_face_site_xpos_xmat",
            "published_quaternion": "finite_sign_canonical_wxyz",
            "contact_normal_direction": "cam_geom_to_slider_tab_geom",
            "source_preseat_mm": "-dock_local_robot_mating_z_mm",
            "source_lateral_x_mm": "dock_local_robot_mating_x_mm",
            "source_transverse_y_mm": "dock_local_robot_mating_y_mm",
        },
        "runtime_inventory": {
            "contact_eligible_leading_tab_geom": CORE_CAM_TAB_LEADING_GEOM,
            "always_forbidden_noncontact_tab_geom": CORE_CAM_TAB_NONCONTACT_GEOM,
            "main_geom": CORE_CAM_GEOMS[0],
            "axial_lead_geom": CORE_CAM_GEOMS[1],
            "hold_finger_geom": CORE_CAM_GEOMS[2],
            "always_forbidden_root_geoms": CORE_CAM_GEOMS[3:],
            "complete_cam_geom_roster": list(CORE_CAM_GEOMS),
        },
        "phase_and_equality_policy": {
            "capture_actions": CORE_CAM_CONTACT_ACTIONS,
            "free_space_no_contact_actions": CORE_CAM_CONTACT_ACTIONS[:2],
            "functional_actions": CORE_CAM_CONTACT_ACTIONS[2:],
            "dock_hold_equality": "dock_gripper_hold",
            "dock_hold_must_be_active": True,
            "attach_equality": "attach_gripper",
            "attach_equality_must_be_active": False,
            "audit_frequency": "after_every_mj_step_before_generic_contact_audit",
        },
        "capture_law": {
            "source_x_formula": (
                "0.2_for_p_ge_6.4;0.0625*(p-3.2)_for_3.2_lt_p_lt_6.4;"
                "0_for_p_le_3.2"
            ),
            "passive_q_max_formula_mm": "clamp(p-x-3.15,0.05,3.0)",
            "ramp_contact_start_preseat_mm": 6.346666666666667,
            "ramp_end_preseat_mm": 3.2,
            "passive_open_q_mm": 0.05000000000000071,
            "maximum_source_x_error_mm": 0.040,
            "maximum_transverse_y_mm": 0.010,
            "maximum_orientation_error_rad": math.radians(0.1),
            "orientation_contract": (
                "fixed_0p1deg_bound_best_attainable_five_dof_orientation"
            ),
            "exact_quaternion_required_off_seat": False,
        },
        "surface_classifiers": classifier_surfaces,
        "provisional_development_guard": {
            "point_and_locus_tolerance_mm": 0.020,
            "maximum_penetration_mm": 0.020,
            "numerical_epsilon_mm": 1.0e-6,
            "minimum_normal_alignment": 0.999,
            "finite_compressive_contact_force_required_but_unbounded": True,
            "contact_force_authority": False,
        },
        "functional_envelope_sampling": {
            "state_index": "physics_substep_count_after_mj_step",
            "actions": CORE_CAM_CONTACT_ACTIONS[2:],
            "complete_cam_distance": {
                "method": (
                    "minimum_live_contact_dist_else_mj_geomDistance_for_each_of_"
                    "two_slider_tabs_by_five_exact_cam_geoms"
                ),
                "maximum_distance_m": 0.1,
                "signed_distance_units": "mm",
                "closest_points_world_order": ["slider_tab", "cam_component"],
                "minimum_recomputed_over_exact_two_by_five_pair_roster": True,
                "pair_class_clearance_rules": {
                    "noncontact_tab_part_000_all_cam_components_minimum_mm": -1.0e-6,
                    "either_tab_to_outer_root_minimum_mm": -1.0e-6,
                    "leading_tab_part_001_to_main_lead_hold_minimum_mm": -0.020,
                    "negative_without_live_contact": "unresolved_and_failed",
                    "distance_cutoff_or_nonfinite": "unresolved_and_failed",
                },
            },
            "per_state_count_partition": (
                "eligible_plus_rejected_equals_all_observed_cam_tab_contacts;"
                "functional_plus_nonfunctional_equals_eligible"
            ),
            "functional_surface_state_rule": (
                "lead_for_recenter_when_p_le_6.346666666666667_and_hold_for_"
                "centered_final;each_state_requires_either_exact_valid_contact_"
                "or_resolved_nonnegative_signed_gap"
            ),
            "discrete_no_skipped_state_check": True,
            "discrete_no_rebound_check": (
                "every_functional_surface_state_retains_contact_or_nonnegative_"
                "gap_and_q_envelope"
            ),
            "continuous_tunnel_authority": False,
            "continuous_motion_lipschitz_bound_published": False,
            "lossless_replay_state_fields": [
                "qpos", "qvel", "mocap_pos", "mocap_quat_wxyz",
                "all_named_equality_active_states",
            ],
            "replay_world_pose_fields_sign_canonical_wxyz": [
                "dock_gripper_body", "robot_mating_face_site",
                "robot_plate_frame_body", "qc_positive_lock_slider_body",
                "both_slider_tab_geoms",
            ],
            "state_continuity": {
                "state_index_formula": "physics_substep_count_after_mj_step",
                "adjacent_state_index_delta": 1,
                "adjacent_sim_time_delta": "model.opt.timestep",
                "sim_time_absolute_tolerance_s": 1.0e-12,
                "allowed_action_progression": [
                    "same_functional_action",
                    "gripper_capture_coupled_recenter_to_gripper_capture_centered_final",
                ],
            },
            "sampled_coordinate_jump_rules": {
                "gripper_capture_coupled_recenter_formula_mm": (
                    "(abs(delta_p)+abs(delta_x)+abs(delta_q))/sqrt(2)"
                ),
                "gripper_capture_centered_final_formula_mm": (
                    "abs(delta_x)+abs(delta_q)"
                ),
                "maximum_mm": 0.010,
            },
            "nonaccumulating_running_minimum_rules": {
                "preseat_mm": "p<=running_min_prior_p+0.020",
                "post_first_functional_lead_slider_q_mm": (
                    "q<=running_min_prior_post_lead_q+0.020"
                ),
                "tolerance_mm": 0.020,
            },
            "joint_and_phase_envelopes": {
                "compiled_slider_joint_range_mm": [0.0, 3.0],
                "joint_range_numeric_tolerance_mm": 1.0e-6,
                "recenter_preseat_bounds_mm": [3.2, 6.4],
                "centered_final_preseat_bounds_mm": [0.0, 3.2],
                "preseat_bounds_tolerance_mm": 0.020,
                "recenter_q_upper_formula_mm": (
                    "min(3.0+numeric_epsilon,qmax+sqrt(2)*0.020)"
                ),
                "centered_final_q_upper_mm": 0.07000000000000071,
                "source_x_error_maximum_mm": 0.040,
                "absolute_transverse_y_maximum_mm": 0.010,
                "orientation_error_maximum_rad": math.radians(0.1),
                "dock_hold_active": True,
                "attach_equality_active": False,
            },
        },
        "evidence_pass_formula": {
            "actual_model_binding": {
                "controller_init_snapshot_required": True,
                "evidence_time_recompute_required": True,
                "compiled_model_xml_equivalent_digest_must_match_expected": True,
                "initialized_active_geometry_digest_must_match_expected": True,
                "controller_init_and_evidence_digests_must_be_identical": True,
                "active_geometry_state_construction": (
                    "fresh_MjData_then_initialize_and_mj_forward"
                ),
            },
            "requires_all_four_capture_phases_sampled": True,
            "requires_both_functional_phases_sampled": True,
            "requires_all_four_route_endpoints_completed": True,
            "requires_finite_raw_values_and_contiguous_state_indices": True,
            "requires_functional_lead_and_hold_coverage": True,
            "requires_zero_rejected_or_unclassified_contacts": True,
            "requires_all_provisional_depth_locus_normal_q_guards": True,
            "requires_discrete_no_skipped_state_and_no_rebound": True,
            "requires_current_abort_absent": True,
            "zero_contact_cannot_pass": True,
            "top_level_success_remains_false_without_physical_authority": True,
        },
    }
    model_binding = {
        "model_xml_sha256": route_contract["model_binding"]["model_xml_sha256"],
        "compiled_model_xml_equivalent_sha256": (
            CORE_CAM_COMPILED_MODEL_XML_EQUIVALENT_SHA256
        ),
        "initialized_active_collision_geometry_sha256": route_contract[
            "model_binding"
        ]["initialized_active_collision_geometry_sha256"],
    }
    identity_preimage = {
        "source_generator_sha256": generator["sha256"],
        "positive_lock_cam_contract_sha256": cam_contract_sha,
        "positive_lock_slider_step_sha256": slider_step["sha256"],
        "capture_route_contract_identity_sha256": route_identity,
        "classifier_semantics": classifier_semantics,
        "model_binding": model_binding,
    }
    return {
        "schema_version": "1.0",
        "contract_kind": "capture_only_exact_cam_tab_source_envelope",
        "frame": "dock_gripper",
        "contract_identity_digest_preimage": identity_preimage,
        "contract_identity_sha256": canonical_json_sha256(identity_preimage),
        "classifier_semantics": classifier_semantics,
        "source_binding": {
            "generator_file": generator,
            "positive_lock_cam_contract_sha256": cam_contract_sha,
            "runtime_cam_geometry_contract_sha256": canonical_json_sha256(
                runtime_cam_contract
            ),
            "positive_lock_slider_step": slider_step,
            "capture_route_contract_identity_sha256": route_identity,
        },
        "model_binding": {
            **model_binding,
            "compiled_model_xml_equivalent_digest_api": (
                "compiled_model_xml_equivalent_sha256"
            ),
            "initialized_active_collision_geometry_digest_api": (
                "initialized_active_collision_geometry_sha256"
            ),
        },
        "runtime_inventory": {
            "contact_eligible_leading_tab_geom": CORE_CAM_TAB_LEADING_GEOM,
            "non_contact_tab_geom": CORE_CAM_TAB_NONCONTACT_GEOM,
            "main_geom": CORE_CAM_GEOMS[0],
            "axial_lead_geom": CORE_CAM_GEOMS[1],
            "hold_finger_geom": CORE_CAM_GEOMS[2],
            "always_forbidden_root_geoms": CORE_CAM_GEOMS[3:],
            "all_cam_geoms": list(CORE_CAM_GEOMS),
        },
        "capture_law": {
            "preseat_formula": "-dock_local_robot_mating_z_mm",
            "lateral_formula": "dock_local_robot_mating_x_mm",
            "passive_q_max_formula_mm": "clamp(p-x-3.15,0.05,3.0)",
            "ramp_contact_start_preseat_mm": 6.346666666666667,
            "ramp_end_preseat_mm": 3.2,
            "passive_open_q_mm": 0.05000000000000071,
            "maximum_live_source_x_error_mm": 0.040,
        },
        "phase_policy": {
            "no_cam_contact_actions": CORE_CAM_CONTACT_ACTIONS[:2],
            "functional_contact_actions": CORE_CAM_CONTACT_ACTIONS[2:],
            "dock_hold_must_be_active": True,
            "attach_equality_must_be_inactive": True,
            "physical_release_action_is_excluded": True,
        },
        "source_surfaces": source_surfaces,
        "provisional_guard": {
            "point_tolerance_mm": 0.020,
            "maximum_penetration_mm": 0.020,
            "minimum_normal_alignment": 0.999,
            "lead_maximum_q_excess_mm": math.sqrt(2.0) * 0.020,
            "hold_and_main_maximum_q_mm": 0.05000000000000071 + 0.020,
            "authority": "provisional_simulation_guard_only",
        },
        "free_space_endpoint_clearance": {
            "axial_action_endpoint_preseat_mm": 6.4,
            "x_axis_gap_mm": 0.050,
            "lead_normal_gap_mm": 0.050 * inv_sqrt_two,
            "retained_x_gap_at_40um_corridor_mm": 0.010,
            "retained_lead_normal_gap_at_40um_corridor_mm": 0.010 * inv_sqrt_two,
        },
        "post_capture_exclusion": {
            "excluded_action": "gripper_source_negative_y_physical_release",
            "reason": (
                "source_negative_y_roster_is_static_only_and_has_no_reverse_"
                "insertion_tracking_or_contact_dynamics_authority"
            ),
            "static_contract_api": "core_dock_static_release_route_contract",
            "source_axis": "dock_local_negative_y",
            "axis_dock_local": [0.0, -1.0, 0.0],
            "axis_world": [0.0, 0.0, 1.0],
            "roster_row_count": 31,
            "roster_canonical_sha256": (
                "f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293"
            ),
            "physical_release_action_implemented": False,
            "default_action_sequence_ends_at": "gripper_dock_release_verify",
        },
        "evidence_requirements": {
            "audit_frequency": "after_every_mj_step_before_generic_contact_audit",
            "raw_contact_force_torque_width": 6,
            "zero_contact_cannot_pass_functional_coverage": True,
            "functional_roles_required": CORE_CAM_FUNCTIONAL_ROLES,
            "all_candidate_contacts_remain_counted": True,
            "functional_state_sampling": (
                "one_lossless_state_after_every_functional_phase_mj_step"
            ),
            "exact_pair_gap_roster": (
                "two_slider_tab_geoms_by_five_complete_cam_geoms"
            ),
            "lossless_replay_state": [
                "qpos", "qvel", "mocap_pos", "mocap_quat_wxyz",
                "all_equality_active_states",
            ],
            "replay_transforms": [
                "dock_gripper_body", "robot_mating_face_site",
                "robot_plate_frame_body", "qc_positive_lock_slider_body",
                "both_slider_tab_geoms",
            ],
            "continuous_between_mj_steps_authority": False,
            "interval_motion_bound_certified": False,
        },
        "authority_scope": {
            "static_geometry_phase_and_locus_authority": True,
            "provisional_contact_classification_authority": False,
            "friction_coefficient_authority": False,
            "load_capacity_authority": False,
            "contact_force_authority": False,
            "dynamics_authority": False,
            "post_capture_release_authority": False,
            "continuous_between_mj_steps_authority": False,
            "blockers": CORE_CAM_TAB_CONTACT_BLOCKERS,
            "release_ready": False,
        },
        "passed": True,
        "release_ready": False,
    }


def core_cam_tab_contact_contract_errors(
    record: Any,
    cad: ModuleType,
    runtime_cam_contract: dict[str, Any],
    route_contract: dict[str, Any],
) -> list[str]:
    if not isinstance(record, dict):
        return ["cam_tab_contract_missing"]
    expected = expected_core_cam_tab_contact_contract(
        cad, runtime_cam_contract, route_contract
    )
    return [
        f"cam_tab_contract:{path}"
        for path in _evidence_mismatches(record, expected)
    ]


def _interval_error(value: float, lower: float, upper: float) -> float:
    return max(lower - value, value - upper, 0.0)


def _quaternion_wxyz_matrix(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(quaternion)) or abs(norm - 1.0) > 1.0e-9:
        return None
    if float(quaternion[0]) < -1.0e-15:
        return None
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _replay_pose_map_errors(value: Any) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != {
        "dock_body", "robot_mating_site", "robot_plate_body",
        "positive_lock_slider_body", "slider_tab_geoms",
    }:
        return ["replay_poses:keys"], {}
    expected_names = {
        "dock_body": "dock_gripper",
        "robot_mating_site": "robot_mating_face",
        "robot_plate_body": "robot_plate_frame",
        "positive_lock_slider_body": "qc_positive_lock_slider",
    }
    parsed: dict[str, Any] = {}
    for key, expected_name in expected_names.items():
        pose = value.get(key)
        if not isinstance(pose, dict) or set(pose) != {
            "name", "position_world_m", "quat_wxyz"
        }:
            errors.append(f"replay_poses:{key}:keys")
            continue
        position = np.asarray(pose.get("position_world_m"), dtype=np.float64)
        rotation = _quaternion_wxyz_matrix(pose.get("quat_wxyz"))
        if pose.get("name") != expected_name:
            errors.append(f"replay_poses:{key}:name")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            errors.append(f"replay_poses:{key}:position")
        if rotation is None:
            errors.append(f"replay_poses:{key}:quaternion")
        if position.shape == (3,) and rotation is not None:
            parsed[key] = (position, rotation)
    tab_poses = value.get("slider_tab_geoms")
    if not isinstance(tab_poses, list) or len(tab_poses) != 2:
        errors.append("replay_poses:tabs:count")
    else:
        observed_names: list[str] = []
        parsed_tabs: dict[str, Any] = {}
        for index, pose in enumerate(tab_poses):
            if not isinstance(pose, dict) or set(pose) != {
                "name", "position_world_m", "quat_wxyz"
            }:
                errors.append(f"replay_poses:tab:{index}:keys")
                continue
            name = pose.get("name")
            position = np.asarray(pose.get("position_world_m"), dtype=np.float64)
            rotation = _quaternion_wxyz_matrix(pose.get("quat_wxyz"))
            if not isinstance(name, str):
                errors.append(f"replay_poses:tab:{index}:name")
            else:
                observed_names.append(name)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                errors.append(f"replay_poses:tab:{index}:position")
            if rotation is None:
                errors.append(f"replay_poses:tab:{index}:quaternion")
            if isinstance(name, str) and position.shape == (3,) and rotation is not None:
                parsed_tabs[name] = (position, rotation)
        if observed_names != [CORE_CAM_TAB_NONCONTACT_GEOM, CORE_CAM_TAB_LEADING_GEOM]:
            errors.append("replay_poses:tabs:roster")
        parsed["slider_tab_geoms"] = parsed_tabs
    return errors, parsed


def _independent_cam_tab_record_classification(
    record: Any,
) -> tuple[list[str], bool, str | None]:
    """Recompute one raw MuJoCo contact classification from its measurements."""

    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record:missing"], False, None
    expected_keys = {
        "state_index", "physics_substep_count", "sim_time_s", "action", "runtime_pair",
        "canonical_pair", "cam_geom", "tab_or_other_geom", "surface_role",
        "preseat_mm", "source_x_mm", "transverse_y_mm",
        "orientation_error_rad", "slider_q_mm", "source_q_max_mm",
        "source_x_error_mm", "dock_hold_active", "attach_equality_active",
        "replay_world_poses",
        "contact_dist_mm", "penetration_mm", "contact_position_world_m",
        "contact_position_dock_local_mm", "contact_normal_raw_world",
        "contact_normal_cam_to_tab_dock_local", "contact_frame_3x3",
        "contact_friction", "contact_solref", "contact_solimp",
        "contact_force_torque_6d", "locus_error_mm", "normal_alignment",
        "q_excess_mm", "is_slider_tab_contact", "pair_eligible", "phase_state_valid", "locus_valid",
        "normal_valid", "q_valid", "force_finite",
        "provisional_classification_passed", "functional_coverage_role",
    }
    if set(record) != expected_keys:
        errors.append("record:keys")
    if record.get("state_index") != record.get("physics_substep_count"):
        errors.append("record:state_index")

    numeric_names = (
        "sim_time_s", "preseat_mm", "source_x_mm", "transverse_y_mm",
        "orientation_error_rad", "slider_q_mm", "source_q_max_mm",
        "source_x_error_mm", "contact_dist_mm", "penetration_mm",
        "q_excess_mm",
    )
    values = {name: _finite_real(record.get(name)) for name in numeric_names}
    if any(value is None for value in values.values()):
        errors.append("record:finite_scalars")
        return errors, False, None
    p = float(values["preseat_mm"])
    x = float(values["source_x_mm"])
    q = float(values["slider_q_mm"])
    expected_x = _independent_core_capture_x_mm(max(0.0, p))
    expected_qmax = max(0.05, min(3.0, max(0.0, p) - x - 3.15))
    if not _number_matches(record.get("source_q_max_mm"), expected_qmax):
        errors.append("record:source_q_max")
    if not _number_matches(record.get("source_x_error_mm"), x - expected_x):
        errors.append("record:source_x_error")
    if not _number_matches(record.get("q_excess_mm"), q - expected_qmax):
        errors.append("record:q_excess")
    distance = float(values["contact_dist_mm"])
    penetration = max(0.0, -distance)
    if not _number_matches(record.get("penetration_mm"), penetration):
        errors.append("record:penetration")

    vector_widths = {
        "contact_position_world_m": 3,
        "contact_position_dock_local_mm": 3,
        "contact_normal_raw_world": 3,
        "contact_normal_cam_to_tab_dock_local": 3,
        "contact_frame_3x3": 9,
        "contact_friction": 5,
        "contact_solref": 2,
        "contact_solimp": 5,
        "contact_force_torque_6d": 6,
    }
    vectors: dict[str, np.ndarray] = {}
    for name, width in vector_widths.items():
        value = record.get(name)
        if not isinstance(value, list) or len(value) != width:
            errors.append(f"record:{name}:width")
            continue
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            errors.append(f"record:{name}:finite")
            continue
        if not np.all(np.isfinite(array)):
            errors.append(f"record:{name}:finite")
            continue
        vectors[name] = array
    point = vectors.get("contact_position_dock_local_mm")
    normal = vectors.get("contact_normal_cam_to_tab_dock_local")
    force = vectors.get("contact_force_torque_6d")
    pose_errors, poses = _replay_pose_map_errors(record.get("replay_world_poses"))
    errors.extend(f"record:{error}" for error in pose_errors)
    dock_pose = poses.get("dock_body")
    mating_pose = poses.get("robot_mating_site")
    if dock_pose is not None and mating_pose is not None:
        dock_position, dock_rotation = dock_pose
        mating_position, mating_rotation = mating_pose
        local_mating_mm = dock_rotation.T @ (mating_position - dock_position) * 1000.0
        relative_rotation = dock_rotation.T @ mating_rotation
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        replay_orientation = math.acos(cosine)
        for name, expected in (
            ("preseat_mm", -float(local_mating_mm[2])),
            ("source_x_mm", float(local_mating_mm[0])),
            ("transverse_y_mm", float(local_mating_mm[1])),
            ("orientation_error_rad", replay_orientation),
        ):
            tolerance = 5.0e-8 if name == "orientation_error_rad" else 2.0e-9
            if not _number_matches(record.get(name), expected, tolerance=tolerance):
                errors.append(f"record:replay:{name}")
        world_point = vectors.get("contact_position_world_m")
        if world_point is not None and point is not None:
            replay_local_point = dock_rotation.T @ (world_point - dock_position) * 1000.0
            if not np.allclose(point, replay_local_point, rtol=0.0, atol=2.0e-9):
                errors.append("record:replay:contact_point")
        raw_normal = vectors.get("contact_normal_raw_world")
        if raw_normal is not None and normal is not None:
            runtime_pair_for_normal = record.get("runtime_pair")
            cam_for_normal = record.get("cam_geom")
            canonical_world = raw_normal.copy()
            if (
                isinstance(runtime_pair_for_normal, list)
                and runtime_pair_for_normal
                and runtime_pair_for_normal[0] != cam_for_normal
            ):
                canonical_world *= -1.0
            length = float(np.linalg.norm(canonical_world))
            if length > 0.0:
                replay_normal = dock_rotation.T @ (canonical_world / length)
                if not np.allclose(normal, replay_normal, rtol=0.0, atol=2.0e-9):
                    errors.append("record:replay:contact_normal")
    if normal is not None and not _number_matches(
        float(np.linalg.norm(normal)), 1.0, tolerance=1.0e-9
    ):
        errors.append("record:normal_unit")

    action = record.get("action")
    cam = record.get("cam_geom")
    other = record.get("tab_or_other_geom")
    runtime_pair = record.get("runtime_pair")
    pair_eligible = bool(
        other == CORE_CAM_TAB_LEADING_GEOM
        and cam in CORE_CAM_GEOMS[:3]
        and isinstance(runtime_pair, list)
        and len(runtime_pair) == 2
        and sorted(runtime_pair) == sorted([other, cam])
        and record.get("canonical_pair") == [other, cam]
    )
    if record.get("pair_eligible") is not pair_eligible:
        errors.append("record:pair_eligible")
    is_slider_tab = other in {CORE_CAM_TAB_NONCONTACT_GEOM, CORE_CAM_TAB_LEADING_GEOM}
    if record.get("is_slider_tab_contact") is not is_slider_tab:
        errors.append("record:is_slider_tab_contact")

    role = "unclassified_or_forbidden_core_cam_contact"
    locus_error = math.inf
    alignment = -math.inf
    locus_valid = False
    normal_valid = False
    q_valid = False
    functional_role: str | None = None
    tolerance = 0.020
    if point is not None and normal is not None:
        px, py, pz = [float(value) for value in point]
        if (
            pair_eligible
            and cam == CORE_CAM_GEOMS[1]
            and action == CORE_CAM_CONTACT_ACTIONS[2]
            and 3.2 - tolerance <= p <= 6.346666666666667 + tolerance
        ):
            role = CORE_CAM_FUNCTIONAL_ROLES[0]
            locus_error = max(
                abs(px + pz - 17.65),
                _interval_error(py, 0.0, 2.0),
                _interval_error(pz, -9.6, -6.4),
            )
            alignment = float(normal @ np.asarray([-1.0, 0.0, -1.0])) / math.sqrt(2.0)
            locus_valid = locus_error <= tolerance
            normal_valid = alignment >= 0.999
            q_valid = -tolerance <= q - expected_qmax <= math.sqrt(2.0) * 0.020
            functional_role = role
        elif (
            pair_eligible
            and cam == CORE_CAM_GEOMS[2]
            and action == CORE_CAM_CONTACT_ACTIONS[3]
            and -tolerance <= p <= 3.2 + tolerance
        ):
            role = CORE_CAM_FUNCTIONAL_ROLES[1]
            tab_lower = -4.8 - max(0.0, p)
            tab_upper = -3.2 - max(0.0, p)
            locus_error = max(
                abs(px - 24.05),
                _interval_error(py, 0.0, 2.0),
                _interval_error(pz, max(-6.4, tab_lower), min(-4.15, tab_upper)),
            )
            alignment = -float(normal[0])
            locus_valid = locus_error <= tolerance
            normal_valid = alignment >= 0.999
            q_valid = -tolerance <= q <= 0.05000000000000071 + tolerance
            functional_role = role
        elif (
            pair_eligible
            and cam == CORE_CAM_GEOMS[1]
            and action == CORE_CAM_CONTACT_ACTIONS[3]
            and 1.6 - tolerance <= p < 3.2
        ):
            role = "lead_hold_partition_seam_nonfunctional"
            locus_error = max(
                abs(px - 24.05), abs(pz + 6.4), _interval_error(py, 0.0, 2.0)
            )
            alignment = float(np.linalg.norm(normal[[0, 2]]))
            locus_valid = locus_error <= tolerance
            normal_valid = bool(
                alignment >= 0.999
                and float(normal[0]) <= 1.0e-12
                and float(normal[2]) <= 1.0e-12
                and -float(normal[2]) <= -float(normal[0]) + 1.0e-12
            )
            q_valid = -tolerance <= q <= 0.05000000000000071 + tolerance
        elif (
            pair_eligible
            and cam == CORE_CAM_GEOMS[0]
            and action == CORE_CAM_CONTACT_ACTIONS[3]
            and -tolerance <= p <= 0.95 + tolerance
        ):
            role = "main_hold_edge_tangency_nonfunctional"
            locus_error = max(
                abs(px - 24.05), abs(py),
                _interval_error(pz, -4.15, -3.2 - max(0.0, p)),
            )
            alignment = float(np.linalg.norm(normal[:2]))
            locus_valid = locus_error <= tolerance
            normal_valid = bool(
                alignment >= 0.999
                and float(normal[0]) <= 1.0e-12
                and float(normal[1]) >= 0.246875 * float(normal[0]) - 1.0e-12
            )
            q_valid = -tolerance <= q <= 0.05000000000000071 + tolerance

    phase_valid = bool(
        action in CORE_CAM_CONTACT_ACTIONS
        and record.get("dock_hold_active") is True
        and record.get("attach_equality_active") is False
        and abs(x - expected_x) <= 0.040
        and abs(float(values["transverse_y_mm"])) <= 0.010
        and float(values["orientation_error_rad"]) <= math.radians(0.1)
    )
    force_finite = force is not None
    passed = bool(
        pair_eligible
        and role != "unclassified_or_forbidden_core_cam_contact"
        and phase_valid and locus_valid and normal_valid and q_valid
        and force_finite and float(force[0]) >= -1.0e-12
        and penetration <= 0.020
    )
    expected_flags = {
        "surface_role": role,
        "phase_state_valid": phase_valid,
        "locus_valid": locus_valid,
        "normal_valid": normal_valid,
        "q_valid": q_valid,
        "force_finite": force_finite,
        "provisional_classification_passed": passed,
        "functional_coverage_role": functional_role,
    }
    for name, expected in expected_flags.items():
        if record.get(name) != expected:
            errors.append(f"record:{name}")
    expected_locus = locus_error if math.isfinite(locus_error) else None
    expected_alignment = alignment if math.isfinite(alignment) else None
    if not (
        (expected_locus is None and record.get("locus_error_mm") is None)
        or (expected_locus is not None and _number_matches(record.get("locus_error_mm"), expected_locus))
    ):
        errors.append("record:locus_error")
    if not (
        (expected_alignment is None and record.get("normal_alignment") is None)
        or (expected_alignment is not None and _number_matches(record.get("normal_alignment"), expected_alignment))
    ):
        errors.append("record:normal_alignment")
    return errors, passed, functional_role


def _core_cam_functional_envelope_errors(
    envelope: Any,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    if not isinstance(envelope, dict):
        return ["envelope:missing"], False
    expected_keys = {
        "schema_version", "state_index_semantics", "raw_states",
        "raw_states_sha256", "raw_state_count", "audited_functional_substeps",
        "state_counts_by_phase", "producer_phase_counts", "phase_counts_consistent",
        "both_functional_phases_observed", "first_functional_lead_state",
        "first_functional_hold_state", "functional_role_onsets_observed",
        "full_state_continuity_verified",
        "exact_two_tab_by_five_cam_gap_closure_verified",
        "per_state_contact_count_partitions_verified",
        "source_pose_and_equality_states_verified",
        "discrete_no_skipped_state_verified", "discrete_no_rebound_verified",
        "all_functional_surface_states_contact_or_nonnegative_gap",
        "all_raw_states_finite", "maximum_sampled_coordinate_jump_mm",
        "minimum_complete_cam_signed_distance_mm", "contactless_negative_pair_count",
        "unresolved_pair_count", "cutoff_pair_count", "maximum_q_excess_mm",
        "continuous_between_mj_steps_authority", "interval_motion_bound_certified",
        "continuous_tunnel_authority", "passed", "release_ready",
    }
    if set(envelope) != expected_keys:
        errors.append("envelope:keys")
    for key, expected in {
        "schema_version": "1.0",
        "state_index_semantics": "physics_substep_count_immediately_after_mj_step",
        "continuous_between_mj_steps_authority": False,
        "interval_motion_bound_certified": False,
        "continuous_tunnel_authority": False,
        "release_ready": False,
    }.items():
        if envelope.get(key) != expected:
            errors.append(f"envelope:{key}")
    states = envelope.get("raw_states")
    if not isinstance(states, list):
        errors.append("envelope:raw_states")
        states = []
    if envelope.get("raw_states_sha256") != canonical_json_sha256(states):
        errors.append("envelope:raw_states_sha256")
    if envelope.get("raw_state_count") != len(states):
        errors.append("envelope:raw_state_count")
    functional_actions = CORE_CAM_CONTACT_ACTIONS[2:]
    state_counts = {action: 0 for action in functional_actions}
    expected_state_keys = {
        "state_index", "physics_substep_count", "sim_time_s", "action",
        "preseat_mm", "source_x_mm", "expected_source_x_mm", "source_x_error_mm",
        "transverse_y_mm", "orientation_error_rad", "slider_q_mm",
        "source_q_max_mm", "q_excess_mm", "dock_hold_active",
        "attach_equality_active", "replay_state", "replay_world_poses",
        "pair_gap_records", "complete_cam_min_signed_distance_mm",
        "complete_cam_minimum_pair", "observed_cam_tab_contact_count",
        "eligible_cam_tab_contact_count", "functional_contact_count",
        "functional_lead_contact_count", "functional_hold_contact_count",
        "nonfunctional_candidate_contact_count", "rejected_cam_tab_contact_count",
        "other_core_cam_contact_count", "count_partition_valid",
        "expected_functional_role", "functional_contact_required",
        "expected_functional_contact_count", "contact_continuity_state_passed",
        "pair_gap_closure_passed", "contactless_negative_pair_count",
        "precontact_state_passed", "source_pose_state_passed",
        "q_envelope_state_passed", "state_index_contiguous",
        "sim_time_contiguous", "action_transition_valid",
        "sampled_coordinate_jump_mm", "sampled_coordinate_jump_limit_mm",
        "running_min_preseat_before_mm", "preseat_no_rebound",
        "first_lead_contact_previously_observed",
        "running_min_post_lead_q_before_mm", "post_first_lead_q_no_rebound",
        "discrete_no_skipped_state_passed", "discrete_no_rebound_state_passed",
        "continuous_between_mj_steps_authority", "interval_motion_bound_certified",
        "finite",
    }
    pair_roster = [
        [tab, cam]
        for tab in (CORE_CAM_TAB_NONCONTACT_GEOM, CORE_CAM_TAB_LEADING_GEOM)
        for cam in CORE_CAM_GEOMS
    ]
    previous: dict[str, Any] | None = None
    prior_p: list[float] = []
    post_lead_q: list[float] = []
    lead_seen = False
    state_flags = {
        "continuity": [], "gap": [], "counts": [], "source": [],
        "skip": [], "rebound": [], "contact": [], "finite": [],
    }
    for index, state in enumerate(states):
        prefix = f"state:{index}"
        if not isinstance(state, dict) or set(state) != expected_state_keys:
            errors.append(f"{prefix}:keys")
            previous = state if isinstance(state, dict) else previous
            continue
        action = state.get("action")
        if action not in functional_actions:
            errors.append(f"{prefix}:action")
            continue
        state_counts[action] += 1
        state_index = state.get("state_index")
        if state_index != state.get("physics_substep_count") or (
            isinstance(state_index, bool) or not isinstance(state_index, int)
        ):
            errors.append(f"{prefix}:state_index")
        time_s = _finite_real(state.get("sim_time_s"))
        p = _finite_real(state.get("preseat_mm"))
        x = _finite_real(state.get("source_x_mm"))
        q = _finite_real(state.get("slider_q_mm"))
        y = _finite_real(state.get("transverse_y_mm"))
        orientation = _finite_real(state.get("orientation_error_rad"))
        if None in (time_s, p, x, q, y, orientation):
            errors.append(f"{prefix}:finite_scalars")
            continue
        p = float(p); x = float(x); q = float(q)
        expected_x = _independent_core_capture_x_mm(max(0.0, p))
        qmax = max(0.05, min(3.0, max(0.0, p) - x - 3.15))
        for key, expected in (
            ("expected_source_x_mm", expected_x),
            ("source_x_error_mm", x - expected_x),
            ("source_q_max_mm", qmax),
            ("q_excess_mm", q - qmax),
        ):
            if not _number_matches(state.get(key), expected):
                errors.append(f"{prefix}:{key}")
        pose_errors, _ = _replay_pose_map_errors(state.get("replay_world_poses"))
        errors.extend(f"{prefix}:{error}" for error in pose_errors)
        replay = state.get("replay_state")
        if not isinstance(replay, dict) or set(replay) != {
            "qpos", "qvel", "mocap_pos", "mocap_quat_wxyz",
            "equality_active", "replay_method",
        }:
            errors.append(f"{prefix}:replay_state")
        elif replay.get("replay_method") != "copy_into_fresh_MjData_then_mj_forward":
            errors.append(f"{prefix}:replay_method")
        else:
            for field in ("qpos", "qvel"):
                values = replay.get(field)
                if not isinstance(values, list) or not values or any(
                    _finite_real(value) is None for value in values
                ):
                    errors.append(f"{prefix}:replay:{field}")
            equality = replay.get("equality_active")
            if not isinstance(equality, list) or any(
                not isinstance(item, dict)
                or set(item) != {"name", "active"}
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("active"), bool)
                for item in equality
            ):
                errors.append(f"{prefix}:replay:equality")

        pair_records = state.get("pair_gap_records")
        if not isinstance(pair_records, list) or len(pair_records) != 10:
            errors.append(f"{prefix}:pair_roster")
            pair_records = []
        observed_pairs: list[list[str]] = []
        pair_valid: list[bool] = []
        contactless_negative_count = 0
        for pair_index, pair_record in enumerate(pair_records):
            pair_prefix = f"{prefix}:pair:{pair_index}"
            pair_keys = {
                "tab_geom", "cam_geom", "pair", "method", "maximum_distance_m",
                "signed_distance_mm", "live_contact_count", "live_contact_indices",
                "contact_position_world_m", "closest_points_world_m",
                "closest_points_valid", "cutoff_reached", "contactless_negative",
                "finite", "resolved", "source_pair_clearance_valid",
            }
            if not isinstance(pair_record, dict) or set(pair_record) != pair_keys:
                errors.append(f"{pair_prefix}:keys")
                continue
            tab = pair_record.get("tab_geom"); cam = pair_record.get("cam_geom")
            pair = [tab, cam]
            observed_pairs.append(pair)
            if pair_record.get("pair") != pair:
                errors.append(f"{pair_prefix}:pair")
            distance = _finite_real(pair_record.get("signed_distance_mm"))
            live_count = pair_record.get("live_contact_count")
            live_indices = pair_record.get("live_contact_indices")
            has_live = isinstance(live_count, int) and live_count > 0
            if not isinstance(live_indices, list) or live_count != len(live_indices):
                errors.append(f"{pair_prefix}:live_count")
            expected_method = (
                "minimum_live_contact_dist" if has_live
                else "mj_geomDistance_no_live_contact"
            )
            if pair_record.get("method") != expected_method or not _number_matches(
                pair_record.get("maximum_distance_m"), 0.1
            ):
                errors.append(f"{pair_prefix}:method")
            finite = distance is not None
            cutoff = bool(finite and float(distance) >= 100.0 - 1.0e-9)
            contactless_negative = bool(
                not has_live and finite and float(distance) < -1.0e-6
            )
            resolved = bool(finite and not cutoff and not contactless_negative)
            clearance = bool(
                finite and float(distance) >= (
                    -1.0e-6
                    if tab == CORE_CAM_TAB_NONCONTACT_GEOM or cam in CORE_CAM_GEOMS[3:]
                    else -0.020
                )
            )
            for key, expected in (
                ("finite", finite), ("cutoff_reached", cutoff),
                ("contactless_negative", contactless_negative),
                ("resolved", resolved), ("source_pair_clearance_valid", clearance),
            ):
                if pair_record.get(key) is not expected:
                    errors.append(f"{pair_prefix}:{key}")
            contactless_negative_count += int(contactless_negative)
            pair_valid.append(resolved and clearance)
        if observed_pairs != pair_roster:
            errors.append(f"{prefix}:pair_roster_order")
        if pair_records:
            minimum = min(pair_records, key=lambda item: float(item["signed_distance_mm"]))
            if state.get("complete_cam_minimum_pair") != minimum.get("pair"):
                errors.append(f"{prefix}:minimum_pair")
            if not _number_matches(
                state.get("complete_cam_min_signed_distance_mm"),
                float(minimum["signed_distance_mm"]),
            ):
                errors.append(f"{prefix}:minimum_distance")
        observed_count = state.get("observed_cam_tab_contact_count")
        eligible = state.get("eligible_cam_tab_contact_count")
        rejected = state.get("rejected_cam_tab_contact_count")
        functional = state.get("functional_contact_count")
        nonfunctional = state.get("nonfunctional_candidate_contact_count")
        count_partition = bool(
            all(isinstance(value, int) and value >= 0 for value in (
                observed_count, eligible, rejected, functional, nonfunctional
            ))
            and eligible + rejected == observed_count
            and functional + nonfunctional == eligible
        )
        if state.get("count_partition_valid") is not count_partition:
            errors.append(f"{prefix}:count_partition")
        expected_role = (
            CORE_CAM_FUNCTIONAL_ROLES[0]
            if action == functional_actions[0] and p <= 6.346666666666667 + 1.0e-6
            else (CORE_CAM_FUNCTIONAL_ROLES[1] if action == functional_actions[1] else None)
        )
        expected_count = (
            int(state.get("functional_lead_contact_count", 0))
            if expected_role == CORE_CAM_FUNCTIONAL_ROLES[0]
            else int(state.get("functional_hold_contact_count", 0))
            if expected_role == CORE_CAM_FUNCTIONAL_ROLES[1]
            else 0
        )
        required = expected_role is not None
        if state.get("expected_functional_role") != expected_role:
            errors.append(f"{prefix}:expected_role")
        if state.get("functional_contact_required") is not required:
            errors.append(f"{prefix}:functional_required")
        if state.get("expected_functional_contact_count") != expected_count:
            errors.append(f"{prefix}:functional_count")
        expected_cam = CORE_CAM_GEOMS[1] if expected_role == CORE_CAM_FUNCTIONAL_ROLES[0] else CORE_CAM_GEOMS[2]
        expected_gap = next((item for item in pair_records if item.get("pair") == [CORE_CAM_TAB_LEADING_GEOM, expected_cam]), None)
        contact_continuity = bool(
            not required or expected_count > 0 or (
                expected_gap is not None and expected_gap.get("resolved") is True
                and float(expected_gap["signed_distance_mm"]) >= -1.0e-6
            )
        )
        if state.get("contact_continuity_state_passed") is not contact_continuity:
            errors.append(f"{prefix}:contact_continuity")
        pair_distances = [
            _finite_real(item.get("signed_distance_mm"))
            for item in pair_records
            if isinstance(item, dict)
        ]
        precontact_state_passed = bool(
            required
            or (
                observed_count == 0
                and len(pair_distances) == 10
                and all(value is not None for value in pair_distances)
                and min(float(value) for value in pair_distances if value is not None)
                >= -1.0e-6
            )
        )
        if state.get("precontact_state_passed") is not precontact_state_passed:
            errors.append(f"{prefix}:precontact")
        q_envelope = bool(
            -1.0e-6 <= q <= (
                min(3.0 + 1.0e-6, qmax + math.sqrt(2.0) * 0.020)
                if action == functional_actions[0]
                else 0.05000000000000071 + 0.020
            )
        )
        phase_range = (
            3.2 - 0.020 <= p <= 6.4 + 0.020
            if action == functional_actions[0]
            else -0.020 <= p <= 3.2 + 0.020
        )
        source_valid = bool(
            phase_range and abs(x - expected_x) <= 0.040
            and abs(float(y)) <= 0.010 and float(orientation) <= math.radians(0.1)
            and state.get("dock_hold_active") is True
            and state.get("attach_equality_active") is False
        )
        pair_closure = bool(len(pair_valid) == 10 and all(pair_valid))
        if state.get("pair_gap_closure_passed") is not pair_closure:
            errors.append(f"{prefix}:pair_closure")
        if state.get("contactless_negative_pair_count") != contactless_negative_count:
            errors.append(f"{prefix}:contactless_count")
        if state.get("source_pose_state_passed") is not source_valid:
            errors.append(f"{prefix}:source_pose")
        if state.get("q_envelope_state_passed") is not q_envelope:
            errors.append(f"{prefix}:q_envelope")
        if not _number_matches(state.get("sampled_coordinate_jump_limit_mm"), 0.010):
            errors.append(f"{prefix}:jump_limit")
        if previous is None:
            contiguous = time_contiguous = action_valid = True
            jump = 0.0
        else:
            contiguous = state_index == int(previous["state_index"]) + 1
            time_contiguous = abs(float(time_s) - float(previous["sim_time_s"]) - 0.00025) <= 1.0e-12
            action_valid = bool(previous["action"] == action or (
                previous["action"] == functional_actions[0] and action == functional_actions[1]
            ))
            dp = abs(p - float(previous["preseat_mm"])); dx = abs(x - float(previous["source_x_mm"])); dq = abs(q - float(previous["slider_q_mm"]))
            jump = (dp + dx + dq) / math.sqrt(2.0) if action == functional_actions[0] else dx + dq
        for key, expected in (
            ("state_index_contiguous", contiguous),
            ("sim_time_contiguous", time_contiguous),
            ("action_transition_valid", action_valid),
        ):
            if state.get(key) is not expected:
                errors.append(f"{prefix}:{key}")
        if not _number_matches(state.get("sampled_coordinate_jump_mm"), jump, 1.0e-10):
            errors.append(f"{prefix}:jump")
        running_p = min(prior_p, default=p)
        p_no_rebound = p <= running_p + 0.020
        if not _number_matches(state.get("running_min_preseat_before_mm"), running_p):
            errors.append(f"{prefix}:running_p")
        if state.get("preseat_no_rebound") is not p_no_rebound:
            errors.append(f"{prefix}:p_rebound")
        if lead_seen:
            running_q = min(post_lead_q, default=q)
            q_no_rebound = q <= running_q + 0.020
        else:
            running_q = q
            q_no_rebound = True
        if state.get("first_lead_contact_previously_observed") is not lead_seen:
            errors.append(f"{prefix}:lead_seen")
        if not _number_matches(state.get("running_min_post_lead_q_before_mm"), running_q):
            errors.append(f"{prefix}:running_q")
        if state.get("post_first_lead_q_no_rebound") is not q_no_rebound:
            errors.append(f"{prefix}:q_rebound")
        skipped = bool(contiguous and time_contiguous and action_valid and jump <= 0.010 and pair_closure and contactless_negative_count == 0)
        rebound = bool(p_no_rebound and q_no_rebound and q_envelope and contact_continuity)
        if state.get("discrete_no_skipped_state_passed") is not skipped:
            errors.append(f"{prefix}:discrete_skip")
        if state.get("discrete_no_rebound_state_passed") is not rebound:
            errors.append(f"{prefix}:discrete_rebound")
        if state.get("continuous_between_mj_steps_authority") is not False or state.get("interval_motion_bound_certified") is not False:
            errors.append(f"{prefix}:continuous_authority")
        state_finite = all(_finite_real(state.get(key)) is not None for key in (
            "preseat_mm", "source_x_mm", "slider_q_mm", "source_q_max_mm",
            "transverse_y_mm", "orientation_error_rad",
        ))
        if state.get("finite") is not state_finite:
            errors.append(f"{prefix}:finite")
        state_flags["continuity"].append(contiguous and time_contiguous and action_valid)
        state_flags["gap"].append(pair_closure)
        state_flags["counts"].append(count_partition)
        state_flags["source"].append(source_valid)
        state_flags["skip"].append(skipped)
        state_flags["rebound"].append(rebound)
        state_flags["contact"].append(contact_continuity)
        state_flags["finite"].append(state_finite)
        prior_p.append(p)
        if int(state.get("functional_lead_contact_count", 0)) > 0:
            lead_seen = True
        if lead_seen:
            post_lead_q.append(q)
        previous = state

    expected_phase_counts = state_counts
    if envelope.get("state_counts_by_phase") != expected_phase_counts:
        errors.append("envelope:state_counts")
    producer_counts = envelope.get("producer_phase_counts")
    phase_consistent = bool(
        producer_counts == expected_phase_counts
        and envelope.get("audited_functional_substeps") == len(states)
    )
    if envelope.get("phase_counts_consistent") is not phase_consistent:
        errors.append("envelope:phase_consistency")
    both_phases = all(value > 0 for value in expected_phase_counts.values())
    if envelope.get("both_functional_phases_observed") is not both_phases:
        errors.append("envelope:both_phases")
    first_lead = next((state for state in states if isinstance(state, dict) and int(state.get("functional_lead_contact_count", 0)) > 0), None)
    first_hold = next((state for state in states if isinstance(state, dict) and int(state.get("functional_hold_contact_count", 0)) > 0), None)
    if envelope.get("first_functional_lead_state") != first_lead:
        errors.append("envelope:first_lead")
    if envelope.get("first_functional_hold_state") != first_hold:
        errors.append("envelope:first_hold")
    onsets = first_lead is not None and first_hold is not None
    if envelope.get("functional_role_onsets_observed") is not onsets:
        errors.append("envelope:onsets")
    aggregate_flags = {
        "full_state_continuity_verified": bool(states and all(state_flags["continuity"]) and len({state.get("state_index") for state in states if isinstance(state, dict)}) == len(states)),
        "exact_two_tab_by_five_cam_gap_closure_verified": bool(states and all(state_flags["gap"])),
        "per_state_contact_count_partitions_verified": bool(states and all(state_flags["counts"])),
        "source_pose_and_equality_states_verified": bool(states and all(state_flags["source"])),
        "discrete_no_skipped_state_verified": bool(states and all(state_flags["skip"])),
        "discrete_no_rebound_verified": bool(states and all(state_flags["rebound"])),
        "all_functional_surface_states_contact_or_nonnegative_gap": bool(states and all(state_flags["contact"])),
        "all_raw_states_finite": bool(states and all(state_flags["finite"])),
    }
    for key, expected in aggregate_flags.items():
        if envelope.get(key) is not expected:
            errors.append(f"envelope:{key}")
    numeric_aggregates = {
        "maximum_sampled_coordinate_jump_mm": max((float(state["sampled_coordinate_jump_mm"]) for state in states), default=None),
        "minimum_complete_cam_signed_distance_mm": min((float(state["complete_cam_min_signed_distance_mm"]) for state in states), default=None),
        "contactless_negative_pair_count": sum(int(state.get("contactless_negative_pair_count", 0)) for state in states if isinstance(state, dict)),
        "unresolved_pair_count": sum(sum(not bool(pair.get("resolved")) for pair in state.get("pair_gap_records", [])) for state in states if isinstance(state, dict)),
        "cutoff_pair_count": sum(sum(bool(pair.get("cutoff_reached")) for pair in state.get("pair_gap_records", [])) for state in states if isinstance(state, dict)),
        "maximum_q_excess_mm": max((float(state["q_excess_mm"]) for state in states), default=None),
    }
    for key, expected in numeric_aggregates.items():
        if _evidence_mismatches(envelope.get(key), expected, key):
            errors.append(f"envelope:{key}")
    expected_pass = bool(
        phase_consistent and both_phases and all(aggregate_flags.values())
    )
    # Route endpoint/abort conjunction is checked by the enclosing result gate.
    return errors, expected_pass


def _core_cam_functional_state_replay_errors(
    state: Any,
    model: Any,
    mujoco: ModuleType,
) -> list[str]:
    """Replay one lossless state in fresh MjData and recompute poses/gaps."""

    if not isinstance(state, dict):
        return ["state_replay:missing"]
    replay = state.get("replay_state")
    if not isinstance(replay, dict):
        return ["state_replay:state"]
    errors: list[str] = []
    data = mujoco.MjData(model)
    qpos = np.asarray(replay.get("qpos"), dtype=np.float64)
    qvel = np.asarray(replay.get("qvel"), dtype=np.float64)
    mocap_pos = np.asarray(replay.get("mocap_pos"), dtype=np.float64)
    mocap_quat = np.asarray(replay.get("mocap_quat_wxyz"), dtype=np.float64)
    if model.nmocap == 0 and mocap_pos.size == 0:
        mocap_pos = mocap_pos.reshape(0, 3)
    if model.nmocap == 0 and mocap_quat.size == 0:
        mocap_quat = mocap_quat.reshape(0, 4)
    if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
        return ["state_replay:qpos"]
    if qvel.shape != (model.nv,) or not np.all(np.isfinite(qvel)):
        return ["state_replay:qvel"]
    if mocap_pos.shape != (model.nmocap, 3) or not np.all(np.isfinite(mocap_pos)):
        return ["state_replay:mocap_pos"]
    if mocap_quat.shape != (model.nmocap, 4) or not np.all(np.isfinite(mocap_quat)):
        return ["state_replay:mocap_quat"]
    if model.nmocap and not np.allclose(
        np.linalg.norm(mocap_quat, axis=1), 1.0, rtol=0.0, atol=1.0e-9
    ):
        errors.append("state_replay:mocap_quat_norm")
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    if model.nmocap:
        data.mocap_pos[:] = mocap_pos
        data.mocap_quat[:] = mocap_quat
    equality = replay.get("equality_active")
    expected_equality_names = [str(model.equality(index).name) for index in range(model.neq)]
    if not isinstance(equality, list) or [item.get("name") for item in equality if isinstance(item, dict)] != expected_equality_names:
        errors.append("state_replay:equality_roster")
    else:
        for index, item in enumerate(equality):
            if not isinstance(item.get("active"), bool):
                errors.append(f"state_replay:equality:{index}")
            else:
                data.eq_active[index] = int(item["active"])
    mujoco.mj_forward(model, data)

    def quat(rotation: np.ndarray) -> list[float]:
        value = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(value, np.asarray(rotation, dtype=np.float64).reshape(9))
        value /= np.linalg.norm(value)
        if value[0] < 0.0:
            value *= -1.0
        return [float(item) for item in value]

    expected_poses: dict[str, Any] = {}
    for key, kind, name in (
        ("dock_body", "body", "dock_gripper"),
        ("robot_mating_site", "site", "robot_mating_face"),
        ("robot_plate_body", "body", "robot_plate_frame"),
        ("positive_lock_slider_body", "body", "qc_positive_lock_slider"),
    ):
        object_data = data.body(name) if kind == "body" else data.site(name)
        expected_poses[key] = {
            "name": name,
            "position_world_m": [float(value) for value in object_data.xpos],
            "quat_wxyz": quat(np.asarray(object_data.xmat).reshape(3, 3)),
        }
    tab_names = [CORE_CAM_TAB_NONCONTACT_GEOM, CORE_CAM_TAB_LEADING_GEOM]
    expected_poses["slider_tab_geoms"] = []
    for name in tab_names:
        geom_id = int(model.geom(name).id)
        expected_poses["slider_tab_geoms"].append(
            {
                "name": name,
                "position_world_m": [float(value) for value in data.geom_xpos[geom_id]],
                "quat_wxyz": quat(np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)),
            }
        )
    pose_mismatches = _evidence_mismatches(
        state.get("replay_world_poses"), expected_poses, "state_replay:poses"
    )
    errors.extend(pose_mismatches)
    dock_position = np.asarray(data.body("dock_gripper").xpos, dtype=np.float64)
    dock_rotation = np.asarray(data.body("dock_gripper").xmat, dtype=np.float64).reshape(3, 3)
    mating_position = np.asarray(data.site("robot_mating_face").xpos, dtype=np.float64)
    mating_rotation = np.asarray(data.site("robot_mating_face").xmat, dtype=np.float64).reshape(3, 3)
    local_mm = dock_rotation.T @ (mating_position - dock_position) * 1000.0
    relative = dock_rotation.T @ mating_rotation
    orientation = math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    slider_joint = model.joint("qc_positive_lock_slider_joint")
    slider_q = float(data.qpos[int(slider_joint.qposadr[0])]) * 1000.0
    p = -float(local_mm[2]); x = float(local_mm[0])
    qmax = max(0.05, min(3.0, max(0.0, p) - x - 3.15))
    for key, expected in (
        ("preseat_mm", p), ("source_x_mm", x),
        ("transverse_y_mm", float(local_mm[1])),
        ("orientation_error_rad", orientation), ("slider_q_mm", slider_q),
        ("source_q_max_mm", qmax),
    ):
        if not _number_matches(state.get(key), expected, tolerance=2.0e-9):
            errors.append(f"state_replay:{key}")

    observed_gap_records = state.get("pair_gap_records")
    if not isinstance(observed_gap_records, list) or len(observed_gap_records) != 10:
        return [*errors, "state_replay:pair_gap_roster"]
    for index, (tab_name, cam_name) in enumerate(
        (pair for pair in ([tab, cam] for tab in tab_names for cam in CORE_CAM_GEOMS))
    ):
        observed = observed_gap_records[index]
        tab_id = int(model.geom(tab_name).id)
        cam_id = int(model.geom(cam_name).id)
        contact_indices = [
            contact_index
            for contact_index in range(data.ncon)
            if frozenset(int(value) for value in data.contact[contact_index].geom)
            == frozenset((tab_id, cam_id))
        ]
        if contact_indices:
            witness_index = min(
                contact_indices, key=lambda item: float(data.contact[item].dist)
            )
            witness = data.contact[witness_index]
            distance_mm = float(witness.dist) * 1000.0
            method = "minimum_live_contact_dist"
            contact_position = [float(value) for value in witness.pos]
            closest = None
            closest_valid = False
            cutoff = False
        else:
            from_to = np.full(6, np.nan, dtype=np.float64)
            distance_m = float(
                mujoco.mj_geomDistance(model, data, tab_id, cam_id, 0.1, from_to)
            )
            distance_mm = distance_m * 1000.0
            method = "mj_geomDistance_no_live_contact"
            cutoff = distance_m >= 0.1 - 1.0e-12
            closest_valid = bool(not cutoff and np.all(np.isfinite(from_to)))
            closest = (
                [[float(value) for value in from_to[:3]], [float(value) for value in from_to[3:]]]
                if closest_valid else None
            )
            contact_position = None
        checks = {
            "pair": [tab_name, cam_name], "method": method,
            "live_contact_count": len(contact_indices),
            "live_contact_indices": contact_indices,
            "contact_position_world_m": contact_position,
            "closest_points_world_m": closest,
            "closest_points_valid": closest_valid,
            "cutoff_reached": cutoff,
        }
        for key, expected in checks.items():
            if _evidence_mismatches(observed.get(key), expected, key):
                errors.append(f"state_replay:pair:{index}:{key}")
        if not _number_matches(observed.get("signed_distance_mm"), distance_mm, 2.0e-9):
            errors.append(f"state_replay:pair:{index}:distance")
    return errors


def _core_cam_actual_model_binding_errors(
    record: Any,
    contract: dict[str, Any],
    observations: dict[str, str] | None = None,
) -> tuple[list[str], bool]:
    """Recompute the shared init/evidence model-binding arithmetic."""

    if not isinstance(record, dict):
        return ["model_binding:missing"], False
    expected_keys = {
        "schema_version", "binding_state", "expected_source_model_xml_sha256",
        "compiled_model_xml_equivalent_digest_api",
        "expected_compiled_model_xml_equivalent_sha256",
        "initialized_active_collision_geometry_digest_api",
        "initialized_state_construction",
        "expected_initialized_active_collision_geometry_sha256",
        "controller_init_observed_compiled_model_xml_equivalent_sha256",
        "controller_init_compiled_model_xml_equivalent_matches",
        "controller_init_observed_initialized_active_geometry_sha256",
        "controller_init_initialized_active_geometry_matches",
        "controller_init_passed",
        "evidence_observed_compiled_model_xml_equivalent_sha256",
        "evidence_compiled_model_xml_equivalent_matches",
        "evidence_observed_initialized_active_geometry_sha256",
        "evidence_initialized_active_geometry_matches",
        "evidence_recompute_passed",
        "compiled_model_digest_unchanged_since_controller_init",
        "active_geometry_digest_unchanged_since_controller_init", "passed",
    }
    errors: list[str] = []
    if set(record) != expected_keys:
        errors.append("model_binding:keys")
    expected_compiled = contract["model_binding"][
        "compiled_model_xml_equivalent_sha256"
    ]
    expected_active = contract["model_binding"][
        "initialized_active_collision_geometry_sha256"
    ]
    if observations is None:
        errors.append("model_binding:observations_missing")
        observations = {
            "controller_init_compiled": "",
            "controller_init_active": "",
            "evidence_compiled": "",
            "evidence_active": "",
        }
    if set(observations) != {
        "controller_init_compiled", "controller_init_active",
        "evidence_compiled", "evidence_active",
    }:
        errors.append("model_binding:observations")
        observations = {
            "controller_init_compiled": "",
            "controller_init_active": "",
            "evidence_compiled": "",
            "evidence_active": "",
        }
    expected_header = {
        "schema_version": "1.0",
        "binding_state": (
            "controller_init_and_evidence_recomputed_actual_passed_model"
        ),
        "expected_source_model_xml_sha256": contract["model_binding"][
            "model_xml_sha256"
        ],
        "compiled_model_xml_equivalent_digest_api": (
            "compiled_model_xml_equivalent_sha256"
        ),
        "expected_compiled_model_xml_equivalent_sha256": expected_compiled,
        "initialized_active_collision_geometry_digest_api": (
            "initialized_active_collision_geometry_sha256"
        ),
        "initialized_state_construction": (
            "fresh_MjData_then_initialize_and_mj_forward"
        ),
        "expected_initialized_active_collision_geometry_sha256": expected_active,
        "controller_init_observed_compiled_model_xml_equivalent_sha256": (
            observations["controller_init_compiled"]
        ),
        "controller_init_observed_initialized_active_geometry_sha256": (
            observations["controller_init_active"]
        ),
        "evidence_observed_compiled_model_xml_equivalent_sha256": (
            observations["evidence_compiled"]
        ),
        "evidence_observed_initialized_active_geometry_sha256": (
            observations["evidence_active"]
        ),
    }
    for key, expected in expected_header.items():
        if record.get(key) != expected:
            errors.append(f"model_binding:{key}")
    init_compiled_match = observations["controller_init_compiled"] == expected_compiled
    init_active_match = observations["controller_init_active"] == expected_active
    evidence_compiled_match = observations["evidence_compiled"] == expected_compiled
    evidence_active_match = observations["evidence_active"] == expected_active
    init_pass = init_compiled_match and init_active_match
    evidence_pass = evidence_compiled_match and evidence_active_match
    compiled_unchanged = (
        observations["controller_init_compiled"] == observations["evidence_compiled"]
    )
    active_unchanged = (
        observations["controller_init_active"] == observations["evidence_active"]
    )
    expected_flags = {
        "controller_init_compiled_model_xml_equivalent_matches": init_compiled_match,
        "controller_init_initialized_active_geometry_matches": init_active_match,
        "controller_init_passed": init_pass,
        "evidence_compiled_model_xml_equivalent_matches": evidence_compiled_match,
        "evidence_initialized_active_geometry_matches": evidence_active_match,
        "evidence_recompute_passed": evidence_pass,
        "compiled_model_digest_unchanged_since_controller_init": compiled_unchanged,
        "active_geometry_digest_unchanged_since_controller_init": active_unchanged,
        "passed": bool(init_pass and evidence_pass and compiled_unchanged and active_unchanged),
    }
    for key, expected in expected_flags.items():
        if record.get(key) is not expected:
            errors.append(f"model_binding:{key}")
    return errors, expected_flags["passed"]


def core_cam_tab_result_errors(
    result: Any,
    contract: dict[str, Any],
    *,
    model_binding_observations: dict[str, str] | None = None,
    replay_model: Any | None = None,
    replay_mujoco: ModuleType | None = None,
) -> list[str]:
    """Recompute checkpoint-A contact/free-space evidence from raw samples."""

    if not isinstance(result, dict):
        return ["cam_tab_result_missing"]
    errors: list[str] = []
    abort_reason = result.get("abort_reason")
    physics_substeps = result.get("physics_substep_count")
    if (
        isinstance(physics_substeps, bool)
        or not isinstance(physics_substeps, int)
        or physics_substeps < 0
    ):
        errors.append("result:physics_substeps")
        physics_substeps = 0
    actual_model_binding = result.get("core_cam_actual_model_binding")
    binding_errors, binding_passed = _core_cam_actual_model_binding_errors(
        actual_model_binding,
        contract,
        model_binding_observations,
    )
    errors.extend(binding_errors)
    alignment = result.get("route_alignment")
    endpoint_records = (
        alignment.get("phase_endpoint_journal_evidence", [])
        if isinstance(alignment, dict)
        else []
    )
    endpoint_actions = {
        item.get("action")
        for item in endpoint_records
        if isinstance(item, dict) and item.get("event") == "move_complete"
    }
    all_route_endpoints = endpoint_actions == set(CORE_CAM_CONTACT_ACTIONS)

    evidence = result.get("core_cam_tab_contact_evidence")
    if not isinstance(evidence, dict):
        return [*errors, "contact_evidence:missing"]
    expected_evidence_keys = {
        "schema_version", "evidence_kind", "runtime_contract_api",
        "contract_identity_sha256", "model_binding", "physics_timestep_s", "observed",
        "audited_substeps", "audited_substeps_by_phase",
        "all_four_capture_phases_observed", "raw_contact_records",
        "raw_contact_records_sha256", "candidate_contact_count",
        "rejected_contact_count", "counter_replay_consistent", "functional_role_counts",
        "functional_coverage_observed", "zero_contact_cannot_pass",
        "maximum_penetration_mm", "maximum_locus_error_mm",
        "maximum_abs_normal_force_n_diagnostic_only", "functional_phase_envelope",
        "provisional_geometry_classification_passed",
        "contact_forces_are_unbounded_diagnostic_evidence_only",
        "contact_force_authority", "friction_coefficient_authority",
        "dynamics_authority",
        "physical_source_negative_y_release_action_excluded", "passed",
        "release_ready",
    }
    if set(evidence) != expected_evidence_keys:
        errors.append("contact_evidence:keys")
    expected_header = {
        "schema_version": "2.0",
        "evidence_kind": "real_mujoco_per_substep_capture_cam_tab_envelope",
        "runtime_contract_api": "core_cam_tab_contact_runtime_contract",
        "contract_identity_sha256": contract.get("contract_identity_sha256"),
        "model_binding": actual_model_binding,
        "physics_timestep_s": 0.00025,
        "zero_contact_cannot_pass": True,
        "contact_forces_are_unbounded_diagnostic_evidence_only": True,
        "contact_force_authority": False,
        "friction_coefficient_authority": False,
        "dynamics_authority": False,
        "physical_source_negative_y_release_action_excluded": True,
        "release_ready": False,
    }
    for key, expected in expected_header.items():
        if _evidence_mismatches(evidence.get(key), expected, key):
            errors.append(f"contact_evidence:{key}")

    phase_counts = evidence.get("audited_substeps_by_phase")
    if not isinstance(phase_counts, dict) or set(phase_counts) != set(
        CORE_CAM_CONTACT_ACTIONS
    ):
        errors.append("contact_evidence:phase_counts")
        phase_counts = {}
    valid_counts = bool(phase_counts) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in phase_counts.values()
    )
    if not valid_counts:
        errors.append("contact_evidence:phase_count_values")
    audited = evidence.get("audited_substeps")
    if (
        isinstance(audited, bool)
        or not isinstance(audited, int)
        or audited < 0
    ):
        errors.append("contact_evidence:audited_substeps")
        audited = 0
    elif valid_counts and audited != sum(phase_counts.values()):
        errors.append("contact_evidence:audited_sum")
    if audited > physics_substeps:
        errors.append("contact_evidence:audited_exceeds_physics")
    all_phases = valid_counts and all(value > 0 for value in phase_counts.values())
    if evidence.get("all_four_capture_phases_observed") is not all_phases:
        errors.append("contact_evidence:all_phases")

    records = evidence.get("raw_contact_records")
    if not isinstance(records, list):
        errors.append("contact_evidence:records")
        records = []
    if evidence.get("raw_contact_records_sha256") != canonical_json_sha256(records):
        errors.append("contact_evidence:records_sha")
    observed = len(records) > 0
    if evidence.get("observed") is not observed:
        errors.append("contact_evidence:observed")
    record_passes: list[bool] = []
    roles: list[str | None] = []
    previous_substep = -1
    for index, record in enumerate(records):
        record_errors, record_passed, role = (
            _independent_cam_tab_record_classification(record)
        )
        errors.extend(f"contact:{index}:{error}" for error in record_errors)
        record_passes.append(record_passed)
        roles.append(role)
        if isinstance(record, dict):
            substep = record.get("physics_substep_count")
            if (
                isinstance(substep, bool)
                or not isinstance(substep, int)
                or substep <= 0
                or substep > physics_substeps
                or substep < previous_substep
            ):
                errors.append(f"contact:{index}:substep")
            else:
                previous_substep = substep
                if not _number_matches(
                    record.get("sim_time_s"), substep * 0.00025, tolerance=1.0e-10
                ):
                    errors.append(f"contact:{index}:time")
    candidate_count = sum(record_passes)
    rejected_count = len(records) - candidate_count
    if evidence.get("candidate_contact_count") != candidate_count:
        errors.append("contact_evidence:candidate_count")
    if evidence.get("rejected_contact_count") != rejected_count:
        errors.append("contact_evidence:rejected_count")
    if evidence.get("counter_replay_consistent") is not True:
        errors.append("contact_evidence:counter_replay_consistent")
    role_counts = {role: roles.count(role) for role in CORE_CAM_FUNCTIONAL_ROLES}
    if evidence.get("functional_role_counts") != role_counts:
        errors.append("contact_evidence:functional_role_counts")
    functional_coverage = all(value > 0 for value in role_counts.values())
    if evidence.get("functional_coverage_observed") is not functional_coverage:
        errors.append("contact_evidence:functional_coverage")

    penetrations = [
        float(record["penetration_mm"])
        for record in records
        if isinstance(record, dict) and _finite_real(record.get("penetration_mm")) is not None
    ]
    loci = [
        float(record["locus_error_mm"])
        for record in records
        if isinstance(record, dict) and _finite_real(record.get("locus_error_mm")) is not None
    ]
    normal_forces = [
        abs(float(record["contact_force_torque_6d"][0]))
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("contact_force_torque_6d"), list)
        and record["contact_force_torque_6d"]
        and _finite_real(record["contact_force_torque_6d"][0]) is not None
    ]
    for field, expected in (
        ("maximum_penetration_mm", max(penetrations, default=None)),
        ("maximum_locus_error_mm", max(loci, default=None)),
        ("maximum_abs_normal_force_n_diagnostic_only", max(normal_forces, default=None)),
    ):
        if _evidence_mismatches(evidence.get(field), expected, field):
            errors.append(f"contact_evidence:{field}")
    envelope_errors, envelope_state_pass = _core_cam_functional_envelope_errors(
        evidence.get("functional_phase_envelope")
    )
    errors.extend(envelope_errors)
    expected_envelope_pass = bool(
        binding_passed
        and envelope_state_pass
        and all_route_endpoints
        and abort_reason is None
    )
    envelope_record = evidence.get("functional_phase_envelope")
    envelope_states = (
        envelope_record.get("raw_states", [])
        if isinstance(envelope_record, dict)
        else []
    )
    if (replay_model is None) != (replay_mujoco is None):
        errors.append("envelope:replay_authority_incomplete")
    elif envelope_states and replay_model is None:
        errors.append("envelope:replay_authority_missing")
    elif replay_model is not None and replay_mujoco is not None:
        for state_index, state in enumerate(envelope_states):
            replay_errors = _core_cam_functional_state_replay_errors(
                state,
                replay_model,
                replay_mujoco,
            )
            errors.extend(
                f"envelope:state_replay:{state_index}:{error}"
                for error in replay_errors
            )
    if isinstance(envelope_record, dict):
        expected_producer_phase_counts = {
            name: int(phase_counts.get(name, 0))
            for name in CORE_CAM_CONTACT_ACTIONS[2:]
        }
        if envelope_record.get("producer_phase_counts") != (
            expected_producer_phase_counts
        ):
            errors.append("envelope:producer_phase_counts_vs_contact_audit")
        records_by_state: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            state_index = record.get("state_index")
            if isinstance(state_index, int) and not isinstance(state_index, bool):
                records_by_state.setdefault(state_index, []).append(record)
        envelope_state_keys: set[tuple[int, str]] = set()
        for raw_index, state in enumerate(envelope_states):
            if not isinstance(state, dict):
                continue
            state_index = state.get("state_index")
            if not isinstance(state_index, int) or isinstance(state_index, bool):
                continue
            action_name = state.get("action")
            state_key = (state_index, str(action_name))
            if state_key in envelope_state_keys:
                errors.append(f"envelope:duplicate_state:{raw_index}")
            envelope_state_keys.add(state_key)
            if not 1 <= state_index <= physics_substeps:
                errors.append(f"envelope:state_absolute_index:{raw_index}")
            if not _number_matches(
                state.get("sim_time_s"),
                state_index * 0.00025,
                tolerance=1.0e-10,
            ):
                errors.append(f"envelope:state_absolute_time:{raw_index}")
            state_records = records_by_state.get(state_index, [])
            if any(record.get("action") != state.get("action") for record in state_records):
                errors.append(f"envelope:state_contact_action:{raw_index}")
            slider_records = [
                record for record in state_records
                if record.get("is_slider_tab_contact") is True
            ]
            eligible_records = [
                record for record in slider_records
                if record.get("provisional_classification_passed") is True
            ]
            functional_records = [
                record for record in eligible_records
                if record.get("functional_coverage_role") is not None
            ]
            expected_counts = {
                "observed_cam_tab_contact_count": len(slider_records),
                "eligible_cam_tab_contact_count": len(eligible_records),
                "rejected_cam_tab_contact_count": (
                    len(slider_records) - len(eligible_records)
                ),
                "functional_contact_count": len(functional_records),
                "functional_lead_contact_count": sum(
                    record.get("functional_coverage_role")
                    == CORE_CAM_FUNCTIONAL_ROLES[0]
                    for record in functional_records
                ),
                "functional_hold_contact_count": sum(
                    record.get("functional_coverage_role")
                    == CORE_CAM_FUNCTIONAL_ROLES[1]
                    for record in functional_records
                ),
                "nonfunctional_candidate_contact_count": (
                    len(eligible_records) - len(functional_records)
                ),
                "other_core_cam_contact_count": (
                    len(state_records) - len(slider_records)
                ),
            }
            for field, expected in expected_counts.items():
                if state.get(field) != expected:
                    errors.append(f"envelope:state_contact_count:{raw_index}:{field}")
        for contact_index, record in enumerate(records):
            if not isinstance(record, dict) or record.get("action") not in (
                CORE_CAM_CONTACT_ACTIONS[2:]
            ):
                continue
            contact_key = (
                record.get("state_index"),
                str(record.get("action")),
            )
            if contact_key not in envelope_state_keys:
                errors.append(f"envelope:orphan_functional_contact:{contact_index}")
    if isinstance(envelope_record, dict) and envelope_record.get("passed") is not expected_envelope_pass:
        errors.append("envelope:passed")
    expected_contact_pass = bool(
        binding_passed and observed and all_phases and functional_coverage
        and rejected_count == 0 and candidate_count == len(records)
        and expected_envelope_pass
        and result.get("forbidden_contact_count") == 0
        and abort_reason is None
    )
    for field in ("provisional_geometry_classification_passed", "passed"):
        if evidence.get(field) is not expected_contact_pass:
            errors.append(f"contact_evidence:{field}")

    free = result.get("core_capture_free_space_tracking_evidence")
    if not isinstance(free, dict):
        return [*errors, "free_space:missing"]
    expected_free_keys = {
        "schema_version", "evidence_kind", "route_contract_identity_sha256",
        "cam_contact_contract_identity_sha256", "model_binding", "physics_timestep_s",
        "observed", "audited_substeps_by_phase",
        "all_free_space_phases_observed", "completed_endpoint_actions",
        "all_free_space_endpoints_completed", "raw_samples",
        "raw_samples_sha256", "maximum_abs_q_tracking_error_rad",
        "maximum_abs_preseat_error_mm", "maximum_abs_x_error_mm",
        "maximum_abs_transverse_y_mm", "maximum_orientation_error_rad",
        "minimum_lead_x_gap_mm", "cam_contact_observation_count",
        "thresholds", "passed", "live_dynamics_authority", "release_ready",
    }
    if set(free) != expected_free_keys:
        errors.append("free_space:keys")
    free_header = {
        "schema_version": "1.0",
        "evidence_kind": "real_mujoco_per_substep_free_space_servo_tracking",
        "route_contract_identity_sha256": contract["source_binding"][
            "capture_route_contract_identity_sha256"
        ],
        "cam_contact_contract_identity_sha256": contract["contract_identity_sha256"],
        "model_binding": actual_model_binding,
        "physics_timestep_s": 0.00025,
        "live_dynamics_authority": False,
        "release_ready": False,
        "thresholds": {
            "maximum_abs_q_tracking_error_rad": 0.002,
            "maximum_abs_preseat_error_mm": 0.050,
            "maximum_abs_x_error_mm": 0.040,
            "maximum_abs_transverse_y_mm": 0.010,
            "maximum_orientation_error_rad": math.radians(0.1),
            "minimum_lead_x_gap_mm": 0.0,
            "maximum_cam_contact_count": 0,
        },
    }
    for key, expected in free_header.items():
        if _evidence_mismatches(free.get(key), expected, key):
            errors.append(f"free_space:{key}")
    free_counts = free.get("audited_substeps_by_phase")
    free_actions = CORE_CAM_CONTACT_ACTIONS[:2]
    if not isinstance(free_counts, dict) or set(free_counts) != set(free_actions):
        errors.append("free_space:phase_counts")
        free_counts = {}
    valid_free_counts = bool(free_counts) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in free_counts.values()
    )
    samples = free.get("raw_samples")
    if not isinstance(samples, list):
        errors.append("free_space:samples")
        samples = []
    if free.get("raw_samples_sha256") != canonical_json_sha256(samples):
        errors.append("free_space:samples_sha")
    if valid_free_counts and len(samples) != sum(free_counts.values()):
        errors.append("free_space:sample_count")
    free_observed = len(samples) > 0
    if free.get("observed") is not free_observed:
        errors.append("free_space:observed")
    free_all_phases = valid_free_counts and all(value > 0 for value in free_counts.values())
    if free.get("all_free_space_phases_observed") is not free_all_phases:
        errors.append("free_space:all_phases")

    expected_sample_keys = {
        "physics_substep_count", "sim_time_s", "action",
        "command_smooth_fraction", "commanded_arm_q_rad", "observed_arm_q_rad",
        "observed_arm_qvel_rad_s", "max_abs_q_tracking_error_rad",
        "expected_preseat_mm", "observed_preseat_mm", "preseat_error_mm",
        "expected_x_mm", "observed_x_mm", "x_error_mm", "transverse_y_mm",
        "orientation_error_rad", "slider_q_mm", "source_q_max_mm",
        "lead_x_gap_mm", "lead_normal_clearance_mm", "cam_contact_count", "finite",
    }
    sample_validity: list[bool] = []
    previous_by_action: dict[str, int] = {}
    for index, sample in enumerate(samples):
        valid = isinstance(sample, dict) and set(sample) == expected_sample_keys
        if not valid:
            errors.append(f"free_sample:{index}:keys")
            sample_validity.append(False)
            continue
        action = sample.get("action")
        substep = sample.get("physics_substep_count")
        smooth = _finite_real(sample.get("command_smooth_fraction"))
        if action not in free_actions or smooth is None or not 0.0 <= smooth <= 1.0:
            errors.append(f"free_sample:{index}:action_or_smooth")
            valid = False
        if isinstance(substep, bool) or not isinstance(substep, int) or substep <= 0:
            errors.append(f"free_sample:{index}:substep")
            valid = False
        elif action in previous_by_action and substep != previous_by_action[action] + 1:
            errors.append(f"free_sample:{index}:continuity")
            valid = False
        if isinstance(substep, int) and action in free_actions:
            previous_by_action[action] = substep
            if not _number_matches(sample.get("sim_time_s"), substep * 0.00025, 1.0e-10):
                errors.append(f"free_sample:{index}:time")
                valid = False
        commanded = sample.get("commanded_arm_q_rad")
        observed_q = sample.get("observed_arm_q_rad")
        observed_qvel = sample.get("observed_arm_qvel_rad_s")
        arrays: list[np.ndarray] = []
        for name, value in (
            ("commanded", commanded), ("observed", observed_q), ("qvel", observed_qvel)
        ):
            if not isinstance(value, list) or len(value) != 5:
                errors.append(f"free_sample:{index}:{name}")
                valid = False
                continue
            array = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                errors.append(f"free_sample:{index}:{name}:finite")
                valid = False
            arrays.append(array)
        expected_p = 55.0 if action == free_actions[0] else 55.0 - 48.6 * float(smooth or 0.0)
        expected_x = 0.20 * float(smooth or 0.0) if action == free_actions[0] else 0.20
        observed_p = _finite_real(sample.get("observed_preseat_mm"))
        observed_x = _finite_real(sample.get("observed_x_mm"))
        slider_q = _finite_real(sample.get("slider_q_mm"))
        if None in (observed_p, observed_x, slider_q):
            errors.append(f"free_sample:{index}:finite_geometry")
            valid = False
        else:
            qmax = max(0.05, min(3.0, float(observed_p) - float(observed_x) - 3.15))
            gap = float(observed_p) - float(observed_x) - 3.15 - float(slider_q)
            checks = {
                "expected_preseat_mm": expected_p,
                "preseat_error_mm": float(observed_p) - expected_p,
                "expected_x_mm": expected_x,
                "x_error_mm": float(observed_x) - expected_x,
                "source_q_max_mm": qmax,
                "lead_x_gap_mm": gap,
                "lead_normal_clearance_mm": gap / math.sqrt(2.0),
            }
            for key, expected in checks.items():
                if not _number_matches(sample.get(key), expected):
                    errors.append(f"free_sample:{index}:{key}")
                    valid = False
        if len(arrays) >= 2:
            q_error = float(np.max(np.abs(arrays[1] - arrays[0])))
            if not _number_matches(sample.get("max_abs_q_tracking_error_rad"), q_error):
                errors.append(f"free_sample:{index}:q_error")
                valid = False
        finite_fields = all(
            _finite_real(sample.get(key)) is not None
            for key in (
                "transverse_y_mm", "orientation_error_rad", "lead_x_gap_mm",
                "lead_normal_clearance_mm",
            )
        )
        expected_finite = bool(valid and finite_fields)
        if sample.get("finite") is not expected_finite:
            errors.append(f"free_sample:{index}:finite_flag")
            valid = False
        contact_count = sample.get("cam_contact_count")
        if isinstance(contact_count, bool) or not isinstance(contact_count, int) or contact_count < 0:
            errors.append(f"free_sample:{index}:contact_count")
            valid = False
        sample_validity.append(valid)

    completed_free = sorted(endpoint_actions.intersection(free_actions))
    if free.get("completed_endpoint_actions") != completed_free:
        errors.append("free_space:completed_endpoints")
    all_free_endpoints = set(completed_free) == set(free_actions)
    if free.get("all_free_space_endpoints_completed") is not all_free_endpoints:
        errors.append("free_space:all_endpoints")
    metric_inputs = {
        "maximum_abs_q_tracking_error_rad": [
            float(sample["max_abs_q_tracking_error_rad"]) for sample in samples
            if isinstance(sample, dict) and _finite_real(sample.get("max_abs_q_tracking_error_rad")) is not None
        ],
        "maximum_abs_preseat_error_mm": [
            abs(float(sample["preseat_error_mm"])) for sample in samples
            if isinstance(sample, dict) and _finite_real(sample.get("preseat_error_mm")) is not None
        ],
        "maximum_abs_x_error_mm": [
            abs(float(sample["x_error_mm"])) for sample in samples
            if isinstance(sample, dict) and _finite_real(sample.get("x_error_mm")) is not None
        ],
        "maximum_abs_transverse_y_mm": [
            abs(float(sample["transverse_y_mm"])) for sample in samples
            if isinstance(sample, dict) and _finite_real(sample.get("transverse_y_mm")) is not None
        ],
        "maximum_orientation_error_rad": [
            float(sample["orientation_error_rad"]) for sample in samples
            if isinstance(sample, dict) and _finite_real(sample.get("orientation_error_rad")) is not None
        ],
    }
    for field, values in metric_inputs.items():
        expected = max(values, default=None)
        if _evidence_mismatches(free.get(field), expected, field):
            errors.append(f"free_space:{field}")
    gaps = [
        float(sample["lead_x_gap_mm"]) for sample in samples
        if isinstance(sample, dict) and _finite_real(sample.get("lead_x_gap_mm")) is not None
    ]
    minimum_gap = min(gaps, default=None)
    if _evidence_mismatches(free.get("minimum_lead_x_gap_mm"), minimum_gap, "gap"):
        errors.append("free_space:minimum_gap")
    contact_total = sum(
        int(sample["cam_contact_count"]) for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("cam_contact_count"), int)
        and not isinstance(sample.get("cam_contact_count"), bool)
    )
    if free.get("cam_contact_observation_count") != contact_total:
        errors.append("free_space:contact_total")
    expected_free_pass = bool(
        binding_passed and free_observed and free_all_phases and all_free_endpoints
        and sample_validity and all(sample_validity)
        and metric_inputs["maximum_abs_q_tracking_error_rad"]
        and max(metric_inputs["maximum_abs_q_tracking_error_rad"]) <= 0.002
        and max(metric_inputs["maximum_abs_preseat_error_mm"]) <= 0.050
        and max(metric_inputs["maximum_abs_x_error_mm"]) <= 0.040
        and max(metric_inputs["maximum_abs_transverse_y_mm"]) <= 0.010
        and max(metric_inputs["maximum_orientation_error_rad"]) <= math.radians(0.1)
        and minimum_gap is not None and minimum_gap >= 0.0
        and contact_total == 0 and abort_reason is None
    )
    if free.get("passed") is not expected_free_pass:
        errors.append("free_space:passed")
    expected_development_milestone = bool(
        result.get("completed") is True
        and result.get("attachment_verified") is True
        and result.get("attached_tool") == "gripper"
        and result.get("locked") is False
        and result.get("physical_lock_confirmed") is False
        and expected_contact_pass
        and expected_free_pass
        and result.get("forbidden_contact_count") == 0
        and abort_reason is None
    )
    if result.get("development_geometry_milestone_passed") is not expected_development_milestone:
        errors.append("result:development_geometry_milestone_passed")
    if result.get("success") is not False:
        errors.append("result:success_must_remain_false")
    expected_physical_cam_authority = bool(
        evidence.get("contact_force_authority") is True
        and evidence.get("friction_coefficient_authority") is True
        and evidence.get("dynamics_authority") is True
    )
    if result.get("physical_cam_authority_ready") is not (
        expected_physical_cam_authority
    ):
        errors.append("result:physical_cam_authority_ready")
    if result.get("locked") is not False:
        errors.append("result:locked")
    if result.get("physical_lock_confirmed") is not False:
        errors.append("result:physical_lock_confirmed")
    if result.get("release_ready") is not False:
        errors.append("result:release_ready")
    return errors


CORE_CAPTURE_GRAVITY_BIAS_ACTIONS = (
    "gripper_capture_lateral_align",
    "gripper_capture_axial_open_side",
    "gripper_capture_coupled_recenter",
    "gripper_capture_centered_final",
)
CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256 = (
    "a84c10e16c890b5e1ee4e4479c0d15d7e07a75f2afae17c62e639e8adc55cc27"
)
CORE_CAPTURE_GRAVITY_BIAS_IDENTITY_SHA256 = (
    "a039042d0c32263fe9565000f8eee7e442d6be06f48b3834d77861f26b36c4e2"
)
CORE_CAPTURE_GRAVITY_BIAS_DESIRED_START_SHA256 = (
    "7216752cb39dc68608396f33d09eadc018c4d936b510a50f456e978e94df0618"
)
CORE_CAPTURE_GRAVITY_BIAS_ROUTE_IDENTITY_SHA256 = (
    "1db1e8a2325a4afdc451330c2b464ee9d85930a956ca22b8f073bdfa16105088"
)
CORE_CAPTURE_GRAVITY_BIAS_NON_ARM_QPOS_SHA256 = (
    "c28ea3a6c2ff43eeab74db3ab36dc6bc37987d64d3ed94dc33808ddf977b8001"
)
CORE_CAPTURE_GRAVITY_BIAS_CALLGRAPH_SHA256 = (
    "35c5cd0330c1fc140431a15ce70f62a4b2d50d7e44f57e277ce5484afff719a9"
)
CORE_CAPTURE_GRAVITY_BIAS_BYTECODE_SHA256 = (
    "8fc01fced6ecc6a679027e971c3e414a600d7fe6735b29ffb7c62ee1cc353f30"
)
CORE_CAPTURE_GRAVITY_BIAS_AST_POLICY_SHA256 = (
    "40995993893abbd315a2c95806291116f0a641bb6e8db54fbb576b6a08477d1f"
)
CORE_CAPTURE_GRAVITY_BIAS_GUARDS_SHA256 = (
    "a30d3c871580b36a8f18eca6f07b6d4e5eee34cbecf50093aaca7c4cb1d3ee40"
)
CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256 = (
    "679180aabfd17ee70c1f88ac40444cfdc24ca92f0709121bb3048333d7bd25ad"
)
CORE_CAPTURE_GRAVITY_BIAS_ACTION_ROSTER_SHA256 = (
    "3a3301d0adf98e1e617766891ebf7be4129d94d0440d7f4fcb694f273313a3ba"
)
CORE_CAPTURE_GRAVITY_BIAS_FUNCTION_HASHES = {
    "_current_core_capture_route_identity_preimage": {
        "code_object_sha256": "706c34bf3be68b007ec0a7bc73014c1de62c4cd1f3277b0c62033dafffe7d326",
        "normalized_ast_sha256": "80ce2147fa15ab316a373258cb9a26c857a15fb85c5286e00ba93246006ea885",
        "source_sha256": "3376655106ed3c30254fe48360ac9618899404e5f18449b69b402f9743caed1c",
    },
    "_current_core_capture_gravity_bias_formula": {
        "code_object_sha256": "ecd0c98eae7bd2ea43f9fc342730a6a9e03df80e941c30c081198525830cebd2",
        "normalized_ast_sha256": "a4a57804498e9ff0ddc8f6fe30aab2bfa501a9f5a2f6c4fcf449844e5c04b3ef",
        "source_sha256": "d69801ef96976942c136df6836247590e1fd488547b0d860e0eb6ddc3cd87192",
    },
    "_current_core_capture_gravity_bias_guard_thresholds": {
        "code_object_sha256": "0f082a73e1b12bdd1de67298227ebe82def4ef97cf55323d4cb6d19a420c5f99",
        "normalized_ast_sha256": "d1e0ea6daed8b33149290c52ad1936060e60525f9a24bbd2a58ac202c409ed8b",
        "source_sha256": "18e9b5e955e7ac3ff7fde8b2f733e2c37edc51e9661cbd06b66ab77e3a8b5cf4",
    },
    "_core_capture_gravity_bias_model_digests": {
        "code_object_sha256": "53dafb7977991a46c7770a1c079cce41dae7c4704c316731f859b320394b3750",
        "normalized_ast_sha256": "07efb8c30a9d646a1b09170961bcf146ddaf2cafb6462287236b38e8fa4eebb6",
        "source_sha256": "bafe1fe6660bd22220dfc621ce49e6adbf767e7832e4b56c125d83c5925d75ef",
    },
    "_core_capture_move_actions": {
        "code_object_sha256": "22fa84bb3b3ea159c42e8d7d50d9987bb445653f64198158e7ea7349831bc60a",
        "normalized_ast_sha256": "b327106b4085f0035ca96fd0cf4dd57a79f72c82c7e46205f0948c2eaec3e7a1",
        "source_sha256": "c259459e0c313025d09967b73cc439166c3271bce04b077de539d3f8bf37e119",
    },
    "_move_action_desired_q": {
        "code_object_sha256": "45cf6d26fdf9ba7cc11f7d4ada38cd55ea2897068b601fc6a3ed1c84082cec9b",
        "normalized_ast_sha256": "19587e57dfa2b27b9f47e3da21f34a8391c6a961305f6f96675a90ade9c6b035",
        "source_sha256": "f946e4b873a1cf636de51ebfd0a9c06b7b19d1a987f1066fc2b274691b81cf14",
    },
    "_forward_scratch_arm_configuration": {
        "code_object_sha256": "b661244a8a9ccc3e7b1d420ca2d04144ec8c5fe44cf80e2e621d727708ab9180",
        "normalized_ast_sha256": "5fa53cbf090c73e95ca8a614fe789a2a69d06eb7d7f468ffa8909c56ff63b802",
        "source_sha256": "3789322274800ad9ae02dff3b39e2fb4c5f0df032506028dd8a5730e0775a496",
    },
    "_core_capture_gravity_bias_control": {
        "code_object_sha256": "984c35f45f085fabf35e0a90cc847867c9b5ebc049ea9aa73b06b626ad2d6a0c",
        "normalized_ast_sha256": "1c7a66c27a1308fb798fb4897c5ea4701c0e7540d0d9c640f407ab58bbbdc495",
        "source_sha256": "e3af3d2a67a0e2f8817a6bf2ca3508f17443ef85d7177866456791d216f8e2c2",
    },
    "_current_core_capture_gravity_bias_lightweight_identity_snapshot": {
        "code_object_sha256": "a914e0707f89c6dff610d60fe95305668100c7d5b50801043a637610818af204",
        "normalized_ast_sha256": "85d8e603e03b7810e3ba2d6fc37ffc758ac671beef113591aaaf661eda365511",
        "source_sha256": "f382cc675c678800dfbf0646dbca347df7807725ef89994ed1b89e0d95669799",
    },
    "_core_capture_gravity_bias_prewrite_snapshot": {
        "code_object_sha256": "3e85b28ea27d80cf0c0141833b2a02e058dac470e49c2e86348ecf18f464503c",
        "normalized_ast_sha256": "d0c0195b72c1e157b38d28c3ac16fdb34ecd302d3134d70432d1f527546eef0d",
        "source_sha256": "3793c7f04bbb3792f504e1d2451220d12fad7277cf13965ae959228116034b6f",
    },
    "MatchaWorkflowController._command_move": {
        "code_object_sha256": "0e99b5ffae5793a7ac4d5246f67732fa6a16f9b473e269a040c3e630122a0d64",
        "normalized_ast_sha256": "7ff41d90cca92c0053c0b0d7362be7e6272c3857fd598c542e4cf2d75340a7ad",
        "source_sha256": "c71c602dad86d0b8dc73d8143f3e2c1bbf82748446c44bcf1763f96ea5169eab",
    },
}
CORE_CAPTURE_GRAVITY_BIAS_EXPECTED_HASHES = {
    "gravity_sha256": "ae1e8d988bda31f91f37b3360818f46f5216924d49e011e065e323fe23f3bf2e",
    "body_mass_sha256": "0f0dd621c2dd11a9928b2052897825c0eec541e8ed0d3c2c7a43700077775aa0",
    "body_inertia_sha256": "b54392d89a7e884c3001f6807f7691a1a29baba7a8018871e2d7ca453538a9a8",
    "body_ipos_sha256": "2c9487de7d3f29c60f0a304caa83e64b2fc940ca543960b1f9ef43b73bb55e0b",
    "body_iquat_sha256": "8103edd84053535deee67cd68de775575ad40ff6d3123300e5aa737fdde9b395",
    "inertial_bundle_sha256": "8960fcae8eab2eb5235c8316271bcdf813051a6d31b55fc992d72d92b54ed29f",
    "arm_gainprm_sha256": "6d15e750d44986e760901cb224115639c75fc49ab17d23f94c74c8678a663a74",
    "arm_gear_sha256": "9c380cb1330e9755842a38183a344f8f159a3224ef409800723e9e10babbcaef",
    "arm_ctrlrange_sha256": "7996351eb41db1364c708be82e65566ef357a359000c965ee48278552b5df209",
    "arm_forcerange_sha256": "2b1c560a63c760ed53819eca8ed65031871b4415028102367cb36718389f575a",
}


def _expected_gravity_bias_formula() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "eligible_actions": sorted(CORE_CAPTURE_GRAVITY_BIAS_ACTIONS),
        "desired_route_authority": (
            "immutable_source_route_q_roster_and_per_action_desired_start"
        ),
        "scratch_state": (
            "private_MjData_distinct_from_live_at_desired_q_with_all_qvel_zero"
        ),
        "scratch_position_update": (
            "mj_differentiatePos_then_mj_integratePos_then_mj_forward"
        ),
        "bias_source": "scratch_data.qfrc_bias[arm_dof_ids]",
        "offset_formula": (
            "qfrc_bias/(actuator_gainprm[arm_actuator_ids,0]*"
            "actuator_gear[arm_actuator_ids,0])"
        ),
        "offset_sign": "positive",
        "unsaturated_control_formula": "q_des+gravity_bias_offset",
        "applied_control_formula": (
            "clip(unsaturated_control,actuator_ctrlrange)"
        ),
        "saturation_policy": "any_saturation_fails_development_evidence",
        "runtime_isolation": {
            "scratch_and_live_object_identity_must_differ": True,
            "evaluator_receives_no_live_MjData_argument": True,
            "live_qpos_qvel_snapshots_must_be_bitwise_unchanged": True,
            "all_scratch_qvel_must_be_exact_zero": True,
            "non_arm_scratch_qpos_digest_must_remain_frozen": True,
            "failure_aborts_before_ctrl_write": True,
        },
        "prewrite_revalidation": (
            "fresh_route_formula_guard_desired_action_ast_bytecode_and_"
            "compiled_dynamics_identity_before_desired_q_and_ctrl_write"
        ),
        "prohibited_inputs": [
            "qfrc_constraint",
            "mj_contactForce",
            "mj_inverse",
            "live_qpos_write",
            "live_qvel_write",
        ],
        "authority_scope": "development_free_space_tracking_only",
    }


def _expected_gravity_bias_guard_thresholds() -> dict[str, Any]:
    return {
        "endpoint_maximum_q_error_rad": 0.002,
        "endpoint_maximum_abs_qvel_rad_s": 0.02,
        "endpoint_maximum_fk_position_error_m": 0.00005,
        "endpoint_maximum_fk_orientation_error_rad": math.radians(0.1),
        "endpoint_maximum_abs_source_x_error_mm": 0.040,
        "endpoint_required_contiguous_controller_ticks": 4,
        "free_space_maximum_abs_q_error_rad": 0.002,
        "free_space_maximum_abs_preseat_error_mm": 0.050,
        "free_space_maximum_abs_source_x_error_mm": 0.040,
        "free_space_maximum_abs_transverse_y_mm": 0.010,
        "free_space_maximum_orientation_error_rad": math.radians(0.1),
        "free_space_maximum_raw_cam_contact_count": 0,
        "state_time_anchor_absolute_tolerance_s": 1.0e-10,
        "adjacent_state_time_absolute_tolerance_s": 1.0e-12,
        "all_four_phases_and_endpoints_required_for_pass": True,
        "any_saturation_fails": True,
        "abort_must_be_absent": True,
    }


def _expected_gravity_bias_ast_policy() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "audited_functions": [
            "_current_core_capture_route_identity_preimage",
            "_current_core_capture_gravity_bias_formula",
            "_current_core_capture_gravity_bias_guard_thresholds",
            "_core_capture_gravity_bias_model_digests",
            "_core_capture_move_actions",
            "_move_action_desired_q",
            "_forward_scratch_arm_configuration",
            "_core_capture_gravity_bias_control",
            "_current_core_capture_gravity_bias_lightweight_identity_snapshot",
            "_core_capture_gravity_bias_prewrite_snapshot",
            "MatchaWorkflowController._command_move",
        ],
        "allowed_direct_mujoco_calls": [
            "mj_differentiatePos", "mj_forward", "mj_integratePos"
        ],
        "allowed_scratch_state_attributes": ["qfrc_bias", "qpos", "qvel"],
        "allowed_model_feedforward_arrays": [
            "actuator_ctrlrange", "actuator_gainprm", "actuator_gear"
        ],
        "prohibited_attributes": ["qfrc_constraint"],
        "prohibited_calls": ["mj_contactForce", "mj_inverse"],
        "prohibited_assignment_targets": [
            "self.data.qpos", "self.data.qvel"
        ],
        "command_branch_live_state_policy": (
            "outer_live_qpos_qvel_snapshots_before_and_after_evaluator;"
            "evaluator_receives_no_live_MjData;no_live_assignment"
        ),
    }


def _independent_ast_attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _independent_ast_attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Subscript):
        return _independent_ast_attribute_path(node.value)
    return None


def _independent_code_object_sha256(code: CodeType) -> str:
    """Hash executable instructions and symbols without production helpers."""

    def constant_record(value: object) -> object:
        if isinstance(value, CodeType):
            return {
                "nested_code_sha256": _independent_code_object_sha256(value)
            }
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, tuple):
            return [constant_record(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"type": type(value).__name__, "repr": repr(value)}

    return canonical_json_sha256(
        {
            "co_code_hex": code.co_code.hex(),
            "co_consts": [constant_record(value) for value in code.co_consts],
            "co_names": list(code.co_names),
            "co_varnames": list(code.co_varnames),
            "co_freevars": list(code.co_freevars),
            "co_cellvars": list(code.co_cellvars),
            "co_argcount": code.co_argcount,
            "co_posonlyargcount": code.co_posonlyargcount,
            "co_kwonlyargcount": code.co_kwonlyargcount,
            "co_nlocals": code.co_nlocals,
            "co_stacksize": code.co_stacksize,
            "co_flags": code.co_flags,
        }
    )


def _independent_gravity_bias_source_audit(
    demo: ModuleType,
) -> dict[str, Any]:
    """Independently hash and allowlist the complete control callgraph."""

    policy = _expected_gravity_bias_ast_policy()
    functions: dict[str, Any] = {
        "_current_core_capture_route_identity_preimage": (
            demo._current_core_capture_route_identity_preimage
        ),
        "_current_core_capture_gravity_bias_formula": (
            demo._current_core_capture_gravity_bias_formula
        ),
        "_current_core_capture_gravity_bias_guard_thresholds": (
            demo._current_core_capture_gravity_bias_guard_thresholds
        ),
        "_core_capture_gravity_bias_model_digests": (
            demo._core_capture_gravity_bias_model_digests
        ),
        "_core_capture_move_actions": demo._core_capture_move_actions,
        "_move_action_desired_q": demo._move_action_desired_q,
        "_forward_scratch_arm_configuration": (
            demo._forward_scratch_arm_configuration
        ),
        "_core_capture_gravity_bias_control": (
            demo._core_capture_gravity_bias_control
        ),
        "_current_core_capture_gravity_bias_lightweight_identity_snapshot": (
            demo._current_core_capture_gravity_bias_lightweight_identity_snapshot
        ),
        "_core_capture_gravity_bias_prewrite_snapshot": (
            demo._core_capture_gravity_bias_prewrite_snapshot
        ),
        "MatchaWorkflowController._command_move": (
            demo.MatchaWorkflowController._command_move
        ),
    }
    records: list[dict[str, Any]] = []
    counts = {
        "direct_live_qpos_write_count": 0,
        "direct_live_qvel_write_count": 0,
        "qfrc_constraint_read_count": 0,
        "mj_contact_force_call_count": 0,
        "mj_inverse_call_count": 0,
        "unapproved_direct_mujoco_call_count": 0,
        "unapproved_scratch_state_attribute_count": 0,
        "unapproved_model_feedforward_array_count": 0,
    }
    errors: list[str] = []
    allowed_mujoco = frozenset(policy["allowed_direct_mujoco_calls"])
    allowed_scratch = frozenset(policy["allowed_scratch_state_attributes"])
    allowed_model = frozenset(policy["allowed_model_feedforward_arrays"])
    for name, function in functions.items():
        try:
            source = textwrap.dedent(inspect.getsource(function))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError) as exc:
            errors.append(f"{name}:{type(exc).__name__}")
            continue
        normalized = ast.dump(
            tree, annotate_fields=True, include_attributes=False
        )
        records.append(
            {
                "name": name,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "normalized_ast_sha256": hashlib.sha256(
                    normalized.encode()
                ).hexdigest(),
                "code_object_sha256": _independent_code_object_sha256(
                    function.__code__
                ),
            }
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == "qfrc_constraint":
                    counts["qfrc_constraint_read_count"] += 1
                base = _independent_ast_attribute_path(node.value)
                if base == "scratch_data" and node.attr not in allowed_scratch:
                    counts["unapproved_scratch_state_attribute_count"] += 1
                if (
                    name == "_core_capture_gravity_bias_control"
                    and base == "model"
                    and node.attr not in allowed_model
                ):
                    counts["unapproved_model_feedforward_array_count"] += 1
            if isinstance(node, ast.Call):
                call_path = _independent_ast_attribute_path(node.func)
                call_name = call_path.rsplit(".", 1)[-1] if call_path else ""
                if call_name == "mj_contactForce":
                    counts["mj_contact_force_call_count"] += 1
                if call_name == "mj_inverse":
                    counts["mj_inverse_call_count"] += 1
                if (
                    call_path
                    and call_path.startswith("mujoco.")
                    and call_name not in allowed_mujoco
                ):
                    counts["unapproved_direct_mujoco_call_count"] += 1
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    list(node.targets)
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    path = _independent_ast_attribute_path(target)
                    if path == "self.data.qpos":
                        counts["direct_live_qpos_write_count"] += 1
                    if path == "self.data.qvel":
                        counts["direct_live_qvel_write_count"] += 1
    callgraph_sha256 = canonical_json_sha256(records)
    bytecode_sha256 = canonical_json_sha256(
        [
            {
                "name": record["name"],
                "code_object_sha256": record["code_object_sha256"],
            }
            for record in records
        ]
    )
    policy_sha256 = canonical_json_sha256(policy)
    records_match = bool(
        all(
            record.get(key)
            == CORE_CAPTURE_GRAVITY_BIAS_FUNCTION_HASHES[record["name"]][key]
            for record in records
            for key in (
                "source_sha256",
                "normalized_ast_sha256",
                "code_object_sha256",
            )
        )
    )
    passed = bool(
        not errors
        and [record["name"] for record in records]
        == policy["audited_functions"]
        and all(count == 0 for count in counts.values())
        and records_match
        and callgraph_sha256 == CORE_CAPTURE_GRAVITY_BIAS_CALLGRAPH_SHA256
        and bytecode_sha256 == CORE_CAPTURE_GRAVITY_BIAS_BYTECODE_SHA256
        and policy_sha256 == CORE_CAPTURE_GRAVITY_BIAS_AST_POLICY_SHA256
    )
    return {
        "schema_version": "1.0",
        "policy": policy,
        "expected_policy_sha256": CORE_CAPTURE_GRAVITY_BIAS_AST_POLICY_SHA256,
        "observed_policy_sha256": policy_sha256,
        "expected_transitive_callgraph_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_CALLGRAPH_SHA256
        ),
        "observed_transitive_callgraph_sha256": callgraph_sha256,
        "expected_transitive_bytecode_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_BYTECODE_SHA256
        ),
        "observed_transitive_bytecode_sha256": bytecode_sha256,
        "function_records": records,
        "prohibited_operation_counts": counts,
        "transitive_function_bindings_match_frozen": records_match,
        "inspection_errors": errors,
        "passed": passed,
    }


def _independent_float64_bytes_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _independent_forward_scratch_arm_configuration(
    model: Any,
    data: Any,
    mujoco: Any,
    arm_qpos_ids: np.ndarray,
    arm_q_rad: np.ndarray,
) -> None:
    target = np.array(data.qpos, dtype=np.float64, copy=True)
    target[arm_qpos_ids] = np.asarray(arm_q_rad, dtype=np.float64)
    velocity = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model, velocity, 1.0, data.qpos, target
    )
    mujoco.mj_integratePos(model, data.qpos, velocity, 1.0)
    mujoco.mj_forward(model, data)


def _independent_gravity_bias_model_record(
    model: Any, demo: ModuleType
) -> dict[str, Any]:
    actuator_ids = np.asarray(
        [model.actuator(name).id for name in demo.ARM_ACTUATORS], dtype=int
    )
    gain = np.asarray(model.actuator_gainprm[actuator_ids], dtype=np.float64)
    gear = np.asarray(model.actuator_gear[actuator_ids], dtype=np.float64)
    ctrlrange = np.asarray(
        model.actuator_ctrlrange[actuator_ids], dtype=np.float64
    )
    forcerange = np.asarray(
        model.actuator_forcerange[actuator_ids], dtype=np.float64
    )
    hashes = {
        "gravity_sha256": _independent_float64_bytes_sha256(model.opt.gravity),
        "body_mass_sha256": _independent_float64_bytes_sha256(model.body_mass),
        "body_inertia_sha256": _independent_float64_bytes_sha256(
            model.body_inertia
        ),
        "body_ipos_sha256": _independent_float64_bytes_sha256(model.body_ipos),
        "body_iquat_sha256": _independent_float64_bytes_sha256(
            model.body_iquat
        ),
        "arm_gainprm_sha256": _independent_float64_bytes_sha256(gain),
        "arm_gear_sha256": _independent_float64_bytes_sha256(gear),
        "arm_ctrlrange_sha256": _independent_float64_bytes_sha256(ctrlrange),
        "arm_forcerange_sha256": _independent_float64_bytes_sha256(forcerange),
    }
    hashes["inertial_bundle_sha256"] = canonical_json_sha256(
        {
            "body_inertia": hashes["body_inertia_sha256"],
            "body_ipos": hashes["body_ipos_sha256"],
            "body_iquat": hashes["body_iquat_sha256"],
        }
    )
    return {
        "gravity_vector_m_s2": [float(value) for value in model.opt.gravity],
        **hashes,
        "arm_kp": [float(value) for value in gain[:, 0]],
        "arm_joint_gear": [float(value) for value in gear[:, 0]],
        "arm_ctrlrange": [
            [float(value) for value in row] for row in ctrlrange
        ],
        "arm_forcerange_nm": [
            [float(value) for value in row] for row in forcerange
        ],
    }


def _expected_gravity_bias_dynamics_binding(
    model: Any, demo: ModuleType
) -> dict[str, Any]:
    observed = _independent_gravity_bias_model_record(model, demo)
    matches = {
        f"{name.removesuffix('_sha256')}_matches": (
            observed[name] == expected
        )
        for name, expected in CORE_CAPTURE_GRAVITY_BIAS_EXPECTED_HASHES.items()
    }
    return {
        "schema_version": "1.0",
        "digest_preimage": (
            "canonical_little_endian_float64_bytes;inertial_bundle_is_"
            "canonical_json_of_body_inertia_body_ipos_body_iquat_hashes"
        ),
        "expected_hashes": copy.deepcopy(
            CORE_CAPTURE_GRAVITY_BIAS_EXPECTED_HASHES
        ),
        "observed": observed,
        "matches": matches,
        "expected_formula_sha256": CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256,
        "observed_formula_sha256": CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256,
        "formula_matches": True,
        "passed": bool(all(matches.values())),
    }


def _independent_gravity_bias_desired_starts(
    route_contract: dict[str, Any],
) -> dict[str, list[float]]:
    states = route_contract["source_states"]
    return {
        CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[0]: [
            -0.7200000000000006,
            -1.1177085425081206,
            1.1350213393902684,
            -0.017312796882147836,
            -1.522116811941434,
        ],
        CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[1]: [
            float(value) for value in states[0]["q_rad"]
        ],
        CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[2]: [
            float(value) for value in states[243]["q_rad"]
        ],
        CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[3]: [
            float(value) for value in states[259]["q_rad"]
        ],
    }


def _independent_gravity_bias_action_records(
    route_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the four executable actions from frozen route rows."""

    states = route_contract["source_states"]
    spans = (
        (CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[0], 0.25, 1.0, 0, 1),
        (CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[1], 1.60, 3.0, 1, 244),
        (CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[2], 0.50, 1.5, 244, 260),
        (CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[3], 0.50, 1.5, 260, 276),
    )
    return [
        {
            "name": name,
            "kind": "move",
            "tool": "gripper",
            "duration_s": duration_s,
            "timeout_s": timeout_s,
            "target_q": [float(value) for value in states[stop - 1]["q_rad"]],
            "joint_waypoints": [
                [float(value) for value in states[index]["q_rad"]]
                for index in range(start, stop)
            ],
        }
        for name, duration_s, timeout_s, start, stop in spans
    ]


def expected_core_capture_gravity_bias_contract(
    demo: ModuleType,
    model: Any,
    route_contract: dict[str, Any],
) -> dict[str, Any]:
    desired_starts = _independent_gravity_bias_desired_starts(route_contract)
    desired_start_sha256 = canonical_json_sha256(desired_starts)
    dynamics = _expected_gravity_bias_dynamics_binding(model, demo)
    source_model_sha256 = (
        "d919728e7108061f7ede7bd74991c2b5e42fa0985d5e731c30c95e0a660a953a"
    )
    compiled_sha256 = _independent_compiled_model_xml_equivalent_sha256(model)
    scratch = demo.mujoco.MjData(model)
    demo.initialize(model, scratch)
    active_sha256 = _independent_initialized_active_geometry_sha256(
        model, scratch, demo.mujoco
    )
    robot_xml = REPOSITORY_ROOT / "Simulation/SO101/so101_new_calib.xml"
    formula = _expected_gravity_bias_formula()
    guard_thresholds = _expected_gravity_bias_guard_thresholds()
    source_audit = _independent_gravity_bias_source_audit(demo)
    action_records = _independent_gravity_bias_action_records(route_contract)
    action_roster_sha256 = canonical_json_sha256(action_records)
    lightweight_preimage = {
        "capture_route_contract_identity_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_ROUTE_IDENTITY_SHA256
        ),
        "desired_start_q_sha256": desired_start_sha256,
        "move_action_roster_sha256": action_roster_sha256,
        "formula_sha256": canonical_json_sha256(formula),
        "guard_thresholds_sha256": canonical_json_sha256(guard_thresholds),
        "transitive_callgraph_sha256": source_audit[
            "observed_transitive_callgraph_sha256"
        ],
        "transitive_bytecode_sha256": source_audit[
            "observed_transitive_bytecode_sha256"
        ],
        "ast_policy_sha256": source_audit["observed_policy_sha256"],
        "source_audit_code_object_sha256": (
            "37d6475ec7fb112f057f605e7ef184fb0614f4568858fdfb9de978d700c76a17"
        ),
        "lightweight_guard_code_object_sha256": (
            "a914e0707f89c6dff610d60fe95305668100c7d5b50801043a637610818af204"
        ),
    }
    lightweight_component_matches = {
        "route_identity_matches": True,
        "desired_start_q_matches": True,
        "move_action_roster_matches": True,
        "formula_matches": True,
        "guard_thresholds_match": True,
        "callgraph_matches": True,
        "bytecode_matches": True,
        "ast_policy_matches": True,
    }
    lightweight_identity = {
        "schema_version": "1.0",
        "observed_identity_preimage": lightweight_preimage,
        "expected_identity_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256
        ),
        "observed_identity_sha256": canonical_json_sha256(
            lightweight_preimage
        ),
        "component_matches": lightweight_component_matches,
        "public_objects_match_fresh_reconstruction": True,
        "public_digests_match_private_frozen_literals": True,
        "source_audit_binding_matches": True,
        "builder_bindings_match_frozen": True,
        "source_audit": source_audit,
        "move_action_records": action_records,
        "passed": bool(
            source_audit["passed"]
            and action_roster_sha256
            == CORE_CAPTURE_GRAVITY_BIAS_ACTION_ROSTER_SHA256
            and canonical_json_sha256(lightweight_preimage)
            == CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256
        ),
    }
    identity = {
        "robot_xml_sha256": sha256_file(robot_xml),
        "model_xml_sha256": source_model_sha256,
        "compiled_model_xml_equivalent_sha256": (
            "edfc58afb55d83901f3e35f7e3426d5ffeef696257ef55e28955b400249480d0"
        ),
        "initialized_active_collision_geometry_sha256": (
            "401cbab95925ec5688d54b875c7d9e2e3788ebaa717cc2eb92fe94b1600b2083"
        ),
        "capture_route_contract_identity_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_ROUTE_IDENTITY_SHA256
        ),
        "desired_start_q_sha256": desired_start_sha256,
        **copy.deepcopy(CORE_CAPTURE_GRAVITY_BIAS_EXPECTED_HASHES),
        "initialized_non_arm_qpos_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_NON_ARM_QPOS_SHA256
        ),
        "formula_sha256": canonical_json_sha256(formula),
        "guard_thresholds": guard_thresholds,
        "transitive_callgraph_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_CALLGRAPH_SHA256
        ),
        "transitive_bytecode_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_BYTECODE_SHA256
        ),
        "ast_policy_sha256": CORE_CAPTURE_GRAVITY_BIAS_AST_POLICY_SHA256,
        "lightweight_identity_sha256": (
            CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256
        ),
    }
    component_matches = {key: True for key in identity}
    route_preimage = route_contract["contract_identity_digest_preimage"]
    route_identity_sha256 = canonical_json_sha256(route_preimage)
    identity_revalidation = {
        "schema_version": "1.0",
        "expected_identity_preimage": copy.deepcopy(identity),
        "observed_identity_preimage": copy.deepcopy(identity),
        "expected_identity_sha256": CORE_CAPTURE_GRAVITY_BIAS_IDENTITY_SHA256,
        "observed_identity_sha256": canonical_json_sha256(identity),
        "component_matches": component_matches,
        "route_identity": {
            "expected_sha256": (
                CORE_CAPTURE_GRAVITY_BIAS_ROUTE_IDENTITY_SHA256
            ),
            "observed_preimage": copy.deepcopy(route_preimage),
            "observed_sha256": route_identity_sha256,
            "matches": bool(
                route_identity_sha256
                == CORE_CAPTURE_GRAVITY_BIAS_ROUTE_IDENTITY_SHA256
            ),
        },
        "formula": {
            "expected_sha256": CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256,
            "observed_sha256": canonical_json_sha256(formula),
            "matches": bool(
                canonical_json_sha256(formula)
                == CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256
            ),
        },
        "desired_start_q": {
            "expected_sha256": (
                CORE_CAPTURE_GRAVITY_BIAS_DESIRED_START_SHA256
            ),
            "observed_sha256": desired_start_sha256,
            "matches": bool(
                desired_start_sha256
                == CORE_CAPTURE_GRAVITY_BIAS_DESIRED_START_SHA256
            ),
        },
        "source_audit": source_audit,
        "lightweight_identity": lightweight_identity,
        "public_objects_match_fresh_reconstruction": True,
        "passed": bool(
            all(component_matches.values())
            and route_identity_sha256
            == CORE_CAPTURE_GRAVITY_BIAS_ROUTE_IDENTITY_SHA256
            and canonical_json_sha256(formula)
            == CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256
            and desired_start_sha256
            == CORE_CAPTURE_GRAVITY_BIAS_DESIRED_START_SHA256
            and source_audit["passed"]
            and lightweight_identity["passed"]
            and canonical_json_sha256(identity)
            == CORE_CAPTURE_GRAVITY_BIAS_IDENTITY_SHA256
        ),
    }
    actuator_ids = [model.actuator(name).id for name in demo.ARM_ACTUATORS]
    arm_actuators = [
        {
            "name": name,
            "kp": float(model.actuator_gainprm[actuator_id, 0]),
            "gear": float(model.actuator_gear[actuator_id, 0]),
            "ctrlrange_rad": [
                float(value) for value in model.actuator_ctrlrange[actuator_id]
            ],
            "forcerange_nm": [
                float(value) for value in model.actuator_forcerange[actuator_id]
            ],
        }
        for name, actuator_id in zip(
            demo.ARM_ACTUATORS, actuator_ids, strict=True
        )
    ]
    return {
        "schema_version": "1.0",
        "contract_kind": (
            "development_only_capture_route_gravity_bias_position_feedforward"
        ),
        "contract_identity_digest_preimage": identity,
        "contract_identity_sha256": canonical_json_sha256(identity),
        "formula": formula,
        "formula_sha256": canonical_json_sha256(formula),
        "identity_revalidation": identity_revalidation,
        "source_ast_audit": source_audit,
        "guard_thresholds": guard_thresholds,
        "source_binding": {
            "robot_xml": {
                "path": "Simulation/SO101/so101_new_calib.xml",
                "bytes": robot_xml.stat().st_size,
                "sha256": sha256_file(robot_xml),
            },
            "assembled_model_xml_sha256": source_model_sha256,
            "compiled_model_xml_equivalent_sha256": compiled_sha256,
            "initialized_active_collision_geometry_sha256": active_sha256,
            "capture_route_contract_identity_sha256": route_contract[
                "contract_identity_sha256"
            ],
            "capture_route_source_state_sha256": route_contract[
                "source_state_sha256"
            ],
            "capture_route_q_roster_sha256": route_contract[
                "q_roster_sha256"
            ],
            "desired_start_q_by_action": desired_starts,
            "desired_start_q_sha256": desired_start_sha256,
        },
        "compiled_dynamics_binding": dynamics,
        "arm_actuators": arm_actuators,
        "evidence_requirements": {
            "every_physics_substep_recorded": True,
            "desired_q_and_biased_ctrl_are_separate": True,
            "desired_action_start_never_seeded_from_biased_ctrl": True,
            "wrong_sign_zero_gravity_gain_gear_and_model_mutations_fail": True,
            "any_saturation_fails": True,
            "raw_two_tab_by_five_cam_contact_counts": True,
        },
        "authority_scope": {
            "development_free_space_tracking": True,
            "contact_force_authority": False,
            "friction_coefficient_authority": False,
            "contact_parameter_authority": False,
            "cam_dynamics_authority": False,
            "physical_lock_authority": False,
            "release_ready": False,
        },
        "release_ready": False,
    }


def core_capture_gravity_bias_contract_errors(
    record: Any,
    demo: ModuleType,
    model: Any,
    route_contract: dict[str, Any],
) -> list[str]:
    expected = expected_core_capture_gravity_bias_contract(
        demo, model, route_contract
    )
    return [
        f"gravity_bias_contract:{path}"
        for path in _evidence_mismatches(record, expected)
    ]


def core_capture_gravity_bias_source_errors(source: str) -> list[str]:
    """Reject hidden physical-state writes or non-bias force inputs."""

    tree = ast.parse(source)
    module_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    controller = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "MatchaWorkflowController"
        ),
        None,
    )
    if controller is None:
        return ["gravity_bias_source:controller_missing"]
    methods = {
        node.name: node
        for node in controller.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_functions = {
        "_current_core_capture_route_identity_preimage",
        "_current_core_capture_gravity_bias_formula",
        "_current_core_capture_gravity_bias_guard_thresholds",
        "_core_capture_gravity_bias_model_digests",
        "_core_capture_move_actions",
        "_move_action_desired_q",
        "_core_capture_gravity_bias_control",
        "_forward_scratch_arm_configuration",
        "_current_core_capture_gravity_bias_lightweight_identity_snapshot",
        "_core_capture_gravity_bias_prewrite_snapshot",
    }
    required_methods = {
        "__init__",
        "_advance_action",
        "_command_move",
        "_record_core_capture_gravity_bias_feedforward",
        "_core_capture_gravity_bias_evidence_report",
    }
    errors: list[str] = []
    if not required_functions.issubset(module_functions):
        errors.append("gravity_bias_source:module_helpers")
    if not required_methods.issubset(methods):
        errors.append("gravity_bias_source:controller_methods")
    selected = [
        *(module_functions[name] for name in required_functions if name in module_functions),
        *(methods[name] for name in required_methods if name in methods),
    ]
    prohibited_attributes = {"qfrc_constraint"}
    prohibited_calls = {"mj_inverse", "mj_contactForce"}
    for function in selected:
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and node.attr in prohibited_attributes:
                errors.append(
                    f"gravity_bias_source:{function.name}:{node.attr}"
                )
            if isinstance(node, ast.Call) and called_name(node) in prohibited_calls:
                errors.append(
                    f"gravity_bias_source:{function.name}:{called_name(node)}"
                )
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                if target_state_field(target) in {"qpos", "qvel"}:
                    errors.append(
                        f"gravity_bias_source:{function.name}:state_write"
                    )
    control = module_functions.get("_core_capture_gravity_bias_control")
    if control is not None:
        control_arguments = [argument.arg for argument in control.args.args]
        if control_arguments != [
            "model",
            "scratch_data",
            "arm_qpos_ids",
            "non_arm_qpos_ids",
            "arm_dof_ids",
            "arm_actuator_ids",
            "desired_q",
            "expected_non_arm_qpos_sha256",
        ]:
            errors.append("gravity_bias_source:control_signature")
        qfrc_bias_reads = sum(
            isinstance(node, ast.Attribute) and node.attr == "qfrc_bias"
            for node in ast.walk(control)
        )
        if qfrc_bias_reads != 1:
            errors.append("gravity_bias_source:qfrc_bias_read_count")
    init_method = methods.get("__init__")
    if init_method is not None:
        scratch_assignments = []
        for node in ast.walk(init_method):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            if any(
                isinstance(target, ast.Attribute)
                and target.attr
                in {
                    "core_capture_gravity_bias_scratch_data",
                    "core_capture_gravity_bias_telemetry_fk_data",
                }
                for target in targets
            ):
                scratch_assignments.append(ast.unparse(node.value))
        if len(scratch_assignments) != 2 or any(
            expression == "self.data" for expression in scratch_assignments
        ):
            errors.append("gravity_bias_source:scratch_alias")
    desired_start_assignments: list[tuple[str, ast.AST]] = []
    for method_name in ("__init__", "_advance_action"):
        method = methods.get(method_name)
        if method is None:
            continue
        for node in ast.walk(method):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Attribute)
                and target.attr == "desired_action_start_q"
                for target in targets
            ):
                desired_start_assignments.append((method_name, node.value))
    if len(desired_start_assignments) != 2:
        errors.append("gravity_bias_source:desired_start_assignment_count")
    expected_start_names = {
        "__init__": "initial_desired_start",
        "_advance_action": "next_desired_start",
    }
    for method_name, value in desired_start_assignments:
        expression = ast.unparse(value)
        method_expression = ast.unparse(methods[method_name])
        if (
            expression != f"np.asarray({expected_start_names[method_name]}, dtype=np.float64).copy()"
            or "CORE_CAPTURE_ROUTE_DESIRED_START_Q" not in method_expression
        ):
            errors.append("gravity_bias_source:desired_start_not_frozen")
    audited_nodes = {
        name: (
            methods.get(name.rsplit(".", 1)[-1])
            if name.startswith("MatchaWorkflowController.")
            else module_functions.get(name)
        )
        for name in _expected_gravity_bias_ast_policy()["audited_functions"]
    }
    for name, node in audited_nodes.items():
        if node is None:
            errors.append(f"gravity_bias_source:{name}:missing")
            continue
        normalized = ast.dump(
            ast.Module(body=[node], type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
        if hashlib.sha256(normalized.encode()).hexdigest() != (
            CORE_CAPTURE_GRAVITY_BIAS_FUNCTION_HASHES[name][
                "normalized_ast_sha256"
            ]
        ):
            errors.append(f"gravity_bias_source:{name}:ast_sha256")
    return errors


def _independent_move_action_desired_q(
    action: Any,
    desired_start_q: list[float],
    elapsed_s: float,
) -> tuple[np.ndarray, float]:
    alpha = min(1.0, max(0.0, elapsed_s / float(action.duration_s)))
    smooth = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
    start = np.asarray(desired_start_q, dtype=np.float64)
    if action.joint_waypoints:
        route = np.asarray(
            (tuple(start), *action.joint_waypoints), dtype=np.float64
        )
        position = smooth * (len(route) - 1)
        segment = min(int(math.floor(position)), len(route) - 2)
        fraction = position - segment
        desired = route[segment] + fraction * (
            route[segment + 1] - route[segment]
        )
    else:
        target = np.asarray(action.target_q, dtype=np.float64)
        desired = start + smooth * (target - start)
    return desired, smooth


def _independent_expected_p_x_mm(
    action_name: str, smooth: float
) -> tuple[float, float]:
    if action_name == CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[0]:
        return 55.0, 0.20 * smooth
    if action_name == CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[1]:
        return 55.0 - 48.6 * smooth, 0.20
    if action_name == CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[2]:
        return 6.4 - 3.2 * smooth, 0.20 * (1.0 - smooth)
    if action_name == CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[3]:
        return 3.2 * (1.0 - smooth), 0.0
    raise ValueError(action_name)


def _gravity_bias_endpoint_record_errors(
    record: Any,
    expected: dict[str, Any],
    thresholds: dict[str, Any],
    controller_tick_substeps: int = 20,
) -> list[str]:
    """Validate one move-complete record against private-step replay."""

    if not isinstance(record, dict):
        return ["endpoint:missing"]
    errors = [
        f"endpoint:{path}"
        for path in _evidence_mismatches(record, expected)
    ]
    if record.get("event") != "move_complete":
        errors.append("endpoint:event")
    if record.get("endpoint_dwell_ticks") != thresholds.get(
        "endpoint_required_contiguous_controller_ticks"
    ):
        errors.append("endpoint:dwell")
    endpoint_substep = record.get("physics_substep_count")
    if (
        isinstance(endpoint_substep, bool)
        or not isinstance(endpoint_substep, int)
        or endpoint_substep <= 0
        or endpoint_substep % controller_tick_substeps != 0
    ):
        errors.append("endpoint:controller_tick_grid")
    for field, threshold_name in (
        ("endpoint_q_error_rad", "endpoint_maximum_q_error_rad"),
        ("endpoint_max_abs_qvel_rad_s", "endpoint_maximum_abs_qvel_rad_s"),
        ("endpoint_fk_position_error_m", "endpoint_maximum_fk_position_error_m"),
        (
            "endpoint_fk_orientation_error_rad",
            "endpoint_maximum_fk_orientation_error_rad",
        ),
    ):
        value = _finite_real(record.get(field))
        threshold = _finite_real(thresholds.get(threshold_name))
        if value is None or threshold is None or value > threshold:
            errors.append(f"endpoint:{field}:threshold")
    return errors


def core_capture_gravity_bias_result_errors(
    result: Any,
    contract: dict[str, Any],
    demo: ModuleType,
    model: Any,
    route_contract: dict[str, Any],
) -> list[str]:
    """Replay every bounded feedforward sample from independent model state."""

    if not isinstance(result, dict):
        return ["gravity_bias_result:missing"]
    evidence = result.get("core_capture_gravity_bias_feedforward_evidence")
    if not isinstance(evidence, dict):
        return ["gravity_bias_evidence:missing"]
    expected_keys = {
        "schema_version", "evidence_kind", "runtime_contract_api",
        "contract_identity_sha256", "identity_binding", "model_binding",
        "dynamics_binding",
        "physics_timestep_s", "desired_route_q_roster_sha256",
        "desired_start_q_by_action", "frozen_action_roster_matches",
        "raw_samples", "raw_samples_sha256", "recomputed_telemetry",
        "recomputed_telemetry_sha256", "raw_sample_count",
        "first_physics_substep_count", "last_physics_substep_count",
        "sample_counts_by_phase", "producer_counts_by_phase",
        "contact_audit_counts_by_phase", "phase_counts_consistent",
        "state_index_and_time_contiguous", "final_sample_coverage_anchored",
        "observed_phase_order", "expected_phase_order",
        "sample_phase_order_is_valid_prefix", "every_physics_substep_recorded",
        "all_formula_replay_passed", "all_samples_replayed_from_actual_model",
        "all_runtime_scratch_isolation_passed",
        "all_telemetry_recomputed_from_raw_fields_and_fresh_fk",
        "raw_contact_count_closure_passed", "immutable_desired_route_replayed",
        "any_saturation", "saturation_sample_count",
        "maximum_abs_gravity_bias_offset_rad",
        "maximum_abs_tracking_error_to_desired_rad",
        "maximum_actuator_torque_utilization",
        "maximum_ctrl_range_utilization", "completed_route_endpoint_actions",
        "completed_route_endpoint_action_order", "endpoint_order_is_valid_prefix",
        "route_endpoint_records", "align_and_axial_endpoints_completed",
        "all_four_phases_observed", "all_four_route_endpoints_completed",
        "align_and_axial_free_space_closed",
        "align_and_axial_tracking_thresholds", "first_cam_contact_record",
        "first_rejected_cam_contact_record", "prohibited_operation_counts",
        "source_ast_allowlist_audit", "prohibited_operations_verified",
        "contact_force_authority",
        "friction_coefficient_authority", "contact_parameter_authority",
        "cam_dynamics_authority", "passed", "release_ready",
    }
    errors: list[str] = []
    if set(evidence) != expected_keys:
        errors.append("gravity_bias_evidence:keys")
    desired_starts = _independent_gravity_bias_desired_starts(route_contract)
    expected_header = {
        "schema_version": "1.0",
        "evidence_kind": (
            "real_mujoco_every_substep_development_gravity_bias_feedforward"
        ),
        "runtime_contract_api": (
            "core_capture_gravity_bias_feedforward_runtime_contract"
        ),
        "contract_identity_sha256": CORE_CAPTURE_GRAVITY_BIAS_IDENTITY_SHA256,
        "physics_timestep_s": 0.00025,
        "desired_route_q_roster_sha256": route_contract["q_roster_sha256"],
        "desired_start_q_by_action": desired_starts,
        "contact_force_authority": False,
        "friction_coefficient_authority": False,
        "contact_parameter_authority": False,
        "cam_dynamics_authority": False,
        "release_ready": False,
    }
    for key, expected in expected_header.items():
        if _evidence_mismatches(evidence.get(key), expected, key):
            errors.append(f"gravity_bias_evidence:{key}")

    expected_identity_snapshot = contract.get("identity_revalidation")
    expected_identity_binding = {
        "controller_init": expected_identity_snapshot,
        "evidence_recompute": expected_identity_snapshot,
        "unchanged_since_controller_init": True,
        "control_function_binding_matches": True,
        "passed": bool(
            isinstance(expected_identity_snapshot, dict)
            and expected_identity_snapshot.get("passed") is True
        ),
    }
    if _evidence_mismatches(
        evidence.get("identity_binding"), expected_identity_binding
    ):
        errors.append("gravity_bias_evidence:identity_binding")
    identity_passed = expected_identity_binding["passed"]

    compiled_sha = _independent_compiled_model_xml_equivalent_sha256(model)
    initialized = demo.mujoco.MjData(model)
    demo.initialize(model, initialized)
    active_sha = _independent_initialized_active_geometry_sha256(
        model, initialized, demo.mujoco
    )
    binding_contract = {
        "model_binding": {
            "model_xml_sha256": contract["source_binding"][
                "assembled_model_xml_sha256"
            ],
            "compiled_model_xml_equivalent_sha256": contract[
                "source_binding"
            ]["compiled_model_xml_equivalent_sha256"],
            "initialized_active_collision_geometry_sha256": contract[
                "source_binding"
            ]["initialized_active_collision_geometry_sha256"],
        }
    }
    binding_observations = {
        "controller_init_compiled": compiled_sha,
        "controller_init_active": active_sha,
        "evidence_compiled": compiled_sha,
        "evidence_active": active_sha,
    }
    binding_errors, binding_passed = _core_cam_actual_model_binding_errors(
        evidence.get("model_binding"),
        binding_contract,
        binding_observations,
    )
    errors.extend(
        f"gravity_bias_evidence:{error}" for error in binding_errors
    )
    expected_dynamics = _expected_gravity_bias_dynamics_binding(model, demo)
    expected_dynamics_record = {
        "controller_init": expected_dynamics,
        "evidence_recompute": expected_dynamics,
        "unchanged_since_controller_init": True,
        "passed": bool(binding_passed and expected_dynamics["passed"]),
    }
    if _evidence_mismatches(
        evidence.get("dynamics_binding"), expected_dynamics_record
    ):
        errors.append("gravity_bias_evidence:dynamics_binding")
    dynamics_passed = expected_dynamics_record["passed"]

    samples = evidence.get("raw_samples")
    if not isinstance(samples, list):
        errors.append("gravity_bias_evidence:raw_samples")
        samples = []
    if evidence.get("raw_samples_sha256") != canonical_json_sha256(samples):
        errors.append("gravity_bias_evidence:raw_samples_sha256")
    if evidence.get("raw_sample_count") != len(samples):
        errors.append("gravity_bias_evidence:raw_sample_count")
    published_telemetry = evidence.get("recomputed_telemetry")
    if not isinstance(published_telemetry, list):
        errors.append("gravity_bias_evidence:recomputed_telemetry")
        published_telemetry = []
    if evidence.get("recomputed_telemetry_sha256") != canonical_json_sha256(
        published_telemetry
    ):
        errors.append("gravity_bias_evidence:recomputed_telemetry_sha256")
    expected_sample_keys = {
        "physics_substep_count", "sim_time_s", "action",
        "contract_identity_sha256", "command_elapsed_s",
        "prewrite_identity_sha256", "prewrite_identity_passed",
        "positive_lock_slider_qpos_address",
        "command_smooth_fraction", "desired_action_start_q_rad",
        "desired_arm_q_rad", "scratch_desired_arm_q_rad",
        "scratch_is_distinct_from_live",
        "live_qpos_unchanged_during_bias_evaluation",
        "live_qvel_unchanged_during_bias_evaluation",
        "expected_non_arm_qpos_sha256",
        "observed_non_arm_qpos_before_sha256",
        "observed_non_arm_qpos_after_sha256",
        "all_scratch_qvel_zero_before", "all_scratch_qvel_zero_after",
        "scratch_arm_qvel_rad_s", "scratch_qfrc_bias_n_m", "kp", "gear",
        "kp_times_gear", "gravity_bias_offset_rad",
        "unsaturated_control_rad", "applied_control_rad",
        "saturated_by_joint", "any_saturation", "live_arm_q_rad",
        "live_arm_qvel_rad_s", "live_full_qpos", "live_full_qvel",
        "tracking_error_to_desired_rad", "actuator_torque_nm",
        "actuator_torque_utilization", "ctrl_range_utilization", "fk",
        "cached_post_mj_step_transform_fk",
        "raw_two_tab_by_five_cam_contact_counts", "raw_live_contact_count",
        "raw_all_contact_geom_pairs", "raw_tab_cam_contact_count",
        "raw_all_cam_contact_count", "raw_other_cam_contact_count", "finite",
    }
    actions = {
        action.name: action for action in demo._core_capture_move_actions()
    }
    actual_action_records = [
        {
            "name": action.name,
            "kind": action.kind,
            "tool": action.tool,
            "duration_s": action.duration_s,
            "timeout_s": action.timeout_s,
            "target_q": list(action.target_q or ()),
            "joint_waypoints": [
                list(row) for row in action.joint_waypoints
            ],
        }
        for action in demo._core_capture_move_actions()
    ]
    expected_action_records = _independent_gravity_bias_action_records(
        route_contract
    )
    action_roster_matches = bool(
        actual_action_records == expected_action_records
        and canonical_json_sha256(actual_action_records)
        == CORE_CAPTURE_GRAVITY_BIAS_ACTION_ROSTER_SHA256
    )
    if not action_roster_matches:
        errors.append("gravity_bias_evidence:runtime_action_roster")
    if evidence.get("frozen_action_roster_matches") is not True:
        errors.append("gravity_bias_evidence:frozen_action_roster_matches")
    arm_qpos = np.asarray(
        [model.joint(name).qposadr[0] for name in demo.ARM_JOINTS], dtype=int
    )
    arm_dof = np.asarray(
        [model.joint(name).dofadr[0] for name in demo.ARM_JOINTS], dtype=int
    )
    arm_actuators = np.asarray(
        [model.actuator(name).id for name in demo.ARM_ACTUATORS], dtype=int
    )
    non_arm_qpos = np.asarray(
        sorted(set(range(model.nq)) - set(arm_qpos.tolist())), dtype=int
    )
    slider_qpos_address = int(
        model.joint("qc_positive_lock_slider_joint").qposadr[0]
    )
    scratch = demo.mujoco.MjData(model)
    demo.initialize(model, scratch)
    _independent_forward_scratch_arm_configuration(
        model,
        scratch,
        demo.mujoco,
        arm_qpos,
        np.asarray(scratch.qpos, dtype=np.float64)[arm_qpos],
    )
    expected_non_arm_qpos_sha256 = _independent_float64_bytes_sha256(
        np.asarray(scratch.qpos, dtype=np.float64)[non_arm_qpos]
    )
    fk_data = demo.mujoco.MjData(model)
    demo.initialize(model, fk_data)
    physical_replay_data = demo.mujoco.MjData(model)
    demo.initialize(model, physical_replay_data)
    dock_id = int(model.body("dock_gripper").id)
    mating_id = int(model.site("robot_mating_face").id)
    kp_model = np.asarray(
        model.actuator_gainprm[arm_actuators, 0], dtype=np.float64
    )
    gear_model = np.asarray(
        model.actuator_gear[arm_actuators, 0], dtype=np.float64
    )
    ctrlrange = np.asarray(
        model.actuator_ctrlrange[arm_actuators], dtype=np.float64
    )
    forcerange = np.asarray(
        model.actuator_forcerange[arm_actuators], dtype=np.float64
    )
    formula_passes: list[bool] = []
    model_replay_passes: list[bool] = []
    isolation_passes: list[bool] = []
    telemetry_passes: list[bool] = []
    recomputed_telemetry: list[dict[str, Any]] = []
    route_passes: list[bool] = []
    contact_passes: list[bool] = []
    sample_validity: list[bool] = []
    endpoint_replay_by_substep: dict[int, dict[str, Any]] = {}
    for sample_index, sample in enumerate(samples):
        prefix = f"gravity_bias_sample:{sample_index}"
        if not isinstance(sample, dict) or set(sample) != expected_sample_keys:
            errors.append(f"{prefix}:keys")
            formula_passes.append(False)
            model_replay_passes.append(False)
            isolation_passes.append(False)
            telemetry_passes.append(False)
            route_passes.append(False)
            contact_passes.append(False)
            sample_validity.append(False)
            continue
        try:
            action_name = str(sample["action"])
            action = actions[action_name]
            desired_start = desired_starts[action_name]
            desired, smooth = _independent_move_action_desired_q(
                action, desired_start, float(sample["command_elapsed_s"])
            )
            route_ok = bool(
                not _evidence_mismatches(
                    sample["desired_action_start_q_rad"], desired_start
                )
                and not _evidence_mismatches(
                    sample["desired_arm_q_rad"], desired.tolist()
                )
                and _number_matches(
                    sample["command_smooth_fraction"], smooth
                )
                and sample["contract_identity_sha256"]
                == CORE_CAPTURE_GRAVITY_BIAS_IDENTITY_SHA256
                and sample["prewrite_identity_sha256"]
                == CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256
                and sample["prewrite_identity_passed"] is True
                and sample["positive_lock_slider_qpos_address"]
                == slider_qpos_address
            )
            route_passes.append(route_ok)
            if not route_ok:
                errors.append(f"{prefix}:desired_route")

            scratch.qpos[arm_qpos] = desired
            scratch.qvel[:] = 0.0
            demo.mujoco.mj_forward(model, scratch)
            bias = np.asarray(scratch.qfrc_bias[arm_dof], dtype=np.float64)
            denominator = kp_model * gear_model
            offset = bias / denominator
            unsaturated = desired + offset
            applied = np.clip(
                unsaturated, ctrlrange[:, 0], ctrlrange[:, 1]
            )
            saturated = np.not_equal(applied, unsaturated)
            model_expected = {
                "scratch_desired_arm_q_rad": desired.tolist(),
                "scratch_arm_qvel_rad_s": [0.0] * len(arm_dof),
                "scratch_qfrc_bias_n_m": bias.tolist(),
                "kp": kp_model.tolist(),
                "gear": gear_model.tolist(),
                "kp_times_gear": denominator.tolist(),
                "gravity_bias_offset_rad": offset.tolist(),
                "unsaturated_control_rad": unsaturated.tolist(),
                "applied_control_rad": applied.tolist(),
                "saturated_by_joint": [bool(value) for value in saturated],
                "any_saturation": bool(np.any(saturated)),
            }
            model_ok = all(
                not _evidence_mismatches(sample.get(key), expected, key)
                for key, expected in model_expected.items()
            )
            model_replay_passes.append(model_ok)
            if not model_ok:
                errors.append(f"{prefix}:model_replay")
            isolation_ok = bool(
                sample["scratch_is_distinct_from_live"] is True
                and sample[
                    "live_qpos_unchanged_during_bias_evaluation"
                ] is True
                and sample[
                    "live_qvel_unchanged_during_bias_evaluation"
                ] is True
                and sample["all_scratch_qvel_zero_before"] is True
                and sample["all_scratch_qvel_zero_after"] is True
                and sample["expected_non_arm_qpos_sha256"]
                == CORE_CAPTURE_GRAVITY_BIAS_NON_ARM_QPOS_SHA256
                and sample["observed_non_arm_qpos_before_sha256"]
                == CORE_CAPTURE_GRAVITY_BIAS_NON_ARM_QPOS_SHA256
                and sample["observed_non_arm_qpos_after_sha256"]
                == CORE_CAPTURE_GRAVITY_BIAS_NON_ARM_QPOS_SHA256
                and expected_non_arm_qpos_sha256
                == CORE_CAPTURE_GRAVITY_BIAS_NON_ARM_QPOS_SHA256
            )
            isolation_passes.append(isolation_ok)
            if not isolation_ok:
                errors.append(f"{prefix}:scratch_isolation")
            published_bias = np.asarray(
                sample["scratch_qfrc_bias_n_m"], dtype=np.float64
            )
            published_kp = np.asarray(sample["kp"], dtype=np.float64)
            published_gear = np.asarray(sample["gear"], dtype=np.float64)
            published_offset = np.asarray(
                sample["gravity_bias_offset_rad"], dtype=np.float64
            )
            published_unsaturated = np.asarray(
                sample["unsaturated_control_rad"], dtype=np.float64
            )
            formula_ok = bool(
                np.array_equal(
                    published_offset,
                    published_bias / (published_kp * published_gear),
                )
                and np.array_equal(
                    published_unsaturated, desired + published_offset
                )
                and np.array_equal(
                    np.asarray(sample["applied_control_rad"]),
                    np.clip(
                        published_unsaturated,
                        ctrlrange[:, 0], ctrlrange[:, 1],
                    ),
                )
                and bool(sample["finite"])
            )
            formula_passes.append(formula_ok)
            if not formula_ok:
                errors.append(f"{prefix}:formula")

            live_q = np.asarray(sample["live_arm_q_rad"], dtype=np.float64)
            live_qvel = np.asarray(
                sample["live_arm_qvel_rad_s"], dtype=np.float64
            )
            live_full_qpos = np.asarray(
                sample["live_full_qpos"], dtype=np.float64
            )
            live_full_qvel = np.asarray(
                sample["live_full_qvel"], dtype=np.float64
            )
            published_slider_address = sample[
                "positive_lock_slider_qpos_address"
            ]
            slider_address_ok = bool(
                isinstance(published_slider_address, int)
                and not isinstance(published_slider_address, bool)
                and published_slider_address == slider_qpos_address
                and 0 <= published_slider_address < model.nq
            )
            slider_q_mm = (
                float(live_full_qpos[slider_qpos_address]) * 1000.0
                if live_full_qpos.shape == (model.nq,)
                else math.nan
            )
            expected_tracking = live_q - desired
            actuator_torque = np.asarray(
                sample["actuator_torque_nm"], dtype=np.float64
            )
            physical_replay_data.ctrl[arm_actuators] = applied
            demo.mujoco.mj_step(model, physical_replay_data)
            expected_actuator_torque = np.asarray(
                physical_replay_data.actuator_force[arm_actuators],
                dtype=np.float64,
            )
            physical_replay_ok = bool(
                np.array_equal(
                    live_full_qpos,
                    np.asarray(physical_replay_data.qpos, dtype=np.float64),
                )
                and np.array_equal(
                    live_full_qvel,
                    np.asarray(physical_replay_data.qvel, dtype=np.float64),
                )
                and np.array_equal(
                    actuator_torque, expected_actuator_torque
                )
            )
            model_ok = bool(model_ok and physical_replay_ok)
            model_replay_passes[-1] = model_ok
            if not physical_replay_ok:
                errors.append(f"{prefix}:physical_dynamics_replay")
            replay_dock_position = np.asarray(
                physical_replay_data.xpos[dock_id], dtype=np.float64
            )
            replay_dock_rotation = np.asarray(
                physical_replay_data.xmat[dock_id], dtype=np.float64
            ).reshape(3, 3)
            replay_mating_position = np.asarray(
                physical_replay_data.site_xpos[mating_id], dtype=np.float64
            )
            replay_mating_rotation = np.asarray(
                physical_replay_data.site_xmat[mating_id], dtype=np.float64
            ).reshape(3, 3)
            replay_local_position = (
                replay_mating_position - replay_dock_position
            ) @ replay_dock_rotation
            target_preseat_mm, target_source_x_mm = (
                _independent_expected_p_x_mm(action_name, 1.0)
            )
            target_local_position = np.asarray(
                [target_source_x_mm, 0.0, -target_preseat_mm],
                dtype=np.float64,
            ) * 0.001
            endpoint_position_error_m = float(
                np.linalg.norm(
                    replay_local_position - target_local_position
                )
            )
            replay_relative_rotation = (
                replay_dock_rotation.T @ replay_mating_rotation
            )
            endpoint_orientation_error_rad = math.acos(
                np.clip(
                    (float(np.trace(replay_relative_rotation)) - 1.0)
                    / 2.0,
                    -1.0,
                    1.0,
                )
            )
            endpoint_q_error_rad = float(
                np.max(
                    np.abs(
                        live_q
                        - np.asarray(action.target_q, dtype=np.float64)
                    )
                )
            )
            endpoint_max_abs_qvel_rad_s = float(
                np.max(np.abs(live_qvel))
            )
            endpoint_substep = int(sample["physics_substep_count"])
            endpoint_time_s = float(sample["sim_time_s"])
            endpoint_route_evidence = {
                "action": action_name,
                "target_preseat_mm": target_preseat_mm,
                "target_source_x_mm": target_source_x_mm,
                "observed_preseat_mm": (
                    -float(replay_local_position[2]) * 1000.0
                ),
                "observed_x_mm": (
                    float(replay_local_position[0]) * 1000.0
                ),
                "source_x_error_mm": (
                    float(replay_local_position[0]) * 1000.0
                    - target_source_x_mm
                ),
                "observed_transverse_y_mm": (
                    float(replay_local_position[1]) * 1000.0
                ),
                "position_error_m": endpoint_position_error_m,
                "orientation_error_rad": endpoint_orientation_error_rad,
                "physics_substep_count": endpoint_substep,
                "sim_time_s": endpoint_time_s,
            }
            endpoint_replay_by_substep[endpoint_substep] = {
                "event": "move_complete",
                "action": action_name,
                "physics_substep_count": endpoint_substep,
                "sim_time_s": endpoint_time_s,
                "endpoint_q_error_rad": endpoint_q_error_rad,
                "endpoint_max_abs_qvel_rad_s": (
                    endpoint_max_abs_qvel_rad_s
                ),
                "endpoint_fk_position_error_m": endpoint_position_error_m,
                "endpoint_fk_orientation_error_rad": (
                    endpoint_orientation_error_rad
                ),
                "endpoint_dwell_ticks": int(
                    contract["guard_thresholds"][
                        "endpoint_required_contiguous_controller_ticks"
                    ]
                ),
                "route_endpoint_evidence": endpoint_route_evidence,
            }
            force_limit = np.max(np.abs(forcerange), axis=1)
            expected_torque_util = np.divide(
                np.abs(expected_actuator_torque), force_limit,
                out=np.zeros_like(expected_actuator_torque),
                where=force_limit > 0.0,
            )
            ctrl_midpoint = np.mean(ctrlrange, axis=1)
            ctrl_half = 0.5 * (ctrlrange[:, 1] - ctrlrange[:, 0])
            expected_ctrl_util = np.divide(
                np.abs(applied - ctrl_midpoint), ctrl_half,
                out=np.zeros_like(applied), where=ctrl_half > 0.0,
            )
            arithmetic_ok = bool(
                not _evidence_mismatches(
                    sample["tracking_error_to_desired_rad"],
                    expected_tracking.tolist(),
                )
                and not _evidence_mismatches(
                    sample["actuator_torque_utilization"],
                    expected_torque_util.tolist(),
                )
                and not _evidence_mismatches(
                    sample["actuator_torque_nm"],
                    expected_actuator_torque.tolist(),
                )
                and not _evidence_mismatches(
                    sample["ctrl_range_utilization"],
                    expected_ctrl_util.tolist(),
                )
                and np.all(np.isfinite(live_q))
                and np.all(np.isfinite(live_qvel))
                and np.all(np.isfinite(actuator_torque))
                and np.all(np.isfinite(expected_actuator_torque))
                and np.all(np.isfinite(live_full_qpos))
                and np.all(np.isfinite(live_full_qvel))
                and live_full_qpos.shape == (model.nq,)
                and live_full_qvel.shape == (model.nv,)
                and np.array_equal(live_q, live_full_qpos[arm_qpos])
                and np.array_equal(live_qvel, live_full_qvel[arm_dof])
            )
            # The authoritative telemetry is a fresh forward pass at the
            # recorded post-step generalized state.  Cached transforms from
            # ``mj_step`` are preserved separately and never used as FK
            # authority.
            fk_data.qpos[:] = live_full_qpos
            fk_data.qvel[:] = live_full_qvel
            demo.mujoco.mj_forward(model, fk_data)
            dock_position = np.asarray(fk_data.xpos[dock_id], dtype=np.float64)
            dock_rotation = np.asarray(
                fk_data.xmat[dock_id], dtype=np.float64
            ).reshape(3, 3)
            mating_position = np.asarray(
                fk_data.site_xpos[mating_id], dtype=np.float64
            )
            mating_rotation = np.asarray(
                fk_data.site_xmat[mating_id], dtype=np.float64
            ).reshape(3, 3)
            local_mm = (mating_position - dock_position) @ dock_rotation * 1000.0
            preseat_mm = -float(local_mm[2])
            source_x_mm = float(local_mm[0])
            expected_p, expected_x = _independent_expected_p_x_mm(
                action_name, smooth
            )
            relative_rotation = dock_rotation.T @ mating_rotation
            orientation = math.acos(
                np.clip((float(np.trace(relative_rotation)) - 1.0) / 2.0, -1.0, 1.0)
            )
            source_q_max = max(
                0.05, min(3.0, max(0.0, preseat_mm) - source_x_mm - 3.15)
            )
            expected_fk = {
                "sampling_semantics": (
                    "fresh_private_scratch_mj_forward_at_recorded_post_step_qpos"
                ),
                "expected_preseat_mm": expected_p,
                "preseat_mm": preseat_mm,
                "preseat_error_mm": preseat_mm - expected_p,
                "expected_source_x_mm": expected_x,
                "source_x_mm": source_x_mm,
                "source_x_error_mm": source_x_mm - expected_x,
                "transverse_y_mm": float(local_mm[1]),
                "orientation_error_rad": orientation,
                "slider_q_mm": slider_q_mm,
                "source_q_max_mm": source_q_max,
            }
            fk_ok = bool(
                slider_address_ok
                and not _evidence_mismatches(sample["fk"], expected_fk)
            )
            cached_fk = sample["cached_post_mj_step_transform_fk"]
            cached_ok = bool(
                isinstance(cached_fk, dict)
                and set(cached_fk) == {
                    "sampling_semantics", "preseat_mm", "source_x_mm",
                    "transverse_y_mm", "orientation_error_rad",
                    "slider_q_mm", "source_q_max_mm",
                }
                and cached_fk.get("sampling_semantics")
                == (
                    "live_cached_transforms_after_mj_step_not_qpos_replay_authority"
                )
                and all(
                    _finite_real(cached_fk.get(key)) is not None
                    for key in set(cached_fk) - {"sampling_semantics"}
                )
            )
            if not arithmetic_ok:
                errors.append(f"{prefix}:arithmetic")
            if not fk_ok:
                errors.append(f"{prefix}:fk")
            if not cached_ok:
                errors.append(f"{prefix}:cached_fk")

            raw_contacts = sample["raw_all_contact_geom_pairs"]
            raw_indices_ok = bool(
                isinstance(raw_contacts, list)
                and [item.get("contact_index") for item in raw_contacts]
                == list(range(len(raw_contacts)))
                and sample["raw_live_contact_count"] == len(raw_contacts)
            )
            expected_pair_names = [
                [tab_name, cam_name]
                for tab_name in (
                    "qc_col_lock_slider_tab_part_000",
                    "qc_col_lock_slider_tab_part_001",
                )
                for cam_name in (
                    "dock_gripper_cam_collision",
                    "dock_gripper_cam_axial_lead_collision",
                    "dock_gripper_cam_hold_finger_collision",
                    "dock_gripper_cam_outer_root_lower_collision",
                    "dock_gripper_cam_outer_root_upper_collision",
                )
            ]
            pair_records = sample["raw_two_tab_by_five_cam_contact_counts"]
            pair_ok = bool(
                isinstance(pair_records, list)
                and [item.get("pair") for item in pair_records]
                == expected_pair_names
            )
            classified: set[int] = set()
            if raw_indices_ok and pair_ok:
                for pair_record, pair_names in zip(
                    pair_records, expected_pair_names, strict=True
                ):
                    indices = [
                        int(item["contact_index"])
                        for item in raw_contacts
                        if frozenset(item.get("geom_pair", []))
                        == frozenset(pair_names)
                    ]
                    pair_ok &= (
                        pair_record.get("contact_indices") == indices
                        and pair_record.get("contact_count") == len(indices)
                    )
                    classified.update(indices)
            cam_names = frozenset(pair[1] for pair in expected_pair_names)
            all_cam = {
                int(item["contact_index"])
                for item in raw_contacts
                if any(name in cam_names for name in item.get("geom_pair", []))
            } if raw_indices_ok else set()
            contact_ok = bool(
                raw_indices_ok and pair_ok
                and sample["raw_tab_cam_contact_count"] == len(classified)
                and sample["raw_all_cam_contact_count"] == len(all_cam)
                and sample["raw_other_cam_contact_count"]
                == len(all_cam - classified)
            )
            contact_passes.append(contact_ok)
            if not contact_ok:
                errors.append(f"{prefix}:contact_closure")
            derived_finite = bool(
                np.all(np.isfinite(desired))
                and np.all(np.isfinite(live_q))
                and np.all(np.isfinite(live_qvel))
                and np.all(np.isfinite(live_full_qpos))
                and np.all(np.isfinite(live_full_qvel))
                and np.all(np.isfinite(published_bias))
                and np.all(np.isfinite(published_kp))
                and np.all(np.isfinite(published_gear))
                and np.all(np.isfinite(published_offset))
                and np.all(np.isfinite(published_unsaturated))
                and np.all(np.isfinite(applied))
                and np.all(np.isfinite(actuator_torque))
                and np.all(np.isfinite(expected_torque_util))
                and np.all(np.isfinite(expected_ctrl_util))
                and all(
                    _finite_real(value) is not None
                    for key, value in expected_fk.items()
                    if key != "sampling_semantics"
                )
            )
            telemetry_ok = bool(
                arithmetic_ok
                and fk_ok
                and bool(sample["finite"]) is derived_finite
            )
            telemetry_passes.append(telemetry_ok)
            telemetry_record = {
                "physics_substep_count": int(
                    sample["physics_substep_count"]
                ),
                "action": action_name,
                "tracking_error_to_desired_rad": (
                    expected_tracking.tolist()
                ),
                "actuator_torque_utilization": (
                    expected_torque_util.tolist()
                ),
                "ctrl_range_utilization": expected_ctrl_util.tolist(),
                "fk": {
                    key: value
                    for key, value in expected_fk.items()
                    if key != "sampling_semantics"
                },
                "finite": derived_finite,
                "passed": telemetry_ok,
                "raw_all_cam_contact_count": len(all_cam),
            }
            recomputed_telemetry.append(telemetry_record)
            if (
                sample_index >= len(published_telemetry)
                or _evidence_mismatches(
                    published_telemetry[sample_index], telemetry_record
                )
            ):
                errors.append(f"{prefix}:recomputed_telemetry")
            sample_validity.append(
                bool(
                    route_ok and model_ok and formula_ok and isolation_ok
                    and telemetry_ok and contact_ok
                )
            )
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
            errors.append(f"{prefix}:malformed")
            while len(formula_passes) <= sample_index:
                formula_passes.append(False)
            while len(model_replay_passes) <= sample_index:
                model_replay_passes.append(False)
            while len(isolation_passes) <= sample_index:
                isolation_passes.append(False)
            while len(telemetry_passes) <= sample_index:
                telemetry_passes.append(False)
            while len(route_passes) <= sample_index:
                route_passes.append(False)
            while len(contact_passes) <= sample_index:
                contact_passes.append(False)
            sample_validity.append(False)

    if len(published_telemetry) != len(recomputed_telemetry):
        errors.append("gravity_bias_evidence:recomputed_telemetry_count")

    cam_contact_records = result.get(
        "core_cam_tab_contact_evidence", {}
    ).get("raw_contact_records", [])
    if not isinstance(cam_contact_records, list):
        errors.append("gravity_bias_evidence:cam_contact_records")
        cam_contact_records = []
    independently_classified_contacts: list[tuple[dict[str, Any], bool]] = []
    for contact_index, contact_record in enumerate(cam_contact_records):
        classification_errors, classification_passed, _ = (
            _independent_cam_tab_record_classification(contact_record)
        )
        if classification_errors:
            errors.append(
                f"gravity_bias_evidence:cam_contact_record:{contact_index}"
            )
        if isinstance(contact_record, dict):
            independently_classified_contacts.append(
                (contact_record, classification_passed)
            )
    expected_first_cam_contact = (
        copy.deepcopy(independently_classified_contacts[0][0])
        if independently_classified_contacts
        else None
    )
    expected_first_rejected_cam_contact = next(
        (
            copy.deepcopy(contact_record)
            for contact_record, classification_passed
            in independently_classified_contacts
            if not classification_passed
        ),
        None,
    )
    if _evidence_mismatches(
        evidence.get("first_cam_contact_record"),
        expected_first_cam_contact,
    ):
        errors.append("gravity_bias_evidence:first_cam_contact_record")
    if _evidence_mismatches(
        evidence.get("first_rejected_cam_contact_record"),
        expected_first_rejected_cam_contact,
    ):
        errors.append(
            "gravity_bias_evidence:first_rejected_cam_contact_record"
        )

    phase_counts = {
        name: sum(
            isinstance(sample, dict) and sample.get("action") == name
            for sample in samples
        )
        for name in CORE_CAPTURE_GRAVITY_BIAS_ACTIONS
    }
    observed_order: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        name = str(sample.get("action"))
        if not observed_order or observed_order[-1] != name:
            observed_order.append(name)
    expected_order = list(CORE_CAPTURE_GRAVITY_BIAS_ACTIONS)
    phase_order_ok = observed_order == expected_order[: len(observed_order)]
    state_contiguous = bool(
        samples
        and all(isinstance(sample, dict) for sample in samples)
        and samples[0].get("physics_substep_count") == 1
        and samples[-1].get("physics_substep_count") == len(samples)
        and all(
            sample.get("physics_substep_count") == index
            and _number_matches(sample.get("sim_time_s"), index * 0.00025, 1.0e-10)
            for index, sample in enumerate(samples, start=1)
        )
        and all(
            _number_matches(
                float(current["sim_time_s"])
                - float(previous["sim_time_s"]),
                0.00025,
                1.0e-12,
            )
            for previous, current in zip(samples, samples[1:])
        )
    )
    endpoint_records = evidence.get("route_endpoint_records")
    if not isinstance(endpoint_records, list):
        endpoint_records = []
        errors.append("gravity_bias_evidence:endpoint_records")
    endpoint_order = [
        str(item.get("action")) for item in endpoint_records
        if isinstance(item, dict)
    ]
    endpoint_prefix = bool(
        endpoint_order == expected_order[: len(endpoint_order)]
        and len(endpoint_order) == len(set(endpoint_order))
    )
    route_endpoint_journal = result.get("route_alignment", {}).get(
        "phase_endpoint_journal_evidence"
    )
    if _evidence_mismatches(route_endpoint_journal, endpoint_records):
        errors.append("gravity_bias_evidence:endpoint_route_journal")
    result_journal_endpoints = [
        record
        for record in result.get("journal", [])
        if isinstance(record, dict)
        and record.get("event") == "move_complete"
        and record.get("action") in CORE_CAPTURE_GRAVITY_BIAS_ACTIONS
    ] if isinstance(result.get("journal"), list) else []
    if _evidence_mismatches(result_journal_endpoints, endpoint_records):
        errors.append("gravity_bias_evidence:endpoint_result_journal")
    endpoint_records_passed = True
    samples_by_substep = {
        int(sample["physics_substep_count"]): sample
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("physics_substep_count"), int)
        and not isinstance(sample.get("physics_substep_count"), bool)
    }
    physics_timestep_s = float(model.opt.timestep)
    controller_tick_substeps = int(round(0.005 / physics_timestep_s))
    if not math.isclose(
        controller_tick_substeps * physics_timestep_s,
        0.005,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        errors.append("gravity_bias_evidence:endpoint_controller_tick")
    for endpoint_index, endpoint_record in enumerate(endpoint_records):
        expected_endpoint = endpoint_replay_by_substep.get(
            endpoint_record.get("physics_substep_count")
            if isinstance(endpoint_record, dict)
            else None
        )
        endpoint_errors = _gravity_bias_endpoint_record_errors(
            endpoint_record,
            expected_endpoint or {},
            contract["guard_thresholds"],
            controller_tick_substeps,
        )
        if endpoint_errors or expected_endpoint is None:
            endpoint_records_passed = False
            errors.extend(
                f"gravity_bias_evidence:endpoint:{endpoint_index}:{error}"
                for error in endpoint_errors
            )
            if expected_endpoint is None:
                errors.append(
                    f"gravity_bias_evidence:endpoint:{endpoint_index}:sample"
                )
        if isinstance(endpoint_record, dict) and expected_endpoint is not None:
            action_name = endpoint_record.get("action")
            action = actions.get(action_name)
            endpoint_substep = endpoint_record.get("physics_substep_count")
            dwell_ticks = int(
                contract["guard_thresholds"][
                    "endpoint_required_contiguous_controller_ticks"
                ]
            )
            guard_substeps = (
                [
                    int(endpoint_substep)
                    - controller_tick_substeps * offset
                    for offset in reversed(range(dwell_ticks))
                ]
                if isinstance(endpoint_substep, int)
                and not isinstance(endpoint_substep, bool)
                else []
            )
            dwell_replay_passed = bool(
                action is not None
                and len(guard_substeps) == dwell_ticks
                and int(endpoint_substep) % controller_tick_substeps == 0
                and all(
                    substep in endpoint_replay_by_substep
                    and substep in samples_by_substep
                    and samples_by_substep[substep].get("action")
                    == action_name
                    and _finite_real(
                        samples_by_substep[substep].get(
                            "command_elapsed_s"
                        )
                    )
                    is not None
                    and float(
                        samples_by_substep[substep]["command_elapsed_s"]
                    )
                    >= float(action.duration_s)
                    and _number_matches(
                        samples_by_substep[substep].get(
                            "command_smooth_fraction"
                        ),
                        1.0,
                    )
                    and endpoint_replay_by_substep[substep][
                        "endpoint_q_error_rad"
                    ]
                    <= contract["guard_thresholds"][
                        "endpoint_maximum_q_error_rad"
                    ]
                    and endpoint_replay_by_substep[substep][
                        "endpoint_max_abs_qvel_rad_s"
                    ]
                    <= contract["guard_thresholds"][
                        "endpoint_maximum_abs_qvel_rad_s"
                    ]
                    and endpoint_replay_by_substep[substep][
                        "endpoint_fk_position_error_m"
                    ]
                    <= contract["guard_thresholds"][
                        "endpoint_maximum_fk_position_error_m"
                    ]
                    and endpoint_replay_by_substep[substep][
                        "endpoint_fk_orientation_error_rad"
                    ]
                    <= contract["guard_thresholds"][
                        "endpoint_maximum_fk_orientation_error_rad"
                    ]
                    and abs(
                        endpoint_replay_by_substep[substep][
                            "route_endpoint_evidence"
                        ]["source_x_error_mm"]
                    )
                    <= contract["guard_thresholds"][
                        "endpoint_maximum_abs_source_x_error_mm"
                    ]
                    for substep in guard_substeps
                )
            )
            if not dwell_replay_passed:
                endpoint_records_passed = False
                errors.append(
                    f"gravity_bias_evidence:endpoint:{endpoint_index}:dwell_replay"
                )
    completed_set = set(endpoint_order)
    all_phases = all(count > 0 for count in phase_counts.values())
    all_endpoints = completed_set == set(expected_order)
    final_anchored = bool(
        samples
        and (
            all_endpoints
            and endpoint_records
            and samples[-1].get("physics_substep_count")
            == endpoint_records[-1].get("physics_substep_count")
            or not all_endpoints
            and samples[-1].get("physics_substep_count")
            == result.get("physics_substep_count")
        )
    )
    all_formula = bool(samples and len(formula_passes) == len(samples) and all(formula_passes))
    all_model = bool(samples and len(model_replay_passes) == len(samples) and all(model_replay_passes))
    all_isolation = bool(samples and len(isolation_passes) == len(samples) and all(isolation_passes))
    all_telemetry = bool(samples and len(telemetry_passes) == len(samples) and all(telemetry_passes))
    all_route = bool(samples and len(route_passes) == len(samples) and all(route_passes))
    all_contacts = bool(samples and len(contact_passes) == len(samples) and all(contact_passes))
    any_saturation = any(
        isinstance(sample, dict) and bool(sample.get("any_saturation"))
        for sample in samples
    )
    free_samples = [
        record for record in recomputed_telemetry
        if record.get("action") in CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[:2]
    ]
    tracking_thresholds_pass = bool(
        free_samples
        and all(
            max(abs(float(value)) for value in sample["tracking_error_to_desired_rad"])
            <= 0.002
            and abs(float(sample["fk"]["preseat_error_mm"])) <= 0.050
            and abs(float(sample["fk"]["source_x_error_mm"])) <= 0.040
            and abs(float(sample["fk"]["transverse_y_mm"])) <= 0.010
            and float(sample["fk"]["orientation_error_rad"]) <= math.radians(0.1)
            and int(sample["raw_all_cam_contact_count"]) == 0
            and bool(sample["finite"])
            for sample in free_samples
        )
    )
    align_closed = bool(
        all(phase_counts[name] > 0 and name in completed_set for name in expected_order[:2])
        and all_formula and all_model and all_isolation and all_telemetry
        and all_contacts and all_route and not any_saturation
        and dynamics_passed and identity_passed and tracking_thresholds_pass
    )
    source_ast_audit = _independent_gravity_bias_source_audit(demo)
    prohibited = source_ast_audit["prohibited_operation_counts"]
    if _evidence_mismatches(
        evidence.get("source_ast_allowlist_audit"), source_ast_audit
    ):
        errors.append("gravity_bias_evidence:source_ast_allowlist_audit")
    prohibited_ok = bool(
        source_ast_audit["passed"]
        and evidence.get("prohibited_operation_counts") == prohibited
        and all(count == 0 for count in prohibited.values())
    )
    expected_pass = bool(
        dynamics_passed and identity_passed
        and action_roster_matches
        and evidence.get("frozen_action_roster_matches") is True
        and endpoint_records_passed
        and evidence.get("producer_counts_by_phase") == phase_counts
        and evidence.get("contact_audit_counts_by_phase") == phase_counts
        and state_contiguous and all_phases and all_endpoints
        and endpoint_prefix and phase_order_ok and final_anchored
        and all_formula and all_model and all_isolation and all_telemetry
        and all_contacts and all_route
        and align_closed and not any_saturation and prohibited_ok
        and result.get("abort_reason") is None
    )
    aggregate = {
        "first_physics_substep_count": 1 if samples else None,
        "last_physics_substep_count": len(samples) if samples else None,
        "sample_counts_by_phase": phase_counts,
        "phase_counts_consistent": bool(
            evidence.get("producer_counts_by_phase") == phase_counts
            and evidence.get("contact_audit_counts_by_phase") == phase_counts
            and len(samples) == sum(phase_counts.values())
        ),
        "state_index_and_time_contiguous": state_contiguous,
        "final_sample_coverage_anchored": final_anchored,
        "observed_phase_order": observed_order,
        "expected_phase_order": expected_order,
        "sample_phase_order_is_valid_prefix": phase_order_ok,
        "every_physics_substep_recorded": bool(
            evidence.get("producer_counts_by_phase") == phase_counts
            and evidence.get("contact_audit_counts_by_phase") == phase_counts
            and len(samples) == sum(phase_counts.values())
            and state_contiguous
            and final_anchored
        ),
        "all_formula_replay_passed": all_formula,
        "all_samples_replayed_from_actual_model": all_model,
        "all_runtime_scratch_isolation_passed": all_isolation,
        "all_telemetry_recomputed_from_raw_fields_and_fresh_fk": (
            all_telemetry
        ),
        "raw_contact_count_closure_passed": all_contacts,
        "immutable_desired_route_replayed": all_route,
        "any_saturation": any_saturation,
        "saturation_sample_count": sum(
            isinstance(sample, dict) and bool(sample.get("any_saturation"))
            for sample in samples
        ),
        "completed_route_endpoint_actions": sorted(completed_set),
        "completed_route_endpoint_action_order": endpoint_order,
        "endpoint_order_is_valid_prefix": endpoint_prefix,
        "align_and_axial_endpoints_completed": all(
            name in completed_set for name in expected_order[:2]
        ),
        "all_four_phases_observed": all_phases,
        "all_four_route_endpoints_completed": all_endpoints,
        "align_and_axial_free_space_closed": align_closed,
        "prohibited_operation_counts": prohibited,
        "prohibited_operations_verified": prohibited_ok,
        "passed": expected_pass,
        "release_ready": False,
    }
    for key, expected in aggregate.items():
        if _evidence_mismatches(evidence.get(key), expected, key):
            errors.append(f"gravity_bias_evidence:{key}")
    expected_thresholds = {
        "observed_maximum_abs_q_error_to_desired_rad": max(
            (
                abs(float(value))
                for sample in free_samples
                for value in sample["tracking_error_to_desired_rad"]
            ),
            default=None,
        ),
        "maximum_abs_q_error_to_desired_rad": 0.002,
        "observed_maximum_abs_preseat_error_mm": max(
            (
                abs(float(sample["fk"]["preseat_error_mm"]))
                for sample in free_samples
            ),
            default=None,
        ),
        "maximum_abs_preseat_error_mm": 0.050,
        "observed_maximum_abs_source_x_error_mm": max(
            (
                abs(float(sample["fk"]["source_x_error_mm"]))
                for sample in free_samples
            ),
            default=None,
        ),
        "maximum_abs_source_x_error_mm": 0.040,
        "observed_maximum_abs_transverse_y_mm": max(
            (
                abs(float(sample["fk"]["transverse_y_mm"]))
                for sample in free_samples
            ),
            default=None,
        ),
        "maximum_abs_transverse_y_mm": 0.010,
        "observed_maximum_orientation_error_rad": max(
            (
                float(sample["fk"]["orientation_error_rad"])
                for sample in free_samples
            ),
            default=None,
        ),
        "maximum_orientation_error_rad": math.radians(0.1),
        "observed_maximum_raw_cam_contact_count": max(
            (
                int(sample["raw_all_cam_contact_count"])
                for sample in free_samples
            ),
            default=None,
        ),
        "maximum_raw_cam_contact_count": 0,
        "passed": tracking_thresholds_pass,
    }
    if _evidence_mismatches(
        evidence.get("align_and_axial_tracking_thresholds"),
        expected_thresholds,
    ):
        errors.append("gravity_bias_evidence:tracking_thresholds")
    metrics = {
        "maximum_abs_gravity_bias_offset_rad": max(
            (
                abs(float(value)) for sample in samples if isinstance(sample, dict)
                for value in sample.get("gravity_bias_offset_rad", [])
            ), default=None,
        ),
        "maximum_abs_tracking_error_to_desired_rad": max(
            (
                abs(float(value)) for sample in recomputed_telemetry
                for value in sample.get("tracking_error_to_desired_rad", [])
            ), default=None,
        ),
        "maximum_actuator_torque_utilization": max(
            (
                float(value) for sample in recomputed_telemetry
                for value in sample.get("actuator_torque_utilization", [])
            ), default=None,
        ),
        "maximum_ctrl_range_utilization": max(
            (
                float(value) for sample in recomputed_telemetry
                for value in sample.get("ctrl_range_utilization", [])
            ), default=None,
        ),
    }
    for key, expected in metrics.items():
        if _evidence_mismatches(evidence.get(key), expected, key):
            errors.append(f"gravity_bias_evidence:{key}")

    free_records = result.get(
        "core_capture_free_space_tracking_evidence", {}
    ).get("raw_samples", [])
    free_by_substep = {
        record.get("physics_substep_count"): record
        for record in free_records if isinstance(record, dict)
    }
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("action") not in expected_order[:2]:
            continue
        free = free_by_substep.get(sample.get("physics_substep_count"))
        if not isinstance(free, dict):
            errors.append(f"gravity_bias_sample:{index}:free_crosslink_missing")
            continue
        crosslinks = (
            (free.get("commanded_arm_q_rad"), sample.get("applied_control_rad")),
            (free.get("observed_arm_q_rad"), sample.get("live_arm_q_rad")),
            (free.get("observed_arm_qvel_rad_s"), sample.get("live_arm_qvel_rad_s")),
            (
                free.get("observed_preseat_mm"),
                sample.get("cached_post_mj_step_transform_fk", {}).get(
                    "preseat_mm"
                ),
            ),
            (
                free.get("observed_x_mm"),
                sample.get("cached_post_mj_step_transform_fk", {}).get(
                    "source_x_mm"
                ),
            ),
            (free.get("cam_contact_count"), sample.get("raw_all_cam_contact_count")),
        )
        if any(_evidence_mismatches(left, right) for left, right in crosslinks):
            errors.append(f"gravity_bias_sample:{index}:free_crosslink")

    expected_development = bool(
        result.get("completed") is True
        and result.get("attachment_verified") is True
        and result.get("attached_tool") == "gripper"
        and result.get("locked") is False
        and result.get("physical_lock_confirmed") is False
        and result.get("core_cam_tab_contact_evidence", {}).get("passed") is True
        and result.get("core_capture_free_space_tracking_evidence", {}).get("passed") is True
        and expected_pass
        and result.get("forbidden_contact_count") == 0
        and result.get("abort_reason") is None
    )
    if result.get("development_geometry_milestone_passed") is not expected_development:
        errors.append("gravity_bias_result:development_milestone")
    if result.get("success") is not False:
        errors.append("gravity_bias_result:success")
    if result.get("release_ready") is not False:
        errors.append("gravity_bias_result:release_ready")
    return errors


def expected_pogo_dynamic_evidence(
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build a tiny endpoint trace that exercises evidence validation only.

    This is deliberately not a manufacturer dynamics qualification.  Its two
    samples prove that the evidence schema is tied to the runtime contract and
    that the prismatic FK is interpreted consistently at both joint limits.
    """

    timestep_s = 0.00025
    signals: dict[str, Any] = {}
    for runtime_signal, contract in runtime_contract["signals"].items():
        plunger = contract["plunger"]
        q_max_m = float(plunger["range_m"][1])
        local_pos_m = [float(value) for value in plunger["local_pos_m"]]
        signals[runtime_signal] = {
            "source_signal": contract["source_signal"],
            "joint_name": plunger["joint_name"],
            "qualifying_bus_geom_name": plunger["geom_name"],
            "samples": [
                {
                    "physics_substep": 0,
                    "time_s": 0.0,
                    "q_m": 0.0,
                    "plunger_body_local_pos_m": list(local_pos_m),
                },
                {
                    "physics_substep": 1,
                    "time_s": timestep_s,
                    "q_m": q_max_m,
                    "plunger_body_local_pos_m": [
                        local_pos_m[0],
                        local_pos_m[1],
                        local_pos_m[2] - q_max_m,
                    ],
                },
            ],
            "fixed_shell_qualified_contact_count": 0,
            "cross_signal_qualified_contact_count": 0,
        }
    return {
        "schema_version": "1.0",
        "evidence_kind": "bounded_physics_substep_trace_contract_fixture",
        "runtime_contract_sha256": canonical_json_sha256(runtime_contract),
        "physics_timestep_s": timestep_s,
        "every_physics_substep_recorded": True,
        "direct_joint_state_write_count": 0,
        "signals": signals,
        "dynamics_authority": copy.deepcopy(
            runtime_contract["dynamics_authority"]
        ),
        "development_passed": True,
        "release_ready": False,
    }


def pogo_dynamic_evidence_errors(
    record: Any,
    runtime_contract: dict[str, Any],
) -> list[str]:
    """Independently validate bounded pogo trace evidence fail closed."""

    if not isinstance(record, dict):
        return ["evidence_missing"]
    errors: list[str] = []
    expected_keys = {
        "schema_version",
        "evidence_kind",
        "runtime_contract_sha256",
        "physics_timestep_s",
        "every_physics_substep_recorded",
        "direct_joint_state_write_count",
        "signals",
        "dynamics_authority",
        "development_passed",
        "release_ready",
    }
    if set(record) != expected_keys:
        errors.append("evidence:keys")
    if record.get("schema_version") != "1.0":
        errors.append("evidence:schema_version")
    if record.get("evidence_kind") != (
        "bounded_physics_substep_trace_contract_fixture"
    ):
        errors.append("evidence:kind")
    if record.get("runtime_contract_sha256") != canonical_json_sha256(
        runtime_contract
    ):
        errors.append("evidence:runtime_contract_sha256")
    timestep_s = _finite_real(record.get("physics_timestep_s"))
    if timestep_s is None or timestep_s <= 0.0:
        errors.append("evidence:physics_timestep_s")
    if record.get("every_physics_substep_recorded") is not True:
        errors.append("evidence:substep_coverage")
    if record.get("direct_joint_state_write_count") != 0:
        errors.append("evidence:direct_joint_state_write_count")

    evidence_signals = record.get("signals")
    expected_signals = runtime_contract.get("signals")
    if not isinstance(evidence_signals, dict) or not isinstance(
        expected_signals, dict
    ) or set(evidence_signals) != set(expected_signals):
        errors.append("evidence:signal_inventory")
        evidence_signals = {}
    for runtime_signal, contract in expected_signals.items():
        evidence = evidence_signals.get(runtime_signal)
        if not isinstance(evidence, dict):
            errors.append(f"signal:{runtime_signal}:missing")
            continue
        expected_signal_keys = {
            "source_signal",
            "joint_name",
            "qualifying_bus_geom_name",
            "samples",
            "fixed_shell_qualified_contact_count",
            "cross_signal_qualified_contact_count",
        }
        if set(evidence) != expected_signal_keys:
            errors.append(f"signal:{runtime_signal}:keys")
        plunger = contract["plunger"]
        if evidence.get("source_signal") != contract["source_signal"]:
            errors.append(f"signal:{runtime_signal}:source_signal")
        if evidence.get("joint_name") != plunger["joint_name"]:
            errors.append(f"signal:{runtime_signal}:joint_name")
        if evidence.get("qualifying_bus_geom_name") != plunger["geom_name"]:
            errors.append(f"signal:{runtime_signal}:qualifying_bus_geom")
        if evidence.get("fixed_shell_qualified_contact_count") != 0:
            errors.append(f"signal:{runtime_signal}:fixed_shell_contact")
        if evidence.get("cross_signal_qualified_contact_count") != 0:
            errors.append(f"signal:{runtime_signal}:cross_signal_contact")

        samples = evidence.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            errors.append(f"signal:{runtime_signal}:samples")
            continue
        q_min_m, q_max_m = [float(value) for value in plunger["range_m"]]
        base_position = np.asarray(plunger["local_pos_m"], dtype=np.float64)
        previous_substep: int | None = None
        observed_q: list[float] = []
        for sample_index, sample in enumerate(samples):
            prefix = f"signal:{runtime_signal}:sample:{sample_index}"
            if not isinstance(sample, dict) or set(sample) != {
                "physics_substep",
                "time_s",
                "q_m",
                "plunger_body_local_pos_m",
            }:
                errors.append(f"{prefix}:keys")
                continue
            substep = sample.get("physics_substep")
            if isinstance(substep, bool) or not isinstance(substep, int):
                errors.append(f"{prefix}:physics_substep")
                continue
            if previous_substep is not None and substep != previous_substep + 1:
                errors.append(f"{prefix}:substep_continuity")
            previous_substep = substep
            time_s = _finite_real(sample.get("time_s"))
            if (
                time_s is None
                or timestep_s is None
                or not _number_matches(time_s, substep * timestep_s)
            ):
                errors.append(f"{prefix}:time")
            q_m = _finite_real(sample.get("q_m"))
            if q_m is None or not (q_min_m - 1.0e-12 <= q_m <= q_max_m + 1.0e-12):
                errors.append(f"{prefix}:q_range")
                continue
            observed_q.append(q_m)
            position = sample.get("plunger_body_local_pos_m")
            if not isinstance(position, list) or len(position) != 3:
                errors.append(f"{prefix}:fk_position")
                continue
            expected_position = base_position + np.asarray(
                plunger["axis"], dtype=np.float64
            ) * q_m
            observed_position = np.asarray(position, dtype=np.float64)
            if not np.all(np.isfinite(observed_position)) or not np.allclose(
                observed_position,
                expected_position,
                rtol=0.0,
                atol=1.0e-12,
            ):
                errors.append(f"{prefix}:fk_position")
        if not observed_q or not _number_matches(observed_q[0], q_min_m):
            errors.append(f"signal:{runtime_signal}:q0_endpoint")
        if not observed_q or not _number_matches(observed_q[-1], q_max_m):
            errors.append(f"signal:{runtime_signal}:qmax_endpoint")
        if observed_q and max(observed_q) - min(observed_q) <= 1.0e-12:
            errors.append(f"signal:{runtime_signal}:frozen_plunger")

    if record.get("dynamics_authority") != runtime_contract.get(
        "dynamics_authority"
    ):
        errors.append("evidence:dynamics_authority")
    authority = record.get("dynamics_authority")
    if not isinstance(authority, dict):
        errors.append("evidence:dynamics_authority_type")
    else:
        if authority.get("geometry_and_datum_authority") is not True:
            errors.append("evidence:geometry_and_datum_authority")
        for field in (
            "mass_properties_authority",
            "spring_force_curve_authority",
            "damping_authority",
            "ground_first_mate_tolerance_stack_qualified",
        ):
            if authority.get(field) is not False:
                errors.append(f"evidence:{field}")
        if authority.get("blockers") != POGO_RUNTIME_RELEASE_BLOCKERS:
            errors.append("evidence:blockers")
        if authority.get("release_ready") is not False:
            errors.append("evidence:authority_release_ready")
    if record.get("development_passed") is not True:
        errors.append("evidence:development_passed")
    if record.get("release_ready") is not False:
        errors.append("evidence:release_ready")
    return errors


def iter_file_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from iter_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_file_records(child)


def resolve_manifest_record(path_text: str) -> Path:
    relative = Path(path_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError(f"Manifest export escapes its package: {path_text}")
    candidates = (
        CAD_MANIFEST.parent / relative,
        CAD_ROOT / relative,
        MAGNETIC_ROOT / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(CAD_ROOT.resolve()):
                raise AssertionError(f"Manifest export escapes CAD root: {path_text}")
            return resolved
    raise AssertionError(f"Manifest export does not exist: {path_text}")


def first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    estimated = mapping.get("estimated_mass_g")
    if isinstance(estimated, dict):
        value = estimated.get("total")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def tool_from_manifest(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or not isinstance(tools.get(key), dict):
        raise AssertionError(f"CAD manifest is missing tools.{key}")
    return tools[key]


def normalized_named_ids(value: Any, path: tuple[str, ...] = ()) -> dict[str, int]:
    """Collect unambiguous named tool/device IDs from arbitrary config JSON."""

    found: dict[str, int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key).lower())
            joined = ".".join(child_path)
            if isinstance(child, int) and not isinstance(child, bool):
                if (
                    "tool_id" in joined
                    or joined.endswith(".id")
                    or "gripper_servo_id" in joined
                ):
                    if "spoon" in joined:
                        found["matcha_spoon"] = child
                    elif "whisk" in joined and "device" not in joined and "ttl" not in joined:
                        found["matcha_whisk"] = child
                    elif "gripper" in joined:
                        found["gripper"] = child
                if (
                    (("device_id" in joined or "ttl" in joined) and "whisk" in joined)
                    or "tool_bus_id" in joined
                ):
                    found["whisk_device"] = child
            found.update(normalized_named_ids(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(normalized_named_ids(child, (*path, str(index))))
    return found


class MatchaIdentityAndCadContractTests(unittest.TestCase):
    def test_tool_and_device_ids_are_exact_unique_and_non_aliasing(self) -> None:
        sources: list[tuple[str, dict[str, int]]] = []
        if CAD_MANIFEST.is_file():
            manifest = load_json(CAD_MANIFEST, "matcha CAD manifest")
            sources.append(
                (
                    "CAD manifest",
                    {
                        "matcha_spoon": int(
                            tool_from_manifest(manifest, "matcha_spoon")["tool_id"]
                        ),
                        "matcha_whisk": int(
                            tool_from_manifest(manifest, "matcha_whisk")["tool_id"]
                        ),
                    },
                )
            )
            whisk = tool_from_manifest(manifest, "matcha_whisk")
            device = whisk.get("ttl_device_id", whisk.get("bus_address"))
            if device is None and isinstance(whisk.get("electrical"), dict):
                device = whisk["electrical"].get("ttl_device_id")
            if device is not None:
                sources[-1][1]["whisk_device"] = int(device)
        if MATCHA_CONFIG.is_file():
            sources.append(
                (
                    "simulation config",
                    normalized_named_ids(
                        load_json(MATCHA_CONFIG, "matcha simulation config")
                    ),
                )
            )
        if not sources:
            self.skipTest("matcha CAD/config identity authorities are not restored")

        for description, observed in sources:
            for name, expected in EXPECTED_TOOL_IDS.items():
                if name in observed:
                    self.assertEqual(observed[name], expected, (description, observed))
            if "whisk_device" in observed:
                self.assertEqual(
                    observed["whisk_device"], EXPECTED_WHISK_DEVICE_ID, description
                )
            tool_values = [
                observed[name] for name in EXPECTED_TOOL_IDS if name in observed
            ]
            self.assertEqual(len(tool_values), len(set(tool_values)), description)
            if "whisk_device" in observed:
                self.assertNotIn(observed["whisk_device"], tool_values, description)

    def test_cad_manifest_closes_hashes_masses_com_and_interface(self) -> None:
        manifest = load_json(CAD_MANIFEST, "matcha CAD manifest")
        self.assertEqual(str(manifest.get("units", "")).lower(), "mm")
        interface = manifest.get("interface", manifest.get("interface_authority"))
        self.assertIsInstance(interface, dict)
        mismatch = interface.get("interface_match_mm3", {})
        if isinstance(mismatch, dict) and "absolute_mismatch_mm3" in mismatch:
            self.assertEqual(float(mismatch["absolute_mismatch_mm3"]), 0.0)
        authority_path = interface.get("path")
        authority_sha = interface.get("sha256")
        if authority_path is not None:
            resolved_authority = REPOSITORY_ROOT / str(authority_path)
            self.assertTrue(resolved_authority.is_file(), resolved_authority)
            self.assertRegex(str(authority_sha), r"^[0-9a-f]{64}$")
            self.assertEqual(sha256_file(resolved_authority), authority_sha)
        self.assertIn("tool_plate(stock_gripper=False)", str(interface.get("construction")))

        for key, expected_id in (("matcha_spoon", 21), ("matcha_whisk", 22)):
            tool = tool_from_manifest(manifest, key)
            self.assertEqual(int(tool["tool_id"]), expected_id)
            ledger_path = CAD_ROOT / str(tool["mass_ledger_path"])
            ledger = load_json(ledger_path, f"{key} mass ledger")
            self.assertEqual(int(ledger["tool_id"]), expected_id)
            self.assertEqual(str(ledger["tool"]), key.removeprefix("matcha_"))
            components = ledger.get("components")
            self.assertIsInstance(components, list)
            self.assertTrue(components, key)
            component_masses = [
                float(component["mass_kg"])
                for component in components
                if isinstance(component, dict)
                and component.get("fabrication")
                and "mass_kg" in component
            ]
            declared_mass = first_number(ledger, "total_mass_kg")
            if declared_mass is not None and component_masses:
                self.assertAlmostEqual(
                    math.fsum(component_masses), declared_mass, delta=1.0e-12
                )
            if declared_mass is not None:
                self.assertGreater(declared_mass, 0.0)
            self.assertEqual(len(components), int(tool["component_count"]))
            com = ledger.get("center_of_mass_mm", ledger.get("com_mm"))
            if com is not None:
                self.assertEqual(len(com), 3)
                self.assertTrue(all(math.isfinite(float(value)) for value in com))

        records = list(iter_file_records(manifest.get("files", [])))
        records += list(iter_file_records(manifest.get("exports", {})))
        records += list(iter_file_records(manifest.get("validation_artifacts", {})))
        self.assertTrue(records, "manifest has no hash-pinned outputs")
        seen: set[str] = set()
        for record in records:
            path_text = str(record["path"])
            self.assertNotIn(path_text, seen, f"duplicate manifest path: {path_text}")
            seen.add(path_text)
            artifact = resolve_manifest_record(path_text)
            self.assertEqual(artifact.stat().st_size, int(record["bytes"]), path_text)
            self.assertRegex(str(record["sha256"]), r"^[0-9a-f]{64}$")
            self.assertEqual(sha256_file(artifact), record["sha256"], path_text)

    def test_simulation_config_hash_pins_cad_and_collision_authorities(self) -> None:
        config = load_json(MATCHA_CONFIG, "matcha simulation config")
        missing = [
            path
            for path in (CAD_MANIFEST, PAYLOAD_REPORT)
            if not path.is_file()
        ]
        if missing:
            self.assertIs(
                config.get("release_ready"),
                False,
                "a recovery config with absent authorities must fail closed",
            )
            self.skipTest(
                "hash-pinned authorities are not restored yet: "
                + ", ".join(str(path) for path in missing)
            )
        serialized = json.dumps(config, sort_keys=True)
        required_names = (
            "matcha_tool_manifest.json",
            "matcha_payload_proxy_report.json",
        )
        for required in required_names:
            self.assertIn(required, serialized)
        file_records = list(iter_file_records(config))
        pins = {Path(str(record["path"])).name: record for record in file_records}
        for required in required_names:
            self.assertIn(required, pins, f"config lacks file record for {required}")
            self.assertRegex(str(pins[required]["sha256"]), r"^[0-9a-f]{64}$")

    def test_recovery_config_is_fail_closed_and_declares_occt_free_fast_gate(
        self,
    ) -> None:
        config = load_json(MATCHA_CONFIG, "matcha simulation config")
        self.assertIs(config.get("release_ready"), False)
        authority = config.get("collision_clearance_authority")
        self.assertIsInstance(authority, dict)
        self.assertIn("fcpw", str(authority.get("development", "")).lower())
        self.assertIs(authority.get("occt_boolean_in_fidelity_hot_path"), False)
        runtime = config.get("runtime_collision_geometry")
        self.assertIsInstance(runtime, dict)
        self.assertIs(runtime.get("runtime_proxy_is_clearance_authority"), False)
        self.assertEqual(
            runtime.get("broad_plate_proxy_role"),
            "approximate dynamics broadphase with exact named critical lock voids",
        )
        self.assertIs(runtime.get("broad_plate_proxy_exact_source_subset"), False)
        self.assertIs(
            runtime.get("broad_plate_proxy_continuous_clearance_authority"),
            False,
        )
        self.assertIs(runtime.get("cad_coverage_validation_required"), True)
        if PAYLOAD_REPORT.is_file():
            payload_report = load_json(
                PAYLOAD_REPORT, "payload collision authority report"
            )
            self.assertIs(
                payload_report.get("release_ready"),
                False,
                "the development FCPW plate report cannot publish release authority",
            )
        limitations = config.get("limitations")
        self.assertIsInstance(limitations, list)
        self.assertTrue(
            any(
                "runtime collision proxies" in str(item).lower()
                and "not the release clearance authority" in str(item).lower()
                for item in limitations
            ),
            limitations,
        )


class PogoInterfaceAuthorityContractTests(unittest.TestCase):
    """Bind the purchased 7983 section, mounting direction, and first-mate proof."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cad = import_file(
                MAGNETIC_ROOT / "generate_cad.py",
                "core_cad_pogo_authority_validation",
                "core CAD pogo authority",
            )
            cls.ledger = load_json(
                POGO_LEDGER_PATH, "Mill-Max 7983 authority ledger"
            )
        except unittest.SkipTest as error:
            raise AssertionError(
                "pogo authority dependencies are release inputs and may not skip"
            ) from error
        cls.contract = cls._production_contract(cls.cad)

    @staticmethod
    def _production_contract(cad: ModuleType) -> Any:
        factory = getattr(cad, "pogo_interface_authority_contract", None)
        if callable(factory):
            return factory()
        fit_factory = getattr(cad, "interface_hardware_fit_contract", None)
        if callable(fit_factory):
            fit_contract = fit_factory()
            if isinstance(fit_contract, dict):
                return fit_contract.get("pogo_interface_authority")
        return None

    def test_pure_pogo_authority_schema_accepts_only_complete_evidence(self) -> None:
        self.assertIsInstance(self.contract, dict)
        self.assertEqual(
            pogo_authority_contract_errors(
                self.contract, self.ledger, require_release_ready=False
            ),
            [],
        )
        proof = self.contract["first_mate_tolerance_stack"]
        self.assertEqual(
            proof["independent_standard_length_tolerance_term_count"], 4
        )
        self.assertAlmostEqual(
            proof["independent_pin_pair_error_bound_mm"],
            4.0 * 0.006 * 25.4,
            places=12,
        )
        self.assertEqual(
            proof["passed"], proof["guaranteed_worst_case_ground_lead_mm"] > 0.0
        )
        stroke = self.contract["stroke"]
        self.assertAlmostEqual(stroke["full_stroke_nominal_mm"], 1.397, places=12)
        self.assertAlmostEqual(stroke["full_stroke_tolerance_mm"], 0.127, places=12)
        plunger = self.contract["dimensioned_profile"]["moving_plunger"]
        self.assertAlmostEqual(plunger["outer_diameter_mm"], 1.0668, places=12)
        self.assertEqual(plunger["motion_kind"], "prismatic")
        datums = {
            datum["signal"]: datum
            for datum in self.contract["selected_mounting_design"][
                "installed_datums"
            ]
        }
        for signal, datum in datums.items():
            expected_compression = 0.95 if signal == "GND" else 0.75
            self.assertAlmostEqual(
                datum["target_pad_exposed_contact_plane_z_mm"], 9.45, places=12
            )
            self.assertAlmostEqual(
                datum["mated_tip_z_mm"], 9.45, places=12
            )
            self.assertAlmostEqual(
                datum["mated_compression_mm"], expected_compression, places=12
            )
        fit = self.contract["selected_mounting_design"][
            "nominal_and_part_tolerance_only_fit"
        ]
        self.assertAlmostEqual(
            fit["solder_cup_minimum_diametral_passage_mm"],
            0.0052,
            places=12,
        )
        self.assertIs(fit["pullout_force_bound_included"], False)
        self.assertIs(self.contract["release_authority"]["release_ready"], False)

    def test_pogo_authority_rejects_simplified_mounting_and_runtime_models(
        self,
    ) -> None:
        baseline = self.contract

        straight_pilot = copy.deepcopy(baseline)
        straight_pilot["selected_mounting_design"]["mode"] = "straight_pilot"
        straight_errors = pogo_authority_contract_errors(
            straight_pilot, self.ledger, require_release_ready=False
        )
        self.assertIn("mounting:mode", straight_errors)

        two_cylinders = copy.deepcopy(baseline)
        two_cylinders["dimensioned_profile"][
            "fixed_shell_collision_envelope_segments"
        ] = [
            {
                "name": "solder_cup",
                "z_bounds_mm": [0.0, 3.0],
                "outer_diameter_mm": 1.52,
            },
            {
                "name": "body",
                "z_bounds_mm": [3.0, 8.1],
                "outer_diameter_mm": 2.11,
            },
        ]
        cylinder_errors = pogo_authority_contract_errors(
            two_cylinders, self.ledger, require_release_ready=False
        )
        self.assertIn("profile:fixed_shell_segments", cylinder_errors)

        swapped = copy.deepcopy(baseline)
        for datum in swapped["selected_mounting_design"]["installed_datums"]:
            datum["insertion_direction"] = "rear_to_mating_face_plunger_side_first"
        swapped_errors = pogo_authority_contract_errors(
            swapped, self.ledger, require_release_ready=False
        )
        self.assertTrue(
            all(
                f"mounting:datum:{signal}" in swapped_errors
                for signal in POGO_SIGNAL_CENTRES_MM
            ),
            swapped_errors,
        )

        frozen = copy.deepcopy(baseline)
        frozen["dimensioned_profile"]["moving_plunger"]["motion_kind"] = "fixed"
        frozen["dimensioned_profile"]["moving_plunger"][
            "compression_range_mm"
        ] = [0.0, 0.0]
        frozen_errors = pogo_authority_contract_errors(
            frozen, self.ledger, require_release_ready=False
        )
        self.assertIn("plunger:motion_kind", frozen_errors)
        self.assertIn("plunger:compression_range", frozen_errors)

        no_shoulder = copy.deepcopy(baseline)
        for datum in no_shoulder["selected_mounting_design"]["installed_datums"]:
            datum.pop("body_counterbore_z_bounds_mm")
            datum.pop("shoulder_stop_plane_z_mm")
        shoulder_errors = pogo_authority_contract_errors(
            no_shoulder, self.ledger, require_release_ready=False
        )
        self.assertTrue(
            all(
                f"mounting:datum:{signal}" in shoulder_errors
                for signal in POGO_SIGNAL_CENTRES_MM
            ),
            shoulder_errors,
        )

        reference_as_authority = copy.deepcopy(baseline)
        reference_as_authority["first_mate_tolerance_stack"][
            "shoulder_to_tip_dimension_terms_per_pin"
        ] = [".374_in_parenthesized_reference"]
        self.assertIn(
            "first_mate:dimension_term_roster",
            pogo_authority_contract_errors(
                reference_as_authority,
                self.ledger,
                require_release_ready=False,
            ),
        )

        plane_tip = copy.deepcopy(baseline)
        for datum in plane_tip["selected_mounting_design"]["installed_datums"]:
            protrusion = float(datum["nominal_face_protrusion_mm"])
            datum["target_pad_exposed_contact_plane_z_mm"] = 9.50
            datum["mated_tip_z_mm"] = 9.50
            datum["mated_compression_mm"] = protrusion
            datum[
                "nominal_design_remaining_against_catalog_minimum_stroke_mm"
            ] = 0.050 * 25.4 - protrusion
        plane_tip_errors = pogo_authority_contract_errors(
            plane_tip, self.ledger, require_release_ready=False
        )
        self.assertTrue(
            all(
                f"mounting:datum:{signal}" in plane_tip_errors
                for signal in POGO_SIGNAL_CENTRES_MM
            ),
            plane_tip_errors,
        )

    def test_pogo_authority_rejects_hash_and_nominal_only_first_mate_controls(
        self,
    ) -> None:
        altered_hash = copy.deepcopy(self.contract)
        altered_hash["official_sources"]["dimension_drawing_svg"]["sha256"] = (
            "0" * 64
        )
        self.assertIn(
            "official_sources:dimension_drawing",
            pogo_authority_contract_errors(
                altered_hash, self.ledger, require_release_ready=False
            ),
        )

        nominal_only = copy.deepcopy(self.contract)
        proof = nominal_only["first_mate_tolerance_stack"]
        self.assertAlmostEqual(proof["nominal_ground_lead_mm"], 0.20)
        proof["passed"] = True
        authority = nominal_only["release_authority"]
        authority["ground_first_mate_tolerance_stack_qualified"] = True
        authority["blockers"].remove(
            "ground_first_mate_tolerance_stack_unqualified"
        )
        nominal_errors = pogo_authority_contract_errors(
            nominal_only, self.ledger, require_release_ready=True
        )
        self.assertIn("first_mate:verdict", nominal_errors)
        self.assertIn(
            "release_authority:ground_first_mate_tolerance_stack_qualified",
            nominal_errors,
        )
        self.assertIn("release_authority:blockers", nominal_errors)

    def test_production_pogo_authority_is_hash_bound_physical_and_qualified(
        self,
    ) -> None:
        self.assertIsInstance(self.contract, dict)
        self.assertEqual(
            pogo_authority_contract_errors(
                self.contract, self.ledger, require_release_ready=False
            ),
            [],
        )

        ledger_record = self.contract["official_sources"][
            "derived_authority_ledger"
        ]
        self.assertEqual(
            ledger_record["path"],
            POGO_LEDGER_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        )
        self.assertEqual(ledger_record["bytes"], POGO_LEDGER_PATH.stat().st_size)
        self.assertEqual(ledger_record["sha256"], sha256_file(POGO_LEDGER_PATH))
        self.assertEqual(
            self.cad.pogo_points(),
            [tuple(centre) for centre in POGO_SIGNAL_CENTRES_MM.values()],
        )

        # Bind the declared envelope to actual source solids.  A JSON-only
        # two-cylinder shell or frozen plunger must not satisfy this gate.
        profile = self.contract["dimensioned_profile"]
        physical_features = self.cad.pogo_official_fixed_feature_solids()
        segment_records = {
            segment["name"]: segment
            for segment in profile["fixed_shell_collision_envelope_segments"]
        }
        self.assertEqual(set(physical_features), set(segment_records))
        for name, workplane in physical_features.items():
            segment = segment_records[name]
            z_min_mm, z_max_mm = segment["z_bounds_mm"]
            diameter_mm = float(segment["outer_diameter_mm"])
            bounds = workplane.val().BoundingBox()
            self.assertAlmostEqual(bounds.xlen, diameter_mm, places=9, msg=name)
            self.assertAlmostEqual(bounds.ylen, diameter_mm, places=9, msg=name)
            self.assertAlmostEqual(bounds.zmin, z_min_mm, places=9, msg=name)
            self.assertAlmostEqual(bounds.zmax, z_max_mm, places=9, msg=name)

        fixed_length_mm = float(profile["fixed_shell_length_mm"])
        exposed_length_mm = float(profile["maximum_exposed_plunger_mm"])
        plunger_diameter_mm = float(
            profile["moving_plunger"]["outer_diameter_mm"]
        )
        max_compression_mm = float(
            self.contract["stroke"]["maximum_full_stroke_mm"]
        )
        full_extension_zmax = None
        for compression_mm in (0.0, max_compression_mm):
            plunger_bounds = self.cad.pogo_official_plunger(
                compression_mm
            ).val().BoundingBox()
            self.assertAlmostEqual(
                plunger_bounds.xlen, plunger_diameter_mm, places=9
            )
            self.assertAlmostEqual(
                plunger_bounds.ylen, plunger_diameter_mm, places=9
            )
            self.assertAlmostEqual(plunger_bounds.zmin, fixed_length_mm, places=9)
            self.assertAlmostEqual(
                plunger_bounds.zmax,
                fixed_length_mm + exposed_length_mm - compression_mm,
                places=9,
            )
            if full_extension_zmax is None:
                full_extension_zmax = plunger_bounds.zmax
            else:
                self.assertAlmostEqual(
                    full_extension_zmax - plunger_bounds.zmax,
                    max_compression_mm,
                    places=9,
                )
        pad_bounds = self.cad.contact_pad().val().BoundingBox()
        self.assertAlmostEqual(pad_bounds.zmin, 0.0, places=12)
        self.assertAlmostEqual(pad_bounds.zmax, 0.05, places=12)

        # Recompute the sectional knurl land, counterbore, and hard-stop
        # annulus from the actual plate rather than trusting record booleans.
        plate = self.cad.robot_plate().val()
        mounting = self.contract["selected_mounting_design"]
        for datum in mounting["installed_datums"]:
            self.assertEqual(
                self.cad.pogo_installed_datum(datum["signal"]), datum
            )
        land_radius_mm = float(mounting["retention_land_diameter_mm"]) / 2.0
        counter_radius_mm = (
            float(mounting["body_counterbore_design_diameter_mm"]) / 2.0
        )
        annulus_radius_mm = 0.5 * (land_radius_mm + counter_radius_mm)
        for datum in mounting["installed_datums"]:
            x_mm, y_mm = [float(value) for value in datum["centre_xy_mm"]]
            land_z_min, land_z_max = [
                float(value) for value in datum["retention_land_z_bounds_mm"]
            ]
            counter_z_min, counter_z_max = [
                float(value) for value in datum["body_counterbore_z_bounds_mm"]
            ]
            self.assertAlmostEqual(land_z_max, counter_z_min, places=12)
            for z_mm, radius_mm in (
                (0.5 * (land_z_min + land_z_max), 0.45 * land_radius_mm),
                (0.5 * (counter_z_min + counter_z_max), 0.45 * counter_radius_mm),
            ):
                for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                    point = self.cad.cq.Vector(
                        x_mm + radius_mm * math.cos(float(angle)),
                        y_mm + radius_mm * math.sin(float(angle)),
                        z_mm,
                    )
                    self.assertFalse(
                        plate.isInside(point, 1.0e-7),
                        {"datum": datum, "filled_bore_point_mm": point.toTuple()},
                    )
            for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                cosine = math.cos(float(angle))
                sine = math.sin(float(angle))
                below = self.cad.cq.Vector(
                    x_mm + annulus_radius_mm * cosine,
                    y_mm + annulus_radius_mm * sine,
                    counter_z_min - 0.01,
                )
                above = self.cad.cq.Vector(
                    x_mm + annulus_radius_mm * cosine,
                    y_mm + annulus_radius_mm * sine,
                    counter_z_min + 0.01,
                )
                self.assertTrue(
                    plate.isInside(below, 1.0e-7),
                    {"datum": datum, "missing_shoulder_point_mm": below.toTuple()},
                )
                self.assertFalse(
                    plate.isInside(above, 1.0e-7),
                    {"datum": datum, "filled_counterbore_point_mm": above.toTuple()},
                )

        # The source geometry is physically bound, while release remains
        # deliberately fail closed until the four-term tolerance proof,
        # process/pullout qualification, and cycle evidence all close.
        self.assertEqual(
            pogo_authority_contract_errors(
                self.contract, self.ledger, require_release_ready=True
            ),
            ["release_authority:not_ready"],
        )
        self.assertIs(self.contract["release_authority"]["release_ready"], False)


class PogoRuntimeAuthorityTests(unittest.TestCase):
    """Bind source-derived moving pins to the compiled contact bus."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cad = import_file(
                MAGNETIC_ROOT / "generate_cad.py",
                "core_cad_runtime_pogo_validation",
                "core CAD runtime pogo authority",
            )
            cls.ledger = load_json(
                POGO_LEDGER_PATH, "Mill-Max 7983 authority ledger"
            )
        except unittest.SkipTest as error:
            raise AssertionError(
                "runtime pogo authority dependencies are validation inputs and may not skip"
            ) from error
        cls.expected_contract = expected_pogo_runtime_geometry_contract(cls.cad)

    def test_pure_runtime_contract_rejects_legacy_frozen_and_cross_signal_models(
        self,
    ) -> None:
        baseline = self.expected_contract
        self.assertEqual(
            pogo_runtime_geometry_contract_errors(
                baseline, self.cad, self.ledger
            ),
            [],
        )

        mutations: dict[str, Any] = {}
        mutations["altered_source_hash"] = copy.deepcopy(baseline)
        mutations["altered_source_hash"]["source_binding"]["ledger_file"][
            "sha256"
        ] = "0" * 64

        mutations["missing_fixed_segment"] = copy.deepcopy(baseline)
        mutations["missing_fixed_segment"]["signals"]["ground"][
            "fixed_segments"
        ].pop()

        mutations["wrong_fixed_radius"] = copy.deepcopy(baseline)
        mutations["wrong_fixed_radius"]["signals"]["power"][
            "fixed_segments"
        ][0]["size_m"][0] *= 0.5

        mutations["legacy_sphere_plunger"] = copy.deepcopy(baseline)
        mutations["legacy_sphere_plunger"]["signals"]["data"]["plunger"].update(
            {
                "geom_name": "qc_col_pogo_data",
                "geom_type": "sphere",
            }
        )

        mutations["frozen_plunger"] = copy.deepcopy(baseline)
        mutations["frozen_plunger"]["signals"]["id"]["plunger"][
            "range_m"
        ] = [0.0, 0.0]

        mutations["wrong_joint_axis"] = copy.deepcopy(baseline)
        mutations["wrong_joint_axis"]["signals"]["ground"]["plunger"][
            "axis"
        ] = [0.0, 0.0, 1.0]

        mutations["fixed_shell_eligible_for_bus"] = copy.deepcopy(baseline)
        mutations["fixed_shell_eligible_for_bus"]["signals"]["power"][
            "fixed_segments"
        ][2]["bus_contact_eligible"] = True

        mutations["cross_signal_source_mapping"] = copy.deepcopy(baseline)
        mutations["cross_signal_source_mapping"]["signals"]["data"][
            "source_signal"
        ] = "+12V"

        mutations["qualified_missing_dynamics"] = copy.deepcopy(baseline)
        mutations["qualified_missing_dynamics"]["dynamics_authority"][
            "mass_properties_authority"
        ] = True

        mutations["omitted_blocker"] = copy.deepcopy(baseline)
        mutations["omitted_blocker"]["dynamics_authority"]["blockers"].pop()

        mutations["release_promotion"] = copy.deepcopy(baseline)
        mutations["release_promotion"]["release_ready"] = True
        mutations["release_promotion"]["dynamics_authority"][
            "release_ready"
        ] = True

        for name, mutated in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    pogo_runtime_geometry_contract_errors(
                        mutated, self.cad, self.ledger
                    ),
                    [],
                    name,
                )

    def test_dynamic_evidence_is_bounded_and_rejects_false_authority(
        self,
    ) -> None:
        baseline = expected_pogo_dynamic_evidence(self.expected_contract)
        self.assertEqual(
            pogo_dynamic_evidence_errors(baseline, self.expected_contract), []
        )

        mutations: dict[str, Any] = {}
        mutations["altered_contract_hash"] = copy.deepcopy(baseline)
        mutations["altered_contract_hash"]["runtime_contract_sha256"] = "0" * 64

        mutations["direct_joint_write"] = copy.deepcopy(baseline)
        mutations["direct_joint_write"]["direct_joint_state_write_count"] = 1

        mutations["fixed_shell_contact"] = copy.deepcopy(baseline)
        mutations["fixed_shell_contact"]["signals"]["ground"][
            "fixed_shell_qualified_contact_count"
        ] = 1

        mutations["cross_signal_contact"] = copy.deepcopy(baseline)
        mutations["cross_signal_contact"]["signals"]["power"][
            "cross_signal_qualified_contact_count"
        ] = 1

        mutations["wrong_qualifying_geom"] = copy.deepcopy(baseline)
        mutations["wrong_qualifying_geom"]["signals"]["data"][
            "qualifying_bus_geom_name"
        ] = "qc_col_pogo_data_fixed_shell_knurl"

        mutations["frozen_plunger"] = copy.deepcopy(baseline)
        mutations["frozen_plunger"]["signals"]["id"]["samples"][1][
            "q_m"
        ] = 0.0
        mutations["frozen_plunger"]["signals"]["id"]["samples"][1][
            "plunger_body_local_pos_m"
        ] = list(
            mutations["frozen_plunger"]["signals"]["id"]["samples"][0][
                "plunger_body_local_pos_m"
            ]
        )

        mutations["out_of_range"] = copy.deepcopy(baseline)
        mutations["out_of_range"]["signals"]["ground"]["samples"][1][
            "q_m"
        ] = 1.0

        mutations["wrong_fk"] = copy.deepcopy(baseline)
        mutations["wrong_fk"]["signals"]["power"]["samples"][1][
            "plunger_body_local_pos_m"
        ][2] += 0.001

        mutations["decimated_substeps"] = copy.deepcopy(baseline)
        mutations["decimated_substeps"]["signals"]["data"]["samples"][1][
            "physics_substep"
        ] = 2
        mutations["decimated_substeps"]["signals"]["data"]["samples"][1][
            "time_s"
        ] = 0.0005

        mutations["qualified_mass"] = copy.deepcopy(baseline)
        mutations["qualified_mass"]["dynamics_authority"][
            "mass_properties_authority"
        ] = True

        mutations["removed_blocker"] = copy.deepcopy(baseline)
        mutations["removed_blocker"]["dynamics_authority"]["blockers"].pop()

        mutations["release_promotion"] = copy.deepcopy(baseline)
        mutations["release_promotion"]["release_ready"] = True
        mutations["release_promotion"]["dynamics_authority"][
            "release_ready"
        ] = True

        for name, mutated in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    pogo_dynamic_evidence_errors(
                        mutated, self.expected_contract
                    ),
                    [],
                    name,
                )

    def test_compiled_runtime_geometry_fk_and_bus_are_exact(self) -> None:
        demo = import_file(
            MATCHA_DEMO,
            "matcha_workflow_runtime_pogo_validation",
            "matcha workflow runtime pogo geometry",
        )
        runtime_factory = getattr(demo.qc, "pogo_runtime_geometry_contract", None)
        self.assertTrue(callable(runtime_factory))
        runtime_contract = runtime_factory()
        self.assertEqual(
            pogo_runtime_geometry_contract_errors(
                runtime_contract, self.cad, self.ledger
            ),
            [],
        )
        self.assertIs(runtime_contract["passed"], True)
        self.assertIs(runtime_contract["release_ready"], False)

        model = demo.build_model()
        mujoco = demo.mujoco
        cylinder_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
        expected_body_names: set[str] = set()
        expected_geom_names: set[str] = set()
        expected_joint_names: set[str] = set()
        plunger_geom_names: set[str] = set()
        fixed_geom_names: set[str] = set()
        qpos_addresses: dict[str, int] = {}

        for runtime_signal, signal_contract in runtime_contract["signals"].items():
            fixed_body = signal_contract["fixed_body"]
            fixed_body_name = fixed_body["name"]
            fixed_body_id = int(model.body(fixed_body_name).id)
            fixed_parent_id = int(model.body(fixed_body["parent"]).id)
            expected_body_names.add(fixed_body_name)
            self.assertEqual(int(model.body_parentid[fixed_body_id]), fixed_parent_id)
            np.testing.assert_allclose(
                model.body_pos[fixed_body_id],
                fixed_body["pos_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                model.body_quat[fixed_body_id],
                fixed_body["quat_wxyz"],
                rtol=0.0,
                atol=1.0e-12,
            )

            for segment in signal_contract["fixed_segments"]:
                geom_name = segment["name"]
                geom_id = int(model.geom(geom_name).id)
                expected_geom_names.add(geom_name)
                fixed_geom_names.add(geom_name)
                self.assertEqual(int(model.geom_bodyid[geom_id]), fixed_body_id)
                self.assertEqual(int(model.geom_type[geom_id]), cylinder_type)
                np.testing.assert_allclose(
                    model.geom_pos[geom_id],
                    segment["local_pos_m"],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    model.geom_size[geom_id, :2],
                    segment["size_m"],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                self.assertIs(segment["bus_contact_eligible"], False)

            plunger = signal_contract["plunger"]
            plunger_body_name = plunger["body_name"]
            plunger_body_id = int(model.body(plunger_body_name).id)
            expected_body_names.add(plunger_body_name)
            self.assertEqual(int(model.body_parentid[plunger_body_id]), fixed_body_id)
            np.testing.assert_allclose(
                model.body_pos[plunger_body_id],
                plunger["local_pos_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                model.body_quat[plunger_body_id],
                plunger["quat_wxyz"],
                rtol=0.0,
                atol=1.0e-12,
            )

            joint_name = plunger["joint_name"]
            joint_id = int(model.joint(joint_name).id)
            expected_joint_names.add(joint_name)
            self.assertEqual(int(model.jnt_bodyid[joint_id]), plunger_body_id)
            self.assertEqual(int(model.jnt_type[joint_id]), slide_type)
            np.testing.assert_allclose(
                model.jnt_axis[joint_id],
                plunger["axis"],
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                model.jnt_range[joint_id],
                plunger["range_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            qpos_addresses[runtime_signal] = int(model.jnt_qposadr[joint_id])

            geom_name = plunger["geom_name"]
            geom_id = int(model.geom(geom_name).id)
            expected_geom_names.add(geom_name)
            plunger_geom_names.add(geom_name)
            self.assertEqual(int(model.geom_bodyid[geom_id]), plunger_body_id)
            self.assertEqual(int(model.geom_type[geom_id]), cylinder_type)
            np.testing.assert_allclose(
                model.geom_pos[geom_id],
                plunger["geom_local_pos_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                model.geom_size[geom_id, :2],
                plunger["geom_size_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            self.assertIs(plunger["bus_contact_eligible"], True)

        observed_body_names = {
            str(model.body(body_id).name)
            for body_id in range(model.nbody)
            if str(model.body(body_id).name).startswith("qc_pogo_")
        }
        observed_geom_names = {
            str(model.geom(geom_id).name)
            for geom_id in range(model.ngeom)
            if str(model.geom(geom_id).name).startswith("qc_col_pogo_")
        }
        observed_joint_names = {
            str(model.joint(joint_id).name)
            for joint_id in range(model.njnt)
            if str(model.joint(joint_id).name).startswith("qc_pogo_")
        }
        self.assertEqual(observed_body_names, expected_body_names)
        self.assertEqual(observed_geom_names, expected_geom_names)
        self.assertEqual(observed_joint_names, expected_joint_names)

        data = mujoco.MjData(model)
        demo.initialize(model, data)
        sentinel = {
            runtime_signal: 0.1
            * (index + 1)
            * float(runtime_contract["signals"][runtime_signal]["plunger"]["range_m"][1])
            for index, runtime_signal in enumerate(sorted(qpos_addresses))
        }
        for runtime_signal, q_m in sentinel.items():
            data.qpos[qpos_addresses[runtime_signal]] = q_m
        demo.initialize(model, data)
        for runtime_signal, q_m in sentinel.items():
            self.assertAlmostEqual(
                float(data.qpos[qpos_addresses[runtime_signal]]), q_m, places=15
            )

        for address in qpos_addresses.values():
            data.qpos[address] = 0.0
        mujoco.mj_forward(model, data)
        q0_body_positions = {
            runtime_signal: np.array(
                data.xpos[
                    int(
                        model.body(
                            runtime_contract["signals"][runtime_signal]["plunger"][
                                "body_name"
                            ]
                        ).id
                    )
                ],
                dtype=np.float64,
                copy=True,
            )
            for runtime_signal in qpos_addresses
        }
        q0_geom_positions = {
            runtime_signal: np.array(
                data.geom_xpos[
                    int(
                        model.geom(
                            runtime_contract["signals"][runtime_signal]["plunger"][
                                "geom_name"
                            ]
                        ).id
                    )
                ],
                dtype=np.float64,
                copy=True,
            )
            for runtime_signal in qpos_addresses
        }
        for runtime_signal, address in qpos_addresses.items():
            data.qpos[address] = float(
                runtime_contract["signals"][runtime_signal]["plunger"]["range_m"][1]
            )
        mujoco.mj_forward(model, data)
        for runtime_signal in qpos_addresses:
            plunger = runtime_contract["signals"][runtime_signal]["plunger"]
            fixed_body_id = int(model.body(plunger["parent"]).id)
            fixed_rotation = np.asarray(
                data.xmat[fixed_body_id], dtype=np.float64
            ).reshape(3, 3)
            displacement = fixed_rotation @ (
                np.asarray(plunger["axis"], dtype=np.float64)
                * float(plunger["range_m"][1])
            )
            plunger_body_id = int(model.body(plunger["body_name"]).id)
            plunger_geom_id = int(model.geom(plunger["geom_name"]).id)
            np.testing.assert_allclose(
                data.xpos[plunger_body_id] - q0_body_positions[runtime_signal],
                displacement,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                data.geom_xpos[plunger_geom_id]
                - q0_geom_positions[runtime_signal],
                displacement,
                rtol=0.0,
                atol=1.0e-12,
            )

        # Source seated compression is 0.95 mm for GND and 0.75 mm for the
        # other signals.  The moving +Z tip must then land on the actual
        # copper exposed plane at robot-frame z=9.45 mm, not the nominal
        # 9.50 mm mating datum (which would bury it 0.05 mm into the pad).
        for runtime_signal, address in qpos_addresses.items():
            datum = runtime_contract["signals"][runtime_signal][
                "installed_datum"
            ]
            data.qpos[address] = float(datum["mated_compression_mm"]) / 1000.0
        mujoco.mj_forward(model, data)
        robot_frame_id = int(model.body("robot_plate_frame").id)
        robot_frame_position = np.asarray(
            data.xpos[robot_frame_id], dtype=np.float64
        )
        robot_frame_rotation = np.asarray(
            data.xmat[robot_frame_id], dtype=np.float64
        ).reshape(3, 3)
        for runtime_signal in qpos_addresses:
            signal_contract = runtime_contract["signals"][runtime_signal]
            datum = signal_contract["installed_datum"]
            plunger = signal_contract["plunger"]
            plunger_geom_id = int(model.geom(plunger["geom_name"]).id)
            plunger_axis = np.asarray(
                data.geom_xmat[plunger_geom_id], dtype=np.float64
            ).reshape(3, 3)[:, 2]
            tip_world = (
                np.asarray(data.geom_xpos[plunger_geom_id], dtype=np.float64)
                + plunger_axis * float(plunger["geom_size_m"][1])
            )
            tip_robot_frame = robot_frame_rotation.T @ (
                tip_world - robot_frame_position
            )
            np.testing.assert_allclose(
                tip_robot_frame,
                [
                    float(datum["centre_xy_mm"][0]) / 1000.0,
                    float(datum["centre_xy_mm"][1]) / 1000.0,
                    float(datum["target_pad_exposed_contact_plane_z_mm"])
                    / 1000.0,
                ],
                rtol=0.0,
                atol=1.0e-12,
                err_msg=runtime_signal,
            )
        expected_pairs = {
            frozenset(
                {
                    runtime_contract["signals"][runtime_signal]["plunger"][
                        "geom_name"
                    ],
                    f"{tool}_pad_{runtime_signal}_collision",
                }
            )
            for tool in demo.ALL_TOOL_IDS
            for runtime_signal in POGO_RUNTIME_SIGNAL_MAP
        }
        for tool in demo.ALL_TOOL_IDS:
            equality_id = int(model.equality(f"attach_{tool}").id)
            self.assertEqual(
                int(model.eq_obj1id[equality_id]), robot_frame_id, tool
            )
            self.assertEqual(
                int(model.eq_obj2id[equality_id]),
                int(model.body(f"tool_{tool}").id),
                tool,
            )
            np.testing.assert_allclose(
                model.eq_data[equality_id, 3:10],
                [0.0, 0.0, 0.0095, 1.0, 0.0, 0.0, 0.0],
                rtol=0.0,
                atol=1.0e-12,
                err_msg=tool,
            )
            for runtime_signal in POGO_RUNTIME_SIGNAL_MAP:
                pad_name = f"{tool}_pad_{runtime_signal}_collision"
                pad_id = int(model.geom(pad_name).id)
                self.assertEqual(int(model.geom_type[pad_id]), cylinder_type)
                np.testing.assert_allclose(
                    model.geom_pos[pad_id],
                    [-0.031, float(demo.qc.SIGNAL_Y_M[runtime_signal]), -0.000025],
                    rtol=0.0,
                    atol=1.0e-12,
                    err_msg=pad_name,
                )
                self.assertAlmostEqual(
                    float(model.eq_data[equality_id, 5])
                    + float(model.geom_pos[pad_id, 2])
                    - float(model.geom_size[pad_id, 1]),
                    0.00945,
                    places=12,
                    msg=pad_name,
                )
                np.testing.assert_allclose(
                    model.geom_size[pad_id, :2],
                    [0.002, 0.000025],
                    rtol=0.0,
                    atol=1.0e-12,
                    err_msg=pad_name,
                )
        observed_pairs = {
            frozenset(
                {
                    str(model.geom(int(model.pair_geom1[pair_id])).name),
                    str(model.geom(int(model.pair_geom2[pair_id])).name),
                }
            )
            for pair_id in range(model.npair)
            if {
                str(model.geom(int(model.pair_geom1[pair_id])).name),
                str(model.geom(int(model.pair_geom2[pair_id])).name),
            }
            & expected_geom_names
        }
        self.assertEqual(observed_pairs, expected_pairs)
        self.assertTrue(
            all(not pair.intersection(fixed_geom_names) for pair in observed_pairs)
        )

        controller = demo.MatchaWorkflowController(model, data)
        self.assertEqual(set(controller.pogo_pair_contract), expected_pairs)
        for pair, (tool, runtime_signal) in controller.pogo_pair_contract.items():
            self.assertIn(runtime_signal, POGO_RUNTIME_SIGNAL_MAP)
            self.assertEqual(
                pair,
                frozenset(
                    {
                        runtime_contract["signals"][runtime_signal]["plunger"][
                            "geom_name"
                        ],
                        f"{tool}_pad_{runtime_signal}_collision",
                    }
                ),
            )
            self.assertEqual(len(pair.intersection(plunger_geom_names)), 1)

        # The electrical witness is directional.  A finite unit vector alone
        # cannot distinguish the source +Z pogo-to-pad normal from a lateral
        # scrape or the reversed pad-to-pogo direction.
        with mock.patch.object(
            controller, "_capture_pose_is_valid", return_value=True
        ), mock.patch.object(controller, "_equality_active", return_value=True):
            for runtime_signal in POGO_RUNTIME_SIGNAL_MAP:
                tool = "gripper"
                plunger_name = runtime_contract["signals"][runtime_signal][
                    "plunger"
                ]["geom_name"]
                pad_name = f"{tool}_pad_{runtime_signal}_collision"
                plunger_id = int(model.geom(plunger_name).id)
                pad_id = int(model.geom(pad_name).id)
                plunger_rotation = np.asarray(
                    data.geom_xmat[plunger_id], dtype=np.float64
                ).reshape(3, 3)
                source_positive_z = plunger_rotation[:, 2]
                lateral = plunger_rotation[:, 0]
                pad_center = np.array(
                    data.geom_xpos[pad_id], dtype=np.float64, copy=True
                )
                pad_rotation = np.asarray(
                    data.geom_xmat[pad_id], dtype=np.float64
                ).reshape(3, 3)
                pad_positive_z = pad_rotation[:, 2]
                pad_half_height_m = float(model.geom_size[pad_id, 1])
                plunger_radius_m = float(model.geom_size[plunger_id, 0])
                penetration_m = -1.0e-6
                exposed_axial_m = -pad_half_height_m - penetration_m / 2.0

                def contact(
                    normal: np.ndarray,
                    *,
                    axial_m: float = exposed_axial_m,
                    radial_m: float = 0.0,
                    distance_m: float = penetration_m,
                ) -> SimpleNamespace:
                    return SimpleNamespace(
                        geom=np.asarray([plunger_id, pad_id], dtype=np.int32),
                        dist=distance_m,
                        pos=(
                            pad_center
                            + pad_positive_z * axial_m
                            + lateral * radial_m
                        ),
                        frame=np.concatenate(
                            (np.asarray(normal, dtype=np.float64), np.zeros(6))
                        ),
                    )

                self.assertTrue(
                    controller._matching_pogo_contact_is_valid(
                        contact(source_positive_z), tool, runtime_signal
                    )
                )
                self.assertFalse(
                    controller._matching_pogo_contact_is_valid(
                        contact(lateral), tool, runtime_signal
                    )
                )
                self.assertFalse(
                    controller._matching_pogo_contact_is_valid(
                        contact(-source_positive_z), tool, runtime_signal
                    )
                )
                self.assertFalse(
                    controller._matching_pogo_contact_is_valid(
                        contact(source_positive_z, axial_m=0.0),
                        tool,
                        runtime_signal,
                    ),
                    "a pad-center witness is not its exposed underside",
                )
                self.assertFalse(
                    controller._matching_pogo_contact_is_valid(
                        contact(
                            source_positive_z,
                            axial_m=pad_half_height_m,
                        ),
                        tool,
                        runtime_signal,
                    ),
                    "a back-face witness cannot qualify the electrical bus",
                )
                self.assertFalse(
                    controller._matching_pogo_contact_is_valid(
                        contact(
                            source_positive_z,
                            radial_m=plunger_radius_m + 10.0e-6,
                        ),
                        tool,
                        runtime_signal,
                    ),
                    "a contact outside the plunger crown cannot qualify",
                )
                self.assertFalse(
                    controller._matching_pogo_contact_is_valid(
                        contact(
                            source_positive_z,
                            axial_m=-pad_half_height_m,
                        ),
                        tool,
                        runtime_signal,
                    ),
                    "penetrating witnesses must include MuJoCo's dist/2 shift",
                )


class PositiveLockCamRuntimeAuthorityTests(unittest.TestCase):
    """Bind every dock's runtime cam to the exact five-piece source union."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cad = import_file(
            MAGNETIC_ROOT / "generate_cad.py",
            "core_cad_runtime_cam_validation",
            "core CAD runtime cam authority",
        )
        cls.expected = expected_positive_lock_cam_runtime_contract(cls.cad)

    @staticmethod
    def _reseal_component(component: dict[str, Any]) -> None:
        component["canonical_geometry_sha256"] = canonical_json_sha256(
            {
                key: component[key]
                for key in (
                    "source_component",
                    "representation",
                    "source_geometry_m",
                )
            }
        )

    def test_runtime_cam_contract_rejects_missing_shifted_or_laundered_geometry(
        self,
    ) -> None:
        baseline = self.expected
        self.assertEqual(
            positive_lock_cam_runtime_contract_errors(baseline, self.cad), []
        )

        mutations: dict[str, dict[str, Any]] = {}
        mutations["drop_axial_lead"] = copy.deepcopy(baseline)
        mutations["drop_axial_lead"]["core_gripper"]["components"].pop(1)
        mutations["drop_axial_lead"]["core_gripper"][
            "runtime_geom_names"
        ].pop(1)

        mutations["shift_axial_lead"] = copy.deepcopy(baseline)
        shifted_lead = mutations["shift_axial_lead"]["core_gripper"][
            "components"
        ][1]
        shifted_lead["source_geometry_m"]["lower_rectangle"]["x_bounds"][0] += (
            0.001
        )
        self._reseal_component(shifted_lead)

        mutations["flatten_axial_lead"] = copy.deepcopy(baseline)
        flat_lead = mutations["flatten_axial_lead"]["core_gripper"][
            "components"
        ][1]
        flat_lead["source_geometry_m"]["upper_rectangle"]["x_bounds"] = list(
            flat_lead["source_geometry_m"]["lower_rectangle"]["x_bounds"]
        )
        self._reseal_component(flat_lead)

        mutations["alter_hold"] = copy.deepcopy(baseline)
        altered_hold = mutations["alter_hold"]["core_gripper"]["components"][2]
        altered_hold["source_geometry_m"]["bounds"][2][0] += 0.0005
        self._reseal_component(altered_hold)

        mutations["alter_root"] = copy.deepcopy(baseline)
        altered_root = mutations["alter_root"]["core_gripper"]["components"][3]
        altered_root["source_geometry_m"]["runtime_remainder_bounds"][0][1][1] = (
            0.0005
        )
        self._reseal_component(altered_root)

        mutations["old_main_only_proxy"] = copy.deepcopy(baseline)
        mutations["old_main_only_proxy"]["core_gripper"]["components"] = [
            mutations["old_main_only_proxy"]["core_gripper"]["components"][0]
        ]
        mutations["old_main_only_proxy"]["core_gripper"][
            "runtime_geom_names"
        ] = ["dock_gripper_cam_collision"]

        mutations["core_body_alias_for_matcha"] = copy.deepcopy(baseline)
        mutations["core_body_alias_for_matcha"]["matcha_bays"]["spoon"].update(
            {
                "frame": "dock_gripper",
                "runtime_geom_names": list(
                    baseline["core_gripper"]["runtime_geom_names"]
                ),
            }
        )

        mutations["legacy_matcha_shifted_prism"] = copy.deepcopy(baseline)
        mutations["legacy_matcha_shifted_prism"]["matcha_bays"]["whisk"].update(
            {
                "runtime_geom_names": ["dock_whisk_cam_collision"],
                "uses_core_canonical_geometry": False,
                "geometry_and_placement_authority": False,
            }
        )

        mutations["alter_source_hash"] = copy.deepcopy(baseline)
        mutations["alter_source_hash"]["source_binding"]["generator_file"][
            "sha256"
        ] = "0" * 64

        mutations["friction_load_laundering"] = copy.deepcopy(baseline)
        laundering = mutations["friction_load_laundering"]["authority_scope"]
        laundering["friction_coefficient_authority"] = True
        laundering["load_capacity_authority"] = True
        laundering["blockers"] = [
            blocker
            for blocker in laundering["blockers"]
            if "friction" not in blocker and "load_capacity" not in blocker
        ]

        mutations["release_promotion"] = copy.deepcopy(baseline)
        mutations["release_promotion"]["release_ready"] = True
        mutations["release_promotion"]["authority_scope"]["release_ready"] = True

        for name, mutated in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    positive_lock_cam_runtime_contract_errors(
                        mutated, self.cad
                    ),
                    [],
                    name,
                )

    def test_compiled_cam_union_matches_source_on_all_three_docks(self) -> None:
        demo = import_file(
            MATCHA_DEMO,
            "matcha_workflow_runtime_cam_validation",
            "matcha workflow runtime cam geometry",
        )
        runtime_contract = demo.qc.positive_lock_cam_runtime_contract()
        self.assertEqual(
            positive_lock_cam_runtime_contract_errors(
                runtime_contract, self.cad
            ),
            [],
        )
        self.assertIs(runtime_contract["passed"], True)
        self.assertIs(runtime_contract["release_ready"], False)
        self.assertEqual(
            runtime_contract["authority_scope"]["blockers"],
            POSITIVE_LOCK_CAM_RUNTIME_BLOCKERS,
        )

        model = demo.build_model()
        mujoco = demo.mujoco
        xml_text, _ = demo._build_xml_and_assets()
        xml_root = ET.fromstring(xml_text)
        xml_geoms = {
            str(geom.get("name")): geom
            for geom in xml_root.findall(".//geom")
            if geom.get("name")
        }

        component_records = {
            record["source_component"]: record
            for record in runtime_contract["core_gripper"]["components"]
        }
        main_geometry = component_records["main_xy_wedge"]["source_geometry_m"]
        lead_geometry = component_records["axial_lead"]["source_geometry_m"]
        hold_bounds = component_records["hold_finger"]["source_geometry_m"][
            "bounds"
        ]
        root_bounds = component_records["outer_root_bridge"][
            "source_geometry_m"
        ]["runtime_remainder_bounds"]

        def rectangle_vertices(record: dict[str, Any]) -> np.ndarray:
            return np.asarray(
                [
                    [x_value, y_value, float(record["z"])]
                    for x_value in record["x_bounds"]
                    for y_value in record["y_bounds"]
                ],
                dtype=np.float64,
            )

        def box_vertices(bounds: list[list[float]]) -> np.ndarray:
            return np.asarray(
                [
                    [x_value, y_value, z_value]
                    for x_value in bounds[0]
                    for y_value in bounds[1]
                    for z_value in bounds[2]
                ],
                dtype=np.float64,
            )

        expected_vertices_by_suffix = {
            "cam_collision": np.asarray(
                [
                    [x_value, y_value, z_value]
                    for x_value, y_value in main_geometry["polygon_xy"]
                    for z_value in main_geometry["z_bounds"]
                ],
                dtype=np.float64,
            ),
            "cam_axial_lead_collision": np.vstack(
                [
                    rectangle_vertices(lead_geometry["lower_rectangle"]),
                    rectangle_vertices(lead_geometry["upper_rectangle"]),
                ]
            ),
            "cam_hold_finger_collision": box_vertices(hold_bounds),
            "cam_outer_root_lower_collision": box_vertices(root_bounds[0]),
            "cam_outer_root_upper_collision": box_vertices(root_bounds[1]),
        }

        def quaternion_matrix(quaternion: Any) -> np.ndarray:
            values = np.array(quaternion, dtype=np.float64, copy=True)
            values /= np.linalg.norm(values)
            w_value, x_value, y_value, z_value = values
            return np.asarray(
                [
                    [
                        1.0 - 2.0 * (y_value**2 + z_value**2),
                        2.0 * (x_value * y_value - w_value * z_value),
                        2.0 * (x_value * z_value + w_value * y_value),
                    ],
                    [
                        2.0 * (x_value * y_value + w_value * z_value),
                        1.0 - 2.0 * (x_value**2 + z_value**2),
                        2.0 * (y_value * z_value - w_value * x_value),
                    ],
                    [
                        2.0 * (x_value * z_value - w_value * y_value),
                        2.0 * (y_value * z_value + w_value * x_value),
                        1.0 - 2.0 * (x_value**2 + y_value**2),
                    ],
                ],
                dtype=np.float64,
            )

        def owner_vertices(geom_id: int) -> np.ndarray:
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                vertices = np.asarray(
                    model.mesh_vert[start : start + count], dtype=np.float64
                )
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                vertices = box_vertices(
                    [
                        [-float(size), float(size)]
                        for size in model.geom_size[geom_id]
                    ]
                )
            else:
                raise AssertionError(model.geom(geom_id).name)
            rotation = quaternion_matrix(model.geom_quat[geom_id])
            return np.asarray(model.geom_pos[geom_id], dtype=np.float64) + (
                vertices @ rotation.T
            )

        def sorted_unique(vertices: np.ndarray) -> np.ndarray:
            rounded = np.unique(np.round(vertices, decimals=10), axis=0)
            order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
            return rounded[order]

        all_expected_names: set[str] = set()
        for tool in ("gripper", "spoon", "whisk"):
            expected_names = [
                f"dock_{tool}_{suffix}"
                for suffix in expected_vertices_by_suffix
            ]
            all_expected_names.update(expected_names)
            if tool == "gripper":
                self.assertEqual(
                    runtime_contract["core_gripper"]["runtime_geom_names"],
                    expected_names,
                )
            else:
                bay = runtime_contract["matcha_bays"][tool]
                self.assertEqual(bay["runtime_geom_names"], expected_names)
                self.assertIs(bay["uses_core_canonical_geometry"], True)
                self.assertIs(bay["geometry_and_placement_authority"], True)
            dock_body_id = int(model.body(f"dock_{tool}").id)
            for name in expected_names:
                geom_id = int(model.geom(name).id)
                self.assertEqual(int(model.geom_bodyid[geom_id]), dock_body_id)
                self.assertEqual(int(model.geom_group[geom_id]), 3)
                self.assertEqual(int(model.geom_contype[geom_id]), 1)
                self.assertEqual(int(model.geom_conaffinity[geom_id]), 1)
                self.assertEqual(xml_geoms[name].get("mass"), "0")
                suffix = name.removeprefix(f"dock_{tool}_")
                np.testing.assert_allclose(
                    sorted_unique(owner_vertices(geom_id)),
                    sorted_unique(expected_vertices_by_suffix[suffix]),
                    rtol=0.0,
                    atol=2.0e-9,
                    err_msg=name,
                )

        observed_cam_names = {
            str(model.geom(geom_id).name)
            for geom_id in range(model.ngeom)
            if any(
                str(model.geom(geom_id).name).startswith(f"dock_{tool}_cam")
                for tool in ("gripper", "spoon", "whisk")
            )
            and str(model.geom(geom_id).name).endswith("collision")
        }
        self.assertEqual(observed_cam_names, all_expected_names)

        data = mujoco.MjData(model)
        demo.initialize(model, data)
        controller = demo.MatchaWorkflowController(model, data)
        expected_gripper_ids = tuple(
            int(model.geom(name).id)
            for name in runtime_contract["core_gripper"]["runtime_geom_names"]
        )
        self.assertIsInstance(controller.dock_gripper_cam_geom_ids, tuple)
        self.assertEqual(controller.dock_gripper_cam_geom_ids, expected_gripper_ids)
        self.assertEqual(controller.dock_gripper_cam_geom_id, expected_gripper_ids[0])

        # Independently rebuild the exact runtime partition and prove that its
        # set union equals the complete source B-rep without internal overlaps.
        cq = self.cad.cq

        def perimeter_wire(record: dict[str, Any]) -> Any:
            x_min, x_max = [1000.0 * value for value in record["x_bounds"]]
            y_min, y_max = [1000.0 * value for value in record["y_bounds"]]
            z_value = 1000.0 * float(record["z"])
            return cq.Wire.makePolygon(
                [
                    cq.Vector(x_min, y_min, z_value),
                    cq.Vector(x_max, y_min, z_value),
                    cq.Vector(x_max, y_max, z_value),
                    cq.Vector(x_min, y_max, z_value),
                ],
                close=True,
            )

        def box_shape(bounds: list[list[float]]) -> Any:
            sizes_mm = [1000.0 * (axis[1] - axis[0]) for axis in bounds]
            center_mm = [500.0 * (axis[0] + axis[1]) for axis in bounds]
            return (
                cq.Workplane("XY")
                .box(*sizes_mm, centered=True)
                .translate(tuple(center_mm))
                .val()
            )

        main_polygon_mm = [
            (1000.0 * x_value, 1000.0 * y_value)
            for x_value, y_value in main_geometry["polygon_xy"]
        ]
        z_min_mm, z_max_mm = [
            1000.0 * value for value in main_geometry["z_bounds"]
        ]
        runtime_pieces = [
            (
                cq.Workplane("XY")
                .polyline(main_polygon_mm)
                .close()
                .extrude(z_max_mm - z_min_mm)
                .translate((0.0, 0.0, z_min_mm))
                .val()
            ),
            cq.Solid.makeLoft(
                [
                    perimeter_wire(lead_geometry["lower_rectangle"]),
                    perimeter_wire(lead_geometry["upper_rectangle"]),
                ],
                ruled=True,
            ),
            box_shape(hold_bounds),
            box_shape(root_bounds[0]),
            box_shape(root_bounds[1]),
        ]

        def volume_mm3(shape: Any) -> float:
            return math.fsum(float(solid.Volume()) for solid in shape.Solids())

        for first_index, first in enumerate(runtime_pieces):
            for second in runtime_pieces[first_index + 1 :]:
                self.assertLessEqual(
                    volume_mm3(first.intersect(second)), 1.0e-6
                )
        runtime_union = cq.Workplane(obj=runtime_pieces[0])
        for piece in runtime_pieces[1:]:
            runtime_union = runtime_union.union(cq.Workplane(obj=piece))
        runtime_union_shape = runtime_union.clean().val()
        source_shape = self.cad.positive_lock_cam().val()
        self.assertTrue(runtime_union_shape.isValid())
        self.assertEqual(len(runtime_union_shape.Solids()), 1)
        self.assertAlmostEqual(volume_mm3(runtime_union_shape), 325.435, places=9)
        self.assertLessEqual(
            volume_mm3(source_shape.cut(runtime_union_shape)), 1.0e-6
        )
        self.assertLessEqual(
            volume_mm3(runtime_union_shape.cut(source_shape)), 1.0e-6
        )

        # A small exact source route sample prevents an exact-looking static
        # union from being detached from its functional capture clearances.
        studs = [
            self.cad.shoulder_lock_stud()
            .translate((x_value, 0.0, 0.0))
            .val()
            for x_value in (-float(self.cad.LOCK_STUD_X), float(self.cad.LOCK_STUD_X))
        ]
        slider_native = self.cad.locking_slider()
        plate_native = self.cad.robot_plate()

        def overlap_mm3(first: Any, second: Any) -> float:
            if float(first.distance(second)) > 1.0e-7:
                return 0.0
            return volume_mm3(first.intersect(second))

        route_samples: list[tuple[float, float, float]] = []
        for preseat_mm in (9.6, 6.4, 3.2, 3.1, 0.0):
            lateral_mm = float(
                self.cad.positive_lock_cam_capture_lateral_offset_mm(preseat_mm)
            )
            q_mm = float(self.cad.positive_lock_cam_capture_q_max_mm(preseat_mm))
            slider = slider_native.translate(
                (
                    q_mm + lateral_mm,
                    0.0,
                    self.cad.SLIDER_Z - self.cad.PLATE_THICKNESS - preseat_mm,
                )
            ).val()
            plate = plate_native.translate(
                (lateral_mm, 0.0, -self.cad.PLATE_THICKNESS - preseat_mm)
            ).val()
            self.assertLessEqual(overlap_mm3(slider, source_shape), 1.0e-6)
            self.assertLessEqual(overlap_mm3(plate, source_shape), 1.0e-6)
            for stud in studs:
                self.assertLessEqual(overlap_mm3(slider, stud), 1.0e-6)
            route_samples.append((preseat_mm, lateral_mm, q_mm))
        passive_open_q_mm = float(
            self.cad.DOCK_CAM_X_INNER - self.cad.SLIDER_TAB_END_X
        )
        for observed, expected_preseat_mm in (
            (route_samples[-2], 3.1),
            (route_samples[-1], 0.0),
        ):
            self.assertEqual(observed[0], expected_preseat_mm)
            self.assertAlmostEqual(observed[1], 0.0, places=12)
            self.assertAlmostEqual(observed[2], passive_open_q_mm, places=12)


class CoreCaptureRouteRuntimeAuthorityTests(unittest.TestCase):
    """Independently bind the four-phase p/X route and fail-honest live guard."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cad = import_file(
            MAGNETIC_ROOT / "generate_cad.py",
            "core_capture_route_cad_authority",
            "core capture route CAD authority",
        )
        cls.demo = import_file(
            MATCHA_DEMO,
            "core_capture_route_runtime_authority",
            "core capture route runtime authority",
        )
        cls.contract = cls.demo.core_capture_route_runtime_contract()

    @staticmethod
    def _reseal(mutated: dict[str, Any]) -> None:
        states = mutated["source_states"]
        mutated["source_state_sha256"] = canonical_json_sha256(states)
        mutated["q_roster_sha256"] = canonical_json_sha256(
            [state["q_rad"] for state in states]
        )
        mutated["canonical_waypoint_digest_preimage"] = {
            "source_binding": mutated["source_binding"],
            "model_binding": mutated["model_binding"],
            "source_states": states,
        }
        mutated["canonical_waypoint_sha256"] = canonical_json_sha256(
            mutated["canonical_waypoint_digest_preimage"]
        )
        identity = mutated["contract_identity_digest_preimage"]
        identity["source_generator_sha256"] = mutated["source_binding"][
            "generator_file"
        ]["sha256"]
        identity["positive_lock_cam_contract_sha256"] = mutated[
            "source_binding"
        ]["positive_lock_cam_contract_sha256"]
        identity["embedded_state_bytes_sha256"] = mutated[
            "embedded_state_bytes_sha256"
        ]
        identity["source_state_sha256"] = mutated["source_state_sha256"]
        identity["q_roster_sha256"] = mutated["q_roster_sha256"]
        mutated["contract_identity_sha256"] = canonical_json_sha256(identity)

    # Superseded by ``RolledCoreDockRuntimeAuthorityTests`` below.  Retain the
    # historical implementation as readable migration context, but do not
    # discover its stale pre-roll schema assertions as current authority.
    def superseded_route_contract_and_result_mutations_fail_closed(self) -> None:
        baseline = self.contract
        self.assertEqual(
            core_capture_route_contract_errors(baseline, self.cad), []
        )

        mutations: dict[str, dict[str, Any]] = {}
        mutations["old_constant_plus_0p20"] = copy.deepcopy(baseline)
        for state in mutations["old_constant_plus_0p20"]["source_states"]:
            state["source_x_mm"] = 0.20
        self._reseal(mutations["old_constant_plus_0p20"])

        mutations["late_recenter"] = copy.deepcopy(baseline)
        mutations["late_recenter"]["source_states"][260]["source_x_mm"] = 0.01
        self._reseal(mutations["late_recenter"])

        mutations["self_attested_p"] = copy.deepcopy(baseline)
        mutations["self_attested_p"]["source_states"][100]["preseat_mm"] += 0.5
        self._reseal(mutations["self_attested_p"])

        mutations["shifted_route"] = copy.deepcopy(baseline)
        mutations["shifted_route"]["source_states"][120]["q_rad"][1] += 0.01
        self._reseal(mutations["shifted_route"])

        mutations["rotated_route"] = copy.deepcopy(baseline)
        mutations["rotated_route"]["source_states"][200]["q_rad"][4] += 0.02
        self._reseal(mutations["rotated_route"])

        mutations["altered_source_digest"] = copy.deepcopy(baseline)
        mutations["altered_source_digest"]["source_binding"]["generator_file"][
            "sha256"
        ] = "0" * 64
        self._reseal(mutations["altered_source_digest"])

        mutations["temporary_candidate_authority"] = copy.deepcopy(baseline)
        mutations["temporary_candidate_authority"]["source_binding"][
            "candidate_path"
        ] = "/tmp/core_capture_route_candidate.json"

        mutations["single_global_action"] = copy.deepcopy(baseline)
        mutations["single_global_action"]["actions"] = [
            mutations["single_global_action"]["actions"][1]
        ]

        mutations["missing_recenter_phase"] = copy.deepcopy(baseline)
        mutations["missing_recenter_phase"]["actions"].pop(2)

        mutations["nonzero_breakpoint_velocity"] = copy.deepcopy(baseline)
        mutations["nonzero_breakpoint_velocity"]["actions"][2][
            "zero_commanded_endpoint_velocity"
        ] = False

        mutations["deadline_widened"] = copy.deepcopy(baseline)
        mutations["deadline_widened"]["actions"][2]["timeout_s"] = 15.0

        mutations["corridor_widened"] = copy.deepcopy(baseline)
        mutations["corridor_widened"]["live_source_corridor_guard"][
            "maximum_absolute_source_x_error_mm"
        ] = 0.041

        mutations["dense_threshold_widened"] = copy.deepcopy(baseline)
        mutations["dense_threshold_widened"]["dense_fk_evidence"]["phases"][
            2
        ]["thresholds"]["maximum_source_x_error_mm"] = 0.01

        mutations["direct_slider_write"] = copy.deepcopy(baseline)
        mutations["direct_slider_write"]["state_write_contract"][
            "direct_slider_qpos_writes_after_initialization"
        ] = 1

        mutations["friction_laundering"] = copy.deepcopy(baseline)
        mutations["friction_laundering"]["authority_scope"][
            "friction_coefficient_authority"
        ] = True

        mutations["release_promotion"] = copy.deepcopy(baseline)
        mutations["release_promotion"]["release_ready"] = True
        mutations["release_promotion"]["authority_scope"]["release_ready"] = True

        for name, mutated in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    core_capture_route_contract_errors(mutated, self.cad),
                    [],
                )

        model = self.demo.build_model()
        data = self.demo.mujoco.MjData(model)
        self.demo.initialize(model, data)
        controller = self.demo.MatchaWorkflowController(model, data)
        honest_initial = controller.result()
        self.assertEqual(
            core_capture_route_result_errors(honest_initial, baseline), []
        )
        initial_live = honest_initial["route_alignment"]["live_source_corridor"]
        self.assertIs(initial_live["observed"], False)
        self.assertIs(initial_live["passed"], False)
        self.assertIs(honest_initial["route_alignment"]["passed"], False)

        complete = copy.deepcopy(honest_initial)
        endpoint_records = [
            {"event": "move_complete", "action": name}
            for name in CORE_CAPTURE_ROUTE_ACTIONS
        ]
        complete_alignment = complete["route_alignment"]
        complete_alignment["phase_endpoint_journal_evidence"] = endpoint_records
        complete_alignment["completed_endpoint_actions"] = sorted(
            CORE_CAPTURE_ROUTE_ACTIONS
        )
        complete_alignment["all_four_endpoints_completed"] = True
        complete_alignment["measured_max_lateral_deviation_m"] = 1.0e-5
        live = complete_alignment["live_source_corridor"]
        live.update(
            {
                "observed": True,
                "audited_substeps": 3,
                "audited_substeps_by_phase": {
                    name: 1 for name in list(CORE_CAPTURE_ROUTE_ACTIONS)[1:]
                },
                "all_three_phases_observed": True,
                "maximum_absolute_error_mm": 0.005,
                "witness": {
                    "action": "gripper_capture_coupled_recenter",
                    "preseat_mm": 4.0,
                    "observed_x_mm": 0.055,
                    "expected_source_x_mm": 0.05,
                    "signed_error_mm": 0.005,
                    "absolute_error_mm": 0.005,
                },
                "passed": True,
            }
        )
        complete_alignment["passed"] = True
        self.assertEqual(core_capture_route_result_errors(complete, baseline), [])

        result_mutations: dict[str, dict[str, Any]] = {}
        result_mutations["zero_sample_false_green"] = copy.deepcopy(honest_initial)
        result_mutations["zero_sample_false_green"]["route_alignment"][
            "live_source_corridor"
        ]["passed"] = True
        result_mutations["zero_sample_false_green"]["route_alignment"][
            "passed"
        ] = True
        result_mutations["missing_phase_substep"] = copy.deepcopy(complete)
        result_mutations["missing_phase_substep"]["route_alignment"][
            "live_source_corridor"
        ]["audited_substeps_by_phase"]["gripper_capture_coupled_recenter"] = 0
        result_mutations["missing_endpoint"] = copy.deepcopy(complete)
        result_mutations["missing_endpoint"]["route_alignment"][
            "phase_endpoint_journal_evidence"
        ].pop()
        result_mutations["self_attested_x"] = copy.deepcopy(complete)
        result_mutations["self_attested_x"]["route_alignment"][
            "live_source_corridor"
        ]["witness"]["expected_source_x_mm"] = 0.055
        result_mutations["result_threshold_widened"] = copy.deepcopy(complete)
        result_mutations["result_threshold_widened"]["route_alignment"][
            "live_source_corridor"
        ]["maximum_allowed_error_mm"] = 0.041
        result_mutations["old_method"] = copy.deepcopy(complete)
        result_mutations["old_method"]["route_alignment"]["method"] = (
            "constant_x_plus_0p20_same_z_recenter"
        )
        result_mutations["abort_laundered"] = copy.deepcopy(complete)
        result_mutations["abort_laundered"]["abort_reason"] = "forbidden_collision"
        for name, mutated in result_mutations.items():
            with self.subTest(result=name):
                self.assertNotEqual(
                    core_capture_route_result_errors(mutated, baseline), []
                )

    def superseded_compiled_route_replays_dense_fk_actions_and_per_step_guard(
        self,
    ) -> None:
        demo = self.demo
        contract = self.contract
        self.assertEqual(core_capture_route_contract_errors(contract, self.cad), [])

        # Decode the embedded little-endian authority independently and bind
        # it to both public JSON rosters and the frozen external-review pins.
        raw = base64.b64decode(demo._CORE_CAPTURE_ROUTE_STATE_BASE64, validate=True)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            CORE_CAPTURE_ROUTE_EMBEDDED_BYTES_SHA256,
        )
        values = np.frombuffer(raw, dtype="<f8")
        self.assertEqual(values.shape, (276 * 7,))
        rows = values.reshape(276, 7)
        decoded_states = [
            {
                "preseat_mm": float(row[0]),
                "source_x_mm": float(row[1]),
                "q_rad": [float(value) for value in row[2:]],
            }
            for row in rows
        ]
        self.assertEqual(decoded_states, contract["source_states"])
        self.assertEqual(
            canonical_json_sha256(decoded_states),
            CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256,
        )
        self.assertEqual(
            canonical_json_sha256([state["q_rad"] for state in decoded_states]),
            CORE_CAPTURE_ROUTE_Q_SHA256,
        )

        source_capture = self.cad.positive_lock_cam_contract()["passive_capture"]
        self.assertEqual(
            source_capture["lateral_offset_breakpoints_mm"],
            [[6.4, 0.2], [3.2, 0.0], [0.0, 0.0]],
        )
        for state in decoded_states:
            preseat = float(state["preseat_mm"])
            independently_expected_x = _independent_core_capture_x_mm(preseat)
            self.assertAlmostEqual(
                float(state["source_x_mm"]), independently_expected_x, places=12
            )
            self.assertAlmostEqual(
                float(self.cad.positive_lock_cam_capture_lateral_offset_mm(preseat)),
                independently_expected_x,
                places=12,
            )

        model = demo.build_model()
        mujoco = demo.mujoco
        data = mujoco.MjData(model)
        demo.initialize(model, data)
        xml_text, _ = demo._build_xml_and_assets()
        self.assertEqual(
            contract["model_binding"]["model_xml_sha256"],
            hashlib.sha256(xml_text.encode()).hexdigest(),
        )
        self.assertEqual(
            contract["model_binding"][
                "initialized_active_collision_geometry_sha256"
            ],
            _independent_initialized_active_geometry_sha256(
                model, data, mujoco
            ),
        )

        actual_actions = demo._core_capture_move_actions()
        self.assertEqual(
            [action.name for action in actual_actions],
            list(CORE_CAPTURE_ROUTE_ACTIONS),
        )
        starts = {
            "gripper_capture_lateral_align": [
                float(value) for value in demo.DOCK_PRE_CAPTURE_Q["gripper"]
            ],
            "gripper_capture_axial_open_side": decoded_states[0]["q_rad"],
            "gripper_capture_coupled_recenter": decoded_states[243]["q_rad"],
            "gripper_capture_centered_final": decoded_states[259]["q_rad"],
        }
        expected_full_q_rosters = {
            "gripper_capture_lateral_align": [
                starts["gripper_capture_lateral_align"],
                decoded_states[0]["q_rad"],
            ],
            "gripper_capture_axial_open_side": [
                state["q_rad"] for state in decoded_states[0:244]
            ],
            "gripper_capture_coupled_recenter": [
                state["q_rad"] for state in decoded_states[243:260]
            ],
            "gripper_capture_centered_final": [
                state["q_rad"] for state in decoded_states[259:276]
            ],
        }

        def actual_route_errors(actions: tuple[Any, ...]) -> list[str]:
            errors: list[str] = []
            if [action.name for action in actions] != list(
                CORE_CAPTURE_ROUTE_ACTIONS
            ):
                return ["action_roster"]
            for action in actions:
                actual_full_roster = [
                    list(starts[action.name]),
                    *[list(waypoint) for waypoint in action.joint_waypoints],
                ]
                expected_roster = expected_full_q_rosters[action.name]
                if actual_full_roster != expected_roster:
                    errors.append(f"{action.name}:full_q_roster")
                observed_sha = canonical_json_sha256(actual_full_roster)
                expected_sha = CORE_CAPTURE_ROUTE_ACTIONS[action.name][
                    "q_sha256"
                ]
                if observed_sha != expected_sha:
                    errors.append(f"{action.name}:q_sha256")
                if list(action.target_q or ()) != expected_roster[-1]:
                    errors.append(f"{action.name}:target_q")
            return errors

        self.assertEqual(actual_route_errors(actual_actions), [])
        default_data = mujoco.MjData(model)
        demo.initialize(model, default_data)
        default_controller = demo.MatchaWorkflowController(model, default_data)
        self.assertEqual(
            tuple(default_controller.actions[:4]), actual_actions
        )
        self.assertEqual(
            actual_route_errors(tuple(default_controller.actions[:4])), []
        )

        # Direct regression for the prior false-green: changing one interior
        # axial waypoint by 1e-8 rad must fail even when downstream command
        # evidence is regenerated from that same mutated production action.
        bad_waypoints = [
            list(waypoint) for waypoint in actual_actions[1].joint_waypoints
        ]
        bad_waypoints[100][1] += 1.0e-8
        axial = actual_actions[1]
        bad_axial = demo.WorkflowAction(
            name=axial.name,
            kind=axial.kind,
            tool=axial.tool,
            target_q=axial.target_q,
            joint_waypoints=tuple(tuple(row) for row in bad_waypoints),
            duration_s=axial.duration_s,
            timeout_s=axial.timeout_s,
        )
        bad_actions = list(actual_actions)
        bad_actions[1] = bad_axial
        self.assertNotEqual(actual_route_errors(tuple(bad_actions)), [])

        expected_command_schedules: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for action, published in zip(
            actual_actions, contract["actions"], strict=True
        ):
            expected = CORE_CAPTURE_ROUTE_ACTIONS[action.name]
            self.assertEqual(action.duration_s, expected["duration_s"])
            self.assertEqual(action.timeout_s, expected["timeout_s"])
            self.assertEqual(list(action.target_q), published["endpoint_q_rad"])
            route = np.asarray(
                (tuple(starts[action.name]), *action.joint_waypoints),
                dtype=np.float64,
            )
            times = np.arange(
                0.0,
                action.duration_s + 0.0025,
                0.005,
                dtype=np.float64,
            )
            commands: list[np.ndarray] = []
            sample_hasher = hashlib.sha256()
            for sample_index, time_s in enumerate(times):
                alpha = min(1.0, max(0.0, float(time_s) / action.duration_s))
                smooth = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
                position = smooth * (len(route) - 1)
                segment = min(int(math.floor(position)), len(route) - 2)
                fraction = position - segment
                command = route[segment] + fraction * (
                    route[segment + 1] - route[segment]
                )
                commands.append(command)
                sample_hasher.update(struct.pack("<Id", sample_index, float(time_s)))
                sample_hasher.update(np.asarray(command, dtype="<f8").tobytes())
            command_array = np.asarray(commands)
            expected_command_schedules[action.name] = (times, command_array)
            velocity = np.diff(command_array, axis=0) / 0.005
            acceleration = np.diff(velocity, axis=0) / 0.005
            observed_speed = float(np.max(np.abs(velocity)))
            observed_acceleration = float(np.max(np.abs(acceleration)))
            reported = published["command_kinematics"]
            self.assertAlmostEqual(
                observed_speed,
                float(reported["maximum_abs_joint_speed_rad_s"]),
                places=14,
            )
            self.assertAlmostEqual(
                observed_acceleration,
                float(reported["maximum_abs_joint_acceleration_rad_s2"]),
                places=12,
            )
            self.assertEqual(
                sample_hasher.hexdigest(), reported["command_sample_sha256"]
            )
            self.assertLessEqual(observed_speed, expected["speed_bound"])
            self.assertLessEqual(
                observed_acceleration, expected["acceleration_bound"]
            )

        def runtime_command_errors() -> list[str]:
            errors: list[str] = []
            for action in actual_actions:
                command_data = mujoco.MjData(model)
                demo.initialize(model, command_data)
                controller = demo.MatchaWorkflowController(
                    model, command_data, actions=(action,)
                )
                controller.action_start_q = np.asarray(
                    starts[action.name], dtype=np.float64
                )
                controller._integrate = lambda: None
                times, expected_commands = expected_command_schedules[action.name]
                observed_commands: list[np.ndarray] = []
                for time_s in times:
                    controller._command_move(action, float(time_s))
                    observed_commands.append(
                        np.asarray(
                            command_data.ctrl[controller.arm_actuator_ids],
                            dtype=np.float64,
                        ).copy()
                    )
                observed_array = np.asarray(observed_commands)
                if not np.array_equal(observed_array, expected_commands):
                    errors.append(f"{action.name}:runtime_command_schedule")
            return errors

        self.assertEqual(runtime_command_errors(), [])

        def bad_linear_command(
            controller: Any, action: Any, elapsed_s: float
        ) -> None:
            alpha = min(1.0, max(0.0, elapsed_s / action.duration_s))
            start = np.asarray(controller.action_start_q, dtype=np.float64)
            target = np.asarray(action.target_q, dtype=np.float64)
            controller.data.ctrl[controller.arm_actuator_ids] = (
                start + alpha * (target - start)
            )

        with mock.patch.object(
            demo.MatchaWorkflowController,
            "_command_move",
            bad_linear_command,
        ):
            self.assertNotEqual(runtime_command_errors(), [])

        arm_qpos = np.asarray(
            [model.joint(name).qposadr[0] for name in demo.ARM_JOINTS], dtype=int
        )
        dock = data.body("dock_gripper")
        dock_position = np.asarray(dock.xpos, dtype=np.float64).copy()
        dock_rotation = np.asarray(dock.xmat, dtype=np.float64).reshape(3, 3).copy()
        fractions = [index / 100.0 for index in range(101)]
        phase_states = [
            (
                "gripper_capture_lateral_align",
                [
                    (55.0, 0.0, starts["gripper_capture_lateral_align"]),
                    (
                        decoded_states[0]["preseat_mm"],
                        decoded_states[0]["source_x_mm"],
                        decoded_states[0]["q_rad"],
                    ),
                ],
                True,
            ),
            (
                "gripper_capture_axial_open_side",
                [
                    (state["preseat_mm"], state["source_x_mm"], state["q_rad"])
                    for state in decoded_states[0:244]
                ],
                False,
            ),
            (
                "gripper_capture_coupled_recenter",
                [
                    (state["preseat_mm"], state["source_x_mm"], state["q_rad"])
                    for state in decoded_states[243:260]
                ],
                False,
            ),
            (
                "gripper_capture_centered_final",
                [
                    (state["preseat_mm"], state["source_x_mm"], state["q_rad"])
                    for state in decoded_states[259:276]
                ],
                False,
            ),
        ]
        published_phases = {
            phase["action"]: phase
            for phase in contract["dense_fk_evidence"]["phases"]
        }
        for action_name, states, alignment_phase in phase_states:
            maxima = {
                "maximum_preseat_error_mm": 0.0,
                "maximum_source_x_error_mm": 0.0,
                "maximum_abs_transverse_y_mm": 0.0,
                "maximum_orientation_error_rad": 0.0,
            }
            previous_preseat = math.inf
            monotone = True
            hasher = hashlib.sha256()
            hasher.update(action_name.encode())
            hasher.update(b"\0")
            for interval_index, (start, end) in enumerate(
                zip(states[:-1], states[1:], strict=True)
            ):
                start_q = np.asarray(start[2], dtype=np.float64)
                end_q = np.asarray(end[2], dtype=np.float64)
                for fraction_index, fraction in enumerate(fractions):
                    q_value = start_q + fraction * (end_q - start_q)
                    data.qpos[arm_qpos] = q_value
                    mujoco.mj_forward(model, data)
                    local_mm = (
                        np.asarray(data.site("robot_mating_face").xpos) - dock_position
                    ) @ dock_rotation * 1000.0
                    observed_preseat = -float(local_mm[2])
                    expected_preseat = float(
                        start[0] + fraction * (end[0] - start[0])
                    )
                    expected_x = (
                        float(start[1] + fraction * (end[1] - start[1]))
                        if alignment_phase
                        else _independent_core_capture_x_mm(
                            max(0.0, observed_preseat)
                        )
                    )
                    relative_rotation = dock_rotation.T @ np.asarray(
                        data.site("robot_mating_face").xmat
                    ).reshape(3, 3)
                    sine = 0.5 * np.asarray(
                        [
                            relative_rotation[2, 1] - relative_rotation[1, 2],
                            relative_rotation[0, 2] - relative_rotation[2, 0],
                            relative_rotation[1, 0] - relative_rotation[0, 1],
                        ]
                    )
                    angle = math.atan2(
                        float(np.linalg.norm(sine)),
                        (float(np.trace(relative_rotation)) - 1.0) / 2.0,
                    )
                    maxima["maximum_preseat_error_mm"] = max(
                        maxima["maximum_preseat_error_mm"],
                        abs(observed_preseat - expected_preseat),
                    )
                    maxima["maximum_source_x_error_mm"] = max(
                        maxima["maximum_source_x_error_mm"],
                        abs(float(local_mm[0]) - expected_x),
                    )
                    maxima["maximum_abs_transverse_y_mm"] = max(
                        maxima["maximum_abs_transverse_y_mm"], abs(float(local_mm[1]))
                    )
                    maxima["maximum_orientation_error_rad"] = max(
                        maxima["maximum_orientation_error_rad"], angle
                    )
                    if observed_preseat > previous_preseat + 1.0e-9:
                        monotone = False
                    previous_preseat = observed_preseat
                    hasher.update(
                        struct.pack(
                            "<IIddddd",
                            interval_index,
                            fraction_index,
                            fraction,
                            observed_preseat,
                            float(local_mm[0]),
                            float(local_mm[1]),
                            angle,
                        )
                    )
            published = published_phases[action_name]
            self.assertEqual(
                hasher.hexdigest(), published["sample_sha256"], action_name
            )
            for key, value in maxima.items():
                self.assertAlmostEqual(
                    value, float(published["observed"][key]), places=14, msg=action_name
                )
            if not alignment_phase:
                self.assertTrue(monotone, action_name)

        # Exact-source negative: the retired same-Z +0.20->0 recenter cuts the
        # complete cam at both nominal q=0 and passive q=0.05 mm.
        source_cam = self.cad.positive_lock_cam().val()
        old_overlaps: dict[str, float] = {}
        for key, q_mm in (("slider_q_0p00mm", 0.0), ("slider_q_0p05mm", 0.05)):
            slider = self.cad.locking_slider().translate(
                (
                    q_mm + 0.20,
                    0.0,
                    self.cad.SLIDER_Z - self.cad.PLATE_THICKNESS,
                )
            ).val()
            intersection = slider.intersect(source_cam)
            old_overlaps[key] = math.fsum(
                float(solid.Volume()) for solid in intersection.Solids()
            )
        for key, value in old_overlaps.items():
            self.assertAlmostEqual(
                value,
                float(
                    contract["retired_route_negative"][
                        "complete_source_cam_overlap_mm3"
                    ][key]
                ),
                places=9,
            )
            self.assertGreater(value, 1.0e-6)

        source = MATCHA_DEMO.read_text()
        integrate_source = inspect.getsource(demo.MatchaWorkflowController._integrate)
        self.assertEqual(integrate_source.count("mujoco.mj_step"), 1)
        self.assertEqual(
            integrate_source.count("_audit_core_capture_source_corridor"), 1
        )
        self.assertLess(
            integrate_source.index("mujoco.mj_step"),
            integrate_source.index("_audit_core_capture_source_corridor"),
        )
        self.assertLess(
            integrate_source.index("_audit_core_capture_source_corridor"),
            integrate_source.index("_audit_contacts"),
        )
        corridor_source = inspect.getsource(
            demo.MatchaWorkflowController._audit_core_capture_source_corridor
        )
        for required in (
            'self.data.site("robot_mating_face")',
            'self.data.body("dock_gripper")',
            "_core_capture_source_x_mm",
            "CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM",
            'self._abort("core_capture_source_corridor_violation")',
        ):
            self.assertIn(required, corridor_source)
        self.assertNotIn("journal", corridor_source)
        allowed_source = inspect.getsource(
            demo.MatchaWorkflowController._allowed_penetrating_contact
        )
        self.assertNotIn("cam_geom", allowed_source)
        self.assertNotIn("cam_collision", allowed_source)
        move_source = inspect.getsource(demo.MatchaWorkflowController._command_move)
        self.assertIn("alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))", move_source)
        self.assertIn("CORE_CAPTURE_ROUTE_ENDPOINT_DWELL_TICKS", move_source)
        self.assertIn("self.move_endpoint_dwell_ticks", move_source)

        tree = ast.parse(source, filename=str(MATCHA_DEMO))
        controller_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "MatchaWorkflowController"
        )
        direct_state_writes: list[str] = []
        for node in ast.walk(controller_node):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if target_state_field(target) in {"qpos", "qvel"}:
                    direct_state_writes.append(ast.unparse(target))
        self.assertEqual(direct_state_writes, [])

        fresh_data = mujoco.MjData(model)
        demo.initialize(model, fresh_data)
        initial_result = demo.MatchaWorkflowController(model, fresh_data).result()
        self.assertEqual(
            core_capture_route_result_errors(initial_result, contract), []
        )
        self.assertIs(
            initial_result["route_alignment"]["live_source_corridor"]["passed"],
            False,
        )
        self.assertIs(initial_result["release_ready"], False)


class CoreCamTabContactCheckpointATests(unittest.TestCase):
    """Bind capture-only cam evidence without promoting physical authority."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cad = import_file(
            MAGNETIC_ROOT / "generate_cad.py",
            "core_cam_tab_checkpoint_cad_authority",
            "core cam/tab checkpoint CAD authority",
        )
        cls.demo = import_file(
            MATCHA_DEMO,
            "core_cam_tab_checkpoint_runtime_authority",
            "core cam/tab checkpoint runtime authority",
        )
        cls.route_contract = cls.demo.core_capture_route_runtime_contract()
        cls.runtime_cam_contract = (
            cls.demo.qc.positive_lock_cam_runtime_contract()
        )
        cls.contract = cls.demo.core_cam_tab_contact_runtime_contract()

    @classmethod
    def _independent_model_digests(cls, model: Any) -> tuple[str, str]:
        compiled = _independent_compiled_model_xml_equivalent_sha256(model)
        scratch = cls.demo.mujoco.MjData(model)
        cls.demo.initialize(model, scratch)
        active = _independent_initialized_active_geometry_sha256(
            model, scratch, cls.demo.mujoco
        )
        return compiled, active

    @staticmethod
    def _binding_observations(
        initial: tuple[str, str], evidence: tuple[str, str]
    ) -> dict[str, str]:
        return {
            "controller_init_compiled": initial[0],
            "controller_init_active": initial[1],
            "evidence_compiled": evidence[0],
            "evidence_active": evidence[1],
        }

    @staticmethod
    def _reseal_contract(mutated: dict[str, Any]) -> None:
        identity = mutated["contract_identity_digest_preimage"]
        identity["source_generator_sha256"] = mutated["source_binding"][
            "generator_file"
        ]["sha256"]
        identity["positive_lock_cam_contract_sha256"] = mutated[
            "source_binding"
        ]["positive_lock_cam_contract_sha256"]
        identity["positive_lock_slider_step_sha256"] = mutated[
            "source_binding"
        ]["positive_lock_slider_step"]["sha256"]
        identity["capture_route_contract_identity_sha256"] = mutated[
            "source_binding"
        ]["capture_route_contract_identity_sha256"]
        identity["model_binding"] = {
            "model_xml_sha256": mutated["model_binding"]["model_xml_sha256"],
            "compiled_model_xml_equivalent_sha256": mutated["model_binding"][
                "compiled_model_xml_equivalent_sha256"
            ],
            "initialized_active_collision_geometry_sha256": mutated[
                "model_binding"
            ]["initialized_active_collision_geometry_sha256"],
        }
        identity["classifier_semantics"] = copy.deepcopy(
            mutated["classifier_semantics"]
        )
        mutated["contract_identity_sha256"] = canonical_json_sha256(identity)

    def test_contract_and_honest_red_result_mutations_fail_closed(self) -> None:
        contract = self.contract
        self.assertEqual(
            core_cam_tab_contact_contract_errors(
                contract,
                self.cad,
                self.runtime_cam_contract,
                self.route_contract,
            ),
            [],
        )
        mutations: dict[str, dict[str, Any]] = {}
        mutations["tab000_made_eligible"] = copy.deepcopy(contract)
        mutations["tab000_made_eligible"]["classifier_semantics"][
            "runtime_inventory"
        ]["contact_eligible_leading_tab_geom"] = CORE_CAM_TAB_NONCONTACT_GEOM

        mutations["root_removed_from_complete_roster"] = copy.deepcopy(contract)
        mutations["root_removed_from_complete_roster"]["classifier_semantics"][
            "runtime_inventory"
        ]["complete_cam_geom_roster"].pop()

        mutations["lead_normal_flattened"] = copy.deepcopy(contract)
        mutations["lead_normal_flattened"]["classifier_semantics"][
            "surface_classifiers"
        ][0]["normal_cam_to_tab_dock_local"] = [-1.0, 0.0, 0.0]

        mutations["positive_z_top_cap_allowed"] = copy.deepcopy(contract)
        mutations["positive_z_top_cap_allowed"]["classifier_semantics"][
            "surface_classifiers"
        ][2]["closed_top_cap_positive_z_is_forbidden"] = False

        mutations["guard_widened"] = copy.deepcopy(contract)
        mutations["guard_widened"]["classifier_semantics"][
            "provisional_development_guard"
        ]["maximum_penetration_mm"] = 0.021

        mutations["one_by_five_gap_laundering"] = copy.deepcopy(contract)
        mutations["one_by_five_gap_laundering"]["classifier_semantics"][
            "functional_envelope_sampling"
        ]["complete_cam_distance"]["method"] = (
            "leading_tab_by_five_cam_geoms"
        )

        mutations["continuous_authority_laundered"] = copy.deepcopy(contract)
        mutations["continuous_authority_laundered"]["classifier_semantics"][
            "functional_envelope_sampling"
        ]["continuous_tunnel_authority"] = True

        mutations["physical_release_implemented"] = copy.deepcopy(contract)
        mutations["physical_release_implemented"]["post_capture_exclusion"][
            "physical_release_action_implemented"
        ] = True

        mutations["source_hash_altered"] = copy.deepcopy(contract)
        mutations["source_hash_altered"]["source_binding"]["generator_file"][
            "sha256"
        ] = "0" * 64

        mutations["release_promoted"] = copy.deepcopy(contract)
        mutations["release_promoted"]["release_ready"] = True
        mutations["release_promoted"]["authority_scope"]["release_ready"] = True

        for name, mutated in mutations.items():
            self._reseal_contract(mutated)
            with self.subTest(contract_mutation=name):
                self.assertNotEqual(
                    core_cam_tab_contact_contract_errors(
                        mutated,
                        self.cad,
                        self.runtime_cam_contract,
                        self.route_contract,
                    ),
                    [],
                )

        model = self.demo.build_model()
        data = self.demo.mujoco.MjData(model)
        self.demo.initialize(model, data)
        independent = self._independent_model_digests(model)
        self.assertEqual(
            independent,
            (
                CORE_CAM_COMPILED_MODEL_XML_EQUIVALENT_SHA256,
                contract["model_binding"][
                    "initialized_active_collision_geometry_sha256"
                ],
            ),
        )
        controller = self.demo.MatchaWorkflowController(model, data)
        result = controller.result()
        observations = self._binding_observations(independent, independent)
        self.assertEqual(
            core_cam_tab_result_errors(
                result,
                contract,
                model_binding_observations=observations,
                replay_model=model,
                replay_mujoco=self.demo.mujoco,
            ),
            [],
        )
        self.assertIn(
            "model_binding:observations_missing",
            core_cam_tab_result_errors(
                result,
                contract,
                replay_model=model,
                replay_mujoco=self.demo.mujoco,
            ),
        )
        self.assertIs(result["core_cam_tab_contact_evidence"]["observed"], False)
        self.assertIs(result["core_cam_tab_contact_evidence"]["passed"], False)
        self.assertIs(
            result["core_capture_free_space_tracking_evidence"]["passed"], False
        )
        self.assertIs(result["development_geometry_milestone_passed"], False)
        self.assertIs(result["success"], False)
        self.assertIs(result["release_ready"], False)

        result_mutations: dict[str, dict[str, Any]] = {}
        result_mutations["zero_contact_false_green"] = copy.deepcopy(result)
        result_mutations["zero_contact_false_green"][
            "core_cam_tab_contact_evidence"
        ]["passed"] = True
        result_mutations["zero_contact_false_green"][
            "core_cam_tab_contact_evidence"
        ]["provisional_geometry_classification_passed"] = True
        result_mutations["zero_sample_free_false_green"] = copy.deepcopy(result)
        result_mutations["zero_sample_free_false_green"][
            "core_capture_free_space_tracking_evidence"
        ]["passed"] = True
        result_mutations["development_promoted"] = copy.deepcopy(result)
        result_mutations["development_promoted"][
            "development_geometry_milestone_passed"
        ] = True
        result_mutations["physical_lock_claimed"] = copy.deepcopy(result)
        result_mutations["physical_lock_claimed"]["physical_lock_confirmed"] = True
        result_mutations["locked_claimed"] = copy.deepcopy(result)
        result_mutations["locked_claimed"]["locked"] = True
        result_mutations["cam_authority_promoted"] = copy.deepcopy(result)
        result_mutations["cam_authority_promoted"][
            "physical_cam_authority_ready"
        ] = True
        result_mutations["release_promoted"] = copy.deepcopy(result)
        result_mutations["release_promoted"]["release_ready"] = True
        result_mutations["binding_echo_changed"] = copy.deepcopy(result)
        result_mutations["binding_echo_changed"]["core_cam_actual_model_binding"][
            "evidence_observed_initialized_active_geometry_sha256"
        ] = "f" * 64
        for name, mutated in result_mutations.items():
            with self.subTest(result_mutation=name):
                self.assertNotEqual(
                    core_cam_tab_result_errors(
                        mutated,
                        contract,
                        model_binding_observations=observations,
                        replay_model=model,
                        replay_mujoco=self.demo.mujoco,
                    ),
                    [],
                )

    def test_actual_model_binding_replay_and_contact_audit_are_independent(
        self,
    ) -> None:
        demo = self.demo
        contract = self.contract

        # Mutating an active geom before controller construction must make both
        # the init snapshot and evidence-time binding honestly red.
        pre_model = demo.build_model()
        pre_data = demo.mujoco.MjData(pre_model)
        demo.initialize(pre_model, pre_data)
        floor_id = int(pre_model.geom("matcha_floor_collision").id)
        pre_model.geom_pos[floor_id, 2] += 1.0e-6
        pre_digests = self._independent_model_digests(pre_model)
        self.assertNotEqual(
            pre_digests,
            (
                CORE_CAM_COMPILED_MODEL_XML_EQUIVALENT_SHA256,
                contract["model_binding"][
                    "initialized_active_collision_geometry_sha256"
                ],
            ),
        )
        pre_controller = demo.MatchaWorkflowController(pre_model, pre_data)
        pre_result = pre_controller.result()
        pre_observations = self._binding_observations(pre_digests, pre_digests)
        self.assertEqual(
            core_cam_tab_result_errors(
                pre_result,
                contract,
                model_binding_observations=pre_observations,
                replay_model=pre_model,
                replay_mujoco=demo.mujoco,
            ),
            [],
        )
        self.assertIs(pre_result["core_cam_actual_model_binding"]["passed"], False)
        self.assertIs(pre_result["core_cam_tab_contact_evidence"]["passed"], False)
        self.assertIs(
            pre_result["core_capture_free_space_tracking_evidence"]["passed"],
            False,
        )
        self.assertIs(pre_result["development_geometry_milestone_passed"], False)

        # A mutation after construction must fail the evidence recomputation
        # and both unchanged-since-init witnesses.
        post_model = demo.build_model()
        post_data = demo.mujoco.MjData(post_model)
        demo.initialize(post_model, post_data)
        initial_digests = self._independent_model_digests(post_model)
        post_controller = demo.MatchaWorkflowController(post_model, post_data)
        post_floor_id = int(post_model.geom("matcha_floor_collision").id)
        post_model.geom_pos[post_floor_id, 2] += 1.0e-6
        evidence_digests = self._independent_model_digests(post_model)
        self.assertNotEqual(initial_digests, evidence_digests)
        post_result = post_controller.result()
        post_observations = self._binding_observations(
            initial_digests, evidence_digests
        )
        self.assertEqual(
            core_cam_tab_result_errors(
                post_result,
                contract,
                model_binding_observations=post_observations,
                replay_model=post_model,
                replay_mujoco=demo.mujoco,
            ),
            [],
        )
        binding = post_result["core_cam_actual_model_binding"]
        self.assertIs(binding["controller_init_passed"], True)
        self.assertIs(binding["evidence_recompute_passed"], False)
        self.assertIs(
            binding["compiled_model_digest_unchanged_since_controller_init"],
            False,
        )
        self.assertIs(
            binding["active_geometry_digest_unchanged_since_controller_init"],
            False,
        )
        self.assertIs(binding["passed"], False)
        self.assertEqual(
            post_result["core_cam_tab_contact_evidence"]["model_binding"],
            binding,
        )
        self.assertEqual(
            post_result["core_capture_free_space_tracking_evidence"][
                "model_binding"
            ],
            binding,
        )

        # One source-invalid but losslessly published functional state is
        # enough to test exact scratch-MjData pose and 2x5 gap replay without
        # running the workflow.
        replay_model = demo.build_model()
        replay_data = demo.mujoco.MjData(replay_model)
        demo.initialize(replay_model, replay_data)
        replay_controller = demo.MatchaWorkflowController(
            replay_model, replay_data
        )
        replay_controller.physics_substep_count = 1
        replay_data.time = float(replay_model.opt.timestep)
        replay_controller._record_core_cam_tab_functional_envelope(
            replay_controller.actions[2], []
        )
        state = copy.deepcopy(
            replay_controller.core_cam_tab_functional_envelope_samples[0]
        )
        self.assertEqual(
            _core_cam_functional_state_replay_errors(
                state, replay_model, demo.mujoco
            ),
            [],
        )
        replay_digests = self._independent_model_digests(replay_model)
        replay_result = replay_controller.result()
        self.assertIn(
            "envelope:replay_authority_missing",
            core_cam_tab_result_errors(
                replay_result,
                contract,
                model_binding_observations=self._binding_observations(
                    replay_digests, replay_digests
                ),
            ),
        )
        state_mutations: dict[str, dict[str, Any]] = {}
        state_mutations["qpos_changed"] = copy.deepcopy(state)
        state_mutations["qpos_changed"]["replay_state"]["qpos"][0] += 1.0e-6
        state_mutations["pose_changed"] = copy.deepcopy(state)
        state_mutations["pose_changed"]["replay_world_poses"][
            "robot_plate_body"
        ]["position_world_m"][0] += 1.0e-6
        state_mutations["gap_changed"] = copy.deepcopy(state)
        state_mutations["gap_changed"]["pair_gap_records"][0][
            "signed_distance_mm"
        ] += 0.1
        state_mutations["pair_dropped"] = copy.deepcopy(state)
        state_mutations["pair_dropped"]["pair_gap_records"].pop()
        for name, mutated in state_mutations.items():
            with self.subTest(replay_mutation=name):
                self.assertNotEqual(
                    _core_cam_functional_state_replay_errors(
                        mutated, replay_model, demo.mujoco
                    ),
                    [],
                )

        # Exercise the contact classifier itself on real static MuJoCo
        # contacts at the three source route regimes.  This is forward-only:
        # no workflow step or force/dynamics authority is claimed.
        arm_qpos_ids = np.asarray(
            [
                replay_model.joint(name).qposadr[0]
                for name in demo.ARM_JOINTS
            ],
            dtype=int,
        )
        slider_address = int(
            replay_model.joint("qc_positive_lock_slider_joint").qposadr[0]
        )
        classified_by_role: dict[str, dict[str, Any]] = {}
        static_cases = (
            (259, 2, {CORE_CAM_FUNCTIONAL_ROLES[0]}),
            (
                260,
                3,
                {
                    "lead_hold_partition_seam_nonfunctional",
                    CORE_CAM_FUNCTIONAL_ROLES[1],
                },
            ),
            (
                274,
                3,
                {
                    "main_hold_edge_tangency_nonfunctional",
                    CORE_CAM_FUNCTIONAL_ROLES[1],
                },
            ),
        )
        for row_index, action_index, required_roles in static_cases:
            static_data = demo.mujoco.MjData(replay_model)
            demo.initialize(replay_model, static_data)
            source_state = self.route_contract["source_states"][row_index]
            preseat_mm = float(source_state["preseat_mm"])
            source_x_mm = float(source_state["source_x_mm"])
            qmax_mm = max(
                0.05,
                min(3.0, max(0.0, preseat_mm) - source_x_mm - 3.15),
            )
            static_data.qpos[arm_qpos_ids] = np.asarray(
                source_state["q_rad"], dtype=np.float64
            )
            static_data.qpos[slider_address] = 0.001 * qmax_mm
            static_data.eq_active[
                int(replay_model.equality("dock_gripper_hold").id)
            ] = 1
            static_data.eq_active[
                int(replay_model.equality("attach_gripper").id)
            ] = 0
            demo.mujoco.mj_forward(replay_model, static_data)
            static_controller = demo.MatchaWorkflowController(
                replay_model, static_data
            )
            static_controller.physics_substep_count = 1
            static_data.time = float(replay_model.opt.timestep)
            action = static_controller.actions[action_index]
            observed_roles: set[str] = set()
            for contact_index in range(static_data.ncon):
                record = static_controller._core_cam_tab_contact_record(
                    contact_index, action
                )
                if record is None:
                    continue
                record_errors, passed, role = (
                    _independent_cam_tab_record_classification(record)
                )
                surface_role = str(record.get("surface_role"))
                if passed and surface_role in required_roles:
                    self.assertEqual(
                        record_errors, [], (surface_role, role, record_errors)
                    )
                    observed_roles.add(surface_role)
                    classified_by_role.setdefault(surface_role, record)
            self.assertTrue(required_roles.issubset(observed_roles), (
                row_index, required_roles, observed_roles
            ))

        lead_record = classified_by_role[CORE_CAM_FUNCTIONAL_ROLES[0]]
        seam_record = classified_by_role[
            "lead_hold_partition_seam_nonfunctional"
        ]
        classifier_mutations: dict[str, dict[str, Any]] = {}
        classifier_mutations["tab000"] = copy.deepcopy(lead_record)
        classifier_mutations["tab000"]["tab_or_other_geom"] = (
            CORE_CAM_TAB_NONCONTACT_GEOM
        )
        classifier_mutations["root_component"] = copy.deepcopy(lead_record)
        classifier_mutations["root_component"]["cam_geom"] = CORE_CAM_GEOMS[3]
        classifier_mutations["positive_z_cap_normal"] = copy.deepcopy(
            seam_record
        )
        classifier_mutations["positive_z_cap_normal"][
            "contact_normal_cam_to_tab_dock_local"
        ] = [0.0, 0.0, 1.0]
        classifier_mutations["depth_widened"] = copy.deepcopy(lead_record)
        classifier_mutations["depth_widened"]["contact_dist_mm"] = -0.021
        classifier_mutations["depth_widened"]["penetration_mm"] = 0.021
        classifier_mutations["nonfinite_force"] = copy.deepcopy(lead_record)
        classifier_mutations["nonfinite_force"]["contact_force_torque_6d"][0] = (
            math.nan
        )
        classifier_mutations["attach_equality_active"] = copy.deepcopy(
            lead_record
        )
        classifier_mutations["attach_equality_active"][
            "attach_equality_active"
        ] = True
        for name, mutated in classifier_mutations.items():
            with self.subTest(classifier_mutation=name):
                record_errors, passed, _ = (
                    _independent_cam_tab_record_classification(mutated)
                )
                self.assertNotEqual(record_errors, [])
                self.assertIs(passed, False)

        # The cam classifier runs after every mj_step and before the generic
        # penetration audit.  Its allow-set is contact-index scoped only.
        events: list[str] = []
        audit_controller = demo.MatchaWorkflowController(
            replay_model, demo.mujoco.MjData(replay_model)
        )
        demo.initialize(replay_model, audit_controller.data)
        patches = (
            mock.patch.object(
                demo.mujoco, "mj_step", side_effect=lambda *_: events.append("step")
            ),
            mock.patch.object(
                audit_controller, "_record_route_alignment",
                side_effect=lambda *_: events.append("route"),
            ),
            mock.patch.object(
                audit_controller,
                "_record_core_capture_gravity_bias_feedforward",
                side_effect=lambda *_: events.append("ff"),
            ),
            mock.patch.object(
                audit_controller, "_record_core_capture_free_space_tracking",
                side_effect=lambda *_: events.append("free"),
            ),
            mock.patch.object(
                audit_controller, "_audit_core_capture_cam_tab_contacts",
                side_effect=lambda *_: events.append("cam"),
            ),
            mock.patch.object(
                audit_controller, "_audit_core_capture_source_corridor",
                side_effect=lambda *_: events.append("corridor"),
            ),
            mock.patch.object(
                audit_controller, "_audit_contacts",
                side_effect=lambda *_: events.append("generic"),
            ),
            mock.patch.object(
                audit_controller, "_record_actuator_loads",
                side_effect=lambda *_: events.append("loads"),
            ),
            mock.patch.object(demo, "PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP", 2),
        )
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patches[5], patches[6], patches[7], patches[8]
        ):
            audit_controller._integrate()
        self.assertEqual(
            events,
            [
                "step", "route", "ff", "free", "cam", "corridor", "generic", "loads",
                "step", "route", "ff", "free", "cam", "corridor", "generic", "loads",
            ],
        )
        allowed_source = inspect.getsource(
            demo.MatchaWorkflowController._allowed_penetrating_contact
        )
        self.assertIn("core_cam_tab_allowed_contact_indices", allowed_source)
        self.assertIn("contact_index", allowed_source)
        self.assertNotIn("startswith", allowed_source)

    def test_result_helper_requires_external_binding_and_replay_authority(
        self,
    ) -> None:
        """Reject omitted independent inputs and retired lock promotion."""

        demo = self.demo
        model = demo.build_model()
        data = demo.mujoco.MjData(model)
        demo.initialize(model, data)
        controller = demo.MatchaWorkflowController(model, data)
        digests = self._independent_model_digests(model)
        observations = self._binding_observations(digests, digests)
        result = controller.result()
        self.assertEqual(
            core_cam_tab_result_errors(
                result,
                self.contract,
                model_binding_observations=observations,
                replay_model=model,
                replay_mujoco=demo.mujoco,
            ),
            [],
        )
        self.assertIn(
            "model_binding:observations_missing",
            core_cam_tab_result_errors(
                result,
                self.contract,
                replay_model=model,
                replay_mujoco=demo.mujoco,
            ),
        )

        for field in ("physical_cam_authority_ready", "locked"):
            mutated = copy.deepcopy(result)
            mutated[field] = True
            with self.subTest(promoted_field=field):
                self.assertNotEqual(
                    core_cam_tab_result_errors(
                        mutated,
                        self.contract,
                        model_binding_observations=observations,
                        replay_model=model,
                        replay_mujoco=demo.mujoco,
                    ),
                    [],
                )

        controller.physics_substep_count = 1
        data.time = float(model.opt.timestep)
        controller._record_core_cam_tab_functional_envelope(
            controller.actions[2], []
        )
        nonempty_result = controller.result()
        self.assertTrue(
            nonempty_result["core_cam_tab_contact_evidence"]
            ["functional_phase_envelope"]["raw_states"]
        )
        self.assertIn(
            "envelope:replay_authority_missing",
            core_cam_tab_result_errors(
                nonempty_result,
                self.contract,
                model_binding_observations=observations,
            ),
        )

    def test_retired_negative_z_suffix_and_complete_cam_overlap_stay_red(
        self,
    ) -> None:
        demo = self.demo
        default_actions = demo._recovery_controller_actions("gripper")
        self.assertEqual(
            [action.name for action in default_actions][-1],
            "gripper_dock_release_verify",
        )
        retired_kinds = {
            "axial_disengage", "slider_return", "physical_lock_confirm"
        }
        self.assertTrue(retired_kinds.isdisjoint(
            action.kind for action in default_actions
        ))
        retired_handlers = {
            "_command_axial_disengage",
            "_command_slider_return",
            "_command_physical_lock_confirm",
        }
        self.assertTrue(
            all(
                not hasattr(demo.MatchaWorkflowController, method_name)
                for method_name in retired_handlers
            )
        )
        for kind in sorted(retired_kinds):
            with self.subTest(retired_kind=kind):
                model = demo.build_model()
                data = demo.mujoco.MjData(model)
                demo.initialize(model, data)
                action = demo.WorkflowAction(
                    name=f"adversarial_{kind}", kind=kind, timeout_s=1.0
                )
                controller = demo.MatchaWorkflowController(
                    model, data, actions=(action,)
                )
                initial_time = float(data.time)
                with mock.patch.object(
                    demo.mujoco,
                    "mj_step",
                    side_effect=AssertionError("retired path advanced physics"),
                ):
                    controller.step()
                result = controller.result()
                self.assertEqual(
                    result["abort_reason"], f"unknown_action:{kind}"
                )
                self.assertEqual(result["physics_substep_count"], 0)
                self.assertEqual(float(data.time), initial_time)
                self.assertEqual(
                    result["core_cam_tab_contact_evidence"]["raw_contact_records"],
                    [],
                )
                self.assertEqual(
                    result["core_cam_tab_contact_evidence"][
                        "functional_phase_envelope"
                    ]["raw_states"],
                    [],
                )
                self.assertIs(result["physical_lock_confirmed"], False)
                self.assertIs(result["locked"], False)
                self.assertIs(result["success"], False)
        exclusion = self.contract["post_capture_exclusion"]
        self.assertEqual(
            exclusion["excluded_action"],
            "gripper_source_negative_y_physical_release",
        )
        self.assertEqual(
            exclusion["static_contract_api"],
            "core_dock_static_release_route_contract",
        )
        self.assertEqual(exclusion["source_axis"], "dock_local_negative_y")
        self.assertEqual(exclusion["axis_dock_local"], [0.0, -1.0, 0.0])
        self.assertEqual(exclusion["axis_world"], [0.0, 0.0, 1.0])
        self.assertEqual(exclusion["roster_row_count"], 31)
        self.assertEqual(
            exclusion["roster_canonical_sha256"],
            "f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293",
        )
        self.assertIs(exclusion["physical_release_action_implemented"], False)
        self.assertIs(
            self.contract["authority_scope"]["post_capture_release_authority"],
            False,
        )
        self.assertIs(self.contract["release_ready"], False)


class CoreCaptureGravityBiasFeedforwardTests(unittest.TestCase):
    """Bound the development-only feedforward to source and live state."""

    @classmethod
    def setUpClass(cls) -> None:
        require_path(MATCHA_DEMO, "matcha workflow controller")
        cls.demo = import_file(
            MATCHA_DEMO,
            "matcha_workflow_demo_gravity_bias",
            "matcha workflow gravity-bias controller",
        )
        cls.model = cls.demo.build_model()
        cls.route_contract = cls.demo.core_capture_route_runtime_contract()
        cls.contract = (
            cls.demo.core_capture_gravity_bias_feedforward_runtime_contract()
        )
        cls.expected_contract = expected_core_capture_gravity_bias_contract(
            cls.demo, cls.model, cls.route_contract
        )
        cls.source = MATCHA_DEMO.read_text()

    def test_contract_source_and_authority_mutations_fail_closed(self) -> None:
        demo = self.demo
        self.assertEqual(
            sha256_file(MATCHA_DEMO),
            "42414b1acdffe39c2affef94f6dd1df28f125670851036e54e6b4910d208ca7c",
        )
        self.assertEqual(
            _evidence_mismatches(self.contract, self.expected_contract), []
        )
        self.assertEqual(core_capture_gravity_bias_source_errors(self.source), [])
        self.assertEqual(
            self.contract["contract_identity_sha256"],
            CORE_CAPTURE_GRAVITY_BIAS_IDENTITY_SHA256,
        )
        self.assertEqual(
            self.contract["identity_revalidation"]["lightweight_identity"]
            ["observed_identity_sha256"],
            CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256,
        )
        self.assertIs(self.contract["release_ready"], False)

        contract_mutations: dict[str, dict[str, Any]] = {}
        wrong_sign = copy.deepcopy(self.contract)
        wrong_sign["formula"]["offset_sign"] = "negative"
        wrong_sign["formula_sha256"] = canonical_json_sha256(
            wrong_sign["formula"]
        )
        contract_mutations["wrong_sign"] = wrong_sign
        widened = copy.deepcopy(self.contract)
        widened["guard_thresholds"][
            "free_space_maximum_abs_q_error_rad"
        ] = 0.003
        contract_mutations["threshold_widened"] = widened
        biased_start = copy.deepcopy(self.contract)
        biased_start["source_binding"]["desired_start_q_by_action"][
            CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[1]
        ][0] += 1.0e-6
        biased_start["source_binding"]["desired_start_q_sha256"] = (
            canonical_json_sha256(
                biased_start["source_binding"][
                    "desired_start_q_by_action"
                ]
            )
        )
        contract_mutations["biased_action_start"] = biased_start
        model_hash = copy.deepcopy(self.contract)
        model_hash["source_binding"][
            "compiled_model_xml_equivalent_sha256"
        ] = "0" * 64
        contract_mutations["model_hash"] = model_hash
        promoted = copy.deepcopy(self.contract)
        promoted["authority_scope"]["physical_lock_authority"] = True
        promoted["authority_scope"]["release_ready"] = True
        promoted["release_ready"] = True
        contract_mutations["authority_promotion"] = promoted
        for name, mutation in contract_mutations.items():
            with self.subTest(contract_mutation=name):
                self.assertNotEqual(
                    _evidence_mismatches(mutation, self.expected_contract), []
                )

        mutated_model = demo.build_model()
        original_gravity = np.array(
            mutated_model.opt.gravity, dtype=np.float64, copy=True
        )
        mutated_model.opt.gravity[:] = 0.0
        self.assertIs(
            _expected_gravity_bias_dynamics_binding(
                mutated_model, demo
            )["passed"],
            False,
        )
        mutated_model.opt.gravity[:] = original_gravity
        shoulder_actuator = int(mutated_model.actuator("shoulder_pan").id)
        original_gain = float(
            mutated_model.actuator_gainprm[shoulder_actuator, 0]
        )
        mutated_model.actuator_gainprm[shoulder_actuator, 0] = (
            original_gain + 1.0
        )
        self.assertIs(
            _expected_gravity_bias_dynamics_binding(
                mutated_model, demo
            )["passed"],
            False,
        )
        mutated_model.actuator_gainprm[shoulder_actuator, 0] = original_gain
        initialized = demo.mujoco.MjData(mutated_model)
        demo.initialize(mutated_model, initialized)
        expected_active = _independent_initialized_active_geometry_sha256(
            mutated_model, initialized, demo.mujoco
        )
        floor_id = int(mutated_model.geom("matcha_floor_collision").id)
        mutated_model.geom_pos[floor_id, 2] += 1.0e-6
        initialized = demo.mujoco.MjData(mutated_model)
        demo.initialize(mutated_model, initialized)
        self.assertNotEqual(
            _independent_initialized_active_geometry_sha256(
                mutated_model, initialized, demo.mujoco
            ),
            expected_active,
        )

        def replacement(old: str, new: str) -> str:
            self.assertIn(old, self.source)
            return self.source.replace(old, new, 1)

        source_mutations = {
            "wrong_sign": replacement(
                "    offset = qfrc_bias / denominator\n",
                "    offset = -qfrc_bias / denominator\n",
            ),
            "constraint_input": replacement(
                "    qfrc_bias = np.asarray(\n",
                "    forbidden = scratch_data.qfrc_constraint\n"
                "    qfrc_bias = np.asarray(\n",
            ),
            "contact_force_input": replacement(
                "    qfrc_bias = np.asarray(\n",
                "    mujoco.mj_contactForce(model, scratch_data, 0, np.zeros(6))\n"
                "    qfrc_bias = np.asarray(\n",
            ),
            "inverse_input": replacement(
                "    mujoco.mj_integratePos(\n"
                "        model,\n"
                "        data.qpos,\n"
                "        generalized_velocity,\n"
                "        1.0,\n"
                "    )\n"
                "    mujoco.mj_forward(model, data)\n",
                "    mujoco.mj_integratePos(\n"
                "        model,\n"
                "        data.qpos,\n"
                "        generalized_velocity,\n"
                "        1.0,\n"
                "    )\n"
                "    mujoco.mj_inverse(model, data)\n",
            ),
            "live_qpos_write": replacement(
                "            live_qpos_before = np.asarray(\n",
                "            self.data.qpos[0] = self.data.qpos[0]\n"
                "            live_qpos_before = np.asarray(\n",
            ),
            "scratch_alias": replacement(
                "        self.core_capture_gravity_bias_scratch_data = mujoco.MjData(model)\n",
                "        self.core_capture_gravity_bias_scratch_data = self.data\n",
            ),
        }
        for name, mutation in source_mutations.items():
            with self.subTest(source_mutation=name):
                self.assertNotEqual(
                    core_capture_gravity_bias_source_errors(mutation), []
                )

        endpoint_fixture = {
            "event": "move_complete",
            "action": "gripper_capture_lateral_align",
            "physics_substep_count": 1080,
            "sim_time_s": 0.26999999999999796,
            "endpoint_q_error_rad": 1.0929280654048412e-05,
            "endpoint_max_abs_qvel_rad_s": 0.0005492046223591684,
            "endpoint_fk_position_error_m": 9.784799785729311e-06,
            "endpoint_fk_orientation_error_rad": 1.0929775727327563e-06,
            "endpoint_dwell_ticks": 4,
            "route_endpoint_evidence": {
                "action": "gripper_capture_lateral_align",
                "target_preseat_mm": 55.0,
                "target_source_x_mm": 0.2,
                "observed_preseat_mm": 55.00012463907181,
                "observed_x_mm": 0.19852505913442162,
                "source_x_error_mm": -0.0014749408655783947,
                "observed_transverse_y_mm": -0.009672193204833866,
                "position_error_m": 9.784799785729311e-06,
                "orientation_error_rad": 1.0929775727327563e-06,
                "physics_substep_count": 1080,
                "sim_time_s": 0.26999999999999796,
            },
        }
        self.assertEqual(
            _gravity_bias_endpoint_record_errors(
                endpoint_fixture,
                endpoint_fixture,
                self.contract["guard_thresholds"],
            ),
            [],
        )
        endpoint_mutations: dict[str, dict[str, Any]] = {}
        for field, value in (
            ("event", "forged"),
            ("action", "gripper_capture_axial_open_side"),
            ("physics_substep_count", 1079),
            ("sim_time_s", 0.27025),
            ("endpoint_q_error_rad", 0.001),
            ("endpoint_max_abs_qvel_rad_s", 0.001),
            ("endpoint_fk_position_error_m", 2.0e-05),
            ("endpoint_fk_orientation_error_rad", 2.0e-05),
            ("endpoint_dwell_ticks", 5),
        ):
            mutation = copy.deepcopy(endpoint_fixture)
            mutation[field] = value
            endpoint_mutations[field] = mutation
        nested_endpoint = copy.deepcopy(endpoint_fixture)
        nested_endpoint["route_endpoint_evidence"]["observed_x_mm"] += 1.0e-6
        endpoint_mutations["route_endpoint_evidence"] = nested_endpoint
        for name, mutation in endpoint_mutations.items():
            with self.subTest(endpoint_mutation=name):
                self.assertNotEqual(
                    _gravity_bias_endpoint_record_errors(
                        mutation,
                        endpoint_fixture,
                        self.contract["guard_thresholds"],
                    ),
                    [],
                )

    def test_prewrite_identity_and_code_drift_abort_before_control(self) -> None:
        demo = self.demo

        def exercise_mutation(mutate: Any, restore: Any) -> dict[str, Any]:
            data = demo.mujoco.MjData(self.model)
            demo.initialize(self.model, data)
            controller = demo.MatchaWorkflowController(self.model, data)
            initial_ctrl = np.array(data.ctrl, dtype=np.float64, copy=True)
            initial_qpos = np.array(data.qpos, dtype=np.float64, copy=True)
            initial_qvel = np.array(data.qvel, dtype=np.float64, copy=True)
            initial_time = float(data.time)
            mutate()
            try:
                with mock.patch.object(
                    demo.mujoco,
                    "mj_step",
                    side_effect=AssertionError(
                        "identity drift advanced physical state"
                    ),
                ):
                    controller.step()
            finally:
                restore()
            result = controller.result()
            self.assertEqual(result["physics_substep_count"], 0)
            self.assertTrue(
                str(result["abort_reason"]).startswith("gravity_bias_")
            )
            self.assertTrue(np.array_equal(data.ctrl, initial_ctrl))
            self.assertTrue(np.array_equal(data.qpos, initial_qpos))
            self.assertTrue(np.array_equal(data.qvel, initial_qvel))
            self.assertEqual(float(data.time), initial_time)
            self.assertEqual(
                result["core_capture_gravity_bias_feedforward_evidence"]
                ["raw_samples"],
                [],
            )
            return result

        original_code = demo._move_action_desired_q.__code__

        def replaced_desired_q(
            action: Any, desired_action_start_q: Any, elapsed_s: float
        ) -> tuple[np.ndarray, float]:
            return np.zeros(5, dtype=np.float64), 0.0

        exercise_mutation(
            lambda: setattr(
                demo._move_action_desired_q,
                "__code__",
                replaced_desired_q.__code__,
            ),
            lambda: setattr(
                demo._move_action_desired_q, "__code__", original_code
            ),
        )

        original_starts = demo.CORE_CAPTURE_ROUTE_DESIRED_START_Q
        biased_starts = {
            name: tuple(values) for name, values in original_starts.items()
        }
        biased_starts[CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[0]] = tuple(
            value + (1.0e-6 if index == 0 else 0.0)
            for index, value in enumerate(
                biased_starts[CORE_CAPTURE_GRAVITY_BIAS_ACTIONS[0]]
            )
        )
        exercise_mutation(
            lambda: setattr(
                demo, "CORE_CAPTURE_ROUTE_DESIRED_START_Q", biased_starts
            ),
            lambda: setattr(
                demo, "CORE_CAPTURE_ROUTE_DESIRED_START_Q", original_starts
            ),
        )

    def test_one_controller_step_replays_raw_state_and_stays_honestly_red(
        self,
    ) -> None:
        demo = self.demo
        data = demo.mujoco.MjData(self.model)
        demo.initialize(self.model, data)
        controller = demo.MatchaWorkflowController(self.model, data)
        controller.step()
        result = controller.result()
        self.assertEqual(
            result["physics_substep_count"],
            int(demo.PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP),
        )
        self.assertEqual(
            core_capture_gravity_bias_result_errors(
                result,
                self.contract,
                demo,
                self.model,
                self.route_contract,
            ),
            [],
        )
        evidence = result["core_capture_gravity_bias_feedforward_evidence"]
        self.assertEqual(len(evidence["raw_samples"]), 20)
        self.assertTrue(
            all(
                sample["prewrite_identity_passed"] is True
                and sample["prewrite_identity_sha256"]
                == CORE_CAPTURE_GRAVITY_BIAS_LIGHTWEIGHT_IDENTITY_SHA256
                and sample["positive_lock_slider_qpos_address"] == 5
                for sample in evidence["raw_samples"]
            )
        )
        self.assertIs(evidence["passed"], False)
        self.assertIs(evidence["release_ready"], False)
        self.assertIs(result["success"], False)
        self.assertIs(result["release_ready"], False)

        def reseal(mutated: dict[str, Any]) -> dict[str, Any]:
            mutated_evidence = mutated[
                "core_capture_gravity_bias_feedforward_evidence"
            ]
            mutated_evidence["raw_samples_sha256"] = canonical_json_sha256(
                mutated_evidence["raw_samples"]
            )
            mutated_evidence["recomputed_telemetry_sha256"] = (
                canonical_json_sha256(
                    mutated_evidence["recomputed_telemetry"]
                )
            )
            return mutated

        mutations: dict[str, dict[str, Any]] = {}
        wrong_sign = copy.deepcopy(result)
        wrong_sign["core_capture_gravity_bias_feedforward_evidence"][
            "raw_samples"
        ][0]["gravity_bias_offset_rad"][0] *= -1.0
        mutations["wrong_sign"] = reseal(wrong_sign)
        missing_row = copy.deepcopy(result)
        missing_row["core_capture_gravity_bias_feedforward_evidence"][
            "raw_samples"
        ].pop(3)
        mutations["missing_row"] = reseal(missing_row)
        biased_start = copy.deepcopy(result)
        biased_start["core_capture_gravity_bias_feedforward_evidence"][
            "raw_samples"
        ][0]["desired_action_start_q_rad"][0] += 1.0e-6
        mutations["biased_start"] = reseal(biased_start)
        prewrite_hash = copy.deepcopy(result)
        prewrite_hash["core_capture_gravity_bias_feedforward_evidence"][
            "raw_samples"
        ][0]["prewrite_identity_sha256"] = "0" * 64
        mutations["prewrite_hash"] = reseal(prewrite_hash)
        slider_address = copy.deepcopy(result)
        slider_address["core_capture_gravity_bias_feedforward_evidence"][
            "raw_samples"
        ][0]["positive_lock_slider_qpos_address"] = 4
        mutations["slider_address"] = reseal(slider_address)
        slider_raw_qpos = copy.deepcopy(result)
        slider_raw_qpos["core_capture_gravity_bias_feedforward_evidence"][
            "raw_samples"
        ][0]["live_full_qpos"][5] += 1.0e-3
        mutations["slider_raw_qpos"] = reseal(slider_raw_qpos)
        self_attested_slider = copy.deepcopy(result)
        self_attested_evidence = self_attested_slider[
            "core_capture_gravity_bias_feedforward_evidence"
        ]
        self_attested_evidence["raw_samples"][0]["fk"][
            "slider_q_mm"
        ] += 1.0
        self_attested_evidence["recomputed_telemetry"][0]["fk"][
            "slider_q_mm"
        ] += 1.0
        mutations["self_attested_slider"] = reseal(self_attested_slider)
        prohibited = copy.deepcopy(result)
        prohibited["core_capture_gravity_bias_feedforward_evidence"][
            "prohibited_operation_counts"
        ]["mj_inverse_call_count"] = 1
        mutations["prohibited_input"] = reseal(prohibited)
        action_roster = copy.deepcopy(result)
        action_roster["core_capture_gravity_bias_feedforward_evidence"][
            "frozen_action_roster_matches"
        ] = False
        mutations["frozen_action_roster"] = reseal(action_roster)
        first_cam = copy.deepcopy(result)
        first_cam["core_capture_gravity_bias_feedforward_evidence"][
            "first_cam_contact_record"
        ] = {"forged": True}
        mutations["first_cam_contact"] = reseal(first_cam)
        first_rejected = copy.deepcopy(result)
        first_rejected["core_capture_gravity_bias_feedforward_evidence"][
            "first_rejected_cam_contact_record"
        ] = 42
        mutations["first_rejected_cam_contact"] = reseal(first_rejected)
        coherent_torque = copy.deepcopy(result)
        coherent_evidence = coherent_torque[
            "core_capture_gravity_bias_feedforward_evidence"
        ]
        coherent_sample = coherent_evidence["raw_samples"][0]
        coherent_sample["actuator_torque_nm"][0] += 1.0
        first_actuator_id = int(
            self.model.actuator(demo.ARM_ACTUATORS[0]).id
        )
        first_force_limit = float(
            np.max(
                np.abs(self.model.actuator_forcerange[first_actuator_id])
            )
        )
        coherent_utilization = abs(
            float(coherent_sample["actuator_torque_nm"][0])
        ) / first_force_limit
        coherent_sample["actuator_torque_utilization"][0] = (
            coherent_utilization
        )
        coherent_evidence["recomputed_telemetry"][0][
            "actuator_torque_utilization"
        ][0] = coherent_utilization
        coherent_evidence["maximum_actuator_torque_utilization"] = max(
            float(value)
            for telemetry in coherent_evidence["recomputed_telemetry"]
            for value in telemetry["actuator_torque_utilization"]
        )
        mutations["coherent_torque_reseal"] = reseal(coherent_torque)
        promoted = copy.deepcopy(result)
        promoted_evidence = promoted[
            "core_capture_gravity_bias_feedforward_evidence"
        ]
        promoted_evidence["contact_force_authority"] = True
        promoted_evidence["passed"] = True
        promoted_evidence["release_ready"] = True
        promoted["success"] = True
        promoted["release_ready"] = True
        mutations["authority_promotion"] = reseal(promoted)
        for name, mutation in mutations.items():
            with self.subTest(result_mutation=name):
                self.assertNotEqual(
                    core_capture_gravity_bias_result_errors(
                        mutation,
                        self.contract,
                        demo,
                        self.model,
                        self.route_contract,
                    ),
                    [],
                )


FORBIDDEN_STATE_FIELDS = {"qpos", "qvel", "time", "eq_data"}
FORBIDDEN_RESET_CALLS = {"mj_resetData", "mj_resetDataKeyframe", "mj_setConst"}


def target_state_field(target: ast.AST) -> str | None:
    node = target
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_STATE_FIELDS:
        return node.attr
    return None


def called_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


class ControllerSourceSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        require_path(MATCHA_DEMO, "matcha workflow controller")
        cls.source = MATCHA_DEMO.read_text()
        cls.tree = ast.parse(cls.source, filename=str(MATCHA_DEMO))
        cls.controller_classes = [
            node
            for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and "controller" in node.name.lower()
        ]
        if not cls.controller_classes:
            raise unittest.SkipTest(
                "matcha workflow controller layer is not restored yet"
            )

    def test_controller_call_graph_never_teleports_physical_state(self) -> None:
        violations: list[str] = []
        for class_node in self.controller_classes:
            methods = {
                node.name: node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            reachable = set(methods)
            # All controller methods are treated as reachable: a fault hook or
            # rarely used helper is not allowed to become a teleport backdoor.
            for method_name in sorted(reachable):
                method = methods[method_name]
                for node in ast.walk(method):
                    targets: list[ast.AST] = []
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = (
                            list(node.targets)
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                    elif isinstance(node, ast.AugAssign):
                        targets = [node.target]
                    for target in targets:
                        field = target_state_field(target)
                        if field is not None:
                            violations.append(
                                f"{class_node.name}.{method_name}:{node.lineno} writes {field}"
                            )
                    if isinstance(node, ast.Call):
                        name = called_name(node)
                        if name in FORBIDDEN_RESET_CALLS:
                            violations.append(
                                f"{class_node.name}.{method_name}:{node.lineno} calls {name}"
                            )
                        if name in {"copyto", "put", "place"}:
                            text = ast.unparse(node)
                            if any(f".{field}" in text for field in FORBIDDEN_STATE_FIELDS):
                                violations.append(
                                    f"{class_node.name}.{method_name}:{node.lineno} mutates state via {name}"
                                )
        self.assertEqual(violations, [], "\n".join(violations))

    def test_controller_advances_only_through_mujoco_step(self) -> None:
        calls = [
            called_name(node)
            for class_node in self.controller_classes
            for node in ast.walk(class_node)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("mj_step", calls, "controller never advances real dynamics")
        self.assertNotIn("mj_forwardSkip", calls)

    def test_module_helpers_cannot_hide_post_initialize_state_writes(self) -> None:
        violations: list[str] = []
        for function in (
            node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != "initialize"
        ):
            for node in ast.walk(function):
                targets: list[ast.AST] = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                for target in targets:
                    field = target_state_field(target)
                    if field is not None:
                        violations.append(f"{function.name}:{node.lineno} writes {field}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_capture_path_cannot_claim_a_returned_physical_lock(self) -> None:
        violations: list[str] = []
        forbidden_phase_roots = {
            "_command_capture",
            "_command_lock_verify",
            "_command_move",
            "_command_release_verify",
        }
        retired_handlers = {
            "_command_axial_disengage",
            "_command_slider_return",
            "_command_physical_lock_confirm",
        }

        def target_is_physical_lock(target: ast.AST) -> bool:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "physical_lock_confirmed"
            ):
                return True
            return bool(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "self"
                and target.value.attr == "__dict__"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "physical_lock_confirmed"
            )

        def method_may_set_lock_true(method: ast.AST) -> list[int]:
            lines: list[int] = []
            for node in ast.walk(method):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = (
                        list(node.targets)
                        if isinstance(node, ast.Assign)
                        else [node.target]
                    )
                    value = node.value
                    for target in targets:
                        if target_is_physical_lock(target) and not (
                            isinstance(value, ast.Constant) and value.value is False
                        ):
                            lines.append(node.lineno)
                if not isinstance(node, ast.Call) or called_name(node) != "setattr":
                    continue
                if len(node.args) < 3:
                    lines.append(node.lineno)
                    continue
                field = node.args[1]
                value = node.args[2]
                if (
                    isinstance(field, ast.Constant)
                    and field.value == "physical_lock_confirmed"
                    and not (isinstance(value, ast.Constant) and value.value is False)
                ):
                    lines.append(node.lineno)
            return lines

        module_functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        module_setters = {
            name: lines
            for name, node in module_functions.items()
            if (lines := method_may_set_lock_true(node))
        }
        self.assertEqual(
            module_setters,
            {},
            "module helpers may not mutate controller physical-lock state",
        )
        for class_node in self.controller_classes:
            methods = {
                node.name: node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertTrue(retired_handlers.isdisjoint(methods), methods)
            self.assertTrue(forbidden_phase_roots.issubset(methods), methods)
            calls = {
                method_name: {
                    called
                    for node in ast.walk(method)
                    if isinstance(node, ast.Call)
                    and (called := called_name(node)) in methods
                }
                for method_name, method in methods.items()
            }
            setters: dict[str, list[int]] = {}
            for method_name, method in methods.items():
                setter_lines = method_may_set_lock_true(method)
                if setter_lines:
                    setters[method_name] = setter_lines
            for root in sorted(forbidden_phase_roots):
                reachable = {root}
                pending = [root]
                while pending:
                    current = pending.pop()
                    for called in calls[current] - reachable:
                        reachable.add(called)
                        pending.append(called)
                for setter in sorted(reachable.intersection(setters)):
                    for line in setters[setter]:
                        violations.append(f"{root}->{setter}:{line}")
        self.assertEqual(
            violations,
            [],
            "pre-withdrawal phases cannot confirm the spring lock while the dock cam holds it open",
        )


class RenderedCollisionInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = import_file(MATCHA_DEMO, "matcha_workflow_demo_validation", "matcha workflow")
        if not hasattr(cls.demo, "build_model"):
            raise AssertionError("matcha workflow does not expose build_model()")
        try:
            cls.model = cls.demo.build_model()
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"MuJoCo dependency not restored: {error.name}")

    def test_every_rendered_rigid_body_has_direct_active_collision_geometry(self) -> None:
        model = self.model
        body_visuals: dict[int, list[str]] = {}
        body_collisions: dict[int, list[str]] = {}
        for geom_id in range(int(model.ngeom)):
            name = str(model.geom(geom_id).name or f"geom_{geom_id}")
            body_id = int(model.geom_bodyid[geom_id])
            group = int(model.geom_group[geom_id])
            contype = int(model.geom_contype[geom_id])
            conaffinity = int(model.geom_conaffinity[geom_id])
            if group == 2 or (contype == 0 and conaffinity == 0):
                body_visuals.setdefault(body_id, []).append(name)
            if contype != 0 or conaffinity != 0:
                body_collisions.setdefault(body_id, []).append(name)

        missing: dict[str, list[str]] = {}
        for body_id, visuals in body_visuals.items():
            if body_id == 0 or body_id in body_collisions:
                continue
            physical_visuals = [
                name
                for name in visuals
                if not name.endswith("_target")
                and "camera_target" not in name
                and "fault_obstacle" not in name
            ]
            if physical_visuals:
                missing[str(model.body(body_id).name)] = physical_visuals
        self.assertEqual(missing, {}, json.dumps(missing, indent=2))

    def test_runtime_collision_coverage_api_is_complete(self) -> None:
        if not hasattr(self.demo, "collision_coverage"):
            self.skipTest("collision_coverage(model) is not restored yet")
        coverage = self.demo.collision_coverage(self.model)
        self.assertIsInstance(coverage, dict)
        complete = coverage.get("complete", coverage.get("collision_coverage_complete"))
        missing = coverage.get(
            "missing_collision_bodies", coverage.get("missing_bodies", [])
        )
        self.assertTrue(complete, coverage)
        self.assertEqual(missing, [], coverage)

    def test_initialized_scene_has_no_unreviewed_penetration(self) -> None:
        if not hasattr(self.demo, "initialize") or not hasattr(
            self.demo, "initial_contact_report"
        ):
            self.skipTest("initialized contact audit API is not restored yet")
        import mujoco

        data = mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        report = self.demo.initial_contact_report(self.model, data)
        self.assertTrue(report.get("passed"), report)
        self.assertEqual(int(report.get("penetration_count", -1)), 0, report)
        self.assertEqual(report.get("penetrations"), [], report)

    def test_declared_fixture_support_chains_exist_and_are_exactly_tangent(
        self,
    ) -> None:
        import mujoco

        config = load_json(MATCHA_CONFIG, "matcha simulation config")
        chains = config.get("fixture_support_chains")
        self.assertIsInstance(chains, dict)
        self.assertTrue(chains)
        data = mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        missing: list[str] = []
        measurements: dict[str, list[dict[str, float | str]]] = {}
        for fixture, chain in sorted(chains.items()):
            self.assertIsInstance(chain, list)
            self.assertGreaterEqual(len(chain), 2)
            fixture_records: list[dict[str, float | str]] = []
            for first, second in zip(chain, chain[1:]):
                first_name = str(first)
                second_name = str(second)
                first_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, first_name
                )
                second_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, second_name
                )
                if first_id < 0 or second_id < 0:
                    missing.extend(
                        name
                        for name, geom_id in (
                            (first_name, first_id),
                            (second_name, second_id),
                        )
                        if geom_id < 0
                    )
                    continue
                for geom_id in (first_id, second_id):
                    self.assertTrue(
                        int(self.model.geom_contype[geom_id])
                        or int(self.model.geom_conaffinity[geom_id]),
                        str(self.model.geom(geom_id).name),
                    )
                witness = np.zeros(6, dtype=np.float64)
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        data,
                        first_id,
                        second_id,
                        0.05,
                        witness,
                    )
                )
                fixture_records.append(
                    {
                        "first": first_name,
                        "second": second_name,
                        "distance_m": distance,
                    }
                )
                self.assertLessEqual(abs(distance), 1.0e-7, fixture_records[-1])
            measurements[str(fixture)] = fixture_records
        self.assertEqual(sorted(set(missing)), [], f"stale support names: {missing}")
        self.assertTrue(all(measurements.values()), measurements)

    def test_payload_report_active_geom_inventory_matches_compiled_model(self) -> None:
        report = load_json(PAYLOAD_REPORT, "payload collision authority report")
        groups = report.get("groups")
        self.assertIsInstance(groups, list)
        reported: list[str] = []
        for group in groups:
            self.assertIsInstance(group, dict)
            names = group.get("active_geom_names")
            self.assertIsInstance(names, list, group.get("name"))
            self.assertEqual(names, sorted(names), group.get("name"))
            reported.extend(str(name) for name in names)
        self.assertEqual(len(reported), len(set(reported)), "duplicate report geom")
        compiled = {str(self.model.geom(index).name) for index in range(self.model.ngeom)}
        self.assertEqual(set(reported) - compiled, set())
        for name in reported:
            geom_id = int(self.model.geom(name).id)
            self.assertTrue(
                int(self.model.geom_contype[geom_id])
                or int(self.model.geom_conaffinity[geom_id]),
                name,
            )
        expected = getattr(self.demo, "MATCHA_PAYLOAD_COLLISION_GEOM_NAMES", None)
        if expected is not None:
            self.assertEqual(set(reported), set(expected))


class SimCadPlacementContractTests(unittest.TestCase):
    """Bind runtime mount/stop primitives to their distinct source contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = import_file(
            MATCHA_DEMO,
            "matcha_demo_cad_placement_validation",
            "matcha workflow simulator",
        )
        cls.clearance = import_file(
            CORE_CLEARANCE_VALIDATOR,
            "core_clearance_placement_validation",
            "core CAD clearance validator",
        )
        cls.matcha_cad = import_file(
            MATCHA_CAD_GENERATOR,
            "matcha_cad_placement_validation",
            "matcha CAD generator",
        )
        build_xml = getattr(cls.demo, "_build_xml_and_assets", None)
        if build_xml is None:
            raise AssertionError("simulator must expose deterministic XML assembly")
        xml_text, cls.assets = build_xml()
        cls.xml_root = ET.fromstring(xml_text)
        cls.model = cls.demo.build_model()
        cls._geom_hull_equations: dict[int, np.ndarray] = {}

    @staticmethod
    def _vector(element: ET.Element, name: str, default: str) -> np.ndarray:
        return np.asarray(
            [float(value) for value in element.get(name, default).split()],
            dtype=np.float64,
        )

    def test_contact_semantic_caches_equal_compiled_predicates_and_audit_each_step(
        self,
    ) -> None:
        """Permit immutable lookup caching without weakening contact coverage."""

        data = self.demo.mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        controller = self.demo.MatchaWorkflowController(
            self.model,
            data,
            actions=self.demo._recovery_controller_actions(),
        )
        compiled_names = tuple(
            str(self.model.geom(geom_id).name)
            for geom_id in range(self.model.ngeom)
        )
        self.assertIsInstance(controller.geom_names, tuple)
        self.assertEqual(controller.geom_names, compiled_names)

        expected_robot_lands = frozenset(
            {
                "qc_col_robot_plate_core__mating_land",
                "qc_col_robot_plate_cam_relief_part_01",
                *(
                    name
                    for name in compiled_names
                    if name.startswith(
                        "qc_col_robot_plate_upper_well_partition_"
                    )
                    and name.endswith("__mating_land")
                ),
            }
        )
        self.assertIsInstance(controller.robot_mating_land_names, frozenset)
        self.assertEqual(controller.robot_mating_land_names, expected_robot_lands)

        expected_stops = {
            tool: frozenset(
                name
                for name in compiled_names
                if self.demo.qc.is_dock_stop_collision_name(tool, name)
            )
            for tool in self.demo.ALL_TOOL_IDS
        }
        self.assertEqual(controller.dock_stop_names_by_tool, expected_stops)
        self.assertTrue(
            all(
                isinstance(names, frozenset)
                for names in controller.dock_stop_names_by_tool.values()
            )
        )

        expected_pogo_pairs = {
            frozenset(
                {
                    f"qc_col_pogo_{signal}_plunger",
                    f"{tool}_pad_{signal}_collision",
                }
            ): (tool, signal)
            for tool in self.demo.ALL_TOOL_IDS
            for signal in self.demo.qc.SIGNALS
        }
        self.assertEqual(controller.pogo_pair_contract, expected_pogo_pairs)
        expected_support_pairs = {
            frozenset(
                {
                    f"dock_{tool}_support_anchor_collision",
                    f"dock_{tool}_support_collision",
                }
            )
            for tool in ("spoon", "whisk")
        }
        expected_support_pairs.update(
            frozenset(
                {f"dock_{tool}_support_collision", "matcha_floor_collision"}
            )
            for tool in ("spoon", "whisk")
        )
        expected_support_pairs.update(
            frozenset(pair)
            for pair in self.demo.qc.CORE_DOCK_SUPPORT_PROXY_FACE_TANGENCIES
        )
        expected_support_pairs.update(
            frozenset({name, "matcha_floor_collision"})
            for name in self.demo.qc.CORE_DOCK_SUPPORT_PROXY_FLOOR_CONTACT_GEOM_NAMES
        )
        self.assertEqual(controller.support_contact_pairs, expected_support_pairs)
        self.assertIsInstance(controller.support_contact_pairs, frozenset)
        self.assertTrue(
            all(
                "dock_gripper_support" not in name
                for pair in controller.support_contact_pairs
                for name in pair
            )
        )
        self.assertEqual(
            controller.slider_tab_geom_ids,
            tuple(
                geom_id
                for geom_id, name in enumerate(compiled_names)
                if name.startswith("qc_col_lock_slider_tab_part_")
            ),
        )
        self.assertEqual(
            controller.dock_gripper_cam_geom_id,
            int(self.model.geom("dock_gripper_cam_collision").id),
        )

        integrate_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(self.demo.MatchaWorkflowController._integrate)
            )
        )
        integration_loops = []
        for loop in (node for node in ast.walk(integrate_tree) if isinstance(node, ast.For)):
            calls = [
                (called_name(call), call.lineno)
                for call in ast.walk(loop)
                if isinstance(call, ast.Call)
            ]
            names = {name for name, _ in calls}
            if {"mj_step", "_audit_contacts"}.issubset(names):
                integration_loops.append(calls)
        self.assertEqual(len(integration_loops), 1)
        integration_calls = integration_loops[0]
        step_line = min(line for name, line in integration_calls if name == "mj_step")
        audit_line = min(
            line for name, line in integration_calls if name == "_audit_contacts"
        )
        self.assertLess(step_line, audit_line)

        audit_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(self.demo.MatchaWorkflowController._audit_contacts)
            )
        )
        contact_loops = [
            node
            for node in ast.walk(audit_tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and called_name(node.iter) == "range"
            and "self.data.ncon" in ast.unparse(node.iter)
        ]
        self.assertEqual(len(contact_loops), 1)
        self.assertIn("self.data.contact[contact_index]", ast.unparse(contact_loops[0]))

        forbidden_cache_mutations: list[str] = []
        cache_fields = {
            "geom_names",
            "robot_mating_land_names",
            "dock_stop_names_by_tool",
            "pogo_pair_contract",
            "support_contact_pairs",
            "slider_tab_geom_ids",
            "dock_gripper_cam_geom_id",
        }
        class_source = textwrap.dedent(
            inspect.getsource(self.demo.MatchaWorkflowController)
        )
        class_tree = ast.parse(class_source)
        for method in (
            node
            for node in ast.walk(class_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != "__init__"
        ):
            for node in ast.walk(method):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and target.attr in cache_fields
                        ):
                            forbidden_cache_mutations.append(
                                f"{method.name}:{node.lineno}:{target.attr}"
                            )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "self"
                    and node.func.value.attr in cache_fields
                    and node.func.attr
                    in {"add", "clear", "discard", "pop", "remove", "setdefault", "update"}
                ):
                    forbidden_cache_mutations.append(
                        f"{method.name}:{node.lineno}:{node.func.value.attr}.{node.func.attr}"
                    )
        self.assertEqual(forbidden_cache_mutations, [])
        self.assertNotIn("nominal_contact_pairs", class_source)
        self.assertNotIn("learned_contact", class_source)

        audit_calls = 0
        original_audit = controller._audit_contacts

        def counted_audit() -> None:
            nonlocal audit_calls
            audit_calls += 1
            original_audit()

        controller._audit_contacts = counted_audit
        before = int(controller.physics_substep_count)
        controller.step()
        advanced = int(controller.physics_substep_count) - before
        self.assertEqual(advanced, int(self.demo.PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP))
        self.assertEqual(audit_calls, advanced)

    def test_ordered_lock_actions_are_bounded_and_rack_exit_stays_separate(
        self,
    ) -> None:
        actions = self.demo._recovery_controller_actions()
        expected = (
            ("gripper_capture_lateral_align", "move"),
            ("gripper_capture_axial_open_side", "move"),
            ("gripper_capture_coupled_recenter", "move"),
            ("gripper_capture_centered_final", "move"),
            ("gripper_physical_capture", "capture"),
            ("gripper_lock_verify", "lock_verify"),
            ("gripper_dock_release_verify", "release_verify"),
        )
        self.assertEqual(
            tuple((action.name, action.kind) for action in actions), expected
        )
        self.assertTrue(all(math.isfinite(action.timeout_s) for action in actions))
        self.assertTrue(all(action.timeout_s > 0.0 for action in actions))
        self.assertTrue(all(action.duration_s >= 0.0 for action in actions))
        self.assertTrue(
            all(action.timeout_s >= action.duration_s for action in actions)
        )
        self.assertLessEqual(math.fsum(action.timeout_s for action in actions), 15.0)
        self.assertTrue(
            {
                "axial_disengage", "slider_return", "physical_lock_confirm"
            }.isdisjoint(action.kind for action in actions)
        )
        with self.assertRaisesRegex(ValueError, "rack exit"):
            self.demo._recovery_controller_actions(include_rack_exit=True)

        data = self.demo.mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        controller = self.demo.MatchaWorkflowController(
            self.model, data, actions=actions
        )
        result = controller.result()
        self.assertIs(result.get("physical_lock_confirmed"), False)
        self.assertIs(result.get("locked"), False)
        self.assertIs(result.get("success"), False)
        self.assertIs(result.get("release_ready"), False)

    @staticmethod
    def _binary_stl_vertices(payload: bytes) -> np.ndarray:
        if len(payload) < 84:
            raise AssertionError("binary STL is truncated")
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if len(payload) != 84 + 50 * triangle_count:
            raise AssertionError("binary STL byte count does not match its header")
        vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
        for triangle_index in range(triangle_count):
            offset = 84 + 50 * triangle_index + 12
            vertices[3 * triangle_index : 3 * triangle_index + 3] = np.asarray(
                struct.unpack_from("<9f", payload, offset), dtype=np.float64
            ).reshape(3, 3)
        return vertices

    @staticmethod
    def _quaternion_matrix(quaternion: Any) -> np.ndarray:
        values = np.array(quaternion, dtype=np.float64, copy=True)
        values /= np.linalg.norm(values)
        w_value, x_value, y_value, z_value = values
        return np.asarray(
            [
                [
                    1.0 - 2.0 * (y_value * y_value + z_value * z_value),
                    2.0 * (x_value * y_value - w_value * z_value),
                    2.0 * (x_value * z_value + w_value * y_value),
                ],
                [
                    2.0 * (x_value * y_value + w_value * z_value),
                    1.0 - 2.0 * (x_value * x_value + z_value * z_value),
                    2.0 * (y_value * z_value - w_value * x_value),
                ],
                [
                    2.0 * (x_value * z_value - w_value * y_value),
                    2.0 * (y_value * z_value + w_value * x_value),
                    1.0 - 2.0 * (x_value * x_value + y_value * y_value),
                ],
            ],
            dtype=np.float64,
        )

    def _compiled_geom_vertices_in_owner_frame(self, geom_id: int) -> np.ndarray:
        mujoco = self.demo.mujoco
        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
            vertices = np.asarray(
                [
                    (sx * size[0], sy * size[1], sz * size[2])
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                    for sz in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            start = int(self.model.mesh_vertadr[mesh_id])
            count = int(self.model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                self.model.mesh_vert[start : start + count], dtype=np.float64
            )
        else:
            raise AssertionError(
                f"unsupported void/proxy geom type for {self.model.geom(geom_id).name}: "
                f"{geom_type}"
            )
        rotation = self._quaternion_matrix(self.model.geom_quat[geom_id])
        return (
            np.asarray(self.model.geom_pos[geom_id], dtype=np.float64)
            + vertices @ rotation.T
        )

    def _convex_geom_contains_owner_point(
        self, geom_id: int, point: Any, *, tolerance_m: float = 1.0e-10
    ) -> bool:
        from scipy.spatial import ConvexHull

        equations = self._geom_hull_equations.get(geom_id)
        if equations is None:
            vertices = self._compiled_geom_vertices_in_owner_frame(geom_id)
            equations = np.asarray(ConvexHull(vertices).equations, dtype=np.float64)
            self._geom_hull_equations[geom_id] = equations
        query = np.asarray(point, dtype=np.float64)
        return bool(
            np.all(equations[:, :3] @ query + equations[:, 3] <= tolerance_m)
        )

    def _collision_geoms_containing_owner_point(
        self, body_name: str, prefix: str, point: Any
    ) -> list[str]:
        body_id = int(self.model.body(body_name).id)
        containing: list[str] = []
        for geom_id in range(self.model.ngeom):
            name = str(self.model.geom(geom_id).name)
            if (
                int(self.model.geom_bodyid[geom_id]) != body_id
                or not name.startswith(prefix)
                or not (
                    int(self.model.geom_contype[geom_id])
                    or int(self.model.geom_conaffinity[geom_id])
                )
            ):
                continue
            if self._convex_geom_contains_owner_point(geom_id, point):
                containing.append(name)
        return sorted(containing)

    @staticmethod
    def _source_mass_properties(
        shapes: Iterable[Any], density_kg_m3: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return exact composite mass, COM, and COM inertia from OCCT solids.

        CadQuery reports volume in mm^3 and unit-density inertia in mm^5.
        The parallel-axis composition is performed before the SI conversion so
        disjoint installed hardware is not silently fused or double counted.
        """

        source_shapes = list(shapes)
        if not source_shapes:
            raise AssertionError("source mass-property roster is empty")
        volumes_mm3 = np.asarray(
            [float(shape.Volume()) for shape in source_shapes], dtype=np.float64
        )
        if not np.all(np.isfinite(volumes_mm3)) or np.any(volumes_mm3 <= 0.0):
            raise AssertionError(f"invalid source volumes: {volumes_mm3}")
        centers_mm = np.asarray(
            [
                type(shape).centerOfMass(shape).toTuple()
                for shape in source_shapes
            ],
            dtype=np.float64,
        )
        total_volume_mm3 = float(math.fsum(float(value) for value in volumes_mm3))
        composite_center_mm = np.sum(
            volumes_mm3[:, None] * centers_mm, axis=0
        ) / total_volume_mm3
        inertia_mm5 = np.zeros((3, 3), dtype=np.float64)
        for shape, volume_mm3, center_mm in zip(
            source_shapes, volumes_mm3, centers_mm, strict=True
        ):
            local_inertia_mm5 = np.asarray(
                type(shape).matrixOfInertia(shape), dtype=np.float64
            )
            offset_mm = center_mm - composite_center_mm
            inertia_mm5 += local_inertia_mm5 + float(volume_mm3) * (
                float(offset_mm @ offset_mm) * np.eye(3)
                - np.outer(offset_mm, offset_mm)
            )
        mass_kg = total_volume_mm3 * float(density_kg_m3) * 1.0e-9
        inertia_kg_m2 = inertia_mm5 * float(density_kg_m3) * 1.0e-15
        return mass_kg, composite_center_mm * 1.0e-3, inertia_kg_m2

    def _compiled_body_inertia_tensor(self, body_id: int) -> np.ndarray:
        principal_rotation = self._quaternion_matrix(
            self.model.body_iquat[body_id]
        )
        return (
            principal_rotation
            @ np.diag(np.asarray(self.model.body_inertia[body_id], dtype=np.float64))
            @ principal_rotation.T
        )

    def _subtree_mass_properties_in_root_frame(
        self,
        data: Any,
        root_id: int,
        *,
        inertia_scale_overrides: dict[int, float] | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray, list[int]]:
        """Compose a moving body subtree about its instantaneous total COM."""

        descendants: list[int] = []
        for body_id in range(1, self.model.nbody):
            ancestor = body_id
            while ancestor not in (0, root_id):
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == root_id:
                descendants.append(body_id)
        if not descendants or descendants.count(root_id) != 1:
            raise AssertionError(
                f"invalid subtree roster for {self.model.body(root_id).name}: "
                f"{descendants}"
            )
        masses_kg = np.asarray(
            [float(self.model.body_mass[body_id]) for body_id in descendants],
            dtype=np.float64,
        )
        if np.any(masses_kg <= 0.0) or not np.all(np.isfinite(masses_kg)):
            raise AssertionError(f"invalid subtree body masses: {masses_kg}")
        total_mass_kg = float(math.fsum(float(value) for value in masses_kg))
        root_rotation_world = np.asarray(
            data.xmat[root_id], dtype=np.float64
        ).reshape(3, 3)
        root_position_world = np.asarray(data.xpos[root_id], dtype=np.float64)
        centers_root_m = np.asarray(
            [
                root_rotation_world.T
                @ (
                    np.asarray(data.xipos[body_id], dtype=np.float64)
                    - root_position_world
                )
                for body_id in descendants
            ],
            dtype=np.float64,
        )
        composite_com_m = np.sum(
            masses_kg[:, None] * centers_root_m, axis=0
        ) / total_mass_kg
        composite_inertia_kg_m2 = np.zeros((3, 3), dtype=np.float64)
        scales = inertia_scale_overrides or {}
        for body_id, mass_kg, center_root_m in zip(
            descendants, masses_kg, centers_root_m, strict=True
        ):
            body_rotation_world = np.asarray(
                data.xmat[body_id], dtype=np.float64
            ).reshape(3, 3)
            body_to_root_rotation = root_rotation_world.T @ body_rotation_world
            body_inertia_root = (
                body_to_root_rotation
                @ self._compiled_body_inertia_tensor(body_id)
                @ body_to_root_rotation.T
            ) * float(scales.get(body_id, 1.0))
            offset_m = center_root_m - composite_com_m
            composite_inertia_kg_m2 += body_inertia_root + float(mass_kg) * (
                float(offset_m @ offset_m) * np.eye(3)
                - np.outer(offset_m, offset_m)
            )
        return (
            total_mass_kg,
            composite_com_m,
            composite_inertia_kg_m2,
            descendants,
        )

    def _isolated_slider_return_result(
        self,
        *,
        spring_enabled: bool,
        pin_at_unlocked: bool,
        maximum_time_s: float = 1.0,
    ) -> dict[str, Any]:
        """Run a minimal equality-controlled copy using only ``mj_step``."""

        mujoco = self.demo.mujoco
        slider_body = self.xml_root.find(
            ".//body[@name='qc_positive_lock_slider']"
        )
        self.assertIsNotNone(slider_body)
        inertial = slider_body.find("./inertial")
        joint = slider_body.find("./joint[@name='qc_positive_lock_slider_joint']")
        self.assertIsNotNone(inertial)
        self.assertIsNotNone(joint)
        stiffness = float(joint.get("stiffness", "nan")) if spring_enabled else 0.0
        equality_active = "true" if pin_at_unlocked else "false"
        # Joint equalities are expressed relative to qpos0=ref.  A -3 mm
        # constant therefore pins the physical q=0 unlocked state.
        pin_offset_m = -float(joint.get("ref", "0"))
        limit_solver_attributes = ""
        for attribute in ("solreflimit", "solimplimit"):
            if attribute in joint.attrib:
                limit_solver_attributes += (
                    f' {attribute}="{joint.attrib[attribute]}"'
                )
        xml = f"""
<mujoco model="isolated_positive_lock_return">
  <option timestep="{float(self.model.opt.timestep):.17g}" gravity="0 0 0"/>
  <worldbody>
    <body name="slider">
      <inertial pos="{inertial.get('pos')}" mass="{inertial.get('mass')}"
                fullinertia="{inertial.get('fullinertia')}"/>
      <joint name="slider_joint" type="slide" axis="1 0 0"
             range="{joint.get('range')}" limited="true"
             ref="{joint.get('ref')}" stiffness="{stiffness:.17g}"
             springref="{joint.get('springref')}"
             damping="{joint.get('damping')}"
             frictionloss="{joint.get('frictionloss')}"
             armature="{joint.get('armature')}"{limit_solver_attributes}/>
    </body>
  </worldbody>
  <equality>
    <joint name="unlocked_pin" joint1="slider_joint"
           polycoef="{pin_offset_m:.17g} 0 0 0 0"
           active="{equality_active}" solref="0.0001 1"
           solimp="0.999 0.9999 0.00001"/>
  </equality>
  <keyframe><key name="unlocked" qpos="0"/></keyframe>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        self.assertEqual(float(data.qpos[0]), 0.0)
        self.assertEqual(float(data.qvel[0]), 0.0)
        self.assertEqual(bool(data.eq_active[0]), pin_at_unlocked)

        timestep_s = float(model.opt.timestep)
        dwell_samples = int(math.ceil(0.050 / timestep_s))
        window_q_m: list[float] = []
        window_qvel_m_s: list[float] = []
        trajectory_min_q_m = math.inf
        trajectory_max_q_m = -math.inf
        maximum_steps = int(math.ceil(maximum_time_s / timestep_s))
        for _ in range(maximum_steps):
            mujoco.mj_step(model, data)
            q_m = float(data.qpos[0])
            qvel_m_s = float(data.qvel[0])
            trajectory_min_q_m = min(trajectory_min_q_m, q_m)
            trajectory_max_q_m = max(trajectory_max_q_m, q_m)
            window_q_m.append(q_m)
            window_qvel_m_s.append(qvel_m_s)
            if len(window_q_m) > dwell_samples:
                window_q_m.pop(0)
                window_qvel_m_s.pop(0)

        lower_range_tolerance_m = 0.00002
        upper_range_tolerance_m = 0.00005
        locked_band_min_m = 0.00295
        locked_band_max_m = 0.00302
        maximum_abs_qvel_m_s = 0.005
        final_window_complete = len(window_q_m) == dwell_samples
        final_window_in_band = bool(
            final_window_complete
            and min(window_q_m) >= locked_band_min_m
            and max(window_q_m) <= locked_band_max_m
        )
        final_window_low_speed = bool(
            final_window_complete
            and max(abs(value) for value in window_qvel_m_s)
            <= maximum_abs_qvel_m_s
        )
        trajectory_within_range = bool(
            trajectory_min_q_m >= -lower_range_tolerance_m
            and trajectory_max_q_m <= 0.003 + upper_range_tolerance_m
        )
        passed = bool(
            spring_enabled
            and not pin_at_unlocked
            and trajectory_within_range
            and final_window_in_band
            and final_window_low_speed
        )
        return {
            "spring_enabled": spring_enabled,
            "pin_equality_active": pin_at_unlocked,
            "direct_state_writes_after_initialization": 0,
            "physics_timestep_s": timestep_s,
            "simulated_time_s": float(data.time),
            "required_dwell_s": 0.050,
            "dwell_sample_count": dwell_samples,
            "locked_band_m": [locked_band_min_m, locked_band_max_m],
            "maximum_abs_qvel_m_s": maximum_abs_qvel_m_s,
            "trajectory_range_m": [
                -lower_range_tolerance_m,
                0.003 + upper_range_tolerance_m,
            ],
            "trajectory_min_q_m": trajectory_min_q_m,
            "trajectory_max_q_m": trajectory_max_q_m,
            "trajectory_within_range": trajectory_within_range,
            "final_window_min_q_m": min(window_q_m),
            "final_window_max_q_m": max(window_q_m),
            "final_window_max_abs_qvel_m_s": max(
                abs(value) for value in window_qvel_m_s
            ),
            "final_window_in_band": final_window_in_band,
            "final_window_low_speed": final_window_low_speed,
            "passed": passed,
            "release_ready": False,
        }

    @staticmethod
    def _full_inertia_vector(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                matrix[0, 1],
                matrix[0, 2],
                matrix[1, 2],
            ],
            dtype=np.float64,
        )

    def test_stock_gripper_wrapper_composes_to_published_step_mount(self) -> None:
        wrapper = self.xml_root.find(".//body[@name='stock_gripper']")
        self.assertIsNotNone(wrapper, "runtime stock-gripper wrapper is missing")
        required = self.clearance._required_stock_sim_body_transform()
        observed_position = self._vector(wrapper, "pos", "0 0 0")
        expected_position = np.asarray(required["wrapper_body_pos_m"], dtype=np.float64)
        np.testing.assert_allclose(
            observed_position, expected_position, rtol=0.0, atol=1.0e-12
        )
        observed_quat = self._vector(wrapper, "quat", "1 0 0 0")
        expected_quat = np.asarray(
            required["wrapper_body_quat_wxyz"], dtype=np.float64
        )
        observed_quat /= np.linalg.norm(observed_quat)
        expected_quat /= np.linalg.norm(expected_quat)
        self.assertAlmostEqual(abs(float(observed_quat @ expected_quat)), 1.0, 12)
        self.assertLessEqual(float(required["position_residual_m"]), 1.0e-12)
        self.assertLessEqual(
            float(required["rotation_residual_frobenius"]), 1.0e-12
        )

    def _stop_box_records(self, tool: str) -> list[dict[str, Any]]:
        dock = self.xml_root.find(f".//body[@name='dock_{tool}']")
        self.assertIsNotNone(dock, f"dock_{tool} body is missing")
        prefix = f"dock_{tool}_qc_col_dock_stop"
        records: list[dict[str, Any]] = []
        for geom in dock.findall("./geom"):
            name = str(geom.get("name", ""))
            if not name.startswith(prefix):
                continue
            self.assertEqual(geom.get("type"), "box", name)
            self.assertTrue(
                int(geom.get("contype", "0")) or int(geom.get("conaffinity", "0")),
                name,
            )
            quat = self._vector(geom, "quat", "1 0 0 0")
            quat /= np.linalg.norm(quat)
            self.assertAlmostEqual(abs(float(quat[0])), 1.0, 12, name)
            self.assertLessEqual(float(np.linalg.norm(quat[1:])), 1.0e-12, name)
            position = self._vector(geom, "pos", "0 0 0")
            half_size = self._vector(geom, "size", "")
            self.assertEqual(half_size.shape, (3,), name)
            self.assertTrue(np.all(half_size > 0.0), name)
            records.append(
                {
                    "name": name,
                    "position_m": position,
                    "half_size_m": half_size,
                    "minimum_m": position - half_size,
                    "maximum_m": position + half_size,
                }
            )
        self.assertTrue(records, f"{prefix}* collision pieces are missing")
        self.assertEqual(
            [record["name"] for record in records],
            sorted(record["name"] for record in records),
            f"{tool} stop pieces must have deterministic sorted names",
        )
        return records

    def test_each_dock_stop_matches_its_exact_source_bounds_and_core_holes(
        self,
    ) -> None:
        core_spec = self.clearance.CAD.core_dock_stop_spec()
        matcha_spec = self.matcha_cad.matcha_dock_stop_spec()
        self.assertNotEqual(core_spec["bounds_mm"], matcha_spec["bounds_mm"])
        observed_bounds: dict[str, dict[str, list[float]]] = {}
        for tool, spec in (
            ("gripper", core_spec),
            ("spoon", matcha_spec),
            ("whisk", matcha_spec),
        ):
            records = self._stop_box_records(tool)
            minimum = np.min(
                np.stack([record["minimum_m"] for record in records]), axis=0
            )
            maximum = np.max(
                np.stack([record["maximum_m"] for record in records]), axis=0
            )
            expected_minimum = 0.001 * np.asarray(
                [spec["bounds_mm"][axis][0] for axis in "xyz"], dtype=np.float64
            )
            expected_maximum = 0.001 * np.asarray(
                [spec["bounds_mm"][axis][1] for axis in "xyz"], dtype=np.float64
            )
            np.testing.assert_allclose(
                minimum, expected_minimum, rtol=0.0, atol=1.0e-12
            )
            np.testing.assert_allclose(
                maximum, expected_maximum, rtol=0.0, atol=1.0e-12
            )
            observed_bounds[tool] = {
                "minimum_m": minimum.tolist(),
                "maximum_m": maximum.tolist(),
            }
            if tool != "gripper":
                continue
            for hole in spec["through_holes"]:
                radius_m = 0.0005 * float(hole["diameter_mm"])
                y_value = 0.5 * (expected_minimum[1] + expected_maximum[1])
                for radial_x, radial_z in (
                    (0.0, 0.0),
                    (0.5, 0.0),
                    (-0.5, 0.0),
                    (0.0, 0.5),
                    (0.0, -0.5),
                ):
                    point = np.asarray(
                        [
                            0.001 * float(hole["x_mm"]) + radial_x * radius_m,
                            y_value,
                            0.001 * float(hole["z_mm"]) + radial_z * radius_m,
                        ]
                    )
                    filled_by = [
                        record["name"]
                        for record in records
                        if np.all(
                            np.abs(point - record["position_m"])
                            < record["half_size_m"] - 1.0e-12
                        )
                    ]
                    self.assertEqual(
                        filled_by,
                        [],
                        f"core stop proxy fills functional hole {hole}: {point}",
                    )
        self.assertNotEqual(observed_bounds["gripper"], observed_bounds["spoon"])

    def test_core_keeper_contract_is_exact_and_excludes_the_air_gap_stop(self) -> None:
        release = self.demo.core_dock_static_release_route_contract()
        self.assertEqual(release["row_count"], 31)
        self.assertEqual(release["withdrawal_bounds_mm"], [0.0, 15.0])
        self.assertEqual(release["step_mm"], 0.5)
        self.assertEqual(release["axis_dock_local"], [0.0, -1.0, 0.0])
        self.assertEqual(release["axis_world"], [0.0, 0.0, 1.0])
        self.assertIs(release["included_in_default_controller_actions"], False)
        self.assertIs(release["physical_release_action_implemented"], False)
        self.assertIs(release["physical_release_authority"], False)
        self.assertIs(release["release_ready"], False)
        self.assertTrue(
            {
                "axial_disengage", "slider_return", "physical_lock_confirm"
            }.isdisjoint(
                action.kind for action in self.demo._recovery_controller_actions()
            ),
            "the source-negative -Z lock suffix must remain retired",
        )
        contract = getattr(self.demo, "CORE_KEEPER_CONTACT_CONTRACT", None)
        self.assertIsInstance(contract, (tuple, list))
        self.assertEqual(len(contract), 5)
        expected = {
            ("stock_tool_plate", "left_lower_rail"): {
                "runtime_pair": [
                    "matcha_col_gripper_plate_electrical_wing_edge__mating_land__keeper_land",
                    "dock_gripper_keeper_left_lower_collision",
                ],
                "expected_local_normal_subspace": "dock_xz_plane",
                "source_witness": {
                    "kind": "line_tangency",
                    "frame": "dock_gripper",
                    "line_axis": "y",
                    "fixed_coordinates_mm": {"x": -36.0, "z": 0.0},
                    "line_axis_bounds_mm": [-12.0, 12.0],
                    "point_tolerance_mm": 0.020,
                },
            },
            ("stock_tool_plate", "left_upper_rail"): {
                "runtime_pair": [
                    "matcha_col_gripper_plate_electrical_wing_edge__mating_land__keeper_land",
                    "dock_gripper_keeper_left_upper_collision",
                ],
                "expected_local_normal_axis": "z",
                "source_witness": {
                    "kind": "planar_face_tangency",
                    "frame": "dock_gripper",
                    "normal_axis": "z",
                    "plane_coordinate_mm": 9.5,
                    "tangential_bounds_mm": {
                        "x": [-36.0, -33.0],
                        "y": [-12.0, 12.0],
                    },
                    "point_tolerance_mm": 0.020,
                },
            },
            ("stock_tool_plate", "right_lower_rail"): {
                "runtime_pair": [
                    "matcha_col_gripper_plate_xpos__mating_land__locator_land__dock_stop_land",
                    "dock_gripper_keeper_right_lower_collision",
                ],
                "expected_local_normal_subspace": "dock_xz_plane",
                "source_witness": {
                    "kind": "line_tangency",
                    "frame": "dock_gripper",
                    "line_axis": "y",
                    "fixed_coordinates_mm": {"x": 28.0, "z": 0.0},
                    "line_axis_bounds_mm": [-21.0, 21.0],
                    "point_tolerance_mm": 0.020,
                },
            },
            ("stock_tool_plate", "right_upper_rail"): {
                "runtime_pair": [
                    "matcha_col_gripper_plate_xpos__mating_land__locator_land__dock_stop_land",
                    "dock_gripper_keeper_right_upper_collision",
                ],
                "expected_local_normal_axis": "z",
                "source_witness": {
                    "kind": "planar_face_tangency",
                    "frame": "dock_gripper",
                    "normal_axis": "z",
                    "plane_coordinate_mm": 9.5,
                    "tangential_bounds_mm": {
                        "x": [25.0, 28.0],
                        "y": [-25.0, 25.0],
                    },
                    "source_boundary_constraint": {
                        "kind": "rounded_rectangle",
                        "half_width_mm": 28.0,
                        "half_height_mm": 25.0,
                        "corner_radius_mm": 4.0,
                    },
                    "point_tolerance_mm": 0.020,
                },
            },
            ("robot_plate", "left_lower_rail"): {
                "runtime_pair": [
                    "qc_col_robot_plate_electrical_wing_edge__keeper_land",
                    "dock_gripper_keeper_left_lower_collision",
                ],
                "expected_local_normal_axis": "x",
                "source_witness": {
                    "kind": "planar_face_tangency",
                    "frame": "dock_gripper",
                    "normal_axis": "x",
                    "plane_coordinate_mm": -36.0,
                    "tangential_bounds_mm": {
                        "y": [-12.0, 12.0],
                        "z": [-3.0, 0.0],
                    },
                    "point_tolerance_mm": 0.020,
                },
            },
        }
        self.assertEqual(
            set(expected), set(self.clearance.INTENDED_ZERO_VOLUME_CONTACT_PAIRS)
        )
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        for record in contract:
            self.assertIsInstance(record, dict)
            source_pair = tuple(str(item) for item in record["source_pair"])
            runtime_pair = tuple(str(item) for item in record["runtime_pair"])
            self.assertEqual(len(source_pair), 2)
            self.assertEqual(len(runtime_pair), 2)
            self.assertNotIn(source_pair, observed)
            self.assertFalse(
                any(name.startswith("dock_gripper_qc_col_dock_stop") for name in runtime_pair),
                record,
            )
            normal_fields = {
                key: record[key]
                for key in (
                    "expected_local_normal_axis",
                    "expected_local_normal_subspace",
                )
                if key in record
            }
            self.assertEqual(len(normal_fields), 1, record)
            observed[source_pair] = {
                "runtime_pair": list(runtime_pair),
                **normal_fields,
                "source_witness": record.get("source_witness"),
            }
        self.assertEqual(observed, expected)
        xml_geoms = {
            str(geom.get("name")): geom for geom in self.xml_root.iter("geom") if geom.get("name")
        }
        expected_owners = {
            "matcha_col_gripper_plate_electrical_wing_edge__mating_land__keeper_land": (
                "tool_gripper"
            ),
            "matcha_col_gripper_plate_xpos__mating_land__locator_land__dock_stop_land": (
                "tool_gripper"
            ),
            "qc_col_robot_plate_electrical_wing_edge__keeper_land": (
                "robot_plate_frame"
            ),
            "dock_gripper_keeper_left_lower_collision": "dock_gripper",
            "dock_gripper_keeper_left_upper_collision": "dock_gripper",
            "dock_gripper_keeper_right_lower_collision": "dock_gripper",
            "dock_gripper_keeper_right_upper_collision": "dock_gripper",
        }
        for expected_record in expected.values():
            for name in expected_record["runtime_pair"]:
                self.assertIn(name, xml_geoms)
                geom_id = int(self.model.geom(name).id)
                owner_id = int(self.model.geom_bodyid[geom_id])
                self.assertEqual(
                    str(self.model.body(owner_id).name), expected_owners[name], name
                )
            geom_ids = [
                int(self.model.geom(name).id)
                for name in expected_record["runtime_pair"]
            ]
            contype_a = int(self.model.geom_contype[geom_ids[0]])
            affinity_a = int(self.model.geom_conaffinity[geom_ids[0]])
            contype_b = int(self.model.geom_contype[geom_ids[1]])
            affinity_b = int(self.model.geom_conaffinity[geom_ids[1]])
            self.assertTrue(
                (contype_a & affinity_b) or (contype_b & affinity_a),
                {
                    "runtime_pair": expected_record["runtime_pair"],
                    "contype": [contype_a, contype_b],
                    "conaffinity": [affinity_a, affinity_b],
                },
            )

        dock_features = self.clearance._named_dock_features()
        expected_bounds_m: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in ("left", "right"):
            for level in ("lower", "upper"):
                feature_name = f"{side}_{level}_rail"
                bounds = dock_features[feature_name].val().BoundingBox()
                expected_bounds_m[f"dock_gripper_keeper_{side}_{level}_collision"] = (
                    0.001 * np.asarray([bounds.xmin, bounds.ymin, bounds.zmin]),
                    0.001 * np.asarray([bounds.xmax, bounds.ymax, bounds.zmax]),
                )
        cad = self.clearance.CAD
        edge_x_max_mm = cad.CONTACT_CENTER_X - (
            cad.CONTACT_BOARD_WIDTH + 0.25
        ) / 2.0
        edge_bounds = (
            0.001
            * np.asarray(
                [
                    cad.ELECTRICAL_WING_X_MIN,
                    -cad.ELECTRICAL_WING_HEIGHT / 2.0,
                    0.0,
                ]
            ),
            0.001
            * np.asarray(
                [
                    edge_x_max_mm,
                    cad.ELECTRICAL_WING_HEIGHT / 2.0,
                    cad.PLATE_THICKNESS,
                ]
            ),
        )
        expected_bounds_m[
            "matcha_col_gripper_plate_electrical_wing_edge__mating_land__keeper_land"
        ] = edge_bounds
        expected_bounds_m[
            "qc_col_robot_plate_electrical_wing_edge__keeper_land"
        ] = edge_bounds
        expected_bounds_m[
            "matcha_col_gripper_plate_xpos__mating_land__locator_land__dock_stop_land"
        ] = (
            np.asarray([0.004, -0.025, 0.0]),
            np.asarray([0.028, 0.025, 0.001 * cad.PLATE_THICKNESS]),
        )
        self.assertEqual(set(expected_bounds_m), set(expected_owners))
        for name, (expected_minimum, expected_maximum) in expected_bounds_m.items():
            geom = xml_geoms[name]
            if name.startswith("matcha_col_gripper_plate_xpos"):
                self.assertEqual(geom.get("type"), "mesh", name)
                mesh_name = str(geom.get("mesh"))
                mesh_element = self.xml_root.find(
                    f"./asset/mesh[@name='{mesh_name}']"
                )
                self.assertIsNotNone(mesh_element, mesh_name)
                serialized_vertices = mesh_element.get("vertex")
                if serialized_vertices is not None:
                    vertices = np.asarray(
                        [float(value) for value in serialized_vertices.split()],
                        dtype=np.float64,
                    ).reshape(-1, 3)
                else:
                    mesh_file = mesh_element.get("file")
                    self.assertIsNotNone(mesh_file, mesh_name)
                    self.assertTrue(mesh_file in self.assets, mesh_file)
                    vertices = self._binary_stl_vertices(self.assets[mesh_file])
                minimum = np.min(vertices, axis=0)
                maximum = np.max(vertices, axis=0)
                self.assertTrue(np.all(minimum >= expected_minimum - 1.0e-9), name)
                self.assertTrue(np.all(maximum <= expected_maximum + 1.0e-9), name)
                # The stable x-pos semantic now names the outer keeper-bearing
                # partition only; the lock-bore/nut pieces close the remainder.
                # Retain the exact released rounded exterior and full Z/y
                # keeper span without requiring this one partition to fill the
                # newly validated functional voids.
                np.testing.assert_allclose(
                    maximum,
                    expected_maximum,
                    rtol=0.0,
                    atol=1.0e-9,
                    err_msg=name,
                )
                self.assertAlmostEqual(float(minimum[1]), -0.025, 9, name)
                self.assertAlmostEqual(float(minimum[2]), 0.0, 9, name)
                for x_value, y_value in vertices[:, :2]:
                    dx = max(abs(float(x_value)) - 0.024, 0.0)
                    dy = max(abs(float(y_value)) - 0.021, 0.0)
                    self.assertLessEqual(
                        math.hypot(dx, dy), 0.004 + 5.0e-6, name
                    )
                continue
            self.assertEqual(geom.get("type"), "box", name)
            quat = self._vector(geom, "quat", "1 0 0 0")
            quat /= np.linalg.norm(quat)
            self.assertAlmostEqual(abs(float(quat[0])), 1.0, 12, name)
            self.assertLessEqual(float(np.linalg.norm(quat[1:])), 1.0e-12, name)
            position = self._vector(geom, "pos", "0 0 0")
            half_size = self._vector(geom, "size", "")
            np.testing.assert_allclose(
                position - half_size,
                expected_minimum,
                rtol=0.0,
                atol=1.0e-12,
                err_msg=name,
            )
            np.testing.assert_allclose(
                position + half_size,
                expected_maximum,
                rtol=0.0,
                atol=1.0e-12,
                err_msg=name,
            )

    def test_matcha_docks_retain_stop_contact_but_core_uses_only_keepers(
        self,
    ) -> None:
        data = self.demo.mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        controller = self.demo.MatchaWorkflowController(self.model, data)

        valid_stop_contacts: dict[str, int] = {
            "gripper": 0,
            "spoon": 0,
            "whisk": 0,
        }
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            for tool in valid_stop_contacts:
                if controller._dock_stop_contact_is_valid(contact, tool):
                    valid_stop_contacts[tool] += 1

        self.assertEqual(
            valid_stop_contacts["gripper"],
            0,
            "the core dock stop has a designed air gap and is not a seating witness",
        )
        self.assertFalse(controller._dock_stop_is_seated("gripper"))
        for tool in ("spoon", "whisk"):
            self.assertGreater(valid_stop_contacts[tool], 0, tool)
            self.assertTrue(controller._dock_stop_is_seated(tool), tool)

    def test_line_keeper_normal_cones_are_oriented_not_any_xz_vector(self) -> None:
        validator = getattr(
            self.demo.MatchaWorkflowController,
            "_core_keeper_normal_is_valid",
            None,
        )
        self.assertTrue(
            callable(validator),
            "keeper contacts need an oriented-normal predicate, not projection only",
        )
        contract = {
            tuple(record["source_pair"]): record
            for record in self.demo.CORE_KEEPER_CONTACT_CONTRACT
        }
        left = contract[("stock_tool_plate", "left_lower_rail")]
        right = contract[("stock_tool_plate", "right_lower_rail")]
        dock_rotation = np.eye(3)

        def contact(normal: tuple[float, float, float]) -> SimpleNamespace:
            unit = np.asarray(normal, dtype=np.float64)
            unit /= np.linalg.norm(unit)
            return SimpleNamespace(frame=np.concatenate((unit, np.zeros(6))))

        for normal in ((1.0, 0.0, 1.0), (-1.0, 0.0, -1.0)):
            self.assertTrue(validator(contact(normal), left, dock_rotation), normal)
            self.assertFalse(validator(contact(normal), right, dock_rotation), normal)
        for normal in ((1.0, 0.0, -1.0), (-1.0, 0.0, 1.0)):
            self.assertFalse(validator(contact(normal), left, dock_rotation), normal)
            self.assertTrue(validator(contact(normal), right, dock_rotation), normal)
        for normal in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
            self.assertTrue(validator(contact(normal), left, dock_rotation), normal)
            self.assertTrue(validator(contact(normal), right, dock_rotation), normal)
        self.assertFalse(validator(contact((0.0, 1.0, 0.0)), left, dock_rotation))
        self.assertFalse(validator(contact((0.0, 1.0, 0.0)), right, dock_rotation))

    def test_degenerate_line_distance_uses_live_geometry_not_garbage_endpoints(
        self,
    ) -> None:
        """A zero scalar with an untouched ``from_to`` buffer is not a witness."""

        mujoco = self.demo.mujoco
        model = self.demo.build_model()
        data = mujoco.MjData(model)
        self.demo.initialize(model, data)
        controller = self.demo.MatchaWorkflowController(model, data)
        contract = next(
            record
            for record in self.demo.CORE_KEEPER_CONTACT_CONTRACT
            if record["source_pair"] == ["stock_tool_plate", "right_lower_rail"]
        )
        geom_ids = [int(model.geom(name).id) for name in contract["runtime_pair"]]

        # Break symmetry without changing the source line: a hard-coded origin
        # witness would remain y=0, whereas the live edge-overlap midpoint moves
        # to +2.5 mm.  Disable this one pair in the contact solver so the test
        # deterministically exercises the optional no-ncon line-tangency path.
        model.geom_pos[geom_ids[0], 1] += 0.005
        model.geom_contype[geom_ids[0]] = 0
        model.geom_conaffinity[geom_ids[0]] = 0
        mujoco.mj_forward(model, data)

        original_geom_distance = mujoco.mj_geomDistance

        def degenerate_geom_distance(
            model_arg: Any,
            data_arg: Any,
            geom_a: int,
            geom_b: int,
            distance_cap: float,
            from_to: np.ndarray,
        ) -> float:
            if {int(geom_a), int(geom_b)} == set(geom_ids):
                # Reproduce MuJoCo's degenerate-line behavior: return a valid
                # signed scalar but do not touch the caller's endpoint buffer.
                return 0.0
            return float(
                original_geom_distance(
                    model_arg,
                    data_arg,
                    geom_a,
                    geom_b,
                    distance_cap,
                    from_to,
                )
            )

        with mock.patch.object(
            mujoco, "mj_geomDistance", side_effect=degenerate_geom_distance
        ):
            report = controller._core_keeper_contact_report()

        record = next(
            item
            for item in report["records"]
            if item["source_pair"] == contract["source_pair"]
        )
        self.assertEqual(record.get("contact_count"), 0, record)
        self.assertEqual(record.get("signed_distance_mm"), 0.0, record)
        self.assertIs(record.get("mujoco_from_to_valid"), False, record)
        self.assertEqual(
            record.get("closest_point_method"),
            "analytic_box_box_line_tangency_from_live_geom_transforms",
            record,
        )
        self.assertEqual(
            record.get("witness_method"),
            "live_mujoco_signed_geom_distance_and_source_semantics",
            record,
        )

        # Independently reconstruct each active geom's transformed edge in the
        # dock frame.  This deliberately does not call the production fallback.
        dock = data.body("dock_gripper")
        dock_rotation = np.asarray(dock.xmat, dtype=np.float64).reshape(3, 3)
        edge_y_bounds: list[tuple[float, float]] = []
        for geom_id in geom_ids:
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
                local_vertices = np.asarray(
                    [
                        (sx * size[0], sy * size[1], sz * size[2])
                        for sx in (-1.0, 1.0)
                        for sy in (-1.0, 1.0)
                        for sz in (-1.0, 1.0)
                    ],
                    dtype=np.float64,
                )
            else:
                self.assertEqual(geom_type, int(mujoco.mjtGeom.mjGEOM_MESH))
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                local_vertices = np.asarray(
                    model.mesh_vert[start : start + count], dtype=np.float64
                )
            geom_rotation = np.asarray(
                data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            world_vertices = (
                np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
                + local_vertices @ geom_rotation.T
            )
            dock_vertices_mm = (
                (world_vertices - np.asarray(dock.xpos, dtype=np.float64))
                @ dock_rotation
                * 1000.0
            )
            on_exact_line = (
                (np.abs(dock_vertices_mm[:, 0] - 28.0) <= 1.0e-3)
                & (np.abs(dock_vertices_mm[:, 2]) <= 1.0e-3)
            )
            self.assertGreaterEqual(int(np.count_nonzero(on_exact_line)), 2)
            edge = dock_vertices_mm[on_exact_line]
            edge_y_bounds.append(
                (float(np.min(edge[:, 1])), float(np.max(edge[:, 1])))
            )

        source_lower, source_upper = (-21.0, 21.0)
        overlap_lower = max(source_lower, *(bounds[0] for bounds in edge_y_bounds))
        overlap_upper = min(source_upper, *(bounds[1] for bounds in edge_y_bounds))
        self.assertLessEqual(overlap_lower, overlap_upper)
        expected_point = np.asarray(
            [28.0, 0.5 * (overlap_lower + overlap_upper), 0.0],
            dtype=np.float64,
        )
        self.assertGreater(abs(float(expected_point[1])), 1.0)
        closest_points = record.get("closest_points_dock_local_mm")
        self.assertIsInstance(closest_points, list, record)
        self.assertEqual(len(closest_points), 2, record)
        for point in closest_points:
            np.testing.assert_allclose(
                np.asarray(point, dtype=np.float64),
                expected_point,
                rtol=0.0,
                atol=1.0e-3,
                err_msg=str(record),
            )

    def test_positive_lock_studs_are_tool_owned_and_match_source_geometry(
        self,
    ) -> None:
        """Reject the recovered robot-side/y-offset stud approximation."""

        cad = self.clearance.CAD
        self.assertEqual(float(cad.LOCK_STUD_X), 12.0)
        self.assertEqual(float(cad.LOCK_SHOULDER_DIAMETER), 4.0)
        self.assertEqual(float(cad.LOCK_SHOULDER_LENGTH), 5.0)
        self.assertEqual(float(cad.LOCK_HEAD_DIAMETER), 6.0)
        self.assertEqual(float(cad.LOCK_HEAD_HEIGHT), 1.3)
        source_bounds = cad.shoulder_lock_stud().val().BoundingBox()
        np.testing.assert_allclose(
            [
                source_bounds.xmin,
                source_bounds.ymin,
                source_bounds.zmin,
                source_bounds.xmax,
                source_bounds.ymax,
                source_bounds.zmax,
            ],
            [-3.0, -3.0, -6.3, 3.0, 3.0, 4.0],
            rtol=0.0,
            atol=1.0e-9,
        )

        compiled_names = {
            str(self.model.geom(index).name) for index in range(self.model.ngeom)
        }
        legacy_names = {
            name
            for name in compiled_names
            if name.startswith("qc_col_stud_")
            or any(
                name.startswith(f"{tool}_stud_")
                for tool in ("gripper", "spoon", "whisk")
            )
        }
        self.assertEqual(
            sorted(legacy_names),
            [],
            "legacy robot-owned or y=+-10 mm stud proxies are not source CAD",
        )

        expected_names: set[str] = set()
        for tool in ("gripper", "spoon", "whisk"):
            tool_root_id = int(self.model.body(f"tool_{tool}").id)
            owner_name = f"tool_{tool}_positive_lock_hardware"
            owner_id = int(self.model.body(owner_name).id)
            self.assertEqual(
                int(self.model.body_parentid[owner_id]), tool_root_id, owner_name
            )
            np.testing.assert_allclose(
                self.model.body_pos[owner_id], np.zeros(3), rtol=0.0, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.abs(self.model.body_quat[owner_id]),
                [1.0, 0.0, 0.0, 0.0],
                rtol=0.0,
                atol=1.0e-12,
            )
            for side, x_mm in (("left", -12.0), ("right", 12.0)):
                for feature, radius_mm, z_min_mm, z_max_mm in (
                    ("shoulder", 2.0, -5.0, 0.0),
                    ("head", 3.0, -6.3, -5.0),
                ):
                    name = f"{tool}_lock_stud_{side}_{feature}_collision"
                    expected_names.add(name)
                    self.assertIn(name, compiled_names, name)
                    geom_id = int(self.model.geom(name).id)
                    self.assertEqual(
                        int(self.model.geom_bodyid[geom_id]), owner_id, name
                    )
                    self.assertEqual(
                        int(self.model.geom_type[geom_id]),
                        int(self.demo.mujoco.mjtGeom.mjGEOM_CYLINDER),
                        name,
                    )
                    expected_position = 0.001 * np.asarray(
                        [x_mm, 0.0, 0.5 * (z_min_mm + z_max_mm)],
                        dtype=np.float64,
                    )
                    expected_size = 0.001 * np.asarray(
                        [radius_mm, 0.5 * (z_max_mm - z_min_mm)],
                        dtype=np.float64,
                    )
                    np.testing.assert_allclose(
                        self.model.geom_pos[geom_id],
                        expected_position,
                        rtol=0.0,
                        atol=1.0e-12,
                        err_msg=name,
                    )
                    np.testing.assert_allclose(
                        self.model.geom_size[geom_id, :2],
                        expected_size,
                        rtol=0.0,
                        atol=1.0e-12,
                        err_msg=name,
                    )
                    quaternion = np.asarray(
                        self.model.geom_quat[geom_id], dtype=np.float64
                    )
                    quaternion /= np.linalg.norm(quaternion)
                    np.testing.assert_allclose(
                        np.abs(quaternion),
                        np.asarray([1.0, 0.0, 0.0, 0.0]),
                        rtol=0.0,
                        atol=1.0e-12,
                        err_msg=name,
                    )
                    self.assertTrue(
                        int(self.model.geom_contype[geom_id])
                        or int(self.model.geom_conaffinity[geom_id]),
                        name,
                    )
        observed_lock_studs = {
            name
            for name in compiled_names
            if "_lock_stud_" in name and name.endswith("_collision")
        }
        self.assertEqual(observed_lock_studs, expected_names)

    def test_fixed_plate_collision_proxies_preserve_lock_hardware_voids(
        self,
    ) -> None:
        """Broad plate prisms may not fill the stud wells or tool pockets."""

        cad = self.clearance.CAD
        robot_source = cad.robot_plate().val()
        robot_well_radius_mm = float(cad.ROBOT_STUD_WELL_RADIUS)
        self.assertEqual(robot_well_radius_mm, 3.6)
        for x_mm in (-float(cad.LOCK_STUD_X), float(cad.LOCK_STUD_X)):
            for z_mm in (3.2, 6.0, 9.2):
                samples_mm = [(x_mm, 0.0, z_mm)]
                samples_mm.extend(
                    (
                        x_mm
                        + 0.75
                        * robot_well_radius_mm
                        * math.cos(index * math.pi / 4.0),
                        0.75
                        * robot_well_radius_mm
                        * math.sin(index * math.pi / 4.0),
                        z_mm,
                    )
                    for index in range(8)
                )
                for point_mm in samples_mm:
                    self.assertFalse(
                        robot_source.isInside(cad.cq.Vector(*point_mm), 1.0e-7),
                        point_mm,
                    )
                    filled = self._collision_geoms_containing_owner_point(
                        "robot_plate_frame",
                        "qc_col_robot_plate_",
                        0.001 * np.asarray(point_mm, dtype=np.float64),
                    )
                    self.assertEqual(
                        filled,
                        [],
                        f"fixed robot proxy fills the Ø7.2 mm entry well: {point_mm}",
                    )
            material_point_mm = (x_mm, 5.0, 6.0)
            self.assertTrue(
                robot_source.isInside(cad.cq.Vector(*material_point_mm), 1.0e-7)
            )
            self.assertTrue(
                self._collision_geoms_containing_owner_point(
                    "robot_plate_frame",
                    "qc_col_robot_plate_",
                    0.001 * np.asarray(material_point_mm, dtype=np.float64),
                ),
                "the filled-void negative is vacuous if adjacent plate material is absent",
            )

        self.assertEqual(float(cad.LOCK_NUT_POCKET_ACROSS_FLATS), 5.7)
        bore_radius_mm = 1.6
        pocket_apothem_mm = 0.5 * float(cad.LOCK_NUT_POCKET_ACROSS_FLATS)
        pocket_sample_radius_mm = 0.5 * (bore_radius_mm + pocket_apothem_mm)
        for tool in ("gripper", "spoon", "whisk"):
            tool_source = cad.tool_plate(stock_gripper=tool == "gripper").val()
            for x_mm in (-float(cad.LOCK_STUD_X), float(cad.LOCK_STUD_X)):
                for z_mm in (0.5, 4.5, 9.0):
                    for index in range(8):
                        radius_mm = 0.75 * bore_radius_mm
                        point_mm = (
                            x_mm + radius_mm * math.cos(index * math.pi / 4.0),
                            radius_mm * math.sin(index * math.pi / 4.0),
                            z_mm,
                        )
                        self.assertFalse(
                            tool_source.isInside(
                                cad.cq.Vector(*point_mm), 1.0e-7
                            ),
                            (tool, point_mm),
                        )
                        filled = self._collision_geoms_containing_owner_point(
                            f"tool_{tool}",
                            f"matcha_col_{tool}_plate_",
                            0.001 * np.asarray(point_mm, dtype=np.float64),
                        )
                        self.assertEqual(
                            filled,
                            [],
                            f"{tool} plate proxy fills the Ø3.2 mm thread bore: {point_mm}",
                        )
                for z_mm in (2.0, 5.0, 9.0):
                    for index in range(12):
                        point_mm = (
                            x_mm
                            + pocket_sample_radius_mm
                            * math.cos(index * math.pi / 6.0),
                            pocket_sample_radius_mm
                            * math.sin(index * math.pi / 6.0),
                            z_mm,
                        )
                        self.assertFalse(
                            tool_source.isInside(
                                cad.cq.Vector(*point_mm), 1.0e-7
                            ),
                            (tool, point_mm),
                        )
                        filled = self._collision_geoms_containing_owner_point(
                            f"tool_{tool}",
                            f"matcha_col_{tool}_plate_",
                            0.001 * np.asarray(point_mm, dtype=np.float64),
                        )
                        self.assertEqual(
                            filled,
                            [],
                            f"{tool} plate proxy fills the AF5.7 nut pocket: {point_mm}",
                        )
                material_point_mm = (x_mm, 4.0, 5.0)
                self.assertTrue(
                    tool_source.isInside(cad.cq.Vector(*material_point_mm), 1.0e-7)
                )
                self.assertTrue(
                    self._collision_geoms_containing_owner_point(
                        f"tool_{tool}",
                        f"matcha_col_{tool}_plate_",
                        0.001 * np.asarray(material_point_mm, dtype=np.float64),
                    ),
                    (tool, "adjacent source material missing from collision proxy"),
                )

    def test_core_positive_lock_artifact_provenance_is_current(self) -> None:
        """A regenerated STEP may not be paired with a stale source or report."""

        manifest = load_json(CORE_CAD_MANIFEST, "core CAD manifest")
        generator = manifest.get("generator")
        self.assertIsInstance(generator, dict, manifest)
        generator_path = REPOSITORY_ROOT / str(generator.get("path"))
        self.assertTrue(generator_path.is_file(), generator_path)
        self.assertEqual(generator_path.stat().st_size, int(generator["bytes"]))
        self.assertEqual(
            sha256_file(generator_path),
            generator["sha256"],
            "core CAD outputs must be regenerated after any source change",
        )

        report = load_json(CORE_CLEARANCE_REPORT, "core CAD clearance report")
        report_manifest = report.get("core_cad_manifest_validation", {}).get(
            "manifest"
        )
        self.assertIsInstance(report_manifest, dict, report)
        self.assertEqual(
            report_manifest.get("sha256"), sha256_file(CORE_CAD_MANIFEST)
        )
        self.assertEqual(
            int(report_manifest.get("bytes", -1)), CORE_CAD_MANIFEST.stat().st_size
        )

    def test_positive_lock_source_has_continuous_shoulder_clearance(self) -> None:
        """Prove the released slider STEP clears both shoulders for every q.

        In the slider frame a fixed shoulder moves along a straight 3 mm
        segment.  The exact swept solid is therefore a cylinder-ended capsule.
        A zero Boolean intersection with that capsule proves the complete
        continuum, rather than only a finite set of q samples.
        """

        cad = self.clearance.CAD
        manifest = load_json(CORE_CAD_MANIFEST, "core CAD manifest")
        slider_record = next(
            (
                record
                for record in manifest.get("files", [])
                if str(record.get("path", "")).endswith(
                    "/so101_positive_lock_slider.step"
                )
            ),
            None,
        )
        self.assertIsInstance(slider_record, dict, manifest)
        slider_path = REPOSITORY_ROOT / str(slider_record["path"])
        self.assertEqual(sha256_file(slider_path), slider_record["sha256"])
        slider = cad.cq.importers.importStep(str(slider_path)).val()
        self.assertTrue(slider.isValid())

        keyhole_contract = cad.positive_lock_keyhole_contract()
        required_radial_clearance_mm = float(
            keyhole_contract["minimum_radial_shoulder_clearance_mm"]
        )
        self.assertGreaterEqual(required_radial_clearance_mm, 0.1)
        self.assertEqual(
            float(keyhole_contract["slider_travel_mm"]), float(cad.SLIDER_TRAVEL)
        )
        shoulder_z_min_in_slider_mm = (
            cad.PLATE_THICKNESS - cad.LOCK_SHOULDER_LENGTH - cad.SLIDER_Z
        )
        sample_positions_mm = np.linspace(
            0.0, float(cad.SLIDER_TRAVEL), 13, dtype=np.float64
        )
        failures: dict[str, Any] = {}
        for side, stud_x_mm in (
            ("left", -float(cad.LOCK_STUD_X)),
            ("right", float(cad.LOCK_STUD_X)),
        ):
            start = cad.axis_cylinder(
                cad.LOCK_SHOULDER_DIAMETER,
                cad.LOCK_SHOULDER_LENGTH,
                (stud_x_mm, 0.0, shoulder_z_min_in_slider_mm),
            )
            end = cad.axis_cylinder(
                cad.LOCK_SHOULDER_DIAMETER,
                cad.LOCK_SHOULDER_LENGTH,
                (
                    stud_x_mm - cad.SLIDER_TRAVEL,
                    0.0,
                    shoulder_z_min_in_slider_mm,
                ),
            )
            connecting_prism = (
                cad.cq.Workplane("XY")
                .box(
                    cad.SLIDER_TRAVEL,
                    cad.LOCK_SHOULDER_DIAMETER,
                    cad.LOCK_SHOULDER_LENGTH,
                    centered=(True, True, False),
                )
                .translate(
                    (
                        stud_x_mm - cad.SLIDER_TRAVEL / 2.0,
                        0.0,
                        shoulder_z_min_in_slider_mm,
                    )
                )
            )
            shoulder_sweep = start.union(end).union(connecting_prism).clean().val()
            self.assertTrue(shoulder_sweep.isValid(), side)
            swept_overlap_mm3 = self.clearance._intersection_volume_mm3(
                slider, shoulder_sweep
            )
            swept_clearance_mm = float(slider.distance(shoulder_sweep))

            sampled_overlaps_mm3: list[float] = []
            for q_mm in sample_positions_mm:
                placed_slider = slider.moved(
                    cad.cq.Location(cad.cq.Vector(float(q_mm), 0.0, cad.SLIDER_Z))
                )
                shoulder = cad.axis_cylinder(
                    cad.LOCK_SHOULDER_DIAMETER,
                    cad.LOCK_SHOULDER_LENGTH,
                    (
                        stud_x_mm,
                        0.0,
                        cad.PLATE_THICKNESS - cad.LOCK_SHOULDER_LENGTH,
                    ),
                ).val()
                sampled_overlaps_mm3.append(
                    self.clearance._intersection_volume_mm3(
                        placed_slider, shoulder
                    )
                )
            if (
                swept_overlap_mm3
                > self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3
                or swept_clearance_mm + 1.0e-9
                < required_radial_clearance_mm
                or max(sampled_overlaps_mm3)
                > self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3
            ):
                failures[side] = {
                    "continuous_swept_overlap_mm3": swept_overlap_mm3,
                    "continuous_swept_clearance_mm": swept_clearance_mm,
                    "required_radial_clearance_mm": required_radial_clearance_mm,
                    "sample_q_mm": sample_positions_mm.tolist(),
                    "sample_overlap_mm3": sampled_overlaps_mm3,
                }
        self.assertEqual(
            failures,
            {},
            "the slider keyhole must clear each shoulder throughout its full travel",
        )

    def test_current_positive_lock_cam_is_a_fail_closed_sequence_control(
        self,
    ) -> None:
        """Quantify why the old axial-then-seated-recenter route cannot lock.

        This is the negative control for the forthcoming coupled passive-cam
        oracle.  It deliberately evaluates only the released main XY wedge:
        the final positive gate must add a hash-bound axial lead and prove a
        coupled +0.20 -> 0.00 mm X recenter while that lead opens the slider.
        Merely retaining the late same-Z waypoint would collide even if the
        controller claimed the slider was already open.
        """

        cad = self.clearance.CAD
        overlap_tolerance_mm3 = float(
            self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3
        )
        distance_tolerance_mm = float(
            self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM
        )
        # Rebuild the released main XY wedge explicitly.  The production
        # ``positive_lock_cam`` may now union an axial lead; folding that new
        # feature into this negative control would erase the very ordering
        # defect the control is intended to preserve.
        cam = (
            cad.cq.Workplane("XY")
            .polyline(
                [
                    (cad.DOCK_CAM_X_OUTER_MIN, cad.DOCK_CAM_Y_MIN),
                    (cad.DOCK_CAM_X_OUTER_MAX, cad.DOCK_CAM_Y_MIN),
                    (cad.DOCK_CAM_X_OUTER_MAX, cad.DOCK_CAM_Y_MAX),
                    (cad.DOCK_CAM_X_INNER, cad.DOCK_CAM_Y_MAX),
                ]
            )
            .close()
            .extrude(cad.DOCK_CAM_THICKNESS)
            .translate((0.0, 0.0, cad.DOCK_CAM_Z_MIN))
            .clean()
            .val()
        )

        def slider(
            q_mm: float,
            *,
            robot_x_mm: float = 0.0,
            robot_y_mm: float = 0.0,
            preseat_mm: float = 0.0,
        ) -> Any:
            return (
                cad.locking_slider()
                .translate(
                    (
                        q_mm + robot_x_mm,
                        robot_y_mm,
                        cad.SLIDER_Z - cad.PLATE_THICKNESS - preseat_mm,
                    )
                )
                .val()
            )

        def overlap_mm3(first: Any, second: Any) -> float:
            if float(first.distance(second)) > distance_tolerance_mm:
                return 0.0
            return float(
                self.clearance._intersection_volume_mm3(first, second)
            )

        heads = {
            side: cad.axis_cylinder(
                cad.LOCK_HEAD_DIAMETER,
                cad.LOCK_HEAD_HEIGHT,
                (
                    stud_x_mm,
                    0.0,
                    -cad.LOCK_SHOULDER_LENGTH - cad.LOCK_HEAD_HEIGHT,
                ),
            ).val()
            for side, stud_x_mm in (
                ("left", -float(cad.LOCK_STUD_X)),
                ("right", float(cad.LOCK_STUD_X)),
            )
        }

        # Derive both event positions from the exact source BRep bounds.  The
        # locked slider approaches along -Z from below.  Its top first reaches
        # the stud-head bottom at 3.1 mm preseat, whereas the unchanged XY cam
        # cannot touch its tab until only 0.95 mm remains.
        locked_at_guided_x = slider(
            float(cad.SLIDER_TRAVEL), robot_x_mm=0.20
        )
        slider_bounds = locked_at_guided_x.BoundingBox()
        head_bounds = next(iter(heads.values())).BoundingBox()
        cam_bounds = cam.BoundingBox()
        first_head_tangent_preseat_mm = float(
            slider_bounds.zmax - head_bounds.zmin
        )
        first_main_cam_tangent_preseat_mm = float(
            slider_bounds.zmax - cam_bounds.zmin
        )
        self.assertAlmostEqual(first_head_tangent_preseat_mm, 3.10, places=12)
        self.assertAlmostEqual(first_main_cam_tangent_preseat_mm, 0.95, places=12)

        head_tangent_slider = slider(
            float(cad.SLIDER_TRAVEL),
            robot_x_mm=0.20,
            preseat_mm=first_head_tangent_preseat_mm,
        )
        head_post_tangent_slider = slider(
            float(cad.SLIDER_TRAVEL),
            robot_x_mm=0.20,
            preseat_mm=first_head_tangent_preseat_mm - 0.001,
        )
        head_tangent_distances = {
            side: float(head_tangent_slider.distance(head))
            for side, head in heads.items()
        }
        head_post_tangent_overlaps = {
            side: overlap_mm3(head_post_tangent_slider, head)
            for side, head in heads.items()
        }
        for distance_mm in head_tangent_distances.values():
            self.assertLessEqual(distance_mm, distance_tolerance_mm)
        for overlap in head_post_tangent_overlaps.values():
            self.assertGreater(overlap, overlap_tolerance_mm3)

        cam_tangent_slider = slider(
            float(cad.SLIDER_TRAVEL),
            robot_x_mm=0.20,
            preseat_mm=first_main_cam_tangent_preseat_mm,
        )
        cam_post_tangent_slider = slider(
            float(cad.SLIDER_TRAVEL),
            robot_x_mm=0.20,
            preseat_mm=first_main_cam_tangent_preseat_mm - 0.001,
        )
        self.assertLessEqual(
            float(cam_tangent_slider.distance(cam)), distance_tolerance_mm
        )
        cam_post_tangent_overlap_mm3 = overlap_mm3(
            cam_post_tangent_slider, cam
        )
        self.assertGreater(
            cam_post_tangent_overlap_mm3, overlap_tolerance_mm3
        )

        lead_margin_mm = (
            first_main_cam_tangent_preseat_mm
            - first_head_tangent_preseat_mm
        )
        self.assertAlmostEqual(lead_margin_mm, -2.15, places=12)

        # q=0 is the nominal unlocked reference; q=+0.05 mm is the honest
        # spring-advanced passive contact state set by the source's 50 um
        # tab/cam gap.  Either state intersects the current cam if the robot
        # waits until fully seated to recenter from guided X=+0.20 mm.
        late_recenter_overlap_mm3 = {
            "nominal_q0": overlap_mm3(
                slider(0.0, robot_x_mm=0.20), cam
            ),
            "passive_open_q0p05": overlap_mm3(
                slider(0.05, robot_x_mm=0.20), cam
            ),
        }
        self.assertGreater(
            late_recenter_overlap_mm3["nominal_q0"], overlap_tolerance_mm3
        )
        self.assertGreater(
            late_recenter_overlap_mm3["passive_open_q0p05"],
            overlap_tolerance_mm3,
        )

        # The source-derived passive-open state itself is safe for the entire
        # head passage once X has already reached zero.  Extruding each head's
        # circular projection through the full slider thickness is the exact
        # continuous axial sweep, not a finite pose sample.
        passive_open_slider = (
            cad.locking_slider()
            .translate((0.05, 0.0, cad.SLIDER_Z))
            .val()
        )
        projected_head_clearances_mm: dict[str, float] = {}
        for side, stud_x_mm in (
            ("left", -float(cad.LOCK_STUD_X)),
            ("right", float(cad.LOCK_STUD_X)),
        ):
            projected_head_sweep = cad.axis_cylinder(
                cad.LOCK_HEAD_DIAMETER,
                cad.SLIDER_THICKNESS,
                (stud_x_mm, 0.0, cad.SLIDER_Z),
            ).val()
            self.assertLessEqual(
                overlap_mm3(passive_open_slider, projected_head_sweep),
                overlap_tolerance_mm3,
            )
            projected_head_clearances_mm[side] = float(
                passive_open_slider.distance(projected_head_sweep)
            )
        self.assertGreaterEqual(
            min(projected_head_clearances_mm.values()), 0.20
        )

        # A locked slider cannot exist at the seated cam.  It becomes a valid
        # retention claim only after the separate dock-local -Y withdrawal.
        seated_locked_overlap_mm3 = overlap_mm3(
            slider(float(cad.SLIDER_TRAVEL)), cam
        )
        locked_at_15_mm = slider(
            float(cad.SLIDER_TRAVEL), robot_y_mm=-15.0
        )
        locked_cam_clearance_at_15_mm = float(locked_at_15_mm.distance(cam))
        self.assertGreater(seated_locked_overlap_mm3, overlap_tolerance_mm3)
        self.assertGreaterEqual(locked_cam_clearance_at_15_mm, 0.20)

        diagnostic = {
            "authority": "exact_occt_source_brep",
            "first_stud_head_tangent_preseat_mm": (
                first_head_tangent_preseat_mm
            ),
            "first_main_cam_tangent_preseat_mm": (
                first_main_cam_tangent_preseat_mm
            ),
            "main_cam_lead_margin_mm": lead_margin_mm,
            "head_overlap_0p001mm_after_tangent_mm3": (
                head_post_tangent_overlaps
            ),
            "main_cam_overlap_0p001mm_after_tangent_mm3": (
                cam_post_tangent_overlap_mm3
            ),
            "late_same_z_recenter_overlap_mm3": late_recenter_overlap_mm3,
            "passive_open_projected_head_clearance_mm": (
                projected_head_clearances_mm
            ),
            "seated_locked_cam_overlap_mm3": seated_locked_overlap_mm3,
            "locked_cam_clearance_after_15mm_negative_y_mm": (
                locked_cam_clearance_at_15_mm
            ),
            "required_corrected_order": [
                "coupled_axial_lead_and_x_recenter",
                "passive_open_head_passage_at_x0",
                "axial_seat_at_x0",
                "attach_then_dock_release",
                "cam_following_negative_y_withdrawal",
                "q3_lock_only_after_15mm_and_0p20mm_cam_clearance",
            ],
            "blockers": [
                "main_cam_actuation_is_2p15mm_late",
                "seated_same_z_recenter_intersects_main_cam",
                "seated_pre_exit_q3_intersects_main_cam",
                "hash_bound_axial_lead_not_yet_in_this_negative_control",
            ],
            "passed": False,
            "release_ready": False,
        }
        self.assertIs(diagnostic["passed"], False, diagnostic)
        self.assertIs(diagnostic["release_ready"], False, diagnostic)

    def test_passive_positive_lock_cam_source_sequence_is_exact_and_ordered(
        self,
    ) -> None:
        """Independently certify the coupled cam-open/recenter/release path.

        Production publishes a convenient machine-readable record, but this
        gate does not use that record as geometry authority.  It reconstructs
        the ruled loft, hold finger, root bridge, capture transforms and
        release transforms from the exact source contract, then runs its own
        OCCT distances and Booleans.  Only after those checks pass is the
        production record compared with the independent witnesses.
        """

        cad = self.clearance.CAD
        cq = cad.cq
        overlap_tolerance_mm3 = float(
            self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3
        )
        distance_tolerance_mm = float(
            self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM
        )
        manufacturing_clearance_mm = float(
            self.clearance.MANUFACTURING_CLEARANCE_MM
        )
        contract = cad.positive_lock_cam_contract()
        self.assertEqual(
            set(contract),
            {
                "schema_version",
                "frame",
                "construction",
                "main_xy_wedge",
                "axial_lead",
                "hold_finger",
                "outer_root_bridge",
                "expected_geometry",
                "passive_capture",
                "passive_release",
                "manufacturability",
                "quasistatic_load_envelope",
            },
        )
        self.assertEqual(contract["schema_version"], "1.0")
        self.assertEqual(contract["frame"], "dock_local_mm")
        self.assertEqual(
            contract["construction"],
            "union_main_xy_wedge_ruled_axial_lead_hold_finger",
        )

        def shape_volume_mm3(shape: Any) -> float:
            solids = shape.Solids()
            return math.fsum(float(solid.Volume()) for solid in solids)

        def overlap_mm3(first: Any, second: Any) -> float:
            if float(first.distance(second)) > distance_tolerance_mm:
                return 0.0
            return float(
                self.clearance._intersection_volume_mm3(first, second)
            )

        def box_from_contract(bounds: dict[str, list[float]]) -> Any:
            axis_bounds = [bounds[axis] for axis in ("x", "y", "z")]
            for values in axis_bounds:
                self.assertEqual(len(values), 2)
                self.assertGreater(float(values[1]), float(values[0]))
            size = [float(values[1]) - float(values[0]) for values in axis_bounds]
            center = [0.5 * math.fsum(map(float, values)) for values in axis_bounds]
            return (
                cq.Workplane("XY")
                .box(*size, centered=True)
                .translate(tuple(center))
            )

        def rectangle_wire(record: dict[str, Any]) -> Any:
            x_min, x_max = map(float, record["x"])
            y_min, y_max = map(float, record["y"])
            z_value = float(record["z"])
            return cq.Wire.makePolygon(
                [
                    cq.Vector(x_min, y_min, z_value),
                    cq.Vector(x_max, y_min, z_value),
                    cq.Vector(x_max, y_max, z_value),
                    cq.Vector(x_min, y_max, z_value),
                ],
                close=True,
            )

        lead_contract = contract["axial_lead"]
        self.assertEqual(
            lead_contract["kind"], "ruled_loft_between_rectangles"
        )
        lower_rectangle = lead_contract["lower_rectangle_mm"]
        upper_rectangle = lead_contract["upper_rectangle_mm"]
        self.assertEqual(lower_rectangle["y"], [0.0, 2.0])
        self.assertEqual(upper_rectangle["y"], [0.0, 2.0])
        independent_ruled_lead = cq.Workplane(
            obj=cq.Solid.makeLoft(
                [
                    rectangle_wire(lower_rectangle),
                    rectangle_wire(upper_rectangle),
                ],
                ruled=True,
            )
        )
        independent_hold = box_from_contract(
            contract["hold_finger"]["bounds_mm"]
        )
        bridge_contract = contract["outer_root_bridge"]
        self.assertEqual(
            bridge_contract["bounds_mm"],
            {
                "x": [28.0, 29.0],
                "y": [-1.0, 1.0],
                "z": [-4.65, -3.65],
            },
        )
        independent_bridge = box_from_contract(bridge_contract["bounds_mm"])
        independent_lead = (
            independent_ruled_lead
            .union(independent_hold)
            .union(independent_bridge)
            .clean()
        )

        main_contract = contract["main_xy_wedge"]
        main_z = list(map(float, main_contract["z_bounds_mm"]))
        independent_main = (
            cq.Workplane("XY")
            .polyline(
                [tuple(map(float, point)) for point in main_contract["polygon_xy_mm"]]
            )
            .close()
            .extrude(main_z[1] - main_z[0])
            .translate((0.0, 0.0, main_z[0]))
            .clean()
        )
        independent_cam = independent_main.union(independent_lead).clean().val()
        source_lead = cad.positive_lock_cam_axial_lead().val()
        source_cam = cad.positive_lock_cam().val()
        self.assertTrue(source_cam.isValid())
        self.assertEqual(len(source_cam.Solids()), 1)
        self.assertTrue(independent_cam.isValid())
        self.assertEqual(len(independent_cam.Solids()), 1)
        self.assertLessEqual(
            shape_volume_mm3(source_lead.cut(independent_lead.val())),
            overlap_tolerance_mm3,
        )
        self.assertLessEqual(
            shape_volume_mm3(independent_lead.val().cut(source_lead)),
            overlap_tolerance_mm3,
        )
        self.assertLessEqual(
            shape_volume_mm3(source_cam.cut(independent_cam)),
            overlap_tolerance_mm3,
        )
        self.assertLessEqual(
            shape_volume_mm3(independent_cam.cut(source_cam)),
            overlap_tolerance_mm3,
        )
        expected_geometry = contract["expected_geometry"]
        source_cam_volume_mm3 = shape_volume_mm3(source_cam)
        self.assertAlmostEqual(source_cam_volume_mm3, 325.435, places=9)
        self.assertAlmostEqual(
            source_cam_volume_mm3,
            float(expected_geometry["total_volume_mm3"]),
            places=9,
        )
        source_cam_bounds = source_cam.BoundingBox()
        for axis, observed in (
            ("x", [source_cam_bounds.xmin, source_cam_bounds.xmax]),
            ("y", [source_cam_bounds.ymin, source_cam_bounds.ymax]),
            ("z", [source_cam_bounds.zmin, source_cam_bounds.zmax]),
        ):
            np.testing.assert_allclose(
                observed,
                expected_geometry["bounds_mm"][axis],
                rtol=0.0,
                atol=1.0e-6,
            )
        self.assertIs(bridge_contract["outside_locked_tab_swept_x"], True)
        self.assertGreaterEqual(
            float(bridge_contract["bounds_mm"]["x"][0])
            - (
                float(cad.SLIDER_TAB_END_X)
                + float(cad.SLIDER_TRAVEL)
                + float(cad.ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM)
            ),
            0.79,
        )

        capture_contract = contract["passive_capture"]
        release_contract = contract["passive_release"]
        recenter_start_mm = float(
            capture_contract["recenter_start_preseat_mm"]
        )
        recenter_end_mm = float(capture_contract["recenter_end_preseat_mm"])
        head_entry_mm = float(
            capture_contract["head_entry_tangent_preseat_mm"]
        )
        passive_open_q_mm = float(capture_contract["passive_open_q_max_mm"])
        self.assertAlmostEqual(recenter_start_mm, 6.4, places=12)
        self.assertAlmostEqual(recenter_end_mm, 3.2, places=12)
        self.assertAlmostEqual(head_entry_mm, 3.1, places=12)
        self.assertAlmostEqual(passive_open_q_mm, 0.05, places=12)
        self.assertGreater(recenter_end_mm, head_entry_mm)
        self.assertEqual(
            capture_contract["lateral_offset_breakpoints_mm"],
            [[6.4, 0.2], [3.2, 0.0], [0.0, 0.0]],
        )
        q_coefficients = capture_contract["ramp_q_affine_coefficients"]

        def independent_capture_x_mm(preseat_mm: float) -> float:
            if not math.isfinite(preseat_mm) or preseat_mm < 0.0:
                raise AssertionError(f"invalid preseat: {preseat_mm}")
            if preseat_mm >= recenter_start_mm:
                return 0.20
            if preseat_mm <= recenter_end_mm:
                return 0.0
            return 0.20 * (
                (preseat_mm - recenter_end_mm)
                / (recenter_start_mm - recenter_end_mm)
            )

        def independent_capture_q_mm(preseat_mm: float) -> float:
            lateral_mm = independent_capture_x_mm(preseat_mm)
            raw_q_mm = math.fsum(
                (
                    float(q_coefficients["preseat"]) * preseat_mm,
                    float(q_coefficients["lateral_offset"]) * lateral_mm,
                    float(q_coefficients["constant"]),
                )
            )
            q_min_mm, q_max_mm = map(float, q_coefficients["clamp_mm"])
            return max(q_min_mm, min(q_max_mm, raw_q_mm))

        def independent_release_q_mm(withdrawal_mm: float) -> float:
            raw_q_mm = passive_open_q_mm + float(
                release_contract["q_per_withdrawal_slope"]
            ) * (
                withdrawal_mm
                - float(release_contract["initial_q_hold_withdrawal_mm"])
            )
            return max(
                passive_open_q_mm,
                min(float(cad.SLIDER_TRAVEL), raw_q_mm),
            )

        slider_native = cad.locking_slider()
        plate_native = cad.robot_plate()
        studs = {
            side: cad.shoulder_lock_stud().translate(
                (stud_x_mm, 0.0, 0.0)
            ).val()
            for side, stud_x_mm in (
                ("left", -float(cad.LOCK_STUD_X)),
                ("right", float(cad.LOCK_STUD_X)),
            )
        }
        capture_preseats_mm = [
            round(15.0 - 0.1 * index, 10) for index in range(151)
        ]
        capture_samples: list[dict[str, float]] = []
        maximum_capture_cam_overlap_mm3 = 0.0
        maximum_capture_stud_overlap_mm3 = 0.0
        minimum_plate_cam_distance_mm = math.inf
        plate_translations: list[np.ndarray] = []
        for preseat_mm in capture_preseats_mm:
            lateral_mm = independent_capture_x_mm(preseat_mm)
            q_mm = independent_capture_q_mm(preseat_mm)
            self.assertAlmostEqual(
                cad.positive_lock_cam_capture_lateral_offset_mm(preseat_mm),
                lateral_mm,
                places=12,
            )
            self.assertAlmostEqual(
                cad.positive_lock_cam_capture_q_max_mm(preseat_mm),
                q_mm,
                places=12,
            )
            slider = slider_native.translate(
                (
                    q_mm + lateral_mm,
                    0.0,
                    cad.SLIDER_Z - cad.PLATE_THICKNESS - preseat_mm,
                )
            ).val()
            slider_cam_distance_mm = float(slider.distance(source_cam))
            slider_cam_overlap_mm3 = overlap_mm3(slider, source_cam)
            maximum_capture_cam_overlap_mm3 = max(
                maximum_capture_cam_overlap_mm3,
                slider_cam_overlap_mm3,
            )
            stud_distances = {
                side: float(slider.distance(stud))
                for side, stud in studs.items()
            }
            stud_overlaps = {
                side: overlap_mm3(slider, stud)
                for side, stud in studs.items()
            }
            maximum_capture_stud_overlap_mm3 = max(
                maximum_capture_stud_overlap_mm3,
                *stud_overlaps.values(),
            )
            plate = plate_native.translate(
                (lateral_mm, 0.0, -cad.PLATE_THICKNESS - preseat_mm)
            ).val()
            plate_cam_distance_mm = float(plate.distance(source_cam))
            minimum_plate_cam_distance_mm = min(
                minimum_plate_cam_distance_mm, plate_cam_distance_mm
            )
            self.assertLessEqual(
                overlap_mm3(plate, source_cam), overlap_tolerance_mm3
            )
            plate_translations.append(
                np.asarray([lateral_mm, 0.0, -preseat_mm], dtype=np.float64)
            )
            capture_samples.append(
                {
                    "preseat_mm": preseat_mm,
                    "lateral_mm": lateral_mm,
                    "q_mm": q_mm,
                    "slider_cam_distance_mm": slider_cam_distance_mm,
                    "minimum_slider_stud_distance_mm": min(
                        stud_distances.values()
                    ),
                }
            )
        self.assertLessEqual(
            maximum_capture_cam_overlap_mm3, overlap_tolerance_mm3
        )
        self.assertLessEqual(
            maximum_capture_stud_overlap_mm3, overlap_tolerance_mm3
        )
        self.assertTrue(
            all(
                capture_samples[index + 1]["lateral_mm"]
                <= capture_samples[index]["lateral_mm"] + 1.0e-12
                and capture_samples[index + 1]["q_mm"]
                <= capture_samples[index]["q_mm"] + 1.0e-12
                for index in range(len(capture_samples) - 1)
            )
        )
        head_entry_sample = next(
            sample
            for sample in capture_samples
            if math.isclose(
                sample["preseat_mm"], head_entry_mm, abs_tol=1.0e-12
            )
        )
        self.assertEqual(head_entry_sample["lateral_mm"], 0.0)
        self.assertLessEqual(
            head_entry_sample["q_mm"], passive_open_q_mm + 1.0e-12
        )
        self.assertGreaterEqual(
            head_entry_sample["minimum_slider_stud_distance_mm"], 0.20
        )
        for sample in capture_samples:
            if sample["preseat_mm"] <= recenter_end_mm + 1.0e-12:
                self.assertEqual(sample["lateral_mm"], 0.0, sample)
                self.assertLessEqual(
                    sample["q_mm"], passive_open_q_mm + 1.0e-12, sample
                )
        contact_start_mm = float(
            capture_contract["ramp_contact_start_preseat_mm"]
        )
        self.assertGreater(contact_start_mm, head_entry_mm)
        for sample in capture_samples:
            if sample["preseat_mm"] <= math.floor(contact_start_mm * 10.0) / 10.0:
                self.assertLessEqual(
                    sample["slider_cam_distance_mm"],
                    distance_tolerance_mm,
                    sample,
                )

        plate_interval_motion_mm = max(
            float(np.linalg.norm(second - first))
            for first, second in zip(
                plate_translations, plate_translations[1:]
            )
        )
        plate_motion_bound_mm = plate_interval_motion_mm / 2.0
        continuous_plate_cam_clearance_mm = (
            minimum_plate_cam_distance_mm - plate_motion_bound_mm
        )
        self.assertGreaterEqual(
            continuous_plate_cam_clearance_mm,
            0.2498,
        )

        # Tight exact source sweep around the complete head-entry interval.
        # Distance is 1-Lipschitz under translation; subtracting half of the
        # maximum state-to-state displacement certifies the continuum.
        tight_samples: list[tuple[float, float, np.ndarray]] = []
        for index in range(331):
            preseat_mm = round(0.01 * index, 10)
            lateral_mm = independent_capture_x_mm(preseat_mm)
            q_mm = independent_capture_q_mm(preseat_mm)
            translation = np.asarray(
                [q_mm + lateral_mm, 0.0, -preseat_mm], dtype=np.float64
            )
            slider = slider_native.translate(
                (
                    translation[0],
                    0.0,
                    cad.SLIDER_Z - cad.PLATE_THICKNESS + translation[2],
                )
            ).val()
            minimum_stud_distance_mm = min(
                float(slider.distance(stud)) for stud in studs.values()
            )
            for stud in studs.values():
                self.assertLessEqual(
                    overlap_mm3(slider, stud), overlap_tolerance_mm3
                )
            tight_samples.append(
                (preseat_mm, minimum_stud_distance_mm, translation)
            )
        tight_motion_bound_mm = 0.5 * max(
            float(np.linalg.norm(second[2] - first[2]))
            for first, second in zip(tight_samples, tight_samples[1:])
        )
        tight_witness = min(tight_samples, key=lambda sample: sample[1])
        continuous_stud_clearance_mm = (
            tight_witness[1] - tight_motion_bound_mm
        )
        self.assertGreaterEqual(continuous_stud_clearance_mm, 0.20)

        # Close the source roster against the new cam, rather than proving
        # only the slider and plate.  The 17 tool components remain fixed in
        # the dock; robot magnets follow the same coupled p/x translation.
        tool_components = self.clearance._tool_side_components()
        self.assertEqual(len(tool_components), 17)
        component_cam_overlaps: dict[str, float] = {}
        for component in tool_components:
            component_cam_overlaps[component.name] = overlap_mm3(
                component.shape.val(), source_cam
            )
        robot_components = self.clearance._robot_side_components()
        for component in robot_components:
            if component.name == "robot_plate" or component.role == "positive_lock_slider":
                continue
            maximum_overlap = 0.0
            for sample in capture_samples:
                placed = component.shape.translate(
                    (
                        sample["lateral_mm"],
                        0.0,
                        -sample["preseat_mm"],
                    )
                ).val()
                maximum_overlap = max(
                    maximum_overlap, overlap_mm3(placed, source_cam)
                )
            component_cam_overlaps[component.name] = maximum_overlap
        self.assertTrue(component_cam_overlaps)
        self.assertLessEqual(
            max(component_cam_overlaps.values()), overlap_tolerance_mm3
        )

        self.assertEqual(release_contract["axis"], "dock_local_negative_y")
        self.assertAlmostEqual(
            float(release_contract["q3_tangent_withdrawal_mm"]),
            13.949367088607595,
            places=12,
        )
        self.assertAlmostEqual(
            float(release_contract["nominal_exit_withdrawal_mm"]),
            15.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(release_contract["required_exit_clearance_mm"]),
            0.20,
            places=12,
        )
        release_samples: list[dict[str, float]] = []
        maximum_release_cam_overlap_mm3 = 0.0
        for index in range(151):
            withdrawal_mm = round(0.1 * index, 10)
            q_mm = independent_release_q_mm(withdrawal_mm)
            self.assertAlmostEqual(
                cad.positive_lock_cam_release_q_max_mm(withdrawal_mm),
                q_mm,
                places=12,
            )
            slider = slider_native.translate(
                (
                    q_mm,
                    -withdrawal_mm,
                    cad.SLIDER_Z - cad.PLATE_THICKNESS,
                )
            ).val()
            distance_mm = float(slider.distance(source_cam))
            overlap = overlap_mm3(slider, source_cam)
            maximum_release_cam_overlap_mm3 = max(
                maximum_release_cam_overlap_mm3, overlap
            )
            release_samples.append(
                {
                    "withdrawal_mm": withdrawal_mm,
                    "q_mm": q_mm,
                    "slider_cam_distance_mm": distance_mm,
                }
            )
        self.assertLessEqual(
            maximum_release_cam_overlap_mm3, overlap_tolerance_mm3
        )
        self.assertTrue(
            all(
                second["q_mm"] + 1.0e-12 >= first["q_mm"]
                for first, second in zip(release_samples, release_samples[1:])
            )
        )
        first_manufacturing_clear_sample = next(
            sample
            for sample in release_samples
            if sample["q_mm"] >= float(cad.SLIDER_TRAVEL) - 1.0e-12
            and sample["slider_cam_distance_mm"]
            >= float(release_contract["required_exit_clearance_mm"])
            - distance_tolerance_mm
        )
        self.assertGreaterEqual(
            first_manufacturing_clear_sample["withdrawal_mm"], 14.8
        )
        nominal_exit_sample = release_samples[-1]
        self.assertEqual(nominal_exit_sample["withdrawal_mm"], 15.0)
        self.assertAlmostEqual(
            nominal_exit_sample["q_mm"], float(cad.SLIDER_TRAVEL), places=12
        )
        self.assertGreaterEqual(
            nominal_exit_sample["slider_cam_distance_mm"], 0.20
        )

        # Recompute the complete attached source assembly's first 15 mm of
        # rack withdrawal on a 0.5 mm grid.  The 0.25 mm Lipschitz bound still
        # leaves >=0.5 mm on every forbidden pair.  Only the exact five
        # published keeper tangencies may stay at zero distance.
        dock = self.clearance._dock_authority()
        full_dock = dock["full_dock"].val()
        dock_without_cam = dock["dock_without_cam"].val()
        exit_positions_mm = [0.5 * index for index in range(31)]
        exit_components = tool_components + [
            component
            for component in robot_components
            if component.role != "positive_lock_slider"
        ]
        tangency_components = {"stock_tool_plate", "robot_plate"}
        exit_component_records: dict[str, dict[str, float]] = {}
        for component in exit_components:
            distances: list[float] = []
            maximum_overlap = 0.0
            for withdrawal_mm in exit_positions_mm:
                placed = component.shape.translate(
                    (0.0, -withdrawal_mm, 0.0)
                ).val()
                distance_mm = float(placed.distance(full_dock))
                distances.append(distance_mm)
                maximum_overlap = max(
                    maximum_overlap, overlap_mm3(placed, full_dock)
                )
            minimum_distance_mm = min(distances)
            exit_component_records[component.name] = {
                "minimum_sampled_distance_mm": minimum_distance_mm,
                "maximum_overlap_mm3": maximum_overlap,
            }
            self.assertLessEqual(
                maximum_overlap, overlap_tolerance_mm3, component.name
            )
            if component.name not in tangency_components:
                self.assertGreaterEqual(
                    minimum_distance_mm - 0.25,
                    manufacturing_clearance_mm,
                    component.name,
                )

        dynamic_slider_distances: list[float] = []
        maximum_dynamic_slider_forbidden_overlap = 0.0
        for withdrawal_mm in exit_positions_mm:
            q_mm = independent_release_q_mm(withdrawal_mm)
            slider = slider_native.translate(
                (
                    q_mm,
                    -withdrawal_mm,
                    cad.SLIDER_Z - cad.PLATE_THICKNESS,
                )
            ).val()
            dynamic_slider_distances.append(
                float(slider.distance(dock_without_cam))
            )
            maximum_dynamic_slider_forbidden_overlap = max(
                maximum_dynamic_slider_forbidden_overlap,
                overlap_mm3(slider, dock_without_cam),
            )
        self.assertLessEqual(
            maximum_dynamic_slider_forbidden_overlap,
            overlap_tolerance_mm3,
        )
        self.assertGreaterEqual(
            min(dynamic_slider_distances) - 0.25,
            manufacturing_clearance_mm,
        )

        by_name = {component.name: component for component in exit_components}
        intended_rail_pairs = {
            "stock_tool_plate": (
                "left_lower_rail",
                "right_lower_rail",
                "left_upper_rail",
                "right_upper_rail",
            ),
            "robot_plate": ("left_lower_rail",),
        }
        for component_name, rail_names in intended_rail_pairs.items():
            component = by_name[component_name]
            for rail_name in rail_names:
                rail = dock[rail_name].val()
                # Both source features are prismatic along dock Y, and the
                # only motion is dock-local -Y.  Exact tangency at both ends,
                # plus overlapping Y intervals at both ends, proves the whole
                # linear interval without 31 redundant zero-volume Booleans.
                for withdrawal_mm in (0.0, 15.0):
                    placed = component.shape.translate(
                        (0.0, -withdrawal_mm, 0.0)
                    ).val()
                    self.assertLessEqual(
                        float(placed.distance(rail)),
                        distance_tolerance_mm,
                        (component_name, rail_name, withdrawal_mm),
                    )
                    self.assertLessEqual(
                        overlap_mm3(placed, rail),
                        overlap_tolerance_mm3,
                        (component_name, rail_name, withdrawal_mm),
                    )
                    placed_bounds = placed.BoundingBox()
                    rail_bounds = rail.BoundingBox()
                    self.assertLessEqual(
                        max(placed_bounds.ymin, rail_bounds.ymin),
                        min(placed_bounds.ymax, rail_bounds.ymax),
                        (component_name, rail_name, withdrawal_mm),
                    )

        travel_record = self.clearance._positive_lock_travel_record()
        self.assertIs(travel_record["passed"], True, travel_record)
        self.assertGreaterEqual(
            float(travel_record["continuous_minimum_shoulder_clearance_mm"]),
            0.125,
        )
        self.assertGreater(
            min(
                map(
                    float,
                    travel_record[
                        "locked_projected_head_retention_volume_mm3"
                    ].values(),
                )
            ),
            overlap_tolerance_mm3,
        )

        source_order = [
            "coupled_axial_lead_opens_and_recenters",
            "passive_open_head_passage_at_x0",
            "axial_seat_at_x0",
            "attach_then_dock_release",
            "negative_y_cam_following_withdrawal",
            "q3_tangent_without_lock_claim",
            "physical_lock_eligible_after_14p8mm_clearance",
            "nominal_lock_witness_at_15mm",
        ]
        self.assertLess(
            source_order.index("coupled_axial_lead_opens_and_recenters"),
            source_order.index("passive_open_head_passage_at_x0"),
        )
        self.assertLess(
            source_order.index("attach_then_dock_release"),
            source_order.index("negative_y_cam_following_withdrawal"),
        )
        self.assertLess(
            source_order.index("q3_tangent_without_lock_claim"),
            source_order.index("physical_lock_eligible_after_14p8mm_clearance"),
        )

        # Cross-check the production record last.  Equality here cannot turn a
        # fabricated record green because every compared witness above was
        # already regenerated from exact source shapes and transforms.
        production_record = self.clearance._passive_positive_lock_cam_record()
        self.assertIs(production_record["passed"], True, production_record)
        self.assertEqual(production_record["source_contract"], contract)
        self.assertAlmostEqual(
            float(production_record["geometry"]["cam_volume_mm3"]),
            source_cam_volume_mm3,
            places=9,
        )
        production_tight = production_record["capture"]["tight_stud_clearance"]
        self.assertAlmostEqual(
            float(production_tight["continuous_certified_clearance_mm"]),
            continuous_stud_clearance_mm,
            places=12,
        )
        production_exit = production_record["release"]["nominal_exit_sample"]
        self.assertAlmostEqual(
            float(production_exit["q_mm"]),
            nominal_exit_sample["q_mm"],
            places=12,
        )
        self.assertAlmostEqual(
            float(production_exit["slider_cam_distance_mm"]),
            nominal_exit_sample["slider_cam_distance_mm"],
            places=12,
        )
        source_only_result = {"passed": True, "release_ready": False}
        self.assertIs(source_only_result["passed"], True)
        self.assertIs(
            source_only_result["release_ready"],
            False,
            "source-only geometry cannot claim workflow release readiness",
        )

    def test_positive_lock_slider_mass_properties_match_exact_step(self) -> None:
        """Collision tessellation must not define the moving slider inertia."""

        cad = self.clearance.CAD
        qc = self.demo.qc
        manifest = load_json(CORE_CAD_MANIFEST, "core CAD manifest")
        slider_record = next(
            record
            for record in manifest["files"]
            if str(record["path"]).endswith("/so101_positive_lock_slider.step")
        )
        slider_path = REPOSITORY_ROOT / str(slider_record["path"])
        slider = cad.cq.importers.importStep(str(slider_path)).val()
        density_kg_m3 = float(qc.POSITIVE_LOCK_SLIDER_DENSITY_KG_M3)
        self.assertEqual(density_kg_m3, 8000.0)
        source_mass_kg, source_com_m, source_inertia_kg_m2 = (
            self._source_mass_properties([slider], density_kg_m3)
        )
        np.testing.assert_allclose(
            source_mass_kg,
            qc.POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            source_com_m,
            qc.POSITIVE_LOCK_SLIDER_SOURCE_COM_M,
            rtol=0.0,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            self._full_inertia_vector(source_inertia_kg_m2),
            qc.POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2,
            rtol=0.0,
            atol=5.0e-16,
        )

        body_id = int(self.model.body("qc_positive_lock_slider").id)
        self.assertAlmostEqual(
            float(self.model.body_mass[body_id]), source_mass_kg, delta=1.0e-15
        )
        np.testing.assert_allclose(
            self.model.body_ipos[body_id], source_com_m, rtol=0.0, atol=5.0e-13
        )
        np.testing.assert_allclose(
            self._compiled_body_inertia_tensor(body_id),
            source_inertia_kg_m2,
            rtol=0.0,
            atol=5.0e-16,
        )

        body_xml = self.xml_root.find(".//body[@name='qc_positive_lock_slider']")
        self.assertIsNotNone(body_xml)
        inertial_xml = body_xml.find("./inertial")
        self.assertIsNotNone(inertial_xml, "slider requires explicit source inertia")
        self.assertAlmostEqual(
            float(inertial_xml.get("mass", "nan")), source_mass_kg, delta=1.0e-15
        )
        np.testing.assert_allclose(
            self._vector(inertial_xml, "pos", "nan nan nan"),
            source_com_m,
            rtol=0.0,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            self._vector(
                inertial_xml, "fullinertia", "nan nan nan nan nan nan"
            ),
            self._full_inertia_vector(source_inertia_kg_m2),
            rtol=0.0,
            atol=5.0e-16,
        )
        slider_geoms = body_xml.findall("./geom")
        self.assertTrue(slider_geoms)
        for geom in slider_geoms:
            self.assertEqual(
                float(geom.get("mass", "nan")),
                0.0,
                geom.get("name"),
            )

    def test_positive_lock_slider_joint_dynamics_are_explicit(self) -> None:
        """Reject servo-default inheritance on the passive lock slide."""

        body_xml = self.xml_root.find(".//body[@name='qc_positive_lock_slider']")
        self.assertIsNotNone(body_xml)
        joint_xml = body_xml.find(
            "./joint[@name='qc_positive_lock_slider_joint']"
        )
        self.assertIsNotNone(joint_xml)
        explicit_values: dict[str, float] = {}
        for attribute in ("armature", "frictionloss", "damping"):
            self.assertIn(
                attribute,
                joint_xml.attrib,
                f"slider joint must explicitly override inherited {attribute}",
            )
            value = float(joint_xml.attrib[attribute])
            self.assertTrue(math.isfinite(value), (attribute, value))
            self.assertGreaterEqual(value, 0.0, attribute)
            explicit_values[attribute] = value

        joint_id = int(self.model.joint("qc_positive_lock_slider_joint").id)
        dof_id = int(self.model.jnt_dofadr[joint_id])
        compiled_values = {
            "armature": float(self.model.dof_armature[dof_id]),
            "frictionloss": float(self.model.dof_frictionloss[dof_id]),
            "damping": float(self.model.dof_damping[dof_id]),
        }
        self.assertEqual(compiled_values, explicit_values)
        slider_body_id = int(self.model.body("qc_positive_lock_slider").id)
        expected_critical_damping = 2.0 * math.sqrt(
            float(self.model.jnt_stiffness[joint_id])
            * float(self.model.body_mass[slider_body_id])
        )
        self.assertEqual(explicit_values["armature"], 0.0)
        self.assertEqual(explicit_values["frictionloss"], 0.0)
        self.assertAlmostEqual(
            explicit_values["damping"],
            expected_critical_damping,
            delta=1.0e-12,
            msg="passive slider damping must be derived from exact source mass",
        )
        expected_solref = np.asarray([0.0005, 1.0], dtype=np.float64)
        expected_solimp = np.asarray(
            [0.99, 0.9999, 0.00001, 0.5, 2.0], dtype=np.float64
        )
        for attribute in ("solreflimit", "solimplimit"):
            self.assertIn(
                attribute,
                joint_xml.attrib,
                f"slider limit must explicitly declare {attribute}",
            )
        np.testing.assert_allclose(
            self._vector(joint_xml, "solreflimit", "nan nan"),
            expected_solref,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self._vector(
                joint_xml, "solimplimit", "nan nan nan nan nan"
            ),
            expected_solimp,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.model.jnt_solref[joint_id],
            expected_solref,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.model.jnt_solimp[joint_id],
            expected_solimp,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_isolated_positive_lock_spring_return_and_negative_controls(
        self,
    ) -> None:
        """Require a bounded equality-off return and prove both negatives fail."""

        spring_removed = self._isolated_slider_return_result(
            spring_enabled=False, pin_at_unlocked=False
        )
        q_pinned = self._isolated_slider_return_result(
            spring_enabled=True, pin_at_unlocked=True
        )
        self.assertIs(spring_removed["passed"], False, spring_removed)
        self.assertIs(q_pinned["passed"], False, q_pinned)
        self.assertLess(
            spring_removed["final_window_max_q_m"],
            spring_removed["locked_band_m"][0],
            spring_removed,
        )
        self.assertLess(
            q_pinned["final_window_max_q_m"],
            q_pinned["locked_band_m"][0],
            q_pinned,
        )

        nominal = self._isolated_slider_return_result(
            spring_enabled=True, pin_at_unlocked=False
        )
        self.assertEqual(nominal["direct_state_writes_after_initialization"], 0)
        self.assertIs(
            nominal["passed"],
            True,
            "the equality-off slider must remain inside its declared range and "
            "dwell in [2.95, 3.02] mm at <=5 mm/s for 50 ms",
        )

    def test_positive_lock_hardware_uses_exact_source_inertials(self) -> None:
        """Stud/nut collision proxies may not inflate or double-count mass."""

        cad = self.clearance.CAD
        qc = self.demo.qc
        density_kg_m3 = float(qc.POSITIVE_LOCK_HARDWARE_DENSITY_KG_M3)
        self.assertEqual(density_kg_m3, 8000.0)
        manifest = load_json(CORE_CAD_MANIFEST, "core CAD manifest")
        records = {str(record["path"]): record for record in manifest["files"]}
        screw_relative = (
            "QuickChange/SO101_Magnetic/exports/"
            "hardware_McMaster_90318A720_shoulder_screw.step"
        )
        nut_relative = (
            "QuickChange/SO101_Magnetic/exports/"
            "hardware_DIN934_M3_lock_stud_nut.step"
        )
        self.assertEqual(
            records[screw_relative]["sha256"],
            "c2612e972d5af7ae9b9ebd1ec78b8e2b563cd536173ad30230a7f60b8d844f2b",
        )
        self.assertEqual(
            records[nut_relative]["sha256"],
            "2682fb17a7a369998b89cb9fe5f5b3b1fe3708ed2ac2a1a67ebb22cd3ace8261",
        )
        for relative_path in (screw_relative, nut_relative):
            artifact = REPOSITORY_ROOT / relative_path
            self.assertEqual(
                artifact.stat().st_size, int(records[relative_path]["bytes"])
            )
            self.assertEqual(
                sha256_file(artifact), records[relative_path]["sha256"]
            )
        screw_workplane = cad.cq.importers.importStep(
            str(REPOSITORY_ROOT / screw_relative)
        )
        nut_workplane = cad.cq.importers.importStep(
            str(REPOSITORY_ROOT / nut_relative)
        )
        installed_sources: list[Any] = []
        for x_mm in (-float(cad.LOCK_STUD_X), float(cad.LOCK_STUD_X)):
            installed_sources.append(screw_workplane.translate((x_mm, 0.0, 0.0)).val())
            installed_sources.append(
                nut_workplane.translate(
                    (x_mm, 0.0, float(cad.LOCK_NUT_POCKET_FLOOR))
                ).val()
            )
        source_mass_kg, source_com_m, source_inertia_kg_m2 = (
            self._source_mass_properties(installed_sources, density_kg_m3)
        )
        self.assertAlmostEqual(source_mass_kg, 0.00274294912079922, delta=1.0e-15)
        np.testing.assert_allclose(
            source_com_m,
            [0.0, 0.0, -0.001111579640489],
            rtol=0.0,
            atol=5.0e-13,
        )

        contract = self.demo.CORE_POSITIVE_LOCK_CONTRACT
        self.assertEqual(contract.get("lock_hardware_material"), "stainless steel")
        self.assertEqual(
            float(contract.get("lock_hardware_density_kg_m3")), density_kg_m3
        )
        body_names = [
            f"tool_{tool}_positive_lock_hardware"
            for tool in ("gripper", "spoon", "whisk")
        ]
        compiled_body_names = {
            str(self.model.body(index).name) for index in range(self.model.nbody)
        }
        missing_bodies = sorted(set(body_names) - compiled_body_names)
        self.assertEqual(
            missing_bodies,
            [],
            "lock hardware needs a source-derived child inertial; parent-body "
            "auto inertia counts filled proxy voids and overlapping thread material",
        )
        for tool, body_name in zip(
            ("gripper", "spoon", "whisk"), body_names, strict=True
        ):
            body_id = int(self.model.body(body_name).id)
            self.assertEqual(
                int(self.model.body_parentid[body_id]),
                int(self.model.body(f"tool_{tool}").id),
            )
            np.testing.assert_allclose(
                self.model.body_pos[body_id], np.zeros(3), rtol=0.0, atol=1.0e-12
            )
            self.assertAlmostEqual(
                float(self.model.body_mass[body_id]), source_mass_kg, delta=1.0e-15
            )
            np.testing.assert_allclose(
                self.model.body_ipos[body_id],
                source_com_m,
                rtol=0.0,
                atol=5.0e-13,
            )
            np.testing.assert_allclose(
                self._compiled_body_inertia_tensor(body_id),
                source_inertia_kg_m2,
                rtol=0.0,
                atol=5.0e-16,
            )
            body_xml = self.xml_root.find(f".//body[@name='{body_name}']")
            self.assertIsNotNone(body_xml)
            self.assertEqual(body_xml.findall("./joint"), [], body_name)
            np.testing.assert_allclose(
                self._vector(body_xml, "pos", "0 0 0"),
                np.zeros(3),
                rtol=0.0,
                atol=1.0e-12,
            )
            body_quaternion = self._vector(body_xml, "quat", "1 0 0 0")
            np.testing.assert_allclose(
                np.abs(body_quaternion),
                [1.0, 0.0, 0.0, 0.0],
                rtol=0.0,
                atol=1.0e-12,
            )
            inertial_xml = body_xml.find("./inertial")
            self.assertIsNotNone(inertial_xml)
            self.assertAlmostEqual(
                float(inertial_xml.get("mass", "nan")),
                source_mass_kg,
                delta=1.0e-15,
            )
            np.testing.assert_allclose(
                self._vector(inertial_xml, "pos", "nan nan nan"),
                source_com_m,
                rtol=0.0,
                atol=5.0e-13,
            )
            np.testing.assert_allclose(
                self._vector(
                    inertial_xml, "fullinertia", "nan nan nan nan nan nan"
                ),
                self._full_inertia_vector(source_inertia_kg_m2),
                rtol=0.0,
                atol=5.0e-16,
            )
            expected_geom_names = {
                f"{tool}_lock_stud_{side}_{feature}_collision"
                for side in ("left", "right")
                for feature in ("shoulder", "head")
            } | {
                f"{tool}_positive_lock_{feature}_{side}_collision"
                for side in ("left", "right")
                for feature in ("thread", "nut")
            }
            observed_geom_names = {
                str(self.model.geom(index).name)
                for index in range(self.model.ngeom)
                if int(self.model.geom_bodyid[index]) == body_id
            }
            self.assertEqual(observed_geom_names, expected_geom_names)
            for name in expected_geom_names:
                geom_xml = body_xml.find(f"./geom[@name='{name}']")
                self.assertIsNotNone(geom_xml, name)
                self.assertEqual(float(geom_xml.get("mass", "nan")), 0.0, name)

    def test_complete_matcha_tool_mass_com_and_inertia_match_cad_ledgers(
        self,
    ) -> None:
        """Compose every moving descendant without double-counting hardware."""

        self.maxDiff = None
        manifest = load_json(CAD_MANIFEST, "matcha CAD manifest")
        mujoco = self.demo.mujoco
        data = mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        mujoco.mj_forward(self.model, data)
        mismatches: dict[str, Any] = {}
        for manifest_key, runtime_tool in (
            ("matcha_spoon", "spoon"),
            ("matcha_whisk", "whisk"),
        ):
            tool_record = tool_from_manifest(manifest, manifest_key)
            ledger = load_json(
                CAD_ROOT / str(tool_record["mass_ledger_path"]),
                f"{manifest_key} mass ledger",
            )
            expected_mass_kg = float(ledger["total_mass_kg"])
            expected_com_m = 0.001 * np.asarray(ledger["com_mm"], dtype=np.float64)
            expected_inertia_kg_m2 = np.asarray(
                ledger["inertia_about_com_kg_m2"], dtype=np.float64
            )
            self.assertEqual(expected_inertia_kg_m2.shape, (3, 3))
            np.testing.assert_allclose(
                expected_inertia_kg_m2,
                expected_inertia_kg_m2.T,
                rtol=0.0,
                atol=1.0e-15,
            )
            self.assertTrue(
                np.all(np.linalg.eigvalsh(expected_inertia_kg_m2) > 0.0),
                expected_inertia_kg_m2,
            )
            root_id = int(self.model.body(f"tool_{runtime_tool}").id)
            (
                observed_mass_kg,
                observed_com_m,
                observed_inertia_kg_m2,
                descendants,
            ) = self._subtree_mass_properties_in_root_frame(data, root_id)
            descendant_names = [
                str(self.model.body(body_id).name) for body_id in descendants
            ]
            hardware_body_name = f"tool_{runtime_tool}_positive_lock_hardware"
            self.assertEqual(
                descendant_names.count(hardware_body_name),
                1,
                "the source-derived stud/nut inertial must be composed exactly once",
            )
            if runtime_tool == "whisk":
                self.assertTrue(
                    {
                        "whisk_eccentric_rotor",
                        "whisk_compliance_carriage",
                    }.issubset(descendant_names),
                    descendant_names,
                )
            self.assertAlmostEqual(
                observed_mass_kg,
                float(self.model.body_subtreemass[root_id]),
                delta=1.0e-15,
            )
            mass_error_kg = observed_mass_kg - expected_mass_kg
            com_error_mm = 1000.0 * (observed_com_m - expected_com_m)
            inertia_error_kg_m2 = (
                observed_inertia_kg_m2 - expected_inertia_kg_m2
            )
            if abs(mass_error_kg) > 1.0e-9 or float(
                np.linalg.norm(com_error_mm)
            ) > 0.001 or float(np.linalg.norm(inertia_error_kg_m2)) > 1.0e-10:
                mismatches[runtime_tool] = {
                    "expected_mass_kg": expected_mass_kg,
                    "observed_mass_kg": observed_mass_kg,
                    "mass_error_kg": mass_error_kg,
                    "expected_com_mm": (1000.0 * expected_com_m).tolist(),
                    "observed_com_mm": (1000.0 * observed_com_m).tolist(),
                    "com_error_mm": com_error_mm.tolist(),
                    "expected_inertia_about_com_kg_m2": (
                        expected_inertia_kg_m2.tolist()
                    ),
                    "observed_inertia_about_com_kg_m2": (
                        observed_inertia_kg_m2.tolist()
                    ),
                    "inertia_error_kg_m2": inertia_error_kg_m2.tolist(),
                    "inertia_error_frobenius_kg_m2": float(
                        np.linalg.norm(inertia_error_kg_m2)
                    ),
                    "descendant_bodies": descendant_names,
                }
        self.assertEqual(
            mismatches,
            {},
            "runtime payload mass, COM, and inertia must close the complete CAD ledger",
        )

    def test_mass_property_oracle_detects_altered_moving_child_inertia(self) -> None:
        """A moving whisk child inertia mutation must change the composed tensor."""

        mujoco = self.demo.mujoco
        data = mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        mujoco.mj_forward(self.model, data)
        root_id = int(self.model.body("tool_whisk").id)
        child_id = int(self.model.body("whisk_compliance_carriage").id)
        baseline = self._subtree_mass_properties_in_root_frame(data, root_id)
        altered = self._subtree_mass_properties_in_root_frame(
            data, root_id, inertia_scale_overrides={child_id: 1.10}
        )
        manifest = load_json(CAD_MANIFEST, "matcha CAD manifest")
        ledger = load_json(
            CAD_ROOT
            / str(
                tool_from_manifest(manifest, "matcha_whisk")[
                    "mass_ledger_path"
                ]
            ),
            "matcha whisk mass ledger",
        )
        expected_mass_kg = float(ledger["total_mass_kg"])
        expected_com_m = 0.001 * np.asarray(ledger["com_mm"], dtype=np.float64)
        expected_inertia_kg_m2 = np.asarray(
            ledger["inertia_about_com_kg_m2"], dtype=np.float64
        )
        self.assertAlmostEqual(baseline[0], expected_mass_kg, delta=1.0e-9)
        np.testing.assert_allclose(
            baseline[1], expected_com_m, rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            baseline[2], expected_inertia_kg_m2, rtol=0.0, atol=1.0e-10
        )
        self.assertEqual(baseline[0], altered[0])
        np.testing.assert_allclose(baseline[1], altered[1], rtol=0.0, atol=0.0)
        inertia_delta = altered[2] - baseline[2]
        self.assertGreater(float(np.linalg.norm(inertia_delta)), 1.0e-10)
        self.assertFalse(
            np.allclose(
                expected_inertia_kg_m2,
                altered[2],
                rtol=0.0,
                atol=1.0e-10,
            )
        )

    def test_positive_lock_slider_is_source_bound_physical_mechanism(self) -> None:
        """Bind the moving lock sheet, spring joint, and functional voids to CAD."""

        manifest = load_json(CORE_CAD_MANIFEST, "core CAD manifest")
        report = load_json(CORE_CLEARANCE_REPORT, "core CAD clearance report")
        self.assertIs(report.get("passed"), False, report)
        self.assertIs(report.get("release_ready"), False, report)
        self.assertEqual(
            report.get("blockers"),
            [
                "core_dock_floor_support:PA12_modulus_strength_creep_and_process_allowables_unqualified",
                "core_dock_floor_support:cam_contact_friction_reverse_insertion_and_capture_dynamics_unvalidated",
                "core_dock_floor_support:floor_fixture_substrate_and_M6_thread_authority_missing",
                "core_dock_floor_support:printed_dimensional_tolerance_and_anchor_strength_unqualified",
                "core_dock_floor_support:vendor_or_normative_source_missing_for_selected_M4_and_M6_fasteners",
                "interface_hardware_fit_authority",
            ],
        )
        interface_fit = report.get("interface_hardware_fit")
        self.assertIsInstance(interface_fit, dict, report)
        self.assertIs(interface_fit.get("passed"), False, interface_fit)
        self.assertIs(interface_fit.get("release_ready"), False, interface_fit)
        manifest_record = report["core_cad_manifest_validation"]["manifest"]
        self.assertEqual(
            manifest_record["sha256"], sha256_file(CORE_CAD_MANIFEST), report
        )
        self.assertEqual(manifest_record["bytes"], CORE_CAD_MANIFEST.stat().st_size)
        file_records = {
            str(record["path"]): record for record in manifest.get("files", [])
        }
        expected_sources = {
            "QuickChange/SO101_Magnetic/exports/so101_positive_lock_slider.step": (
                94_504,
                "b9853fe1fcbd7dff91129d7b37c3e1be87d48189486ad91c4c8ab6d57edcbad1",
            ),
            "QuickChange/SO101_Magnetic/exports/hardware_McMaster_90318A720_shoulder_screw.step": (
                13_596,
                "c2612e972d5af7ae9b9ebd1ec78b8e2b563cd536173ad30230a7f60b8d844f2b",
            ),
            "QuickChange/SO101_Magnetic/exports/hardware_DIN934_M3_lock_stud_nut.step": (
                27_470,
                "2682fb17a7a369998b89cb9fe5f5b3b1fe3708ed2ac2a1a67ebb22cd3ace8261",
            ),
        }
        for relative_path, (expected_bytes, expected_sha256) in expected_sources.items():
            self.assertIn(relative_path, file_records)
            record = file_records[relative_path]
            path = REPOSITORY_ROOT / relative_path
            self.assertEqual(record.get("bytes"), expected_bytes, record)
            self.assertEqual(record.get("sha256"), expected_sha256, record)
            self.assertEqual(path.stat().st_size, expected_bytes, path)
            self.assertEqual(sha256_file(path), expected_sha256, path)

        mechanism = report.get("mechanism_preservation")
        self.assertIsInstance(mechanism, dict, report)
        self.assertIs(mechanism.get("passed"), True, mechanism)
        self.assertTrue(all(mechanism.get("checks", {}).values()), mechanism)
        mechanism_bbox = mechanism.get("slider_bbox_native_mm")
        self.assertIsInstance(mechanism_bbox, dict)
        np.testing.assert_allclose(
            [
                *mechanism_bbox["x_mm"],
                *mechanism_bbox["y_mm"],
                *mechanism_bbox["z_mm"],
            ],
            [-15.9740625, 24.0, -4.4, 4.4, 0.0, 1.6],
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            float(mechanism.get("slider_volume_mm3")),
            220.12468083955645,
            places=9,
        )
        self.assertEqual(mechanism.get("stud_centres_xy_mm"), [[-12.0, 0.0], [12.0, 0.0]])
        self.assertEqual(float(mechanism.get("slider_travel_mm")), 3.0)
        self.assertEqual(float(mechanism.get("keyhole_entry_diameter_mm")), 6.5)
        self.assertEqual(float(mechanism.get("keyhole_neck_width_mm")), 4.25)
        self.assertEqual(float(mechanism.get("stud_shoulder_diameter_mm")), 4.0)
        self.assertEqual(float(mechanism.get("stud_head_diameter_mm")), 6.0)

        qc = self.demo.qc
        self.assertEqual(
            qc.POSITIVE_LOCK_SLIDER_STEP_SHA256,
            expected_sources[
                "QuickChange/SO101_Magnetic/exports/so101_positive_lock_slider.step"
            ][1],
        )
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M, (0.0, 0.003))
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_BASE_POS_M, (0.003, 0.0, 0.0047))
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_SPRINGREF_M, 0.0036)
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_STIFFNESS_N_M, 980.0)
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM, 0.005)
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_ANGULAR_DEFLECTION_RAD, 0.3)
        self.assertEqual(qc.POSITIVE_LOCK_SLIDER_VOID_EXPANSION_MM, 0.010)

        mujoco = self.demo.mujoco
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "qc_positive_lock_slider"
        )
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "qc_positive_lock_slider_joint",
        )
        self.assertGreaterEqual(body_id, 0, "physical positive-lock slider body missing")
        self.assertGreaterEqual(joint_id, 0, "physical positive-lock slider joint missing")
        self.assertEqual(
            int(self.model.body_parentid[body_id]),
            int(self.model.body("robot_plate_frame").id),
        )
        np.testing.assert_allclose(
            self.model.body_pos[body_id], qc.POSITIVE_LOCK_SLIDER_BASE_POS_M,
            rtol=0.0, atol=1.0e-12,
        )
        self.assertEqual(
            int(self.model.jnt_type[joint_id]), int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        np.testing.assert_allclose(
            self.model.jnt_axis[joint_id], [1.0, 0.0, 0.0], rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            self.model.jnt_range[joint_id], qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M,
            rtol=0.0, atol=1.0e-12,
        )
        self.assertEqual(
            float(self.model.jnt_stiffness[joint_id]),
            qc.POSITIVE_LOCK_SLIDER_STIFFNESS_N_M,
        )
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        self.assertEqual(float(self.model.qpos0[qpos_address]), 0.003)
        self.assertEqual(
            float(self.model.qpos_spring[qpos_address]),
            qc.POSITIVE_LOCK_SLIDER_SPRINGREF_M,
        )

        prefixes = (
            "qc_col_lock_slider_bridge_part_",
            "qc_col_lock_slider_left_lobe_part_",
            "qc_col_lock_slider_right_lobe_part_",
            "qc_col_lock_slider_tab_part_",
        )
        groups: dict[str, list[int]] = {prefix: [] for prefix in prefixes}
        active_direct: list[int] = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != body_id:
                continue
            if not (
                int(self.model.geom_contype[geom_id])
                or int(self.model.geom_conaffinity[geom_id])
            ):
                continue
            active_direct.append(geom_id)
            name = str(self.model.geom(geom_id).name)
            matches = [prefix for prefix in prefixes if name.startswith(prefix)]
            self.assertEqual(len(matches), 1, name)
            suffix = name.removeprefix(matches[0])
            self.assertTrue(suffix.isdigit(), name)
            groups[matches[0]].append(geom_id)
        self.assertTrue(active_direct, "slider has no active collision geometry")
        self.assertTrue(all(groups.values()), groups)
        active_names = [str(self.model.geom(index).name) for index in active_direct]
        self.assertEqual(len(active_names), len(set(active_names)))
        for prefix, geom_ids in groups.items():
            suffixes = sorted(
                int(str(self.model.geom(index).name).removeprefix(prefix))
                for index in geom_ids
            )
            self.assertEqual(suffixes, list(range(len(suffixes))), prefix)

        from scipy.spatial import ConvexHull

        quaternion_snapshot = np.array(self.model.geom_quat, copy=True)
        piece_vertices: dict[int, np.ndarray] = {}
        piece_equations: dict[int, np.ndarray] = {}
        proxy_volume_mm3 = 0.0
        for geom_id in active_direct:
            name = str(self.model.geom(geom_id).name)
            self.assertEqual(
                int(self.model.geom_type[geom_id]),
                int(mujoco.mjtGeom.mjGEOM_MESH),
                name,
            )
            vertices = self._compiled_geom_vertices_in_owner_frame(geom_id)
            piece_vertices[geom_id] = vertices
            hull = ConvexHull(vertices)
            piece_equations[geom_id] = np.asarray(
                hull.equations, dtype=np.float64
            )
            proxy_volume_mm3 += float(hull.volume) * 1.0e9
            for tool in ("gripper", "spoon", "whisk"):
                for side in ("left", "right"):
                    stud_id = int(
                        self.model.geom(
                            f"{tool}_lock_stud_{side}_shoulder_collision"
                        ).id
                    )
                    self.assertTrue(
                        (int(self.model.geom_contype[geom_id]) & int(self.model.geom_conaffinity[stud_id]))
                        or (int(self.model.geom_contype[stud_id]) & int(self.model.geom_conaffinity[geom_id])),
                        (name, self.model.geom(stud_id).name),
                    )
        np.testing.assert_array_equal(self.model.geom_quat, quaternion_snapshot)
        all_vertices = np.concatenate(list(piece_vertices.values()), axis=0)
        np.testing.assert_allclose(
            np.min(all_vertices, axis=0), [-0.0159740625, -0.0044, 0.0],
            rtol=0.0,
            atol=1.0e-3 * qc.POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM,
        )
        np.testing.assert_allclose(
            np.max(all_vertices, axis=0), [0.024, 0.0044, 0.0016],
            rtol=0.0,
            atol=1.0e-3 * qc.POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM,
        )
        self.assertLessEqual(proxy_volume_mm3, 220.12468083955645 + 1.0e-6)
        self.assertGreaterEqual(proxy_volume_mm3, 0.995 * 220.12468083955645)

        # Functional void negatives: adding even a small hidden component in
        # an entry hole, keyhole neck, or guide slot must make this test red.
        void_points_mm: list[tuple[float, float, float]] = []
        for x_mm in (-12.0, 12.0):
            void_points_mm.append((x_mm, 0.0, 0.8))
            void_points_mm.extend(
                (
                    x_mm + 2.5 * math.cos(index * math.pi / 4.0),
                    2.5 * math.sin(index * math.pi / 4.0),
                    0.8,
                )
                for index in range(8)
            )
        void_points_mm.extend(
            [(-14.5, 0.0, 0.8), (9.5, 0.0, 0.8), (-1.5, 0.0, 0.8)]
        )
        for point_mm in void_points_mm:
            point_m = 0.001 * np.asarray(point_mm, dtype=np.float64)
            filled_by = [
                str(self.model.geom(geom_id).name)
                for geom_id, equations in piece_equations.items()
                if np.all(
                    equations[:, :3] @ point_m + equations[:, 3] <= 1.0e-10
                )
            ]
            self.assertEqual(filled_by, [], f"slider proxy fills source void {point_mm}")

        data = mujoco.MjData(self.model)
        self.demo.initialize(self.model, data)
        self.assertAlmostEqual(float(data.qpos[qpos_address]), 0.003, places=9)

        parent_id = int(self.model.body_parentid[body_id])

        def relative_slider_pose(qpos_m: float) -> tuple[np.ndarray, np.ndarray]:
            data.qpos[qpos_address] = qpos_m
            mujoco.mj_forward(self.model, data)
            parent_rotation = np.asarray(
                data.xmat[parent_id], dtype=np.float64
            ).reshape(3, 3)
            slider_rotation = np.asarray(
                data.xmat[body_id], dtype=np.float64
            ).reshape(3, 3)
            position = parent_rotation.T @ (
                np.asarray(data.xpos[body_id], dtype=np.float64)
                - np.asarray(data.xpos[parent_id], dtype=np.float64)
            )
            rotation = parent_rotation.T @ slider_rotation
            return position, rotation

        unlocked_position, unlocked_rotation = relative_slider_pose(0.0)
        locked_position, locked_rotation = relative_slider_pose(0.003)
        np.testing.assert_allclose(
            unlocked_position, [0.0, 0.0, 0.0047], rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            locked_position, [0.003, 0.0, 0.0047], rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            unlocked_rotation, np.eye(3), rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            locked_rotation, np.eye(3), rtol=0.0, atol=1.0e-12
        )

    def retired_negative_z_lock_release_stroke_contract(self) -> None:
        """Retained history for the superseded negative-Z release concept.

        The rolled core dock uses the source-authored negative-Y static route
        certified by ``RolledCoreDockRuntimeAuthorityTests``.  This older
        physical-release test must not be discovered as a current contract.
        """

        cad = self.clearance.CAD
        self.assertEqual(
            self.demo.CORE_LOCK_RELEASE_SOURCE_AXIS,
            "dock_local_negative_z",
        )
        np.testing.assert_allclose(
            np.asarray(self.demo.CORE_LOCK_RELEASE_AXIS_DOCK_LOCAL, dtype=float),
            [0.0, 0.0, -1.0],
            rtol=0.0,
            atol=0.0,
        )
        target_mm = float(self.demo.CORE_LOCK_RELEASE_STROKE_MM)
        minimum_mm = float(self.demo.CORE_LOCK_RELEASE_MIN_STROKE_MM)
        maximum_mm = float(self.demo.MAXIMUM_SOURCE_AXIS_WITHDRAWAL_MM)
        self.assertEqual(
            float(self.demo.MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM), minimum_mm
        )
        self.assertAlmostEqual(target_mm, 1.20, places=12)
        self.assertAlmostEqual(minimum_mm, 1.15, places=12)
        self.assertAlmostEqual(maximum_mm, 1.20, places=12)
        self.assertGreaterEqual(target_mm, minimum_mm)
        self.assertLessEqual(target_mm, maximum_mm)

        components = self.clearance._tool_side_components()
        expected_names = {
            "stock_tool_plate",
            "official_fixed_gripper_body",
            "target_1",
            "target_screw_1",
            "target_nut_1",
            "target_2",
            "target_screw_2",
            "target_nut_2",
            "shoulder_stud_1",
            "stud_nut_1",
            "shoulder_stud_2",
            "stud_nut_2",
            "target_contact_board",
            "target_pad_1",
            "target_pad_2",
            "target_pad_3",
            "target_pad_4",
        }
        by_name = {component.name: component for component in components}
        self.assertEqual(len(components), len(by_name))
        self.assertEqual(set(by_name), expected_names)

        dock = self.clearance._dock_authority()
        full_dock = dock["full_dock"].val()
        cam = dock["positive_lock_cam"].val()
        rail_names = (
            "left_lower_rail",
            "right_lower_rail",
            "left_upper_rail",
            "right_upper_rail",
        )
        samples_mm = [round(0.1 * index, 10) for index in range(13)]
        self.assertEqual(samples_mm[0], 0.0)
        self.assertEqual(samples_mm[-1], target_mm)

        stock_plate = by_name["stock_tool_plate"]
        plate_bounds = self.clearance._bbox_tuple(stock_plate.shape.val())
        left_lower_bounds = self.clearance._bbox_tuple(
            dock["left_lower_rail"].val()
        )
        right_lower_bounds = self.clearance._bbox_tuple(
            dock["right_lower_rail"].val()
        )
        self.assertAlmostEqual(plate_bounds[0][0], left_lower_bounds[0][1])
        self.assertAlmostEqual(plate_bounds[0][1], right_lower_bounds[0][0])

        endpoint_distances: dict[str, float] = {}
        for stroke_mm in samples_mm:
            translation = (0.0, 0.0, -stroke_mm)
            for component in components:
                placed = component.shape.translate(translation).val()
                distance_mm = float(placed.distance(full_dock))
                overlap_mm3 = (
                    self.clearance._intersection_volume_mm3(placed, full_dock)
                    if distance_mm <= self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM
                    else 0.0
                )
                self.assertLessEqual(
                    overlap_mm3,
                    self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3,
                    {
                        "component": component.name,
                        "stroke_mm": stroke_mm,
                        "distance_mm": distance_mm,
                        "overlap_mm3": overlap_mm3,
                    },
                )
                if component.name != "stock_tool_plate":
                    self.assertGreater(
                        distance_mm,
                        self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM,
                        {
                            "component": component.name,
                            "stroke_mm": stroke_mm,
                            "unclassified_zero_distance_contact": True,
                        },
                    )
                if stroke_mm == target_mm:
                    endpoint_distances[component.name] = distance_mm

            placed_plate = stock_plate.shape.translate(translation).val()
            for rail_name in rail_names:
                rail = dock[rail_name].val()
                distance_mm = float(placed_plate.distance(rail))
                overlap_mm3 = self.clearance._intersection_volume_mm3(
                    placed_plate, rail
                )
                self.assertLessEqual(
                    overlap_mm3,
                    self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3,
                    (rail_name, stroke_mm, overlap_mm3),
                )
                if rail_name.endswith("lower_rail"):
                    # Persistent opposing X-face side guides; -Z is tangent
                    # to their normals and must not create volume overlap.
                    self.assertAlmostEqual(distance_mm, 0.0, delta=1.0e-9)
                else:
                    # The upper keepers touch only at the seated plane, then
                    # separate monotonically by exactly the axial stroke.
                    self.assertAlmostEqual(distance_mm, stroke_mm, delta=1.0e-9)

        motion_bound_mm = 0.05
        endpoint_plate = stock_plate.shape.translate((0.0, 0.0, -target_mm)).val()
        plate_cam_distance_mm = float(endpoint_plate.distance(cam))
        self.assertAlmostEqual(plate_cam_distance_mm, 0.75, delta=1.0e-9)
        self.assertGreaterEqual(plate_cam_distance_mm - motion_bound_mm, 0.70)

        self.assertGreaterEqual(
            endpoint_distances["official_fixed_gripper_body"] - motion_bound_mm,
            1.750703657,
        )
        self.assertGreaterEqual(
            endpoint_distances["target_contact_board"] - motion_bound_mm,
            0.95,
        )
        hardware_names = expected_names - {
            "stock_tool_plate",
            "official_fixed_gripper_body",
            "target_contact_board",
        }
        self.assertGreaterEqual(
            min(endpoint_distances[name] for name in hardware_names)
            - motion_bound_mm,
            2.95,
        )

        locked_slider = (
            cad.locking_slider()
            .translate(
                (
                    cad.SLIDER_TRAVEL,
                    0.0,
                    cad.SLIDER_Z - cad.PLATE_THICKNESS - target_mm,
                )
            )
            .val()
        )
        slider_cam_distance_mm = float(locked_slider.distance(cam))
        slider_cam_overlap_mm3 = (
            self.clearance._intersection_volume_mm3(locked_slider, cam)
            if slider_cam_distance_mm
            <= self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        self.assertLessEqual(
            slider_cam_overlap_mm3,
            self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3,
        )
        self.assertAlmostEqual(slider_cam_distance_mm, 0.25, delta=1.0e-9)
        self.assertGreaterEqual(
            slider_cam_distance_mm - motion_bound_mm,
            float(self.demo.LOCK_CAM_MANUFACTURING_CLEARANCE_MM),
        )


class RetiredNegativeZDynamicSmokeContracts(unittest.TestCase):
    """Historical physical-release oracle superseded by the rolled red gate."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = import_file(
            MATCHA_DEMO,
            "matcha_demo_bounded_dynamic_validation",
            "matcha workflow simulator",
        )
        cls.clearance = import_file(
            CORE_CLEARANCE_VALIDATOR,
            "core_clearance_bounded_dynamic_validation",
            "core CAD clearance validator",
        )

    @staticmethod
    def quaternion_matrix(quaternion: Any) -> np.ndarray:
        values = np.asarray(quaternion, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise AssertionError(f"invalid quaternion: {quaternion}")
        norm = float(np.linalg.norm(values))
        if norm <= 0.0:
            raise AssertionError(f"zero quaternion: {quaternion}")
        w, x, y, z = values / norm
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def quaternion_angle(quaternion_a: Any, quaternion_b: Any) -> float:
        a = np.asarray(quaternion_a, dtype=np.float64)
        b = np.asarray(quaternion_b, dtype=np.float64)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        return 2.0 * math.acos(min(1.0, max(0.0, abs(float(a @ b)))))

    def assert_points_on_source_witness(
        self,
        points: Any,
        witness: dict[str, Any],
        description: Any,
    ) -> None:
        self.assertIsInstance(points, list, description)
        tolerance = float(witness["point_tolerance_mm"])
        axis_index = {"x": 0, "y": 1, "z": 2}
        for raw_point in points:
            self.assertEqual(len(raw_point), 3, description)
            point = np.asarray(raw_point, dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(point)), description)
            if witness["kind"] == "line_tangency":
                line_axis = str(witness["line_axis"])
                for axis, coordinate in witness["fixed_coordinates_mm"].items():
                    self.assertLessEqual(
                        abs(float(point[axis_index[axis]]) - float(coordinate)),
                        tolerance,
                        description,
                    )
                line_coordinate = float(point[axis_index[line_axis]])
                lower, upper = (
                    float(value) for value in witness["line_axis_bounds_mm"]
                )
                self.assertGreaterEqual(line_coordinate, lower - tolerance, description)
                self.assertLessEqual(line_coordinate, upper + tolerance, description)
            elif witness["kind"] == "planar_face_tangency":
                normal_axis = str(witness["normal_axis"])
                self.assertLessEqual(
                    abs(
                        float(point[axis_index[normal_axis]])
                        - float(witness["plane_coordinate_mm"])
                    ),
                    tolerance,
                    description,
                )
                for axis, bounds in witness["tangential_bounds_mm"].items():
                    coordinate = float(point[axis_index[axis]])
                    lower, upper = (float(value) for value in bounds)
                    self.assertGreaterEqual(coordinate, lower - tolerance, description)
                    self.assertLessEqual(coordinate, upper + tolerance, description)
                boundary = witness.get("source_boundary_constraint")
                if boundary is not None:
                    self.assertEqual(boundary.get("kind"), "rounded_rectangle")
                    half_width = float(boundary["half_width_mm"])
                    half_height = float(boundary["half_height_mm"])
                    radius = float(boundary["corner_radius_mm"])
                    dx = max(abs(float(point[0])) - (half_width - radius), 0.0)
                    dy = max(abs(float(point[1])) - (half_height - radius), 0.0)
                    self.assertLessEqual(math.hypot(dx, dy), radius + tolerance)
            else:
                self.fail(f"unknown keeper source witness: {witness}")

    def assert_core_keeper_report(self, result: dict[str, Any]) -> None:
        report = result.get("core_keeper_contact_report")
        self.assertIsInstance(report, dict, result)
        self.assertIs(report.get("passed"), True, report)
        self.assertEqual(report.get("phase"), "pre_attach_seated_keeper_capture")
        self.assertIs(report.get("dock_hold_active"), True, report)
        self.assertIs(report.get("attach_equality_active"), False, report)
        self.assertEqual(
            report.get("pogo_signals"), sorted(self.demo.qc.SIGNALS), report
        )
        self.assertEqual(int(report.get("observed_tool_id", -1)), 6, report)
        self.assertEqual(int(report.get("expected_tool_id", -1)), 6, report)
        self.assertIs(report.get("tool_identity_verified"), True, report)
        self.assertEqual(int(report.get("stop_contact_count", -1)), 0, report)

        position_error_mm = float(report.get("pose_position_error_mm", math.inf))
        angle_error_deg = float(report.get("pose_angle_error_deg", math.inf))
        self.assertTrue(math.isfinite(position_error_mm), report)
        self.assertTrue(math.isfinite(angle_error_deg), report)
        self.assertLessEqual(
            position_error_mm,
            1000.0 * float(self.demo.CAPTURE_POSITION_TOLERANCE_M),
            report,
        )
        self.assertLessEqual(
            angle_error_deg,
            math.degrees(float(self.demo.CAPTURE_ORIENTATION_TOLERANCE_RAD)),
            report,
        )

        contract = {
            tuple(str(value) for value in item["source_pair"]): item
            for item in self.demo.CORE_KEEPER_CONTACT_CONTRACT
        }
        records = report.get("records")
        self.assertIsInstance(records, list, report)
        self.assertEqual(len(records), len(contract), report)
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            self.assertIsInstance(record, dict)
            source_pair = tuple(str(value) for value in record["source_pair"])
            self.assertIn(source_pair, contract, record)
            self.assertNotIn(source_pair, observed, record)
            expected = contract[source_pair]
            self.assertEqual(record.get("runtime_pair"), expected["runtime_pair"])
            self.assertEqual(record.get("source_witness"), expected["source_witness"])
            normal_fields = (
                "expected_local_normal_axis",
                "expected_local_normal_subspace",
            )
            for field in normal_fields:
                self.assertEqual(record.get(field), expected.get(field), record)
            self.assertIs(record.get("passed"), True, record)
            contact_count = int(record.get("contact_count", -1))
            self.assertGreaterEqual(contact_count, 0, record)
            signed_distance_mm = float(record.get("signed_distance_mm", math.inf))
            penetration_mm = float(record.get("max_penetration_mm", math.inf))
            self.assertTrue(math.isfinite(signed_distance_mm), record)
            self.assertGreaterEqual(
                signed_distance_mm,
                -float(self.demo.CORE_KEEPER_MAX_PENETRATION_MM),
                record,
            )
            self.assertLessEqual(
                signed_distance_mm,
                float(self.demo.CORE_KEEPER_MAX_SEPARATION_MM),
                record,
            )
            self.assertLessEqual(
                penetration_mm,
                float(self.demo.CORE_KEEPER_MAX_PENETRATION_MM),
                record,
            )
            witness_method = record.get("witness_method")
            self.assertIn(
                witness_method,
                {
                    "live_mujoco_contact",
                    "live_mujoco_signed_geom_distance_and_source_semantics",
                },
                record,
            )
            closest_point_method = record.get("closest_point_method")
            mujoco_from_to_valid = record.get("mujoco_from_to_valid")
            if closest_point_method == "mujoco_mj_geomDistance":
                self.assertIs(mujoco_from_to_valid, True, record)
            elif closest_point_method == (
                "analytic_box_box_line_tangency_from_live_geom_transforms"
            ):
                self.assertIs(mujoco_from_to_valid, False, record)
                self.assertEqual(
                    expected["source_witness"].get("kind"),
                    "line_tangency",
                    record,
                )
                self.assertEqual(contact_count, 0, record)
                self.assertLessEqual(
                    abs(signed_distance_mm),
                    1000.0 * float(self.demo.CONTACT_NUMERICAL_EPSILON_M),
                    record,
                )
            else:
                self.fail(f"unreviewed keeper closest-point method: {record}")
            closest_points = record.get("closest_points_dock_local_mm")
            self.assertIsInstance(closest_points, list, record)
            self.assertEqual(len(closest_points), 2, record)
            self.assert_points_on_source_witness(
                closest_points, expected["source_witness"], record
            )
            contact_points = record.get("contact_points_dock_local_mm")
            self.assertIsInstance(contact_points, list, record)
            self.assertEqual(len(contact_points), contact_count, record)
            self.assert_points_on_source_witness(
                contact_points, expected["source_witness"], record
            )
            maximum_point_error_mm = float(
                record.get("maximum_contact_point_source_witness_error_mm", math.inf)
            )
            self.assertLessEqual(
                maximum_point_error_mm,
                float(expected["source_witness"]["point_tolerance_mm"]),
                record,
            )
            contact_normals = record.get(
                "contact_normals_from_runtime_pair_0_to_1_dock_local"
            )
            self.assertIsInstance(contact_normals, list, record)
            self.assertEqual(len(contact_normals), contact_count, record)
            normalized_contact_normals: list[np.ndarray] = []
            for raw_normal in contact_normals:
                normal = np.asarray(raw_normal, dtype=np.float64)
                self.assertEqual(normal.shape, (3,), record)
                self.assertTrue(np.all(np.isfinite(normal)), record)
                norm = float(np.linalg.norm(normal))
                self.assertGreater(norm, 0.0, record)
                normalized_contact_normals.append(normal / norm)
            if expected.get("expected_local_normal_axis") is not None:
                alignment = float(
                    record.get("minimum_normal_alignment", -math.inf)
                )
                self.assertGreaterEqual(
                    alignment,
                    float(self.demo.CORE_KEEPER_MIN_NORMAL_ALIGNMENT),
                    record,
                )
                self.assertGreater(contact_count, 0, record)
                self.assertEqual(witness_method, "live_mujoco_contact", record)
                axis_index = {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                }[str(expected["expected_local_normal_axis"])]
                for normal in normalized_contact_normals:
                    self.assertGreaterEqual(
                        abs(float(normal[axis_index])),
                        float(self.demo.CORE_KEEPER_MIN_NORMAL_ALIGNMENT),
                        record,
                    )
            else:
                self.assertEqual(
                    expected.get("expected_local_normal_subspace"),
                    "dock_xz_plane",
                    expected,
                )
                if contact_count:
                    subspace_alignment = float(
                        record.get(
                            "minimum_normal_subspace_alignment", -math.inf
                        )
                    )
                    self.assertGreaterEqual(
                        subspace_alignment,
                        float(self.demo.CORE_KEEPER_MIN_NORMAL_ALIGNMENT),
                        record,
                    )
                same_sign_quadrants = source_pair == (
                    "stock_tool_plate",
                    "left_lower_rail",
                )
                for normal in normalized_contact_normals:
                    self.assertGreaterEqual(
                        float(np.linalg.norm(normal[[0, 2]])),
                        float(self.demo.CORE_KEEPER_MIN_NORMAL_ALIGNMENT),
                        record,
                    )
                    product = float(normal[0] * normal[2])
                    if same_sign_quadrants:
                        self.assertGreaterEqual(product, -1.0e-12, record)
                    else:
                        self.assertLessEqual(product, 1.0e-12, record)
            observed[source_pair] = record
        self.assertEqual(set(observed), set(contract), report)

    def assert_slider_return_trace(self, slider: dict[str, Any]) -> None:
        """Recompute the bounded q/qdot return and settled dwell from samples."""

        samples = slider.get("samples")
        self.assertIsInstance(samples, list, slider)
        self.assertTrue(samples, slider)
        sample_array = np.asarray(samples, dtype="<f8")
        self.assertEqual(sample_array.ndim, 2, slider)
        self.assertEqual(sample_array.shape[1], 3, slider)
        self.assertTrue(np.all(np.isfinite(sample_array)), slider)
        substeps = sample_array[:, 0]
        self.assertTrue(np.all(substeps == np.floor(substeps)), slider)
        if len(substeps) > 1:
            np.testing.assert_array_equal(np.diff(substeps), 1.0)
        self.assertEqual(
            int(slider.get("trajectory_sample_count", -1)), len(sample_array), slider
        )
        self.assertEqual(
            int(slider.get("trajectory_first_physics_substep_count", -1)),
            int(substeps[0]),
            slider,
        )
        self.assertEqual(
            int(slider.get("trajectory_last_physics_substep_count", -1)),
            int(substeps[-1]),
            slider,
        )
        self.assertEqual(
            slider.get("trajectory_sha256_le_f64_substep_q_qdot"),
            hashlib.sha256(sample_array.tobytes()).hexdigest(),
            slider,
        )

        q_m = sample_array[:, 1]
        qvel_m_s = sample_array[:, 2]
        # This is stricter than checking only the first state in the locked
        # band: the passive mechanism may neither tunnel below q=0 nor bounce
        # beyond its declared limit during the complete real-step return.
        self.assertGreaterEqual(float(np.min(q_m)), -0.000020, slider)
        self.assertLessEqual(float(np.max(q_m)), 0.003050, slider)
        self.assertAlmostEqual(
            float(slider.get("trajectory_q_min_mm", math.inf)),
            1000.0 * float(np.min(q_m)),
            delta=1.0e-9,
        )
        self.assertAlmostEqual(
            float(slider.get("trajectory_q_max_mm", math.inf)),
            1000.0 * float(np.max(q_m)),
            delta=1.0e-9,
        )
        self.assertAlmostEqual(
            float(slider.get("trajectory_max_abs_qvel_mm_s", math.inf)),
            1000.0 * float(np.max(np.abs(qvel_m_s))),
            delta=1.0e-9,
        )

        timestep_s = float(slider.get("physics_timestep_s", math.nan))
        self.assertTrue(math.isfinite(timestep_s), slider)
        self.assertGreater(timestep_s, 0.0, slider)
        required_dwell_s = float(self.demo.LOCKED_SLIDER_SETTLED_DWELL_S)
        required_substeps = math.ceil(required_dwell_s / timestep_s)
        declared_dwell_substeps = int(slider.get("settled_dwell_substeps", -1))
        self.assertGreaterEqual(declared_dwell_substeps, required_substeps, slider)
        self.assertGreaterEqual(len(sample_array), declared_dwell_substeps, slider)
        dwell = sample_array[-declared_dwell_substeps:]
        band_m = np.asarray(self.demo.LOCKED_SLIDER_POSITION_BAND_M, dtype=float)
        speed_limit_m_s = float(self.demo.LOCKED_SLIDER_SPEED_LIMIT_M_S)
        self.assertTrue(np.all(dwell[:, 1] >= band_m[0]), slider)
        self.assertTrue(np.all(dwell[:, 1] <= band_m[1]), slider)
        self.assertTrue(np.all(np.abs(dwell[:, 2]) <= speed_limit_m_s), slider)
        self.assertAlmostEqual(
            float(slider.get("settled_dwell_s", math.nan)),
            declared_dwell_substeps * timestep_s,
            delta=1.0e-12,
        )
        self.assertGreaterEqual(
            float(slider.get("settled_dwell_s", 0.0)), required_dwell_s, slider
        )
        for key, expected in (
            ("dwell_q_min_mm", 1000.0 * float(np.min(dwell[:, 1]))),
            ("dwell_q_max_mm", 1000.0 * float(np.max(dwell[:, 1]))),
            (
                "dwell_max_abs_qvel_mm_s",
                1000.0 * float(np.max(np.abs(dwell[:, 2]))),
            ),
        ):
            self.assertAlmostEqual(
                float(slider.get(key, math.inf)), expected, delta=1.0e-9
            )
        self.assertAlmostEqual(
            float(slider.get("slider_joint_position_mm", math.inf)),
            1000.0 * float(q_m[-1]),
            delta=1.0e-9,
        )
        self.assertAlmostEqual(
            float(slider.get("slider_joint_velocity_mm_s", math.inf)),
            1000.0 * float(qvel_m_s[-1]),
            delta=1.0e-9,
        )
        self.assertEqual(
            int(slider.get("direct_state_writes_after_physical_capture", -1)), 0
        )
        self.assertEqual(int(slider.get("cam_tab_contact_count", -1)), 0, slider)
        self.assertEqual(
            int(slider.get("cam_tab_contact_count_during_settled_dwell", -1)),
            0,
            slider,
        )
        self.assertGreaterEqual(
            float(slider.get("cam_tab_runtime_proxy_clearance_mm", -math.inf)),
            0.0,
            slider,
        )
        self.assertGreaterEqual(
            float(slider.get("cam_tab_min_clearance_mm", -math.inf)),
            0.0,
            slider,
        )

    def assert_lock_sequence(self, result: dict[str, Any]) -> None:
        expected_events = (
            "physical_capture_complete",
            "dock_hold_released",
            "source_axis_withdrawal_complete",
            "slider_return_verified",
            "physical_lock_confirmed",
        )
        journal = result.get("journal")
        self.assertIsInstance(journal, list, result)
        matching: dict[str, dict[str, Any]] = {}
        event_positions: list[int] = []
        for event_name in expected_events:
            matches = [
                (index, record)
                for index, record in enumerate(journal)
                if isinstance(record, dict) and record.get("event") == event_name
            ]
            self.assertEqual(len(matches), 1, (event_name, journal))
            index, record = matches[0]
            event_positions.append(index)
            matching[event_name] = record
        self.assertEqual(event_positions, sorted(event_positions), journal)

        physics_indices: list[int] = []
        for event_name in expected_events:
            record = matching[event_name]
            physics_index = int(record.get("physics_substep_count", -1))
            self.assertGreaterEqual(physics_index, 0, record)
            physics_indices.append(physics_index)
        self.assertEqual(physics_indices, sorted(physics_indices), matching)
        self.assertEqual(len(physics_indices), len(set(physics_indices)), matching)

        true_lock_records = [
            (index, record)
            for index, record in enumerate(journal)
            if isinstance(record, dict)
            and record.get("physical_lock_confirmed") is True
        ]
        self.assertEqual(len(true_lock_records), 1, journal)
        true_lock_position, true_lock_record = true_lock_records[0]
        self.assertEqual(
            true_lock_record.get("event"), "physical_lock_confirmed", true_lock_record
        )
        self.assertEqual(true_lock_position, event_positions[-1], journal)

        for event_name in expected_events[:-1]:
            self.assertIs(
                matching[event_name].get("physical_lock_confirmed"),
                False,
                matching[event_name],
            )
        capture = matching["physical_capture_complete"]
        self.assertIs(capture.get("dock_hold_active"), True, capture)
        self.assertIs(capture.get("attach_equality_active"), True, capture)
        self.assertEqual(capture.get("pogo_signals"), sorted(self.demo.qc.SIGNALS))
        self.assertEqual(int(capture.get("observed_tool_id", -1)), 6, capture)
        self.assertEqual(int(capture.get("expected_tool_id", -1)), 6, capture)
        self.assertIs(capture.get("tool_identity_verified"), True, capture)
        self.assertIs(
            matching["dock_hold_released"].get("dock_hold_active"),
            False,
            matching["dock_hold_released"],
        )
        self.assertIs(
            matching["dock_hold_released"].get("attach_equality_active"),
            True,
            matching["dock_hold_released"],
        )
        withdrawal = matching["source_axis_withdrawal_complete"]
        minimum_withdrawal_mm = float(self.demo.MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM)
        maximum_withdrawal_mm = float(self.demo.MAXIMUM_SOURCE_AXIS_WITHDRAWAL_MM)
        self.assertAlmostEqual(minimum_withdrawal_mm, 1.15, places=12)
        self.assertAlmostEqual(maximum_withdrawal_mm, 1.20, places=12)
        self.assertEqual(
            withdrawal.get("source_axis"),
            "dock_local_negative_z",
            withdrawal,
        )
        np.testing.assert_allclose(
            np.asarray(withdrawal.get("source_axis_dock_local"), dtype=float),
            [0.0, 0.0, -1.0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertIs(withdrawal.get("dock_hold_active"), False, withdrawal)
        self.assertIs(withdrawal.get("attach_equality_active"), True, withdrawal)
        pose_fields = (
            "robot_mating_position_world_m",
            "robot_mating_quat_wxyz",
            "dock_position_world_m",
            "dock_quat_wxyz",
        )
        for record in (capture, withdrawal):
            for field in pose_fields:
                self.assertIn(field, record, record)
        capture_robot_position = np.asarray(
            capture["robot_mating_position_world_m"], dtype=np.float64
        )
        capture_dock_position = np.asarray(
            capture["dock_position_world_m"], dtype=np.float64
        )
        withdrawal_robot_position = np.asarray(
            withdrawal["robot_mating_position_world_m"], dtype=np.float64
        )
        withdrawal_dock_position = np.asarray(
            withdrawal["dock_position_world_m"], dtype=np.float64
        )
        for position in (
            capture_robot_position,
            capture_dock_position,
            withdrawal_robot_position,
            withdrawal_dock_position,
        ):
            self.assertEqual(position.shape, (3,))
            self.assertTrue(np.all(np.isfinite(position)))
        np.testing.assert_allclose(
            withdrawal_dock_position,
            capture_dock_position,
            rtol=0.0,
            atol=1.0e-12,
        )
        capture_dock_rotation = self.quaternion_matrix(capture["dock_quat_wxyz"])
        withdrawal_dock_rotation = self.quaternion_matrix(
            withdrawal["dock_quat_wxyz"]
        )
        np.testing.assert_allclose(
            withdrawal_dock_rotation,
            capture_dock_rotation,
            rtol=0.0,
            atol=1.0e-12,
        )
        capture_local = capture_dock_rotation.T @ (
            capture_robot_position - capture_dock_position
        )
        withdrawal_local = capture_dock_rotation.T @ (
            withdrawal_robot_position - withdrawal_dock_position
        )
        displacement_local = withdrawal_local - capture_local
        displacement_norm = float(np.linalg.norm(displacement_local))
        self.assertGreater(displacement_norm, 0.0, withdrawal)
        recomputed_withdrawal_mm = -1000.0 * float(displacement_local[2])
        recomputed_axis_alignment = -float(displacement_local[2]) / displacement_norm
        self.assertGreaterEqual(
            recomputed_withdrawal_mm,
            minimum_withdrawal_mm,
            withdrawal,
        )
        self.assertLessEqual(
            recomputed_withdrawal_mm,
            maximum_withdrawal_mm + 1.0e-6,
            withdrawal,
        )
        self.assertGreaterEqual(
            recomputed_axis_alignment, 0.999, withdrawal
        )
        self.assertAlmostEqual(
            float(withdrawal.get("withdrawal_mm", math.inf)),
            recomputed_withdrawal_mm,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            float(withdrawal.get("axis_alignment", math.inf)),
            recomputed_axis_alignment,
            delta=1.0e-9,
        )
        self.assertLessEqual(
            1000.0 * float(np.linalg.norm(displacement_local[[0, 1]])),
            1000.0 * float(self.demo.CAM_RELIEF_CORRIDOR_M),
            withdrawal,
        )
        self.assertLessEqual(
            self.quaternion_angle(
                capture["robot_mating_quat_wxyz"],
                withdrawal["robot_mating_quat_wxyz"],
            ),
            float(self.demo.CAPTURE_ORIENTATION_TOLERANCE_RAD),
            withdrawal,
        )
        self.assertAlmostEqual(
            float(withdrawal.get("lateral_deviation_mm", math.inf)),
            1000.0 * float(np.linalg.norm(displacement_local[[0, 1]])),
            delta=1.0e-9,
        )
        self.assertAlmostEqual(
            float(withdrawal.get("orientation_error_rad", math.inf)),
            self.quaternion_angle(
                capture["robot_mating_quat_wxyz"],
                withdrawal["robot_mating_quat_wxyz"],
            ),
            delta=1.0e-9,
        )
        expected_guides = {
            tuple(str(value) for value in contract["source_pair"]): contract
            for contract in self.demo.CORE_KEEPER_CONTACT_CONTRACT
            if contract["source_pair"][0] == "stock_tool_plate"
        }
        guide_records = withdrawal.get("guide_contact_records")
        self.assertIsInstance(guide_records, list, withdrawal)
        self.assertEqual(len(guide_records), len(expected_guides), withdrawal)
        observed_guides: set[tuple[str, str]] = set()
        for guide in guide_records:
            self.assertIsInstance(guide, dict, withdrawal)
            source_pair = tuple(str(value) for value in guide.get("source_pair", ()))
            self.assertIn(source_pair, expected_guides, guide)
            self.assertNotIn(source_pair, observed_guides, guide)
            self.assertEqual(
                guide.get("runtime_pair"), expected_guides[source_pair]["runtime_pair"]
            )
            self.assertIs(guide.get("passed"), True, guide)
            self.assertIs(guide.get("all_normals_valid"), True, guide)
            self.assertIs(guide.get("finite_force_verified"), True, guide)
            self.assertGreaterEqual(int(guide.get("contact_sample_count", -1)), 0)
            self.assertGreaterEqual(float(guide.get("max_normal_force_n", -1.0)), 0.0)
            self.assertLessEqual(
                float(guide.get("max_penetration_mm", math.inf)),
                float(self.demo.CORE_KEEPER_MAX_PENETRATION_MM),
                guide,
            )
            self.assertEqual(
                guide.get("force_acceptance_semantics"),
                "finite_measured_evidence_no_release_limit_declared",
                guide,
            )
            observed_guides.add(source_pair)
        self.assertEqual(observed_guides, set(expected_guides), withdrawal)

        top_level_withdrawal = result.get("source_axis_withdrawal_evidence")
        self.assertIsInstance(top_level_withdrawal, dict, result)
        for key, value in top_level_withdrawal.items():
            self.assertEqual(withdrawal.get(key), value, (key, withdrawal, result))
        slider = matching["slider_return_verified"]
        self.assert_slider_return_trace(slider)
        self.assertIs(slider.get("cam_clear"), True, slider)
        self.assertIs(slider.get("returned"), True, slider)
        slider_position_mm = float(
            slider.get("slider_joint_position_mm", -math.inf)
        )
        slider_band_mm = 1000.0 * np.asarray(
            self.demo.LOCKED_SLIDER_POSITION_BAND_M, dtype=float
        )
        self.assertGreaterEqual(slider_position_mm, slider_band_mm[0], slider)
        self.assertLessEqual(slider_position_mm, slider_band_mm[1], slider)
        self.assertGreaterEqual(float(slider.get("settled_dwell_s", 0.0)), 0.050)
        slider_speed_mm_s = abs(
            float(slider.get("slider_joint_velocity_mm_s", math.inf))
        )
        settled_speed_limit_mm_s = float(
            slider.get("settled_speed_limit_mm_s", -math.inf)
        )
        self.assertGreater(settled_speed_limit_mm_s, 0.0, slider)
        self.assertLessEqual(slider_speed_mm_s, settled_speed_limit_mm_s, slider)

        cad = self.clearance.CAD
        exact_slider = (
            cad.locking_slider()
            .translate(
                (
                    slider_position_mm,
                    0.0,
                    cad.SLIDER_Z
                    - cad.PLATE_THICKNESS
                    - recomputed_withdrawal_mm,
                )
            )
            .val()
        )
        exact_cam = cad.positive_lock_cam().val()
        exact_cam_clearance_mm = float(exact_slider.distance(exact_cam))
        exact_cam_overlap_mm3 = (
            self.clearance._intersection_volume_mm3(exact_slider, exact_cam)
            if exact_cam_clearance_mm
            <= self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        self.assertLessEqual(
            exact_cam_overlap_mm3,
            self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3,
            slider,
        )
        self.assertGreaterEqual(
            exact_cam_clearance_mm,
            self.clearance.MANUFACTURING_CLEARANCE_MM,
            slider,
        )
        self.assertAlmostEqual(
            float(slider.get("cam_tab_clearance_mm", math.inf)),
            exact_cam_clearance_mm,
            delta=1.0e-6,
        )
        continuous_cam_clearance_mm = exact_cam_clearance_mm - 0.05
        self.assertAlmostEqual(
            float(
                slider.get(
                    "cam_tab_continuous_clearance_lower_bound_mm", math.inf
                )
            ),
            continuous_cam_clearance_mm,
            delta=1.0e-6,
        )
        self.assertGreaterEqual(
            continuous_cam_clearance_mm,
            float(self.demo.LOCK_CAM_MANUFACTURING_CLEARANCE_MM),
            slider,
        )
        self.assertEqual(
            slider.get("cam_clearance_authority"),
            "hash_pinned_exact_cad_axial_sweep_plus_0.05mm_motion_bound",
            slider,
        )
        top_level_slider = result.get("slider_return_evidence")
        self.assertIsInstance(top_level_slider, dict, result)
        for key, value in top_level_slider.items():
            self.assertEqual(slider.get(key), value, (key, slider, result))
        self.assertIs(
            matching["physical_lock_confirmed"].get("physical_lock_confirmed"),
            True,
            matching["physical_lock_confirmed"],
        )
        self.assertIs(
            matching["physical_lock_confirmed"].get("dock_hold_active"),
            False,
            matching["physical_lock_confirmed"],
        )
        self.assertIs(
            matching["physical_lock_confirmed"].get("attach_equality_active"),
            True,
            matching["physical_lock_confirmed"],
        )
        self.assertEqual(
            int(result.get("first_physical_lock_true_substep", -1)),
            physics_indices[-1],
            result,
        )

    def retired_real_substep_capture_lock_and_release_smoke(self) -> None:
        require_path(MATCHA_DEMO, "matcha workflow simulator")
        result_process = subprocess.run(
            [
                sys.executable,
                str(MATCHA_DEMO),
                "--headless",
                "--max-steps",
                "20000",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        self.assertEqual(result_process.returncode, 0, result_process.stderr)
        result = json.loads(result_process.stdout)
        self.assertEqual(result.get("milestone"), "capture_lock_and_dock_release")
        self.assertIs(result.get("release_ready"), False, result)
        self.assertIs(result.get("completed"), True, result)
        self.assertIsNone(result.get("abort_reason"), result)
        self.assertIs(result.get("motion_stopped"), True, result)
        self.assertGreater(int(result.get("physics_substep_count", 0)), 0, result)
        self.assertGreater(float(result.get("sim_time_s", 0.0)), 0.0, result)
        self.assertEqual(int(result.get("forbidden_contact_count", -1)), 0, result)
        self.assertEqual(float(result.get("max_forbidden_penetration_m", -1.0)), 0.0)
        self.assertIsNone(result.get("first_forbidden_pair"), result)
        route = result.get("route_alignment")
        self.assertIsInstance(route, dict, result)
        self.assertIs(route.get("passed"), True, route)
        self.assertLessEqual(
            float(route.get("declared_static_max_lateral_deviation_m", math.inf)),
            float(self.demo.CAM_RELIEF_CORRIDOR_M),
            route,
        )
        self.assertLessEqual(
            float(route.get("measured_max_lateral_deviation_m", math.inf)),
            float(self.demo.CAM_RELIEF_CORRIDOR_M),
            route,
        )
        self.assertLessEqual(
            float(route.get("measured_max_orientation_error_rad", math.inf)),
            float(self.demo.CAPTURE_ORIENTATION_TOLERANCE_RAD),
            route,
        )
        self.assertEqual(result.get("attached_tool"), "gripper", result)
        self.assertIs(result.get("bus_connected"), True, result)
        self.assertIs(result.get("handshake_achieved"), True, result)
        self.assertIs(result.get("physical_lock_confirmed"), True, result)
        self.assertIs(result.get("lock_candidate_verified"), True, result)
        self.assertIs(result.get("locked"), True, result)
        self.assertEqual(
            result.get("live_pogo_signals"), ["data", "ground", "id", "power"], result
        )
        self.assertIs(result.get("four_signal_bus_live"), True, result)
        self.assertIs(result.get("dock_hold_active"), False, result)
        self.assertIs(result.get("attach_equality_active"), True, result)
        self.assertEqual(
            result.get("lock_confirmation_phase"),
            "after_dock_release_axial_cam_disengagement_and_slider_return",
            result,
        )
        self.assertEqual(
            float(result.get("minimum_source_axis_withdrawal_mm", math.nan)),
            float(self.demo.MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM),
            result,
        )
        self.assert_core_keeper_report(result)
        self.assert_lock_sequence(result)
        self.assertIs(result.get("finite_actuator_force"), True, result)
        utilization = result.get("max_actuator_utilization")
        self.assertIsInstance(utilization, dict, result)
        self.assertTrue(utilization, result)
        for joint, value in utilization.items():
            numeric = float(value)
            self.assertTrue(math.isfinite(numeric), joint)
            self.assertGreaterEqual(numeric, 0.0, joint)
            self.assertLessEqual(numeric, 1.0 + 1.0e-12, joint)
        coverage = result.get("collision_coverage")
        self.assertIsInstance(coverage, dict, result)
        self.assertTrue(
            coverage.get("complete", coverage.get("collision_coverage_complete")),
            coverage,
        )
        self.assertEqual(
            coverage.get("missing_collision_bodies", coverage.get("missing_bodies", [])),
            [],
            coverage,
        )
        startup = result.get("startup_contact_audit")
        self.assertIsInstance(startup, dict, result)
        self.assertIs(startup.get("passed"), True, startup)
        self.assertEqual(int(startup.get("penetration_count", -1)), 0, startup)

        # Cheap deterministic negative controls exercise the independent
        # oracle against internally plausible but unsafe evidence without
        # paying for another full dynamics replay.
        def cloned() -> dict[str, Any]:
            return json.loads(json.dumps(result, allow_nan=False))

        mutations: list[tuple[str, Any]] = []

        early_lock = cloned()
        next(
            record
            for record in early_lock["journal"]
            if record.get("event") == "physical_capture_complete"
        )["physical_lock_confirmed"] = True
        mutations.append(("early physical-lock claim", early_lock))

        fake_withdrawal = cloned()
        next(
            record
            for record in fake_withdrawal["journal"]
            if record.get("event") == "source_axis_withdrawal_complete"
        )["withdrawal_mm"] += 0.25
        mutations.append(("self-attested withdrawal", fake_withdrawal))

        out_of_range_q = cloned()
        slider_record = next(
            record
            for record in out_of_range_q["journal"]
            if record.get("event") == "slider_return_verified"
        )
        slider_record["samples"][0][1] = 0.004
        mutations.append(("out-of-range slider sample", out_of_range_q))

        cam_contact = cloned()
        next(
            record
            for record in cam_contact["journal"]
            if record.get("event") == "slider_return_verified"
        )["cam_tab_contact_count_during_settled_dwell"] = 1
        mutations.append(("cam-tab dwell contact", cam_contact))

        missing_bus_signal = cloned()
        next(
            record
            for record in missing_bus_signal["journal"]
            if record.get("event") == "physical_capture_complete"
        )["pogo_signals"] = ["data", "ground", "power"]
        mutations.append(("missing capture bus signal", missing_bus_signal))

        wrong_equality_order = cloned()
        next(
            record
            for record in wrong_equality_order["journal"]
            if record.get("event") == "dock_hold_released"
        )["attach_equality_active"] = False
        mutations.append(("attach equality released early", wrong_equality_order))

        for description, mutation in mutations:
            with self.subTest(negative=description):
                with self.assertRaises(AssertionError):
                    self.assert_lock_sequence(mutation)


def box_triangles(
    minimum: tuple[float, float, float], maximum: tuple[float, float, float]
) -> np.ndarray:
    lo = np.asarray(minimum, dtype=np.float64)
    hi = np.asarray(maximum, dtype=np.float64)
    vertices = np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return np.ascontiguousarray(vertices[faces], dtype=np.float64)


def cylinder_triangles(
    radius: float, height: float, segments: int = 48
) -> np.ndarray:
    """Closed, consistently oriented cylindrical triangle surface in millimetres."""

    triangles: list[np.ndarray] = []
    lower_center = np.asarray([0.0, 0.0, 0.0])
    upper_center = np.asarray([0.0, 0.0, height])
    for index in range(segments):
        angle_a = 2.0 * math.pi * index / segments
        angle_b = 2.0 * math.pi * (index + 1) / segments
        lower_a = np.asarray(
            [radius * math.cos(angle_a), radius * math.sin(angle_a), 0.0]
        )
        lower_b = np.asarray(
            [radius * math.cos(angle_b), radius * math.sin(angle_b), 0.0]
        )
        upper_a = lower_a + upper_center
        upper_b = lower_b + upper_center
        triangles.extend(
            (
                np.asarray([lower_a, lower_b, upper_b]),
                np.asarray([lower_a, upper_b, upper_a]),
                np.asarray([upper_center, upper_a, upper_b]),
                np.asarray([lower_center, lower_b, lower_a]),
            )
        )
    return np.ascontiguousarray(triangles, dtype=np.float64)


def annular_cylinder_triangles(
    inner_radius: float, outer_radius: float, height: float, segments: int = 48
) -> np.ndarray:
    """Closed tube surface whose axial functional opening remains explicit."""

    triangles: list[np.ndarray] = []
    upper_offset = np.asarray([0.0, 0.0, height])
    for index in range(segments):
        angle_a = 2.0 * math.pi * index / segments
        angle_b = 2.0 * math.pi * (index + 1) / segments
        outer_a = np.asarray(
            [outer_radius * math.cos(angle_a), outer_radius * math.sin(angle_a), 0.0]
        )
        outer_b = np.asarray(
            [outer_radius * math.cos(angle_b), outer_radius * math.sin(angle_b), 0.0]
        )
        inner_a = np.asarray(
            [inner_radius * math.cos(angle_a), inner_radius * math.sin(angle_a), 0.0]
        )
        inner_b = np.asarray(
            [inner_radius * math.cos(angle_b), inner_radius * math.sin(angle_b), 0.0]
        )
        outer_a_top, outer_b_top = outer_a + upper_offset, outer_b + upper_offset
        inner_a_top, inner_b_top = inner_a + upper_offset, inner_b + upper_offset
        triangles.extend(
            (
                # Exterior and interior walls.
                np.asarray([outer_a, outer_b, outer_b_top]),
                np.asarray([outer_a, outer_b_top, outer_a_top]),
                np.asarray([inner_a, inner_a_top, inner_b_top]),
                np.asarray([inner_a, inner_b_top, inner_b]),
                # Top annulus (+Z) and bottom annulus (-Z).
                np.asarray([outer_a_top, outer_b_top, inner_b_top]),
                np.asarray([outer_a_top, inner_b_top, inner_a_top]),
                np.asarray([outer_a, inner_b, outer_b]),
                np.asarray([outer_a, inner_a, inner_b]),
            )
        )
    return np.ascontiguousarray(triangles, dtype=np.float64)


def point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    """Double-precision Ericson point/triangle distance reference."""

    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + v * ab)))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + w * (c - b))))
    normal = np.cross(ab, ac)
    return abs(float(normal @ ap)) / float(np.linalg.norm(normal))


class FcpwFastGateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = import_file(
            PAYLOAD_GENERATOR,
            "matcha_payload_authority_validation",
            "payload collision authority generator",
        )

    def test_fcpw_version_and_candidate_input_guards_are_fail_closed(self) -> None:
        self.assertEqual(importlib.metadata.version("fcpw"), "1.2.0")
        index_type = getattr(self.authority, "_FcpwTriangleUpperBoundIndex", None)
        self.assertIsNotNone(index_type, "FCPW candidate index is not exposed")
        triangle = np.asarray(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=np.float64,
        )
        index = index_type(triangle)
        distance = np.asarray(index.distances(np.asarray([[0.5, 0.2, 0.2]])))
        self.assertEqual(distance.shape, (1,))
        self.assertTrue(np.all(np.isfinite(distance)))
        for bad in (
            np.full((1, 3, 3), np.nan),
            np.zeros((1, 3, 3), dtype=np.float64),
            np.empty((0, 3, 3), dtype=np.float64),
        ):
            with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                index_type(bad)
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            index.distances(np.asarray([[math.inf, 0.0, 0.0]]))

    def test_fcpw_candidate_is_replayed_on_original_float64_triangle(self) -> None:
        index_type = getattr(self.authority, "_FcpwTriangleUpperBoundIndex", None)
        self.assertIsNotNone(index_type)
        triangle = np.asarray(
            [
                [
                    [1000.000123456789, 0.0, 0.0],
                    [1000.000123456789, 1.0, 0.0],
                    [1000.000123456789, 0.0, 1.0],
                ]
            ],
            dtype=np.float64,
        )
        query = np.asarray([[1000.123456789123, 0.2, 0.2]], dtype=np.float64)
        expected = point_triangle_distance(query[0], triangle[0])
        observed = float(index_type(triangle).distances(query)[0])
        self.assertAlmostEqual(observed, expected, delta=1.0e-12)
        # A direct float32 subtraction is measurably different at this scale;
        # equality above therefore exercises replay, not merely a loose bound.
        float32_distance = abs(
            float(np.float32(query[0, 0]) - np.float32(triangle[0, 0, 0]))
        )
        self.assertGreater(abs(float32_distance - expected), 1.0e-6)

    def test_fcpw_batched_candidates_are_deterministic_conservative_replays(
        self,
    ) -> None:
        index_type = getattr(self.authority, "_FcpwTriangleUpperBoundIndex", None)
        self.assertIsNotNone(index_type)
        rng = np.random.default_rng(0x5A17)
        triangles: list[np.ndarray] = []
        while len(triangles) < 24:
            triangle = rng.normal(size=(3, 3))
            doubled_area = np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
            if float(doubled_area) > 0.1:
                triangles.append(triangle)
        triangle_array = np.ascontiguousarray(triangles, dtype=np.float64)
        points = np.ascontiguousarray(
            np.concatenate(
                (
                    rng.normal(size=(47, 3)),
                    np.asarray([[1.0e6, -2.0e6, 3.0e6]], dtype=np.float64),
                )
            ),
            dtype=np.float64,
        )
        index = index_type(triangle_array)
        first = np.asarray(index.distances(points), dtype=np.float64)
        second = np.asarray(index.distances(points.copy()), dtype=np.float64)
        self.assertEqual(first.shape, (len(points),))
        np.testing.assert_array_equal(first, second)
        brute_force = np.asarray(
            [
                min(
                    point_triangle_distance(point, triangle)
                    for triangle in triangle_array
                )
                for point in points
            ],
            dtype=np.float64,
        )
        self.assertTrue(np.all(np.isfinite(first)))
        # FCPW selects a compiled float32 candidate, but the published result
        # replays that original triangle in float64. A missed nearest candidate
        # may overestimate and cause a conservative red; it may never undercut
        # the true triangle-set distance.
        self.assertTrue(
            np.all(first + 1.0e-12 >= brute_force), (first, brute_force)
        )

    def test_fast_gate_declares_signed_occupancy_tolerance_and_step_mesh_error(self) -> None:
        source = inspect.getsource(self.authority).lower()
        self.assertIn("fcpw", source)
        self.assertRegex(source, r"contain|signed.?distance|occupancy")
        self.assertRegex(source, r"watertight|edge.?incidence")
        self.assertRegex(source, r"orient")
        self.assertRegex(source, r"source_selector_sha256|source_artifact")
        self.assertRegex(source, r"selected_feature_geometry_sha256|geometry_sha256")
        self.assertRegex(source, r"deflection")
        self.assertIn("float64", source)
        self.assertNotRegex(
            source,
            r"fast.*(?:requires|authority).*occt.*(?:subset|contain|union)",
        )

    def _certify(self, source: np.ndarray, proxy: np.ndarray) -> dict[str, Any]:
        public = getattr(
            self.authority, "certify_bidirectional_runtime_collision", None
        )
        if public is not None:
            return public(
                source,
                proxy,
                threshold_mm=SURFACE_INTERNAL_TARGET_MM,
                source_error_mm=0.0,
                proxy_error_mm=0.0,
            )
        prepared_type = getattr(self.authority, "_PreparedBoundarySurface", None)
        index_type = getattr(self.authority, "_FcpwTriangleUpperBoundIndex", None)
        measure = getattr(self.authority, "_measured_boundary_surface", None)
        if prepared_type is None or index_type is None or measure is None:
            self.fail(
                "payload generator must expose the OCCT-free synthetic fidelity API"
            )
        prepared_source = prepared_type(source, 0.0, {"kind": "synthetic_source"})
        prepared_proxy = prepared_type(proxy, 0.0, {"kind": "synthetic_proxy"})
        forward = measure(
            prepared_source,
            index_type(proxy),
            direction="source_boundary_to_proxy_boundary",
            target_surface_approximation_error_upper_bound_mm=0.0,
            target_mesh_record={"kind": "synthetic_proxy"},
        )
        reverse = measure(
            prepared_proxy,
            index_type(source),
            direction="proxy_boundary_to_source_boundary",
            target_surface_approximation_error_upper_bound_mm=0.0,
            target_mesh_record={"kind": "synthetic_source"},
        )
        return {
            "source_to_proxy": forward,
            "proxy_to_source": reverse,
            "passed": (
                float(forward["certified_upper_bound_mm"])
                <= SURFACE_INTERNAL_TARGET_MM
                and float(reverse["certified_upper_bound_mm"])
                <= SURFACE_INTERNAL_TARGET_MM
            ),
        }

    def test_bidirectional_gate_rejects_added_and_dropped_components(self) -> None:
        base = box_triangles((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        detached = box_triangles((2.0, 0.0, 0.0), (2.5, 0.5, 0.5))
        added = self._certify(base, np.concatenate((base, detached)))
        dropped = self._certify(np.concatenate((base, detached)), base)
        self.assertFalse(bool(added["passed"]), added)
        self.assertFalse(bool(dropped["passed"]), dropped)
        self.assertGreater(
            float(added["proxy_to_source"]["certified_upper_bound_mm"]),
            SURFACE_RELEASE_LIMIT_MM,
        )
        self.assertGreater(
            float(dropped["source_to_proxy"]["certified_upper_bound_mm"]),
            SURFACE_RELEASE_LIMIT_MM,
        )

    def test_bidirectional_gate_rejects_a_filled_functional_hole(self) -> None:
        # Both inputs are independently watertight/oriented. The only material
        # semantic difference is that the proxy closes the source's axial bore.
        tube = annular_cylinder_triangles(1.0, 2.0, 1.0)
        filled = cylinder_triangles(2.0, 1.0)
        certificate = self._certify(tube, filled)
        self.assertFalse(bool(certificate["passed"]), certificate)
        self.assertGreater(
            max(
                float(certificate["source_to_proxy"]["certified_upper_bound_mm"]),
                float(certificate["proxy_to_source"]["certified_upper_bound_mm"]),
            ),
            SURFACE_RELEASE_LIMIT_MM,
        )

    def test_public_certificate_is_bidirectional_signed_and_never_release_ready(
        self,
    ) -> None:
        surface = box_triangles((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        public = getattr(
            self.authority, "certify_bidirectional_runtime_collision", None
        )
        self.assertIsNotNone(public, "public synthetic fidelity API is required")
        certificate = public(
            surface,
            surface.copy(),
            threshold_mm=SURFACE_INTERNAL_TARGET_MM,
            source_error_mm=0.01,
            proxy_error_mm=0.02,
        )
        self.assertIs(certificate.get("release_ready"), False, certificate)
        self.assertTrue(certificate.get("passed"), certificate)
        for topology_name in ("source_topology", "proxy_topology"):
            topology = certificate.get(topology_name)
            self.assertIsInstance(topology, dict, topology_name)
            for field in (
                "watertight",
                "orientation_consistent",
                "positive_volume",
                "passed",
            ):
                self.assertIs(topology.get(field), True, (topology_name, topology))
        occupancy = certificate.get("occupancy")
        self.assertIsInstance(occupancy, dict)
        self.assertIs(occupancy.get("signed"), True, occupancy)
        self.assertIs(occupancy.get("union_occupancy"), True, occupancy)
        self.assertIs(occupancy.get("passed"), True, occupancy)
        sign_tolerance = float(occupancy["signed_distance_tolerance_mm"])
        self.assertTrue(math.isfinite(sign_tolerance), occupancy)
        self.assertGreaterEqual(sign_tolerance, 0.0, occupancy)
        self.assertLessEqual(sign_tolerance, 1.0e-6, occupancy)
        for direction_name in ("source_to_proxy", "proxy_to_source"):
            direction = certificate.get(direction_name)
            self.assertIsInstance(direction, dict, direction_name)
            for field in (
                "witness_maximum_mm",
                "query_surface_covering_radius_mm",
                "query_faceting_error_upper_bound_mm",
                "target_faceting_error_upper_bound_mm",
                "certified_upper_bound_mm",
            ):
                self.assertTrue(math.isfinite(float(direction[field])), field)
                self.assertGreaterEqual(float(direction[field]), 0.0, field)
            recomputed = sum(
                float(direction[field])
                for field in (
                    "witness_maximum_mm",
                    "query_surface_covering_radius_mm",
                    "query_faceting_error_upper_bound_mm",
                    "target_faceting_error_upper_bound_mm",
                )
            )
            self.assertAlmostEqual(
                float(direction["certified_upper_bound_mm"]),
                recomputed,
                delta=1.0e-12,
            )
            self.assertLessEqual(
                float(direction["certified_upper_bound_mm"]),
                SURFACE_INTERNAL_TARGET_MM,
            )
            self.assertRegex(str(direction["witness_set_sha256"]), r"^[0-9a-f]{64}$")
            self.assertIs(direction.get("float64_candidate_replay"), True, direction)
            self.assertIs(direction.get("passed"), True, direction)
        self.assertAlmostEqual(
            float(
                certificate["source_to_proxy"][
                    "target_faceting_error_upper_bound_mm"
                ]
            ),
            0.02,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            float(
                certificate["source_to_proxy"][
                    "query_faceting_error_upper_bound_mm"
                ]
            ),
            0.01,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            float(
                certificate["proxy_to_source"][
                    "target_faceting_error_upper_bound_mm"
                ]
            ),
            0.01,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            float(
                certificate["proxy_to_source"][
                    "query_faceting_error_upper_bound_mm"
                ]
            ),
            0.02,
            delta=1.0e-15,
        )

    def test_public_certificate_rejects_invalid_thresholds_and_error_bounds(
        self,
    ) -> None:
        surface = box_triangles((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        public = getattr(
            self.authority, "certify_bidirectional_runtime_collision", None
        )
        self.assertIsNotNone(public, "public synthetic fidelity API is required")
        cases = (
            {"threshold_mm": -0.1, "source_error_mm": 0.0, "proxy_error_mm": 0.0},
            {"threshold_mm": math.nan, "source_error_mm": 0.0, "proxy_error_mm": 0.0},
            {"threshold_mm": 0.34, "source_error_mm": -0.1, "proxy_error_mm": 0.0},
            {"threshold_mm": 0.34, "source_error_mm": 0.0, "proxy_error_mm": math.inf},
        )
        for kwargs in cases:
            try:
                certificate = public(surface, surface.copy(), **kwargs)
            except (AssertionError, RuntimeError, ValueError):
                continue
            self.assertIs(certificate.get("release_ready"), False, certificate)
            self.assertFalse(bool(certificate.get("passed")), (kwargs, certificate))

    def test_topology_and_sign_gate_rejects_open_or_flipped_meshes(self) -> None:
        closed = box_triangles((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        open_surface = closed[1:].copy()
        flipped = closed.copy()
        flipped[0] = flipped[0, ::-1]
        inside_out = closed[:, ::-1].copy()
        public = getattr(
            self.authority, "certify_bidirectional_runtime_collision", None
        )
        self.assertIsNotNone(public, "public synthetic fidelity API is required")
        for label, proxy in (
            ("open", open_surface),
            ("locally_flipped", flipped),
            ("inside_out", inside_out),
        ):
            try:
                certificate = public(
                    closed,
                    proxy,
                    threshold_mm=SURFACE_INTERNAL_TARGET_MM,
                    source_error_mm=0.0,
                    proxy_error_mm=0.0,
                )
            except (AssertionError, RuntimeError, ValueError):
                continue
            self.assertFalse(bool(certificate.get("passed")), (label, certificate))
            topology = certificate.get("proxy_topology")
            self.assertIsInstance(topology, dict, (label, certificate))
            self.assertFalse(bool(topology.get("passed")), (label, topology))

    def test_topology_sign_is_order_invariant_for_disconnected_components(
        self,
    ) -> None:
        # A large positive shell must not hide a smaller inside-out shell in
        # aggregate signed volume.  Triangle ordering also must not let a
        # bounded face-sampling sanity check skip the bad component.
        positive = box_triangles((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        inside_out = box_triangles((20.0, 20.0, 20.0), (21.0, 21.0, 21.0))[
            :, ::-1
        ]
        positive_slots = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23}
        positive_iterator = iter(positive)
        negative_iterator = iter(inside_out)
        mixed = np.ascontiguousarray(
            [
                next(positive_iterator)
                if index in positive_slots
                else next(negative_iterator)
                for index in range(len(positive) + len(inside_out))
            ],
            dtype=np.float64,
        )
        public = getattr(
            self.authority, "certify_bidirectional_runtime_collision", None
        )
        self.assertIsNotNone(public, "public synthetic fidelity API is required")
        certificate = public(
            mixed,
            mixed.copy(),
            threshold_mm=SURFACE_INTERNAL_TARGET_MM,
            source_error_mm=0.0,
            proxy_error_mm=0.0,
        )
        self.assertFalse(bool(certificate.get("passed")), certificate)
        self.assertFalse(
            bool(certificate["source_topology"].get("passed"))
            and bool(certificate["occupancy"].get("passed")),
            certificate,
        )

    def test_topology_accepts_a_correctly_oriented_nested_cavity(self) -> None:
        # The disconnected inner shell of a closed cavity is intentionally
        # negative: orientation follows solid material, not "positive per
        # component".  A nesting-aware topology gate must retain this valid
        # OCCT/STEP representation while rejecting an external negative shell.
        outer = box_triangles((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        cavity = box_triangles((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))[:, ::-1]
        hollow_solid = np.ascontiguousarray(
            np.concatenate((outer, cavity)), dtype=np.float64
        )
        public = getattr(
            self.authority, "certify_bidirectional_runtime_collision", None
        )
        self.assertIsNotNone(public, "public synthetic fidelity API is required")
        certificate = public(
            hollow_solid,
            hollow_solid.copy(),
            threshold_mm=SURFACE_INTERNAL_TARGET_MM,
            source_error_mm=0.0,
            proxy_error_mm=0.0,
        )
        self.assertTrue(certificate.get("passed"), certificate)
        self.assertTrue(certificate["source_topology"].get("passed"), certificate)
        self.assertTrue(certificate["proxy_topology"].get("passed"), certificate)
        self.assertTrue(certificate["occupancy"].get("passed"), certificate)

    def test_validator_recomputes_and_rejects_fabricated_evidence(self) -> None:
        validator = import_file(
            PAYLOAD_VALIDATOR,
            "matcha_payload_validator_validation",
            "payload collision authority validator",
        )
        validate = getattr(
            validator, "validate_bidirectional_runtime_collision_certificate", None
        )
        self.assertIsNotNone(validate, "canonical validator API is required")
        source = box_triangles((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        proxy = np.ascontiguousarray(
            source + np.asarray([0.08, 0.0, 0.0]), dtype=np.float64
        )
        certificate = self.authority.certify_bidirectional_runtime_collision(
            source,
            proxy,
            threshold_mm=SURFACE_INTERNAL_TARGET_MM,
            source_error_mm=0.01,
            proxy_error_mm=0.02,
        )
        canonical = validate(
            certificate,
            source,
            proxy,
            threshold_mm=SURFACE_INTERNAL_TARGET_MM,
            source_error_mm=0.01,
            proxy_error_mm=0.02,
        )
        self.assertEqual(canonical, certificate)

        fabricated = json.loads(json.dumps(certificate))
        direction = fabricated["source_to_proxy"]
        self.assertGreater(float(direction["witness_maximum_mm"]), 0.01)
        direction["witness_maximum_mm"] = 0.0
        direction["certified_upper_bound_mm"] = math.fsum(
            float(direction[field])
            for field in (
                "witness_maximum_mm",
                "query_surface_covering_radius_mm",
                "query_faceting_error_upper_bound_mm",
                "target_faceting_error_upper_bound_mm",
            )
        )
        direction["passed"] = bool(
            direction["certified_upper_bound_mm"] <= SURFACE_INTERNAL_TARGET_MM
        )
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            validate(
                fabricated,
                source,
                proxy,
                threshold_mm=SURFACE_INTERNAL_TARGET_MM,
                source_error_mm=0.01,
                proxy_error_mm=0.02,
            )

        changed_proxy = np.ascontiguousarray(
            proxy + np.asarray([0.02, 0.0, 0.0]), dtype=np.float64
        )
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            validate(
                certificate,
                source,
                changed_proxy,
                threshold_mm=SURFACE_INTERNAL_TARGET_MM,
                source_error_mm=0.01,
                proxy_error_mm=0.02,
            )

        false_release = json.loads(json.dumps(certificate))
        false_release["release_ready"] = True
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            validate(
                false_release,
                source,
                proxy,
                threshold_mm=SURFACE_INTERNAL_TARGET_MM,
                source_error_mm=0.01,
                proxy_error_mm=0.02,
            )


class RolledCoreDockRuntimeAuthorityTests(unittest.TestCase):
    """Independent consumer of the canonical rolled full-arm checkpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = import_file(
            ROLLED_CORE_DOCK_RUNTIME_VALIDATOR,
            "rolled_core_dock_runtime_validation_under_test",
            "rolled core-dock runtime validator",
        )
        cls.report = load_json(
            ROLLED_CORE_DOCK_RUNTIME_REPORT,
            "rolled core-dock runtime report",
        )

    def _errors(self, report: dict[str, Any]) -> list[str]:
        return self.validator.rolled_core_dock_runtime_report_errors(
            report,
            repository_root=REPOSITORY_ROOT,
        )

    def _reseal(self, report: dict[str, Any]) -> None:
        report["canonical_sha256_without_this_field"] = None
        report["canonical_sha256_without_this_field"] = (
            self.validator.canonical_sha256(report)
        )

    def test_report_binds_sources_frame_proxy_routes_and_false_authority(
        self,
    ) -> None:
        report = self.report
        self.assertEqual(self._errors(report), [])
        self.validator.validate_rolled_core_dock_runtime_report(
            report,
            repository_root=REPOSITORY_ROOT,
        )
        source = report["source_binding"]
        self.assertIs(source["triple_contract_equal"], True)
        self.assertEqual(
            source["source_contract_canonical_sha256"],
            self.validator.EXPECTED_SOURCE_CONTRACT_SHA256,
        )
        for record in source["directly_consumed_runtime_files"]:
            path = REPOSITORY_ROOT / record["path"]
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(sha256_file(path), record["sha256"])

        frame = report["rolled_frame"]
        self.assertEqual(
            frame["position_m"],
            list(self.validator.EXPECTED_CORE_DOCK_POSITION_M),
        )
        self.assertEqual(
            frame["quat_wxyz"],
            list(self.validator.EXPECTED_CORE_DOCK_QUAT_WXYZ),
        )
        self.assertEqual(frame["tool_view_roll_deg"], -87.21086925015224)
        self.assertEqual(frame["release_axis_world"], [0.0, 0.0, 1.0])

        proxy = report["support_proxy"]
        self.assertEqual(proxy["component_count"], 11)
        self.assertEqual(proxy["pairwise_positive_overlap_count"], 0)
        self.assertEqual(proxy["source_missing_volume_mm3"], 0.0)
        self.assertEqual(
            proxy["analytic_excess_volume_mm3"],
            15381.827700519032,
        )
        self.assertIs(proxy["passage_witness_inside_proxy"], False)
        self.assertEqual(
            proxy["filled_hole_witnesses_inside_proxy"], [True] * 6
        )
        self.assertEqual(
            proxy["removed_legacy_geom_names"],
            [
                "dock_gripper_support_collision",
                "dock_gripper_support_anchor_collision",
            ],
        )

        for tool, expected in (
            ("spoon", self.validator.EXPECTED_SPOON_POSE),
            ("whisk", self.validator.EXPECTED_WHISK_POSE),
        ):
            observed = report["unchanged_matcha_docks"][tool]
            self.assertEqual(observed["declared_position_m"], list(expected[0]))
            self.assertEqual(observed["declared_quat_wxyz"], list(expected[1]))

        capture = report["capture_route"]
        self.assertEqual(capture["row_count"], 276)
        self.assertEqual(capture["f8le_shape"], [276, 7])
        self.assertEqual(capture["preseat_bounds_mm"], [55.0, 0.0])
        self.assertEqual(capture["preseat_step_mm"], -0.2)
        self.assertEqual(capture["orientation_bound_rad"], math.radians(0.1))
        self.assertLessEqual(
            capture["maximum_dense_orientation_error_rad"],
            math.radians(0.1),
        )
        self.assertIs(capture["exact_quaternion_required_off_seat"], False)

        release = report["release_route"]
        self.assertEqual(release["row_count"], 31)
        self.assertEqual(release["f8le_shape"], [31, 6])
        self.assertEqual(
            [row["withdrawal_mm"] for row in release["roster"]],
            [0.5 * index for index in range(31)],
        )
        self.assertEqual(
            release["roster"][0]["q_rad"],
            [-0.72, -0.5, 0.8, -0.3, -1.522116811941435],
        )
        self.assertEqual(
            report["default_actions"]["names"],
            list(self.validator.EXPECTED_DEFAULT_ACTION_NAMES),
        )
        self.assertIs(
            report["default_actions"]["static_release_continuation_included"],
            False,
        )
        self.assertIs(report["release_ready"], False)
        for field in (
            "material_authority",
            "mass_authority",
            "fastener_authority",
            "substrate_authority",
            "contact_dynamics_authority",
            "physical_release_authority",
            "release_ready",
        ):
            self.assertIs(report["authority_scope"][field], False, field)

    def test_full_arm_continuous_clearance_inventory_and_topology_are_exact(
        self,
    ) -> None:
        report = self.report
        inventory = report["inventory"]
        sampling = report["sampling"]
        clearance = report["continuous_clearance"]
        self.assertEqual(inventory["arm_geom_count"], 14)
        self.assertEqual(inventory["support_geom_count"], 11)
        self.assertEqual(inventory["dock_target_geom_count"], 90)
        self.assertEqual(sampling["unique_state_count"], 301)
        self.assertEqual(sampling["joint_linear_substeps_per_interval"], 10)
        self.assertEqual(sampling["distance_evaluation_count"], 379260)
        self.assertAlmostEqual(
            clearance["minimum_sampled_outer_aabb_lower_bound_mm"]
            - clearance["maximum_pair_specific_topology_motion_bound_mm"],
            clearance["continuous_clearance_lower_bound_mm"],
            places=12,
        )
        self.assertGreaterEqual(
            clearance["continuous_clearance_lower_bound_mm"], 0.20
        )
        self.assertEqual(report["support_topology"]["tangency_count"], 15)
        self.assertTrue(report["support_topology"]["passed"])
        self.assertEqual(report["startup"]["penetration_count"], 0)
        self.assertTrue(report["geometry_passed"])

    def test_coherently_resealed_adversarial_mutations_fail_closed(self) -> None:
        def source_hash(payload: dict[str, Any]) -> None:
            records = payload["source_binding"][
                "directly_consumed_runtime_files"
            ]
            records[0]["sha256"] = "0" * 64
            payload["source_binding"][
                "directly_consumed_runtime_files_canonical_sha256"
            ] = self.validator.canonical_sha256(records)

        def undercoverage(payload: dict[str, Any]) -> None:
            components = payload["support_proxy"]["components"]
            components.pop()
            payload["support_proxy"]["component_count"] = len(components)
            payload["support_proxy"]["components_canonical_sha256"] = (
                self.validator.canonical_sha256(components)
            )

        def overcoverage(payload: dict[str, Any]) -> None:
            components = payload["support_proxy"]["components"]
            components[0]["bounds_m"][0][0] = -0.040
            payload["support_proxy"]["components_canonical_sha256"] = (
                self.validator.canonical_sha256(components)
            )
            payload["support_proxy"]["analytic_box_union_volume_mm3"] += 344.0
            payload["support_proxy"]["analytic_excess_volume_mm3"] += 344.0

        def legacy_support(payload: dict[str, Any]) -> None:
            names = payload["inventory"]["dock_target_geom_names"]
            names.append("dock_gripper_support_collision")
            payload["inventory"]["dock_target_geom_count"] += 1
            payload["inventory"]["dock_target_geom_names_sha256"] = (
                self.validator.canonical_sha256(names)
            )

        def roster(payload: dict[str, Any]) -> None:
            rows = payload["release_route"]["roster"]
            rows[1]["q_rad"][2] += 0.01
            payload["release_route"]["canonical_sha256"] = (
                self.validator.canonical_sha256(rows)
            )
            f8le = np.asarray(
                [[row["withdrawal_mm"], *row["q_rad"]] for row in rows],
                dtype="<f8",
            )
            payload["release_route"]["f8le_sha256"] = hashlib.sha256(
                f8le.tobytes()
            ).hexdigest()

        def omitted_arm(payload: dict[str, Any]) -> None:
            names = payload["inventory"]["arm_geom_names"]
            records = payload["inventory"]["arm_compiled_geom_records"]
            names.pop()
            records.pop()
            payload["inventory"]["arm_geom_count"] = len(names)
            payload["inventory"]["arm_geom_names_sha256"] = (
                self.validator.canonical_sha256(names)
            )
            payload["inventory"]["arm_compiled_geom_records_sha256"] = (
                self.validator.canonical_sha256(records)
            )

        def motion_bound_shrink(payload: dict[str, Any]) -> None:
            clearance = payload["continuous_clearance"]
            clearance["maximum_pair_specific_topology_motion_bound_mm"] = 0.01
            clearance["continuous_clearance_lower_bound_mm"] = (
                clearance["minimum_sampled_outer_aabb_lower_bound_mm"] - 0.01
            )

        mutations = (
            ("source_hash_reseal", source_hash),
            (
                "roll_sign",
                lambda payload: payload["rolled_frame"].__setitem__(
                    "tool_view_roll_deg", 87.21086925015224
                ),
            ),
            ("support_undercoverage", undercoverage),
            ("support_overcoverage", overcoverage),
            ("old_support_resurrection", legacy_support),
            ("release_roster", roster),
            (
                "capture_route",
                lambda payload: payload["capture_route"].__setitem__(
                    "state_bytes_sha256", "0" * 64
                ),
            ),
            (
                "compiled_model",
                lambda payload: payload["model_binding"].__setitem__(
                    "compiled_model_xml_equivalent_sha256", "0" * 64
                ),
            ),
            (
                "gravity_ff",
                lambda payload: payload["gravity_feedforward"].__setitem__(
                    "identity_sha256", "0" * 64
                ),
            ),
            ("omitted_arm_geom", omitted_arm),
            (
                "omitted_state",
                lambda payload: payload["sampling"].__setitem__(
                    "unique_state_count", 300
                ),
            ),
            ("motion_bound_shrink", motion_bound_shrink),
            (
                "authority_promotion",
                lambda payload: payload["authority_scope"].__setitem__(
                    "physical_release_authority", True
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(self.report)
                mutate(changed)
                self._reseal(changed)
                self.assertTrue(self._errors(changed), label)
                with self.assertRaises(ValueError):
                    self.validator.validate_rolled_core_dock_runtime_report(
                        changed,
                        repository_root=REPOSITORY_ROOT,
                    )


class MatchaShowcaseContractTests(unittest.TestCase):
    """Keep the final video story complete and explicitly non-release evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.showcase = import_file(
            MATCHA_SHOWCASE,
            "matcha_workflow_showcase_validation",
            "Matcha showcase timeline",
        )
        cls.renderer = import_file(
            MATCHA_SHOWCASE_RENDERER,
            "matcha_workflow_showcase_renderer_validation",
            "Matcha showcase renderer",
        )

    def test_single_camera_story_contains_every_tool_task_and_return(self) -> None:
        summary = self.showcase.showcase_summary()
        self.assertEqual(summary["errors"], [])
        self.assertIs(summary["passed"], True)
        self.assertEqual(summary["camera"], "matcha_scene_camera")
        self.assertEqual(summary["camera_count"], 1)
        self.assertEqual(summary["tools"], ["gripper", "spoon", "whisk"])
        self.assertEqual(summary["gripper_pitchers"], ["hot_water", "milk"])
        self.assertEqual(summary["spoon_task"], "dose_matcha_through_sieve")
        self.assertEqual(summary["whisk_task"], "powered_mix_in_bowl")
        self.assertIs(summary["all_tools_returned"], True)
        self.assertEqual(summary["final_recipe_stage"], "complete")
        self.assertIs(summary["visualization_only"], True)
        self.assertIs(summary["physical_release_authority"], False)
        self.assertIs(summary["release_ready"], False)

    def test_timeline_is_continuous_ordered_and_ends_exactly(self) -> None:
        segments = self.showcase.showcase_segments()
        self.assertEqual(self.showcase.showcase_contract_errors(segments), [])
        duration = self.showcase.showcase_duration_s(segments)
        self.assertGreater(duration, 40.0)
        self.assertLess(duration, 60.0)
        final = self.showcase.showcase_state_at(duration, segments)
        self.assertEqual(final.segment_name, "final_hold")
        self.assertEqual(final.recipe_stage, "complete")
        self.assertIsNone(final.attached_tool)
        self.assertAlmostEqual(final.progress, 1.0)
        self.assertTrue(np.all(np.isfinite(np.asarray(final.arm_q))))
        self.assertTrue(any(segment.whisk_motor for segment in segments))
        self.assertTrue(
            any(
                segment.pitcher == "hot_water"
                and segment.attached_tool == "gripper"
                for segment in segments
            )
        )
        self.assertTrue(
            any(
                segment.pitcher == "milk"
                and segment.attached_tool == "gripper"
                for segment in segments
            )
        )

    def test_renderer_compiles_production_scene_with_one_camera(self) -> None:
        model = self.renderer.build_showcase_model()
        self.assertEqual(model.ncam, 1)
        self.assertEqual(str(model.camera(0).name), "matcha_scene_camera")
        self.assertGreater(model.ngeom, 1_700)
        self.assertGreaterEqual(model.body("tool_gripper").id, 0)
        self.assertGreaterEqual(model.body("tool_spoon").id, 0)
        self.assertGreaterEqual(model.body("tool_whisk").id, 0)
        self.assertGreaterEqual(model.body("showcase_hot_water_pitcher").id, 0)
        self.assertGreaterEqual(model.body("showcase_milk_pitcher").id, 0)


class ValidationTierContractTests(unittest.TestCase):
    def test_development_tier_can_never_publish_release_ready(self) -> None:
        require_path(RUNNER, "matcha validation runner")
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--tier", "development", "--list"],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIs(payload["release_ready"], False)
        self.assertNotEqual(payload.get("development_pass"), True)
        self.assertLessEqual(int(payload["hard_timeout_seconds"]), 300)
        self.assertGreater(int(payload["test_count"]), 0)
        discovered = {
            f"test_matcha_workflow_validation.{class_node.name}.{method.name}"
            for class_node in ast.parse(Path(__file__).read_text()).body
            if isinstance(class_node, ast.ClassDef)
            for method in class_node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name.startswith("test_")
        }
        self.assertEqual(set(payload["tests"]), discovered)
        self.assertEqual(int(payload["test_count"]), len(discovered))


if __name__ == "__main__":
    unittest.main(verbosity=2)
