#!/usr/bin/env python3
"""Fail-closed load and timing authority for the passive positive-lock cam.

The CAD source publishes nominal geometry and a frictionless quasistatic
summary.  Neither is sufficient to authorize a passive mechanism.  This
module independently derives the two cam slopes, propagates qualified
dimensional tolerances, applies adverse Coulomb-friction signs in both travel
directions, and closes timing, contact, root-strength, and robot-route load
evidence.

No evidence is embedded here.  The current source is therefore intentionally
release-red.  A screening friction interval of 0.20--0.25 is used only to make
the present self-lock risk explicit; it is not promoted to a qualified
material-pair coefficient.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
QUICK_CHANGE_DIR = HERE.parent
CAD_GENERATOR_PATH = QUICK_CHANGE_DIR / "generate_cad.py"

SCHEMA_VERSION = "1.0"
SCREENING_MU_INTERVAL = (0.20, 0.25)
REQUIRED_ROUTE_DIRECTIONS = (
    "capture_positive_z",
    "release_negative_y",
    "reverse_insertion_positive_y",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_cad_source() -> Any:
    spec = importlib.util.spec_from_file_location(
        "positive_lock_cam_load_cad_source", CAD_GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CAD source: {CAD_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAD = _load_cad_source()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _qualified_provenance(
    record: Any, *, source_sha256: str
) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["record_missing"]
    if record.get("qualified") is not True:
        errors.append("qualified_flag_missing")
    if record.get("source_sha256") != source_sha256:
        errors.append("source_sha256_mismatch")
    if not _valid_sha256(record.get("qualification_report_sha256")):
        errors.append("qualification_report_sha256_missing")
    if not isinstance(record.get("method"), str) or not record.get("method"):
        errors.append("method_missing")
    return errors


def _source_slopes(contract: Mapping[str, Any]) -> dict[str, float | bool]:
    """Derive both nominal slopes from geometry, not published slope fields."""

    lead = contract["axial_lead"]
    lower = lead["lower_rectangle_mm"]
    upper = lead["upper_rectangle_mm"]
    capture_run = float(lower["x"][0]) - float(upper["x"][0])
    capture_rise = float(upper["z"]) - float(lower["z"])

    polygon = contract["main_xy_wedge"]["polygon_xy_mm"]
    y_min = min(float(point[1]) for point in polygon)
    y_max = max(float(point[1]) for point in polygon)
    outer_min_x = min(
        float(point[0])
        for point in polygon
        if math.isclose(float(point[1]), y_min, abs_tol=1.0e-12)
    )
    inner_x = min(
        float(point[0])
        for point in polygon
        if math.isclose(float(point[1]), y_max, abs_tol=1.0e-12)
    )
    return_run = outer_min_x - inner_x
    return_rise = y_max - y_min

    values = (capture_run, capture_rise, return_run, return_rise)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("cam run/rise geometry must be finite and positive")
    capture_slope = capture_run / capture_rise
    return_slope = return_run / return_rise
    published_capture = float(lead["run_mm"]) / float(lead["rise_mm"])
    published_return = float(
        contract["passive_release"]["q_per_withdrawal_slope"]
    )
    return {
        "capture_run_mm": capture_run,
        "capture_rise_mm": capture_rise,
        "capture_nominal": capture_slope,
        "return_run_mm": return_run,
        "return_rise_mm": return_rise,
        "return_nominal": return_slope,
        "published_values_consistent": bool(
            math.isclose(capture_slope, published_capture, abs_tol=1.0e-12)
            and math.isclose(return_slope, published_return, abs_tol=1.0e-12)
        ),
    }


def _slope_interval(
    *,
    name: str,
    run_mm: float,
    rise_mm: float,
    qualification: Any,
    source_sha256: str,
) -> dict[str, Any]:
    nominal = run_mm / rise_mm
    errors = _qualified_provenance(
        qualification, source_sha256=source_sha256
    )
    item = qualification.get(name, {}) if isinstance(qualification, Mapping) else {}
    run_tolerance = item.get("run_absolute_tolerance_mm")
    rise_tolerance = item.get("rise_absolute_tolerance_mm")
    if not _positive(run_tolerance):
        errors.append(f"{name}_run_tolerance_not_positive")
    if not _positive(rise_tolerance):
        errors.append(f"{name}_rise_tolerance_not_positive")

    qualified = not errors
    lower = nominal
    upper = nominal
    if qualified:
        run_tolerance = float(run_tolerance)
        rise_tolerance = float(rise_tolerance)
        if run_tolerance >= run_mm:
            errors.append(f"{name}_run_tolerance_exhausts_dimension")
        if rise_tolerance >= rise_mm:
            errors.append(f"{name}_rise_tolerance_exhausts_dimension")
        qualified = not errors
        if qualified:
            lower = (run_mm - run_tolerance) / (rise_mm + rise_tolerance)
            upper = (run_mm + run_tolerance) / (rise_mm - rise_tolerance)

    return {
        "name": name,
        "run_nominal_mm": run_mm,
        "rise_nominal_mm": rise_mm,
        "nominal_slope": nominal,
        "run_absolute_tolerance_mm": (
            float(run_tolerance) if _finite(run_tolerance) else None
        ),
        "rise_absolute_tolerance_mm": (
            float(rise_tolerance) if _finite(rise_tolerance) else None
        ),
        "lower_slope": lower,
        "upper_slope": upper,
        "nominal_only": not qualified,
        "qualified": qualified,
        "errors": sorted(set(errors)),
        "formula": "[(run-t_run)/(rise+t_rise),(run+t_run)/(rise-t_rise)]",
    }


def _friction_interval(
    qualification: Any, *, source_sha256: str
) -> dict[str, Any]:
    errors = _qualified_provenance(
        qualification, source_sha256=source_sha256
    )
    mu_min = qualification.get("mu_min") if isinstance(qualification, Mapping) else None
    mu_max = qualification.get("mu_max") if isinstance(qualification, Mapping) else None
    if not _finite(mu_min) or float(mu_min) < 0.0:
        errors.append("mu_min_invalid")
    if not _finite(mu_max) or float(mu_max) < 0.0:
        errors.append("mu_max_invalid")
    if _finite(mu_min) and _finite(mu_max):
        if float(mu_min) >= float(mu_max):
            errors.append("mu_interval_not_strictly_increasing")
    if not isinstance(qualification, Mapping) or not qualification.get("material_pair"):
        errors.append("material_pair_missing")
    if not isinstance(qualification, Mapping) or not qualification.get(
        "environment_envelope"
    ):
        errors.append("environment_envelope_missing")

    qualified = not errors
    used_min, used_max = SCREENING_MU_INTERVAL
    if qualified:
        used_min, used_max = float(mu_min), float(mu_max)
    return {
        "qualified": qualified,
        "qualified_interval": [float(mu_min), float(mu_max)]
        if qualified
        else None,
        "analysis_interval": [used_min, used_max],
        "analysis_interval_role": (
            "qualified_material_pair_envelope"
            if qualified
            else "screening_only_not_release_authority"
        ),
        "adverse_coefficient": used_max,
        "sensitivity_coefficient": used_min,
        "errors": sorted(set(errors)),
    }


def _opening_timing(
    contract: Mapping[str, Any],
    qualification: Any,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    capture = contract["passive_capture"]
    breakpoints = capture["lateral_offset_breakpoints_mm"]
    start_p, start_x = map(float, breakpoints[0])
    end_p, end_x = map(float, breakpoints[1])
    if not start_p > end_p:
        raise ValueError("capture recenter breakpoints are not ordered")
    lateral_slope = (start_x - end_x) / (start_p - end_p)
    lateral_intercept = end_x - lateral_slope * end_p

    coefficients = capture["ramp_q_affine_coefficients"]
    a = float(coefficients["preseat"])
    b = float(coefficients["lateral_offset"])
    c = float(coefficients["constant"])
    q_slope = a + b * lateral_slope
    q_intercept = b * lateral_intercept + c
    if math.isclose(q_slope, 0.0, abs_tol=1.0e-15):
        raise ValueError("capture q slope is degenerate")
    q_open = float(capture["passive_open_q_max_mm"])
    q_locked = float(coefficients["clamp_mm"][1])
    fully_open_p = (q_open - q_intercept) / q_slope
    contact_start_p = (q_locked - q_intercept) / q_slope
    head_entry_p = float(capture["head_entry_tangent_preseat_mm"])
    nominal_margin = fully_open_p - head_entry_p

    errors = _qualified_provenance(
        qualification, source_sha256=source_sha256
    )
    open_tol = (
        qualification.get("fully_open_preseat_absolute_tolerance_mm")
        if isinstance(qualification, Mapping)
        else None
    )
    head_tol = (
        qualification.get("head_entry_absolute_tolerance_mm")
        if isinstance(qualification, Mapping)
        else None
    )
    if not _positive(open_tol):
        errors.append("fully_open_tolerance_not_positive")
    if not _positive(head_tol):
        errors.append("head_entry_tolerance_not_positive")
    if not isinstance(qualification, Mapping) or qualification.get(
        "includes_slope_interval_propagation"
    ) is not True:
        errors.append("slope_interval_propagation_missing")
    qualified = not errors
    certified_margin = None
    if qualified:
        certified_margin = nominal_margin - float(open_tol) - float(head_tol)
    passed = bool(qualified and certified_margin is not None and certified_margin > 0.0)
    return {
        "lateral_recenter_slope": lateral_slope,
        "q_per_preseat_slope": q_slope,
        "ramp_contact_start_preseat_mm": contact_start_p,
        "fully_open_preseat_mm": fully_open_p,
        "head_entry_tangent_preseat_mm": head_entry_p,
        "nominal_open_before_head_margin_mm": nominal_margin,
        "fully_open_preseat_absolute_tolerance_mm": (
            float(open_tol) if _finite(open_tol) else None
        ),
        "head_entry_absolute_tolerance_mm": (
            float(head_tol) if _finite(head_tol) else None
        ),
        "certified_open_before_head_margin_mm": certified_margin,
        "nominal_ordering_passed": nominal_margin > 0.0,
        "qualified": qualified,
        "passed": passed,
        "errors": sorted(set(errors)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator <= 0.0:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _mechanics(
    contract: Mapping[str, Any],
    capture_interval: Mapping[str, Any],
    return_interval: Mapping[str, Any],
    friction: Mapping[str, Any],
) -> dict[str, Any]:
    load = contract["quasistatic_load_envelope"]
    spring_rate = float(load["return_spring_rate_n_per_mm"])
    maximum_deflection = float(load["maximum_spring_deflection_mm"])
    maximum_spring_force = spring_rate * maximum_deflection
    travel = float(contract["passive_capture"]["ramp_q_affine_coefficients"]["clamp_mm"][1])
    open_q = float(contract["passive_capture"]["passive_open_q_max_mm"])
    spring_force_at_open = spring_rate * (maximum_deflection - open_q)
    spring_force_at_locked = spring_rate * (maximum_deflection - travel)
    spring_consistent = math.isclose(
        maximum_spring_force,
        float(load["maximum_spring_force_n"]),
        abs_tol=1.0e-12,
    )

    mu = float(friction["adverse_coefficient"])
    mu_sensitivity = float(friction["sensitivity_coefficient"])
    capture_k = float(capture_interval["upper_slope"])
    return_k = float(return_interval["lower_slope"])

    capture_denominator = 1.0 - mu * capture_k
    capture_axial_ratio = _safe_ratio(capture_k + mu, capture_denominator)
    capture_normal_ratio = _safe_ratio(
        math.hypot(1.0, capture_k), capture_denominator
    )
    capture_axial_force = (
        maximum_spring_force * capture_axial_ratio
        if capture_axial_ratio is not None
        else None
    )
    capture_normal_force = (
        maximum_spring_force * capture_normal_ratio
        if capture_normal_ratio is not None
        else None
    )

    return_denominator = 1.0 + mu * return_k
    return_jam_margin = return_k - mu
    return_ratio = _safe_ratio(return_jam_margin, return_denominator)
    return_force_open = (
        spring_force_at_open * return_ratio if return_ratio is not None else None
    )
    return_force_locked = (
        spring_force_at_locked * return_ratio if return_ratio is not None else None
    )

    reverse_denominator = return_k - mu
    reverse_drive_ratio = _safe_ratio(
        1.0 + mu * return_k, reverse_denominator
    )
    reverse_normal_ratio = _safe_ratio(
        math.hypot(1.0, return_k), reverse_denominator
    )
    reverse_drive_force = (
        spring_force_at_open * reverse_drive_ratio
        if reverse_drive_ratio is not None
        else None
    )
    reverse_normal_force = (
        spring_force_at_open * reverse_normal_ratio
        if reverse_normal_ratio is not None
        else None
    )

    sensitivity_return_margin = return_k - mu_sensitivity
    sensitivity_return_ratio = _safe_ratio(
        sensitivity_return_margin, 1.0 + mu_sensitivity * return_k
    )
    sensitivity_reverse_ratio = _safe_ratio(
        1.0 + mu_sensitivity * return_k,
        sensitivity_return_margin,
    )

    force_requirements = {
        "capture_positive_z": capture_axial_force,
        "release_negative_y": (
            abs(return_force_open) if return_force_open is not None else None
        ),
        "reverse_insertion_positive_y": reverse_drive_force,
    }
    return {
        "spring": {
            "rate_n_per_mm": spring_rate,
            "maximum_deflection_mm": maximum_deflection,
            "maximum_force_n": maximum_spring_force,
            "force_at_passive_open_q_n": spring_force_at_open,
            "force_at_locked_q_n": spring_force_at_locked,
            "published_maximum_force_consistent": spring_consistent,
        },
        "adverse_mu": mu,
        "capture": {
            "adverse_slope": capture_k,
            "denominator_margin": capture_denominator,
            "axial_force_n": capture_axial_force,
            "normal_force_n": capture_normal_force,
            "formula_axial": "Fs*(k+mu)/(1-mu*k)",
            "formula_normal": "Fs*sqrt(1+k^2)/(1-mu*k)",
            "jam_free": capture_denominator > 0.0,
        },
        "passive_return_negative_y": {
            "adverse_slope": return_k,
            "self_lock_margin_k_minus_mu": return_jam_margin,
            "positive_denominator": return_denominator,
            "available_force_at_open_q_n": return_force_open,
            "available_force_at_locked_q_n": return_force_locked,
            "formula": "Fs*(k-mu)/(1+mu*k)",
            "jam_free": return_jam_margin > 0.0,
        },
        "reverse_insertion_positive_y": {
            "adverse_slope": return_k,
            "denominator_margin_k_minus_mu": reverse_denominator,
            "required_drive_force_n": reverse_drive_force,
            "normal_force_n": reverse_normal_force,
            "formula_drive": "Fs*(1+mu*k)/(k-mu)",
            "formula_normal": "Fs*sqrt(1+k^2)/(k-mu)",
            "jam_free": reverse_denominator > 0.0,
        },
        "mu_min_sensitivity": {
            "mu": mu_sensitivity,
            "return_available_force_at_open_q_n": (
                spring_force_at_open * sensitivity_return_ratio
                if sensitivity_return_ratio is not None
                else None
            ),
            "reverse_required_drive_force_n": (
                spring_force_at_open * sensitivity_reverse_ratio
                if sensitivity_reverse_ratio is not None
                else None
            ),
        },
        "route_force_requirements_n": force_requirements,
    }


def _finite_contact_evidence(
    evidence: Any,
    *,
    source_sha256: str,
    source_full_face_area_mm2: float,
    mechanics: Mapping[str, Any],
) -> dict[str, Any]:
    errors = _qualified_provenance(evidence, source_sha256=source_sha256)
    if not isinstance(evidence, Mapping):
        evidence = {}
    method = evidence.get("contact_patch_method")
    forbidden_methods = {
        "line_contact",
        "nominal_line_contact",
        "full_source_face_mean_pressure",
        "full_face_pressure",
    }
    if method in forbidden_methods or not isinstance(method, str) or not method:
        errors.append("finite_contact_patch_method_missing_or_nonfinite")
    if evidence.get("uses_full_source_face_area") is not False:
        errors.append("full_source_face_area_assumption_not_rejected")
    if evidence.get("finite_patch_lower_bound") is not True:
        errors.append("finite_patch_lower_bound_missing")
    area = evidence.get("minimum_contact_area_mm2")
    if not _positive(area):
        errors.append("minimum_contact_area_missing")
    elif float(area) >= source_full_face_area_mm2:
        errors.append("minimum_contact_area_is_full_source_face")
    allowable = evidence.get("allowable_contact_pressure_mpa")
    required_sf = evidence.get("required_safety_factor")
    if not _positive(allowable):
        errors.append("contact_allowable_missing")
    if not _positive(required_sf) or float(required_sf) <= 1.0:
        errors.append("contact_safety_factor_missing")
    if not _valid_sha256(evidence.get("load_case_sha256")):
        errors.append("contact_load_case_sha256_missing")

    normals = [
        mechanics["capture"]["normal_force_n"],
        mechanics["reverse_insertion_positive_y"]["normal_force_n"],
    ]
    if any(value is None for value in normals):
        errors.append("finite_adverse_normal_force_unavailable")
        worst_normal = None
    else:
        worst_normal = max(float(value) for value in normals)
    pressure = None
    achieved_sf = None
    if worst_normal is not None and _positive(area):
        pressure = worst_normal / float(area)
        if _positive(allowable) and pressure > 0.0:
            achieved_sf = float(allowable) / pressure
            if _positive(required_sf) and achieved_sf < float(required_sf):
                errors.append("contact_pressure_safety_factor_failed")
    return {
        "required_semantics": "finite_contact_patch_lower_bound_not_line_or_full_face",
        "source_full_face_area_mm2_diagnostic_only": source_full_face_area_mm2,
        "method": method,
        "minimum_contact_area_mm2": float(area) if _finite(area) else None,
        "worst_adverse_normal_force_n": worst_normal,
        "recomputed_maximum_pressure_mpa": pressure,
        "recomputed_safety_factor": achieved_sf,
        "qualified": not errors,
        "passed": not errors,
        "errors": sorted(set(errors)),
    }


def _root_strength_evidence(
    evidence: Any, *, source_sha256: str
) -> dict[str, Any]:
    errors = _qualified_provenance(evidence, source_sha256=source_sha256)
    if not isinstance(evidence, Mapping):
        evidence = {}
    if evidence.get("geometry_method") != "hash_pinned_exact_solid_section":
        errors.append("exact_root_section_method_missing")
    for key in (
        "minimum_root_ligament_width_mm",
        "root_thickness_mm",
        "minimum_section_modulus_mm3",
        "maximum_bending_stress_mpa",
        "allowable_bending_stress_mpa",
        "required_safety_factor",
    ):
        if not _positive(evidence.get(key)):
            errors.append(f"{key}_missing")
    if _positive(evidence.get("required_safety_factor")) and float(
        evidence["required_safety_factor"]
    ) <= 1.0:
        errors.append("root_required_safety_factor_not_above_one")
    if not _valid_sha256(evidence.get("load_case_sha256")):
        errors.append("root_load_case_sha256_missing")

    achieved_sf = None
    stress = evidence.get("maximum_bending_stress_mpa")
    allowable = evidence.get("allowable_bending_stress_mpa")
    required_sf = evidence.get("required_safety_factor")
    if _positive(stress) and _positive(allowable):
        achieved_sf = float(allowable) / float(stress)
        if _positive(required_sf) and achieved_sf < float(required_sf):
            errors.append("root_strength_safety_factor_failed")
    return {
        "required_semantics": "exact_root_section_and_adverse_load_case",
        "recomputed_safety_factor": achieved_sf,
        "qualified": not errors,
        "passed": not errors,
        "errors": sorted(set(errors)),
    }


def _route_torque_evidence(
    evidence: Any,
    *,
    source_sha256: str,
    force_requirements: Mapping[str, float | None],
) -> dict[str, Any]:
    errors = _qualified_provenance(evidence, source_sha256=source_sha256)
    if not isinstance(evidence, Mapping):
        evidence = {}
    for key in ("model_sha256", "payload_sha256", "load_case_sha256"):
        if not _valid_sha256(evidence.get(key)):
            errors.append(f"{key}_missing")
    directions = evidence.get("directions")
    if not isinstance(directions, Mapping):
        directions = {}
        errors.append("route_directions_missing")
    missing = sorted(set(REQUIRED_ROUTE_DIRECTIONS) - set(directions))
    extra = sorted(set(directions) - set(REQUIRED_ROUTE_DIRECTIONS))
    if missing:
        errors.append("missing_route_directions:" + ",".join(missing))
    if extra:
        errors.append("unexpected_route_directions:" + ",".join(extra))

    records: dict[str, Any] = {}
    for name in REQUIRED_ROUTE_DIRECTIONS:
        item = directions.get(name, {}) if isinstance(directions, Mapping) else {}
        item_errors: list[str] = []
        if not isinstance(item, Mapping):
            item = {}
        for flag in (
            "swept_every_route_state",
            "uses_actual_slider_tab_contact_point",
            "includes_payload_gravity",
            "includes_commanded_acceleration",
        ):
            if item.get(flag) is not True:
                item_errors.append(f"{flag}_missing")
        if not _valid_sha256(item.get("route_samples_sha256")):
            item_errors.append("route_samples_sha256_missing")
        required_force = force_requirements.get(name)
        applied_force = item.get("applied_cam_force_bound_n")
        if required_force is None:
            item_errors.append("finite_required_cam_force_unavailable")
        elif not _finite(applied_force) or float(applied_force) + 1.0e-12 < float(
            required_force
        ):
            item_errors.append("applied_cam_force_does_not_cover_authority")

        demand = item.get("joint_torque_demand_nm")
        limit = item.get("joint_torque_limit_nm")
        margin = None
        if (
            not isinstance(demand, list)
            or not isinstance(limit, list)
            or not demand
            or len(demand) != len(limit)
            or not all(_finite(value) for value in demand + limit)
            or not all(float(value) > 0.0 for value in limit)
        ):
            item_errors.append("joint_torque_vectors_invalid")
        else:
            margin = min(
                float(bound) - abs(float(value))
                for value, bound in zip(demand, limit)
            )
            if margin <= 0.0:
                item_errors.append("joint_torque_margin_nonpositive")
        records[name] = {
            "required_cam_force_n": required_force,
            "applied_cam_force_bound_n": (
                float(applied_force) if _finite(applied_force) else None
            ),
            "recomputed_minimum_joint_torque_margin_nm": margin,
            "passed": not item_errors,
            "errors": item_errors,
        }
        errors.extend(f"{name}:{error}" for error in item_errors)
    return {
        "required_directions": list(REQUIRED_ROUTE_DIRECTIONS),
        "records": records,
        "qualified": not errors,
        "passed": not errors,
        "errors": sorted(set(errors)),
    }


def build_report(
    *,
    evidence: Mapping[str, Any] | None = None,
    source_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical report from source plus external qualification evidence."""

    evidence = copy.deepcopy(dict(evidence or {}))
    contract = copy.deepcopy(
        dict(source_contract or CAD.positive_lock_cam_contract())
    )
    source_sha256 = _sha256(CAD_GENERATOR_PATH)
    slopes = _source_slopes(contract)
    slope_qualification = evidence.get("slope_tolerance_qualification")
    capture_interval = _slope_interval(
        name="capture",
        run_mm=float(slopes["capture_run_mm"]),
        rise_mm=float(slopes["capture_rise_mm"]),
        qualification=slope_qualification,
        source_sha256=source_sha256,
    )
    return_interval = _slope_interval(
        name="return",
        run_mm=float(slopes["return_run_mm"]),
        rise_mm=float(slopes["return_rise_mm"]),
        qualification=slope_qualification,
        source_sha256=source_sha256,
    )
    friction = _friction_interval(
        evidence.get("friction_qualification"),
        source_sha256=source_sha256,
    )
    timing = _opening_timing(
        contract,
        evidence.get("timing_tolerance_qualification"),
        source_sha256=source_sha256,
    )
    mechanics = _mechanics(
        contract, capture_interval, return_interval, friction
    )
    contact = _finite_contact_evidence(
        evidence.get("finite_contact_patch_evidence"),
        source_sha256=source_sha256,
        source_full_face_area_mm2=float(
            contract["quasistatic_load_envelope"]["contact_face_area_mm2"]
        ),
        mechanics=mechanics,
    )
    root = _root_strength_evidence(
        evidence.get("root_strength_evidence"),
        source_sha256=source_sha256,
    )
    route = _route_torque_evidence(
        evidence.get("route_torque_evidence"),
        source_sha256=source_sha256,
        force_requirements=mechanics["route_force_requirements_n"],
    )

    checks = {
        "source_slope_fields_recomputed_consistently": bool(
            slopes["published_values_consistent"]
        ),
        "capture_slope_interval_qualified": capture_interval["qualified"],
        "return_slope_interval_qualified": return_interval["qualified"],
        "friction_interval_qualified": friction["qualified"],
        "opening_timing_certified_before_head_entry": timing["passed"],
        "spring_force_recomputed_consistently": mechanics["spring"][
            "published_maximum_force_consistent"
        ],
        "capture_denominator_positive": mechanics["capture"]["jam_free"],
        "passive_return_k_exceeds_adverse_mu": mechanics[
            "passive_return_negative_y"
        ]["jam_free"],
        "reverse_insertion_denominator_positive": mechanics[
            "reverse_insertion_positive_y"
        ]["jam_free"],
        "finite_contact_patch_pressure_qualified": contact["passed"],
        "root_strength_qualified": root["passed"],
        "full_route_torque_provenance_qualified": route["passed"],
    }
    blocker_names = {
        "source_slope_fields_recomputed_consistently": "source_slope_contract_inconsistent",
        "capture_slope_interval_qualified": "capture_slope_tolerance_unqualified",
        "return_slope_interval_qualified": "return_slope_tolerance_unqualified",
        "friction_interval_qualified": "friction_interval_unqualified",
        "opening_timing_certified_before_head_entry": "opening_timing_tolerance_unqualified",
        "spring_force_recomputed_consistently": "spring_force_contract_inconsistent",
        "capture_denominator_positive": "capture_friction_denominator_nonpositive",
        "passive_return_k_exceeds_adverse_mu": "passive_return_self_lock_risk",
        "reverse_insertion_denominator_positive": "reverse_insertion_jam_risk",
        "finite_contact_patch_pressure_qualified": "finite_contact_patch_evidence_missing_or_invalid",
        "root_strength_qualified": "root_strength_evidence_missing_or_invalid",
        "full_route_torque_provenance_qualified": "route_torque_provenance_missing_or_invalid",
    }
    blockers = [
        blocker_names[name] for name, passed in checks.items() if not passed
    ]
    release_ready = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "release_ready": release_ready,
        "passed": release_ready,
        "blockers": blockers,
        "authorities": {
            "cad_source": {
                "path": str(CAD_GENERATOR_PATH.relative_to(QUICK_CHANGE_DIR.parents[1])),
                "sha256": source_sha256,
            },
            "validator": {
                "path": str(Path(__file__).relative_to(QUICK_CHANGE_DIR.parents[1])),
                "sha256": _sha256(Path(__file__)),
            },
            "source_contract_sha256": _canonical_sha256(contract),
            "evidence_sha256": _canonical_sha256(evidence),
        },
        "screening_mu_interval": {
            "interval": list(SCREENING_MU_INTERVAL),
            "role": "screening_only_not_material_pair_qualification",
        },
        "source_nominal_slopes": slopes,
        "slope_intervals": {
            "capture": capture_interval,
            "return": return_interval,
        },
        "friction": friction,
        "opening_timing": timing,
        "mechanics": mechanics,
        "finite_contact_patch": contact,
        "root_strength": root,
        "route_torque": route,
        "evidence_requirements": {
            "slope_intervals": "nonzero run/rise bounds with source and report hashes",
            "friction": "qualified material pair and environment mu interval",
            "timing": "open/head tolerances including slope interval propagation",
            "contact": "finite patch lower bound; line/full-face mean pressure forbidden",
            "root": "hash-pinned exact section and adverse bending load case",
            "route_torque": {
                "directions": list(REQUIRED_ROUTE_DIRECTIONS),
                "requirements": "actual tab contact point, every route state, gravity and commanded acceleration",
            },
        },
        "checks": checks,
    }


def validate_report(
    report: Any,
    *,
    evidence: Mapping[str, Any] | None = None,
    source_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """Reject any report that is not the canonical fresh recomputation."""

    if not isinstance(report, Mapping):
        return ["report_not_object"]
    expected = build_report(evidence=evidence, source_contract=source_contract)
    errors: list[str] = []
    if dict(report) != expected:
        errors.append("report_recomputation_mismatch")
    if report.get("passed") is not report.get("release_ready"):
        errors.append("passed_release_ready_mismatch")
    if report.get("release_ready") is True and report.get("blockers"):
        errors.append("release_ready_with_blockers")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print the current fail-closed report as one canonical JSON line",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report()
    if validate_report(report):
        raise RuntimeError("fresh positive-lock cam load report did not self-validate")
    if args.compact:
        print(_canonical_bytes(report).decode("utf-8"))
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
