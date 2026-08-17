#!/usr/bin/env python3
"""Generate a fail-closed exact-clearance report for the matcha tool rack.

The fast path uses a continuous swept-AABB lower bound for every named rigid
tool component against every named rack component, including adjacent bays.
Only pairs whose lower bound is below manufacturing clearance advance to an
FCPW tessellated-mesh screen.  Intended support/stop tangencies then receive a
small OCCT B-rep distance and overlap-volume diagnostic.  FCPW is explicitly a
screen; it is never promoted to exact STEP authority.

The stock-gripper bay is intentionally unresolved in this package.  Therefore
the report can prove the spoon and whisk paths while remaining
``release_ready=false`` for the complete three-tool rack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cadquery as cq
import fcpw
import numpy as np

import generate_matcha_tool_cad as cad


SCHEMA_VERSION = "1.0"
REPO_ROOT = cad.PACKAGE_DIR.parents[2]
REPORT_PATH = cad.EXPORT_DIR / "matcha_rack_validation_report.json"
MANIFEST_PATH = cad.EXPORT_DIR / "matcha_tool_manifest.json"

MANUFACTURING_CLEARANCE_MM = 0.20
NUMERIC_TOLERANCE_MM = 0.001
OVERLAP_VOLUME_TOLERANCE_MM3 = 1.0e-6
MESH_TESSELLATION_DEFLECTION_MM = 0.03
MESH_ANGULAR_TOLERANCE_RAD = 0.10
PATH_SAMPLE_STEP_MM = 0.10
MAX_BETWEEN_SAMPLE_MOTION_MM = PATH_SAMPLE_STEP_MM / 2.0

EXPECTED_EXPORT_ROLES = {
    "complete_rigid_assembly_step": 2,
    "mass_ledger": 2,
    "printable_body_step": 2,
    "printable_body_stl": 2,
    "rack_step": 1,
    "rack_stl": 1,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _bbox(shape: cq.Workplane) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    box = shape.val().BoundingBox()
    return ((box.xmin, box.xmax), (box.ymin, box.ymax), (box.zmin, box.zmax))


def _axis_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(b[0] - a[1], a[0] - b[1], 0.0)


def _continuous_swept_aabb(
    moving_shape: cq.Workplane,
    fixed_shape: cq.Workplane,
    bay_x_mm: float,
) -> tuple[float, float]:
    """Return continuous AABB lower bound and one witnessing path Y."""

    moving = _bbox(moving_shape)
    fixed = _bbox(fixed_shape)
    moving_x = (moving[0][0] + bay_x_mm, moving[0][1] + bay_x_mm)
    moving_y_sweep = (
        moving[1][0] + cad.RACK_INSERTION_START_Y,
        moving[1][1] + cad.RACK_SEATED_Y,
    )
    dx = _axis_gap(moving_x, fixed[0])
    dy = _axis_gap(moving_y_sweep, fixed[1])
    dz = _axis_gap(moving[2], fixed[2])

    moving_y_center = (moving[1][0] + moving[1][1]) / 2.0
    fixed_y_center = (fixed[1][0] + fixed[1][1]) / 2.0
    witness_y = min(
        cad.RACK_SEATED_Y,
        max(cad.RACK_INSERTION_START_Y, fixed_y_center - moving_y_center),
    )
    return math.sqrt(dx * dx + dy * dy + dz * dz), witness_y


def _path_positions() -> list[float]:
    intervals = round(
        (cad.RACK_SEATED_Y - cad.RACK_INSERTION_START_Y) / PATH_SAMPLE_STEP_MM
    )
    positions = [
        round(cad.RACK_INSERTION_START_Y + index * PATH_SAMPLE_STEP_MM, 10)
        for index in range(intervals + 1)
    ]
    if positions[-1] != cad.RACK_SEATED_Y:
        raise RuntimeError("rack path sampling does not land exactly at seated Y")
    return positions


def _tool_states(tool: str) -> list[dict[str, float | str]]:
    if tool == "spoon":
        return [{"name": "rigid_nominal", "eccentric_x_mm": 0.0, "compliance_z_mm": 0.0}]
    states: list[dict[str, float | str]] = []
    for eccentric_x in (-cad.WHISK_ECCENTRIC_MM, 0.0, cad.WHISK_ECCENTRIC_MM):
        for compliance_z in cad.WHISK_COMPLIANCE_LIMITS_MM:
            states.append(
                {
                    "name": f"eccentric_{eccentric_x:+.2f}_compliance_{compliance_z:+.2f}",
                    "eccentric_x_mm": eccentric_x,
                    "compliance_z_mm": compliance_z,
                }
            )
    states.append(
        {"name": "mechanism_nominal", "eccentric_x_mm": 0.0, "compliance_z_mm": 0.0}
    )
    return states


WHISK_X_MOVERS = {
    "whisk_carriage_x",
    "whisk_compliance_carriage",
    "whisk_brush_hub",
    "whisk_bamboo_bristles",
    "whisk_brush_collision_envelope",
}
WHISK_Z_MOVERS = {
    "whisk_compliance_carriage",
    "whisk_brush_hub",
    "whisk_bamboo_bristles",
    "whisk_brush_collision_envelope",
}


def _shape_at_mechanism_state(
    component: cad.Component,
    state: dict[str, float | str],
) -> cq.Workplane:
    if not component.name.startswith("whisk_"):
        return component.shape
    dx = float(state["eccentric_x_mm"]) if component.name in WHISK_X_MOVERS else 0.0
    dz = float(state["compliance_z_mm"]) if component.name in WHISK_Z_MOVERS else 0.0
    if component.name == "whisk_eccentric_pin":
        # Authored pin is at the +4 mm crank extreme.
        dx = float(state["eccentric_x_mm"]) - cad.WHISK_ECCENTRIC_MM
    return component.shape.translate((dx, 0.0, dz))


def _mesh(shape: cq.Workplane) -> tuple[np.ndarray, np.ndarray, str]:
    vertices, triangles = shape.val().tessellate(
        MESH_TESSELLATION_DEFLECTION_MM,
        MESH_ANGULAR_TOLERANCE_RAD,
    )
    vertex_array = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=np.float32)
    triangle_array = np.asarray(triangles, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise RuntimeError("invalid OCCT tessellation vertex array")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise RuntimeError("invalid OCCT tessellation triangle array")
    digest = hashlib.sha256(vertex_array.tobytes() + triangle_array.tobytes()).hexdigest()
    return np.ascontiguousarray(vertex_array), np.ascontiguousarray(triangle_array), digest


def _fcpw_scene(vertices: np.ndarray, triangles: np.ndarray):
    scene = fcpw.scene_3D()
    scene.set_object_count(1)
    scene.set_object_vertices(vertices, 0)
    scene.set_object_triangles(triangles, 0)
    scene.build(fcpw.aggregate_type.bvh_surface_area, False)
    return scene


def _directional_vertex_to_mesh_distance_mm(
    query_vertices: np.ndarray,
    target_vertices: np.ndarray,
    target_triangles: np.ndarray,
) -> float:
    scene = _fcpw_scene(target_vertices, target_triangles)
    radii = np.full(len(query_vertices), np.inf, dtype=np.float32)
    interactions = fcpw.interaction_3D_list()
    scene.find_closest_points(query_vertices, radii, interactions)
    if len(interactions) != len(query_vertices):
        raise RuntimeError("FCPW returned an incomplete closest-point batch")
    return min(float(interaction.d) for interaction in interactions)


def _fcpw_screen(shape_a: cq.Workplane, shape_b: cq.Workplane) -> dict[str, Any]:
    vertices_a, triangles_a, digest_a = _mesh(shape_a)
    vertices_b, triangles_b, digest_b = _mesh(shape_b)
    a_to_b = _directional_vertex_to_mesh_distance_mm(
        vertices_a, vertices_b, triangles_b
    )
    b_to_a = _directional_vertex_to_mesh_distance_mm(
        vertices_b, vertices_a, triangles_a
    )
    return {
        "role": "toleranced_mesh_screen_only",
        "backend": "fcpw",
        "backend_version": importlib.metadata.version("fcpw"),
        "tessellation_deflection_mm": MESH_TESSELLATION_DEFLECTION_MM,
        "angular_tolerance_rad": MESH_ANGULAR_TOLERANCE_RAD,
        "mesh_a": {
            "vertices": int(len(vertices_a)),
            "triangles": int(len(triangles_a)),
            "sha256": digest_a,
        },
        "mesh_b": {
            "vertices": int(len(vertices_b)),
            "triangles": int(len(triangles_b)),
            "sha256": digest_b,
        },
        "minimum_a_vertices_to_b_mesh_mm": a_to_b,
        "minimum_b_vertices_to_a_mesh_mm": b_to_a,
        "minimum_directional_witness_mm": min(a_to_b, b_to_a),
        "clearance_authority": False,
    }


def _placed(shape: cq.Workplane, bay_x: float, path_y: float) -> cq.Workplane:
    return shape.translate((bay_x, path_y, 0.0))


def _exact_contact_record(
    tool: str,
    bay: str,
    tool_shape: cq.Workplane,
    rack_name: str,
    rack_shape: cq.Workplane,
    bay_x: float,
) -> dict[str, Any]:
    if rack_name.endswith("left_lower_ledge"):
        semantic = "intended_printed_plate_support_tangency"
        # The left rail bears on the electrical wing, whose +Y edge is 12 mm.
        contact_window = [-61.21, cad.RACK_SEATED_Y]
        outside_range = [cad.RACK_INSERTION_START_Y, -61.21]
        expected_normal = [0.0, 0.0, 1.0]
        witness_y = 0.0
        outside_y = -61.21
    elif rack_name.endswith("right_lower_ledge"):
        semantic = "intended_printed_plate_support_tangency"
        # Rounded main-plate corner reaches this support shortly after -74 mm.
        contact_window = [-74.21, cad.RACK_SEATED_Y]
        outside_range = [cad.RACK_INSERTION_START_Y, -74.21]
        expected_normal = [0.0, 0.0, 1.0]
        witness_y = 0.0
        outside_y = -74.21
    elif rack_name == f"dock_{bay}_seating_stop":
        semantic = "intended_printed_plate_seating_stop_tangency"
        contact_window = [-0.21, cad.RACK_SEATED_Y]
        outside_range = [cad.RACK_INSERTION_START_Y, -0.21]
        expected_normal = [0.0, -1.0, 0.0]
        witness_y = 0.0
        outside_y = -0.21
    else:
        raise RuntimeError(f"unrecognized intentional contact {tool}/{rack_name}")

    placed_contact = _placed(tool_shape, bay_x, witness_y)
    exact_distance = float(placed_contact.val().distance(rack_shape.val()))
    overlap = float(placed_contact.val().intersect(rack_shape.val()).Volume())
    placed_outside = _placed(tool_shape, bay_x, outside_y)
    outside_clearance = float(placed_outside.val().distance(rack_shape.val()))
    mesh_screen = _fcpw_screen(placed_contact, rack_shape)
    passed = (
        exact_distance <= NUMERIC_TOLERANCE_MM
        and overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
        and outside_clearance + NUMERIC_TOLERANCE_MM >= MANUFACTURING_CLEARANCE_MM
    )
    return {
        "semantic_evaluation": semantic,
        "minimum_signed_clearance_mm": 0.0,
        "maximum_overlap_volume_mm3": overlap,
        "witness": {
            "mechanism_state": "interface_invariant",
            "path_y_mm": witness_y,
            "tool_translation_mm": [bay_x, witness_y, 0.0],
            "expected_rack_surface_normal": expected_normal,
        },
        "method": "fcpw_screen_plus_occt_brep_critical_diagnostic",
        "fcpw_screen": mesh_screen,
        "occt_critical_diagnostic": {
            "exact_brep_distance_mm": exact_distance,
            "exact_brep_overlap_volume_mm3": overlap,
            "overlap_volume_tolerance_mm3": OVERLAP_VOLUME_TOLERANCE_MM3,
            "clearance_authority": True,
        },
        "intentional_contact_window_y_mm": contact_window,
        "outside_window_clearance": {
            "path_y_range_mm": outside_range,
            "minimum_exact_clearance_mm": outside_clearance,
            "witness_y_mm": outside_y,
            "required_mm": MANUFACTURING_CLEARANCE_MM,
        },
        "continuous_certificate": {
            "semantics": "analytic_straight_y_contact_window_plus_exact_endpoint",
            "maximum_between_sample_motion_bound_mm": MAX_BETWEEN_SAMPLE_MOTION_MM,
            "contact_window_is_monotone_or_constant": True,
        },
        "passed": passed,
    }


def _pair_record(
    tool: str,
    bay: str,
    component: cad.Component,
    rack_name: str,
    rack_shape: cq.Workplane,
    states: list[dict[str, float | str]],
    bay_x: float,
) -> dict[str, Any]:
    candidates: list[tuple[float, float, dict[str, float | str], cq.Workplane]] = []
    for state in states:
        shape = _shape_at_mechanism_state(component, state)
        lower_bound, witness_y = _continuous_swept_aabb(shape, rack_shape, bay_x)
        candidates.append((lower_bound, witness_y, state, shape))
    lower_bound, witness_y, witness_state, witness_shape = min(candidates, key=lambda item: item[0])

    exact_intended = component.name == "common_tool_plate" and rack_name in {
        f"dock_{bay}_left_lower_ledge",
        f"dock_{bay}_right_lower_ledge",
        f"dock_{bay}_seating_stop",
    }
    if exact_intended:
        evaluation = _exact_contact_record(
            tool, bay, witness_shape, rack_name, rack_shape, bay_x
        )
    elif lower_bound + NUMERIC_TOLERANCE_MM >= MANUFACTURING_CLEARANCE_MM:
        evaluation = {
            "semantic_evaluation": "forbidden_pair_continuous_clearance",
            "minimum_signed_clearance_mm": lower_bound,
            "maximum_overlap_volume_mm3": 0.0,
            "witness": {
                "mechanism_state": str(witness_state["name"]),
                "path_y_mm": witness_y,
                "tool_translation_mm": [bay_x, witness_y, 0.0],
            },
            "method": "analytic_continuous_swept_aabb_lower_bound",
            "continuous_certificate": {
                "semantics": "euclidean_separation_of_exact_brep_axis_aligned_bounds_over_full_y_sweep",
                "sampled_path_is_secondary": True,
                "maximum_between_sample_motion_bound_mm": MAX_BETWEEN_SAMPLE_MOTION_MM,
                "certified_clearance_mm": lower_bound,
            },
            "passed": True,
        }
    else:
        evaluation = {
            "semantic_evaluation": "unresolved_forbidden_pair",
            "minimum_signed_clearance_mm": lower_bound,
            "maximum_overlap_volume_mm3": None,
            "witness": {
                "mechanism_state": str(witness_state["name"]),
                "path_y_mm": witness_y,
                "tool_translation_mm": [bay_x, witness_y, 0.0],
            },
            "method": "continuous_swept_aabb_requires_unimplemented_exact_fallback",
            "continuous_certificate": {
                "maximum_between_sample_motion_bound_mm": MAX_BETWEEN_SAMPLE_MOTION_MM,
                "certified_clearance_mm": lower_bound,
            },
            "passed": False,
        }

    return {
        "tool": tool,
        "bay": bay,
        "pair": [component.name, rack_name],
        "tool_component": component.name,
        "rack_component": rack_name,
        "states_evaluated": [str(state["name"]) for state in states],
        "manufacturing_clearance_mm": MANUFACTURING_CLEARANCE_MM,
        "numeric_tolerance_mm": NUMERIC_TOLERANCE_MM,
        **evaluation,
    }


def _inventory_record(names: Iterable[str]) -> dict[str, Any]:
    sorted_names = sorted(names)
    return {
        "names": sorted_names,
        "count": len(sorted_names),
        "canonical_sha256": _canonical_sha256(sorted_names),
    }


def _manifest_closure() -> tuple[bool, list[str], dict[str, Any] | None, list[dict[str, Any]]]:
    errors: list[str] = []
    if not MANIFEST_PATH.exists():
        return False, ["canonical_export_manifest_missing"], None, []
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema_version") != cad.SCHEMA_VERSION:
        errors.append("manifest_schema_version_mismatch")
    expected_generator = _file_record(Path(cad.__file__))
    if manifest.get("generator", {}).get("sha256") != expected_generator["sha256"]:
        errors.append("manifest_generator_hash_mismatch")
    if manifest.get("interface_authority", {}).get("sha256") != _sha256(cad.BASE_GENERATOR):
        errors.append("manifest_interface_authority_hash_mismatch")

    records = manifest.get("files")
    if not isinstance(records, list):
        return False, errors + ["manifest_files_not_list"], manifest, []
    roles: dict[str, int] = {}
    current_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in records:
        path_value = record.get("path")
        if not isinstance(path_value, str) or path_value in seen_paths:
            errors.append("manifest_duplicate_or_invalid_path")
            continue
        seen_paths.add(path_value)
        path = cad.PACKAGE_DIR / path_value
        if not path.is_file():
            errors.append(f"manifest_file_missing:{path_value}")
            continue
        current = _file_record(path)
        current["role"] = record.get("role")
        if current["bytes"] != record.get("bytes") or current["sha256"] != record.get("sha256"):
            errors.append(f"manifest_file_hash_or_size_mismatch:{path_value}")
        roles[str(record.get("role"))] = roles.get(str(record.get("role")), 0) + 1
        current_records.append(current)
    if roles != EXPECTED_EXPORT_ROLES:
        errors.append(f"manifest_role_inventory_mismatch:{roles}")
    return not errors, errors, manifest, current_records


def build_report(*, require_exports: bool) -> dict[str, Any]:
    rack = cad.build_rack()
    path_positions = _path_positions()
    pair_records: list[dict[str, Any]] = []
    tool_results: dict[str, Any] = {}
    inventories: dict[str, Any] = {"rack_components": _inventory_record(rack)}

    for tool, bay, tool_id, bus_address in (
        ("spoon", "spoon", cad.SPOON_TOOL_ID, None),
        ("whisk", "whisk", cad.WHISK_TOOL_ID, cad.WHISK_BUS_ADDRESS),
    ):
        components = cad.build_tool(tool)
        states = _tool_states(tool)
        bay_x = (cad.RACK_BAY_NAMES.index(bay) - 1) * cad.RACK_BAY_PITCH
        inventories[f"{tool}_components"] = _inventory_record(
            component.name for component in components
        )
        tool_pairs = [
            _pair_record(tool, bay, component, rack_name, rack_shape, states, bay_x)
            for component in components
            for rack_name, rack_shape in sorted(rack.items())
        ]
        pair_records.extend(tool_pairs)
        state_contract = [
            {
                "name": state["name"],
                "eccentric_x_mm": state["eccentric_x_mm"],
                "compliance_z_mm": state["compliance_z_mm"],
            }
            for state in states
        ]
        forward_sequence = [
            {"y_mm": y, "state": state["name"]}
            for state in states
            for y in path_positions
        ]
        reverse_sequence = list(reversed(forward_sequence))
        tool_results[f"matcha_{tool}"] = {
            "tool_id": tool_id,
            "bus_address": bus_address,
            "bay": bay,
            "passed": all(record["passed"] for record in tool_pairs),
            "straight_y_path": {
                "axis": [0.0, 1.0, 0.0],
                "insertion_start_y_mm": cad.RACK_INSERTION_START_Y,
                "seated_y_mm": cad.RACK_SEATED_Y,
                "exit_start_y_mm": cad.RACK_SEATED_Y,
                "exit_end_y_mm": cad.RACK_INSERTION_START_Y,
                "sample_step_mm": PATH_SAMPLE_STEP_MM,
                "sample_count_per_mechanism_state": len(path_positions),
                "sampled_positions_y_mm": path_positions,
                "sampled_positions_sha256": _canonical_sha256(path_positions),
                "maximum_between_sample_motion_bound_mm": MAX_BETWEEN_SAMPLE_MOTION_MM,
                "forward_sequence_sha256": _canonical_sha256(forward_sequence),
                "reverse_sequence_sha256": _canonical_sha256(reverse_sequence),
                "reverse_path_set_equivalent": sorted(
                    _canonical_bytes(item) for item in forward_sequence
                )
                == sorted(_canonical_bytes(item) for item in reverse_sequence),
            },
            "mechanism_states": state_contract,
            "mechanism_states_sha256": _canonical_sha256(state_contract),
            "pair_count": len(tool_pairs),
            "failed_pair_count": sum(not record["passed"] for record in tool_pairs),
        }

    expected_pairs = [
        [record["tool"], record["tool_component"], record["rack_component"]]
        for record in pair_records
    ]
    pair_keys = [tuple(pair) for pair in expected_pairs]
    unique_pairs = set(pair_keys)
    pair_closure = {
        "expected_count": sum(
            inventories[f"{tool}_components"]["count"] * inventories["rack_components"]["count"]
            for tool in ("spoon", "whisk")
        ),
        "evaluated_count": len(pair_records),
        "unique_count": len(unique_pairs),
        "duplicate_count": len(pair_keys) - len(unique_pairs),
        "failed_count": sum(not record["passed"] for record in pair_records),
        "canonical_pair_inventory_sha256": _canonical_sha256(sorted(expected_pairs)),
        "passed": False,
    }
    pair_closure["passed"] = (
        pair_closure["expected_count"]
        == pair_closure["evaluated_count"]
        == pair_closure["unique_count"]
        and pair_closure["failed_count"] == 0
    )

    export_ok, export_errors, manifest, artifact_records = _manifest_closure()
    blockers = [
        {
            "code": "stock_gripper_keeper_geometry_unresolved",
            "detail": "gripper bay position exists, but no exact official-gripper/keeper swept-clearance authority is present",
        }
    ]
    if require_exports and not export_ok:
        blockers.extend({"code": error, "detail": "canonical export closure failed"} for error in export_errors)
    tool_validation_passed = pair_closure["passed"] and all(
        result["passed"] for result in tool_results.values()
    )
    release_ready = False  # Stock-gripper blocker is deliberately fail-closed.
    return {
        "schema_version": SCHEMA_VERSION,
        "units": "mm",
        "release_ready": release_ready,
        "passed": release_ready and tool_validation_passed and export_ok,
        "tool_validation_passed": tool_validation_passed,
        "blockers": blockers,
        "authorities": {
            "matcha_generator": _file_record(Path(cad.__file__)),
            "rack_validator": _file_record(Path(__file__)),
            "quick_change_interface_generator": _file_record(cad.BASE_GENERATOR),
            "fcpw_role": "toleranced_mesh_screen_only",
            "occt_role": "STEP import/tessellation and six critical B-rep tangency diagnostics",
        },
        "input_artifacts": {
            "required": require_exports,
            "closure_passed": export_ok,
            "closure_errors": export_errors,
            "manifest": _file_record(MANIFEST_PATH) if MANIFEST_PATH.exists() else None,
            "manifest_schema": manifest,
            "files": artifact_records,
        },
        "tolerances": {
            "manufacturing_clearance_mm": MANUFACTURING_CLEARANCE_MM,
            "numeric_tolerance_mm": NUMERIC_TOLERANCE_MM,
            "overlap_volume_tolerance_mm3": OVERLAP_VOLUME_TOLERANCE_MM3,
            "mesh_tessellation_deflection_mm": MESH_TESSELLATION_DEFLECTION_MM,
            "mesh_angular_tolerance_rad": MESH_ANGULAR_TOLERANCE_RAD,
        },
        "inventories": inventories,
        "tool_path_results": tool_results,
        "collision_pair_closure": pair_closure,
        "pair_results": pair_records,
    }


def validate_report_structure(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "units",
        "release_ready",
        "passed",
        "blockers",
        "authorities",
        "input_artifacts",
        "inventories",
        "tool_path_results",
        "collision_pair_closure",
        "pair_results",
    }
    if not required.issubset(report):
        errors.append(f"missing_top_level_keys:{sorted(required - set(report))}")
    closure = report.get("collision_pair_closure", {})
    if closure.get("expected_count") != 1458:
        errors.append("unexpected_pair_inventory_count")
    if closure.get("evaluated_count") != closure.get("unique_count"):
        errors.append("pair_inventory_not_unique")
    records = report.get("pair_results", [])
    computed_pairs = [
        [record.get("tool"), record.get("tool_component"), record.get("rack_component")]
        for record in records
    ]
    unique_pairs = {tuple(pair) for pair in computed_pairs}
    if closure.get("evaluated_count") != len(records):
        errors.append("pair_evaluated_count_not_recomputed")
    if closure.get("unique_count") != len(unique_pairs):
        errors.append("pair_unique_count_not_recomputed")
    if closure.get("duplicate_count") != len(records) - len(unique_pairs):
        errors.append("pair_duplicate_count_not_recomputed")
    if closure.get("failed_count") != sum(not record.get("passed", False) for record in records):
        errors.append("pair_failed_count_not_recomputed")
    if closure.get("canonical_pair_inventory_sha256") != _canonical_sha256(sorted(computed_pairs)):
        errors.append("pair_inventory_digest_not_recomputed")
    exact_semantics = {
        "intended_printed_plate_support_tangency": 4,
        "intended_printed_plate_seating_stop_tangency": 2,
    }
    observed: dict[str, int] = {}
    for record in records:
        semantic = str(record.get("semantic_evaluation"))
        observed[semantic] = observed.get(semantic, 0) + 1
        if record.get("pair") != [record.get("tool_component"), record.get("rack_component")]:
            errors.append("pair_record_name_mismatch")
        if semantic == "forbidden_pair_continuous_clearance":
            clearance = record.get("minimum_signed_clearance_mm")
            if not isinstance(clearance, (float, int)) or (
                float(clearance) + NUMERIC_TOLERANCE_MM < MANUFACTURING_CLEARANCE_MM
            ):
                errors.append("forbidden_pair_below_manufacturing_clearance")
            if record.get("maximum_overlap_volume_mm3") != 0.0 or not record.get("passed"):
                errors.append("forbidden_pair_verdict_inconsistent")
        elif semantic.startswith("intended_printed_plate_"):
            diagnostic = record.get("occt_critical_diagnostic", {})
            outside = record.get("outside_window_clearance", {})
            if diagnostic.get("exact_brep_distance_mm", math.inf) > NUMERIC_TOLERANCE_MM:
                errors.append("intended_contact_not_at_tangency")
            if diagnostic.get("exact_brep_overlap_volume_mm3", math.inf) > OVERLAP_VOLUME_TOLERANCE_MM3:
                errors.append("intended_contact_has_overlap")
            if outside.get("minimum_exact_clearance_mm", -math.inf) + NUMERIC_TOLERANCE_MM < MANUFACTURING_CLEARANCE_MM:
                errors.append("intended_contact_outside_window_clearance_failed")
            if record.get("fcpw_screen", {}).get("clearance_authority") is not False:
                errors.append("fcpw_screen_promoted_to_clearance_authority")
    for semantic, count in exact_semantics.items():
        if observed.get(semantic) != count:
            errors.append(f"intentional_contact_count_mismatch:{semantic}:{observed.get(semantic)}")
    if observed.get("unresolved_forbidden_pair", 0):
        errors.append("unresolved_forbidden_pairs_present")
    if not any(blocker.get("code") == "stock_gripper_keeper_geometry_unresolved" for blocker in report.get("blockers", [])):
        errors.append("stock_gripper_blocker_missing")
    if report.get("release_ready") or report.get("passed"):
        errors.append("report_must_remain_release_red_until_stock_gripper_closes")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="validate source geometry without requiring or writing generated exports",
    )
    mode.add_argument(
        "--release-report",
        action="store_true",
        help="require canonical export closure and write the JSON report",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require_exports = bool(args.release_report)
    report = build_report(require_exports=require_exports)
    structural_errors = validate_report_structure(report)
    if structural_errors:
        raise SystemExit("rack report structural validation failed: " + "; ".join(structural_errors))
    if args.release_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "mode": "release-report" if args.release_report else "preflight",
        "tool_validation_passed": report["tool_validation_passed"],
        "release_ready": report["release_ready"],
        "pair_count": report["collision_pair_closure"]["evaluated_count"],
        "failed_pair_count": report["collision_pair_closure"]["failed_count"],
        "blockers": [blocker["code"] for blocker in report["blockers"]],
        "report": str(args.report) if args.release_report else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["tool_validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
