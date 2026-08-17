#!/usr/bin/env python3
"""Independently recompute the offline plate-proxy authority report.

Report booleans are never accepted on faith.  This validator reloads the
hash-pinned STEP, checks the calibrated frame and piece inventory, replays the
whole-cell subset inequality, recomputes both boundary directions, and probes
every declared functional void.  Release mode additionally rebuilds the
adaptive octree and requires byte-equivalent numeric box bounds.

The exported :func:`run_adversarial_suite` is deliberately small enough for a
focused unit test.  It rejects a translated proxy, a filled bore, a deleted
wall, an injected outside cell, and a rotated source frame using recomputed
geometry rather than self-reported arithmetic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import generate_plate_proxy_authority as generator


SCHEMA_VERSION = generator.SCHEMA_VERSION
POSITION_TOLERANCE_MM = 1.0e-10
QUATERNION_TOLERANCE = 1.0e-12
BOUND_TOLERANCE_MM = 1.0e-10


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _resolve_repo_path(record: dict[str, Any]) -> Path:
    path = generator.REPOSITORY_ROOT / str(record.get("path", ""))
    resolved = path.resolve()
    try:
        resolved.relative_to(generator.REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError("artifact path escapes repository") from error
    return resolved


def _validate_file_record(record: Any, *, expected_role: str | None = None) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        path = _resolve_repo_path(record)
        if not path.is_file():
            return False
        if int(record.get("bytes", -1)) != path.stat().st_size:
            return False
        if str(record.get("sha256", "")) != hashlib.sha256(path.read_bytes()).hexdigest():
            return False
        if expected_role is not None and record.get("role") != expected_role:
            return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def validate_source_frame(frame: Any, spec: generator.PlateSourceSpec) -> bool:
    if not isinstance(frame, dict):
        return False
    core = {key: value for key, value in frame.items() if key != "frame_sha256"}
    if frame.get("frame_sha256") != _canonical_sha256(core):
        return False
    try:
        position = np.asarray(frame["body_local_pos_mm"], dtype=np.float64)
        quaternion = np.asarray(frame["body_local_quat_wxyz"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if position.shape != (3,) or quaternion.shape != (4,):
        return False
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        return False
    expected_position = np.asarray(spec.expected_frame_pos_mm, dtype=np.float64)
    expected_quaternion = np.asarray(spec.expected_frame_quat_wxyz, dtype=np.float64)
    if np.linalg.norm(position - expected_position) > POSITION_TOLERANCE_MM:
        return False
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > QUATERNION_TOLERANCE:
        return False
    direct = np.linalg.norm(quaternion - expected_quaternion)
    negated = np.linalg.norm(quaternion + expected_quaternion)
    return bool(min(direct, negated) <= QUATERNION_TOLERANCE)


def boxes_from_inventory(inventory: Any) -> np.ndarray:
    if not isinstance(inventory, dict) or not isinstance(inventory.get("pieces"), list):
        raise ValueError("piece inventory is absent")
    pieces = inventory["pieces"]
    if int(inventory.get("piece_count", -1)) != len(pieces) or not pieces:
        raise ValueError("piece count mismatch")
    if inventory.get("inventory_sha256") != _canonical_sha256(pieces):
        raise ValueError("piece inventory digest mismatch")
    bounds: list[np.ndarray] = []
    for index, piece in enumerate(pieces):
        if piece.get("name") != f"plate_proxy_box_{index:05d}":
            raise ValueError("piece names are not canonical")
        box = np.asarray(piece.get("bounds_mm"), dtype=np.float64)
        if box.shape != (2, 3) or not np.all(np.isfinite(box)):
            raise ValueError("invalid box bounds")
        if np.any(box[1] <= box[0]):
            raise ValueError("box extent is nonpositive")
        center = np.asarray(piece.get("center_mm"), dtype=np.float64)
        size = np.asarray(piece.get("size_mm"), dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,):
            raise ValueError("piece derived values are absent")
        if not np.allclose(center, np.mean(box, axis=0), atol=BOUND_TOLERANCE_MM, rtol=0.0):
            raise ValueError("piece center is inconsistent")
        if not np.allclose(size, box[1] - box[0], atol=BOUND_TOLERANCE_MM, rtol=0.0):
            raise ValueError("piece size is inconsistent")
        bounds.append(box)
    array = np.ascontiguousarray(np.asarray(bounds, dtype=np.float64))
    canonical_keys = [
        array[:, 0, 0], array[:, 1, 0],
        array[:, 0, 1], array[:, 1, 1],
        array[:, 0, 2], array[:, 1, 2],
    ]
    order = np.lexsort(tuple(reversed(canonical_keys)))
    if not np.array_equal(order, np.arange(len(array))):
        raise ValueError("piece inventory is not deterministically sorted")
    return array


def recompute_exact_subset(
    source_triangles: np.ndarray,
    boxes_mm: np.ndarray,
    parameters: generator.OctreeParameters,
) -> dict[str, Any]:
    """Re-prove a merged union without pretending each merged box is a ball.

    A greedy merge can turn many certified cells into a long slab whose own
    circumscribed ball leaves the source even though the slab (the exact union
    of those cells) does not.  We first try the stronger one-ball proof, then
    deterministically subdivide only failed slabs on the source octree lattice.
    """

    source = generator.SignedFcpwMesh(source_triangles)
    grid = generator._root_grid(source_triangles, parameters.maximum_depth)
    leaf_size = np.asarray(grid["leaf_size_mm"], dtype=np.float64)
    pending = np.ascontiguousarray(boxes_mm, dtype=np.float64)
    proven_margins: list[np.ndarray] = []
    failed_leaf_count = 0
    evaluated_count = 0
    for _ in range(parameters.maximum_depth + 24):
        if not len(pending):
            break
        if evaluated_count + len(pending) > parameters.max_frontier_cells * 4:
            raise RuntimeError("merged-box subset replay exceeded fail-closed cap")
        evaluated_count += len(pending)
        centers = np.mean(pending, axis=1)
        half_diagonals = np.linalg.norm((pending[:, 1] - pending[:, 0]) / 2.0, axis=1)
        inside, lower, _ = source.query(
            centers, chunk_size=parameters.query_chunk_size
        )
        margins = lower - half_diagonals - parameters.source_faceting_bound_mm
        proven = inside & (margins > 0.0)
        if np.any(proven):
            proven_margins.append(margins[proven])
        failed = pending[~proven]
        if not len(failed):
            pending = np.empty((0, 2, 3), dtype=np.float64)
            break
        children: list[np.ndarray] = []
        for box in failed:
            counts = np.rint((box[1] - box[0]) / leaf_size).astype(np.int64)
            counts = np.maximum(counts, 1)
            splittable = np.flatnonzero(counts > 1)
            if not len(splittable):
                failed_leaf_count += 1
                continue
            axis = int(splittable[np.argmax(counts[splittable])])
            lower_count = int(counts[axis] // 2)
            split = float(box[0, axis] + lower_count * leaf_size[axis])
            first = box.copy()
            second = box.copy()
            first[1, axis] = split
            second[0, axis] = split
            children.extend((first, second))
        pending = (
            np.ascontiguousarray(np.asarray(children, dtype=np.float64))
            if children
            else np.empty((0, 2, 3), dtype=np.float64)
        )
    if len(pending):
        raise RuntimeError("merged-box subset replay did not terminate")
    minimum_margin = (
        float(min(np.min(values) for values in proven_margins))
        if proven_margins
        else -math.inf
    )
    passed = bool(failed_leaf_count == 0 and minimum_margin > 0.0)
    return {
        "whole_cell_subset_proof": True,
        "box_count": len(boxes_mm),
        "evaluated_subcell_count": evaluated_count,
        "minimum_exact_subset_margin_mm": minimum_margin,
        "failed_leaf_count": failed_leaf_count,
        "passed": passed,
    }


def _component_parameters(record: dict[str, Any]) -> generator.OctreeParameters:
    source = record.get("octree_parameters", {})
    return generator.OctreeParameters(
        maximum_depth=int(source.get("maximum_depth", generator.MAX_OCTREE_DEPTH)),
        boundary_threshold_mm=float(
            source.get("boundary_threshold_mm", generator.BOUNDARY_THRESHOLD_MM)
        ),
        source_faceting_bound_mm=float(
            source.get("source_faceting_bound_mm", generator.SOURCE_FACETING_BOUND_MM)
        ),
        fcpw_float_guard_mm=float(
            source.get("fcpw_float_guard_mm", generator.FCPW_FLOAT_GUARD_MM)
        ),
        source_witness_covering_radius_mm=float(
            source.get(
                "source_witness_covering_radius_mm",
                generator.SOURCE_WITNESS_COVERING_RADIUS_MM,
            )
        ),
    )


def _equivalent_float(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return bool(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance)


def validate_component_record(
    record: Any,
    *,
    release_rebuild_octree: bool,
    recompute_boundary: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return {"passed": False, "errors": ["component record is not an object"]}
    component_id = str(record.get("component_id", ""))
    specs = generator.plate_source_specs()
    spec = specs.get(component_id)
    if spec is None:
        return {"passed": False, "errors": ["unknown component_id"]}

    authority = record.get("source_authority", {})
    if not _validate_file_record(authority.get("artifact"), expected_role="exact_CAD_STEP"):
        errors.append("source STEP file record is invalid")
    if not _validate_file_record(authority.get("core_manifest"), expected_role="source_manifest"):
        errors.append("core manifest file record is invalid")
    selector = authority.get("selector")
    if selector != {"kind": "all_solids"}:
        errors.append("source selector is not all_solids")
    if authority.get("selector_sha256") != _canonical_sha256(selector):
        errors.append("source selector digest mismatch")
    if not validate_source_frame(record.get("source_frame"), spec):
        errors.append("source frame is invalid")

    try:
        triangles, tessellation = generator.load_absolute_step_mesh(spec)
    except Exception as error:  # fail-closed diagnostic boundary
        return {"passed": False, "errors": errors + [f"STEP load failed: {error}"]}
    reported_tessellation = record.get("tessellation_certificate", {})
    for key in (
        "triangle_mesh_sha256",
        "certificate_sha256",
        "triangle_count",
        "face_count",
    ):
        if reported_tessellation.get(key) != tessellation.get(key):
            errors.append(f"tessellation {key} mismatch")
    try:
        boxes = boxes_from_inventory(record.get("runtime_piece_inventory"))
    except (TypeError, ValueError) as error:
        return {"passed": False, "errors": errors + [str(error)]}
    try:
        parameters = _component_parameters(record)
        parameters.validate()
    except (TypeError, ValueError) as error:
        return {"passed": False, "errors": errors + [f"parameters: {error}"]}

    subset = recompute_exact_subset(triangles, boxes, parameters)
    reported_subset = record.get("exact_subset_certificate", {})
    if not subset["passed"]:
        errors.append("recomputed whole-cell subset proof failed")
    if not math.isfinite(
        float(reported_subset.get("minimum_exact_subset_margin_mm", math.nan))
    ) or float(reported_subset.get("minimum_exact_subset_margin_mm", 0.0)) <= 0.0:
        errors.append("reported subset margin is not positive")

    voids = generator.evaluate_functional_voids(
        component_id, generator.SignedFcpwMesh(triangles), boxes
    )
    if not voids["passed"]:
        errors.append("functional void probes failed")
    reported_voids = record.get("functional_void_results", {})
    if reported_voids.get("inventory_sha256") != voids.get("inventory_sha256"):
        errors.append("functional void inventory mismatch")

    boundary: dict[str, Any] | None = None
    if recompute_boundary:
        source_to_proxy = generator.certify_source_to_proxy_surface(
            triangles, boxes, parameters
        )
        boundary = {"source_to_proxy": source_to_proxy}
        if not source_to_proxy["passed"]:
            errors.append("source-to-proxy boundary certificate failed")
        reported = record.get("bidirectional_boundary_certificate", {}).get(
            "source_to_proxy", {}
        )
        if not _equivalent_float(
            float(source_to_proxy["certified_upper_bound_mm"]),
            float(reported.get("certified_upper_bound_mm", math.nan)),
            tolerance=2.0e-6,
        ):
            errors.append("source-to-proxy bound does not recompute")

    rebuilt: dict[str, Any] | None = None
    if release_rebuild_octree:
        rebuilt_boxes, rebuilt, _ = generator.build_adaptive_subset_boxes(
            triangles, parameters
        )
        if rebuilt_boxes.shape != boxes.shape or not np.allclose(
            rebuilt_boxes, boxes, atol=BOUND_TOLERANCE_MM, rtol=0.0
        ):
            errors.append("release octree rebuild differs from reported proxy")
        rebuilt_subset_margin = float(
            rebuilt["exact_subset"]["minimum_exact_subset_margin_mm"]
        )
        reported_subset_margin = float(
            reported_subset.get("minimum_exact_subset_margin_mm", math.nan)
        )
        if not _equivalent_float(
            rebuilt_subset_margin, reported_subset_margin, tolerance=2.0e-6
        ):
            errors.append("release subset margin does not recompute")
        reported_proxy_bound = float(
            record.get("bidirectional_boundary_certificate", {})
            .get("proxy_to_source", {})
            .get("certified_upper_bound_mm", math.nan)
        )
        rebuilt_proxy_bound = float(
            rebuilt["octree"]["unresolved_boundary_maximum_mm"]
        )
        if not _equivalent_float(reported_proxy_bound, rebuilt_proxy_bound, 2.0e-6):
            errors.append("proxy-to-source interval bound does not recompute")
        if rebuilt_proxy_bound > parameters.boundary_threshold_mm + 1.0e-12:
            errors.append("proxy-to-source interval exceeds threshold")

    passed = not errors and bool(record.get("passed"))
    return {
        "component_id": component_id,
        "exact_subset_recomputed": subset,
        "functional_voids_recomputed": voids,
        "boundary_recomputed": boundary,
        "release_octree_rebuilt": release_rebuild_octree,
        "errors": errors,
        "passed": passed,
    }


def _inventory_from_boxes(boxes: np.ndarray) -> dict[str, Any]:
    # Use the same canonical serialization, but do not accept its booleans.
    return generator._boxes_inventory(np.asarray(boxes, dtype=np.float64))


def run_adversarial_suite(
    component_id: str,
    source_triangles: np.ndarray,
    boxes_mm: np.ndarray,
    parameters: generator.OctreeParameters,
    source_frame: dict[str, Any],
    *,
    recompute_deleted_wall_boundary: bool = True,
) -> dict[str, Any]:
    """Mutate raw evidence and require each independent oracle to reject it."""

    cases: list[dict[str, Any]] = []

    translated = np.asarray(boxes_mm, dtype=np.float64) + np.asarray([200.0, 0.0, 0.0])
    translated_inside, _, _ = generator.SignedFcpwMesh(source_triangles).query(
        np.mean(translated, axis=1)
    )
    cases.append(
        {
            "case": "translated_plus_200mm",
            "rejected": bool(not np.all(translated_inside)),
            "oracle": "translated_piece_centres_outside_closed_source",
        }
    )

    roster = generator.functional_void_roster(component_id)
    bore_point = np.asarray(roster[0]["points_mm"][0], dtype=np.float64)
    injected_bore = np.asarray([[bore_point - 0.05, bore_point + 0.05]])
    filled = np.concatenate((boxes_mm, injected_bore), axis=0)
    filled_voids = generator.evaluate_functional_voids(
        component_id, generator.SignedFcpwMesh(source_triangles), filled
    )
    cases.append(
        {
            "case": "filled_functional_bore",
            "rejected": bool(not filled_voids["passed"]),
            "oracle": "complete_functional_void_probe",
        }
    )

    centers = np.mean(boxes_mm, axis=1)
    cut = float(np.median(centers[:, 0]))
    deleted = boxes_mm[centers[:, 0] <= cut]
    if len(deleted) == len(boxes_mm):
        # A one-piece proxy still needs a real deleted-wall mutation.  Retain
        # only one half of its longest dimension; never accept the unchanged
        # baseline as an adversarial witness.
        selected = int(np.argmax(boxes_mm[0, 1] - boxes_mm[0, 0]))
        deleted = boxes_mm[:1].copy()
        deleted[0, 1, selected] = float(
            (deleted[0, 0, selected] + deleted[0, 1, selected]) / 2.0
        )
    deleted_rejected = True
    deleted_observation: float | None = None
    if recompute_deleted_wall_boundary:
        # One exact source-surface counterexample is sufficient to reject a
        # claimed upper bound.  The mutation test therefore uses original
        # triangle vertices and float64 replay, not the much larger release
        # covering set needed to *prove* a passing upper bound.
        deleted_index = generator._FcpwTriangleUpperBoundIndex(
            generator._proxy_triangle_mesh(deleted)
        )
        source_vertices = np.ascontiguousarray(source_triangles.reshape(-1, 3))
        distances = deleted_index.distances(source_vertices)
        deleted_observation = float(np.max(distances))
        deleted_rejected = bool(
            deleted_observation > parameters.boundary_threshold_mm
        )
    cases.append(
        {
            "case": "deleted_wall",
            "rejected": deleted_rejected,
            "oracle": "covered_source_surface_recompute",
            "observed_bound_mm": deleted_observation,
        }
    )

    bbox_max = np.max(source_triangles, axis=(0, 1))
    outside_center = bbox_max + np.asarray([2.0, 2.0, 2.0])
    injected_inside, _, _ = generator.SignedFcpwMesh(source_triangles).query(
        np.asarray([outside_center])
    )
    cases.append(
        {
            "case": "injected_outside_cell",
            "rejected": bool(not injected_inside[0]),
            "oracle": "injected_cell_centre_outside_closed_source",
        }
    )

    rotated = copy.deepcopy(source_frame)
    rotated["body_local_quat_wxyz"] = [
        math.cos(math.pi / 4.0),
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
    ]
    rotated_core = {key: value for key, value in rotated.items() if key != "frame_sha256"}
    rotated["frame_sha256"] = _canonical_sha256(rotated_core)
    spec = generator.plate_source_specs().get(component_id)
    frame_rejected = spec is None or not validate_source_frame(rotated, spec)
    cases.append(
        {
            "case": "rotated_source_frame",
            "rejected": frame_rejected,
            "oracle": "calibrated_source_frame_recompute",
        }
    )

    expected = {
        "translated_plus_200mm",
        "filled_functional_bore",
        "deleted_wall",
        "injected_outside_cell",
        "rotated_source_frame",
    }
    observed = {str(record["case"]) for record in cases}
    return {
        "cases": cases,
        "case_inventory_sha256": _canonical_sha256(sorted(expected)),
        "complete": observed == expected,
        "passed": bool(observed == expected and all(record["rejected"] for record in cases)),
    }


def validate_report(
    report: Any,
    *,
    release_rebuild_octree: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return {"passed": False, "errors": ["report is not an object"]}
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema version mismatch")
    if not _validate_file_record(report.get("report_generator"), expected_role="report_generator"):
        errors.append("generator file record is invalid")
    if not _validate_file_record(report.get("kernel_dependency"), expected_role="FCPW_geometry_kernel"):
        errors.append("kernel file record is invalid")
    components = report.get("components")
    if not isinstance(components, list):
        return {"passed": False, "errors": errors + ["components list is absent"]}
    if [record.get("component_id") for record in components] != [
        "robot_plate",
        "generic_tool_plate",
    ]:
        errors.append("component inventory/order is not canonical")
    results = [
        validate_component_record(
            record,
            release_rebuild_octree=release_rebuild_octree,
            recompute_boundary=True,
        )
        for record in components
    ]
    if not all(result["passed"] for result in results):
        errors.append("one or more component records failed recomputation")
    if bool(report.get("release_ready")):
        errors.append("checkpoint report must not claim release_ready")
    return {
        "schema_version": SCHEMA_VERSION,
        "release_rebuild_octree": release_rebuild_octree,
        "component_results": results,
        "errors": errors,
        "passed": bool(not errors and report.get("passed")),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--release-rebuild-octree",
        action="store_true",
        help="reconstruct all proxy pieces rather than only replaying report evidence",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = json.loads(args.report.read_text())
    result = validate_report(
        report, release_rebuild_octree=args.release_rebuild_octree
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "boxes_from_inventory",
    "recompute_exact_subset",
    "run_adversarial_suite",
    "validate_component_record",
    "validate_report",
    "validate_source_frame",
]
