#!/usr/bin/env python3
"""Build conservative box proxies for the two released quick-change plates.

The runtime collision representation is deliberately *not* trusted as CAD
authority.  This offline tool imports a hash-pinned STEP file, asks OCCT only
for an absolute-deflection triangle surface, and then uses FCPW queries to
construct an eroded union of axis-aligned boxes.  A box is admitted only when
its complete circumscribed ball is proven to remain inside the source solid.

The hot-path geometry after STEP tessellation contains no OCCT Boolean.  It is
also useful without producing a report: ``--preflight`` verifies source
hashes, frames, topology, and the complete functional-void roster in seconds.
The comparatively expensive adaptive octree is run only by ``--write-report``.

This module is intentionally separate from the workflow/controller.  It does
not edit or import the MuJoCo scene and it does not claim that a finite set of
surface samples is a bound: every directed boundary result includes either a
surface covering radius or an octree interval radius.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import cadquery as cq
import fcpw
import numpy as np

from generate_matcha_payload_proxy_report import (
    FCPW_REQUIRED_VERSION,
    _FcpwTriangleUpperBoundIndex,
    _point_triangle_distance_float64,
    _subdivided_surface_witnesses,
    _surface_topology,
)


SCHEMA_VERSION = "1.0.0-plate-proxy-checkpoint"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPOSITORY_ROOT / "QuickChange" / "SO101_Magnetic"
EXPORT_DIR = CORE_DIR / "exports"
CORE_MANIFEST = EXPORT_DIR / "core_cad_manifest.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "plate_proxy_authority_report.json"

ABSOLUTE_TESSELLATION_REQUEST_MM = 0.005
# OCCT's absolute linear request is guarded rather than repeated as if it were
# an independently measured B-rep Hausdorff result.  The release report must
# carry this larger source-to-facet allowance in every relevant inequality.
SOURCE_FACETING_BOUND_MM = 0.010
ANGULAR_DEFLECTION_RAD = 0.15
BOUNDARY_THRESHOLD_MM = 0.35
FCPW_FLOAT_GUARD_MM = 2.0e-5
SOURCE_WITNESS_COVERING_RADIUS_MM = 0.080
MAX_OCTREE_DEPTH = 10
MAX_FRONTIER_CELLS = 3_000_000
QUERY_CHUNK_SIZE = 250_000
MAX_PROXY_BOXES_AFTER_MERGE = 100_000
FRAME_POSITION_TOLERANCE_MM = 1.0e-12


if importlib.metadata.version("fcpw") != FCPW_REQUIRED_VERSION:
    raise RuntimeError(f"fcpw {FCPW_REQUIRED_VERSION} is required")


@dataclass(frozen=True)
class PlateSourceSpec:
    component_id: str
    step_path: Path
    expected_step_sha256: str
    expected_frame_pos_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    expected_frame_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class OctreeParameters:
    maximum_depth: int = MAX_OCTREE_DEPTH
    boundary_threshold_mm: float = BOUNDARY_THRESHOLD_MM
    source_faceting_bound_mm: float = SOURCE_FACETING_BOUND_MM
    fcpw_float_guard_mm: float = FCPW_FLOAT_GUARD_MM
    source_witness_covering_radius_mm: float = SOURCE_WITNESS_COVERING_RADIUS_MM
    query_chunk_size: int = QUERY_CHUNK_SIZE
    max_frontier_cells: int = MAX_FRONTIER_CELLS
    max_proxy_boxes_after_merge: int = MAX_PROXY_BOXES_AFTER_MERGE

    def validate(self) -> None:
        if self.maximum_depth < 1 or self.maximum_depth > 16:
            raise ValueError("maximum_depth must be in [1, 16]")
        for name in (
            "boundary_threshold_mm",
            "source_faceting_bound_mm",
            "fcpw_float_guard_mm",
            "source_witness_covering_radius_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.source_faceting_bound_mm >= self.boundary_threshold_mm:
            raise ValueError("faceting allowance consumes the boundary threshold")
        if self.boundary_threshold_mm > BOUNDARY_THRESHOLD_MM + 1.0e-12:
            raise ValueError(
                f"boundary threshold cannot exceed {BOUNDARY_THRESHOLD_MM} mm"
            )
        if self.source_faceting_bound_mm < SOURCE_FACETING_BOUND_MM - 1.0e-12:
            raise ValueError("source faceting allowance cannot be weakened")
        if self.fcpw_float_guard_mm < FCPW_FLOAT_GUARD_MM - 1.0e-12:
            raise ValueError("FCPW numeric guard cannot be weakened")
        if self.query_chunk_size < 1 or self.max_frontier_cells < 8:
            raise ValueError("invalid bounded-work parameters")
        if self.max_proxy_boxes_after_merge < 1:
            raise ValueError("proxy box cap must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact escapes repository: {resolved}") from error
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        "role": role,
    }


def _manifest_step_hashes() -> dict[str, str]:
    manifest = json.loads(CORE_MANIFEST.read_text())
    result: dict[str, str] = {}
    for record in manifest.get("files", []):
        path = str(record.get("path", ""))
        result[Path(path).name] = str(record.get("sha256", ""))
    return result


def plate_source_specs() -> dict[str, PlateSourceSpec]:
    hashes = _manifest_step_hashes()
    names = {
        "robot_plate": "so101_robot_plate.step",
        "generic_tool_plate": "so101_tool_plate.step",
    }
    specs: dict[str, PlateSourceSpec] = {}
    for component_id, filename in names.items():
        digest = hashes.get(filename, "")
        if len(digest) != 64:
            raise RuntimeError(f"core manifest does not pin {filename}")
        specs[component_id] = PlateSourceSpec(
            component_id=component_id,
            step_path=EXPORT_DIR / filename,
            expected_step_sha256=digest,
        )
    return specs


def _frame_payload(spec: PlateSourceSpec) -> dict[str, Any]:
    payload = {
        "frame_id": f"{spec.component_id}_step_native_mm",
        "body_local_pos_mm": list(spec.expected_frame_pos_mm),
        "body_local_quat_wxyz": list(spec.expected_frame_quat_wxyz),
        "composition": "runtime_body_point = R(quat) * STEP_native_mm + pos_mm",
        "units": "millimeter",
    }
    payload["frame_sha256"] = _canonical_sha256(payload)
    return payload


def _mesh_sha256(triangles: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(triangles, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def load_absolute_step_mesh(
    spec: PlateSourceSpec,
    *,
    requested_deflection_mm: float = ABSOLUTE_TESSELLATION_REQUEST_MM,
    angular_deflection_rad: float = ANGULAR_DEFLECTION_RAD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Import every STEP solid and return one oriented, closed triangle mesh."""

    observed = _sha256(spec.step_path)
    if observed != spec.expected_step_sha256:
        raise RuntimeError(
            f"{spec.component_id} STEP hash mismatch: expected "
            f"{spec.expected_step_sha256}, got {observed}"
        )

    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    imported = cq.importers.importStep(str(spec.step_path))
    selected = imported.val()
    selected_solid_count = len(selected.Solids())
    if selected_solid_count != 1:
        raise RuntimeError(
            f"{spec.component_id} all_solids selector expected one released solid, "
            f"observed {selected_solid_count}"
        )
    shape = selected.wrapped
    BRepTools.Clean_s(shape)
    mesher = BRepMesh_IncrementalMesh(
        shape,
        float(requested_deflection_mm),
        False,
        float(angular_deflection_rad),
        False,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError(f"absolute tessellation failed for {spec.component_id}")

    triangles: list[list[list[float]]] = []
    face_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face_count += 1
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            raise RuntimeError(f"unmeshed STEP face {face_count} in {spec.component_id}")
        transform = location.Transformation()
        nodes = [
            triangulation.Node(index).Transformed(transform)
            for index in range(1, triangulation.NbNodes() + 1)
        ]
        for index in range(1, triangulation.NbTriangles() + 1):
            indices = [int(value) - 1 for value in triangulation.Triangle(index).Get()]
            if face.Orientation() == TopAbs_REVERSED:
                indices = [indices[0], indices[2], indices[1]]
            triangles.append(
                [
                    [
                        float(nodes[node].X()),
                        float(nodes[node].Y()),
                        float(nodes[node].Z()),
                    ]
                    for node in indices
                ]
            )
        explorer.Next()

    mesh = np.ascontiguousarray(np.asarray(triangles, dtype=np.float64))
    topology = _surface_topology(mesh)
    if not bool(topology["passed"]):
        raise RuntimeError(f"{spec.component_id} tessellation is not a closed solid")
    bbox_min = np.min(mesh, axis=(0, 1))
    bbox_max = np.max(mesh, axis=(0, 1))
    float32_error = float(
        np.max(
            np.linalg.norm(
                mesh - np.asarray(mesh, dtype=np.float32).astype(np.float64), axis=2
            )
        )
    )
    certificate = {
        "method": "OCCT_BRepMesh_IncrementalMesh_absolute",
        "relative_mode": False,
        "requested_absolute_linear_deflection_mm": float(requested_deflection_mm),
        "angular_deflection_rad": float(angular_deflection_rad),
        "source_to_faceted_surface_bound_mm": SOURCE_FACETING_BOUND_MM,
        "source_to_faceted_bound_role": (
            "release_conservative_allowance_not_measured_Hausdorff"
        ),
        "face_count": face_count,
        "selected_solid_count": selected_solid_count,
        "triangle_count": len(mesh),
        "triangle_mesh_sha256": _mesh_sha256(mesh),
        "bbox_min_mm": bbox_min.tolist(),
        "bbox_max_mm": bbox_max.tolist(),
        "float64_to_float32_vertex_error_upper_bound_mm": float32_error,
        "topology": topology,
        "passed": True,
    }
    certificate["certificate_sha256"] = _canonical_sha256(certificate)
    return mesh, certificate


class SignedFcpwMesh:
    """Fast signed queries with conservative float64-source distance bounds."""

    def __init__(self, triangles: np.ndarray) -> None:
        self.triangles = np.ascontiguousarray(triangles, dtype=np.float64)
        self._index = _FcpwTriangleUpperBoundIndex(self.triangles)
        cast = self.triangles - self.triangles.astype(np.float32).astype(np.float64)
        self.mesh_cast_error_mm = float(np.max(np.linalg.norm(cast, axis=2)))

    @property
    def scene(self):
        return self._index._scene

    def query(
        self,
        points: np.ndarray,
        *,
        chunk_size: int = QUERY_CHUNK_SIZE,
        numeric_guard_mm: float = FCPW_FLOAT_GUARD_MM,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return inside flags plus lower/upper bounds to the float64 mesh."""

        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
            raise ValueError("query points must be finite (N,3)")
        inside_parts: list[np.ndarray] = []
        lower_parts: list[np.ndarray] = []
        upper_parts: list[np.ndarray] = []
        for offset in range(0, len(query), int(chunk_size)):
            original = query[offset : offset + int(chunk_size)]
            query32 = np.asfortranarray(original, dtype=np.float32)
            squared_radii = np.full(
                len(query32), np.finfo(np.float32).max, dtype=np.float32
            )
            interactions = fcpw.interaction_3D_list()
            self.scene.find_closest_points(
                query32, squared_radii, interactions, False
            )
            if len(interactions) != len(query32):
                raise RuntimeError("FCPW closest-point batch is incomplete")
            distances = np.fromiter(
                (float(interaction.d) for interaction in interactions),
                dtype=np.float64,
                count=len(interactions),
            )
            query_cast_error = np.linalg.norm(
                original - query32.astype(np.float64), axis=1
            )
            guard = float(numeric_guard_mm)
            if not math.isfinite(guard) or guard < FCPW_FLOAT_GUARD_MM - 1.0e-12:
                raise ValueError("FCPW numeric guard cannot be weakened")
            total_error = query_cast_error + self.mesh_cast_error_mm + guard
            lower_parts.append(np.maximum(0.0, distances - total_error))
            upper_parts.append(distances + total_error)
            inside_parts.append(
                np.fromiter(
                    (
                        bool(self.scene.contains(np.ascontiguousarray(point)))
                        for point in query32
                    ),
                    dtype=np.bool_,
                    count=len(query32),
                )
            )
        return (
            np.concatenate(inside_parts) if inside_parts else np.empty(0, dtype=bool),
            np.concatenate(lower_parts) if lower_parts else np.empty(0),
            np.concatenate(upper_parts) if upper_parts else np.empty(0),
        )


def _power_of_two_extent(extent: np.ndarray) -> np.ndarray:
    if np.any(extent <= 0.0):
        raise ValueError("source bbox must have positive extent in every axis")
    return np.power(2.0, np.ceil(np.log2(extent)))


def _root_grid(triangles: np.ndarray, maximum_depth: int) -> dict[str, Any]:
    bbox_min = np.min(triangles, axis=(0, 1))
    bbox_max = np.max(triangles, axis=(0, 1))
    side = _power_of_two_extent(bbox_max - bbox_min)
    center = (bbox_min + bbox_max) / 2.0
    root_min = center - side / 2.0
    root_max = center + side / 2.0
    grid_units = 1 << int(maximum_depth)
    leaf_size = side / grid_units
    return {
        "root_min_mm": root_min,
        "root_max_mm": root_max,
        "root_size_mm": side,
        "leaf_size_mm": leaf_size,
        "grid_units_per_axis": grid_units,
    }


def _split_starts(starts: np.ndarray, child_units: int) -> np.ndarray:
    offsets = np.asarray(
        [
            [x * child_units, y * child_units, z * child_units]
            for x in (0, 1)
            for y in (0, 1)
            for z in (0, 1)
        ],
        dtype=np.int32,
    )
    return np.ascontiguousarray(
        (starts[:, None, :] + offsets[None, :, :]).reshape(-1, 3),
        dtype=np.int32,
    )


def _merge_axis(records: np.ndarray, margins: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedily merge exactly adjacent boxes with matching other extents."""

    if len(records) < 2:
        return records, margins
    low_col = 2 * axis
    high_col = low_col + 1
    other = [column for column in range(6) if column not in (low_col, high_col)]
    sort_columns = other + [low_col, high_col]
    order = np.lexsort(tuple(records[:, column] for column in reversed(sort_columns)))
    sorted_records = records[order]
    sorted_margins = margins[order]
    output_records: list[np.ndarray] = []
    output_margins: list[float] = []
    current = sorted_records[0].copy()
    current_margin = float(sorted_margins[0])
    for record, margin in zip(
        sorted_records[1:], sorted_margins[1:], strict=True
    ):
        matching_cross_section = all(
            int(record[column]) == int(current[column]) for column in other
        )
        if matching_cross_section and int(record[low_col]) == int(current[high_col]):
            current[high_col] = record[high_col]
            current_margin = min(current_margin, float(margin))
        else:
            output_records.append(current)
            output_margins.append(current_margin)
            current = record.copy()
            current_margin = float(margin)
    output_records.append(current)
    output_margins.append(current_margin)
    return (
        np.ascontiguousarray(np.asarray(output_records, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(output_margins, dtype=np.float64)),
    )


def greedy_merge_boxes(
    records: np.ndarray, margins: np.ndarray, *, maximum_passes: int = 9
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Merge a proven cell union without adding a single point of material."""

    boxes = np.ascontiguousarray(records, dtype=np.int32)
    proof_margins = np.ascontiguousarray(margins, dtype=np.float64)
    initial_count = len(boxes)
    history = [initial_count]
    for pass_index in range(maximum_passes):
        before = len(boxes)
        for axis in (0, 1, 2):
            boxes, proof_margins = _merge_axis(boxes, proof_margins, axis)
        history.append(len(boxes))
        if len(boxes) == before:
            break
    order = np.lexsort(tuple(boxes[:, column] for column in reversed(range(6))))
    boxes = boxes[order]
    proof_margins = proof_margins[order]
    return boxes, proof_margins, {
        "method": "deterministic_exact_adjacent_axis_box_merge",
        "initial_cell_count": initial_count,
        "pass_counts": history,
        "final_box_count": len(boxes),
        "union_preserved_exactly": True,
        "material_added": False,
    }


def _integer_boxes_to_mm(
    records: np.ndarray, root_min: np.ndarray, leaf_size: np.ndarray
) -> np.ndarray:
    result = np.empty((len(records), 2, 3), dtype=np.float64)
    result[:, 0, :] = root_min + records[:, (0, 2, 4)] * leaf_size
    result[:, 1, :] = root_min + records[:, (1, 3, 5)] * leaf_size
    return result


def build_adaptive_subset_boxes(
    triangles: np.ndarray,
    parameters: OctreeParameters,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Return merged exact-subset boxes, octree certificate, unresolved cells."""

    parameters.validate()
    grid = _root_grid(triangles, parameters.maximum_depth)
    root_min = np.asarray(grid["root_min_mm"], dtype=np.float64)
    leaf_size = np.asarray(grid["leaf_size_mm"], dtype=np.float64)
    signed_mesh = SignedFcpwMesh(triangles)

    frontier = np.zeros((1, 3), dtype=np.int32)
    accepted_records: list[np.ndarray] = []
    accepted_margins: list[np.ndarray] = []
    unresolved_records: list[np.ndarray] = []
    unresolved_bounds: list[np.ndarray] = []
    depth_records: list[dict[str, Any]] = []
    outside_count = 0
    started = time.monotonic()

    for depth in range(parameters.maximum_depth + 1):
        if len(frontier) > parameters.max_frontier_cells:
            raise RuntimeError(
                f"octree frontier {len(frontier)} exceeds fail-closed cap"
            )
        units = 1 << (parameters.maximum_depth - depth)
        half_size = leaf_size * (units / 2.0)
        half_diagonal = float(np.linalg.norm(half_size))
        centers = root_min + (frontier + units / 2.0) * leaf_size
        inside, distance_lower, distance_upper = signed_mesh.query(
            centers,
            chunk_size=parameters.query_chunk_size,
            numeric_guard_mm=parameters.fcpw_float_guard_mm,
        )
        exact_subset_margin = (
            distance_lower
            - half_diagonal
            - parameters.source_faceting_bound_mm
        )
        safe = exact_subset_margin > 0.0
        accept = safe & inside
        outside = safe & ~inside
        proxy_to_source_bound = (
            distance_upper
            + half_diagonal
            + parameters.source_faceting_bound_mm
        )
        stop_unresolved = (~safe) & (
            proxy_to_source_bound <= parameters.boundary_threshold_mm
        )
        if depth == parameters.maximum_depth:
            stop_unresolved |= ~safe
        continue_refinement = ~(accept | outside | stop_unresolved)

        selected = frontier[accept]
        if len(selected):
            ends = selected + units
            accepted_records.append(
                np.column_stack(
                    (
                        selected[:, 0], ends[:, 0],
                        selected[:, 1], ends[:, 1],
                        selected[:, 2], ends[:, 2],
                    )
                ).astype(np.int32, copy=False)
            )
            accepted_margins.append(exact_subset_margin[accept])
        unresolved = frontier[stop_unresolved]
        if len(unresolved):
            ends = unresolved + units
            unresolved_records.append(
                np.column_stack(
                    (
                        unresolved[:, 0], ends[:, 0],
                        unresolved[:, 1], ends[:, 1],
                        unresolved[:, 2], ends[:, 2],
                    )
                ).astype(np.int32, copy=False)
            )
            unresolved_bounds.append(proxy_to_source_bound[stop_unresolved])
        outside_count += int(np.count_nonzero(outside))
        depth_records.append(
            {
                "depth": depth,
                "evaluated_cell_count": len(frontier),
                "accepted_cell_count": int(np.count_nonzero(accept)),
                "outside_cell_count": int(np.count_nonzero(outside)),
                "resolved_boundary_cell_count": int(np.count_nonzero(stop_unresolved)),
                "refined_cell_count": int(np.count_nonzero(continue_refinement)),
                "cell_size_mm": (leaf_size * units).tolist(),
                "half_diagonal_mm": half_diagonal,
            }
        )
        refining = frontier[continue_refinement]
        if not len(refining):
            frontier = np.empty((0, 3), dtype=np.int32)
            break
        child_units = units // 2
        if child_units < 1:
            raise RuntimeError("unresolved cells remain below maximum octree depth")
        frontier = _split_starts(refining, child_units)

    if len(frontier):
        raise RuntimeError("octree did not close its frontier")
    if not accepted_records or not unresolved_records:
        raise RuntimeError("octree failed to produce material and boundary records")
    accepted = np.concatenate(accepted_records, axis=0)
    margins = np.concatenate(accepted_margins)
    unresolved = np.concatenate(unresolved_records, axis=0)
    boundary_bounds = np.concatenate(unresolved_bounds)
    if np.any(boundary_bounds > parameters.boundary_threshold_mm + 1.0e-12):
        raise RuntimeError("maximum depth leaves an uncertified boundary interval")

    merged, merged_margins, merge_record = greedy_merge_boxes(accepted, margins)
    if len(merged) > parameters.max_proxy_boxes_after_merge:
        raise RuntimeError(
            f"merged proxy has {len(merged)} boxes, cap is "
            f"{parameters.max_proxy_boxes_after_merge}"
        )
    boxes_mm = _integer_boxes_to_mm(merged, root_min, leaf_size)
    unresolved_mm = _integer_boxes_to_mm(unresolved, root_min, leaf_size)
    grid_limit = int(grid["grid_units_per_axis"])
    accepted_root_boundary_face_count = int(
        np.count_nonzero(
            (merged[:, 0] == 0)
            | (merged[:, 2] == 0)
            | (merged[:, 4] == 0)
            | (merged[:, 1] == grid_limit)
            | (merged[:, 3] == grid_limit)
            | (merged[:, 5] == grid_limit)
        )
    )
    if accepted_root_boundary_face_count:
        raise RuntimeError("accepted proxy reaches the octree root boundary")
    integer_digest_payload = {
        "root_min_mm": root_min.tolist(),
        "leaf_size_mm": leaf_size.tolist(),
        "box_integer_bounds": merged.tolist(),
    }
    subset = {
        "method": (
            "oriented_FCPW_containment_plus_float32_distance_lower_bound_"
            "minus_cell_half_diagonal_minus_STEP_faceting"
        ),
        "whole_cell_subset_proof": True,
        "source_closed_oriented_topology_required": True,
        "fcpw_float32_mesh_cast_error_mm": signed_mesh.mesh_cast_error_mm,
        "fcpw_numeric_guard_mm": parameters.fcpw_float_guard_mm,
        "source_faceting_bound_mm": parameters.source_faceting_bound_mm,
        "minimum_exact_subset_margin_mm": float(np.min(merged_margins)),
        "accepted_cell_count_before_merge": len(accepted),
        "proxy_box_count": len(merged),
        "proxy_integer_inventory_sha256": _canonical_sha256(integer_digest_payload),
        "merge": merge_record,
        "passed": True,
    }
    octree = {
        "maximum_depth": parameters.maximum_depth,
        "root_min_mm": root_min.tolist(),
        "root_max_mm": np.asarray(grid["root_max_mm"]).tolist(),
        "root_size_mm": np.asarray(grid["root_size_mm"]).tolist(),
        "leaf_size_mm": leaf_size.tolist(),
        "grid_units_per_axis": grid["grid_units_per_axis"],
        "depth_records": depth_records,
        "outside_cell_count": outside_count,
        "unresolved_boundary_cell_count": len(unresolved),
        "unresolved_boundary_maximum_mm": float(np.max(boundary_bounds)),
        "unresolved_boundary_record_sha256": hashlib.sha256(
            np.ascontiguousarray(unresolved, dtype="<i4").tobytes()
            + np.ascontiguousarray(boundary_bounds, dtype="<f8").tobytes()
        ).hexdigest(),
        "proxy_boundary_interval_theorem": (
            "every external accepted-box face borders an exhaustively classified "
            "unresolved cell; its entire distance to the faceted source is no "
            "greater than center_upper_distance plus cell_half_diagonal"
        ),
        "accepted_root_boundary_face_count": accepted_root_boundary_face_count,
        "all_external_faces_have_unresolved_neighbor_interval": True,
        "bounded_work": {
            "query_chunk_size": parameters.query_chunk_size,
            "max_frontier_cells": parameters.max_frontier_cells,
            "max_proxy_boxes_after_merge": parameters.max_proxy_boxes_after_merge,
        },
        "wall_seconds_observed": time.monotonic() - started,
        "passed": True,
    }
    return boxes_mm, {"octree": octree, "exact_subset": subset}, unresolved_mm


def _box_triangles(bounds: np.ndarray) -> np.ndarray:
    low = np.asarray(bounds[0], dtype=np.float64)
    high = np.asarray(bounds[1], dtype=np.float64)
    if np.any(high <= low):
        raise ValueError("box has nonpositive extent")
    vertices = np.asarray(
        [
            [low[0], low[1], low[2]], [high[0], low[1], low[2]],
            [high[0], high[1], low[2]], [low[0], high[1], low[2]],
            [low[0], low[1], high[2]], [high[0], low[1], high[2]],
            [high[0], high[1], high[2]], [low[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return np.ascontiguousarray(vertices[faces])


def _proxy_triangle_mesh(boxes_mm: np.ndarray) -> np.ndarray:
    if not len(boxes_mm):
        raise ValueError("proxy cannot be empty")
    return np.ascontiguousarray(
        np.concatenate([_box_triangles(box) for box in boxes_mm], axis=0)
    )


def certify_source_to_proxy_surface(
    source_triangles: np.ndarray,
    boxes_mm: np.ndarray,
    parameters: OctreeParameters,
) -> dict[str, Any]:
    """Bound every source-surface point to the exact box union."""

    proxy_triangles = _proxy_triangle_mesh(boxes_mm)
    proxy_index = _FcpwTriangleUpperBoundIndex(proxy_triangles)
    witnesses, covering_radius = _subdivided_surface_witnesses(
        source_triangles, parameters.source_witness_covering_radius_mm
    )
    distances = proxy_index.distances(witnesses)
    witness_maximum = float(np.max(distances))
    certified = math.fsum(
        (witness_maximum, covering_radius, parameters.source_faceting_bound_mm)
    )
    witness_digest = hashlib.sha256(
        np.ascontiguousarray(witnesses, dtype="<f8").tobytes()
        + np.ascontiguousarray(distances, dtype="<f8").tobytes()
    ).hexdigest()
    return {
        "direction": "exact_STEP_boundary_to_proxy_box_union",
        "method": "adaptive_source_triangle_cover_FCPW_candidates_float64_replay",
        "witness_count": len(witnesses),
        "witness_maximum_mm": witness_maximum,
        "source_surface_covering_radius_mm": covering_radius,
        "source_faceting_bound_mm": parameters.source_faceting_bound_mm,
        "certified_upper_bound_mm": certified,
        "witness_set_sha256": witness_digest,
        "passed": bool(certified <= parameters.boundary_threshold_mm + 1.0e-12),
    }


def _point_in_boxes(points: np.ndarray, boxes_mm: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64)
    result = np.zeros(len(query), dtype=bool)
    # The functional roster is small.  Chunk over boxes to avoid allocating
    # an unbounded points x pieces tensor when the release proxy is large.
    for offset in range(0, len(boxes_mm), 4096):
        chunk = boxes_mm[offset : offset + 4096]
        occupied = np.any(
            np.all(query[:, None, :] >= chunk[None, :, 0, :] - tolerance, axis=2)
            & np.all(query[:, None, :] <= chunk[None, :, 1, :] + tolerance, axis=2),
            axis=1,
        )
        result |= occupied
    return result


def _cylinder_probe_points(
    center_xy: tuple[float, float],
    z_range: tuple[float, float],
    radius: float,
    *,
    axial_count: int = 5,
) -> list[list[float]]:
    x, y = center_xy
    points: list[list[float]] = []
    for z in np.linspace(z_range[0], z_range[1], axial_count):
        points.append([x, y, float(z)])
        for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
            points.append(
                [
                    x + 0.55 * radius * math.cos(angle),
                    y + 0.55 * radius * math.sin(angle),
                    float(z),
                ]
            )
    return points


def functional_void_roster(component_id: str) -> list[dict[str, Any]]:
    """Return every source cutter family that must remain recognisably void."""

    # Probe coordinates stay strictly inside the released cutter, not on its
    # tolerance-sensitive boundary.  Families with several repeated cutters
    # carry every instance in one record and no wildcard/prefix exception.
    if component_id == "robot_plate":
        roster = [
            {
                "family": "wrist_horn_four_through_holes",
                "points_mm": sum(
                    (
                        _cylinder_probe_points((x, y), (0.35, 9.15), 1.65)
                        for x in (-4.95, 4.95) for y in (-4.95, 4.95)
                    ),
                    [],
                ),
            },
            {
                "family": "wrist_horn_central_rear_recess",
                "points_mm": _cylinder_probe_points((0.0, 0.0), (0.10, 2.0), 3.5),
            },
            {
                "family": "magnet_fastener_through_bores",
                "points_mm": sum(
                    (_cylinder_probe_points((0.0, y), (0.2, 9.3), 1.7) for y in (-16.0, 16.0)),
                    [],
                ),
            },
            {
                "family": "positive_lock_stud_head_wells",
                "points_mm": sum(
                    (_cylinder_probe_points((x, 0.0), (3.1, 9.25), 3.325) for x in (-12.0, 12.0)),
                    [],
                ),
            },
            {
                "family": "guide_screw_axis",
                "points_mm": _cylinder_probe_points((0.0, 0.0), (2.6, 9.3), 0.85),
            },
            {
                "family": "pogo_press_fit_four_bores",
                "points_mm": sum(
                    (_cylinder_probe_points((-31.0, y), (0.2, 9.3), 0.7875) for y in (-7.5, -2.5, 2.5, 7.5)),
                    [],
                ),
            },
            {
                "family": "cam_relief_full_height",
                "points_mm": [[24.0, y, z] for y in (-15.5, 0.0, 12.0, 23.5) for z in (0.5, 4.75, 9.0)],
            },
            {
                "family": "rear_wiring_pocket",
                "points_mm": [[x, y, z] for x in (-33.0, -30.0, -27.0) for y in (-8.0, 0.0, 8.0) for z in (0.2, 1.4)],
            },
            {
                "family": "return_spring_side_channel",
                "points_mm": [[x, 0.0, 4.2] for x in (-22.4, -18.0, -13.0)],
            },
            {
                "family": "slider_swept_track",
                "points_mm": [[x, y, z] for x in (-15.0, -12.0, 0.0, 12.0, 15.0, 22.0) for y in (-1.5, 0.0, 1.5) for z in (4.8, 5.5, 6.2)],
            },
        ]
    elif component_id == "generic_tool_plate":
        roster = [
            {
                "family": "target_fastener_two_through_bores",
                "points_mm": sum(
                    (_cylinder_probe_points((0.0, y), (0.2, 9.3), 2.9) for y in (-16.0, 16.0)),
                    [],
                ),
            },
            {
                "family": "locator_socket_left",
                "points_mm": _cylinder_probe_points((-20.0, 0.0), (0.15, 3.3), 2.4),
            },
            {
                "family": "locator_socket_right_relieved_pair_union",
                "points_mm": sum(
                    (_cylinder_probe_points((20.0, y), (0.15, 3.3), 2.4) for y in (-0.35, 0.35)),
                    [],
                ),
            },
            {
                "family": "lock_stud_two_through_bores",
                "points_mm": sum(
                    (_cylinder_probe_points((x, 0.0), (0.2, 9.3), 1.6) for x in (-12.0, 12.0)),
                    [],
                ),
            },
            {
                "family": "contact_board_front_pocket",
                "points_mm": [[x, y, z] for x in (-34.0, -30.0, -26.0) for y in (-9.5, 0.0, 9.5) for z in (0.15, 0.7)],
            },
            {
                "family": "contact_board_rear_window",
                "points_mm": [[-26.5, y, z] for y in (-7.0, 0.0, 7.0) for z in (1.2, 5.0, 9.0)],
            },
            {
                "family": "tool_mount_four_through_bores",
                "points_mm": sum(
                    (_cylinder_probe_points((x, y), (0.2, 9.3), 1.65) for x in (-21.0, 21.0) for y in (-17.0, 17.0)),
                    [],
                ),
            },
            {
                "family": "tool_mount_central_through_bore",
                "points_mm": _cylinder_probe_points((0.0, 0.0), (0.2, 9.3), 4.0),
            },
        ]
    else:
        raise KeyError(component_id)
    for record in roster:
        record["probe_count"] = len(record["points_mm"])
        record["probe_sha256"] = _canonical_sha256(record["points_mm"])
    return roster


EXPECTED_VOID_FAMILIES = {
    "robot_plate": {
        "wrist_horn_four_through_holes",
        "wrist_horn_central_rear_recess",
        "magnet_fastener_through_bores",
        "positive_lock_stud_head_wells",
        "guide_screw_axis",
        "pogo_press_fit_four_bores",
        "cam_relief_full_height",
        "rear_wiring_pocket",
        "return_spring_side_channel",
        "slider_swept_track",
    },
    "generic_tool_plate": {
        "target_fastener_two_through_bores",
        "locator_socket_left",
        "locator_socket_right_relieved_pair_union",
        "lock_stud_two_through_bores",
        "contact_board_front_pocket",
        "contact_board_rear_window",
        "tool_mount_four_through_bores",
        "tool_mount_central_through_bore",
    },
}


def evaluate_functional_voids(
    component_id: str,
    source_mesh: SignedFcpwMesh,
    boxes_mm: np.ndarray,
) -> dict[str, Any]:
    roster = functional_void_roster(component_id)
    observed = {str(record["family"]) for record in roster}
    complete = observed == EXPECTED_VOID_FAMILIES[component_id]
    results: list[dict[str, Any]] = []
    for record in roster:
        points = np.asarray(record["points_mm"], dtype=np.float64)
        source_occupied, _, _ = source_mesh.query(points)
        proxy_occupied = _point_in_boxes(points, boxes_mm)
        passed = bool(not np.any(source_occupied) and not np.any(proxy_occupied))
        results.append(
            {
                "family": record["family"],
                "probe_count": len(points),
                "probe_sha256": record["probe_sha256"],
                "source_occupied_probe_count": int(np.count_nonzero(source_occupied)),
                "proxy_occupied_probe_count": int(np.count_nonzero(proxy_occupied)),
                "passed": passed,
            }
        )
    payload = {
        "declared_family_count": len(EXPECTED_VOID_FAMILIES[component_id]),
        "observed_family_count": len(observed),
        "complete_declared_family_closure": complete,
        "family_results": results,
        "passed": bool(complete and all(record["passed"] for record in results)),
    }
    payload["inventory_sha256"] = _canonical_sha256(
        sorted(EXPECTED_VOID_FAMILIES[component_id])
    )
    return payload


def _boxes_inventory(boxes_mm: np.ndarray) -> dict[str, Any]:
    serialized = [
        {
            "name": f"plate_proxy_box_{index:05d}",
            "bounds_mm": [box[0].tolist(), box[1].tolist()],
            "center_mm": ((box[0] + box[1]) / 2.0).tolist(),
            "size_mm": (box[1] - box[0]).tolist(),
        }
        for index, box in enumerate(boxes_mm)
    ]
    return {
        "piece_count": len(serialized),
        "pieces": serialized,
        "inventory_sha256": _canonical_sha256(serialized),
    }


def build_component_record(
    spec: PlateSourceSpec,
    parameters: OctreeParameters | None = None,
) -> dict[str, Any]:
    params = parameters or OctreeParameters()
    params.validate()
    started = time.monotonic()
    source_triangles, tessellation = load_absolute_step_mesh(spec)
    boxes_mm, construction, _ = build_adaptive_subset_boxes(source_triangles, params)
    source_to_proxy = certify_source_to_proxy_surface(source_triangles, boxes_mm, params)
    proxy_to_source = {
        "direction": "proxy_box_union_boundary_to_exact_STEP_boundary",
        "method": "exhaustive_octree_unresolved_neighbor_interval_bound",
        "certified_upper_bound_mm": construction["octree"][
            "unresolved_boundary_maximum_mm"
        ],
        "finite_sample_only": False,
        "accepted_root_boundary_face_count": construction["octree"][
            "accepted_root_boundary_face_count"
        ],
        "all_external_faces_have_unresolved_neighbor_interval": construction[
            "octree"
        ]["all_external_faces_have_unresolved_neighbor_interval"],
        "passed": bool(
            construction["octree"]["unresolved_boundary_maximum_mm"]
            <= params.boundary_threshold_mm + 1.0e-12
        ),
    }
    voids = evaluate_functional_voids(
        spec.component_id, SignedFcpwMesh(source_triangles), boxes_mm
    )
    boundary = {
        "threshold_mm": params.boundary_threshold_mm,
        "source_to_proxy": source_to_proxy,
        "proxy_to_source": proxy_to_source,
        "passed": bool(source_to_proxy["passed"] and proxy_to_source["passed"]),
    }
    record = {
        "component_id": spec.component_id,
        "source_authority": {
            "artifact": _file_record(spec.step_path, role="exact_CAD_STEP"),
            "selector": {"kind": "all_solids"},
            "selector_sha256": _canonical_sha256({"kind": "all_solids"}),
            "core_manifest": _file_record(CORE_MANIFEST, role="source_manifest"),
        },
        "source_frame": _frame_payload(spec),
        "tessellation_certificate": tessellation,
        "octree_parameters": {
            "maximum_depth": params.maximum_depth,
            "boundary_threshold_mm": params.boundary_threshold_mm,
            "source_faceting_bound_mm": params.source_faceting_bound_mm,
            "fcpw_float_guard_mm": params.fcpw_float_guard_mm,
            "source_witness_covering_radius_mm": params.source_witness_covering_radius_mm,
        },
        "octree_result": construction["octree"],
        "runtime_piece_inventory": _boxes_inventory(boxes_mm),
        "exact_subset_certificate": construction["exact_subset"],
        "bidirectional_boundary_certificate": boundary,
        "functional_void_results": voids,
        "wall_seconds_observed": time.monotonic() - started,
    }
    record["passed"] = bool(
        record["exact_subset_certificate"]["passed"]
        and boundary["passed"]
        and voids["passed"]
    )
    return record


def preflight_component(spec: PlateSourceSpec) -> dict[str, Any]:
    triangles, tessellation = load_absolute_step_mesh(spec)
    roster = functional_void_roster(spec.component_id)
    frame = _frame_payload(spec)
    return {
        "component_id": spec.component_id,
        "source_authority": _file_record(spec.step_path, role="exact_CAD_STEP"),
        "source_frame": frame,
        "tessellation_certificate": tessellation,
        "functional_void_family_count": len(roster),
        "functional_void_inventory_sha256": _canonical_sha256(
            sorted(record["family"] for record in roster)
        ),
        "source_triangle_count": len(triangles),
        "passed": bool(
            tessellation["passed"]
            and {record["family"] for record in roster}
            == EXPECTED_VOID_FAMILIES[spec.component_id]
        ),
    }


def build_report(
    component_ids: Sequence[str] = ("robot_plate", "generic_tool_plate"),
    parameters: OctreeParameters | None = None,
) -> dict[str, Any]:
    specs = plate_source_specs()
    params = parameters or OctreeParameters()
    components = [build_component_record(specs[name], params) for name in component_ids]
    report = {
        "schema_version": SCHEMA_VERSION,
        "release_ready": False,
        "passed": bool(all(record["passed"] for record in components)),
        "parameter_authority": {
            "units": "millimeter",
            "boundary_threshold_mm": params.boundary_threshold_mm,
            "manufacturing_clearance_authority": False,
            "runtime_role": "conservative_broadphase_and_contact_proxy",
            "exact_CAD_continuous_sweep_still_required_for_near_pairs": True,
        },
        "report_generator": _file_record(Path(__file__), role="report_generator"),
        "kernel_dependency": _file_record(
            Path(__file__).with_name("generate_matcha_payload_proxy_report.py"),
            role="FCPW_geometry_kernel",
        ),
        "components": components,
        "adversarial_results": {
            "status": "validator_recomputed",
            "required_cases": [
                "translated_plus_200mm",
                "filled_functional_bore",
                "deleted_wall",
                "injected_outside_cell",
                "rotated_source_frame",
            ],
        },
    }
    return report


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("all", "robot_plate", "generic_tool_plate"),
        default="all",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight",
        action="store_true",
        help="verify source hashes/frames/topology/void inventory without octree",
    )
    action.add_argument(
        "--write-report",
        type=Path,
        metavar="PATH",
        help="run the bounded adaptive construction and write canonical JSON",
    )
    parser.add_argument("--maximum-depth", type=int, default=MAX_OCTREE_DEPTH)
    parser.add_argument(
        "--boundary-threshold-mm", type=float, default=BOUNDARY_THRESHOLD_MM
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    names = (
        ("robot_plate", "generic_tool_plate")
        if args.component == "all"
        else (args.component,)
    )
    specs = plate_source_specs()
    if args.preflight:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": "preflight",
            "components": [preflight_component(specs[name]) for name in names],
        }
        payload["passed"] = all(record["passed"] for record in payload["components"])
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0 if payload["passed"] else 1
    parameters = OctreeParameters(
        maximum_depth=args.maximum_depth,
        boundary_threshold_mm=args.boundary_threshold_mm,
    )
    report = build_report(names, parameters)
    _atomic_json_write(args.write_report, report)
    print(args.write_report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_THRESHOLD_MM",
    "EXPECTED_VOID_FAMILIES",
    "OctreeParameters",
    "PlateSourceSpec",
    "SignedFcpwMesh",
    "build_adaptive_subset_boxes",
    "build_component_record",
    "build_report",
    "certify_source_to_proxy_surface",
    "evaluate_functional_voids",
    "functional_void_roster",
    "greedy_merge_boxes",
    "load_absolute_step_mesh",
    "plate_source_specs",
    "preflight_component",
]
