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
INTERNAL_BOUNDARY_TARGET_MM = 0.34
FCPW_FLOAT_GUARD_MM = 2.0e-5
SOURCE_WITNESS_COVERING_RADIUS_MM = 0.080
MAX_OCTREE_DEPTH = 10
MAX_FRONTIER_CELLS = 3_000_000
QUERY_CHUNK_SIZE = 250_000
MAX_PROXY_BOXES_AFTER_MERGE = 100_000
TARGETED_SEED_DISTANCE_MM = INTERNAL_BOUNDARY_TARGET_MM - SOURCE_FACETING_BOUND_MM
TARGETED_REFINEMENT_RADIUS_MM = 0.10
TARGETED_MAX_CELL_EDGE_MM = 0.0625
MAX_TARGETED_FRONTIER_CELLS = 3_000_000
MAX_ADAPTIVE_SURFACE_PATCHES = 3_000_000
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
    internal_boundary_target_mm: float = INTERNAL_BOUNDARY_TARGET_MM
    targeted_seed_distance_mm: float = TARGETED_SEED_DISTANCE_MM
    targeted_refinement_radius_mm: float = TARGETED_REFINEMENT_RADIUS_MM
    targeted_max_cell_edge_mm: float = TARGETED_MAX_CELL_EDGE_MM
    max_targeted_frontier_cells: int = MAX_TARGETED_FRONTIER_CELLS
    max_adaptive_surface_patches: int = MAX_ADAPTIVE_SURFACE_PATCHES

    def validate(self) -> None:
        if self.maximum_depth < 1 or self.maximum_depth > 16:
            raise ValueError("maximum_depth must be in [1, 16]")
        for name in (
            "boundary_threshold_mm",
            "source_faceting_bound_mm",
            "fcpw_float_guard_mm",
            "source_witness_covering_radius_mm",
            "internal_boundary_target_mm",
            "targeted_seed_distance_mm",
            "targeted_refinement_radius_mm",
            "targeted_max_cell_edge_mm",
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
        if self.internal_boundary_target_mm > INTERNAL_BOUNDARY_TARGET_MM + 1.0e-12:
            raise ValueError("internal boundary target cannot be weakened")
        if self.internal_boundary_target_mm >= self.boundary_threshold_mm:
            raise ValueError("internal boundary target must be below release threshold")
        if self.targeted_seed_distance_mm > (
            self.internal_boundary_target_mm - self.source_faceting_bound_mm
        ) + 1.0e-12:
            raise ValueError("targeted seed distance leaves an unguarded source witness")
        if self.targeted_max_cell_edge_mm >= 0.10:
            raise ValueError("targeted sub-leaf edge must remain below 0.10 mm")
        if self.query_chunk_size < 1 or self.max_frontier_cells < 8:
            raise ValueError("invalid bounded-work parameters")
        if self.max_proxy_boxes_after_merge < 1:
            raise ValueError("proxy box cap must be positive")
        if self.max_targeted_frontier_cells < 8 or self.max_adaptive_surface_patches < 8:
            raise ValueError("targeted work caps must be positive")


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

    def distance_bounds(
        self,
        points: np.ndarray,
        *,
        chunk_size: int = QUERY_CHUNK_SIZE,
        numeric_guard_mm: float = FCPW_FLOAT_GUARD_MM,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative float64-source distance bounds without winding.

        Source-surface covering checks need only unsigned distance.  Avoiding
        ``contains`` here is important because a proxy is an exact union of
        boxes whose triangle soup deliberately retains coincident internal
        faces; those faces are valid distance authority but not a winding
        surface, and one winding query per adaptive patch is wasted work.
        """

        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
            raise ValueError("query points must be finite (N,3)")
        guard = float(numeric_guard_mm)
        if not math.isfinite(guard) or guard < FCPW_FLOAT_GUARD_MM - 1.0e-12:
            raise ValueError("FCPW numeric guard cannot be weakened")
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
            total_error = query_cast_error + self.mesh_cast_error_mm + guard
            lower_parts.append(np.maximum(0.0, distances - total_error))
            upper_parts.append(distances + total_error)
        return (
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


def _proxy_point_distances_float64(
    boxes_mm: np.ndarray,
    points_mm: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Return conservative point-to-box-union upper bounds in float64."""

    index = _FcpwTriangleUpperBoundIndex(_proxy_triangle_mesh(boxes_mm))
    points = np.asarray(points_mm, dtype=np.float64)
    parts = [
        index.distances(points[offset : offset + chunk_size])
        for offset in range(0, len(points), chunk_size)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _scan_source_patches_for_undercoverage(
    source_triangles: np.ndarray,
    boxes_mm: np.ndarray,
    parameters: OctreeParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Partition the complete surface into safe leaves or refinement patches.

    Safe leaves use an upper distance plus their physical covering radius.
    Refinement seeds use a lower distance, so float32/FCPW rounding cannot
    hide a certificate-red patch.  ``red`` here means that the current
    conservative mesh-plus-faceting inequality cannot pass, not that a point
    of the underlying B-rep has independently been proved farther away.
    Ambiguous patches alone receive longest-edge splits.
    """

    proxy = SignedFcpwMesh(_proxy_triangle_mesh(boxes_mm))
    pending = np.ascontiguousarray(source_triangles, dtype=np.float64)
    refinement_parts: list[np.ndarray] = []
    digest = hashlib.sha256()
    safe_leaf_count = 0
    red_leaf_count = 0
    unresolved_seed_leaf_count = 0
    maximum_safe_upper = 0.0
    maximum_red_lower = 0.0
    worst_red_point: list[float] | None = None
    iterations: list[dict[str, Any]] = []
    for iteration in range(48):
        if not len(pending):
            break
        if (
            safe_leaf_count
            + red_leaf_count
            + unresolved_seed_leaf_count
            + len(pending)
            > parameters.max_adaptive_surface_patches
        ):
            raise RuntimeError("adaptive source scan exceeded fail-closed patch cap")
        centroids = np.mean(pending, axis=1)
        radii = np.max(
            np.linalg.norm(pending - centroids[:, None, :], axis=2), axis=1
        )
        distance_lower, distance_upper = proxy.distance_bounds(
            centroids,
            chunk_size=parameters.query_chunk_size,
            numeric_guard_mm=parameters.fcpw_float_guard_mm,
        )
        safe_upper = (
            distance_upper + radii + parameters.source_faceting_bound_mm
        )
        red_lower = distance_lower + parameters.source_faceting_bound_mm
        safe = safe_upper <= parameters.internal_boundary_target_mm + 1.0e-12
        red = red_lower > parameters.internal_boundary_target_mm + 1.0e-12
        if np.any(safe & red):
            raise RuntimeError("source patch classified simultaneously safe and red")
        ambiguous = ~(safe | red)
        unresolved_seed = ambiguous & (
            radii <= parameters.source_witness_covering_radius_mm + 1.0e-12
        )
        subdivide = ambiguous & ~unresolved_seed
        if np.any(safe):
            safe_centroids = np.ascontiguousarray(centroids[safe], dtype="<f8")
            safe_radii = np.ascontiguousarray(radii[safe], dtype="<f8")
            safe_distances = np.ascontiguousarray(distance_upper[safe], dtype="<f8")
            digest.update(b"safe")
            digest.update(safe_centroids.tobytes())
            digest.update(safe_radii.tobytes())
            digest.update(safe_distances.tobytes())
            safe_leaf_count += int(np.count_nonzero(safe))
            maximum_safe_upper = max(
                maximum_safe_upper, float(np.max(safe_upper[safe]))
            )
        if np.any(red):
            selected = np.ascontiguousarray(pending[red])
            refinement_parts.append(selected)
            digest.update(b"red")
            digest.update(np.ascontiguousarray(centroids[red], dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(distance_lower[red], dtype="<f8").tobytes())
            red_leaf_count += len(selected)
            local = int(np.argmax(red_lower[red]))
            local_value = float(red_lower[red][local])
            if local_value > maximum_red_lower:
                maximum_red_lower = local_value
                worst_red_point = centroids[red][local].tolist()
        if np.any(unresolved_seed):
            selected = np.ascontiguousarray(pending[unresolved_seed])
            refinement_parts.append(selected)
            digest.update(b"unresolved_seed")
            digest.update(
                np.ascontiguousarray(
                    centroids[unresolved_seed], dtype="<f8"
                ).tobytes()
            )
            digest.update(
                np.ascontiguousarray(
                    distance_upper[unresolved_seed], dtype="<f8"
                ).tobytes()
            )
            digest.update(
                np.ascontiguousarray(radii[unresolved_seed], dtype="<f8").tobytes()
            )
            unresolved_seed_leaf_count += len(selected)
        iterations.append(
            {
                "iteration": iteration,
                "evaluated_patch_count": len(pending),
                "certified_safe_patch_count": int(np.count_nonzero(safe)),
                "certificate_red_patch_count": int(np.count_nonzero(red)),
                "unresolved_refinement_seed_patch_count": int(
                    np.count_nonzero(unresolved_seed)
                ),
                "ambiguous_subdivided_patch_count": int(
                    np.count_nonzero(subdivide)
                ),
                "maximum_safe_candidate_bound_mm": float(np.max(safe_upper)),
                "maximum_certificate_red_lower_plus_faceting_mm": float(
                    np.max(red_lower)
                ),
            }
        )
        uncertain = pending[subdivide]
        if not len(uncertain):
            pending = np.empty((0, 3, 3), dtype=np.float64)
            break
        if (
            safe_leaf_count
            + red_leaf_count
            + unresolved_seed_leaf_count
            + 2 * len(uncertain)
            > parameters.max_adaptive_surface_patches
        ):
            raise RuntimeError(
                "adaptive source scan cannot resolve safe/red patches under cap"
            )
        pending = _split_triangle_longest_edge(uncertain)
    if len(pending):
        raise RuntimeError("adaptive source scan did not terminate")
    refinement_patches = (
        np.concatenate(refinement_parts, axis=0)
        if refinement_parts
        else np.empty((0, 3, 3), dtype=np.float64)
    )
    result = {
        "method": (
            "adaptive_longest_edge_complete_source_cover_with_"
            "FCPW_lower_red_and_upper_plus_radius_safe_bounds"
        ),
        "internal_boundary_target_mm": parameters.internal_boundary_target_mm,
        "complete_surface_partition": True,
        "certified_safe_patch_count": safe_leaf_count,
        "certificate_red_patch_count": red_leaf_count,
        "unresolved_refinement_seed_patch_count": unresolved_seed_leaf_count,
        "unresolved_refinement_seed_radius_mm": (
            parameters.source_witness_covering_radius_mm
        ),
        "maximum_certified_safe_upper_mm": maximum_safe_upper,
        "maximum_certificate_red_lower_plus_faceting_mm": maximum_red_lower,
        "worst_certificate_red_point_mm": worst_red_point,
        "partition_sha256": digest.hexdigest(),
        "iteration_records": iterations,
        "passed": red_leaf_count == 0 and unresolved_seed_leaf_count == 0,
    }
    return refinement_patches, result


def _finite_certificate_red_seed_patches(
    source_triangles: np.ndarray,
    boxes_mm: np.ndarray,
    parameters: OctreeParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return cheap, conservative refinement seeds before the complete cover.

    Vertices and centroids are not a surface certificate.  They are used only
    to avoid spending the complete adaptive-cover budget subdividing broad
    regions already known to fail the conservative distance-plus-faceting
    inequality.  FCPW lower bounds ensure numeric error cannot manufacture a
    red seed.
    """

    probes = np.concatenate(
        (source_triangles, np.mean(source_triangles, axis=1)[:, None, :]), axis=1
    )
    proxy = SignedFcpwMesh(_proxy_triangle_mesh(boxes_mm))
    lower, _ = proxy.distance_bounds(
        probes.reshape(-1, 3),
        chunk_size=parameters.query_chunk_size,
        numeric_guard_mm=parameters.fcpw_float_guard_mm,
    )
    lower = lower.reshape(len(source_triangles), 4)
    maxima = np.max(lower + parameters.source_faceting_bound_mm, axis=1)
    selected = maxima > parameters.internal_boundary_target_mm + 1.0e-12
    worst_triangle = int(np.argmax(maxima))
    worst_probe = int(np.argmax(lower[worst_triangle]))
    patches = np.ascontiguousarray(source_triangles[selected])
    record = {
        "method": (
            "source_triangle_vertices_and_centroid_FCPW_lower_bound_plus_"
            "faceting_refinement_seed_only"
        ),
        "triangle_count": len(source_triangles),
        "probe_count": int(probes.shape[0] * probes.shape[1]),
        "selected_triangle_count": len(patches),
        "selected_triangle_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(selected, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "maximum_lower_plus_faceting_mm": float(maxima[worst_triangle]),
        "worst_triangle_index": worst_triangle,
        "worst_probe_index": worst_probe,
        "worst_probe_point_mm": probes[worst_triangle, worst_probe].tolist(),
        "finite_seed_is_not_a_complete_surface_certificate": True,
        "passed": True,
    }
    return patches, record


def _cells_near_triangle_patches(
    unresolved_mm: np.ndarray,
    selected_patches: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Return cells intersecting deterministic expanded red-patch AABBs."""

    if not len(selected_patches):
        return np.zeros(len(unresolved_mm), dtype=bool)
    patches = np.asarray(selected_patches, dtype=np.float64)
    patch_low = np.min(patches, axis=1) - radius_mm
    patch_high = np.max(patches, axis=1) + radius_mm
    selected = np.zeros(len(unresolved_mm), dtype=bool)
    # Avoid an unbounded cells x patches tensor.  AABB intersection is a
    # conservative neighborhood selector; it is not a subset proof.
    for cell_offset in range(0, len(unresolved_mm), 50_000):
        cells = unresolved_mm[cell_offset : cell_offset + 50_000]
        hit = np.zeros(len(cells), dtype=bool)
        for patch_offset in range(0, len(patch_low), 256):
            low = patch_low[patch_offset : patch_offset + 256]
            high = patch_high[patch_offset : patch_offset + 256]
            hit |= np.any(
                np.all(cells[:, None, 1, :] >= low[None, :, :], axis=2)
                & np.all(cells[:, None, 0, :] <= high[None, :, :], axis=2),
                axis=1,
            )
            if np.all(hit):
                break
        selected[cell_offset : cell_offset + len(cells)] = hit
    return selected


def _split_integer_records(records: np.ndarray) -> np.ndarray:
    children: list[np.ndarray] = []
    for record in records:
        widths = record[[1, 3, 5]] - record[[0, 2, 4]]
        if np.any(widths <= 1) or not np.all(widths == widths[0]):
            raise RuntimeError("targeted refinement received a non-octree cell")
        child_units = int(widths[0] // 2)
        start = record[[0, 2, 4]]
        child_starts = _split_starts(start[None, :], child_units)
        ends = child_starts + child_units
        children.append(
            np.column_stack(
                (
                    child_starts[:, 0], ends[:, 0],
                    child_starts[:, 1], ends[:, 1],
                    child_starts[:, 2], ends[:, 2],
                )
            ).astype(np.int32, copy=False)
        )
    return np.concatenate(children, axis=0) if children else np.empty((0, 6), dtype=np.int32)


def _targeted_refine_cells(
    source_triangles: np.ndarray,
    selected_records: np.ndarray,
    *,
    root_min: np.ndarray,
    leaf_size: np.ndarray,
    parameters: OctreeParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine selected unresolved cells with the unchanged subset inequality."""

    source = SignedFcpwMesh(source_triangles)
    frontier = np.ascontiguousarray(selected_records, dtype=np.int32)
    accepted_parts: list[np.ndarray] = []
    margin_parts: list[np.ndarray] = []
    unresolved_parts: list[np.ndarray] = []
    iterations: list[dict[str, Any]] = []
    for iteration in range(parameters.maximum_depth + 2):
        if not len(frontier):
            break
        if len(frontier) > parameters.max_targeted_frontier_cells:
            raise RuntimeError("targeted refinement exceeded its fail-closed cell cap")
        lows = root_min + frontier[:, (0, 2, 4)] * leaf_size
        highs = root_min + frontier[:, (1, 3, 5)] * leaf_size
        centers = (lows + highs) / 2.0
        half_diagonals = np.linalg.norm((highs - lows) / 2.0, axis=1)
        maximum_edges = np.max(highs - lows, axis=1)
        inside, distance_lower, _ = source.query(
            centers,
            chunk_size=parameters.query_chunk_size,
            numeric_guard_mm=parameters.fcpw_float_guard_mm,
        )
        margins = (
            distance_lower
            - half_diagonals
            - parameters.source_faceting_bound_mm
        )
        safe = margins > 0.0
        accept = safe & inside
        outside = safe & ~inside
        terminal = (~safe) & (
            maximum_edges <= parameters.targeted_max_cell_edge_mm + 1.0e-12
        )
        refine = ~(accept | outside | terminal)
        if np.any(accept):
            accepted_parts.append(frontier[accept])
            margin_parts.append(margins[accept])
        if np.any(terminal):
            unresolved_parts.append(frontier[terminal])
        iterations.append(
            {
                "iteration": iteration,
                "evaluated_cell_count": len(frontier),
                "accepted_cell_count": int(np.count_nonzero(accept)),
                "outside_cell_count": int(np.count_nonzero(outside)),
                "terminal_unresolved_cell_count": int(np.count_nonzero(terminal)),
                "refined_cell_count": int(np.count_nonzero(refine)),
                "maximum_cell_edge_mm": float(np.max(maximum_edges)),
            }
        )
        frontier = _split_integer_records(frontier[refine])
    if len(frontier):
        raise RuntimeError("targeted refinement did not close")
    accepted = (
        np.concatenate(accepted_parts, axis=0)
        if accepted_parts
        else np.empty((0, 6), dtype=np.int32)
    )
    margins = (
        np.concatenate(margin_parts)
        if margin_parts
        else np.empty(0, dtype=np.float64)
    )
    unresolved = (
        np.concatenate(unresolved_parts, axis=0)
        if unresolved_parts
        else np.empty((0, 6), dtype=np.int32)
    )
    return accepted, margins, unresolved, {
        "method": "targeted_isotropic_octree_same_signed_distance_Lipschitz_subset",
        "target_maximum_cell_edge_mm": parameters.targeted_max_cell_edge_mm,
        "input_unresolved_cell_count": len(selected_records),
        "new_accepted_cell_count": len(accepted),
        "terminal_unresolved_cell_count": len(unresolved),
        "iterations": iterations,
        "passed": True,
    }


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

    first_merged, first_merged_margins, first_merge_record = greedy_merge_boxes(
        accepted, margins
    )
    first_boxes_mm = _integer_boxes_to_mm(first_merged, root_min, leaf_size)
    first_unresolved_mm = _integer_boxes_to_mm(unresolved, root_min, leaf_size)
    targeted_passes: list[dict[str, Any]] = []
    total_selected_unresolved = 0
    merged, merged_margins, merge_record = (
        first_merged, first_merged_margins, first_merge_record
    )
    boxes_mm = first_boxes_mm
    final_surface_scan: dict[str, Any] | None = None

    # Close the broad, cheap finite-witness reds before asking the complete
    # adaptive surface partition to spend its bounded patch budget near the
    # final 0.34 mm frontier.  This pass is explicitly not clearance authority.
    seed_patches, seed_record = _finite_certificate_red_seed_patches(
        triangles, boxes_mm, parameters
    )
    seed_pass: dict[str, Any] = {
        "pass_index": 0,
        "mode": "finite_certificate_red_seed_prepass",
        "seed_record": seed_record,
    }
    if len(seed_patches):
        selected_mask = _cells_near_triangle_patches(
            first_unresolved_mm,
            seed_patches,
            parameters.targeted_refinement_radius_mm,
        )
        selected_count = int(np.count_nonzero(selected_mask))
        if selected_count == 0:
            raise RuntimeError("finite red seed patches selected no octree cells")
        total_selected_unresolved += selected_count
        targeted_accepted, targeted_margins, targeted_unresolved, refinement = (
            _targeted_refine_cells(
                triangles,
                unresolved[selected_mask],
                root_min=root_min,
                leaf_size=leaf_size,
                parameters=parameters,
            )
        )
        if not len(targeted_accepted):
            raise RuntimeError("finite seed pass added no exact-subset material")
        accepted = np.concatenate((accepted, targeted_accepted), axis=0)
        margins = np.concatenate((margins, targeted_margins))
        unresolved = np.concatenate(
            (unresolved[~selected_mask], targeted_unresolved), axis=0
        )
        merged, merged_margins, merge_record = greedy_merge_boxes(accepted, margins)
        if len(merged) > parameters.max_proxy_boxes_after_merge:
            raise RuntimeError(
                f"targeted proxy has {len(merged)} boxes, cap is "
                f"{parameters.max_proxy_boxes_after_merge}"
            )
        boxes_mm = _integer_boxes_to_mm(merged, root_min, leaf_size)
        seed_pass["selected_unresolved_cell_count"] = selected_count
        seed_pass["refinement"] = refinement
        seed_pass["proxy_box_count_after_pass"] = len(merged)
    else:
        seed_pass["selected_unresolved_cell_count"] = 0
        seed_pass["refinement"] = {
            "method": "not_required_no_finite_certificate_red_seed",
            "passed": True,
        }
    targeted_passes.append(seed_pass)

    for adaptive_index in range(6):
        pass_index = adaptive_index + 1
        refinement_patches, surface_scan = _scan_source_patches_for_undercoverage(
            triangles, boxes_mm, parameters
        )
        pass_record: dict[str, Any] = {
            "pass_index": pass_index,
            "mode": "complete_adaptive_surface_partition",
            "surface_scan": surface_scan,
        }
        if not len(refinement_patches):
            final_surface_scan = surface_scan
            pass_record["selected_unresolved_cell_count"] = 0
            pass_record["refinement"] = {
                "method": "not_required_complete_surface_scan_green",
                "passed": True,
            }
            targeted_passes.append(pass_record)
            break
        unresolved_mm_current = _integer_boxes_to_mm(
            unresolved, root_min, leaf_size
        )
        selected_mask = _cells_near_triangle_patches(
            unresolved_mm_current,
            refinement_patches,
            parameters.targeted_refinement_radius_mm,
        )
        selected_count = int(np.count_nonzero(selected_mask))
        if selected_count == 0:
            raise RuntimeError("surface refinement patches selected no octree cells")
        total_selected_unresolved += selected_count
        targeted_accepted, targeted_margins, targeted_unresolved, refinement = (
            _targeted_refine_cells(
                triangles,
                unresolved[selected_mask],
                root_min=root_min,
                leaf_size=leaf_size,
                parameters=parameters,
            )
        )
        if not len(targeted_accepted):
            raise RuntimeError("targeted pass added no exact-subset material")
        accepted = np.concatenate((accepted, targeted_accepted), axis=0)
        margins = np.concatenate((margins, targeted_margins))
        unresolved = np.concatenate(
            (unresolved[~selected_mask], targeted_unresolved), axis=0
        )
        merged, merged_margins, merge_record = greedy_merge_boxes(accepted, margins)
        if len(merged) > parameters.max_proxy_boxes_after_merge:
            raise RuntimeError(
                f"targeted proxy has {len(merged)} boxes, cap is "
                f"{parameters.max_proxy_boxes_after_merge}"
            )
        boxes_mm = _integer_boxes_to_mm(merged, root_min, leaf_size)
        pass_record["selected_unresolved_cell_count"] = selected_count
        pass_record["refinement_patch_geometry_sha256"] = hashlib.sha256(
            np.ascontiguousarray(refinement_patches, dtype="<f8").tobytes()
        ).hexdigest()
        pass_record["refinement"] = refinement
        pass_record["proxy_box_count_after_pass"] = len(merged)
        targeted_passes.append(pass_record)
    if final_surface_scan is None:
        raise RuntimeError("targeted refinement did not close source undercoverage")

    if len(merged) > parameters.max_proxy_boxes_after_merge:
        raise RuntimeError(
            f"merged proxy has {len(merged)} boxes, cap is "
            f"{parameters.max_proxy_boxes_after_merge}"
        )
    boxes_mm = _integer_boxes_to_mm(merged, root_min, leaf_size)
    unresolved_mm = _integer_boxes_to_mm(unresolved, root_min, leaf_size)
    unresolved_centers = np.mean(unresolved_mm, axis=1)
    unresolved_half_diagonals = np.linalg.norm(
        (unresolved_mm[:, 1] - unresolved_mm[:, 0]) / 2.0, axis=1
    )
    _, _, unresolved_distance_upper = signed_mesh.query(
        unresolved_centers,
        chunk_size=parameters.query_chunk_size,
        numeric_guard_mm=parameters.fcpw_float_guard_mm,
    )
    boundary_bounds = (
        unresolved_distance_upper
        + unresolved_half_diagonals
        + parameters.source_faceting_bound_mm
    )
    if np.any(boundary_bounds > parameters.boundary_threshold_mm + 1.0e-12):
        raise RuntimeError("targeted refinement weakened proxy boundary coverage")
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
        "first_pass": {
            "accepted_cell_count_before_merge": first_merge_record[
                "initial_cell_count"
            ],
            "proxy_box_count": len(first_merged),
            "minimum_exact_subset_margin_mm": float(
                np.min(first_merged_margins)
            ),
            "merge": first_merge_record,
        },
        "targeted_refinement_passes": targeted_passes,
        "final_complete_source_surface_scan": final_surface_scan,
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
        "first_pass_unresolved_boundary_cell_count": len(first_unresolved_mm),
        "targeted_selected_unresolved_cell_count": total_selected_unresolved,
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
            "max_targeted_frontier_cells": parameters.max_targeted_frontier_cells,
            "max_adaptive_surface_patches": parameters.max_adaptive_surface_patches,
            "maximum_targeted_pass_count_including_seed_prepass": 7,
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


def _split_triangle_longest_edge(triangles: np.ndarray) -> np.ndarray:
    """Bisect each patch on its longest edge without four-way skinny growth."""

    result = np.empty((2 * len(triangles), 3, 3), dtype=np.float64)
    for index, triangle in enumerate(triangles):
        a, b, c = triangle
        lengths = (
            float(np.linalg.norm(b - a)),
            float(np.linalg.norm(c - b)),
            float(np.linalg.norm(a - c)),
        )
        edge = int(np.argmax(lengths))
        if edge == 0:
            midpoint = (a + b) / 2.0
            first, second = (a, midpoint, c), (midpoint, b, c)
        elif edge == 1:
            midpoint = (b + c) / 2.0
            first, second = (b, midpoint, a), (midpoint, c, a)
        else:
            midpoint = (c + a) / 2.0
            first, second = (c, midpoint, b), (midpoint, a, b)
        result[2 * index] = first
        result[2 * index + 1] = second
    return np.ascontiguousarray(result)


def _certificate_from_surface_scan(
    scan: dict[str, Any], parameters: OctreeParameters
) -> dict[str, Any]:
    """Normalise a complete adaptive scan into the published certificate."""

    certified = float(scan["maximum_certified_safe_upper_mm"])
    return {
        "direction": "exact_STEP_boundary_to_proxy_box_union",
        "method": scan["method"],
        "internal_acceptance_bound_mm": parameters.internal_boundary_target_mm,
        "release_threshold_mm": parameters.boundary_threshold_mm,
        "certified_safe_patch_count": scan["certified_safe_patch_count"],
        "certificate_red_patch_count": scan["certificate_red_patch_count"],
        "unresolved_refinement_seed_patch_count": scan[
            "unresolved_refinement_seed_patch_count"
        ],
        "source_faceting_bound_mm": parameters.source_faceting_bound_mm,
        "certified_upper_bound_mm": certified,
        "complete_source_surface_partition": scan["complete_surface_partition"],
        "surface_partition_sha256": scan["partition_sha256"],
        "iteration_records": scan["iteration_records"],
        "finite_witness_maximum_is_not_the_bound": True,
        "passed": bool(
            int(scan["certificate_red_patch_count"]) == 0
            and int(scan["unresolved_refinement_seed_patch_count"]) == 0
            and scan["passed"]
            and certified <= parameters.internal_boundary_target_mm + 1.0e-12
            and certified <= parameters.boundary_threshold_mm + 1.0e-12
        ),
    }


def certify_source_to_proxy_surface(
    source_triangles: np.ndarray,
    boxes_mm: np.ndarray,
    parameters: OctreeParameters,
) -> dict[str, Any]:
    """Adaptively cover every source triangle under the actual distance field.

    Each leaf patch publishes ``distance(centroid, proxy) + patch_radius +
    STEP_faceting``.  Point-to-set distance is 1-Lipschitz, so this is a bound
    for every point of that complete patch, not a finite-sample claim.  Only
    ambiguous patches are bisected, and longest-edge bisection avoids the
    former four-way explosion on skinny OCCT triangles.
    """

    _, scan = _scan_source_patches_for_undercoverage(
        source_triangles, boxes_mm, parameters
    )
    return _certificate_from_surface_scan(scan, parameters)


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
    source_to_proxy = _certificate_from_surface_scan(
        construction["exact_subset"]["final_complete_source_surface_scan"],
        params,
    )
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
            "internal_boundary_target_mm": params.internal_boundary_target_mm,
            "targeted_seed_distance_mm": params.targeted_seed_distance_mm,
            "targeted_refinement_radius_mm": params.targeted_refinement_radius_mm,
            "targeted_max_cell_edge_mm": params.targeted_max_cell_edge_mm,
            "max_targeted_frontier_cells": params.max_targeted_frontier_cells,
            "max_adaptive_surface_patches": params.max_adaptive_surface_patches,
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
