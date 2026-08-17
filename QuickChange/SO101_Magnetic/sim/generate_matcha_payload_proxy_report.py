#!/usr/bin/env python3
"""OCCT-free runtime-collision fidelity certificates for the matcha payloads.

This recovery checkpoint intentionally exposes the small, independently
testable geometry kernel before it emits a repository report.  STEP import is
performed elsewhere with an absolute-deflection tessellator; this module sees
only hash-pinned triangle surfaces and their proven faceting-error bounds.

FCPW is used only to select closest-triangle candidates.  Every published
distance is replayed against the original triangle coordinates in float64, so
float32 acceleration can conservatively overestimate but can never create a
false-small clearance/fidelity witness.  Runtime proxy Boolean union or OCCT
containment is not part of this fast authority.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import fcpw
import numpy as np


FCPW_REQUIRED_VERSION = "1.2.0"
SIGNED_DISTANCE_TOLERANCE_MM = 1.0e-6
VERTEX_WELD_TOLERANCE_MM = 1.0e-9
MAX_SUBDIVIDED_TRIANGLES = 2_000_000
_MIN_DOUBLED_TRIANGLE_AREA_MM2 = 1.0e-15


if importlib.metadata.version("fcpw") != FCPW_REQUIRED_VERSION:
    raise RuntimeError(
        f"fcpw {FCPW_REQUIRED_VERSION} is required for deterministic candidates"
    )


@dataclass(frozen=True)
class AbsoluteStepMeshAuthority:
    """Hash-pinned provenance for an absolutely tessellated STEP feature.

    ``absolute_deflection_mm`` is the independently measured OCCT/STEP mesh
    error supplied to :func:`certify_bidirectional_runtime_collision`.  The
    selector and selected-geometry hashes make feature scoping unambiguous;
    no compiled/recentered runtime frame is accepted as source authority.
    """

    source_artifact_sha256: str
    source_selector_sha256: str
    selected_feature_geometry_sha256: str
    absolute_deflection_mm: float

    def __post_init__(self) -> None:
        for name in (
            "source_artifact_sha256",
            "source_selector_sha256",
            "selected_feature_geometry_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid {name}")
        _finite_nonnegative(self.absolute_deflection_mm, "absolute_deflection_mm")


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validated_triangles(value: Any, *, label: str = "triangles") -> np.ndarray:
    triangles = np.asarray(value, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"{label} must have finite shape (N, 3, 3)")
    if len(triangles) == 0:
        raise ValueError(f"{label} cannot be empty")
    if not np.all(np.isfinite(triangles)):
        raise ValueError(f"{label} contains a nonfinite coordinate")
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(doubled_areas <= _MIN_DOUBLED_TRIANGLE_AREA_MM2):
        bad = np.flatnonzero(doubled_areas <= _MIN_DOUBLED_TRIANGLE_AREA_MM2)
        raise ValueError(f"{label} contains degenerate triangles at {bad[:8].tolist()}")
    return np.ascontiguousarray(triangles, dtype=np.float64)


def _point_triangle_distance_float64(point: np.ndarray, triangle: np.ndarray) -> float:
    """Ericson point/triangle distance evaluated only in original float64."""

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
        fraction = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + fraction * ab)))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + fraction * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + fraction * (c - b))))
    normal = np.cross(ab, ac)
    return abs(float(normal @ ap)) / float(np.linalg.norm(normal))


class _FcpwTriangleUpperBoundIndex:
    """Batched FCPW candidates with conservative original-float64 replay."""

    def __init__(self, triangles: Any) -> None:
        self.triangles = _validated_triangles(triangles)
        vertices = np.asfortranarray(self.triangles.reshape(-1, 3), dtype=np.float32)
        indices = np.asfortranarray(
            np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        )
        scene = fcpw.scene_3D()
        scene.set_object_count(1)
        scene.set_object_vertices(vertices, 0)
        scene.set_object_triangles(indices, 0)
        scene.build(fcpw.aggregate_type.bvh_surface_area, False, False, False)
        self._scene = scene

    def candidate_triangle_indices(self, points: Any) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query points must have shape (N, 3)")
        if not np.all(np.isfinite(query)):
            raise ValueError("query points contain a nonfinite coordinate")
        if len(query) == 0:
            return np.empty(0, dtype=np.int64)
        query32 = np.asfortranarray(query, dtype=np.float32)
        squared_max_radii = np.full(
            len(query32), np.finfo(np.float32).max, dtype=np.float32
        )
        interactions = fcpw.interaction_3D_list()
        self._scene.find_closest_points(
            query32, squared_max_radii, interactions, False
        )
        if len(interactions) != len(query):
            raise RuntimeError("FCPW did not return one candidate per query")
        candidates = np.asarray(
            [int(interaction.primitive_index) for interaction in interactions],
            dtype=np.int64,
        )
        if np.any(candidates < 0) or np.any(candidates >= len(self.triangles)):
            raise RuntimeError("FCPW returned an invalid closest-triangle candidate")
        return candidates

    def distances(self, points: Any) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        candidates = self.candidate_triangle_indices(query)
        distances = np.fromiter(
            (
                _point_triangle_distance_float64(point, self.triangles[candidate])
                for point, candidate in zip(query, candidates, strict=True)
            ),
            dtype=np.float64,
            count=len(query),
        )
        if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
            raise RuntimeError("float64 candidate replay produced an invalid distance")
        return distances


def _normalise_proxy_pieces(proxy_piece_triangles: Any) -> tuple[np.ndarray, ...]:
    if isinstance(proxy_piece_triangles, np.ndarray):
        array = np.asarray(proxy_piece_triangles)
        if array.ndim == 3:
            return (_validated_triangles(array, label="proxy triangles"),)
        if array.ndim == 4 and array.shape[2:] == (3, 3):
            return tuple(
                _validated_triangles(piece, label=f"proxy piece {index}")
                for index, piece in enumerate(array)
            )
        raise ValueError("proxy triangles must be (N,3,3) or a sequence of pieces")
    try:
        pieces = tuple(proxy_piece_triangles)
    except TypeError as error:
        raise ValueError("proxy pieces must be an iterable of triangle arrays") from error
    if not pieces:
        raise ValueError("proxy pieces cannot be empty")
    return tuple(
        _validated_triangles(piece, label=f"proxy piece {index}")
        for index, piece in enumerate(pieces)
    )


def _vertex_key(vertex: np.ndarray) -> tuple[int, int, int]:
    scaled = np.rint(np.asarray(vertex, dtype=np.float64) / VERTEX_WELD_TOLERANCE_MM)
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > np.iinfo(np.int64).max):
        raise ValueError("vertex cannot be represented by topology quantisation")
    return tuple(int(value) for value in scaled)


def _topology_index(
    triangles: np.ndarray,
) -> tuple[
    dict[tuple[int, int, int], int],
    dict[tuple[int, int], list[int]],
    dict[tuple[int, int], int],
]:
    vertex_ids: dict[tuple[int, int, int], int] = {}
    edge_triangles: dict[tuple[int, int], list[int]] = {}
    edge_orientations: dict[tuple[int, int], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        ids: list[int] = []
        for vertex in triangle:
            key = _vertex_key(vertex)
            if key not in vertex_ids:
                vertex_ids[key] = len(vertex_ids)
            ids.append(vertex_ids[key])
        if len(set(ids)) != 3:
            raise ValueError("topology quantisation collapsed a triangle")
        for first, second in zip(ids, (ids[1], ids[2], ids[0]), strict=True):
            edge = (min(first, second), max(first, second))
            edge_triangles.setdefault(edge, []).append(triangle_index)
            edge_orientations[edge] = edge_orientations.get(edge, 0) + (
                1 if first < second else -1
            )
    return vertex_ids, edge_triangles, edge_orientations


def _triangle_connected_components(triangles: np.ndarray) -> tuple[np.ndarray, ...]:
    _, edge_triangles, _ = _topology_index(triangles)
    adjacency: list[set[int]] = [set() for _ in range(len(triangles))]
    for incident in edge_triangles.values():
        for first in incident:
            adjacency[first].update(second for second in incident if second != first)
    remaining = set(range(len(triangles)))
    components: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        pending = [seed]
        component: list[int] = []
        remaining.remove(seed)
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    pending.append(neighbour)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return tuple(components)


def _shell_interior_point(
    triangles: np.ndarray, index: _FcpwTriangleUpperBoundIndex
) -> np.ndarray:
    """Find a deterministic parity-interior witness independent of winding."""

    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        shortest_edge = min(
            float(np.linalg.norm(triangle[(edge + 1) % 3] - triangle[edge]))
            for edge in range(3)
        )
        base_offset = max(1.0e-4, min(1.0e-3, shortest_edge * 1.0e-3))
        centroid = np.mean(triangle, axis=0)
        for multiplier in (1.0, 4.0, 16.0):
            offset = base_offset * multiplier
            minus = np.ascontiguousarray(centroid - offset * normal, dtype=np.float32)
            plus = np.ascontiguousarray(centroid + offset * normal, dtype=np.float32)
            minus_inside = bool(index._scene.contains(minus))
            plus_inside = bool(index._scene.contains(plus))
            if minus_inside != plus_inside:
                return np.asarray(minus if minus_inside else plus, dtype=np.float64)
    raise RuntimeError("could not establish a signed containment side for a shell")


def _shell_nesting_records(
    triangles: np.ndarray,
) -> tuple[tuple[np.ndarray, _FcpwTriangleUpperBoundIndex, np.ndarray, int], ...]:
    components = _triangle_connected_components(triangles)
    prepared: list[tuple[np.ndarray, _FcpwTriangleUpperBoundIndex, np.ndarray]] = []
    for component in components:
        shell = np.ascontiguousarray(triangles[component])
        index = _FcpwTriangleUpperBoundIndex(shell)
        prepared.append((shell, index, _shell_interior_point(shell, index)))
    records: list[tuple[np.ndarray, _FcpwTriangleUpperBoundIndex, np.ndarray, int]] = []
    for shell_index, (shell, index, interior) in enumerate(prepared):
        depth = sum(
            bool(other_index._scene.contains(np.ascontiguousarray(interior, dtype=np.float32)))
            for other_shell_index, (_, other_index, _) in enumerate(prepared)
            if other_shell_index != shell_index
        )
        records.append((shell, index, interior, int(depth)))
    return tuple(records)


def _surface_topology(triangles: np.ndarray) -> dict[str, Any]:
    vertex_ids, edge_triangles, edge_orientations = _topology_index(triangles)
    watertight = bool(edge_triangles) and all(
        len(incident) == 2 for incident in edge_triangles.values()
    )
    orientation_consistent = watertight and all(
        orientation == 0 for orientation in edge_orientations.values()
    )
    shell_records = _shell_nesting_records(triangles)
    component_volumes: list[float] = []
    component_depths: list[int] = []
    component_expected_signs: list[int] = []
    component_parity_matches: list[bool] = []
    for component_triangles, _, _, depth in shell_records:
        signed_volume = math.fsum(
            float(a @ np.cross(b, c)) / 6.0 for a, b, c in component_triangles
        )
        extent = np.ptp(component_triangles.reshape(-1, 3), axis=0)
        volume_tolerance = max(1.0e-15, float(np.prod(extent)) * 1.0e-12)
        component_volumes.append(float(signed_volume))
        expected_sign = 1 if depth % 2 == 0 else -1
        component_depths.append(depth)
        component_expected_signs.append(expected_sign)
        component_parity_matches.append(
            bool(
                math.isfinite(signed_volume)
                and expected_sign * signed_volume > volume_tolerance
            )
        )
    positive_volume = bool(component_parity_matches) and all(component_parity_matches)
    signed_volume = math.fsum(component_volumes)
    if not math.isfinite(signed_volume) or signed_volume <= 1.0e-15:
        positive_volume = False
    passed = bool(watertight and orientation_consistent and positive_volume)
    return {
        "watertight": watertight,
        "orientation_consistent": orientation_consistent,
        "positive_volume": positive_volume,
        "signed_volume_mm3": float(signed_volume),
        "connected_component_count": len(shell_records),
        "component_signed_volumes_mm3": component_volumes,
        "component_nesting_depths": component_depths,
        "component_expected_orientation_signs": component_expected_signs,
        "component_orientation_matches_nesting_parity": component_parity_matches,
        "unique_vertex_count": len(vertex_ids),
        "unique_edge_count": len(edge_triangles),
        "passed": passed,
    }


def _combined_topology(pieces: Iterable[np.ndarray]) -> dict[str, Any]:
    records = [_surface_topology(piece) for piece in pieces]
    return {
        "watertight": all(bool(record["watertight"]) for record in records),
        "orientation_consistent": all(
            bool(record["orientation_consistent"]) for record in records
        ),
        "positive_volume": all(bool(record["positive_volume"]) for record in records),
        "signed_volume_mm3": math.fsum(
            float(record["signed_volume_mm3"]) for record in records
        ),
        "piece_count": len(records),
        "passed": all(bool(record["passed"]) for record in records),
    }


def _signed_piece_occupancy_passes(triangles: np.ndarray) -> bool:
    """Check both material sides of every shell against the full parity union."""

    shell_records = _shell_nesting_records(triangles)

    def material_contains(point: np.ndarray) -> bool:
        query = np.ascontiguousarray(point, dtype=np.float32)
        return bool(
            sum(bool(index._scene.contains(query)) for _, index, _, _ in shell_records)
            % 2
        )

    for component_triangles, _, _, _ in shell_records:
        sample_indices = np.linspace(
            0,
            len(component_triangles) - 1,
            min(12, len(component_triangles)),
            dtype=int,
        )
        for triangle_index in sample_indices:
            triangle = component_triangles[int(triangle_index)]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= np.linalg.norm(normal)
            shortest_edge = min(
                float(np.linalg.norm(triangle[(edge + 1) % 3] - triangle[edge]))
                for edge in range(3)
            )
            offset = max(1.0e-4, min(1.0e-3, shortest_edge * 1.0e-3))
            centroid = np.mean(triangle, axis=0)
            inward = np.ascontiguousarray(centroid - offset * normal, dtype=np.float32)
            outward = np.ascontiguousarray(centroid + offset * normal, dtype=np.float32)
            if not material_contains(inward):
                return False
            if material_contains(outward):
                return False
    return True


def _signed_union_occupancy(
    source_pieces: tuple[np.ndarray, ...], proxy_pieces: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    source_passed = all(_signed_piece_occupancy_passes(piece) for piece in source_pieces)
    proxy_passed = all(_signed_piece_occupancy_passes(piece) for piece in proxy_pieces)
    return {
        "signed": True,
        "union_occupancy": True,
        "signed_distance_tolerance_mm": SIGNED_DISTANCE_TOLERANCE_MM,
        "source_signed_witnesses_passed": source_passed,
        "proxy_signed_witnesses_passed": proxy_passed,
        "functional_hole_authority": "bidirectional_boundary_and_signed_union",
        "passed": bool(source_passed and proxy_passed),
    }


def _subdivided_surface_witnesses(
    triangles: np.ndarray, requested_covering_radius_mm: float
) -> tuple[np.ndarray, float]:
    requested = max(1.0e-6, _finite_nonnegative(requested_covering_radius_mm, "covering radius"))
    cells = np.ascontiguousarray(triangles, dtype=np.float64)
    while True:
        centroids = np.mean(cells, axis=1)
        radii = np.max(np.linalg.norm(cells - centroids[:, None, :], axis=2), axis=1)
        selected = radii > requested
        if not np.any(selected):
            return np.ascontiguousarray(centroids), float(np.max(radii))
        selected_count = int(np.count_nonzero(selected))
        new_count = len(cells) + 3 * selected_count
        if new_count > MAX_SUBDIVIDED_TRIANGLES:
            raise RuntimeError("surface witness subdivision exceeded the fail-closed cap")
        keep = cells[~selected]
        chosen = cells[selected]
        a, b, c = chosen[:, 0], chosen[:, 1], chosen[:, 2]
        ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
        split = np.concatenate(
            (
                np.stack((a, ab, ca), axis=1),
                np.stack((ab, b, bc), axis=1),
                np.stack((ca, bc, c), axis=1),
                np.stack((ab, bc, ca), axis=1),
            ),
            axis=0,
        )
        cells = np.ascontiguousarray(np.concatenate((keep, split), axis=0))


def _witness_sha256(points: np.ndarray, distances: np.ndarray) -> str:
    digest = hashlib.sha256()
    header = json.dumps(
        {"count": len(points), "point_shape": list(points.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(distances, dtype="<f8").tobytes())
    return digest.hexdigest()


def _directed_surface_certificate(
    query_triangles: np.ndarray,
    target_index: _FcpwTriangleUpperBoundIndex,
    *,
    threshold_mm: float,
    query_error_mm: float,
    target_error_mm: float,
) -> dict[str, Any]:
    available = max(0.0, threshold_mm - query_error_mm - target_error_mm)
    requested_cover = min(0.20, max(0.005, available * 0.50))
    witnesses, covering_radius = _subdivided_surface_witnesses(
        query_triangles, requested_cover
    )
    distances = target_index.distances(witnesses)
    witness_maximum = float(np.max(distances))
    certified = math.fsum(
        (witness_maximum, covering_radius, query_error_mm, target_error_mm)
    )
    return {
        "witness_maximum_mm": witness_maximum,
        "query_surface_covering_radius_mm": covering_radius,
        "query_faceting_error_upper_bound_mm": query_error_mm,
        "target_faceting_error_upper_bound_mm": target_error_mm,
        "certified_upper_bound_mm": certified,
        "witness_count": len(witnesses),
        "witness_set_sha256": _witness_sha256(witnesses, distances),
        "float64_candidate_replay": True,
        "passed": bool(certified <= threshold_mm + 1.0e-12),
    }


def certify_bidirectional_runtime_collision(
    source_triangles: Any,
    proxy_piece_triangles: Any,
    *,
    threshold_mm: float,
    source_error_mm: float,
    proxy_error_mm: float,
) -> dict[str, Any]:
    """Certify two directed boundary distances with signed/topology guards.

    The finite witness maximum is an observation, never the full-surface
    bound.  Each direction adds its physical query-cell covering radius and
    both absolute STEP/runtime mesh faceting errors.  Functional holes,
    dropped parts, and extra components are therefore fail-closed in either
    direction even when their aggregate volumes happen to look plausible.
    """

    threshold = _finite_nonnegative(threshold_mm, "threshold_mm")
    source_error = _finite_nonnegative(source_error_mm, "source_error_mm")
    proxy_error = _finite_nonnegative(proxy_error_mm, "proxy_error_mm")
    source = _validated_triangles(source_triangles, label="source triangles")
    proxy_pieces = _normalise_proxy_pieces(proxy_piece_triangles)
    proxy = np.ascontiguousarray(np.concatenate(proxy_pieces, axis=0))

    source_topology = _combined_topology((source,))
    proxy_topology = _combined_topology(proxy_pieces)
    occupancy = (
        _signed_union_occupancy((source,), proxy_pieces)
        if source_topology["passed"] and proxy_topology["passed"]
        else {
            "signed": True,
            "union_occupancy": True,
            "signed_distance_tolerance_mm": SIGNED_DISTANCE_TOLERANCE_MM,
            "source_signed_witnesses_passed": False,
            "proxy_signed_witnesses_passed": False,
            "functional_hole_authority": "bidirectional_boundary_and_signed_union",
            "passed": False,
        }
    )
    source_to_proxy = _directed_surface_certificate(
        source,
        _FcpwTriangleUpperBoundIndex(proxy),
        threshold_mm=threshold,
        query_error_mm=source_error,
        target_error_mm=proxy_error,
    )
    proxy_to_source = _directed_surface_certificate(
        proxy,
        _FcpwTriangleUpperBoundIndex(source),
        threshold_mm=threshold,
        query_error_mm=proxy_error,
        target_error_mm=source_error,
    )
    passed = bool(
        source_topology["passed"]
        and proxy_topology["passed"]
        and occupancy["passed"]
        and source_to_proxy["passed"]
        and proxy_to_source["passed"]
    )
    return {
        "schema_version": "1.0.0-fcpw-recovery",
        "method": "fcpw_candidates_float64_replay_bidirectional_boundary_cover",
        "release_ready": False,
        "threshold_mm": threshold,
        "source_topology": source_topology,
        "proxy_topology": proxy_topology,
        "occupancy": occupancy,
        "source_to_proxy": source_to_proxy,
        "proxy_to_source": proxy_to_source,
        "passed": passed,
    }


__all__ = [
    "AbsoluteStepMeshAuthority",
    "_FcpwTriangleUpperBoundIndex",
    "certify_bidirectional_runtime_collision",
]
