#!/usr/bin/env python3
"""Compact runtime-collision fidelity certificates for the matcha payloads.

This recovery checkpoint intentionally exposes the small, independently
testable geometry kernel before it emits a repository report.  STEP import is
performed elsewhere with an absolute-deflection tessellator; this module sees
only hash-pinned triangle surfaces and their proven faceting-error bounds.

FCPW is used only to select closest-triangle candidates. Every published
distance is replayed against the original triangle coordinates in float64, so
float32 acceleration can conservatively overestimate but can never create a
false-small fidelity witness. The contract is deliberately only a bounded,
bidirectional surface comparison; topology reconstruction, signed occupancy,
Boolean unions, and CAD-kernel containment are outside this verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcpw
import numpy as np


FCPW_REQUIRED_VERSION = "1.2.0"
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
    """Certify bounded surface distance in both directions.

    The finite witness maximum is an observation, never the full-surface
    bound. Each direction adds its query-cell covering radius and both mesh
    faceting-error bounds. Functional holes, dropped parts, and extra
    components are therefore fail-closed because one direction must cover
    every part of each surface.
    """

    threshold = _finite_nonnegative(threshold_mm, "threshold_mm")
    source_error = _finite_nonnegative(source_error_mm, "source_error_mm")
    proxy_error = _finite_nonnegative(proxy_error_mm, "proxy_error_mm")
    source = _validated_triangles(source_triangles, label="source triangles")
    proxy_pieces = _normalise_proxy_pieces(proxy_piece_triangles)
    proxy = np.ascontiguousarray(np.concatenate(proxy_pieces, axis=0))

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
    passed = bool(source_to_proxy["passed"] and proxy_to_source["passed"])
    return {
        "schema_version": "2.0.0-fcpw-surface-bound",
        "method": "fcpw_candidates_float64_replay_bidirectional_boundary_cover",
        "release_ready": False,
        "threshold_mm": threshold,
        "source_to_proxy": source_to_proxy,
        "proxy_to_source": proxy_to_source,
        "passed": passed,
    }


__all__ = [
    "AbsoluteStepMeshAuthority",
    "_FcpwTriangleUpperBoundIndex",
    "build_runtime_payload_report",
    "certify_bidirectional_runtime_collision",
]


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
DEFAULT_REPORT_PATH = HERE / "matcha_payload_proxy_report.json"
PAYLOAD_TOOLS = ("gripper", "spoon", "whisk")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(REPOSITORY_ROOT).as_posix(),
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _tool_active_geom_names(model: Any, tool: str) -> list[str]:
    root_id = int(model.body(f"tool_{tool}").id)
    names: list[str] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        ancestor = body_id
        while ancestor not in (0, root_id):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != root_id:
            continue
        if not (
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        ):
            continue
        name = str(model.geom(geom_id).name)
        if not name:
            raise RuntimeError(f"active payload geom {geom_id} has no name")
        names.append(name)
    if not names or len(names) != len(set(names)):
        raise RuntimeError(f"invalid active collision inventory for {tool}")
    return sorted(names)


def build_runtime_payload_report() -> dict[str, Any]:
    """Build the hash-pinned active payload collision inventory.

    The report deliberately remains a development authority.  It proves that
    every declared payload collision geom is active in the compiled model and
    uniquely assigned to one tool.  The bidirectional FCPW certificate kernel
    above remains the source-vs-proxy fidelity primitive; neither artifact is
    promoted to continuous swept-clearance or fabrication authority.
    """

    import matcha_workflow_demo as demo

    model = demo.build_model()
    groups = [
        {
            "name": tool,
            "tool_id": int(demo.ALL_TOOL_IDS[tool]),
            "root_body": f"tool_{tool}",
            "active_geom_names": _tool_active_geom_names(model, tool),
        }
        for tool in PAYLOAD_TOOLS
    ]
    all_names = [
        name for group in groups for name in group["active_geom_names"]
    ]
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("payload collision groups overlap")
    sources = [
        _file_record(HERE / "matcha_workflow_demo.py", "compiled_model_builder"),
        _file_record(
            HERE.parent / "matcha_tools" / "exports" / "matcha_tool_manifest.json",
            "matcha_CAD_manifest",
        ),
        _file_record(Path(__file__), "payload_certificate_generator"),
        _file_record(
            HERE / "validate_matcha_payload_proxy_report.py",
            "independent_report_validator",
        ),
    ]
    report: dict[str, Any] = {
        "schema_version": "1.0.0-matcha-payload-runtime-inventory",
        "method": "compiled_active_geom_inventory_plus_fcpw_certificate_kernel",
        "groups": groups,
        "active_geom_count": len(all_names),
        "active_geom_inventory_sha256": _canonical_sha256(all_names),
        "sources": sources,
        "source_records_sha256": _canonical_sha256(sources),
        "fcpw_required_version": FCPW_REQUIRED_VERSION,
        "runtime_proxy_is_release_clearance_authority": False,
        "continuous_swept_clearance_authority": False,
        "passed": True,
        "release_ready": False,
    }
    report["canonical_sha256_without_this_field"] = _canonical_sha256(report)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_runtime_payload_report()
    _write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "active_geom_count": report["active_geom_count"],
                "group_count": len(report["groups"]),
                "passed": report["passed"],
                "release_ready": report["release_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
