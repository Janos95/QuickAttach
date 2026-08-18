#!/usr/bin/env python3
"""Fail-closed validator API for matcha runtime-collision certificates.

No release report is emitted at this recovery checkpoint.  The validator is
nevertheless geometry-backed: callers must provide the hash-pinned source and
runtime triangle inputs, and the certificate is accepted only when a fresh
canonical recomputation exactly matches every evidence record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any


def _load_authority() -> Any:
    module_name = "_matcha_payload_authority_canonical_validator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("generate_matcha_payload_proxy_report.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import collision authority: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


authority = _load_authority()


_DIRECTION_FIELDS = (
    "witness_maximum_mm",
    "query_surface_covering_radius_mm",
    "query_faceting_error_upper_bound_mm",
    "target_faceting_error_upper_bound_mm",
    "certified_upper_bound_mm",
)


def _validate_direction_schema(direction: Any, *, label: str) -> None:
    if not isinstance(direction, dict):
        raise ValueError(f"{label} must be a mapping")
    values: dict[str, float] = {}
    for field in _DIRECTION_FIELDS:
        value = float(direction[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label}.{field} must be finite and nonnegative")
        values[field] = value
    recomputed = math.fsum(values[field] for field in _DIRECTION_FIELDS[:-1])
    if abs(recomputed - values["certified_upper_bound_mm"]) > 1.0e-12:
        raise ValueError(f"{label} certified-bound arithmetic drifted")
    witness_sha256 = str(direction.get("witness_set_sha256", ""))
    if len(witness_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in witness_sha256
    ):
        raise ValueError(f"{label} has an invalid witness digest")
    if direction.get("float64_candidate_replay") is not True:
        raise ValueError(f"{label} did not replay the original float64 triangles")


def _validate_topology_schema(topology: Any, *, label: str) -> None:
    if not isinstance(topology, dict):
        raise ValueError(f"{label} must be a mapping")
    for field in ("watertight", "orientation_consistent", "positive_volume", "passed"):
        if not isinstance(topology.get(field), bool):
            raise ValueError(f"{label}.{field} must be boolean")


def validate_bidirectional_runtime_collision_certificate(
    certificate: Any,
    source_triangles: Any,
    proxy_piece_triangles: Any,
    *,
    threshold_mm: float,
    source_error_mm: float,
    proxy_error_mm: float,
) -> dict[str, Any]:
    """Recompute and canonical-compare one synthetic or component certificate."""

    if not isinstance(certificate, dict):
        raise ValueError("certificate must be a mapping")
    if certificate.get("release_ready") is not False:
        raise ValueError("component/development certificates cannot publish release readiness")
    _validate_topology_schema(certificate.get("source_topology"), label="source_topology")
    _validate_topology_schema(certificate.get("proxy_topology"), label="proxy_topology")
    occupancy = certificate.get("occupancy")
    if not isinstance(occupancy, dict):
        raise ValueError("occupancy must be a mapping")
    if occupancy.get("signed") is not True or occupancy.get("union_occupancy") is not True:
        raise ValueError("occupancy must use signed union semantics")
    tolerance = float(occupancy["signed_distance_tolerance_mm"])
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0e-6:
        raise ValueError("signed-distance tolerance is invalid")
    for label in ("source_to_proxy", "proxy_to_source"):
        _validate_direction_schema(certificate.get(label), label=label)

    canonical = authority.certify_bidirectional_runtime_collision(
        source_triangles,
        proxy_piece_triangles,
        threshold_mm=threshold_mm,
        source_error_mm=source_error_mm,
        proxy_error_mm=proxy_error_mm,
    )
    if certificate != canonical:
        raise ValueError("certificate differs from canonical geometry recomputation")
    return canonical


def validate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Stable short alias used by report tooling and independent gates."""

    return validate_bidirectional_runtime_collision_certificate(*args, **kwargs)


__all__ = ["validate", "validate_bidirectional_runtime_collision_certificate"]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_runtime_payload_report(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report_not_mapping"]
    observed = dict(report)
    advertised = observed.pop("canonical_sha256_without_this_field", None)
    if advertised != _canonical_sha256(observed):
        errors.append("canonical_sha256")
    if report.get("release_ready") is not False:
        errors.append("release_ready")
    if report.get("passed") is not True:
        errors.append("passed")
    try:
        expected = authority.build_runtime_payload_report()
    except Exception as exc:
        return [*errors, f"recompute:{type(exc).__name__}:{exc}"]
    if report != expected:
        errors.append("canonical_recomputation")
    groups = report.get("groups")
    if not isinstance(groups, list) or not groups:
        errors.append("groups")
    else:
        names = [
            str(name)
            for group in groups
            if isinstance(group, dict)
            for name in group.get("active_geom_names", [])
        ]
        if names != [
            name
            for group in groups
            for name in sorted(group.get("active_geom_names", []))
        ]:
            errors.append("group_name_order")
        if len(names) != len(set(names)):
            errors.append("duplicate_active_geom")
        if report.get("active_geom_count") != len(names):
            errors.append("active_geom_count")
        if report.get("active_geom_inventory_sha256") != _canonical_sha256(names):
            errors.append("active_geom_inventory_sha256")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate_runtime_payload_report(report)
    print(json.dumps({"report": str(args.report), "errors": errors, "passed": not errors}, sort_keys=True))
    return 0 if not errors else 1


__all__.extend(["validate_runtime_payload_report"])


if __name__ == "__main__":
    raise SystemExit(main())
