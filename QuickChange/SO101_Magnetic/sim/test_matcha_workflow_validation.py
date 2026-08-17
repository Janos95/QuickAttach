#!/usr/bin/env python3
"""Independent, fail-closed validation for the optional matcha workflow.

The matcha production files are restored independently from this test module.
Tests whose authority input is not present skip with an explicit reason; they
must become ordinary pass/fail gates as soon as that input lands.  This file
does not generate CAD, patch the scene, or mutate the controller.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
MAGNETIC_ROOT = HERE.parent
REPOSITORY_ROOT = HERE.parents[2]
MATCHA_DEMO = HERE / "matcha_workflow_demo.py"
MATCHA_SCENE = HERE / "matcha_workflow_scene.xml"
MATCHA_CONFIG = HERE / "matcha_tool_geometry.json"
PAYLOAD_GENERATOR = HERE / "generate_matcha_payload_proxy_report.py"
PAYLOAD_VALIDATOR = HERE / "validate_matcha_payload_proxy_report.py"
PAYLOAD_REPORT = HERE / "matcha_payload_proxy_report.json"
CAD_ROOT = MAGNETIC_ROOT / "matcha_tools"
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
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        raise unittest.SkipTest(
            f"{description} dependency is not installed yet: {error.name}"
        ) from error
    return module


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
            device = whisk.get("ttl_device_id")
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
        interface = manifest.get("interface")
        self.assertIsInstance(interface, dict)
        mismatch = interface.get("interface_match_mm3", {})
        if isinstance(mismatch, dict) and "absolute_mismatch_mm3" in mismatch:
            self.assertEqual(float(mismatch["absolute_mismatch_mm3"]), 0.0)

        for key, expected_id in (("matcha_spoon", 21), ("matcha_whisk", 22)):
            tool = tool_from_manifest(manifest, key)
            self.assertEqual(int(tool["tool_id"]), expected_id)
            components = tool.get("components")
            self.assertIsInstance(components, list)
            self.assertTrue(components, key)
            component_masses = [
                float(component["mass_g"])
                for component in components
                if isinstance(component, dict) and "mass_g" in component
            ]
            declared_mass = first_number(tool, "mass_g", "total_mass_g")
            if declared_mass is not None and component_masses:
                self.assertAlmostEqual(
                    math.fsum(component_masses), declared_mass, delta=1.0e-3
                )
            if declared_mass is not None:
                self.assertGreater(declared_mass, 0.0)
            com = tool.get("center_of_mass_mm", tool.get("com_mm"))
            if com is not None:
                self.assertEqual(len(com), 3)
                self.assertTrue(all(math.isfinite(float(value)) for value in com))

        records = list(iter_file_records(manifest.get("exports", {})))
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
        limitations = config.get("limitations")
        self.assertIsInstance(limitations, list)
        self.assertTrue(limitations)


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
            raise AssertionError("matcha controller class is missing")

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
            self.fail("matcha workflow must expose collision_coverage(model)")
        coverage = self.demo.collision_coverage(self.model)
        self.assertIsInstance(coverage, dict)
        complete = coverage.get("complete", coverage.get("collision_coverage_complete"))
        missing = coverage.get(
            "missing_collision_bodies", coverage.get("missing_bodies", [])
        )
        self.assertTrue(complete, coverage)
        self.assertEqual(missing, [], coverage)

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
        # Four closed bars form a 2 x 2 through-opening inside a 4 x 4 frame.
        frame = np.concatenate(
            (
                box_triangles((0.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
                box_triangles((0.0, 3.0, 0.0), (4.0, 4.0, 1.0)),
                box_triangles((0.0, 1.0, 0.0), (1.0, 3.0, 1.0)),
                box_triangles((3.0, 1.0, 0.0), (4.0, 3.0, 1.0)),
            )
        )
        filled = box_triangles((0.0, 0.0, 0.0), (4.0, 4.0, 1.0))
        certificate = self._certify(frame, filled)
        self.assertFalse(bool(certificate["passed"]), certificate)
        self.assertGreater(
            max(
                float(certificate["source_to_proxy"]["certified_upper_bound_mm"]),
                float(certificate["proxy_to_source"]["certified_upper_bound_mm"]),
            ),
            SURFACE_RELEASE_LIMIT_MM,
        )


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
