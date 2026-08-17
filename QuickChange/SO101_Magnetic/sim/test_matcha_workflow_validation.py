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
import struct
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable
from unittest import mock

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
CORE_CLEARANCE_VALIDATOR = HERE / "validate_cad_clearance.py"
CORE_CLEARANCE_REPORT = HERE / "cad_clearance_report.json"
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
        self.assertGreaterEqual(
            float(getattr(self.demo, "MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM", -math.inf)),
            15.0,
            "the core cam still overlaps the slider below the audited 15 mm withdrawal",
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
        self.assertEqual(robot_well_radius_mm, 3.325)
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
                        f"fixed robot proxy fills the Ø6.65 mm entry well: {point_mm}",
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
            "1a302d3674952d881df75e25e47b64a60b23a72b66fde48c35a0924ca1df6990",
        )
        self.assertEqual(
            records[nut_relative]["sha256"],
            "7f19e5abdb33083e7d179df78d5dc1a130140eadb74a3b22aa2d8f3c078266c7",
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
        self.assertIs(report.get("passed"), True, report)
        self.assertIs(report.get("release_ready"), True, report)
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
                "37771b10b4fe82614f0f7b460d44cdc81ea8505b4e6c7ac7b20a1f413b7ca848",
            ),
            "QuickChange/SO101_Magnetic/exports/hardware_McMaster_90318A720_shoulder_screw.step": (
                13_596,
                "1a302d3674952d881df75e25e47b64a60b23a72b66fde48c35a0924ca1df6990",
            ),
            "QuickChange/SO101_Magnetic/exports/hardware_DIN934_M3_lock_stud_nut.step": (
                27_468,
                "7f19e5abdb33083e7d179df78d5dc1a130140eadb74a3b22aa2d8f3c078266c7",
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

    def test_withdrawal_threshold_is_exactly_safe_against_source_cam(self) -> None:
        cad = self.clearance.CAD
        threshold_mm = float(self.demo.MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM)
        self.assertGreaterEqual(threshold_mm, 15.0)

        def locked_slider(withdrawal_mm: float) -> Any:
            return (
                cad.locking_slider()
                .translate(
                    (
                        cad.SLIDER_TRAVEL,
                        -withdrawal_mm,
                        cad.SLIDER_Z - cad.PLATE_THICKNESS,
                    )
                )
                .val()
            )

        cam = cad.positive_lock_cam().val()
        twelve_mm_slider = locked_slider(12.0)
        twelve_mm_overlap = self.clearance._intersection_volume_mm3(
            twelve_mm_slider, cam
        )
        self.assertGreater(
            twelve_mm_overlap,
            self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3,
            "12 mm still leaves the locked slider inside the dock cam",
        )
        threshold_slider = locked_slider(threshold_mm)
        threshold_distance = float(threshold_slider.distance(cam))
        threshold_overlap = (
            self.clearance._intersection_volume_mm3(threshold_slider, cam)
            if threshold_distance <= self.clearance.NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        self.assertLessEqual(
            threshold_overlap, self.clearance.OVERLAP_VOLUME_TOLERANCE_MM3
        )
        self.assertGreaterEqual(
            threshold_distance,
            self.clearance.MANUFACTURING_CLEARANCE_MM,
            {
                "withdrawal_mm": threshold_mm,
                "distance_mm": threshold_distance,
                "overlap_mm3": threshold_overlap,
            },
        )


class BoundedDynamicSmokeTests(unittest.TestCase):
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

        for event_name in expected_events[:-1]:
            self.assertIs(
                matching[event_name].get("physical_lock_confirmed"),
                False,
                matching[event_name],
            )
        self.assertIs(
            matching["dock_hold_released"].get("dock_hold_active"),
            False,
            matching["dock_hold_released"],
        )
        withdrawal = matching["source_axis_withdrawal_complete"]
        minimum_withdrawal_mm = float(self.demo.MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM)
        self.assertGreaterEqual(minimum_withdrawal_mm, 15.0)
        capture = matching["physical_capture_complete"]
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
        recomputed_withdrawal_mm = -1000.0 * float(displacement_local[1])
        recomputed_axis_alignment = -float(displacement_local[1]) / displacement_norm
        self.assertGreaterEqual(
            recomputed_withdrawal_mm,
            minimum_withdrawal_mm,
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
            1000.0 * float(np.linalg.norm(displacement_local[[0, 2]])),
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
        slider = matching["slider_return_verified"]
        self.assertIs(slider.get("cam_clear"), True, slider)
        self.assertIs(slider.get("returned"), True, slider)
        slider_position_mm = float(
            slider.get("slider_joint_position_mm", -math.inf)
        )
        self.assertGreaterEqual(slider_position_mm, 2.95, slider)
        self.assertLessEqual(slider_position_mm, 3.0 + 1.0e-6, slider)
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
                    -recomputed_withdrawal_mm,
                    cad.SLIDER_Z - cad.PLATE_THICKNESS,
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
        self.assertIs(
            matching["physical_lock_confirmed"].get("physical_lock_confirmed"),
            True,
            matching["physical_lock_confirmed"],
        )
        self.assertEqual(
            int(result.get("first_physical_lock_true_substep", -1)),
            physics_indices[-1],
            result,
        )

    def test_real_substep_capture_lock_and_release_is_collision_safe(self) -> None:
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
            timeout=30,
        )
        self.assertEqual(result_process.returncode, 0, result_process.stderr)
        result = json.loads(result_process.stdout)
        self.assertEqual(result.get("milestone"), "capture_lock_and_dock_release")
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
            "after_dock_release_source_axis_withdrawal_and_slider_return",
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
