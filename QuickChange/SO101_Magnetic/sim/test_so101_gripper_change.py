#!/usr/bin/env python3
"""Cheap core physics/collision regressions for the SO-101 quick changer.

These tests intentionally exercise the production builder/controller without
editing them.  Initialization-only pose writes are used to reach a witness
configuration quickly; no test runs the twelve-second demo trajectory.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

import so101_gripper_change_demo as demo


HERE = Path(__file__).resolve().parent
DEMO_PATH = HERE / "so101_gripper_change_demo.py"
QUICK_CHANGE_ROOTS = (
    "robot_plate_frame",
    "tool_dock",
    "tool_plate",
)
FORBIDDEN_POST_INITIALIZE_STATE = {"qpos", "qvel", "time", "eq_data"}


def is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = body_id
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return False


def target_field(target: ast.AST) -> str | None:
    node = target
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_POST_INITIALIZE_STATE:
        return node.attr
    return None


class CoreBuildAndCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = demo.build_model()
        cls.data = mujoco.MjData(cls.model)
        cls.capture_position = demo.initialize(cls.model, cls.data)

    def test_split_topology_and_tool_id_are_exact(self) -> None:
        model = self.model
        self.assertEqual(demo.TOOL_SERVO_ID, 6)
        self.assertEqual(
            int(model.body_parentid[model.body("stock_gripper").id]),
            int(model.body("tool_plate").id),
        )
        self.assertEqual(
            int(model.body_parentid[model.body("robot_plate_frame").id]),
            int(model.body("wrist_output").id),
        )
        self.assertEqual(
            int(model.actuator_trnid[model.actuator("gripper").id, 0]),
            int(model.joint("gripper").id),
        )
        numeric = model.numeric("tool_servo_id")
        self.assertEqual(float(model.numeric_data[int(numeric.adr[0])]), 6.0)

    def test_every_rendered_quick_change_body_has_direct_active_collision(self) -> None:
        model = self.model
        roots = [int(model.body(name).id) for name in QUICK_CHANGE_ROOTS]
        missing: dict[str, list[str]] = {}
        for body_id in range(1, int(model.nbody)):
            if not any(is_descendant(model, body_id, root) for root in roots):
                continue
            visual_names: list[str] = []
            collision_names: list[str] = []
            for geom_id in range(int(model.ngeom)):
                if int(model.geom_bodyid[geom_id]) != body_id:
                    continue
                name = str(model.geom(geom_id).name or f"geom_{geom_id}")
                if int(model.geom_contype[geom_id]) or int(
                    model.geom_conaffinity[geom_id]
                ):
                    collision_names.append(name)
                else:
                    visual_names.append(name)
            if visual_names and not collision_names:
                missing[str(model.body(body_id).name)] = visual_names
        self.assertEqual(missing, {}, json.dumps(missing, indent=2))

    def test_collision_engine_has_active_quick_change_pairs(self) -> None:
        model = self.model
        roots = [int(model.body(name).id) for name in QUICK_CHANGE_ROOTS]
        active = []
        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            if any(is_descendant(model, body_id, root) for root in roots) and (
                int(model.geom_contype[geom_id])
                or int(model.geom_conaffinity[geom_id])
            ):
                active.append(str(model.geom(geom_id).name or f"geom_{geom_id}"))
        self.assertTrue(active, "quick-change assembly has no active contact geometry")


class CoreControllerSourceSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DEMO_PATH.read_text()
        cls.tree = ast.parse(cls.source, filename=str(DEMO_PATH))

    def test_controller_and_reachable_helpers_never_teleport_or_rewrite_weld_pose(
        self,
    ) -> None:
        module_functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        controller = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "QuickChangeController"
        )
        work: list[tuple[str, ast.AST]] = [
            (f"QuickChangeController.{node.name}", node)
            for node in controller.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        seen_helpers: set[str] = set()
        violations: list[str] = []
        while work:
            label, function = work.pop()
            for node in ast.walk(function):
                targets: list[ast.AST] = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                for target in targets:
                    field = target_field(target)
                    if field is not None:
                        violations.append(f"{label}:{node.lineno} writes {field}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    helper_name = node.func.id
                    if helper_name in module_functions and helper_name not in seen_helpers:
                        seen_helpers.add(helper_name)
                        work.append((helper_name, module_functions[helper_name]))
        self.assertEqual(violations, [], "\n".join(sorted(violations)))

    def test_builder_does_not_blanket_disable_stock_gripper_contact(self) -> None:
        split = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_split_stock_gripper"
        )
        blanket_loops: list[int] = []
        for node in ast.walk(split):
            if not isinstance(node, ast.For):
                continue
            iterator = ast.unparse(node.iter).replace('"', "'")
            if iterator != "stock_gripper.iter('geom')":
                continue
            disabled_fields: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call) or not isinstance(
                    child.func, ast.Attribute
                ):
                    continue
                if child.func.attr != "set" or len(child.args) < 2:
                    continue
                key = child.args[0]
                value = child.args[1]
                if (
                    isinstance(key, ast.Constant)
                    and key.value in {"contype", "conaffinity"}
                    and isinstance(value, ast.Constant)
                    and str(value.value) == "0"
                ):
                    disabled_fields.add(str(key.value))
            if disabled_fields == {"contype", "conaffinity"}:
                blanket_loops.append(node.lineno)
        self.assertEqual(
            blanket_loops,
            [],
            f"blanket stock-gripper collision disable at lines {blanket_loops}",
        )


class CoreDeterministicNegativeControlTests(unittest.TestCase):
    def fresh(self) -> tuple[mujoco.MjModel, mujoco.MjData, demo.QuickChangeController]:
        model = demo.build_model()
        data = mujoco.MjData(model)
        demo.initialize(model, data)
        controller = demo.QuickChangeController(model, data)
        arm_qpos = np.asarray(
            [model.joint(name).qposadr[0] for name in demo.ARM_JOINTS], dtype=int
        )
        data.qpos[arm_qpos] = demo.CAPTURE_Q
        mujoco.mj_forward(model, data)
        return model, data, controller

    def test_capture_rejects_a_dock_translated_100_mm(self) -> None:
        model, data, controller = self.fresh()
        dock_mocap = int(model.body("tool_dock").mocapid[0])
        data.mocap_pos[dock_mocap, 0] += 0.100
        mujoco.mj_forward(model, data)
        separation = float(
            np.linalg.norm(
                data.body("tool_dock").xpos - data.body("tool_plate").xpos
            )
        )
        self.assertGreater(separation, 0.095)
        data.time = 2.25
        controller.update()
        self.assertFalse(controller.captured)
        self.assertFalse(controller.handshake_achieved)

    def test_wrong_tool_id_cannot_complete_the_bus_handshake(self) -> None:
        model, data, controller = self.fresh()
        numeric = model.numeric("tool_servo_id")
        model.numeric_data[int(numeric.adr[0])] = 5.0
        data.time = 2.25
        controller.update()
        self.assertFalse(controller.handshake_achieved)
        self.assertFalse(controller.bus_connected)

    def test_missing_positive_lock_cannot_set_logical_locked_state(self) -> None:
        model, data, controller = self.fresh()
        data.time = 2.25
        controller.update()
        self.assertTrue(controller.captured, "negative did not reach capture witness")
        original_set_weld = demo.set_weld

        def drop_positive_lock(
            model_arg: mujoco.MjModel,
            data_arg: mujoco.MjData,
            name: str,
            active: bool,
            update_pose: bool = False,
        ) -> None:
            if name == "positive_lock":
                return
            original_set_weld(model_arg, data_arg, name, active, update_pose)

        data.time = 2.90
        with mock.patch.object(demo, "set_weld", side_effect=drop_positive_lock):
            controller.update()
        lock_active = bool(data.eq_active[model.equality("positive_lock").id])
        self.assertFalse(lock_active)
        self.assertFalse(controller.locked)
        self.assertFalse(controller.lock_achieved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
