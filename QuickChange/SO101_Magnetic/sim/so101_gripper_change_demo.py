#!/usr/bin/env python3
"""Full SO-101 dock-and-use demo for the detachable stock gripper.

The calibrated upstream robot XML is assembled with the quick-change scene at
load time.  The original stock gripper subtree is removed from the wrist and
inserted behind the v0.2 stock-gripper tool plate, so the arm genuinely begins
without an end effector while the original gripper joint/actuator remain usable.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROBOT_XML = REPO_ROOT / "Simulation" / "SO101" / "so101_new_calib.xml"
SCENE_XML = HERE / "so101_gripper_change_scene.xml"

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CAPTURE_Q = np.array([0.0, -0.5, 0.8, -0.3, 0.0])
# Same tool orientation and height as CAPTURE_Q, but 55 mm back along the
# coupling axis.  This was solved against the calibrated SO-101 kinematics.
PRE_CAPTURE_Q = np.array([0.0, -1.11771, 1.13502, -0.01731, 0.0])
GRIPPER_HOLD = 0.15
GRIPPER_OPEN = 1.25
GRIPPER_CLOSED = 0.0
TOOL_SERVO_ID = 6
DEMO_SECONDS = 12.0


def _mesh_assets(
    root: ET.Element,
    xml_dir: Path,
    prefix: str,
    assets: dict[str, bytes],
) -> None:
    """Put external STL bytes in MuJoCo's in-memory asset dictionary."""
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for index, mesh in enumerate(root.findall("./asset/mesh")):
        source_name = mesh.get("file")
        if source_name is None:
            continue
        source = (xml_dir / mesh_dir / source_name).resolve()
        key = f"{prefix}_{index}_{source.name}"
        assets[key] = source.read_bytes()
        # MuJoCo normally derives an omitted mesh name from the file stem.
        # Preserve that name before replacing the file with an in-memory key.
        if mesh.get("name") is None:
            mesh.set("name", Path(source_name).stem)
        mesh.set("file", key)


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in root.iter():
        if child in list(parent):
            return parent
    raise ValueError("Element has no parent")


def _make_robot_plate_frame(wrist_output: ET.Element) -> None:
    frame = ET.SubElement(
        wrist_output,
        "body",
        {"name": "robot_plate_frame", "quat": "0 1 0 0"},
    )
    ET.SubElement(
        frame,
        "geom",
        {
            "type": "mesh",
            "mesh": "qc_robot_plate_mesh",
            "material": "qc_robot_yellow",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for y in (-0.016, 0.016):
        ET.SubElement(
            frame,
            "geom",
            {
                "type": "box",
                "pos": f"0 {y} 0.0075",
                "size": "0.006 0.006 0.002",
                "material": "qc_magnet",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.SubElement(
        frame,
        "geom",
        {
            "type": "box",
            "pos": "0.003 0 0.0055",
            "size": "0.020 0.0024 0.0008",
            "material": "qc_lock",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for y in (-0.0075, -0.0025, 0.0025, 0.0075):
        ET.SubElement(
            frame,
            "geom",
            {
                "type": "cylinder",
                "pos": f"-0.031 {y} 0.0097",
                "size": "0.00105 0.0007",
                "material": "qc_contact",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.SubElement(
        frame,
        "site",
        {
            "name": "robot_mating_face",
            "pos": "0 0 0.0095",
            "size": "0.002",
            "rgba": "1 0.75 0.1 1",
        },
    )


def _split_stock_gripper(robot_root: ET.Element, scene_root: ET.Element) -> None:
    """Move the stock end-effector subtree from the wrist to the free tool."""
    original = robot_root.find(".//body[@name='gripper']")
    placeholder = scene_root.find(".//body[@name='stock_gripper_placeholder']")
    if original is None or placeholder is None:
        raise RuntimeError("Expected stock gripper and scene placeholder were not found")

    stock_gripper = copy.deepcopy(original)
    stock_gripper.set("name", "stock_gripper")
    stock_gripper.set("pos", "0 0 0.0095")
    stock_gripper.set("quat", "0 1 0 0")
    wrist_roll_joint = stock_gripper.find("./joint[@name='wrist_roll']")
    if wrist_roll_joint is None:
        raise RuntimeError("Stock model no longer contains the wrist_roll joint")
    stock_gripper.remove(wrist_roll_joint)
    # These meshes are retained for exact appearance but disabled for contact;
    # the demo validates changer state/control, not grasp or rack contact.
    for geom in stock_gripper.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    wrist_joint = original.find("./joint[@name='wrist_roll']")
    if wrist_joint is None:
        raise RuntimeError("Could not preserve wrist_roll on the bare arm")
    saved_attributes = dict(original.attrib)
    for child in list(original):
        original.remove(child)
    original.attrib.clear()
    original.attrib.update(saved_attributes)
    original.set("name", "wrist_output")
    original.append(copy.deepcopy(wrist_joint))
    ET.SubElement(
        original,
        "inertial",
        {"pos": "0 0 0", "mass": "0.025", "diaginertia": "0.00001 0.00001 0.00001"},
    )
    _make_robot_plate_frame(original)

    placeholder_parent = _find_parent(scene_root, placeholder)
    insert_at = list(placeholder_parent).index(placeholder)
    placeholder_parent.remove(placeholder)
    placeholder_parent.insert(insert_at, stock_gripper)


def _merge_scene(robot_root: ET.Element, scene_root: ET.Element) -> None:
    robot_root.set("model", scene_root.get("model", "SO-101 quick-change demo"))
    for tag in ("option", "statistic", "visual"):
        overlay = scene_root.find(tag)
        if overlay is not None:
            existing = robot_root.find(tag)
            if existing is not None:
                robot_root.remove(existing)
            robot_root.append(copy.deepcopy(overlay))

    for container_name in ("asset", "worldbody", "equality", "custom"):
        overlay = scene_root.find(container_name)
        if overlay is None:
            continue
        destination = robot_root.find(container_name)
        if destination is None:
            destination = ET.SubElement(robot_root, container_name)
        for child in list(overlay):
            destination.append(copy.deepcopy(child))


def build_model() -> mujoco.MjModel:
    """Compile the upstream robot plus the quick-change scene entirely in memory."""
    robot_root = ET.parse(ROBOT_XML).getroot()
    scene_root = ET.parse(SCENE_XML).getroot()
    assets: dict[str, bytes] = {}
    _mesh_assets(robot_root, ROBOT_XML.parent, "so101", assets)
    _mesh_assets(scene_root, SCENE_XML.parent, "quickchange", assets)
    _split_stock_gripper(robot_root, scene_root)
    _merge_scene(robot_root, scene_root)
    xml = ET.tostring(robot_root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.reshape(9))
    return quat


def relative_pose(data: mujoco.MjData, body1: int, body2: int) -> np.ndarray:
    rotation1 = data.xmat[body1].reshape(3, 3)
    relative_position = rotation1.T @ (data.xpos[body2] - data.xpos[body1])
    quat1 = data.xquat[body1]
    quat2 = data.xquat[body2]
    inverse1 = np.array([quat1[0], -quat1[1], -quat1[2], -quat1[3]])
    relative_quat = np.empty(4)
    mujoco.mju_mulQuat(relative_quat, inverse1, quat2)
    return np.concatenate((relative_position, relative_quat))


def set_weld(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    active: bool,
    update_pose: bool = False,
) -> None:
    equality_id = model.equality(name).id
    if active and update_pose:
        body1 = model.eq_obj1id[equality_id]
        body2 = model.eq_obj2id[equality_id]
        model.eq_data[equality_id, 3:10] = relative_pose(data, body1, body2)
    data.eq_active[equality_id] = int(active)
    mujoco.mj_forward(model, data)


def pose_error(data: mujoco.MjData, site_id: int, body_id: int) -> tuple[float, float]:
    position_error = float(np.linalg.norm(data.site_xpos[site_id] - data.xpos[body_id]))
    site_quat = quaternion_from_matrix(data.site_xmat[site_id])
    dot = float(np.clip(abs(np.dot(site_quat, data.xquat[body_id])), 0.0, 1.0))
    angle_error = float(2.0 * np.arccos(dot))
    return position_error, angle_error


class QuickChangeController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.arm_qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self.arm_actuators = np.array([model.actuator(name).id for name in ARM_JOINTS])
        self.gripper_qpos = model.joint("gripper").qposadr[0]
        self.gripper_actuator = model.actuator("gripper").id
        self.mating_site = model.site("robot_mating_face").id
        self.tool_body = model.body("tool_plate").id
        self.dock_body = model.body("tool_dock").id
        self.captured = False
        self.locked = False
        self.bus_connected = False
        self.simulated_id6_handshake = False
        self.capture_achieved = False
        self.lock_achieved = False
        self.handshake_achieved = False
        self.released_to_dock = False
        self.gripper_min = float(data.qpos[self.gripper_qpos])
        self.gripper_max = self.gripper_min
        self.max_locked_position_error = 0.0
        self.max_locked_angle_error = 0.0
        self.last_phase = ""

    def _phase(self, sim_time: float) -> str:
        if sim_time < 0.5:
            return "bare arm and docked gripper"
        if sim_time < 2.0:
            return "axial approach"
        if sim_time < 2.9:
            return "magnetic capture and simulated bus gate"
        if sim_time < 4.5:
            return "positive lock and rack withdrawal"
        if sim_time < 5.5:
            return "ID-6 gripper open"
        if sim_time < 6.5:
            return "ID-6 gripper close"
        if sim_time < 7.0:
            return "attached hold"
        if sim_time < 8.5:
            return "locked return to rack"
        if sim_time < 9.0:
            return "rack unlock and bus disconnect"
        if sim_time < 10.5:
            return "bare wrist withdrawal"
        return "cycle complete"

    def _arm_command(self, sim_time: float) -> np.ndarray:
        if sim_time < 0.5:
            return PRE_CAPTURE_Q
        if sim_time < 2.0:
            alpha = smoothstep((sim_time - 0.5) / 1.5)
            return PRE_CAPTURE_Q + alpha * (CAPTURE_Q - PRE_CAPTURE_Q)
        if sim_time < 3.0:
            return CAPTURE_Q
        if sim_time < 4.5:
            alpha = smoothstep((sim_time - 3.0) / 1.5)
            return CAPTURE_Q + alpha * (PRE_CAPTURE_Q - CAPTURE_Q)
        if sim_time < 7.0:
            return PRE_CAPTURE_Q
        if sim_time < 8.5:
            alpha = smoothstep((sim_time - 7.0) / 1.5)
            return PRE_CAPTURE_Q + alpha * (CAPTURE_Q - PRE_CAPTURE_Q)
        if sim_time < 9.0:
            return CAPTURE_Q
        if sim_time < 10.5:
            alpha = smoothstep((sim_time - 9.0) / 1.5)
            return CAPTURE_Q + alpha * (PRE_CAPTURE_Q - CAPTURE_Q)
        return PRE_CAPTURE_Q

    def update(self) -> None:
        sim_time = float(self.data.time)
        phase = self._phase(sim_time)
        if phase != self.last_phase:
            print(f"[{sim_time:5.2f}s] {phase}")
            self.last_phase = phase

        self.data.ctrl[self.arm_actuators] = self._arm_command(sim_time)
        gripper_command = float(self.data.qpos[self.gripper_qpos])
        if self.bus_connected and self.locked and 4.5 <= sim_time < 5.5:
            gripper_command = GRIPPER_OPEN
        elif self.bus_connected and self.locked and sim_time >= 5.5:
            gripper_command = GRIPPER_CLOSED
        self.data.ctrl[self.gripper_actuator] = gripper_command

        if sim_time >= 2.25 and not self.captured and not self.released_to_dock:
            position_error, angle_error = pose_error(
                self.data, self.mating_site, self.tool_body
            )
            if position_error < 0.002 and angle_error < np.deg2rad(2.0):
                set_weld(self.model, self.data, "magnetic_capture", True, update_pose=True)
                set_weld(self.model, self.data, "tool_in_dock", False)
                self.captured = True
                self.bus_connected = True
                self.simulated_id6_handshake = True
                self.capture_achieved = True
                self.handshake_achieved = True
                print(
                    f"[{sim_time:5.2f}s] captured at {position_error * 1000:.2f} mm; "
                    f"simulated TTL handshake exposes gripper servo ID {TOOL_SERVO_ID} "
                    "with torque held"
                )

        if sim_time >= 2.9 and self.captured and not self.locked:
            set_weld(self.model, self.data, "positive_lock", True, update_pose=True)
            set_weld(self.model, self.data, "magnetic_capture", False)
            self.locked = True
            self.lock_achieved = True
            print(f"[{sim_time:5.2f}s] rack cleared; spring-closed positive lock engaged")

        if self.locked:
            position_error, angle_error = pose_error(
                self.data, self.mating_site, self.tool_body
            )
            self.max_locked_position_error = max(
                self.max_locked_position_error, position_error
            )
            self.max_locked_angle_error = max(
                self.max_locked_angle_error, angle_error
            )

        if self.locked and 4.5 <= sim_time < 8.5:
            gripper_value = float(self.data.qpos[self.gripper_qpos])
            self.gripper_min = min(self.gripper_min, gripper_value)
            self.gripper_max = max(self.gripper_max, gripper_value)

        if sim_time >= 8.5 and self.locked and not self.released_to_dock:
            set_weld(self.model, self.data, "tool_in_dock", True, update_pose=True)
            set_weld(self.model, self.data, "positive_lock", False)
            self.locked = False
            self.captured = False
            self.bus_connected = False
            self.simulated_id6_handshake = False
            self.released_to_dock = True
            print(
                f"[{sim_time:5.2f}s] rack cam opened lock; tool docked and ID-6 bus disconnected"
            )


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    arm_qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
    arm_actuators = np.array([model.actuator(name).id for name in ARM_JOINTS])
    gripper_qpos = model.joint("gripper").qposadr[0]
    gripper_actuator = model.actuator("gripper").id

    data.qpos[arm_qpos] = CAPTURE_Q
    data.qpos[gripper_qpos] = GRIPPER_HOLD
    mujoco.mj_forward(model, data)
    mating_site = model.site("robot_mating_face").id
    capture_position = data.site_xpos[mating_site].copy()
    capture_quat = quaternion_from_matrix(data.site_xmat[mating_site])

    data.qpos[arm_qpos] = PRE_CAPTURE_Q
    tool_free_qpos = model.joint("tool_plate_free").qposadr[0]
    data.qpos[tool_free_qpos : tool_free_qpos + 3] = capture_position
    data.qpos[tool_free_qpos + 3 : tool_free_qpos + 7] = capture_quat
    dock_mocap = model.body("tool_dock").mocapid[0]
    data.mocap_pos[dock_mocap] = capture_position
    data.mocap_quat[dock_mocap] = capture_quat
    data.ctrl[arm_actuators] = PRE_CAPTURE_Q
    data.ctrl[gripper_actuator] = GRIPPER_HOLD
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return capture_position


def metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: QuickChangeController,
    capture_position: np.ndarray,
) -> dict[str, object]:
    tool_position = data.body("tool_plate").xpos.copy()
    tool_dock_error = float(np.linalg.norm(tool_position - capture_position))
    wrist_position = data.site_xpos[controller.mating_site].copy()
    bare_wrist_withdrawal = float(np.linalg.norm(wrist_position - capture_position))
    gripper_span = controller.gripper_max - controller.gripper_min
    positive_active = bool(data.eq_active[model.equality("positive_lock").id])
    magnetic_active = bool(data.eq_active[model.equality("magnetic_capture").id])
    dock_active = bool(data.eq_active[model.equality("tool_in_dock").id])
    topology_split = (
        model.body_parentid[model.body("stock_gripper").id] == model.body("tool_plate").id
        and model.body_parentid[model.body("robot_plate_frame").id]
        == model.body("wrist_output").id
        and model.actuator_trnid[model.actuator("gripper").id, 0]
        == model.joint("gripper").id
    )
    success = (
        topology_split
        and controller.capture_achieved
        and controller.lock_achieved
        and controller.handshake_achieved
        and controller.released_to_dock
        and not controller.captured
        and not controller.locked
        and not controller.bus_connected
        and not controller.simulated_id6_handshake
        and not positive_active
        and not magnetic_active
        and dock_active
        and bare_wrist_withdrawal > 0.045
        and tool_dock_error < 0.003
        and controller.max_locked_position_error < 0.003
        and controller.max_locked_angle_error < np.deg2rad(2.0)
        and gripper_span > 0.80
        and float(data.qpos[controller.gripper_qpos]) < 0.15
    )
    return {
        "success": bool(success),
        "arm_started_without_end_effector": bool(topology_split),
        "tool_servo_id": TOOL_SERVO_ID,
        "capture_achieved": controller.capture_achieved,
        "positive_lock_achieved": controller.lock_achieved,
        "simulated_id6_handshake_achieved": controller.handshake_achieved,
        "released_to_dock": controller.released_to_dock,
        "positive_lock_active": positive_active,
        "magnetic_capture_active": magnetic_active,
        "tool_in_dock_active": dock_active,
        "bus_connected": controller.bus_connected,
        "bare_wrist_withdrawal_mm": round(bare_wrist_withdrawal * 1000.0, 2),
        "tool_dock_position_error_mm": round(tool_dock_error * 1000.0, 3),
        "max_locked_coupling_error_mm": round(
            controller.max_locked_position_error * 1000.0, 3
        ),
        "max_locked_coupling_error_deg": round(
            float(np.rad2deg(controller.max_locked_angle_error)), 3
        ),
        "gripper_joint_min_rad": round(controller.gripper_min, 4),
        "gripper_joint_max_rad": round(controller.gripper_max, 4),
        "gripper_joint_span_rad": round(gripper_span, 4),
        "sim_time": round(float(data.time), 3),
    }


def save_preview(model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> None:
    from PIL import Image

    renderer = mujoco.Renderer(model, height=540, width=720)
    renderer.update_scene(data, camera="quick_change_overview")
    Image.fromarray(renderer.render()).save(path)
    renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--realtime", action="store_true", help="Pace a headless run at wall-clock speed")
    parser.add_argument("--max-seconds", type=float, default=DEMO_SECONDS)
    parser.add_argument("--save-preview", type=Path, help="Write a PNG of the final attached state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = build_model()
    data = mujoco.MjData(model)
    capture_position = initialize(model, data)
    controller = QuickChangeController(model, data)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    try:
        while data.time < args.max_seconds:
            tick = time.perf_counter()
            controller.update()
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            if args.realtime or viewer is not None:
                remaining = model.opt.timestep - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)

        result = metrics(model, data, controller, capture_position)
        print(json.dumps(result, indent=2))
        if args.save_preview:
            save_preview(model, data, args.save_preview)
            print(f"Saved preview to {args.save_preview}")
        return 0 if result["success"] else 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
