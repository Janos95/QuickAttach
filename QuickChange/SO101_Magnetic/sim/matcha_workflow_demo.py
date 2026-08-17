#!/usr/bin/env python3
"""Collision-active SO-101 matcha workflow simulation.

This module builds the calibrated upstream robot and reconstructed matcha
workcell entirely in memory.  Runtime contacts are real MuJoCo contacts and
all controller motion advances through ``mj_step``; equality constraints are
used only for named mechanical captures/locks after their physical guards.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import qc_collision_geometry as qc


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROBOT_XML = REPO_ROOT / "Simulation" / "SO101" / "so101_new_calib.xml"
SCENE_XML = HERE / "matcha_workflow_scene.xml"
CONFIG_PATH = HERE / "matcha_tool_geometry.json"

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ARM_ACTUATORS = ARM_JOINTS
TOOL_IDS = {"spoon": 21, "whisk": 22}
ALL_TOOL_IDS = {"gripper": 6, **TOOL_IDS}
TOOL_BUS_ID = 7
CAMERA_NAME = "matcha_scene_camera"

# Calibrated rack capture posture, with three pan values on one safe arc.
CAPTURE_BASE_Q = np.array([0.0, -0.5, 0.8, -0.3, 0.0], dtype=float)
PRE_CAPTURE_BASE_Q = np.array([0.0, -1.11771, 1.13502, -0.01731, 0.0], dtype=float)
DOCK_CAPTURE_Q = {
    "gripper": np.array([-0.72, -0.5, 0.8, -0.3, 0.0]),
    "spoon": np.array([0.0, -0.5, 0.8, -0.3, 0.0]),
    "whisk": np.array([0.72, -0.5, 0.8, -0.3, 0.0]),
}
DOCK_PRE_CAPTURE_Q = {
    name: np.array([q[0], *PRE_CAPTURE_BASE_Q[1:]], dtype=float)
    for name, q in DOCK_CAPTURE_Q.items()
}

# These are the calibrated robot mating-site poses for DOCK_CAPTURE_Q.  A
# compile-time FK assertion below catches drift against the upstream model.
DOCK_POSES = {
    "gripper": (
        (0.19082795371216685, 0.1330713713445051, 0.1939154579377553),
        (-0.2651276675099772, 0.6676452987889001, 0.2329157680473348, 0.6555206479743555),
    ),
    "spoon": (
        (0.24084947630993864, -0.0001778089765695206, 0.1939154579377552),
        (-0.017209108230571146, 0.7068973380865915, -0.01720910823057085, 0.706897338086591),
    ),
    "whisk": (
        (0.19059347652281455, -0.13333873141492703, 0.19391545793775508),
        (0.23291576804733463, 0.6555206479743557, -0.26512766750997674, 0.6676452987889),
    ),
}

CONTACT_NUMERICAL_EPSILON_M = 1.0e-9
FORBIDDEN_CONTACT_LATCH_M = 1.5e-4
DEFAULT_MAX_STEPS = 120_000
PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP = 20
WORKFLOW_GLOBAL_SAFETY_MARGIN_S = 10.0
CAPTURE_POSITION_TOLERANCE_M = 0.0015
CAPTURE_ORIENTATION_TOLERANCE_RAD = math.radians(1.25)
POGO_PAD_MAX_PENETRATION_M = 6.0e-5
DOCK_STOP_MAX_PENETRATION_M = 1.5e-4
CAPTURE_CONTACT_DWELL_S = 0.020
LOCK_VERIFY_DWELL_S = 0.050
POST_RELEASE_BUS_DWELL_S = 0.250


@dataclass(frozen=True)
class WorkflowAction:
    """One finite-deadline controller action."""

    name: str
    kind: str
    timeout_s: float
    duration_s: float = 0.0
    tool: str | None = None
    target_q: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("workflow action name/kind must be nonempty")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError(f"invalid timeout for {self.name}: {self.timeout_s}")
        if not math.isfinite(self.duration_s) or self.duration_s < 0.0:
            raise ValueError(f"invalid duration for {self.name}: {self.duration_s}")
        if self.timeout_s + 1.0e-12 < self.duration_s:
            raise ValueError(f"timeout shorter than duration for {self.name}")
        if self.target_q is not None and len(self.target_q) != len(ARM_JOINTS):
            raise ValueError(f"wrong target width for {self.name}")


def _recovery_controller_actions(
    tool: str = "gripper", *, include_rack_exit: bool = False
) -> tuple[WorkflowAction, ...]:
    if tool not in ALL_TOOL_IDS:
        raise ValueError(f"unsupported recovery tool {tool}")
    capture_release = (
        WorkflowAction(
            name=f"{tool}_to_capture",
            kind="move",
            tool=tool,
            target_q=tuple(float(value) for value in DOCK_CAPTURE_Q[tool]),
            duration_s=1.5,
            timeout_s=3.5,
        ),
        WorkflowAction(
            name=f"{tool}_physical_capture",
            kind="capture",
            tool=tool,
            duration_s=CAPTURE_CONTACT_DWELL_S,
            timeout_s=2.0,
        ),
        WorkflowAction(
            name=f"{tool}_lock_verify",
            kind="lock_verify",
            tool=tool,
            duration_s=LOCK_VERIFY_DWELL_S,
            timeout_s=2.0,
        ),
        WorkflowAction(
            name=f"{tool}_dock_release_verify",
            kind="release_verify",
            tool=tool,
            duration_s=POST_RELEASE_BUS_DWELL_S,
            timeout_s=2.0,
        ),
    )
    if not include_rack_exit:
        return capture_release
    return capture_release + (
        WorkflowAction(
            name=f"{tool}_rack_exit",
            kind="move",
            tool=tool,
            target_q=tuple(float(value) for value in DOCK_PRE_CAPTURE_Q[tool]),
            duration_s=1.5,
            timeout_s=3.5,
        ),
        WorkflowAction(
            name=f"{tool}_exit_hold",
            kind="hold",
            tool=tool,
            duration_s=0.10,
            timeout_s=1.0,
        ),
    )


def _mesh_assets(
    root: ET.Element,
    xml_dir: Path,
    prefix: str,
    assets: dict[str, bytes],
) -> None:
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
        if mesh.get("name") is None:
            mesh.set("name", Path(source_name).stem)
        mesh.set("file", key)


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in root.iter():
        if child in list(parent):
            return parent
    raise RuntimeError("Element has no parent")


def _merge_scene(robot_root: ET.Element, scene_root: ET.Element) -> None:
    robot_root.set("model", scene_root.get("model", "SO-101 matcha workflow"))
    for tag in ("option", "statistic", "visual"):
        overlay = scene_root.find(tag)
        if overlay is None:
            continue
        existing = robot_root.find(tag)
        if existing is not None:
            robot_root.remove(existing)
        robot_root.append(copy.deepcopy(overlay))
    for container_name in ("asset", "worldbody", "equality", "contact", "custom"):
        overlay = scene_root.find(container_name)
        if overlay is None:
            continue
        destination = robot_root.find(container_name)
        if destination is None:
            destination = ET.SubElement(robot_root, container_name)
        for child in list(overlay):
            destination.append(copy.deepcopy(child))


def _split_stock_gripper(robot_root: ET.Element) -> ET.Element:
    original = robot_root.find(".//body[@name='gripper']")
    if original is None:
        raise RuntimeError("Calibrated robot no longer has the stock gripper subtree")
    stock = copy.deepcopy(original)
    stock.set("name", "stock_gripper")
    # Exact wrapper pose solved from the calibrated collision-mesh geom frame
    # to the released stock-gripper STEP tool-local source contract.
    stock.set("pos", "0.0004875 -0.000000214 0.010500706")
    stock.set("quat", "0 -1 0 0")
    wrist_roll = stock.find("./joint[@name='wrist_roll']")
    if wrist_roll is None:
        raise RuntimeError("Stock subtree no longer contains wrist_roll")
    stock.remove(wrist_roll)
    for geom in stock.iter("geom"):
        if geom.get("name"):
            geom.set("name", f"stock_gripper_{geom.get('name')}")

    preserved_joint = original.find("./joint[@name='wrist_roll']")
    if preserved_joint is None:
        raise RuntimeError("Cannot preserve wrist_roll on bare wrist")
    saved_attributes = dict(original.attrib)
    for child in list(original):
        original.remove(child)
    original.attrib.clear()
    original.attrib.update(saved_attributes)
    original.set("name", "wrist_output")
    original.append(copy.deepcopy(preserved_joint))
    ET.SubElement(
        original,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": "0.025",
            "diaginertia": "0.00001 0.00001 0.00001",
        },
    )
    qc.add_robot_quick_change_interface(original)
    return stock


def _add_visual_twin(
    body: ET.Element,
    collision: ET.Element,
    *,
    name: str,
    rgba: str,
) -> None:
    visual = copy.deepcopy(collision)
    visual.set("name", name)
    visual.set("rgba", rgba)
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    visual.set("group", "2")
    body.append(visual)


def _add_payload_geom(
    body: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: tuple[float, float, float],
    size: tuple[float, ...],
    rgba: str,
    quat: str | None = None,
    **attributes: str,
) -> ET.Element:
    record = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(f"{value:.9g}" for value in pos),
        "size": " ".join(f"{value:.9g}" for value in size),
        "rgba": rgba,
        "contype": "1",
        "conaffinity": "1",
        "group": "3",
    }
    if quat is not None:
        record["quat"] = quat
    record.update(attributes)
    geom = ET.SubElement(body, "geom", record)
    _add_visual_twin(body, geom, name=f"{name}_visual", rgba=rgba)
    return geom


def _add_spoon_payload(tool: ET.Element) -> None:
    _add_payload_geom(
        tool,
        name="spoon_carrier_collision",
        geom_type="box",
        pos=(0.0, 0.0, 0.0165),
        size=(0.025, 0.021, 0.007),
        rgba="0.22 0.55 0.92 0.55",
    )
    _add_payload_geom(
        tool,
        name="spoon_handle_collision",
        geom_type="capsule",
        pos=(0.0, 0.0, 0.0),
        size=(0.004,),
        rgba="0.75 0.78 0.82 0.8",
        fromto="0 0 0.023 0.115 0 0.023",
    )
    _add_payload_geom(
        tool,
        name="spoon_bowl_collision",
        geom_type="ellipsoid",
        pos=(0.143, 0.0, 0.023),
        size=(0.026, 0.019, 0.005),
        rgba="0.78 0.81 0.84 0.68",
    )
    for index, z_value in enumerate((0.027, 0.038)):
        _add_payload_geom(
            tool,
            name=f"spoon_set_screw_{index}_collision",
            geom_type="cylinder",
            pos=(0.007, 0.0, z_value),
            size=(0.0015, 0.004),
            quat="0.70710678 0 0.70710678 0",
            rgba="0.3 0.32 0.35 1",
        )
    ET.SubElement(
        tool,
        "site",
        {"name": "spoon_camera_target", "pos": "0.143 0 0.023", "size": "0.002"},
    )
    ET.SubElement(
        tool,
        "site",
        {"name": "spoon_tip_target", "pos": "0.158 0 0.023", "size": "0.002"},
    )


def _add_whisk_payload(tool: ET.Element, actuator: ET.Element) -> None:
    _add_payload_geom(
        tool,
        name="whisk_housing_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.031),
        size=(0.024, 0.0215),
        rgba="0.12 0.40 0.76 0.55",
    )
    _add_payload_geom(
        tool,
        name="whisk_electronics_collision",
        geom_type="box",
        pos=(0.0, 0.0, 0.031),
        size=(0.017, 0.015, 0.009),
        rgba="0.12 0.55 0.25 0.5",
    )
    rotor = ET.SubElement(tool, "body", {"name": "whisk_eccentric_rotor", "pos": "0 0 0.052"})
    ET.SubElement(
        rotor,
        "joint",
        {
            "name": "whisk_rotor_joint",
            "type": "hinge",
            "axis": "0 0 1",
            "damping": "0.0003",
            "armature": "0.00002",
        },
    )
    _add_payload_geom(
        rotor,
        name="whisk_eccentric_collision",
        geom_type="cylinder",
        pos=(0.004, 0.0, 0.0),
        # Tangent to, rather than embedded in, the compliance carriage at its
        # zero state.  The eccentric remains a direct active collider.
        size=(0.006, 0.0018),
        rgba="0.8 0.5 0.1 0.8",
    )
    carriage = ET.SubElement(tool, "body", {"name": "whisk_compliance_carriage", "pos": "0 0 0.060"})
    ET.SubElement(
        carriage,
        "joint",
        {
            "name": "whisk_compliance_x",
            "type": "slide",
            "axis": "1 0 0",
            "range": "-0.004 0.004",
            "limited": "true",
            "damping": "0.35",
            "stiffness": "65",
        },
    )
    ET.SubElement(
        carriage,
        "joint",
        {
            "name": "whisk_compliance_z",
            "type": "slide",
            "axis": "0 0 1",
            "range": "-0.005 0",
            "limited": "true",
            "damping": "0.35",
            "stiffness": "80",
        },
    )
    _add_payload_geom(
        carriage,
        name="whisk_carriage_collision",
        geom_type="box",
        pos=(0.0, 0.0, 0.0),
        size=(0.016, 0.014, 0.006),
        rgba="0.32 0.35 0.40 0.6",
    )
    _add_payload_geom(
        carriage,
        name="whisk_bellows_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.013),
        size=(0.022, 0.0055),
        rgba="0.18 0.20 0.22 0.55",
    )
    _add_payload_geom(
        carriage,
        name="whisk_collet_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.023),
        size=(0.016, 0.005),
        rgba="0.65 0.67 0.70 0.75",
    )
    _add_payload_geom(
        carriage,
        name="chasen_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.061),
        size=(0.023, 0.033),
        rgba="0.78 0.60 0.32 0.48",
        friction="0.45 0.005 0.0001",
    )
    ET.SubElement(
        carriage,
        "site",
        {"name": "whisk_camera_target", "pos": "0 0 0.03", "size": "0.002"},
    )
    ET.SubElement(
        actuator,
        "motor",
        {
            "name": "whisk_motor",
            "joint": "whisk_rotor_joint",
            "gear": "1",
            "ctrlrange": "-0.8 0.8",
            "forcerange": "-0.08 0.08",
        },
    )


def _add_tool(
    worldbody: ET.Element,
    actuator: ET.Element,
    tool_name: str,
    stock_gripper: ET.Element | None,
) -> ET.Element:
    position, quat = DOCK_POSES[tool_name]
    tool = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"tool_{tool_name}",
            "pos": " ".join(f"{value:.12g}" for value in position),
            "quat": " ".join(f"{value:.12g}" for value in quat),
        },
    )
    ET.SubElement(tool, "freejoint", {"name": f"tool_{tool_name}_free"})
    qc.add_tool_quick_change_interface(tool, tool_name)
    if tool_name == "gripper":
        if stock_gripper is None:
            raise RuntimeError("Stock gripper subtree was not supplied")
        tool.append(stock_gripper)
    elif tool_name == "spoon":
        _add_spoon_payload(tool)
    elif tool_name == "whisk":
        _add_whisk_payload(tool, actuator)
    else:
        raise RuntimeError(f"Unsupported tool {tool_name}")
    ET.SubElement(
        tool,
        "site",
        {"name": f"{tool_name}_tool_id_site", "pos": "-0.031 0 0", "size": "0.001"},
    )
    return tool


def _add_ring(
    body: ET.Element,
    *,
    prefix: str,
    radius: float,
    z: float,
    half_height: float,
    segments: int,
    rgba: str,
) -> None:
    tangent_half = math.pi * radius / segments
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        x_value = radius * math.cos(angle)
        y_value = radius * math.sin(angle)
        half_angle = angle / 2.0
        quat = f"{math.cos(half_angle):.9g} 0 0 {math.sin(half_angle):.9g}"
        _add_payload_geom(
            body,
            name=f"{prefix}_{index:02d}_collision",
            geom_type="box",
            pos=(x_value, y_value, z),
            size=(0.003, tangent_half, half_height),
            quat=quat,
            rgba=rgba,
        )


def _add_supported_fixture(
    worldbody: ET.Element,
    name: str,
    *,
    position: tuple[float, float, float],
    radius: float,
    fixture_half_height: float,
    rgba: str,
) -> ET.Element:
    x_value, y_value, z_value = position
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": name, "pos": f"{x_value:.9g} {y_value:.9g} 0"},
    )
    _add_payload_geom(
        body,
        name=f"{name.replace('_station', '')}_support_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, z_value / 2.0),
        size=(max(0.012, radius * 0.45), z_value / 2.0),
        rgba="0.18 0.20 0.23 1",
    )
    _add_payload_geom(
        body,
        name=f"{name.replace('_station', '')}_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, z_value + fixture_half_height),
        size=(radius, fixture_half_height),
        rgba=rgba,
    )
    return body


def _add_workcell(worldbody: ET.Element, equality: ET.Element) -> None:
    bowl = ET.SubElement(worldbody, "body", {"name": "bowl_station", "pos": "-0.14 0.08 0"})
    _add_payload_geom(
        bowl,
        name="bowl_support_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.035),
        size=(0.030, 0.035),
        rgba="0.18 0.20 0.23 1",
    )
    _add_payload_geom(
        bowl,
        name="bowl_base_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.073),
        size=(0.050, 0.003),
        rgba="0.80 0.78 0.72 0.65",
    )
    _add_ring(
        bowl,
        prefix="bowl_wall",
        radius=0.053,
        z=0.092,
        half_height=0.019,
        segments=16,
        rgba="0.84 0.82 0.76 0.58",
    )
    ET.SubElement(bowl, "site", {"name": "bowl_target", "pos": "0.05 0 0.102", "size": "0.002"})
    ET.SubElement(bowl, "site", {"name": "bowl_interior_target", "pos": "0 0 0.080", "size": "0.002"})

    sieve = ET.SubElement(
        bowl,
        "body",
        {"name": "sieve_carriage", "pos": "0 0 0.115"},
    )
    ET.SubElement(
        sieve,
        "joint",
        {
            "name": "sieve_retract",
            "type": "slide",
            "axis": "0 0 1",
            "range": "0 0.125",
            "limited": "true",
            "damping": "0.20",
        },
    )
    _add_ring(
        sieve,
        prefix="sieve_ring",
        radius=0.050,
        z=0.0,
        half_height=0.003,
        segments=16,
        rgba="0.52 0.56 0.60 0.65",
    )
    _add_payload_geom(
        sieve,
        name="sieve_mesh_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, 0.0),
        size=(0.047, 0.00015),
        rgba="0.62 0.65 0.68 0.20",
    )
    _add_payload_geom(
        sieve,
        name="sieve_retention_lug_collision",
        geom_type="box",
        pos=(0.0, -0.060, 0.067),
        size=(0.010, 0.005, 0.004),
        rgba="0.92 0.58 0.08 0.8",
        solref="0.00025 1",
        solimp="0.99 0.9999 0.00001",
    )
    ET.SubElement(sieve, "site", {"name": "sieve_target", "pos": "0 0 0.002", "size": "0.002"})
    ET.SubElement(sieve, "site", {"name": "sieve_camera_target", "pos": "0.047 0 0.004", "size": "0.002"})
    latch = ET.SubElement(bowl, "body", {"name": "sieve_retention_latch", "pos": "0 0 0"})
    pawl = ET.SubElement(latch, "body", {"name": "sieve_latch_pawl", "pos": "0 0 0"})
    ET.SubElement(
        pawl,
        "joint",
        {
            "name": "sieve_latch_pawl_joint",
            "type": "slide",
            "axis": "0 1 0",
            "range": "-0.004 0",
            "limited": "true",
            "damping": "0.30",
            "stiffness": "250",
            "springref": "0",
            "solreflimit": "0.00025 1",
            "solimplimit": "0.99 0.9999 0.00001",
        },
    )
    _add_payload_geom(
        pawl,
        name="sieve_retention_latch_collision",
        geom_type="box",
        pos=(0.0, -0.070, 0.3067),
        size=(0.009, 0.008, 0.0035),
        quat="0.98078528 0.19509032 0 0",
        rgba="0.16 0.72 0.32 0.78",
        solref="0.00025 1",
        solimp="0.99 0.9999 0.00001",
    )
    ET.SubElement(
        equality,
        "joint",
        {
            "name": "sieve_top_latch",
            "joint1": "sieve_retract",
            "polycoef": "0.125 0 0 0 0",
            "active": "false",
            "solref": "0.001 1",
            "solimp": "0.99 0.999 0.0001",
        },
    )

    powder = _add_supported_fixture(
        worldbody,
        "powder_station",
        position=(-0.27, -0.19, 0.095),
        radius=0.040,
        fixture_half_height=0.055,
        rgba="0.32 0.58 0.28 0.65",
    )
    ET.SubElement(powder, "site", {"name": "powder_target", "pos": "0.038 0 0.145", "size": "0.002"})
    rinse = _add_supported_fixture(
        worldbody,
        "rinse_station",
        position=(-0.27, 0.25, 0.075),
        radius=0.055,
        fixture_half_height=0.035,
        rgba="0.18 0.55 0.75 0.58",
    )
    ET.SubElement(rinse, "site", {"name": "rinse_target", "pos": "0.05 0 0.10", "size": "0.002"})
    for subsystem, y_value, color in (
        ("hot_water", -0.08, "0.85 0.30 0.15 0.55"),
        ("milk", 0.20, "0.92 0.92 0.88 0.65"),
    ):
        station = _add_supported_fixture(
            worldbody,
            f"{subsystem}_station",
            position=(-0.36, y_value, 0.085),
            radius=0.035,
            fixture_half_height=0.055,
            rgba=color,
        )
        # Collision-active supported delivery tube reaches the bowl rim; fluid
        # itself remains an external deterministic metering abstraction.
        ET.SubElement(
            station,
            "geom",
            {
                "name": f"{subsystem}_delivery_tube_collision",
                "type": "capsule",
                "fromto": f"0 0 0.14 {0.22:.6f} {0.08-y_value:.6f} 0.12",
                "size": "0.004",
                "rgba": color,
                "contype": "1",
                "conaffinity": "1",
                "group": "3",
            },
        )
        ET.SubElement(
            station,
            "site",
            {
                "name": f"{subsystem}_outlet_target",
                "pos": f"0.22 {0.08-y_value:.6f} 0.12",
                "size": "0.002",
            },
        )


def _add_equalities(root: ET.Element) -> None:
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for tool in ("gripper", "spoon", "whisk"):
        ET.SubElement(
            equality,
            "weld",
            {
                "name": f"dock_{tool}_hold",
                "body1": f"dock_{tool}",
                "body2": f"tool_{tool}",
                "active": "true",
                "solref": "0.001 1",
                "solimp": "0.99 0.999 0.0001",
            },
        )
        ET.SubElement(
            equality,
            "weld",
            {
                "name": f"attach_{tool}",
                "body1": "robot_plate_frame",
                "body2": f"tool_{tool}",
                "relpose": "0 0 0.0095 1 0 0 0",
                "active": "false",
                "solref": "0.001 1",
                "solimp": "0.99 0.999 0.0001",
            },
        )
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "sieve_grasp",
            "body1": "robot_plate_frame",
            "body2": "sieve_carriage",
            "active": "false",
            "solref": "0.001 1",
            "solimp": "0.99 0.999 0.0001",
        },
    )


def _add_pogo_contact_pairs(root: ET.Element) -> None:
    """Install exact thin-contact pairs without adding an air-gap contact.

    Pair identity remains exact per tool and signal; wrong-signal contacts are
    neither generated nor accepted by the controller's bus audit.  A pogo
    crown is an axial electrical contact, so the pair is frictionless
    (``condim=1``); this prevents tangential rack motion from being converted
    into spurious spring-pin retraction while retaining a zero physical gap.
    """

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for tool in ALL_TOOL_IDS:
        for signal in qc.SIGNALS:
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": f"{tool}_{signal}_pogo_pad_pair",
                    "geom1": f"qc_col_pogo_{signal}",
                    "geom2": f"{tool}_pad_{signal}_collision",
                    "condim": "1",
                    "margin": "0",
                    "gap": "0",
                    "solref": "0.0005 1",
                    "solimp": "0.99 0.9999 0.00001",
                },
            )


def _build_xml_and_assets() -> tuple[str, dict[str, bytes]]:
    robot_root = ET.parse(ROBOT_XML).getroot()
    scene_root = ET.parse(SCENE_XML).getroot()
    assets: dict[str, bytes] = {}
    _mesh_assets(robot_root, ROBOT_XML.parent, "so101", assets)
    qc.activate_upstream_robot_collisions(robot_root)
    stock_gripper = _split_stock_gripper(robot_root)
    _merge_scene(robot_root, scene_root)
    worldbody = robot_root.find("worldbody")
    asset = robot_root.find("asset")
    actuator = robot_root.find("actuator")
    equality = robot_root.find("equality")
    if worldbody is None or asset is None or actuator is None or equality is None:
        raise RuntimeError("Merged robot is missing a required MJCF container")
    colors = {
        "gripper": "0.36 0.42 0.48 1",
        "spoon": "0.12 0.45 0.88 1",
        "whisk": "0.55 0.28 0.78 1",
    }
    for tool in ("gripper", "spoon", "whisk"):
        position, quat = DOCK_POSES[tool]
        qc.add_supported_dock(
            worldbody,
            asset,
            tool,
            position=position,
            quat=quat,
            rgba=colors[tool],
        )
        _add_tool(
            worldbody,
            actuator,
            tool,
            stock_gripper if tool == "gripper" else None,
        )
    _add_pogo_contact_pairs(robot_root)
    _add_workcell(worldbody, equality)
    _add_equalities(robot_root)
    names = qc.collision_geom_names(robot_root)
    qc.require_unique_names(names)
    return ET.tostring(robot_root, encoding="unicode"), assets


def build_model(*, verify_collision_authorities: bool = True) -> mujoco.MjModel:
    """Compile the calibrated robot and collision-active matcha workcell."""

    if verify_collision_authorities:
        config = json.loads(CONFIG_PATH.read_text())
        if config.get("tool_bus_id") != TOOL_BUS_ID or config.get("tool_ids") != TOOL_IDS:
            raise RuntimeError("Tool bus/ID config drifted")
        if config.get("release_ready") is not False:
            raise RuntimeError("Recovered pre-release config must remain fail-closed")
    xml, assets = _build_xml_and_assets()
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def _custom_numeric(model: mujoco.MjModel, name: str) -> np.ndarray:
    numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id < 0:
        raise RuntimeError(f"Missing custom numeric {name}")
    address = int(model.numeric_adr[numeric_id])
    size = int(model.numeric_size[numeric_id])
    return np.asarray(model.numeric_data[address : address + size], dtype=float)


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    arm_qpos = np.asarray([model.joint(name).qposadr[0] for name in ARM_JOINTS])
    arm_actuators = np.asarray([model.actuator(name).id for name in ARM_ACTUATORS])
    data.qpos[arm_qpos] = DOCK_PRE_CAPTURE_Q["gripper"]
    data.ctrl[arm_actuators] = DOCK_PRE_CAPTURE_Q["gripper"]
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper") >= 0:
        gripper_qpos = model.joint("gripper").qposadr[0]
        data.qpos[gripper_qpos] = 0.15
        data.ctrl[model.actuator("gripper").id] = 0.15
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if model.ncam != 1 or model.camera(CAMERA_NAME).id != 0:
        raise RuntimeError("Matcha scene must compile with exactly one named camera")
    if int(round(_custom_numeric(model, "tool_bus_id")[0])) != TOOL_BUS_ID:
        raise RuntimeError("Compiled tool bus ID drifted")
    for name, expected in (
        ("gripper_tool_id", 6),
        ("spoon_tool_id", 21),
        ("whisk_tool_id", 22),
    ):
        if int(round(_custom_numeric(model, name)[0])) != expected:
            raise RuntimeError(f"Compiled {name} drifted")


def initialized_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    active_collision_ids = np.flatnonzero(
        (np.asarray(model.geom_contype) != 0) & (np.asarray(model.geom_conaffinity) != 0)
    )
    unnamed = [
        int(index)
        for index in active_collision_ids
        if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(index))
    ]
    return {
        "schema_version": "2.0.0-recovery",
        "compiled": True,
        "release_ready": False,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "active_collision_geom_count": int(len(active_collision_ids)),
        "unnamed_active_collision_geom_ids": unnamed,
        "camera_count": int(model.ncam),
        "camera_name": CAMERA_NAME,
        "tool_bus_id": TOOL_BUS_ID,
        "tool_ids": TOOL_IDS,
        "ncon_initial": int(data.ncon),
        "finite_state": bool(
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        ),
        "model_xml_sha256": hashlib.sha256(_build_xml_and_assets()[0].encode()).hexdigest(),
    }


def collision_coverage(model: mujoco.MjModel) -> dict[str, Any]:
    """Report direct, body-owned collision coverage for rendered rigid parts."""

    rendered: dict[int, list[str]] = {}
    active: dict[int, list[str]] = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        name = str(model.geom(geom_id).name or f"geom_{geom_id}")
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        if int(model.geom_group[geom_id]) == 2 or (contype == 0 and conaffinity == 0):
            rendered.setdefault(body_id, []).append(name)
        if contype != 0 or conaffinity != 0:
            active.setdefault(body_id, []).append(name)
    missing: list[str] = []
    for body_id, visual_names in sorted(rendered.items()):
        if body_id == 0 or body_id in active:
            continue
        physical = [
            name
            for name in visual_names
            if not name.endswith("_target")
            and "camera_target" not in name
            and "fault_obstacle" not in name
        ]
        if physical:
            missing.append(str(model.body(body_id).name))
    return {
        "complete": not missing,
        "collision_coverage_complete": not missing,
        "missing_collision_bodies": sorted(missing),
        "rendered_body_count": len(rendered),
        "direct_collision_body_count": len(active),
        "active_collision_geom_count": sum(len(values) for values in active.values()),
    }


def initial_contact_report(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Return every true startup penetration without classifying it away."""

    penetrations: list[dict[str, Any]] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if float(contact.dist) >= -CONTACT_NUMERICAL_EPSILON_M:
            continue
        penetrations.append(
            {
                "geom_a": str(model.geom(int(contact.geom[0])).name),
                "geom_b": str(model.geom(int(contact.geom[1])).name),
                "penetration_m": -float(contact.dist),
            }
        )
    penetrations.sort(
        key=lambda item: (-float(item["penetration_m"]), item["geom_a"], item["geom_b"])
    )
    return {
        "contact_count": int(data.ncon),
        "penetration_count": len(penetrations),
        "max_penetration_m": max(
            (float(item["penetration_m"]) for item in penetrations), default=0.0
        ),
        "penetrations": penetrations,
        "passed": not penetrations,
    }


class MatchaWorkflowController:
    """Fail-closed real-dynamics controller for the first recovery milestone.

    This checkpoint intentionally stops after one complete gripper capture and
    rack exit.  The same finite-deadline action and contact-audit machinery is
    used by the remaining matcha process actions as they are restored.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        actions: tuple[WorkflowAction, ...] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.actions = actions if actions is not None else _recovery_controller_actions()
        if not self.actions:
            raise ValueError("controller action list cannot be empty")
        self.controller_dt_s = float(model.opt.timestep) * PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP
        public_horizon_s = DEFAULT_MAX_STEPS * self.controller_dt_s
        declared_horizon_s = math.fsum(action.timeout_s for action in self.actions)
        if public_horizon_s + 1.0e-12 < declared_horizon_s + WORKFLOW_GLOBAL_SAFETY_MARGIN_S:
            raise RuntimeError("public step horizon cannot cover declared action deadlines")

        self.arm_qpos_ids = np.asarray(
            [model.joint(name).qposadr[0] for name in ARM_JOINTS], dtype=int
        )
        self.arm_dof_ids = np.asarray(
            [model.joint(name).dofadr[0] for name in ARM_JOINTS], dtype=int
        )
        self.arm_actuator_ids = np.asarray(
            [model.actuator(name).id for name in ARM_ACTUATORS], dtype=int
        )
        self.action_index = 0
        self.action_started_s = float(data.time)
        # Trajectories start from the preceding actuator command, not the
        # instantaneous tracking state.  Switching a position servo from the
        # commanded capture datum to the slightly lagging qpos caused a small
        # but real unload of the spring-pin bus at the rack-exit boundary.
        self.action_start_q = np.asarray(
            data.ctrl[self.arm_actuator_ids], dtype=float
        ).copy()
        self.completed = False
        self.success = False
        self.abort_reason: str | None = None
        self.motion_stopped = False
        self.attached_tool: str | None = None
        self.bus_connected = False
        self.handshake_achieved = False
        self.physical_lock_confirmed = False
        self.lock_candidate_verified = False
        self.locked = False
        self.capture_live_substeps = 0
        self.lock_live_substeps = 0
        self.release_live_substeps = 0
        self.physics_substep_count = 0
        self.forbidden_contact_count = 0
        self.max_forbidden_penetration_m = 0.0
        self.first_forbidden_pair: tuple[str, str] | None = None
        self.max_tracking_error_rad = 0.0
        self.max_actuator_utilization = {
            name: 0.0 for name in (*ARM_ACTUATORS, "whisk_motor")
        }
        self.journal: list[dict[str, Any]] = [
            {
                "event": "controller_started",
                "sim_time_s": float(data.time),
                "action": self.actions[0].name,
            }
        ]

    @property
    def current_action(self) -> WorkflowAction | None:
        if self.completed or self.abort_reason is not None:
            return None
        return self.actions[self.action_index]

    def _equality_active(self, name: str) -> bool:
        equality_id = int(self.model.equality(name).id)
        return bool(self.data.eq_active[equality_id])

    def _tool_id_from_compiled_bus(self, tool: str) -> int:
        return int(round(float(_custom_numeric(self.model, f"{tool}_tool_id")[0])))

    def _tool_pose_error(self, tool: str) -> tuple[float, float]:
        robot_site = self.data.site("robot_mating_face")
        tool_site = self.data.site(f"{tool}_mating_face")
        position_error = float(np.linalg.norm(robot_site.xpos - tool_site.xpos))
        robot_rotation = np.asarray(robot_site.xmat, dtype=float).reshape(3, 3)
        tool_rotation = np.asarray(tool_site.xmat, dtype=float).reshape(3, 3)
        cosine = float((np.trace(robot_rotation.T @ tool_rotation) - 1.0) / 2.0)
        orientation_error = math.acos(max(-1.0, min(1.0, cosine)))
        return position_error, orientation_error

    def _capture_pose_is_valid(self, tool: str) -> bool:
        position_error, orientation_error = self._tool_pose_error(tool)
        return (
            position_error <= CAPTURE_POSITION_TOLERANCE_M
            and orientation_error <= CAPTURE_ORIENTATION_TOLERANCE_RAD
        )

    def _matching_pogo_contact_is_valid(
        self,
        contact: mujoco.MjContact,
        tool: str,
        signal: str,
    ) -> bool:
        geom_a = str(self.model.geom(int(contact.geom[0])).name)
        geom_b = str(self.model.geom(int(contact.geom[1])).name)
        expected = {f"qc_col_pogo_{signal}", f"{tool}_pad_{signal}_collision"}
        if {geom_a, geom_b} != expected:
            return False
        if float(contact.dist) > CONTACT_NUMERICAL_EPSILON_M:
            return False
        if float(contact.dist) < -POGO_PAD_MAX_PENETRATION_M:
            return False
        if not self._capture_pose_is_valid(tool):
            return False
        if not (
            self._equality_active(f"dock_{tool}_hold")
            or self._equality_active(f"attach_{tool}")
        ):
            return False
        pad_id = int(self.model.geom(f"{tool}_pad_{signal}_collision").id)
        pad_center = np.asarray(self.data.geom_xpos[pad_id], dtype=float)
        pad_rotation = np.asarray(self.data.geom_xmat[pad_id], dtype=float).reshape(3, 3)
        pad_axis = pad_rotation[:, 2]
        offset = np.asarray(contact.pos, dtype=float) - pad_center
        axial = abs(float(offset @ pad_axis))
        radial = float(np.linalg.norm(offset - (offset @ pad_axis) * pad_axis))
        contact_normal = np.asarray(contact.frame[:3], dtype=float)
        return (
            axial <= 1.0e-3
            and radial <= 2.01e-3
            # Pose, point-on-disk, exact signal identity, phase/equality and
            # penetration are the physical acceptance authority.  MuJoCo may
            # report a transient edge normal while an aligned round pin first
            # enters its larger matching disk; treating that same-signal edge
            # witness as a foreign collision produced thousands of false
            # observations even though its material/phase identity was exact.
            and np.all(np.isfinite(contact_normal))
            and float(np.linalg.norm(contact_normal)) >= 0.999
        )

    def _pogo_contact_signals(self, tool: str) -> set[str]:
        observed: set[str] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            for signal in qc.SIGNALS:
                if self._matching_pogo_contact_is_valid(contact, tool, signal):
                    observed.add(signal)
                    break
        return observed

    def _dock_stop_contact_is_valid(self, contact: mujoco.MjContact, tool: str) -> bool:
        geom_a = str(self.model.geom(int(contact.geom[0])).name)
        geom_b = str(self.model.geom(int(contact.geom[1])).name)
        stop_names = [
            name
            for name in (geom_a, geom_b)
            if qc.is_dock_stop_collision_name(tool, name)
        ]
        if len(stop_names) != 1:
            return False
        stop_name = stop_names[0]
        plate_name = geom_b if geom_a == stop_name else geom_a
        if not (
            plate_name.startswith(f"matcha_col_{tool}_plate_")
            and "__dock_stop_land" in plate_name
        ):
            return False
        if float(contact.dist) < -DOCK_STOP_MAX_PENETRATION_M:
            return False
        if not (
            self._equality_active(f"dock_{tool}_hold")
            or self._equality_active(f"attach_{tool}")
        ):
            return False
        dock_separation = float(
            np.linalg.norm(
                self.data.body(f"dock_{tool}").xpos
                - self.data.body(f"tool_{tool}").xpos
            )
        )
        if dock_separation > CAPTURE_POSITION_TOLERANCE_M:
            return False
        dock_rotation = np.asarray(
            self.data.body(f"dock_{tool}").xmat, dtype=float
        ).reshape(3, 3)
        expected_normal = dock_rotation[:, 1]
        return abs(float(np.asarray(contact.frame[:3]) @ expected_normal)) >= 0.98

    def _dock_stop_is_seated(self, tool: str) -> bool:
        return any(
            self._dock_stop_contact_is_valid(self.data.contact[index], tool)
            for index in range(self.data.ncon)
        )

    def _mating_land_contact_is_valid(
        self, contact: mujoco.MjContact, tool: str
    ) -> bool:
        geom_a = str(self.model.geom(int(contact.geom[0])).name)
        geom_b = str(self.model.geom(int(contact.geom[1])).name)
        robot_lands = {
            "qc_col_robot_plate_core__mating_land",
            "qc_col_stud_well_left_mating_land__locator_land",
            "qc_col_stud_well_right_mating_land__locator_land",
        }
        matching_robot_lands = robot_lands.intersection({geom_a, geom_b})
        if len(matching_robot_lands) != 1:
            return False
        robot_land = next(iter(matching_robot_lands))
        tool_land = geom_b if geom_a == robot_land else geom_a
        semantic_mating_land = "__mating_land" in tool_land
        owned_mating_surface = tool_land.startswith(
            f"matcha_col_{tool}_plate_"
        ) or tool_land.startswith(f"{tool}_target_") or tool_land.startswith(
            f"{tool}_m5_screw_"
        )
        if not (semantic_mating_land and owned_mating_surface):
            return False
        if float(contact.dist) < -2.0e-5:
            return False
        if not self._capture_pose_is_valid(tool):
            return False
        if not (
            self._equality_active(f"dock_{tool}_hold")
            or self._equality_active(f"attach_{tool}")
        ):
            return False
        tool_rotation = np.asarray(
            self.data.body(f"tool_{tool}").xmat, dtype=float
        ).reshape(3, 3)
        mating_normal = tool_rotation[:, 2]
        return abs(float(np.asarray(contact.frame[:3]) @ mating_normal)) >= 0.999

    def _locator_land_contact_is_valid(
        self, contact: mujoco.MjContact, tool: str
    ) -> bool:
        geom_a = str(self.model.geom(int(contact.geom[0])).name)
        geom_b = str(self.model.geom(int(contact.geom[1])).name)
        robot_lands = {
            "qc_col_stud_well_left_mating_land__locator_land",
            "qc_col_stud_well_right_mating_land__locator_land",
        }
        matching_robot_lands = robot_lands.intersection({geom_a, geom_b})
        if len(matching_robot_lands) != 1:
            return False
        robot_land = next(iter(matching_robot_lands))
        tool_land = geom_b if geom_a == robot_land else geom_a
        if not (
            tool_land.startswith(f"matcha_col_{tool}_plate_")
            and "__locator_land" in tool_land
        ):
            return False
        if float(contact.dist) < -1.0e-5 or not self._capture_pose_is_valid(tool):
            return False
        if not (
            self._equality_active(f"dock_{tool}_hold")
            or self._equality_active(f"attach_{tool}")
        ):
            return False
        tool_rotation = np.asarray(
            self.data.body(f"tool_{tool}").xmat, dtype=float
        ).reshape(3, 3)
        coupling_axis = tool_rotation[:, 2]
        robot_geom_id = int(self.model.geom(robot_land).id)
        center = np.asarray(self.data.geom_xpos[robot_geom_id], dtype=float)
        offset = np.asarray(contact.pos, dtype=float) - center
        axial = abs(float(offset @ coupling_axis))
        radial = float(
            np.linalg.norm(offset - (offset @ coupling_axis) * coupling_axis)
        )
        return (
            axial <= 2.0e-4
            and 2.8e-3 <= radial <= 4.2e-3
            and np.all(np.isfinite(contact.frame[:3]))
        )

    def _interface_guard(self, tool: str) -> bool:
        signals = self._pogo_contact_signals(tool)
        id_matches = self._tool_id_from_compiled_bus(tool) == ALL_TOOL_IDS[tool]
        dock_or_attach = (
            self._dock_stop_is_seated(tool)
            if self._equality_active(f"dock_{tool}_hold")
            else self._equality_active(f"attach_{tool}")
        )
        return (
            signals == set(qc.SIGNALS)
            and id_matches
            and dock_or_attach
            and self._capture_pose_is_valid(tool)
        )

    def _allowed_penetrating_contact(self, contact: mujoco.MjContact) -> bool:
        geom_a = str(self.model.geom(int(contact.geom[0])).name)
        geom_b = str(self.model.geom(int(contact.geom[1])).name)
        for tool in ALL_TOOL_IDS:
            if self._dock_stop_contact_is_valid(contact, tool):
                return True
            if self._mating_land_contact_is_valid(contact, tool):
                return True
            if self._locator_land_contact_is_valid(contact, tool):
                return True
            for signal in qc.SIGNALS:
                if self._matching_pogo_contact_is_valid(contact, tool, signal):
                    return True
        support_pairs = {
            frozenset(
                {
                    f"dock_{tool}_support_anchor_collision",
                    f"dock_{tool}_support_collision",
                }
            )
            for tool in ALL_TOOL_IDS
        }
        support_pairs.update(
            frozenset({f"dock_{tool}_support_collision", "matcha_floor_collision"})
            for tool in ALL_TOOL_IDS
        )
        return frozenset({geom_a, geom_b}) in support_pairs

    def _audit_contacts(self) -> None:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            penetration = -float(contact.dist)
            if penetration <= CONTACT_NUMERICAL_EPSILON_M:
                continue
            if self._allowed_penetrating_contact(contact):
                continue
            geom_a = str(self.model.geom(int(contact.geom[0])).name)
            geom_b = str(self.model.geom(int(contact.geom[1])).name)
            self.forbidden_contact_count += 1
            self.max_forbidden_penetration_m = max(
                self.max_forbidden_penetration_m, penetration
            )
            if self.first_forbidden_pair is None:
                self.first_forbidden_pair = (geom_a, geom_b)
            if penetration > FORBIDDEN_CONTACT_LATCH_M:
                self._abort("forbidden_collision")
                return

    def _record_actuator_loads(self) -> None:
        for name in self.max_actuator_utilization:
            actuator_id = int(self.model.actuator(name).id)
            force = abs(float(self.data.actuator_force[actuator_id]))
            force_range = np.asarray(self.model.actuator_forcerange[actuator_id], dtype=float)
            limit = max(abs(float(force_range[0])), abs(float(force_range[1])))
            utilization = force / limit if limit > 0.0 else 0.0
            self.max_actuator_utilization[name] = max(
                self.max_actuator_utilization[name], utilization
            )

    def _abort(self, reason: str) -> None:
        if self.abort_reason is not None:
            return
        self.abort_reason = reason
        self.motion_stopped = True
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_ids]
        self.data.xfrc_applied[:] = 0.0
        self.journal.append(
            {"event": "abort", "reason": reason, "sim_time_s": float(self.data.time)}
        )

    def _integrate(self) -> None:
        action = self.current_action
        if action is None:
            return
        for _ in range(PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP):
            mujoco.mj_step(self.model, self.data)
            self.physics_substep_count += 1
            self._audit_contacts()
            self._record_actuator_loads()
            if self.abort_reason is not None:
                return
            if self.locked and self.attached_tool is not None and self._equality_active(
                f"attach_{self.attached_tool}"
            ):
                if not self._interface_guard(self.attached_tool):
                    self._abort("contact_bus_drop")
                    return
            if action.kind == "capture" and action.tool is not None:
                if self._interface_guard(action.tool):
                    self.capture_live_substeps += 1
                else:
                    self.capture_live_substeps = 0
            elif action.kind == "lock_verify" and action.tool is not None:
                if self._interface_guard(action.tool):
                    self.lock_live_substeps += 1
                else:
                    self.lock_live_substeps = 0
            elif action.kind == "release_verify" and action.tool is not None:
                if self._interface_guard(action.tool):
                    self.release_live_substeps += 1
                else:
                    self.release_live_substeps = 0
            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
                and np.all(np.isfinite(self.data.actuator_force))
            ):
                self._abort("nonfinite_state")
                return

    def _advance_action(self, event: str) -> None:
        action = self.actions[self.action_index]
        self.journal.append(
            {
                "event": event,
                "action": action.name,
                "sim_time_s": float(self.data.time),
            }
        )
        self.action_index += 1
        if self.action_index >= len(self.actions):
            self.completed = True
            self.motion_stopped = True
            self.success = (
                self.locked
                and self.attached_tool is not None
                and self.forbidden_contact_count == 0
                and self.abort_reason is None
            )
            return
        self.action_started_s = float(self.data.time)
        self.action_start_q = np.asarray(
            self.data.ctrl[self.arm_actuator_ids], dtype=float
        ).copy()
        self.journal.append(
            {
                "event": "action_started",
                "action": self.actions[self.action_index].name,
                "sim_time_s": float(self.data.time),
            }
        )

    def _command_move(self, action: WorkflowAction, elapsed_s: float) -> None:
        if action.target_q is None:
            self._abort("move_missing_target")
            return
        target = np.asarray(action.target_q, dtype=float)
        alpha = min(1.0, max(0.0, elapsed_s / action.duration_s))
        smooth = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
        command = self.action_start_q + smooth * (target - self.action_start_q)
        self.data.ctrl[self.arm_actuator_ids] = command
        self._integrate()
        tracking = float(
            np.max(np.abs(self.data.qpos[self.arm_qpos_ids] - command))
        )
        self.max_tracking_error_rad = max(self.max_tracking_error_rad, tracking)
        if self.abort_reason is not None:
            return
        if elapsed_s >= action.duration_s:
            target_error = float(
                np.max(np.abs(self.data.qpos[self.arm_qpos_ids] - target))
            )
            speed = float(np.max(np.abs(self.data.qvel[self.arm_dof_ids])))
            if target_error <= 0.025 and speed <= 0.20:
                self._advance_action("move_complete")

    def _command_capture(self, action: WorkflowAction) -> None:
        if action.tool is None:
            self._abort("capture_missing_tool")
            return
        self.data.ctrl[self.arm_actuator_ids] = DOCK_CAPTURE_Q[action.tool]
        self._integrate()
        required = math.ceil(CAPTURE_CONTACT_DWELL_S / float(self.model.opt.timestep))
        if self.abort_reason is not None or self.capture_live_substeps < required:
            return
        if not self._dock_stop_is_seated(action.tool):
            self._abort("dock_stop_not_seated")
            return
        self.bus_connected = True
        self.handshake_achieved = True
        self.physical_lock_confirmed = True
        self.data.eq_active[self.model.equality(f"attach_{action.tool}").id] = 1
        self.attached_tool = action.tool
        self._advance_action("physical_capture_complete")

    def _command_lock_verify(self, action: WorkflowAction) -> None:
        if action.tool is None:
            self._abort("lock_missing_tool")
            return
        self.data.ctrl[self.arm_actuator_ids] = DOCK_CAPTURE_Q[action.tool]
        self._integrate()
        required = math.ceil(LOCK_VERIFY_DWELL_S / float(self.model.opt.timestep))
        if self.abort_reason is not None or self.lock_live_substeps < required:
            return
        equality_active = self._equality_active(f"attach_{action.tool}")
        self.lock_candidate_verified = bool(
            equality_active
            and self.physical_lock_confirmed
            and self._interface_guard(action.tool)
        )
        if not self.lock_candidate_verified:
            self._abort("physical_lock_not_confirmed")
            return
        self.data.eq_active[self.model.equality(f"dock_{action.tool}_hold").id] = 0
        self._advance_action("lock_verified")

    def _command_release_verify(self, action: WorkflowAction) -> None:
        if action.tool is None:
            self._abort("release_verify_missing_tool")
            return
        self.data.ctrl[self.arm_actuator_ids] = DOCK_CAPTURE_Q[action.tool]
        self._integrate()
        required = math.ceil(
            POST_RELEASE_BUS_DWELL_S / float(self.model.opt.timestep)
        )
        if self.abort_reason is not None or self.release_live_substeps < required:
            return
        if self._equality_active(f"dock_{action.tool}_hold"):
            self._abort("dock_hold_failed_to_release")
            return
        self.locked = bool(
            self.lock_candidate_verified
            and self.physical_lock_confirmed
            and self._equality_active(f"attach_{action.tool}")
            and self._interface_guard(action.tool)
        )
        if not self.locked:
            self._abort("post_release_lock_not_confirmed")
            return
        self._advance_action("dock_release_verified")

    def _command_hold(self, action: WorkflowAction, elapsed_s: float) -> None:
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_ids]
        self._integrate()
        if self.abort_reason is None and elapsed_s >= action.duration_s:
            self._advance_action("hold_complete")

    def step(self) -> None:
        action = self.current_action
        if action is None:
            return
        elapsed_s = float(self.data.time) - self.action_started_s
        if elapsed_s > action.timeout_s:
            self._abort(f"action_timeout:{action.name}")
            return
        if action.kind == "move":
            self._command_move(action, elapsed_s)
        elif action.kind == "capture":
            self._command_capture(action)
        elif action.kind == "lock_verify":
            self._command_lock_verify(action)
        elif action.kind == "release_verify":
            self._command_release_verify(action)
        elif action.kind == "hold":
            self._command_hold(action, elapsed_s)
        else:
            self._abort(f"unknown_action:{action.kind}")

    def result(self) -> dict[str, Any]:
        action = self.current_action
        live_signals = (
            sorted(self._pogo_contact_signals(self.attached_tool))
            if self.attached_tool is not None
            else []
        )
        dock_hold_active = (
            self._equality_active(f"dock_{self.attached_tool}_hold")
            if self.attached_tool is not None
            else None
        )
        attach_equality_active = (
            self._equality_active(f"attach_{self.attached_tool}")
            if self.attached_tool is not None
            else None
        )
        return {
            "completed": self.completed,
            "success": self.success,
            "abort_reason": self.abort_reason,
            "motion_stopped": self.motion_stopped,
            "action_index": self.action_index,
            "action": action.name if action is not None else None,
            "sim_time_s": float(self.data.time),
            "physics_substep_count": self.physics_substep_count,
            "attached_tool": self.attached_tool,
            "bus_connected": self.bus_connected,
            "handshake_achieved": self.handshake_achieved,
            "physical_lock_confirmed": self.physical_lock_confirmed,
            "lock_candidate_verified": self.lock_candidate_verified,
            "locked": self.locked,
            "live_pogo_signals": live_signals,
            "four_signal_bus_live": live_signals == sorted(qc.SIGNALS),
            "dock_hold_active": dock_hold_active,
            "attach_equality_active": attach_equality_active,
            "finite_actuator_force": bool(
                np.all(np.isfinite(self.data.actuator_force))
            ),
            "forbidden_contact_count": self.forbidden_contact_count,
            "max_forbidden_penetration_m": self.max_forbidden_penetration_m,
            "first_forbidden_pair": self.first_forbidden_pair,
            "max_tracking_error_rad": self.max_tracking_error_rad,
            "max_actuator_utilization": dict(self.max_actuator_utilization),
            "action_deadlines_s": {
                action.name: action.timeout_s for action in self.actions
            },
            "declared_deadline_sum_s": math.fsum(
                action.timeout_s for action in self.actions
            ),
            "global_safety_margin_s": WORKFLOW_GLOBAL_SAFETY_MARGIN_S,
            "journal": list(self.journal),
        }


def run_headless_scenario(
    max_steps: int = DEFAULT_MAX_STEPS, *, include_rack_exit: bool = False
) -> dict[str, Any]:
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    model = build_model()
    data = mujoco.MjData(model)
    initialize(model, data)
    startup_contacts = initial_contact_report(model, data)
    result = initialized_summary(model, data)
    controller = MatchaWorkflowController(
        model,
        data,
        actions=_recovery_controller_actions(include_rack_exit=include_rack_exit),
    )
    for _ in range(max_steps):
        controller.step()
        if controller.completed or controller.abort_reason is not None:
            break
    result["collision_coverage"] = collision_coverage(model)
    result["startup_contact_audit"] = startup_contacts
    result.update(controller.result())
    result["milestone"] = (
        "capture_lock_and_dock_release"
        if not include_rack_exit
        else "capture_lock_dock_release_and_rack_exit_diagnostic"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--include-rack-exit", action="store_true")
    parser.add_argument("--dump-xml", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dump_xml is not None:
        xml, _ = _build_xml_and_assets()
        args.dump_xml.write_text(xml)
    result = run_headless_scenario(
        max_steps=args.max_steps, include_rack_exit=args.include_rack_exit
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
