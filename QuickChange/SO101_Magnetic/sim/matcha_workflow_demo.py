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
    stock.set("pos", "0 0 0.0095")
    stock.set("quat", "0 1 0 0")
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
        size=(0.006, 0.002),
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


def _build_xml_and_assets() -> tuple[str, dict[str, bytes]]:
    robot_root = ET.parse(ROBOT_XML).getroot()
    scene_root = ET.parse(SCENE_XML).getroot()
    assets: dict[str, bytes] = {}
    _mesh_assets(robot_root, ROBOT_XML.parent, "so101", assets)
    qc.activate_upstream_robot_collisions(robot_root)
    stock_gripper = _split_stock_gripper(robot_root)
    _merge_scene(robot_root, scene_root)
    worldbody = robot_root.find("worldbody")
    actuator = robot_root.find("actuator")
    equality = robot_root.find("equality")
    if worldbody is None or actuator is None or equality is None:
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
    for name, expected in (("spoon_tool_id", 21), ("whisk_tool_id", 22)):
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


def run_headless_scenario(max_steps: int = 100) -> dict[str, Any]:
    model = build_model()
    data = mujoco.MjData(model)
    initialize(model, data)
    startup_contacts = initial_contact_report(model, data)
    arm_actuators = np.asarray([model.actuator(name).id for name in ARM_ACTUATORS])
    for _ in range(max_steps):
        data.ctrl[arm_actuators] = DOCK_PRE_CAPTURE_Q["gripper"]
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError("Non-finite state during initialized smoke")
    result = initialized_summary(model, data)
    result["collision_coverage"] = collision_coverage(model)
    result["startup_contact_audit"] = startup_contacts
    result["sim_time"] = float(data.time)
    result["completed"] = True
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--dump-xml", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dump_xml is not None:
        xml, _ = _build_xml_and_assets()
        args.dump_xml.write_text(xml)
    result = run_headless_scenario(max_steps=args.max_steps)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
