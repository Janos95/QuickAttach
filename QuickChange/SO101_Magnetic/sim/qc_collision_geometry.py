#!/usr/bin/env python3
"""Compact feature-aware collision installation for the SO-101 quick-change.

The runtime model intentionally keeps the calibrated upstream link meshes and
adds small analytic solids only for functional quick-change features.  It does
not load an undifferentiated hundreds-of-parts decomposition.  Exact CAD and
continuous clearance remain separate validation authorities.
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable


SIGNALS = ("ground", "power", "data", "id")
SIGNAL_Y_M = {
    "ground": -0.0075,
    "power": -0.0025,
    "data": 0.0025,
    "id": 0.0075,
}
SIGNAL_TO_BORE = {"ground": 1, "power": 2, "data": 3, "id": 4}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "unnamed"


def _rotate_by_quat(
    vector: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate one vector by a normalized MuJoCo wxyz quaternion."""

    vx, vy, vz = vector
    w_value, qx, qy, qz = quat
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w_value * tx + (qy * tz - qz * ty),
        vy + w_value * ty + (qz * tx - qx * tz),
        vz + w_value * tz + (qx * ty - qy * tx),
    )


def _geom(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, ...],
    rgba: str,
    quat: str | None = None,
    collision: bool = True,
    group: int = 3,
    **attributes: str,
) -> ET.Element:
    record = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(f"{value:.9g}" for value in pos),
        "size": " ".join(f"{value:.9g}" for value in size),
        "rgba": rgba,
        "group": str(group),
        "contype": "1" if collision else "0",
        "conaffinity": "1" if collision else "0",
    }
    if quat is not None:
        record["quat"] = quat
    record.update(attributes)
    return ET.SubElement(parent, "geom", record)


def activate_upstream_robot_collisions(robot_root: ET.Element) -> list[str]:
    """Name and activate every calibrated upstream collision mesh.

    The stock XML already carries one collision mesh for each moving component
    but leaves the base visuals non-colliding.  Base visual meshes are copied as
    collision twins so the complete stationary link also participates.
    """

    active: list[str] = []
    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Robot XML has no worldbody")
    link_order = {
        "base": 0,
        "shoulder": 1,
        "upper_arm": 2,
        "lower_arm": 3,
        "wrist": 4,
        "gripper": 5,
        "moving_jaw_so101_v1": 6,
    }
    all_link_bits = sum(1 << (index + 1) for index in link_order.values())
    for body in worldbody.iter("body"):
        body_name = _safe_name(body.get("name", "body"))
        link_index = link_order.get(body.get("name", ""))
        collision_bit = 1 << ((link_index if link_index is not None else 7) + 1)
        mask = 1  # environment/tools use bit 1
        if link_index is not None:
            for other_index in link_order.values():
                if abs(other_index - link_index) > 1:
                    mask |= 1 << (other_index + 1)
        else:
            mask |= all_link_bits
        collision_index = 0
        for geom in body.findall("./geom"):
            if geom.get("class") != "collision":
                continue
            name = f"robot_col_{body_name}_{collision_index:02d}"
            collision_index += 1
            geom.set("name", name)
            geom.set("contype", str(collision_bit))
            geom.set("conaffinity", str(mask))
            geom.set("group", "3")
            active.append(name)
        if body.get("name") != "base" or collision_index:
            continue
        for visual_index, visual in enumerate(body.findall("./geom")):
            if visual.get("class") != "visual":
                continue
            twin = copy.deepcopy(visual)
            name = f"robot_col_base_{visual_index:02d}"
            twin.set("name", name)
            twin.set("class", "collision")
            twin.set("material", visual.get("material", ""))
            twin.set("rgba", "0.8 0.55 0.08 0.12")
            twin.set("contype", str(collision_bit))
            twin.set("conaffinity", str(mask))
            twin.set("group", "3")
            body.append(twin)
            active.append(name)
    if not active:
        raise RuntimeError("No upstream collision geometry was activated")
    return active


def add_robot_quick_change_interface(wrist_output: ET.Element) -> ET.Element:
    """Install the compact powered robot-side coupling on the bare wrist."""

    frame = ET.SubElement(
        wrist_output,
        "body",
        {"name": "robot_plate_frame", "quat": "0 1 0 0"},
    )
    _geom(
        frame,
        name="qc_col_robot_plate_core",
        geom_type="box",
        pos=(0.0, 0.0, 0.00475),
        size=(0.024, 0.024, 0.00475),
        rgba="0.93 0.62 0.04 0.28",
        contype="64",
        conaffinity="31",
    )
    # Shoulder-stud wells are represented by the mating lands around the two
    # retained studs; the actual stud heads remain separately active.
    for side, y_value in (("left", -0.010), ("right", 0.010)):
        _geom(
            frame,
            name=f"qc_col_stud_{side}_head",
            geom_type="cylinder",
            pos=(0.0, y_value, 0.0110),
            size=(0.0030, 0.0015),
            rgba="0.72 0.75 0.78 1",
            contype="64",
            conaffinity="31",
        )
        _geom(
            frame,
            name=f"qc_col_stud_well_{side}_mating_land",
            geom_type="cylinder",
            pos=(0.0, y_value, 0.00945),
            size=(0.0040, 0.00005),
            rgba="0.25 0.8 0.4 0.35",
            contype="64",
            conaffinity="31",
        )
    for signal in SIGNALS:
        body = ET.SubElement(
            frame,
            "body",
            {
                "name": f"qc_pogo_{signal}_body",
                "pos": (
                    f"-0.031 {SIGNAL_Y_M[signal]:.7f} "
                    f"{0.00875 if signal == 'ground' else 0.00885:.7f}"
                ),
            },
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"qc_pogo_{signal}",
                "type": "slide",
                "axis": "0 0 -1",
                "range": "0 0.0012",
                "limited": "true",
                "damping": "0.12",
                "stiffness": "300",
                "springref": "0",
            },
        )
        half_height = 0.0008 if signal == "ground" else 0.0007
        _geom(
            body,
            name=f"qc_col_pogo_{signal}",
            geom_type="cylinder",
            pos=(0.0, 0.0, 0.0),
            size=(0.0008, half_height),
            rgba="1 0.62 0.05 1",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
            contype="64",
            conaffinity="31",
        )
    ET.SubElement(
        frame,
        "site",
        {
            "name": "robot_mating_face",
            "pos": "0 0 0.0095",
            "size": "0.0015",
            "rgba": "1 0.75 0.1 1",
        },
    )
    return frame


def _add_partitioned_contact_board(body: ET.Element, tool: str) -> list[str]:
    names: list[str] = []
    boxes = (
        ("left", -0.0340, 0.0, 0.000999, 0.011),
        ("right", -0.0270, 0.0, 0.001999, 0.011),
        ("row0", -0.0310, -0.01025, 0.002001, 0.00075),
        ("row1", -0.0310, -0.0050, 0.002001, 0.0005),
        ("row2", -0.0310, 0.0, 0.002001, 0.0005),
        ("row3", -0.0310, 0.0050, 0.002001, 0.0005),
        ("row4", -0.0310, 0.01025, 0.002001, 0.00075),
    )
    for suffix, x_value, y_value, x_half, y_half in boxes:
        name = f"{tool}_contact_board_{suffix}_collision"
        _geom(
            body,
            name=name,
            geom_type="box",
            pos=(x_value, y_value, 0.0005),
            size=(x_half, y_half, 0.0005),
            rgba="0.08 0.42 0.18 0.75",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(name)
    return names


def _add_target_with_bore(body: ET.Element, tool: str, side: str, y_value: float) -> list[str]:
    names: list[str] = []
    # Four boxes form an exact conservative square-minus-square proxy around
    # the M5 countersunk passage.  The surface authority measures the remaining
    # square-vs-round difference; the central bore is never filled.
    for suffix, x, y, sx, sy in (
        ("xneg", -0.00425, y_value, 0.00125, 0.0055),
        ("xpos", 0.00425, y_value, 0.00125, 0.0055),
        ("yneg", 0.0, y_value - 0.00425, 0.0030, 0.00125),
        ("ypos", 0.0, y_value + 0.00425, 0.0030, 0.00125),
    ):
        name = f"{tool}_target_{side}_{suffix}_collision"
        _geom(
            body,
            name=name,
            geom_type="box",
            pos=(x, y, 0.0015),
            size=(sx, sy, 0.0015),
            rgba="0.62 0.65 0.68 1",
        )
        names.append(name)
    _geom(
        body,
        name=f"{tool}_m5_screw_{side}_collision",
        geom_type="cylinder",
        pos=(0.0, y_value, 0.0045),
        size=(0.00245, 0.0045),
        rgba="0.35 0.37 0.40 1",
    )
    _geom(
        body,
        name=f"{tool}_m5_nut_{side}_collision",
        geom_type="cylinder",
        pos=(0.0, y_value, 0.0080),
        size=(0.0040, 0.0020),
        rgba="0.28 0.30 0.33 1",
    )
    names.extend(
        [f"{tool}_m5_screw_{side}_collision", f"{tool}_m5_nut_{side}_collision"]
    )
    return names


def add_tool_quick_change_interface(body: ET.Element, tool: str) -> list[str]:
    """Install one complete tool-side interface with semantic collision names."""

    names: list[str] = []
    # The robot's two retained shoulder studs enter square clearance wells at
    # y=+/-10 mm.  A single plate box would fill those wells and overlap each
    # stud by 3 mm at the exact mating pose, so the compact runtime plate is an
    # explicit five-box square-well partition.  Only pieces reaching the
    # +Y datum carry the dock_stop_land semantic.
    plate_pieces = (
        ("xneg", -0.016, 0.0, 0.012, 0.025, False),
        ("xpos", 0.016, 0.0, 0.012, 0.025, False),
        ("center_yneg", 0.0, -0.0195, 0.004, 0.0055, False),
        ("center_middle", 0.0, 0.0, 0.004, 0.006, False),
        ("center_ypos", 0.0, 0.0195, 0.004, 0.0055, True),
    )
    for suffix, x_value, y_value, x_half, y_half, stop_land in plate_pieces:
        semantics = "__mating_land"
        if stop_land or suffix in {"xneg", "xpos"}:
            semantics += "__dock_stop_land"
        plate_name = f"matcha_col_{tool}_plate_{suffix}{semantics}"
        _geom(
            body,
            name=plate_name,
            geom_type="box",
            pos=(x_value, y_value, 0.00475),
            size=(x_half, y_half, 0.00475),
            rgba="0.08 0.35 0.9 0.34",
        )
        names.append(plate_name)
    for signal in SIGNALS:
        pad_name = f"{tool}_pad_{signal}_collision"
        _geom(
            body,
            name=pad_name,
            geom_type="cylinder",
            pos=(-0.031, SIGNAL_Y_M[signal], 0.000025),
            size=(0.0020, 0.000025),
            rgba="0.95 0.55 0.05 1",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(pad_name)
    names.extend(_add_partitioned_contact_board(body, tool))
    # The two M5 target stacks sit outside the retained-stud wells; their
    # 5.5 mm outer envelope remains inside the 25 mm plate datum.
    for side, y_value in (("neg", -0.0190), ("pos", 0.0190)):
        names.extend(_add_target_with_bore(body, tool, side, y_value))
    for side, y_value in (("left", -0.010), ("right", 0.010)):
        _geom(
            body,
            name=f"{tool}_stud_{side}_collision",
            geom_type="cylinder",
            pos=(0.0, y_value, 0.0065),
            size=(0.0020, 0.0025),
            rgba="0.72 0.75 0.78 1",
        )
        _geom(
            body,
            name=f"{tool}_m3_nut_{side}_collision",
            geom_type="cylinder",
            pos=(0.0, y_value, 0.0090),
            size=(0.00315, 0.0012),
            rgba="0.42 0.44 0.47 1",
        )
        names.extend(
            [f"{tool}_stud_{side}_collision", f"{tool}_m3_nut_{side}_collision"]
        )
    ET.SubElement(
        body,
        "site",
        {
            "name": f"{tool}_mating_face",
            "pos": "0 0 0",
            "size": "0.0015",
            "rgba": "0.2 0.65 1 1",
        },
    )
    return names


def add_supported_dock(
    worldbody: ET.Element,
    tool: str,
    *,
    position: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    rgba: str,
) -> ET.Element:
    """Add a floor-connected passive rack with narrow functional contacts."""

    x_value, y_value, z_value = position
    # A vertical post terminates at the lowest corner of the rotated rear
    # anchor beam.  This produces a true zero-distance support chain without
    # burying a large cylinder inside the dock or the seated tool.
    anchor_corners: list[tuple[float, float, float]] = []
    for local_x in (-0.020, 0.020):
        for local_y in (-0.075, -0.055):
            for local_z in (-0.005, 0.005):
                rotated = _rotate_by_quat((local_x, local_y, local_z), quat)
                anchor_corners.append(
                    (
                        x_value + rotated[0],
                        y_value + rotated[1],
                        z_value + rotated[2],
                    )
                )
    anchor_world = min(anchor_corners, key=lambda point: (point[2], point[0], point[1]))
    support_height = max(0.002, anchor_world[2])
    support = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"dock_{tool}_support",
            "pos": f"{anchor_world[0]:.9g} {anchor_world[1]:.9g} 0",
        },
    )
    _geom(
        support,
        name=f"dock_{tool}_support_collision",
        geom_type="cylinder",
        pos=(0.0, 0.0, support_height / 2.0),
        size=(0.012, support_height / 2.0),
        rgba="0.16 0.18 0.21 1",
    )
    dock = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"dock_{tool}",
            "pos": f"{x_value:.9g} {y_value:.9g} {z_value:.9g}",
            "quat": " ".join(f"{value:.12g}" for value in quat),
        },
    )
    _geom(
        dock,
        name=f"dock_{tool}_support_anchor_collision",
        geom_type="box",
        pos=(0.0, -0.065, 0.0),
        size=(0.020, 0.010, 0.005),
        rgba="0.16 0.18 0.21 1",
    )
    _geom(
        dock,
        name=f"dock_{tool}_qc_col_dock_stop",
        geom_type="box",
        pos=(0.0, 0.029, 0.00475),
        size=(0.031, 0.004, 0.00475),
        rgba=rgba,
        solref="0.0005 1",
        solimp="0.99 0.9999 0.00001",
    )
    # The guide cheeks constrain the plate across local Y.  Keeping the long
    # local-X service corridor open is essential: the spoon handle and whisk
    # payload leave the rack through that corridor.
    for side, y_rail in (("left", -0.033), ("right", 0.033)):
        _geom(
            dock,
            name=f"dock_{tool}_rail_{side}_collision",
            geom_type="box",
            pos=(0.0, y_rail, 0.010),
            size=(0.034, 0.003, 0.010),
            rgba=rgba,
        )
    _geom(
        dock,
        name=f"dock_{tool}_cam_collision",
        geom_type="box",
        pos=(0.020, -0.033, 0.013),
        size=(0.003, 0.003, 0.003),
        rgba="0.12 0.75 0.35 1",
    )
    ET.SubElement(
        dock,
        "site",
        {
            "name": f"dock_{tool}_target",
            "pos": "0 0 0",
            "size": "0.002",
            "rgba": rgba,
        },
    )
    return dock


def collision_geom_names(root: ET.Element) -> list[str]:
    return sorted(
        geom.get("name", "")
        for geom in root.iter("geom")
        if geom.get("contype", "1") != "0" and geom.get("name")
    )


def require_unique_names(names: Iterable[str]) -> None:
    values = list(names)
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate collision geom names: {duplicates[:8]}")
