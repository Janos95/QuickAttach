#!/usr/bin/env python3
"""Export the source-bound QuickAttach MuJoCo scene for the browser controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path


REMOVED_WORKCELL_BODIES = {
    "bowl_station",
    "powder_station",
    "rinse_station",
    "hot_water_station",
    "milk_station",
    "dock_spoon_support",
    "dock_whisk_support",
}

DOCK_POSITIONS = {
    "gripper": (0.1905934765, 0.1333387314, 0.1939154579),
    "spoon": (0.2408494763, -0.0001778090, 0.1939154579),
    "whisk": (0.1905934765, -0.1333387314, 0.1939154579),
}

FIXTURE_LINK_PAIRS = (("gripper", "spoon"), ("spoon", "whisk"))

CAD_PARTS = (
    {
        "id": "robot-plate",
        "label": "Robot plate",
        "category": "Quick changer",
        "source": "QuickChange/SO101_Magnetic/exports/so101_robot_plate.stl",
        "sourceFile": "so101_robot_plate.step",
        "bodyRoots": ["robot_plate_frame"],
        "geomPrefixes": ["qc_col_robot_plate_"],
    },
    {
        "id": "positive-lock-slider",
        "label": "Positive-lock slider",
        "category": "Quick changer",
        "source": "QuickChange/SO101_Magnetic/exports/so101_positive_lock_slider.stl",
        "sourceFile": "so101_positive_lock_slider.step",
        "bodyRoots": ["qc_positive_lock_slider"],
        "geomPrefixes": ["qc_col_lock_slider_"],
    },
    {
        "id": "passive-tool-dock",
        "label": "Passive tool dock",
        "category": "Quick changer",
        "source": "QuickChange/SO101_Magnetic/exports/so101_passive_tool_dock.stl",
        "sourceFile": "so101_passive_tool_dock.step",
        "bodyRoots": ["dock_gripper"],
        "geomPrefixes": ["dock_gripper_qc_col_", "dock_gripper_cam", "dock_gripper_keeper"],
    },
    {
        "id": "dock-support-bracket",
        "label": "Core dock support bracket",
        "category": "Quick changer",
        "source": "QuickChange/SO101_Magnetic/exports/so101_core_dock_support_bracket.stl",
        "sourceFile": "so101_core_dock_support_bracket.step",
        "bodyRoots": ["dock_gripper"],
        "geomPrefixes": ["dock_gripper_floor_support_"],
    },
    {
        "id": "gripper-tool-plate",
        "label": "Stock gripper tool plate",
        "category": "Tool interfaces",
        "source": "QuickChange/SO101_Magnetic/exports/so101_stock_gripper_tool_plate.stl",
        "sourceFile": "so101_stock_gripper_tool_plate.step",
        "bodyRoots": ["tool_gripper"],
        "geomPrefixes": ["matcha_col_gripper_plate_"],
    },
    {
        "id": "generic-tool-plate",
        "label": "Generic tool plate",
        "category": "Tool interfaces",
        "source": "QuickChange/SO101_Magnetic/exports/so101_tool_plate.stl",
        "sourceFile": "so101_tool_plate.step",
        "bodyRoots": ["tool_spoon"],
        "geomPrefixes": ["matcha_col_spoon_plate_"],
    },
    {
        "id": "contact-board",
        "label": "Tool contact board",
        "category": "Tool interfaces",
        "source": "QuickChange/SO101_Magnetic/exports/so101_tool_contact_board_reference.stl",
        "sourceFile": "so101_tool_contact_board_reference.step",
        "bodyRoots": ["tool_spoon"],
        "geomPrefixes": ["spoon_pad_", "spoon_contact_board_"],
    },
    {
        "id": "spoon-body",
        "label": "Matcha spoon body",
        "category": "Tools",
        "source": "QuickChange/SO101_Magnetic/matcha_tools/exports/so101_matcha_spoon_printed_body.stl",
        "sourceFile": "so101_matcha_spoon_printed_body.step",
        "bodyRoots": ["tool_spoon"],
        "geomPrefixes": ["spoon_carrier_", "spoon_handle_", "spoon_bowl_", "spoon_set_screw_"],
    },
    {
        "id": "whisk-body",
        "label": "Matcha whisk body",
        "category": "Tools",
        "source": "QuickChange/SO101_Magnetic/matcha_tools/exports/so101_matcha_whisk_printed_body.stl",
        "sourceFile": "so101_matcha_whisk_printed_body.step",
        "bodyRoots": ["tool_whisk"],
        "geomPrefixes": ["whisk_housing_", "whisk_electronics_", "whisk_eccentric_", "whisk_carriage_", "whisk_bellows_", "whisk_collet_", "chasen_"],
    },
)


def _xyz_on_station(radius_mm: float, angle: float, z_mm: float) -> tuple[float, float, float]:
    return (
        radius_mm * math.cos(angle),
        radius_mm * math.sin(angle),
        z_mm,
    )


def _cq_planar_capsule(cq, plane, p1, p2, radius_mm: float, thickness_mm: float):
    """A support-friendly rounded bar extruded normal to ``plane``."""

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length <= 1.0e-9:
        raise ValueError("fixture capsule endpoints must differ")
    nx = -dy * radius_mm / length
    ny = dx * radius_mm / length
    result = (
        cq.Workplane(plane)
        .moveTo(p1[0] + nx, p1[1] + ny)
        .lineTo(p2[0] + nx, p2[1] + ny)
        .lineTo(p2[0] - nx, p2[1] - ny)
        .lineTo(p1[0] - nx, p1[1] - ny)
        .close()
        .extrude(thickness_mm / 2.0, both=True)
    )
    for point in (p1, p2):
        result = result.union(
            cq.Workplane(plane)
            .center(point[0], point[1])
            .circle(radius_mm)
            .extrude(thickness_mm / 2.0, both=True)
        )
    return result


def _cut_cylinder(cq, part, point, direction, radius_mm: float, length_mm: float):
    start = cq.Vector(
        point[0] - direction[0] * length_mm / 2.0,
        point[1] - direction[1] * length_mm / 2.0,
        point[2] - direction[2] * length_mm / 2.0,
    )
    cutter = cq.Solid.makeCylinder(
        radius_mm,
        length_mm,
        start,
        cq.Vector(*direction),
    )
    return cq.Workplane(obj=part.val().cut(cutter))


def _build_printable_fixture():
    """Build the rounded, under-slung rack in assembled millimetre coordinates.

    Each passive dock rests on a small two-bolt saddle.  A hollow A-frame
    carries that saddle entirely below the parked tool, so there is no ring or
    post behind a tool for the arm to hit.  Four removable base links turn the
    three stations into one rigid fixture while keeping every printed part
    small enough for a typical desktop printer.
    """

    try:
        import cadquery as cq  # type: ignore
    except ImportError as exc:  # pragma: no cover - export environment contract
        raise RuntimeError(
            "CadQuery is required to regenerate the printable fixture assets"
        ) from exc

    station_data = {}
    station_parts = {}
    saddle_parts = {}
    front_nodes = {}
    rear_nodes = {}

    base_front_mm = 292.0
    base_rear_mm = 365.0
    crown_radius_mm = 340.0
    crown_z_mm = 129.0
    saddle_z_mm = 139.0
    link_z_mm = 11.5

    for tool, (dock_x, dock_y, dock_z) in DOCK_POSITIONS.items():
        angle = math.atan2(dock_y, dock_x)
        radial = (math.cos(angle), math.sin(angle), 0.0)
        tangent = (-math.sin(angle), math.cos(angle), 0.0)
        dock_radius_mm = math.hypot(dock_x, dock_y) * 1000.0
        saddle_radius_mm = dock_radius_mm + 3.0

        # A hollow triangular load path replaces the former solid blocks.  Its
        # 135 mm-high envelope remains at least 16 mm below every dock/tool
        # collision envelope in the parked scene.
        station_plane = cq.Plane(
            origin=(0.0, 0.0, 0.0),
            xDir=radial,
            normal=(math.sin(angle), -math.cos(angle), 0.0),
        )
        front = (base_front_mm, 8.0)
        rear = (base_rear_mm, 8.0)
        crown = (crown_radius_mm, crown_z_mm)
        station = _cq_planar_capsule(cq, station_plane, front, rear, 7.5, 10.0)
        station = station.union(
            _cq_planar_capsule(cq, station_plane, front, crown, 5.5, 10.0)
        )
        station = station.union(
            _cq_planar_capsule(cq, station_plane, rear, crown, 5.5, 10.0)
        )
        station = station.union(
            _cq_planar_capsule(
                cq,
                station_plane,
                crown,
                (dock_radius_mm + 24.0, crown_z_mm),
                6.0,
                10.0,
            )
        )

        # Broad, rounded feet provide a real load path into the bench and a
        # flat mounting face for the removable base links.
        foot_plane = cq.Plane(
            origin=(0.0, 0.0, 4.0),
            xDir=(1.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        )
        for radius_mm in (base_front_mm, base_rear_mm):
            start = (
                radial[0] * radius_mm - tangent[0] * 22.0,
                radial[1] * radius_mm - tangent[1] * 22.0,
            )
            end = (
                radial[0] * radius_mm + tangent[0] * 22.0,
                radial[1] * radius_mm + tangent[1] * 22.0,
            )
            station = station.union(
                _cq_planar_capsule(cq, foot_plane, start, end, 7.0, 8.0)
            )

        # Two M4 inserts in the cantilever clamp the saddle across a 0.5 mm
        # washer gap.  The parts are mechanically joined but never overlap.
        for radial_offset in (25.0, 36.0):
            station = _cut_cylinder(
                cq,
                station,
                _xyz_on_station(dock_radius_mm + radial_offset, angle, crown_z_mm),
                (0.0, 0.0, 1.0),
                2.15,
                30.0,
            )

        # The separate T-shaped saddle prints flat.  Its narrow radial tongue
        # bolts to the A-frame and its crossbar carries two dock mounting
        # bosses.  The central flange approach remains completely open.
        saddle_plane = cq.Plane(
            origin=(0.0, 0.0, saddle_z_mm),
            xDir=(1.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        )
        cross_start = (
            radial[0] * saddle_radius_mm - tangent[0] * 33.0,
            radial[1] * saddle_radius_mm - tangent[1] * 33.0,
        )
        cross_end = (
            radial[0] * saddle_radius_mm + tangent[0] * 33.0,
            radial[1] * saddle_radius_mm + tangent[1] * 33.0,
        )
        tongue_start = (
            radial[0] * saddle_radius_mm,
            radial[1] * saddle_radius_mm,
        )
        tongue_end = (
            radial[0] * (dock_radius_mm + 40.0),
            radial[1] * (dock_radius_mm + 40.0),
        )
        saddle = _cq_planar_capsule(
            cq, saddle_plane, cross_start, cross_end, 6.5, 7.0
        ).union(
            _cq_planar_capsule(cq, saddle_plane, tongue_start, tongue_end, 6.5, 7.0)
        )

        if tool == "gripper":
            # These are the two M4 centres carried by the source core-dock
            # support head (local X -25/+21 mm, local Z +4.75 mm).
            dock_mount_sites = [
                (dock_radius_mm + 4.75, -21.0),
                (dock_radius_mm + 4.75, 25.0),
            ]
        else:
            # Spoon and whisk already expose a rear support anchor at local
            # Y=-65 mm.  A small radial stay pad takes two diagonal M4 straps
            # to that anchor without entering the tool envelope.
            stay_pad_start = (
                radial[0] * (dock_radius_mm - 14.0) + tangent[0] * 27.0,
                radial[1] * (dock_radius_mm - 14.0) + tangent[1] * 27.0,
            )
            stay_pad_end = (
                radial[0] * (dock_radius_mm + 14.0) + tangent[0] * 27.0,
                radial[1] * (dock_radius_mm + 14.0) + tangent[1] * 27.0,
            )
            saddle = saddle.union(
                _cq_planar_capsule(
                    cq, saddle_plane, stay_pad_start, stay_pad_end, 5.5, 7.0
                )
            )
            dock_mount_sites = [
                (dock_radius_mm - 10.0, 27.0),
                (dock_radius_mm + 10.0, 27.0),
            ]

        # Printed bosses stop below the dock collision shell.  M4 screws pass
        # through them into either the source head or the diagonal stay pair.
        for radius_mm, tangent_offset in dock_mount_sites:
            boss_point = (
                radial[0] * radius_mm + tangent[0] * tangent_offset,
                radial[1] * radius_mm + tangent[1] * tangent_offset,
                141.5,
            )
            boss = cq.Solid.makeCylinder(
                4.75,
                8.5,
                cq.Vector(*boss_point),
                cq.Vector(0.0, 0.0, 1.0),
            )
            saddle = cq.Workplane(obj=saddle.val().fuse(boss))

        saddle_holes = [
            (dock_radius_mm + 25.0, 0.0),
            (dock_radius_mm + 36.0, 0.0),
            *dock_mount_sites,
        ]
        for radius_mm, tangent_offset in saddle_holes:
            point = (
                radial[0] * radius_mm + tangent[0] * tangent_offset,
                radial[1] * radius_mm + tangent[1] * tangent_offset,
                saddle_z_mm,
            )
            saddle = _cut_cylinder(
                cq, saddle, point, (0.0, 0.0, 1.0), 2.15, 30.0
            )

        station_parts[f"station_{tool}"] = station.clean()
        saddle_parts[f"saddle_{tool}"] = saddle.clean()
        front_nodes[tool] = _xyz_on_station(base_front_mm, angle, link_z_mm)
        rear_nodes[tool] = _xyz_on_station(base_rear_mm, angle, link_z_mm)
        station_data[tool] = {
            "angle": angle,
            "radial": radial,
            "tangent": tangent,
            "dock_radius_mm": dock_radius_mm,
            "saddle_radius_mm": saddle_radius_mm,
            "base_front_mm": base_front_mm,
            "base_rear_mm": base_rear_mm,
            "crown_radius_mm": crown_radius_mm,
            "crown_z_mm": crown_z_mm,
            "saddle_z_mm": saddle_z_mm,
            "link_z_mm": link_z_mm,
            "dock_mount_sites_mm": dock_mount_sites,
        }

    # The front and rear base links form a light two-rail chassis.  Endpoints
    # are shifted to opposite sides of each station, so adjacent links meet the
    # same foot without consuming the same printed volume.
    link_parts = {}
    station_link_holes = {tool: {"front": [], "rear": []} for tool in DOCK_POSITIONS}
    link_plane = cq.Plane(
        origin=(0.0, 0.0, link_z_mm),
        xDir=(1.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    for rail, nodes in (("front", front_nodes), ("rear", rear_nodes)):
        for first, second in FIXTURE_LINK_PAIRS:
            p1 = nodes[first]
            p2 = nodes[second]
            first_tangent = station_data[first]["tangent"]
            second_tangent = station_data[second]["tangent"]
            first_sign = 1.0 if (
                (p2[0] - p1[0]) * first_tangent[0]
                + (p2[1] - p1[1]) * first_tangent[1]
            ) >= 0.0 else -1.0
            second_sign = 1.0 if (
                (p1[0] - p2[0]) * second_tangent[0]
                + (p1[1] - p2[1]) * second_tangent[1]
            ) >= 0.0 else -1.0
            start = (
                p1[0] + first_tangent[0] * first_sign * 16.0,
                p1[1] + first_tangent[1] * first_sign * 16.0,
            )
            end = (
                p2[0] + second_tangent[0] * second_sign * 16.0,
                p2[1] + second_tangent[1] * second_sign * 16.0,
            )
            link = _cq_planar_capsule(cq, link_plane, start, end, 5.0, 5.5)
            for point in (start, end):
                link = _cut_cylinder(
                    cq,
                    link,
                    (point[0], point[1], link_z_mm),
                    (0.0, 0.0, 1.0),
                    2.15,
                    16.0,
                )
            link_parts[f"{rail}_{first}_{second}"] = link.clean()
            station_link_holes[first][rail].append(start)
            station_link_holes[second][rail].append(end)

    # Drill the matching M4 link holes and two M5 bench-anchor holes in each
    # station only after the non-overlapping link landing points are known.
    for part_name, station in station_parts.items():
        tool = part_name.removeprefix("station_")
        for rail in ("front", "rear"):
            for point in station_link_holes[tool][rail]:
                station = _cut_cylinder(
                    cq,
                    station,
                    (point[0], point[1], 4.0),
                    (0.0, 0.0, 1.0),
                    2.15,
                    24.0,
                )
        angle = station_data[tool]["angle"]
        for radius_mm in (base_front_mm, base_rear_mm):
            station = _cut_cylinder(
                cq,
                station,
                _xyz_on_station(radius_mm, angle, 4.0),
                (0.0, 0.0, 1.0),
                2.75,
                24.0,
            )
        station_parts[part_name] = station.clean()

    parts = {**station_parts, **saddle_parts, **link_parts}
    part_metadata = []
    total_volume_mm3 = 0.0
    largest_dimension_mm = 0.0
    for name, part in parts.items():
        shape = part.val()
        solids = list(shape.Solids())
        if not shape.isValid() or len(solids) != 1:
            raise RuntimeError(f"fixture part {name} is not one valid printable solid")
        volume_mm3 = float(shape.Volume())
        total_volume_mm3 += volume_mm3
        bounds = shape.BoundingBox()
        envelope = sorted([bounds.xlen, bounds.ylen, bounds.zlen], reverse=True)
        largest_dimension_mm = max(largest_dimension_mm, envelope[0])
        if name.startswith("station_"):
            orientation = "print upright on rounded feet; support the short cantilever"
        elif name.startswith("saddle_"):
            orientation = "lay flat with dock bosses facing up"
        else:
            orientation = "lay broad face on bed"
        part_metadata.append(
            {
                "name": name,
                "stl": f"printable_fixture_{name}.stl",
                "volumeMm3": round(volume_mm3, 3),
                "assembledBoundsMm": [
                    round(bounds.xlen, 3),
                    round(bounds.ylen, 3),
                    round(bounds.zlen, 3),
                ],
                "printEnvelopeMm": [round(value, 2) for value in envelope],
                "printOrientation": orientation,
            }
        )

    # Separate printed parts may touch at their intended bolted faces, but may
    # not consume the same volume.  This catches the visual intersections that
    # made the earlier block fixture physically impossible.
    overlaps = []
    for (name_a, part_a), (name_b, part_b) in combinations(parts.items(), 2):
        intersection = part_a.val().intersect(part_b.val())
        overlap_mm3 = float(intersection.Volume())
        if overlap_mm3 > 0.05:
            bounds = intersection.BoundingBox()
            overlaps.append(
                (
                    name_a,
                    name_b,
                    overlap_mm3,
                    (bounds.xlen, bounds.ylen, bounds.zlen),
                )
            )
    if overlaps:
        raise RuntimeError(f"printable fixture parts overlap: {overlaps[:4]}")

    fixture_metadata = {
        "design": "modular-rounded-underslung-truss-v2",
        "units": "mm",
        "totalPrintedVolumeCm3": round(total_volume_mm3 / 1000.0, 2),
        "largestPrintDimensionMm": round(largest_dimension_mm, 2),
        "partCount": len(parts),
        "parts": part_metadata,
        "hardware": {
            "thread": "M4",
            "baseLinkFasteners": 8,
            "saddleFasteners": 6,
            "dockMountFasteners": 10,
            "benchAnchors": "6 x M5",
            "recommended": "M4 screws, heat-set inserts, 0.75 mm link washers, 0.5 mm saddle washers, and two 3.5 mm metal stays per side-anchor dock",
        },
        "assembly": "three under-slung A-frame stations, three T-saddles, two front links, and two rear links",
        "validation": {
            "allPartsValidSingleSolids": True,
            "printedPartOverlapMm3": 0.0,
            "largestPrintDimensionMm": round(largest_dimension_mm, 2),
            "minimumDockClearanceMm": 1.0,
            "stationTopZMm": 135.0,
        },
    }
    return cq, parts, station_data, fixture_metadata


def _fixture_asset_bytes(cq, parts) -> dict[str, bytes]:
    assets = {}
    with tempfile.TemporaryDirectory(prefix="quickattach-fixture-") as directory:
        for name, part in parts.items():
            filename = f"printable_fixture_{name}.stl"
            path = Path(directory) / filename
            cq.exporters.export(
                part,
                str(path),
                tolerance=0.65,
                angularTolerance=0.22,
            )
            assets[filename] = path.read_bytes()
    return assets


def _add_geom(
    body: ET.Element,
    *,
    name: str,
    geom_type: str,
    rgba: str,
    collision: bool,
    pos: tuple[float, float, float] | None = None,
    size: tuple[float, ...] | None = None,
    fromto: tuple[float, float, float, float, float, float] | None = None,
    quat: tuple[float, float, float, float] | None = None,
    mesh: str | None = None,
    group: int | None = None,
) -> None:
    attributes = {
        "name": name,
        "type": geom_type,
        "rgba": rgba,
        "group": str(group if group is not None else (3 if collision else 2)),
        "contype": "1" if collision else "0",
        "conaffinity": "1" if collision else "0",
        "mass": "0",
    }
    if pos is not None:
        attributes["pos"] = " ".join(f"{value:.7g}" for value in pos)
    if size is not None:
        attributes["size"] = " ".join(f"{value:.7g}" for value in size)
    if fromto is not None:
        attributes["fromto"] = " ".join(f"{value:.7g}" for value in fromto)
    if quat is not None:
        attributes["quat"] = " ".join(f"{value:.9g}" for value in quat)
    if mesh is not None:
        attributes["mesh"] = mesh
    ET.SubElement(body, "geom", attributes)


def _add_capsule(
    body: ET.Element,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    *,
    collision: bool,
    rgba: str,
    group: int | None = None,
) -> None:
    _add_geom(
        body,
        name=name,
        geom_type="capsule",
        rgba=rgba,
        collision=collision,
        size=(radius,),
        fromto=(*start, *end),
        group=group,
    )


def _yaw_quat(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _add_box_segment(
    body: ET.Element,
    name: str,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    z: float,
    half_width: float,
    half_height: float,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    _add_geom(
        body,
        name=name,
        geom_type="box",
        rgba="0.10 0.52 0.68 0.18",
        collision=True,
        pos=((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, z),
        size=(length / 2.0, half_width, half_height),
        quat=_yaw_quat(math.atan2(dy, dx)),
    )


def _presentation_xml(xml: str, fixture_parts, station_data) -> str:
    """Reduce the validation workcell to a robot and one supported tool rack."""

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MuJoCo XML is missing worldbody")

    for body in list(worldbody.findall("body")):
        if body.get("name") in REMOVED_WORKCELL_BODIES:
            worldbody.remove(body)

    # The source model gave each dock an independent floor support. Replace
    # the long floor members with one shared fixture, while retaining the
    # source head/rear-anchor datums that physically fasten each dock to it.
    retained_dock_adapters = {
        "dock_gripper_floor_support_head_collision",
        "dock_gripper_floor_support_reinforcement_collision",
        "dock_spoon_support_anchor_collision",
        "dock_whisk_support_anchor_collision",
    }
    for body in worldbody.iter("body"):
        for geom in list(body.findall("geom")):
            name = geom.get("name", "")
            if name.startswith("dock_gripper_floor_support_") and name not in retained_dock_adapters:
                body.remove(geom)

    equality = root.find("equality")
    if equality is not None:
        for constraint in list(equality):
            if constraint.get("name") in {"sieve_top_latch", "sieve_grasp"}:
                equality.remove(constraint)

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    for part_name in fixture_parts:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": f"printable_fixture_{part_name}",
                "file": f"printable_fixture_{part_name}.stl",
                "scale": "0.001 0.001 0.001",
            },
        )

    fixture = ET.SubElement(worldbody, "body", {"name": "radial_three_tool_fixture"})
    for part_name in fixture_parts:
        if part_name.startswith("saddle_"):
            rgba = "0.18 0.46 0.50 1"
        elif part_name.startswith("station_"):
            rgba = "0.08 0.32 0.38 1"
        else:
            rgba = "0.12 0.38 0.43 1"
        _add_geom(
            fixture,
            name=f"tool_fixture_{part_name}_visual",
            geom_type="mesh",
            rgba=rgba,
            collision=False,
            mesh=f"printable_fixture_{part_name}",
        )

    # Lightweight rounded proxies serve two purposes: inexpensive contact
    # validation and an honest non-blocky representation in browsers that fall
    # back to the SVG renderer.  WebGL uses the printable STL meshes above.
    fixture_visual = "0.08 0.32 0.38 1"
    saddle_visual = "0.18 0.46 0.50 1"
    link_visual = "0.12 0.38 0.43 1"
    collision_visual = "0.10 0.52 0.68 0.18"

    def add_proxy_capsule(
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float,
        rgba: str,
    ) -> None:
        _add_capsule(
            fixture,
            f"{name}_fallback_visual",
            start,
            end,
            radius,
            collision=False,
            rgba=rgba,
            group=4,
        )
        _add_capsule(
            fixture,
            f"{name}_collision",
            start,
            end,
            radius,
            collision=True,
            rgba=collision_visual,
        )

    def add_proxy_cylinder(
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float,
        rgba: str,
    ) -> None:
        for suffix, collision, group, color in (
            ("fallback_visual", False, 4, rgba),
            ("collision", True, 3, collision_visual),
        ):
            _add_geom(
                fixture,
                name=f"{name}_{suffix}",
                geom_type="cylinder",
                rgba=color,
                collision=collision,
                size=(radius,),
                fromto=(*start, *end),
                group=group,
            )

    def add_hardware_capsule(
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float = 0.0035,
    ) -> None:
        for suffix, collision, group, color in (
            ("visual", False, 2, "0.62 0.69 0.72 1"),
            ("fallback_visual", False, 4, "0.62 0.69 0.72 1"),
            ("collision", True, 3, collision_visual),
        ):
            _add_capsule(
                fixture,
                f"{name}_{suffix}",
                start,
                end,
                radius,
                collision=collision,
                rgba=color,
                group=group,
            )

    def add_adapter_box(
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        angle: float,
    ) -> None:
        for suffix, group in (("visual", 2), ("fallback_visual", 4)):
            _add_geom(
                fixture,
                name=f"{name}_{suffix}",
                geom_type="box",
                rgba="0.48 0.56 0.60 1",
                collision=False,
                pos=pos,
                size=size,
                quat=_yaw_quat(angle),
                group=group,
            )

    front_nodes = {}
    rear_nodes = {}
    for tool, data in station_data.items():
        angle = data["angle"]
        radial = (math.cos(angle), math.sin(angle))
        tangent = (-math.sin(angle), math.cos(angle))
        dock_radius = data["dock_radius_mm"] / 1000.0
        saddle_radius = data["saddle_radius_mm"] / 1000.0
        base_front = data["base_front_mm"] / 1000.0
        base_rear = data["base_rear_mm"] / 1000.0
        crown_radius = data["crown_radius_mm"] / 1000.0
        crown_z = data["crown_z_mm"] / 1000.0
        saddle_z = data["saddle_z_mm"] / 1000.0
        link_z = data["link_z_mm"] / 1000.0

        def point(radius: float, z: float) -> tuple[float, float, float]:
            return (radius * radial[0], radius * radial[1], z)

        def offset_point(
            radius: float, tangent_offset: float, z: float
        ) -> tuple[float, float, float]:
            return (
                radius * radial[0] + tangent_offset * tangent[0],
                radius * radial[1] + tangent_offset * tangent[1],
                z,
            )

        front = point(base_front, 0.008)
        rear = point(base_rear, 0.008)
        crown = point(crown_radius, crown_z)
        for member_name, start, end, radius in (
            ("base", front, rear, 0.008),
            ("front_rib", front, crown, 0.006),
            ("rear_rib", rear, crown, 0.006),
            ("cantilever", crown, point(dock_radius + 0.024, crown_z), 0.006),
        ):
            add_proxy_capsule(
                f"tool_fixture_{tool}_{member_name}",
                start,
                end,
                radius,
                fixture_visual,
            )

        for rail_name, rail_radius in (("front", base_front), ("rear", base_rear)):
            add_proxy_capsule(
                f"tool_fixture_{tool}_{rail_name}_foot",
                offset_point(rail_radius, -0.022, 0.006),
                offset_point(rail_radius, 0.022, 0.006),
                0.006,
                fixture_visual,
            )

        add_proxy_capsule(
            f"tool_fixture_{tool}_saddle_crossbar",
            offset_point(saddle_radius, -0.033, saddle_z),
            offset_point(saddle_radius, 0.033, saddle_z),
            0.005,
            saddle_visual,
        )
        add_proxy_capsule(
            f"tool_fixture_{tool}_saddle_tongue",
            point(saddle_radius, saddle_z),
            point(dock_radius + 0.040, saddle_z),
            0.005,
            saddle_visual,
        )

        mount_sites = [
            (radius_mm / 1000.0, tangent_mm / 1000.0)
            for radius_mm, tangent_mm in data["dock_mount_sites_mm"]
        ]
        if tool != "gripper":
            add_proxy_capsule(
                f"tool_fixture_{tool}_stay_pad",
                offset_point(dock_radius - 0.014, 0.027, saddle_z),
                offset_point(dock_radius + 0.014, 0.027, saddle_z),
                0.005,
                saddle_visual,
            )

        for mount_index, (mount_radius, tangent_offset) in enumerate(mount_sites):
            add_proxy_cylinder(
                f"tool_fixture_{tool}_dock_boss_{mount_index}",
                offset_point(mount_radius, tangent_offset, 0.1415),
                offset_point(mount_radius, tangent_offset, 0.1500),
                0.005,
                "0.58 0.64 0.68 1",
            )

        if tool == "gripper":
            # Retain and visibly bridge into the source core-dock head.  Its
            # M4 centres are hash-bound in the upstream CAD contract.
            add_adapter_box(
                "tool_fixture_gripper_source_head",
                offset_point(dock_radius + 0.00475, 0.004, 0.158),
                (0.0215, 0.028, 0.004),
                angle,
            )
            for mount_index, (mount_radius, tangent_offset) in enumerate(mount_sites):
                add_hardware_capsule(
                    f"tool_fixture_gripper_head_standoff_{mount_index}",
                    offset_point(mount_radius, tangent_offset, 0.1500),
                    offset_point(mount_radius, tangent_offset, 0.1535),
                    0.0038,
                )
        else:
            # The source rear-anchor block remains owned by the passive dock.
            # Two short metal stays close the printed saddle-to-anchor path.
            add_adapter_box(
                f"tool_fixture_{tool}_source_rear_anchor",
                offset_point(dock_radius, 0.065, 0.1971),
                (0.020, 0.010, 0.020),
                angle,
            )
            for mount_index, (mount_radius, tangent_offset) in enumerate(mount_sites):
                add_hardware_capsule(
                    f"tool_fixture_{tool}_anchor_stay_{mount_index}",
                    offset_point(mount_radius, tangent_offset, 0.1500),
                    offset_point(mount_radius, 0.055, 0.1745),
                )

        front_nodes[tool] = (base_front * radial[0], base_front * radial[1])
        rear_nodes[tool] = (base_rear * radial[0], base_rear * radial[1])

    for rail, nodes in (("front", front_nodes), ("rear", rear_nodes)):
        for first, second in FIXTURE_LINK_PAIRS:
            p1 = nodes[first]
            p2 = nodes[second]
            first_tangent = station_data[first]["tangent"]
            second_tangent = station_data[second]["tangent"]
            first_sign = 1.0 if (
                (p2[0] - p1[0]) * first_tangent[0]
                + (p2[1] - p1[1]) * first_tangent[1]
            ) >= 0.0 else -1.0
            second_sign = 1.0 if (
                (p1[0] - p2[0]) * second_tangent[0]
                + (p1[1] - p2[1]) * second_tangent[1]
            ) >= 0.0 else -1.0
            start = (
                p1[0] + first_tangent[0] * first_sign * 0.016,
                p1[1] + first_tangent[1] * first_sign * 0.016,
            )
            end = (
                p2[0] + second_tangent[0] * second_sign * 0.016,
                p2[1] + second_tangent[1] * second_sign * 0.016,
            )
            z = station_data[first]["link_z_mm"] / 1000.0
            add_proxy_capsule(
                f"tool_fixture_{rail}_{first}_{second}",
                (start[0], start[1], z),
                (end[0], end[1], z),
                0.005,
                link_visual,
            )

    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="QuickAttach repository root (defaults to this checkout)",
    )
    parser.add_argument("--output", type=Path, default=Path("public/model"))
    args = parser.parse_args()

    sim_dir = args.repo / "QuickChange" / "SO101_Magnetic" / "sim"
    sys.path.insert(0, str(sim_dir))

    import mujoco  # type: ignore
    import matcha_workflow_demo as demo  # type: ignore

    xml, assets = demo._build_xml_and_assets()
    cq, fixture_parts, station_data, fixture_metadata = _build_printable_fixture()
    fixture_assets = _fixture_asset_bytes(cq, fixture_parts)
    assets.update(fixture_assets)
    xml = _presentation_xml(xml, fixture_parts, station_data)
    xml = "\n".join(line.rstrip() for line in xml.splitlines()) + "\n"
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cad_output = output / "cad"
    cad_output.mkdir(parents=True, exist_ok=True)
    for stale_cad in cad_output.glob("*.stl"):
        stale_cad.unlink()

    cad_assets = []
    for record in CAD_PARTS:
        source_path = args.repo / record["source"]
        if not source_path.is_file():
            raise RuntimeError(f"CAD inspector source is missing: {source_path}")
        asset_name = f"cad/{record['id']}.stl"
        shutil.copyfile(source_path, output / asset_name)
        cad_assets.append(
            {
                **{key: value for key, value in record.items() if key != "source"},
                "asset": asset_name,
            }
        )

    for part in fixture_metadata["parts"]:
        name = part["name"]
        if name.startswith("station_"):
            tool = name.removeprefix("station_")
            prefixes = [
                f"tool_fixture_{tool}_{member}_"
                for member in (
                    "base",
                    "front_rib",
                    "rear_rib",
                    "cantilever",
                    "front_foot",
                    "rear_foot",
                )
            ]
        elif name.startswith("saddle_"):
            tool = name.removeprefix("saddle_")
            prefixes = [
                f"tool_fixture_{tool}_saddle_",
                f"tool_fixture_{tool}_stay_pad_",
                f"tool_fixture_{tool}_dock_boss_",
            ]
        else:
            prefixes = [f"tool_fixture_{name}_"]
        cad_assets.append(
            {
                "id": f"fixture-{name.replace('_', '-')}",
                "label": name.replace("_", " ").title(),
                "category": "Printable fixture",
                "asset": part["stl"],
                "sourceFile": part["stl"],
                "bodyRoots": ["radial_three_tool_fixture"],
                "geomPrefixes": prefixes,
                "printEnvelopeMm": part["printEnvelopeMm"],
                "volumeMm3": part["volumeMm3"],
            }
        )

    for stale_fixture in output.glob("printable_fixture_*.stl"):
        if stale_fixture.name not in fixture_assets:
            stale_fixture.unlink()
    (output / "model.xml").write_text(xml, encoding="utf-8")

    asset_names: list[str] = []
    for name, contents in sorted(assets.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        asset_names.append(name)

    joints = {}
    for name in (*demo.ARM_JOINTS, "gripper", "qc_positive_lock_slider_joint"):
        joint = model.joint(name)
        joints[name] = {
            "range": [float(value) for value in joint.range],
            "qposAddress": int(joint.qposadr[0]),
        }

    manifest = {
        "schemaVersion": 1,
        "source": "QuickAttach/QuickChange/SO101_Magnetic/sim/matcha_workflow_demo.py",
        "releaseReady": False,
        "qualification": "digital validation only; physical qualification remains open",
        "xmlSha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "assets": asset_names,
        "counts": {
            "bodies": int(model.nbody),
            "geoms": int(model.ngeom),
            "meshes": int(model.nmesh),
            "joints": int(model.njnt),
        },
        "armJoints": list(demo.ARM_JOINTS),
        "joints": joints,
        "presets": {
            "home": [0.0, -0.5, 0.8, -0.3, 0.0],
            "parked": [-0.07, -1.65, 0.02, 1.35, 0.0],
            "gripperDock": [float(value) for value in demo.DOCK_PRE_CAPTURE_Q["gripper"]],
            "spoonDock": [float(value) for value in demo.DOCK_PRE_CAPTURE_Q["spoon"]],
            "whiskDock": [float(value) for value in demo.DOCK_PRE_CAPTURE_Q["whisk"]],
        },
        "dockCapture": {
            tool: [float(value) for value in values]
            for tool, values in demo.DOCK_CAPTURE_Q.items()
        },
        "tools": ["gripper", "spoon", "whisk"],
        "cadAssets": cad_assets,
        "fixture": fixture_metadata,
        "camera": {
            "name": demo.CAMERA_NAME,
            "position": [float(value) for value in model.cam_pos[0]],
            "quaternionWxyz": [float(value) for value in model.cam_quat[0]],
            "verticalFovDegrees": float(model.cam_fovy[0]),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "printable_fixture_manifest.json").write_text(
        json.dumps(fixture_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(asset_names)} assets, {model.ngeom} geoms, "
        f"and XML {manifest['xmlSha256'][:12]}… to {output}"
    )


if __name__ == "__main__":
    main()
