#!/usr/bin/env python3
"""Compact feature-aware collision installation for the SO-101 quick-change.

The runtime model intentionally keeps the calibrated upstream link meshes and
adds small analytic solids only for functional quick-change features.  It does
not load an undifferentiated hundreds-of-parts decomposition.  Exact CAD and
continuous clearance remain separate validation authorities.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from functools import cache
from pathlib import Path


SIGNALS = ("ground", "power", "data", "id")
SIGNAL_Y_M = {
    "ground": -0.0075,
    "power": -0.0025,
    "data": 0.0025,
    "id": 0.0075,
}
SIGNAL_TO_BORE = {"ground": 1, "power": 2, "data": 3, "id": 4}

# Frozen source contracts.  The stock-gripper dock is the released core
# quick-change design; spoon and whisk use the separate two-bay matcha rack.
# Keeping these datums distinct prevents the convenient recovered rack box
# from silently becoming collision authority for three different parts.
CORE_DOCK_STOP_BOUNDS_M = (
    (-0.045, 0.037),
    (0.026, 0.032),
    (-0.003, 0.0125),
)
CORE_DOCK_STOP_HOLES_M = (
    (-0.025, 0.00475, 0.0022),
    (0.021, 0.00475, 0.0022),
)
MATCHA_DOCK_STOP_BOUNDS_M = (
    (-0.041, 0.033),
    (0.025, 0.031),
    (-0.003, 0.0125),
)
CORE_DOCK_CAM_POLYGON_M = (
    (0.028, -0.016),
    (0.034, -0.016),
    (0.034, 0.0),
    (0.02405, 0.0),
)
CORE_DOCK_CAM_Z_BOUNDS_M = (-0.00415, -0.00195)
MATCHA_DOCK_CAM_POLYGON_M = (
    (0.028, -0.017),
    (0.034, -0.017),
    (0.034, 0.001),
    (0.024, 0.001),
)
MATCHA_DOCK_CAM_Z_BOUNDS_M = (-0.0042, -0.0020)

# The released core plate carries a local fixed-side recess around the full
# passive-cam sweep.  Values are the published 0.50 mm guarded source cutter;
# plate partitions below naturally clip it to finite plate material.
ROBOT_CAM_RELIEF_BOUNDS_M = (
    (0.02355, 0.025),
    (-0.0165, 0.0245),
    (0.0, 0.0095),
)

# A 0.20 mm Z partition leaves a conservative staircase around each round
# core-stop passage.  Its maximum planar miss is below the 0.35 mm runtime
# proxy release limit, while no box ever fills source hole material.
CORE_STOP_HOLE_PARTITION_STEP_M = 0.0002

# The lock slider is a thin, non-convex sheet with two keyholes and a guide
# slot.  One MuJoCo mesh geom would convexify it and silently fill those voids,
# so the source top face is decomposed into convex triangular prisms.  The STEP
# hash and absolute OCCT meshing controls are part of the runtime contract.
POSITIVE_LOCK_SLIDER_STEP = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "so101_positive_lock_slider.step"
)
POSITIVE_LOCK_SLIDER_STEP_SHA256 = (
    "37771b10b4fe82614f0f7b460d44cdc81ea8505b4e6c7ac7b20a1f413b7ca848"
)
POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM = 0.005
POSITIVE_LOCK_SLIDER_ANGULAR_DEFLECTION_RAD = 0.3
POSITIVE_LOCK_SLIDER_VOID_EXPANSION_MM = 0.010
POSITIVE_LOCK_SLIDER_Z_BOUNDS_M = (0.0, 0.0016)
# The MJCF joint reference is the physical locked q=3 mm state.  Placing the
# body at +3 mm makes q=3 retain locked source geometry while q=0 applies the
# expected -3 mm displacement back to the unlocked source profile.
POSITIVE_LOCK_SLIDER_BASE_POS_M = (0.003, 0.0, 0.0047)
POSITIVE_LOCK_SLIDER_JOINT_RANGE_M = (0.0, 0.003)
POSITIVE_LOCK_SLIDER_SPRINGREF_M = 0.0036
POSITIVE_LOCK_SLIDER_STIFFNESS_N_M = 980.0
POSITIVE_LOCK_SLIDER_DENSITY_KG_M3 = 8000.0
POSITIVE_LOCK_HARDWARE_DENSITY_KG_M3 = 8000.0
POSITIVE_LOCK_SLIDER_SOURCE_VOLUME_MM3 = 220.12468083955645
POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG = 0.0017609974467164514
POSITIVE_LOCK_SLIDER_SOURCE_COM_M = (0.00482206920910109, 0.0, 0.0008)
POSITIVE_LOCK_SLIDER_SOURCE_INERTIA_KG_M2 = (
    7.815117222854e-9,
    2.266197667455e-7,
    2.336835250577e-7,
)
POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2 = (
    *POSITIVE_LOCK_SLIDER_SOURCE_INERTIA_KG_M2,
    0.0,
    0.0,
    0.0,
)

# The positive-lock hardware is a rigid, tool-owned source assembly.  Its
# dynamics are derived from the two hash-pinned shoulder-screw and holed-nut
# STEP sources in the tool-root frame, never from overlapping collision
# primitives.  Each nut is translated +1.5 mm along tool-local Z; the two
# screw/nut stacks are translated to X=+/-12 mm, Y=0.
POSITIVE_LOCK_SHOULDER_SCREW_STEP = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "hardware_McMaster_90318A720_shoulder_screw.step"
)
POSITIVE_LOCK_SHOULDER_SCREW_STEP_SHA256 = (
    "1a302d3674952d881df75e25e47b64a60b23a72b66fde48c35a0924ca1df6990"
)
POSITIVE_LOCK_STUD_NUT_STEP = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "hardware_DIN934_M3_lock_stud_nut.step"
)
POSITIVE_LOCK_STUD_NUT_STEP_SHA256 = (
    "7f19e5abdb33083e7d179df78d5dc1a130140eadb74a3b22aa2d8f3c078266c7"
)
POSITIVE_LOCK_HARDWARE_STACK_X_M = (-0.012, 0.012)
POSITIVE_LOCK_HARDWARE_NUT_TRANSLATION_M = (0.0, 0.0, 0.0015)
POSITIVE_LOCK_HARDWARE_SOURCE_VOLUME_MM3 = 342.8686400999023
POSITIVE_LOCK_HARDWARE_SOURCE_MASS_KG = 0.0027429491207992185
POSITIVE_LOCK_HARDWARE_SOURCE_COM_M = (0.0, 0.0, -0.0011115796404887526)
POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2 = (
    3.6173174131513016e-8,
    4.31157847526606e-7,
    4.0398189716095705e-7,
)
POSITIVE_LOCK_HARDWARE_SOURCE_FULL_INERTIA_KG_M2 = (
    *POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2,
    0.0,
    0.0,
    0.0,
)


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
    size: tuple[float, ...] | None,
    rgba: str,
    quat: str | None = None,
    collision: bool = True,
    group: int = 3,
    **attributes: str,
) -> ET.Element:
    record = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(f"{value:.12g}" for value in pos),
        "rgba": rgba,
        "group": str(group),
        "contype": "1" if collision else "0",
        "conaffinity": "1" if collision else "0",
    }
    if size is not None:
        record["size"] = " ".join(f"{value:.12g}" for value in size)
    if quat is not None:
        record["quat"] = quat
    record.update(attributes)
    return ET.SubElement(parent, "geom", record)


def _add_convex_prism_mesh(
    asset: ET.Element,
    *,
    name: str,
    polygon_xy: tuple[tuple[float, float], ...],
    z_bounds: tuple[float, float],
) -> str:
    """Install one deterministic convex source-contract prism mesh."""

    if len(polygon_xy) < 3:
        raise ValueError("a prism needs at least three polygon vertices")
    existing = asset.find(f"./mesh[@name='{name}']")
    if existing is not None:
        return name
    z_min, z_max = z_bounds
    vertices = [(*point, z_min) for point in polygon_xy]
    vertices.extend((*point, z_max) for point in polygon_xy)
    count = len(polygon_xy)
    faces: list[tuple[int, int, int]] = []
    # Input polygons are counter-clockwise.  Reverse the lower cap, retain the
    # upper cap and triangulate each side with deterministic vertex ordering.
    for index in range(1, count - 1):
        faces.append((0, index + 1, index))
        faces.append((count, count + index, count + index + 1))
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following))
        faces.append((index, count + following, count + index))
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": name,
            "vertex": " ".join(
                f"{coordinate:.12g}" for vertex in vertices for coordinate in vertex
            ),
            "face": " ".join(str(item) for face in faces for item in face),
        },
    )
    return name


def _expand_slider_void_boundary_vertex_mm(
    x_value: float, y_value: float
) -> tuple[float, float]:
    """Move one known STEP void-boundary node conservatively into material.

    OCCT places polygon nodes on exact circular hole boundaries.  Their chords
    lie inside the void and would therefore add a few microns of false runtime
    material.  Expanding only identified inner-boundary nodes by 10 um makes
    every chord conservative; the expansion plus OCCT's reported faceting
    error remains below the 20 um functional-interface budget.
    """

    margin = POSITIVE_LOCK_SLIDER_VOID_EXPANSION_MM
    boundary_tolerance = 1.0e-6

    # Two 6.5 mm entry holes.  Only exposed circular arcs occur on the final
    # face boundary; straight neck portions are handled below.
    for center_x in (-12.0, 12.0):
        dx = x_value - center_x
        radius = math.hypot(dx, y_value)
        if abs(radius - 3.25) <= boundary_tolerance:
            scale = (3.25 + margin) / radius
            return center_x + dx * scale, y_value * scale

    # M2 guide slot: overall length 5.4 mm, width 2.4 mm, centered at x=-1.5.
    # CadQuery's slot has cap centers x=-3 and x=0.
    for center_x, cap_side in ((-3.0, "left"), (0.0, "right")):
        dx = x_value - center_x
        radius = math.hypot(dx, y_value)
        on_exposed_cap = (
            cap_side == "left" and x_value <= center_x + boundary_tolerance
        ) or (cap_side == "right" and x_value >= center_x - boundary_tolerance)
        if on_exposed_cap and abs(radius - 1.2) <= boundary_tolerance:
            scale = (1.2 + margin) / radius
            return center_x + dx * scale, y_value * scale
    if -3.0 - boundary_tolerance <= x_value <= boundary_tolerance and abs(
        abs(y_value) - 1.2
    ) <= boundary_tolerance:
        return x_value, math.copysign(1.2 + margin, y_value)

    # Each released keyhole neck is a 7.25 x 4.25 mm capsule whose centreline
    # runs from stud_x-3 mm (locked shoulder centre) to stud_x (unlocked
    # shoulder centre).  The entry-circle test above owns the larger R3.25
    # boundary around the unlocked endpoint.  Expand the exposed R2.125 left
    # cap and straight flanks into source material; boundary-node filtering in
    # the caller ensures covered/internal capsule arcs are never adjusted.
    neck_radius = 2.125
    for stud_x in (-12.0, 12.0):
        locked_center_x = stud_x - 3.0
        dx = x_value - locked_center_x
        radius = math.hypot(dx, y_value)
        if (
            x_value <= locked_center_x + boundary_tolerance
            and abs(radius - neck_radius) <= boundary_tolerance
        ):
            scale = (neck_radius + margin) / radius
            return locked_center_x + dx * scale, y_value * scale
        if (
            locked_center_x - boundary_tolerance
            <= x_value
            <= stud_x + boundary_tolerance
            and abs(abs(y_value) - neck_radius) <= boundary_tolerance
        ):
            return x_value, math.copysign(neck_radius + margin, y_value)
    return x_value, y_value


@cache
def positive_lock_slider_profile_triangles_m() -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
]:
    """Return a deterministic conservative triangle partition of the slider.

    STEP import is the only B-rep operation in this runtime builder.  Collision
    is performed by MuJoCo against the returned convex prisms; no OCCT Boolean
    result is used as a runtime contact decision.
    """

    source_bytes = POSITIVE_LOCK_SLIDER_STEP.read_bytes()
    observed_hash = hashlib.sha256(source_bytes).hexdigest()
    if observed_hash != POSITIVE_LOCK_SLIDER_STEP_SHA256:
        raise RuntimeError(
            "positive-lock slider STEP hash mismatch: "
            f"expected {POSITIVE_LOCK_SLIDER_STEP_SHA256}, got {observed_hash}"
        )

    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    shape = cq.importers.importStep(str(POSITIVE_LOCK_SLIDER_STEP)).val().wrapped
    BRepTools.Clean_s(shape)
    mesher = BRepMesh_IncrementalMesh(
        shape,
        POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM,
        False,
        POSITIVE_LOCK_SLIDER_ANGULAR_DEFLECTION_RAD,
        False,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("absolute OCCT slider tessellation did not complete")

    top_nodes_mm: list[tuple[float, float, float]] | None = None
    top_faces: list[tuple[int, int, int]] | None = None
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            nodes = [
                triangulation.Node(index).Transformed(transform)
                for index in range(1, triangulation.NbNodes() + 1)
            ]
            if nodes and min(float(point.Z()) for point in nodes) >= 1.6 - 1.0e-8:
                if top_nodes_mm is not None:
                    raise RuntimeError("slider STEP has more than one top face")
                top_nodes_mm = [
                    (float(point.X()), float(point.Y()), float(point.Z()))
                    for point in nodes
                ]
                top_faces = [
                    tuple(int(value) - 1 for value in triangulation.Triangle(index).Get())
                    for index in range(1, triangulation.NbTriangles() + 1)
                ]
        explorer.Next()
    if top_nodes_mm is None or top_faces is None:
        raise RuntimeError("slider STEP top-face triangulation is absent")

    edge_counts: dict[tuple[int, int], int] = {}
    for face in top_faces:
        for index in range(3):
            edge = tuple(sorted((face[index], face[(index + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_nodes = {
        node_index
        for edge, count in edge_counts.items()
        if count == 1
        for node_index in edge
    }
    adjusted_xy_mm = [(x_value, y_value) for x_value, y_value, _ in top_nodes_mm]
    for index in boundary_nodes:
        adjusted_xy_mm[index] = _expand_slider_void_boundary_vertex_mm(
            *adjusted_xy_mm[index]
        )

    triangles: list[tuple[tuple[float, float], ...]] = []
    for face in top_faces:
        polygon_mm = [adjusted_xy_mm[index] for index in face]
        signed_twice_area = sum(
            polygon_mm[index][0] * polygon_mm[(index + 1) % 3][1]
            - polygon_mm[(index + 1) % 3][0] * polygon_mm[index][1]
            for index in range(3)
        )
        if abs(signed_twice_area) <= 1.0e-12:
            raise RuntimeError("slider source triangulation contains a degenerate face")
        if signed_twice_area < 0.0:
            polygon_mm.reverse()
        polygon_m = tuple(
            (float(x_value) * 1.0e-3, float(y_value) * 1.0e-3)
            for x_value, y_value in polygon_mm
        )
        first = min(range(3), key=lambda index: polygon_m[index])
        canonical = tuple(polygon_m[(first + offset) % 3] for offset in range(3))
        triangles.append(canonical)
    triangles.sort()
    if len(triangles) < 100 or len(set(triangles)) != len(triangles):
        raise RuntimeError("slider triangle partition is incomplete or duplicated")
    return tuple(triangles)


def add_positive_lock_slider(frame: ET.Element, asset: ET.Element) -> list[str]:
    """Add the spring-return slider as a moving union of convex prisms."""

    body = ET.SubElement(
        frame,
        "body",
        {
            "name": "qc_positive_lock_slider",
            "pos": " ".join(f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_BASE_POS_M),
        },
    )
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": " ".join(
                f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_SOURCE_COM_M
            ),
            "mass": f"{POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG:.15g}",
            "fullinertia": " ".join(
                f"{value:.15g}"
                for value in POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2
            ),
        },
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": "qc_positive_lock_slider_joint",
            "type": "slide",
            "axis": "1 0 0",
            "range": " ".join(
                f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_JOINT_RANGE_M
            ),
            "limited": "true",
            "ref": f"{POSITIVE_LOCK_SLIDER_JOINT_RANGE_M[1]:.12g}",
            "stiffness": f"{POSITIVE_LOCK_SLIDER_STIFFNESS_N_M:.12g}",
            "springref": f"{POSITIVE_LOCK_SLIDER_SPRINGREF_M:.12g}",
        },
    )
    names: list[str] = []
    role_counts = {"bridge": 0, "left_lobe": 0, "right_lobe": 0, "tab": 0}
    for triangle in positive_lock_slider_profile_triangles_m():
        centroid_x = sum(point[0] for point in triangle) / 3.0
        centroid_y = sum(point[1] for point in triangle) / 3.0
        if centroid_x >= 0.0159 and abs(centroid_y) <= 0.00201:
            role = "tab"
        elif centroid_x < -0.012 or abs(centroid_y) > 0.0024:
            role = "left_lobe" if centroid_x < 0.0 else "right_lobe"
        else:
            role = "bridge"
        role_index = role_counts[role]
        role_counts[role] += 1
        name = f"qc_col_lock_slider_{role}_part_{role_index:03d}"
        mesh_name = _add_convex_prism_mesh(
            asset,
            name=f"qc_lock_slider_{role}_source_prism_{role_index:03d}",
            polygon_xy=triangle,
            z_bounds=POSITIVE_LOCK_SLIDER_Z_BOUNDS_M,
        )
        _geom(
            body,
            name=name,
            geom_type="mesh",
            size=None,
            mesh=mesh_name,
            rgba="0.10 0.75 0.34 0.82",
            contype="64",
            conaffinity="31",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(name)
    if any(count == 0 for count in role_counts.values()):
        raise RuntimeError(f"slider role partition is incomplete: {role_counts}")
    return names


def _add_box_from_bounds(
    parent: ET.Element,
    *,
    name: str,
    bounds: tuple[tuple[float, float], ...],
    rgba: str,
    **attributes: str,
) -> ET.Element:
    if len(bounds) != 3 or any(hi <= lo for lo, hi in bounds):
        raise ValueError(f"invalid box bounds for {name}: {bounds}")
    return _geom(
        parent,
        name=name,
        geom_type="box",
        pos=tuple((lo + hi) / 2.0 for lo, hi in bounds),
        size=tuple((hi - lo) / 2.0 for lo, hi in bounds),
        rgba=rgba,
        **attributes,
    )


def _add_triangle_prism_geom(
    body: ET.Element,
    asset: ET.Element,
    *,
    name: str,
    triangle_xy: tuple[tuple[float, float], ...],
    z_bounds: tuple[float, float],
    rgba: str,
    **attributes: str,
) -> str:
    signed_twice_area = sum(
        triangle_xy[index][0] * triangle_xy[(index + 1) % 3][1]
        - triangle_xy[(index + 1) % 3][0] * triangle_xy[index][1]
        for index in range(3)
    )
    if abs(signed_twice_area) <= 1.0e-18:
        raise ValueError(f"degenerate triangle for {name}")
    polygon = triangle_xy if signed_twice_area > 0.0 else tuple(reversed(triangle_xy))
    mesh_name = _add_convex_prism_mesh(
        asset,
        name=f"{name}_source_mesh",
        polygon_xy=polygon,
        z_bounds=z_bounds,
    )
    _geom(
        body,
        name=name,
        geom_type="mesh",
        size=None,
        mesh=mesh_name,
        rgba=rgba,
        **attributes,
    )
    return name


def _add_square_minus_regular_void_partition(
    body: ET.Element,
    asset: ET.Element,
    *,
    name_prefix: str,
    center_xy: tuple[float, float],
    outer_half_size: float,
    void_vertex_radius: float,
    sides: int,
    z_bounds: tuple[float, float],
    rgba: str,
    semantic_suffix: str = "",
    **attributes: str,
) -> list[str]:
    """Partition a square-minus-convex-void ring into convex prisms."""

    if sides < 6 or outer_half_size <= void_vertex_radius:
        raise ValueError("invalid regular-void partition")
    center_x, center_y = center_xy
    names: list[str] = []
    inner: list[tuple[float, float]] = []
    outer: list[tuple[float, float]] = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        cosine = math.cos(angle)
        sine = math.sin(angle)
        inner.append(
            (
                center_x + void_vertex_radius * cosine,
                center_y + void_vertex_radius * sine,
            )
        )
        ray_scale = outer_half_size / max(abs(cosine), abs(sine))
        outer.append(
            (center_x + ray_scale * cosine, center_y + ray_scale * sine)
        )
    triangle_index = 0
    for index in range(sides):
        following = (index + 1) % sides
        for triangle in (
            (inner[index], outer[index], outer[following]),
            (inner[index], outer[following], inner[following]),
        ):
            name = f"{name_prefix}_{triangle_index:03d}{semantic_suffix}"
            names.append(
                _add_triangle_prism_geom(
                    body,
                    asset,
                    name=name,
                    triangle_xy=triangle,
                    z_bounds=z_bounds,
                    rgba=rgba,
                    **attributes,
                )
            )
            triangle_index += 1
    return names


def _add_two_well_plate_layer(
    body: ET.Element,
    asset: ET.Element,
    *,
    name_prefix: str,
    plate_bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    z_bounds: tuple[float, float],
    well_x_values: tuple[float, float],
    source_well_radius: float,
    semantic_suffix: str,
    rgba: str,
    first_box_name: str | None = None,
    **attributes: str,
) -> list[str]:
    """Add a rectangular layer with two conservative circular head wells."""

    sides = 40
    void_radius = source_well_radius / math.cos(math.pi / sides) + 1.0e-6
    outer_half = void_radius + 1.0e-5
    x_min, x_max = plate_bounds_xy[0]
    y_min, y_max = plate_bounds_xy[1]
    left_x, right_x = well_x_values
    strips = (
        ((x_min, left_x - outer_half), (y_min, y_max)),
        ((left_x + outer_half, right_x - outer_half), (y_min, y_max)),
        ((right_x + outer_half, x_max), (y_min, y_max)),
        ((left_x - outer_half, left_x + outer_half), (y_min, -outer_half)),
        ((left_x - outer_half, left_x + outer_half), (outer_half, y_max)),
        ((right_x - outer_half, right_x + outer_half), (y_min, -outer_half)),
        ((right_x - outer_half, right_x + outer_half), (outer_half, y_max)),
    )
    names: list[str] = []
    for index, bounds_xy in enumerate(strips):
        name = (
            first_box_name
            if index == 0 and first_box_name is not None
            else f"{name_prefix}_box_{index:02d}{semantic_suffix}"
        )
        _add_box_from_bounds(
            body,
            name=name,
            bounds=(*bounds_xy, z_bounds),
            rgba=rgba,
            **attributes,
        )
        names.append(name)
    for side, center_x in (("left", left_x), ("right", right_x)):
        names.extend(
            _add_square_minus_regular_void_partition(
                body,
                asset,
                name_prefix=f"{name_prefix}_{side}_well_part",
                center_xy=(center_x, 0.0),
                outer_half_size=outer_half,
                void_vertex_radius=void_radius,
                sides=sides,
                z_bounds=z_bounds,
                rgba=rgba,
                semantic_suffix=semantic_suffix,
                **attributes,
            )
        )
    return names


def _slider_track_upper_boundary_m() -> tuple[tuple[float, float], ...]:
    """Return an x-monotone conservative upper boundary of the track void."""

    margin = 1.0e-5
    radius = 0.0044 + 0.00022 + margin
    bridge_y = 0.0024 + 0.00022 + margin
    tab_y = 0.0020 + 0.00022 + margin
    points: list[tuple[float, float]] = []

    def append_arc(center_x: float, start: float, end: float) -> None:
        count = max(1, int(math.ceil(abs(end - start) / 0.08)))
        for index in range(count + 1):
            if points and index == 0:
                continue
            angle = start + (end - start) * index / count
            points.append(
                (center_x + radius * math.cos(angle), radius * math.sin(angle))
            )

    bridge_angle = math.asin(bridge_y / radius)
    tab_angle = math.asin(tab_y / radius)
    append_arc(-0.012, math.pi, math.pi / 2.0)
    points.append((-0.009, radius))
    append_arc(-0.009, math.pi / 2.0, bridge_angle)
    points.append((0.012 - radius * math.cos(bridge_angle), bridge_y))
    append_arc(0.012, math.pi - bridge_angle, math.pi / 2.0)
    points.append((0.015, radius))
    append_arc(0.015, math.pi / 2.0, tab_angle)
    points.append((0.02725 + margin, tab_y))

    ordered: list[tuple[float, float]] = []
    for point in sorted(points):
        if ordered and abs(point[0] - ordered[-1][0]) <= 1.0e-12:
            ordered[-1] = (point[0], max(point[1], ordered[-1][1]))
        else:
            ordered.append(point)
    if any(ordered[index][0] >= ordered[index + 1][0] for index in range(len(ordered) - 1)):
        raise RuntimeError("slider track boundary is not x-monotone")
    return tuple(ordered)


def _add_slider_track_plate_layer(
    body: ET.Element,
    asset: ET.Element,
    *,
    name_prefix: str,
    plate_bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    z_bounds: tuple[float, float],
    rgba: str,
    **attributes: str,
) -> list[str]:
    """Partition a plate layer around the conservative swept slider void."""

    x_min, x_max = plate_bounds_xy[0]
    y_min, y_max = plate_bounds_xy[1]
    source_points = _slider_track_upper_boundary_m()
    points = [point for point in source_points if x_min < point[0] < x_max]
    # The tab leaves through the relieved +X edge; clip its flat upper edge at
    # the remaining plate-material boundary.
    if points[-1][0] < x_max:
        points.append((x_max, points[-1][1]))
    elif points[-1][0] > x_max:
        previous = points[-2]
        following = points[-1]
        alpha = (x_max - previous[0]) / (following[0] - previous[0])
        points[-1] = (
            x_max,
            previous[1] + alpha * (following[1] - previous[1]),
        )

    names: list[str] = []
    left_void_x = points[0][0]
    if left_void_x > x_min:
        name = f"{name_prefix}_left_box_000"
        _add_box_from_bounds(
            body,
            name=name,
            bounds=((x_min, left_void_x), (y_min, y_max), z_bounds),
            rgba=rgba,
            **attributes,
        )
        names.append(name)
    triangle_index = 0
    for lower, upper in zip(points, points[1:]):
        for triangle in (
            ((lower[0], lower[1]), (upper[0], upper[1]), (upper[0], y_max)),
            ((lower[0], lower[1]), (upper[0], y_max), (lower[0], y_max)),
            ((lower[0], -lower[1]), (upper[0], y_min), (upper[0], -upper[1])),
            ((lower[0], -lower[1]), (lower[0], y_min), (upper[0], y_min)),
        ):
            name = f"{name_prefix}_part_{triangle_index:03d}"
            names.append(
                _add_triangle_prism_geom(
                    body,
                    asset,
                    name=name,
                    triangle_xy=triangle,
                    z_bounds=z_bounds,
                    rgba=rgba,
                    **attributes,
                )
            )
            triangle_index += 1
    return names


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


def add_robot_quick_change_interface(
    wrist_output: ET.Element, asset: ET.Element
) -> ET.Element:
    """Install the compact powered robot-side coupling on the bare wrist."""

    frame = ET.SubElement(
        wrist_output,
        "body",
        {"name": "robot_plate_frame", "quat": "0 1 0 0"},
    )
    plate_attributes = {
        "contype": "64",
        "conaffinity": "31",
        "solref": "0.0005 1",
        "solimp": "0.99 0.9999 0.00001",
    }
    plate_rgba = "0.93 0.62 0.04 0.28"
    main_xy = ((-0.024, 0.02355), (-0.024, 0.024))
    # Below the fixed head wells the source is continuous material.
    _add_box_from_bounds(
        frame,
        name="qc_col_robot_plate_lower_base_part_00",
        bounds=(*main_xy, (0.0, 0.0029)),
        rgba=plate_rgba,
        **plate_attributes,
    )
    # Above z=2.9 mm, both Ø6.65 entry wells are real voids.  The two layers
    # around the swept slider track use circumscribed 40-gon voids (<18 um
    # source miss) and no additive well geometry.
    _add_two_well_plate_layer(
        frame,
        asset,
        name_prefix="qc_col_robot_plate_lower_well_partition",
        plate_bounds_xy=main_xy,
        z_bounds=(0.0029, 0.0045),
        well_x_values=(-0.012, 0.012),
        source_well_radius=0.003325,
        semantic_suffix="",
        rgba=plate_rgba,
        **plate_attributes,
    )
    _add_slider_track_plate_layer(
        frame,
        asset,
        name_prefix="qc_col_robot_plate_slider_track_partition",
        plate_bounds_xy=main_xy,
        z_bounds=(0.0045, 0.0065),
        rgba=plate_rgba,
        **plate_attributes,
    )
    _add_two_well_plate_layer(
        frame,
        asset,
        name_prefix="qc_col_robot_plate_upper_well_partition",
        plate_bounds_xy=main_xy,
        z_bounds=(0.0065, 0.0095),
        well_x_values=(-0.012, 0.012),
        source_well_radius=0.003325,
        semantic_suffix="__mating_land",
        first_box_name="qc_col_robot_plate_core__mating_land",
        rgba=plate_rgba,
        **plate_attributes,
    )
    # The full-depth +X cam relief leaves only this exact -Y strip of source
    # plate material outside x=23.55 mm.
    _add_box_from_bounds(
        frame,
        name="qc_col_robot_plate_cam_relief_part_01",
        bounds=((0.02355, 0.024), (-0.024, -0.0165), (0.0, 0.0095)),
        rgba=plate_rgba,
        **plate_attributes,
    )
    # The source electrical wing reaches the core dock's left lower keeper.
    # This narrow edge strip lies outside the PCB pocket and is an exact
    # source-material subset; it supplies the fifth published seated tangency.
    _add_box_from_bounds(
        frame,
        name="qc_col_robot_plate_electrical_wing_edge__keeper_land",
        bounds=((-0.036, -0.035125), (-0.012, 0.012), (0.0, 0.0095)),
        rgba="0.93 0.62 0.04 0.28",
        contype="64",
        conaffinity="31",
        solref="0.0005 1",
        solimp="0.99 0.9999 0.00001",
    )
    # The head-entry wells and slider track are source voids, never additive
    # robot-side collision geoms.  The only moving lock material is the
    # hash-pinned, hole-preserving slider partition below.
    add_positive_lock_slider(frame, asset)
    for signal in SIGNALS:
        body = ET.SubElement(
            frame,
            "body",
            {
                "name": f"qc_pogo_{signal}_body",
                "pos": (
                    f"-0.031 {SIGNAL_Y_M[signal]:.7f} "
                    # The recovered spring-pin stack has a 0.875 mm ground
                    # reach and 0.675 mm P/D/ID reach beyond the mating plane.
                    # Continuous dynamics then settles near 0.858/0.662 mm,
                    # leaving a real preload instead of numerical tangency.
                    # The shorter signal pins use the same rounded crown but
                    # a 0.1 mm lower body datum, preserving their calibrated
                    # 0.675 mm reach.
                    f"{0.009575 if signal == 'ground' else 0.009375:.7f}"
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
        # A rounded pogo crown provides a stable axial contact witness.  The
        # former flat-ended cylinder met the larger pad exactly edge-on, so
        # MuJoCo could report only transient radial edge contacts while the
        # rigid interface remained aligned.
        _geom(
            body,
            name=f"qc_col_pogo_{signal}",
            geom_type="sphere",
            pos=(0.0, 0.0, 0.0),
            size=(0.0008,),
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
        name = f"{tool}_target_{side}_{suffix}_collision__mating_land"
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
        name=f"{tool}_m5_screw_{side}_collision__mating_land",
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
        [
            f"{tool}_m5_screw_{side}_collision__mating_land",
            f"{tool}_m5_nut_{side}_collision",
        ]
    )
    return names


@cache
def _require_positive_lock_hardware_sources() -> None:
    """Fail closed if either source used by the rigid hardware inertia drifts."""

    for path, expected_sha256 in (
        (
            POSITIVE_LOCK_SHOULDER_SCREW_STEP,
            POSITIVE_LOCK_SHOULDER_SCREW_STEP_SHA256,
        ),
        (POSITIVE_LOCK_STUD_NUT_STEP, POSITIVE_LOCK_STUD_NUT_STEP_SHA256),
    ):
        observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"positive-lock hardware STEP hash mismatch for {path.name}: "
                f"expected {expected_sha256}, got {observed_sha256}"
            )


def _add_positive_lock_hardware(
    tool_body: ET.Element, asset: ET.Element, tool: str
) -> list[str]:
    """Add one rigid, source-inertial shoulder-screw/nut assembly.

    The collision pieces deliberately contribute zero mass.  Their overlaps
    and faceting therefore cannot perturb the exact pair inertia recomputed
    from the two pinned STEP sources and declared placements.
    """

    _require_positive_lock_hardware_sources()
    hardware_body = ET.SubElement(
        tool_body,
        "body",
        {
            "name": f"tool_{tool}_positive_lock_hardware",
            "pos": "0 0 0",
        },
    )
    ET.SubElement(
        hardware_body,
        "inertial",
        {
            "pos": " ".join(
                f"{value:.15g}" for value in POSITIVE_LOCK_HARDWARE_SOURCE_COM_M
            ),
            "mass": f"{POSITIVE_LOCK_HARDWARE_SOURCE_MASS_KG:.15g}",
            "fullinertia": " ".join(
                f"{value:.15g}"
                for value in POSITIVE_LOCK_HARDWARE_SOURCE_FULL_INERTIA_KG_M2
            ),
        },
    )
    nut_radius = 0.0055 / math.sqrt(3.0)
    nut_polygon = tuple(
        (
            nut_radius * math.cos(index * math.pi / 3.0),
            nut_radius * math.sin(index * math.pi / 3.0),
        )
        for index in range(6)
    )
    nut_mesh = _add_convex_prism_mesh(
        asset,
        name=f"{tool}_positive_lock_nut_source_mesh",
        polygon_xy=nut_polygon,
        z_bounds=(0.0015, 0.0039),
    )
    names: list[str] = []
    for side, x_value in zip(
        ("left", "right"), POSITIVE_LOCK_HARDWARE_STACK_X_M, strict=True
    ):
        _geom(
            hardware_body,
            name=f"{tool}_lock_stud_{side}_shoulder_collision",
            geom_type="cylinder",
            pos=(x_value, 0.0, -0.0025),
            size=(0.0020, 0.0025),
            rgba="0.72 0.75 0.78 1",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        _geom(
            hardware_body,
            name=f"{tool}_lock_stud_{side}_head_collision",
            geom_type="cylinder",
            pos=(x_value, 0.0, -0.00565),
            size=(0.0030, 0.00065),
            rgba="0.72 0.75 0.78 1",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        _geom(
            hardware_body,
            name=f"{tool}_positive_lock_thread_{side}_collision",
            geom_type="cylinder",
            pos=(x_value, 0.0, 0.0020),
            size=(0.0015, 0.0020),
            rgba="0.42 0.44 0.47 1",
            mass="0",
        )
        _geom(
            hardware_body,
            name=f"{tool}_positive_lock_nut_{side}_collision",
            geom_type="mesh",
            size=None,
            pos=(x_value, 0.0, 0.0),
            mesh=nut_mesh,
            rgba="0.42 0.44 0.47 1",
            mass="0",
        )
        names.extend(
            [
                f"{tool}_lock_stud_{side}_shoulder_collision",
                f"{tool}_lock_stud_{side}_head_collision",
                f"{tool}_positive_lock_thread_{side}_collision",
                f"{tool}_positive_lock_nut_{side}_collision",
            ]
        )
    return names


def add_tool_quick_change_interface(
    body: ET.Element, asset: ET.Element, tool: str
) -> list[str]:
    """Install one complete tool-side interface with semantic collision names."""

    names: list[str] = []
    # Plate pieces retain the released outer mating datums.  The positive-lock
    # studs are tool-owned at x=+/-12 mm, y=0; no robot-side duplicate exists.
    plate_pieces = (
        ("center_yneg", 0.0, -0.0195, 0.004, 0.0055, False),
        ("center_middle", 0.0, 0.0, 0.004, 0.006, False),
        ("center_ypos", 0.0, 0.0195, 0.004, 0.0055, True),
    )
    for suffix, x_value, y_value, x_half, y_half, stop_land in plate_pieces:
        semantics = "__mating_land__locator_land"
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
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(plate_name)
    outer_semantics = "__mating_land__locator_land__dock_stop_land"
    lock_void_outer_half = 0.00331
    if tool == "gripper":
        # The core keeper witness reaches the released R4 mm right corner.  A
        # box would create false contact at (x=+/-28,y=+/-25) mm, outside the
        # rounded source boundary.  Sixteen chords per quarter keep the
        # inscribed curve's sagitta below 4.82 um inside the 20 um witness
        # budget.  Matcha rack tools retain their analytic stop boxes below.
        arc_steps = 16
        right_outline: list[tuple[float, float]] = [
            (0.012 + lock_void_outer_half, -0.025),
            (0.024, -0.025),
        ]
        right_outline.extend(
            (
                0.024
                + 0.004
                * math.cos(-math.pi / 2.0 + index * math.pi / (2 * arc_steps)),
                -0.021
                + 0.004
                * math.sin(-math.pi / 2.0 + index * math.pi / (2 * arc_steps)),
            )
            for index in range(1, arc_steps + 1)
        )
        right_outline.append((0.028, 0.021))
        right_outline.extend(
            (
                0.024 + 0.004 * math.cos(index * math.pi / (2 * arc_steps)),
                0.021 + 0.004 * math.sin(index * math.pi / (2 * arc_steps)),
            )
            for index in range(1, arc_steps + 1)
        )
        right_outline.append((0.012 + lock_void_outer_half, 0.025))
        left_outline = [(-x_value, -y_value) for x_value, y_value in right_outline]
        for suffix, outline in (("xneg", left_outline), ("xpos", right_outline)):
            plate_name = f"matcha_col_{tool}_plate_{suffix}{outer_semantics}"
            mesh_name = _add_convex_prism_mesh(
                asset,
                name=f"matcha_{tool}_plate_{suffix}_rounded_source_mesh",
                polygon_xy=tuple(outline),
                z_bounds=(0.0, 0.0095),
            )
            _geom(
                body,
                name=plate_name,
                geom_type="mesh",
                size=None,
                mesh=mesh_name,
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(plate_name)
    else:
        for suffix, x_bounds in (
            ("xneg", (-0.028, -0.012 - lock_void_outer_half)),
            ("xpos", (0.012 + lock_void_outer_half, 0.028)),
        ):
            plate_name = f"matcha_col_{tool}_plate_{suffix}{outer_semantics}"
            _add_box_from_bounds(
                body,
                name=plate_name,
                bounds=(x_bounds, (-0.025, 0.025), (0.0, 0.0095)),
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(plate_name)

    # Partition the material surrounding each x=+/-12 mm lock-stud feature.
    # The lower 1.5 mm layer preserves the Ø3.2 through-bore as a circumscribed
    # 24-gon (<15 um miss); the upper layer preserves the AF5.7 nut pocket as
    # an outward-offset exact hexagon.  No plate proxy overlaps the thread or
    # captive nut hardware installed below.
    inner_semantics = "__mating_land__locator_land"
    for side, center_x in (("left", -0.012), ("right", 0.012)):
        if center_x < 0.0:
            inner_x_bounds = (-0.012 + lock_void_outer_half, -0.004)
        else:
            inner_x_bounds = (0.004, 0.012 - lock_void_outer_half)
        inner_name = (
            f"matcha_col_{tool}_plate_{side}_lock_inner"
            f"{inner_semantics}__dock_stop_land"
        )
        _add_box_from_bounds(
            body,
            name=inner_name,
            bounds=(inner_x_bounds, (-0.025, 0.025), (0.0, 0.0095)),
            rgba="0.08 0.35 0.9 0.34",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(inner_name)
        for y_side, y_bounds in (
            ("yneg", (-0.025, -lock_void_outer_half)),
            ("ypos", (lock_void_outer_half, 0.025)),
        ):
            stop_semantic = "__dock_stop_land" if y_side == "ypos" else ""
            strip_name = (
                f"matcha_col_{tool}_plate_{side}_lock_{y_side}"
                f"{inner_semantics}{stop_semantic}"
            )
            _add_box_from_bounds(
                body,
                name=strip_name,
                bounds=(
                    (center_x - lock_void_outer_half, center_x + lock_void_outer_half),
                    y_bounds,
                    (0.0, 0.0095),
                ),
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(strip_name)
        bore_sides = 24
        bore_void_radius = 0.0016 / math.cos(math.pi / bore_sides) + 1.0e-6
        names.extend(
            _add_square_minus_regular_void_partition(
                body,
                asset,
                name_prefix=f"matcha_col_{tool}_plate_{side}_lock_bore_part",
                center_xy=(center_x, 0.0),
                outer_half_size=lock_void_outer_half,
                void_vertex_radius=bore_void_radius,
                sides=bore_sides,
                z_bounds=(0.0, 0.0015),
                rgba="0.08 0.35 0.9 0.34",
                semantic_suffix=inner_semantics,
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        )
        nut_void_radius = (0.0057 / 2.0 + 1.0e-5) / math.cos(math.pi / 6.0)
        names.extend(
            _add_square_minus_regular_void_partition(
                body,
                asset,
                name_prefix=f"matcha_col_{tool}_plate_{side}_lock_nut_part",
                center_xy=(center_x, 0.0),
                outer_half_size=lock_void_outer_half,
                void_vertex_radius=nut_void_radius,
                sides=6,
                z_bounds=(0.0015, 0.0095),
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        )
    # Exact source-material edge outside the recessed contact-board pocket.
    # On the stock-gripper adapter it bears on both left keeper rails; the
    # same source feature is present on every generic tool plate.
    wing_name = (
        f"matcha_col_{tool}_plate_electrical_wing_edge"
        "__mating_land__keeper_land"
    )
    _add_box_from_bounds(
        body,
        name=wing_name,
        bounds=((-0.036, -0.035125), (-0.012, 0.012), (0.0, 0.0095)),
        rgba="0.08 0.35 0.9 0.34",
        solref="0.0005 1",
        solimp="0.99 0.9999 0.00001",
    )
    names.append(wing_name)
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
    # The two M5 target stacks sit outside the lock-stud bores; their
    # 5.5 mm outer envelope remains inside the 25 mm plate datum.
    for side, y_value in (("neg", -0.0190), ("pos", 0.0190)):
        names.extend(_add_target_with_bore(body, tool, side, y_value))
    names.extend(_add_positive_lock_hardware(body, asset, tool))
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
    asset: ET.Element,
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
    if tool == "gripper":
        x_bounds, y_bounds, z_bounds = CORE_DOCK_STOP_BOUNDS_M
        hole_z = CORE_DOCK_STOP_HOLES_M[0][1]
        radius = CORE_DOCK_STOP_HOLES_M[0][2]
        middle_min = hole_z - radius
        middle_max = hole_z + radius
        stop_bounds: list[tuple[tuple[float, float], ...]] = [
            (x_bounds, y_bounds, (z_bounds[0], middle_min)),
        ]
        slice_count = round(
            (middle_max - middle_min) / CORE_STOP_HOLE_PARTITION_STEP_M
        )
        for slice_index in range(slice_count):
            z_min = middle_min + slice_index * CORE_STOP_HOLE_PARTITION_STEP_M
            z_max = min(
                middle_max, z_min + CORE_STOP_HOLE_PARTITION_STEP_M
            )
            nearest_delta = (
                0.0
                if z_min <= hole_z <= z_max
                else min(abs(z_min - hole_z), abs(z_max - hole_z))
            )
            hole_half_width = math.sqrt(
                max(0.0, radius * radius - nearest_delta * nearest_delta)
            )
            left_x, right_x = (hole[0] for hole in CORE_DOCK_STOP_HOLES_M)
            for interval in (
                (x_bounds[0], left_x - hole_half_width),
                (left_x + hole_half_width, right_x - hole_half_width),
                (right_x + hole_half_width, x_bounds[1]),
            ):
                stop_bounds.append((interval, y_bounds, (z_min, z_max)))
        stop_bounds.append((x_bounds, y_bounds, (middle_max, z_bounds[1])))
        for index, bounds in enumerate(stop_bounds):
            _add_box_from_bounds(
                dock,
                name=(
                    f"dock_{tool}_qc_col_dock_stop_part_{index:03d}"
                    "__dock_stop_land"
                ),
                bounds=bounds,
                rgba=rgba,
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        cam_polygon = CORE_DOCK_CAM_POLYGON_M
        cam_z_bounds = CORE_DOCK_CAM_Z_BOUNDS_M
    elif tool in {"spoon", "whisk"}:
        _add_box_from_bounds(
            dock,
            name=f"dock_{tool}_qc_col_dock_stop",
            bounds=MATCHA_DOCK_STOP_BOUNDS_M,
            rgba=rgba,
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        cam_polygon = MATCHA_DOCK_CAM_POLYGON_M
        cam_z_bounds = MATCHA_DOCK_CAM_Z_BOUNDS_M
    else:
        raise ValueError(f"unsupported dock source contract for {tool!r}")
    if tool == "gripper":
        # Exact core-dock keeper solids.  At the seated pose the stock plate
        # has four zero-volume tangencies (two lower X lands, two upper Z
        # lands); the robot electrical wing adds the fifth left-lower contact.
        keeper_bounds = {
            "left_lower": ((-0.043, -0.036), (-0.040, 0.036), (-0.003, 0.0)),
            "left_upper": ((-0.041, -0.033), (-0.040, 0.036), (0.0095, 0.0125)),
            "right_lower": ((0.028, 0.035), (-0.040, 0.036), (-0.003, 0.0)),
            "right_upper": ((0.025, 0.033), (-0.040, 0.036), (0.0095, 0.0125)),
        }
        for keeper_name, bounds in keeper_bounds.items():
            _add_box_from_bounds(
                dock,
                name=f"dock_gripper_keeper_{keeper_name}_collision",
                bounds=bounds,
                rgba=rgba,
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        for side, bounds in (
            ("left", ((-0.045, -0.041), (-0.040, 0.036), (-0.003, 0.0125))),
            ("right", ((0.033, 0.037), (-0.040, 0.036), (-0.003, 0.0125))),
        ):
            _add_box_from_bounds(
                dock,
                name=f"dock_gripper_wall_{side}_collision",
                bounds=bounds,
                rgba=rgba,
            )
    else:
        # The recovered spoon/whisk guide cheeks remain isolated from the core
        # witness; their source-faithful rack replacement belongs with the
        # pending CAD-derived payload/rack checkpoint.
        for side, y_rail in (("left", -0.033), ("right", 0.033)):
            _geom(
                dock,
                name=f"dock_{tool}_rail_{side}_collision",
                geom_type="box",
                pos=(0.0, y_rail, 0.010),
                size=(0.034, 0.003, 0.010),
                rgba=rgba,
            )
    cam_mesh_name = _add_convex_prism_mesh(
        asset,
        name=f"dock_{tool}_positive_lock_cam_source_mesh",
        polygon_xy=cam_polygon,
        z_bounds=cam_z_bounds,
    )
    _geom(
        dock,
        name=f"dock_{tool}_cam_collision",
        geom_type="mesh",
        size=None,
        mesh=cam_mesh_name,
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


def is_dock_stop_collision_name(tool: str, name: str) -> bool:
    """Return whether *name* is an exact member of one dock's stop family."""

    if tool == "gripper":
        return re.fullmatch(
            r"dock_gripper_qc_col_dock_stop_part_[0-9]{3}__dock_stop_land",
            name,
        ) is not None
    if tool in {"spoon", "whisk"}:
        return name == f"dock_{tool}_qc_col_dock_stop"
    return False


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
