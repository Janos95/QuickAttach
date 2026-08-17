#!/usr/bin/env python3
"""Exact/source-mesh clearance regression for the core SO-101 quick changer.

This validator is deliberately independent of the animation controller.  It
reconstructs the released stock-gripper mount composition from the CAD source,
the official fixed-body STEP, the calibrated SO-101 XML, and the moving-jaw
STL.  It then checks straight rack withdrawal from 0 through 80 mm.

OCCT B-rep distance/Boolean operations are authoritative for STEP solids.
FCPW is used only as a fast closest-point backend for the calibrated STL jaw;
positive continuous AABB separation remains the fail-closed authority for that
mesh.  The dock cam is removed only from the slider's forbidden solid and is
audited separately against the lock geometry.

All authored CAD units are millimetres.  The machine JSON report records every
source hash, transform, state, threshold, witness, and arithmetic result.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import cadquery as cq
import fcpw
import numpy as np


HERE = Path(__file__).resolve().parent
QUICK_CHANGE_DIR = HERE.parent
REPO_ROOT = QUICK_CHANGE_DIR.parents[1]
CAD_GENERATOR_PATH = QUICK_CHANGE_DIR / "generate_cad.py"
CORE_EXPORT_DIR = QUICK_CHANGE_DIR / "exports"
CORE_MANIFEST_PATH = CORE_EXPORT_DIR / "core_cad_manifest.json"
ROBOT_XML_PATH = REPO_ROOT / "Simulation/SO101/so101_new_calib.xml"
FIXED_GRIPPER_STEP_PATH = (
    REPO_ROOT / "STEP/SO101/Follower_Specific/Wrist_Roll_Follower_SO101.step"
)
FIXED_GRIPPER_STL_PATH = (
    REPO_ROOT / "Simulation/SO101/assets/wrist_roll_follower_so101_v1.stl"
)
MOVING_JAW_STL_PATH = REPO_ROOT / "Simulation/SO101/assets/moving_jaw_so101_v1.stl"
REPORT_PATH = HERE / "cad_clearance_report.json"

SCHEMA_VERSION = "1.2"
SWEEP_START_MM = 0.0
SWEEP_END_MM = 80.0
DEFAULT_SWEEP_STEP_MM = 1.0
MANUFACTURING_CLEARANCE_MM = 0.20
NUMERIC_DISTANCE_TOLERANCE_MM = 0.001
OVERLAP_VOLUME_TOLERANCE_MM3 = 1.0e-6
STOP_GAP_MIN_MM = 0.50
STOP_GAP_MAX_MM = 1.50
STOP_FORWARD_ENVELOPE_MM = 1.0
MIN_STUD_TO_CAM_X_MARGIN_MM = 1.0
MESH_TESSELLATION_DEFLECTION_MM = 0.03
MESH_ANGULAR_TOLERANCE_RAD = 0.10
CAM_CLEARANCE_SWEEP_STEP_MM = 0.10
AXIAL_CAPTURE_PRESEAT_MM = 15.0
AXIAL_CAPTURE_SWEEP_STEP_MM = 0.10
ROBOT_PLATE_REFERENCE_VOLUME_MM3 = 21130.316397677383
ROBOT_PLATE_EXPECTED_SOURCE_VOLUME_MM3 = 20872.829782079527
ROBOT_PLATE_MINIMUM_FUNCTIONAL_LIGAMENT_MM = 5.0
POSITIVE_LOCK_TRAVEL_SWEEP_STEP_MM = 0.05
POSITIVE_LOCK_RELEASE_SLIDER_VOLUME_MM3 = 220.12468083955645
POSITIVE_LOCK_RELEASE_SLIDER_BOUNDS_MM = (
    (-15.974062500000002, 24.0),
    (-4.4, 4.4),
    (0.0, 1.6),
)

EXPECTED_CORE_EXPORTS = (
    "so101_robot_plate.step",
    "so101_stock_gripper_tool_plate.step",
    "so101_passive_tool_dock.step",
    "so101_stock_gripper_retrofit_assembly.step",
)

DOCK_FEATURE_NAMES = (
    "left_lower_rail",
    "right_lower_rail",
    "left_upper_rail",
    "right_upper_rail",
    "left_wall",
    "right_wall",
    "seating_stop",
    "positive_lock_cam",
)

# These are complete, exact semantic pairs.  Nothing may enter this policy by
# prefix or role.  The robot plate's left edge and the stock plate keeper faces
# are deliberately coincident in the released CAD; every other dock contact is
# forbidden except the separately audited slider/cam actuation envelope.
INTENDED_ZERO_VOLUME_CONTACT_PAIRS = frozenset(
    {
        ("stock_tool_plate", "left_lower_rail"),
        ("stock_tool_plate", "right_lower_rail"),
        ("stock_tool_plate", "left_upper_rail"),
        ("stock_tool_plate", "right_upper_rail"),
        ("robot_plate", "left_lower_rail"),
    }
)


def _load_cad_generator():
    spec = importlib.util.spec_from_file_location("core_quick_change_cad", CAD_GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CAD generator {CAD_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAD = _load_cad_generator()
FIXED_STEP_TO_TOOL_POS_MM = tuple(CAD.STOCK_FIXED_STEP_TOOL_LOCAL_POS_MM)
FIXED_STEP_TO_TOOL_QUAT_WXYZ = tuple(
    CAD.STOCK_FIXED_STEP_TOOL_LOCAL_QUAT_WXYZ
)


@dataclass(frozen=True)
class BRepComponent:
    name: str
    shape: cq.Workplane
    role: str
    state: str = "nominal"


@dataclass(frozen=True)
class TriangleMesh:
    name: str
    vertices_mm: np.ndarray
    triangles: np.ndarray
    source_path: Path
    state: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _shape_volume_mm3(shape: cq.Shape) -> float:
    solids = list(shape.Solids())
    return float(sum(solid.Volume() for solid in solids))


def _intersection_volume_mm3(a: cq.Shape, b: cq.Shape) -> float:
    intersection = a.intersect(b)
    return _shape_volume_mm3(intersection)


def _bbox_tuple(shape: cq.Shape) -> tuple[tuple[float, float], ...]:
    box = shape.BoundingBox()
    return ((box.xmin, box.xmax), (box.ymin, box.ymax), (box.zmin, box.zmax))


def _bbox_record(bounds: tuple[tuple[float, float], ...]) -> dict[str, list[float]]:
    return {
        "x_mm": list(bounds[0]),
        "y_mm": list(bounds[1]),
        "z_mm": list(bounds[2]),
    }


def _axis_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(b[0] - a[1], a[0] - b[1], 0.0)


def _bbox_distance(a: tuple[tuple[float, float], ...], b: tuple[tuple[float, float], ...]) -> float:
    gaps = [_axis_gap(a[index], b[index]) for index in range(3)]
    return math.sqrt(sum(value * value for value in gaps))


def _translated_bounds(
    bounds: tuple[tuple[float, float], ...],
    translation: tuple[float, float, float],
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (bounds[index][0] + translation[index], bounds[index][1] + translation[index])
        for index in range(3)
    )


def _quat_matrix_wxyz(quat: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(quat), dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"invalid quaternion {values}")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    c_value, s_value = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(
        [[c_value, -s_value, 0.0], [s_value, c_value, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _parse_vector(value: str, count: int) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise RuntimeError(f"expected {count} finite values, got {value!r}")
    return values


def _read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise RuntimeError(f"STL is too short: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_bytes = 84 + triangle_count * 50
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"only canonical binary STL is accepted: {path} has {len(payload)} bytes, expected {expected_bytes}"
        )
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    records = np.frombuffer(payload, dtype=dtype, count=triangle_count, offset=84)
    raw_vertices = np.asarray(records["vertices"].reshape(-1, 3), dtype=np.float64)
    if not np.isfinite(raw_vertices).all():
        raise RuntimeError(f"non-finite STL vertices in {path}")
    unique_vertices, inverse = np.unique(raw_vertices, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.int32)
    return unique_vertices, triangles


def _xml_calibration() -> dict[str, Any]:
    root = ET.parse(ROBOT_XML_PATH).getroot()
    gripper = root.find(".//body[@name='gripper']")
    jaw = root.find(".//body[@name='moving_jaw_so101_v1']")
    if gripper is None or jaw is None:
        raise RuntimeError("calibrated XML no longer contains gripper/moving jaw")
    fixed_geom = next(
        (
            geom
            for geom in gripper.findall("./geom")
            if geom.get("mesh") == "wrist_roll_follower_so101_v1"
            and geom.get("class") == "collision"
        ),
        None,
    )
    jaw_geom = next(
        (
            geom
            for geom in jaw.findall("./geom")
            if geom.get("mesh") == "moving_jaw_so101_v1" and geom.get("class") == "collision"
        ),
        None,
    )
    joint = jaw.find("./joint[@name='gripper']")
    if fixed_geom is None or jaw_geom is None or joint is None:
        raise RuntimeError("calibrated gripper transform chain is incomplete")
    return {
        "fixed_geom_pos_m": _parse_vector(fixed_geom.get("pos", "0 0 0"), 3),
        "fixed_geom_quat_wxyz": _parse_vector(fixed_geom.get("quat", "1 0 0 0"), 4),
        "jaw_body_pos_m": _parse_vector(jaw.get("pos", "0 0 0"), 3),
        "jaw_body_quat_wxyz": _parse_vector(jaw.get("quat", "1 0 0 0"), 4),
        "jaw_geom_pos_m": _parse_vector(jaw_geom.get("pos", "0 0 0"), 3),
        "jaw_geom_quat_wxyz": _parse_vector(jaw_geom.get("quat", "1 0 0 0"), 4),
        "jaw_joint_axis": _parse_vector(joint.get("axis", "0 0 1"), 3),
        "jaw_joint_range_rad": _parse_vector(joint.get("range", "0 0"), 2),
    }


def _moving_jaw_tool_mesh(joint_angle_rad: float) -> TriangleMesh:
    raw_vertices_m, triangles = _read_binary_stl(MOVING_JAW_STL_PATH)
    calibration = _xml_calibration()
    if not np.allclose(calibration["jaw_joint_axis"], (0.0, 0.0, 1.0), atol=1.0e-9):
        raise RuntimeError("only calibrated local-Z jaw rotation is supported")

    r_fixed = _quat_matrix_wxyz(calibration["fixed_geom_quat_wxyz"])
    t_fixed = calibration["fixed_geom_pos_m"]
    r_jaw_body = _quat_matrix_wxyz(calibration["jaw_body_quat_wxyz"])
    t_jaw_body = calibration["jaw_body_pos_m"]
    r_jaw_geom = _quat_matrix_wxyz(calibration["jaw_geom_quat_wxyz"])
    t_jaw_geom = calibration["jaw_geom_pos_m"]

    jaw_geom_points = raw_vertices_m @ r_jaw_geom.T + t_jaw_geom
    jaw_rotated = jaw_geom_points @ _rotation_z(joint_angle_rad).T
    gripper_points = jaw_rotated @ r_jaw_body.T + t_jaw_body
    fixed_raw_points = (gripper_points - t_fixed) @ r_fixed
    tool_points_mm = fixed_raw_points * 1000.0 + np.asarray(FIXED_STEP_TO_TOOL_POS_MM)
    return TriangleMesh(
        name="moving_jaw_closed_calibrated_stl",
        vertices_mm=np.ascontiguousarray(tool_points_mm, dtype=np.float32),
        triangles=np.ascontiguousarray(triangles, dtype=np.int32),
        source_path=MOVING_JAW_STL_PATH,
        state=f"jaw_angle_rad={joint_angle_rad:.12g}",
    )


def _named_dock_features() -> dict[str, cq.Workplane]:
    rail_length = 76.0
    rail_center_y = -2.0
    features: dict[str, cq.Workplane] = {}
    for side, lower_x, upper_x, wall_x in (
        ("left", -39.5, -37.0, -43.0),
        ("right", 31.5, 29.0, 35.0),
    ):
        features[f"{side}_lower_rail"] = (
            cq.Workplane("XY").box(7.0, rail_length, 3.0, centered=True).translate(
                (lower_x, rail_center_y, -1.5)
            )
        )
        features[f"{side}_upper_rail"] = (
            cq.Workplane("XY").box(8.0, rail_length, 3.0, centered=True).translate(
                (upper_x, rail_center_y, CAD.PLATE_THICKNESS + 1.5)
            )
        )
        features[f"{side}_wall"] = (
            cq.Workplane("XY")
            .box(4.0, rail_length, CAD.PLATE_THICKNESS + 6.0, centered=True)
            .translate((wall_x, rail_center_y, CAD.PLATE_THICKNESS / 2.0))
        )
    features["seating_stop"] = CAD.core_dock_stop()
    features["positive_lock_cam"] = CAD.positive_lock_cam()
    return features


def _dock_authority() -> dict[str, cq.Workplane]:
    features = _named_dock_features()
    full = CAD.tool_dock()
    no_cam = full.cut(features["positive_lock_cam"]).clean()
    return {"full_dock": full, "dock_without_cam": no_cam, **features}


def _countersunk_screw(diameter: float, head_diameter: float, head_height: float, length: float):
    head = cq.Solid.makeCone(
        head_diameter / 2.0,
        diameter / 2.0,
        head_height,
        cq.Vector(0.0, 0.0, 0.0),
        cq.Vector(0.0, 0.0, 1.0),
    )
    shaft = cq.Solid.makeCylinder(
        diameter / 2.0,
        length - head_height,
        cq.Vector(0.0, 0.0, head_height),
        cq.Vector(0.0, 0.0, 1.0),
    )
    return cq.Workplane(obj=head.fuse(shaft))


def _hex_nut(across_flats: float, height: float, bore: float) -> cq.Workplane:
    outer = cq.Workplane("XY").polygon(6, across_flats / math.cos(math.pi / 6)).extrude(height)
    return outer.cut(CAD.axis_cylinder(bore, height + 0.2, (0.0, 0.0, -0.1))).clean()


def _tool_side_components() -> list[BRepComponent]:
    exported_plate = cq.importers.importStep(
        str(CORE_EXPORT_DIR / "so101_stock_gripper_tool_plate.step")
    )
    fixed_body = cq.importers.importStep(str(FIXED_GRIPPER_STEP_PATH)).translate(
        FIXED_STEP_TO_TOOL_POS_MM
    )
    components = [
        BRepComponent("stock_tool_plate", exported_plate, "printed_tool_structure"),
        BRepComponent("official_fixed_gripper_body", fixed_body, "official_step_payload"),
    ]
    m5_screw = _countersunk_screw(5.0, 10.0, 2.7, 10.0)
    m5_nut = _hex_nut(8.0, 4.0, 5.0)
    for index, (x_value, y_value) in enumerate(CAD.magnet_points(), start=1):
        components.extend(
            [
                BRepComponent(
                    f"target_{index}",
                    CAD.steel_target().translate(
                        (
                            x_value,
                            y_value,
                            CAD.MAGNETIC_HARDWARE_FACE_RECESS,
                        )
                    ),
                    "interface_hardware",
                ),
                BRepComponent(
                    f"target_screw_{index}",
                    m5_screw.translate((x_value, y_value, 0.0)),
                    "interface_hardware",
                ),
                BRepComponent(
                    f"target_nut_{index}",
                    m5_nut.translate((x_value, y_value, CAD.PLATE_THICKNESS - 4.0)),
                    "interface_hardware",
                ),
            ]
        )
    for index, x_value in enumerate((-CAD.LOCK_STUD_X, CAD.LOCK_STUD_X), start=1):
        components.extend(
            [
                BRepComponent(
                    f"shoulder_stud_{index}",
                    CAD.shoulder_lock_stud().translate((x_value, 0.0, 0.0)),
                    "positive_lock_hardware",
                ),
                BRepComponent(
                    f"stud_nut_{index}",
                    CAD.lock_stud_nut().translate(
                        (x_value, 0.0, CAD.LOCK_NUT_POCKET_FLOOR)
                    ),
                    "positive_lock_hardware",
                ),
            ]
        )
    components.append(
        BRepComponent(
            "target_contact_board",
            CAD.contact_board().translate((CAD.CONTACT_CENTER_X, 0.0, 0.0)),
            "interface_electronics",
        )
    )
    for index, (x_value, y_value) in enumerate(CAD.pogo_points(), start=1):
        components.append(
            BRepComponent(
                f"target_pad_{index}",
                CAD.contact_pad().translate((x_value, y_value, -0.05)),
                "interface_electronics",
            )
        )
    return components


def _robot_side_components() -> list[BRepComponent]:
    components = [
        BRepComponent(
            "robot_plate",
            CAD.robot_plate().translate((0.0, 0.0, -CAD.PLATE_THICKNESS)),
            "attached_robot_structure",
        )
    ]
    for index, (x_value, y_value) in enumerate(CAD.magnet_points(), start=1):
        components.append(
            BRepComponent(
                f"robot_magnet_{index}",
                CAD.screw_on_magnet().translate(
                    (
                        x_value,
                        y_value,
                        -CAD.MAGNET_HEIGHT
                        - CAD.MAGNETIC_HARDWARE_FACE_RECESS,
                    )
                ),
                "attached_robot_hardware",
            )
        )
    for state, x_offset in (("unlocked", 0.0), ("locked", CAD.SLIDER_TRAVEL)):
        components.append(
            BRepComponent(
                f"positive_lock_slider_{state}",
                CAD.locking_slider().translate(
                    (x_offset, 0.0, CAD.SLIDER_Z - CAD.PLATE_THICKNESS)
                ),
                "positive_lock_slider",
                state,
            )
        )
    return components


def _source_mount_contract() -> dict[str, Any]:
    source = CAD_GENERATOR_PATH.read_text()
    tree = ast.parse(source, filename=str(CAD_GENERATOR_PATH))
    normalized_functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    save_source = normalized_functions.get("save_assemblies", "")
    required_fragments = (
        "cq.Vector(*STOCK_TOOL_PLATE_ASSEMBLY_POS_MM)",
        "cq.Vector(*STOCK_FIXED_STEP_ASSEMBLY_POS_MM)",
    )
    missing = [fragment for fragment in required_fragments if fragment not in save_source]
    if missing:
        raise RuntimeError(f"stock-gripper mount composition drifted: {missing}")
    source_contract = CAD.stock_gripper_mount_contract()
    return {
        "source_fragments": list(required_fragments),
        "source_function_sha256": hashlib.sha256(save_source.encode()).hexdigest(),
        "robot_assembly_stock_plate_pos_mm": source_contract[
            "tool_plate_assembly_pos_mm"
        ],
        "robot_assembly_fixed_step_pos_mm": source_contract[
            "fixed_step_assembly_pos_mm"
        ],
        "tool_local_fixed_step_pos_mm": list(FIXED_STEP_TO_TOOL_POS_MM),
        "tool_local_fixed_step_quat_wxyz": list(FIXED_STEP_TO_TOOL_QUAT_WXYZ),
    }


def _required_stock_sim_body_transform() -> dict[str, Any]:
    """Solve the wrapper-body pose that composes the live mesh geom to CAD."""

    calibration = _xml_calibration()
    geom_quat = np.asarray(calibration["fixed_geom_quat_wxyz"], dtype=np.float64)
    geom_quat /= np.linalg.norm(geom_quat)
    body_quat = geom_quat * np.asarray((1.0, -1.0, -1.0, -1.0))
    body_rotation = _quat_matrix_wxyz(body_quat)
    geom_rotation = _quat_matrix_wxyz(geom_quat)
    desired_pos_m = np.asarray(FIXED_STEP_TO_TOOL_POS_MM, dtype=np.float64) * 1.0e-3
    body_pos_m = desired_pos_m - body_rotation @ calibration["fixed_geom_pos_m"]
    composed_pos_m = body_pos_m + body_rotation @ calibration["fixed_geom_pos_m"]
    composed_rotation = body_rotation @ geom_rotation
    return {
        "wrapper_body_pos_m": body_pos_m.tolist(),
        "wrapper_body_quat_wxyz": body_quat.tolist(),
        "composed_source_pos_m": composed_pos_m.tolist(),
        "composed_source_quat_wxyz": list(FIXED_STEP_TO_TOOL_QUAT_WXYZ),
        "position_residual_m": float(np.linalg.norm(composed_pos_m - desired_pos_m)),
        "rotation_residual_frobenius": float(
            np.linalg.norm(composed_rotation - np.eye(3))
        ),
        "composition": "world_wrapper * wrapper_child_geom = CAD tool-local STEP",
    }


def _mesh_from_brep(shape: cq.Shape) -> tuple[np.ndarray, np.ndarray, str]:
    vertices, triangles = shape.tessellate(
        MESH_TESSELLATION_DEFLECTION_MM, MESH_ANGULAR_TOLERANCE_RAD
    )
    vertex_array = np.ascontiguousarray(
        np.asarray([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=np.float32)
    )
    triangle_array = np.ascontiguousarray(np.asarray(triangles, dtype=np.int32))
    digest = hashlib.sha256(vertex_array.tobytes() + triangle_array.tobytes()).hexdigest()
    return vertex_array, triangle_array, digest


def _fcpw_scene(vertices: np.ndarray, triangles: np.ndarray):
    scene = fcpw.scene_3D()
    scene.set_object_count(1)
    scene.set_object_vertices(np.ascontiguousarray(vertices, dtype=np.float32), 0)
    scene.set_object_triangles(np.ascontiguousarray(triangles, dtype=np.int32), 0)
    scene.build(fcpw.aggregate_type.bvh_surface_area, False)
    return scene


def _directional_distance(
    query_vertices: np.ndarray,
    target_vertices: np.ndarray,
    target_triangles: np.ndarray,
) -> float:
    scene = _fcpw_scene(target_vertices, target_triangles)
    radii = np.full(len(query_vertices), np.inf, dtype=np.float32)
    interactions = fcpw.interaction_3D_list()
    scene.find_closest_points(
        np.ascontiguousarray(query_vertices, dtype=np.float32), radii, interactions
    )
    if len(interactions) != len(query_vertices):
        raise RuntimeError("FCPW closest-point batch was incomplete")
    return min(float(interaction.d) for interaction in interactions)


def _mesh_screen(
    moving: TriangleMesh,
    fixed_shape: cq.Shape,
    translation_mm: tuple[float, float, float],
) -> dict[str, Any]:
    fixed_vertices, fixed_triangles, fixed_digest = _mesh_from_brep(fixed_shape)
    moving_vertices = moving.vertices_mm + np.asarray(translation_mm, dtype=np.float32)
    moving_to_fixed = _directional_distance(
        moving_vertices, fixed_vertices, fixed_triangles
    )
    fixed_to_moving = _directional_distance(
        fixed_vertices, moving_vertices, moving.triangles
    )
    return {
        "backend": "fcpw",
        "backend_version": importlib.metadata.version("fcpw"),
        "role": "source_triangle_mesh_screen_only",
        "moving_source_sha256": _sha256(moving.source_path),
        "fixed_tessellation_sha256": fixed_digest,
        "tessellation_deflection_mm": MESH_TESSELLATION_DEFLECTION_MM,
        "minimum_moving_vertices_to_fixed_mesh_mm": moving_to_fixed,
        "minimum_fixed_vertices_to_moving_mesh_mm": fixed_to_moving,
        "minimum_directional_witness_mm": min(moving_to_fixed, fixed_to_moving),
        "clearance_authority": False,
    }


def _sweep_positions(step_mm: float) -> list[float]:
    if step_mm <= 0.0:
        raise ValueError("sweep step must be positive")
    intervals = round((SWEEP_END_MM - SWEEP_START_MM) / step_mm)
    positions = [round(SWEEP_START_MM + index * step_mm, 10) for index in range(intervals + 1)]
    if positions[-1] != SWEEP_END_MM:
        raise RuntimeError("sweep step does not land exactly at 80 mm")
    return positions


def _axial_capture_offsets() -> list[float]:
    """Return the exact 15 mm pre-seat to seated axial approach grid."""

    intervals = round(AXIAL_CAPTURE_PRESEAT_MM / AXIAL_CAPTURE_SWEEP_STEP_MM)
    offsets = [
        round(AXIAL_CAPTURE_PRESEAT_MM - index * AXIAL_CAPTURE_SWEEP_STEP_MM, 10)
        for index in range(intervals + 1)
    ]
    if offsets[0] != AXIAL_CAPTURE_PRESEAT_MM or offsets[-1] != 0.0:
        raise RuntimeError("axial capture grid does not close exactly at seat")
    return offsets


def _brep_sweep_record(
    component: BRepComponent,
    dock_shape: cq.Shape,
    positions_mm: list[float],
    *,
    dock_component: str,
    intended_zero_volume_contact: bool,
) -> dict[str, Any]:
    observations: list[tuple[float, float, float]] = []
    for withdrawal_mm in positions_mm:
        placed = component.shape.translate((0.0, -withdrawal_mm, 0.0)).val()
        distance = float(placed.distance(dock_shape))
        overlap = 0.0
        if distance <= NUMERIC_DISTANCE_TOLERANCE_MM:
            overlap = _intersection_volume_mm3(placed, dock_shape)
        observations.append((withdrawal_mm, distance, overlap))
    witness = min(observations, key=lambda item: (item[1], -item[2]))
    half_step = (positions_mm[1] - positions_mm[0]) / 2.0
    continuous_clearance = witness[1] - half_step
    maximum_overlap = max(item[2] for item in observations)
    if intended_zero_volume_contact:
        # Only the four explicitly named keeper rails may touch the stock
        # plate.  Requiring the seated state to be tangent prevents a missing
        # or displaced keeper from being accepted merely because it does not
        # collide.
        passed = (
            observations[0][1] <= NUMERIC_DISTANCE_TOLERANCE_MM
            and maximum_overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
        )
        semantics = "intended_stock_plate_keeper_tangency"
    else:
        passed = (
            continuous_clearance + NUMERIC_DISTANCE_TOLERANCE_MM
            >= MANUFACTURING_CLEARANCE_MM
            and maximum_overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
        )
        semantics = "forbidden_component_continuous_clearance"
    return {
        "component": component.name,
        "dock_component": dock_component,
        "component_state": component.state,
        "component_role": component.role,
        "semantics": semantics,
        "initial_distance_mm": observations[0][1],
        "sample_count": len(positions_mm),
        "sample_step_mm": positions_mm[1] - positions_mm[0],
        "sampled_positions_sha256": _canonical_sha256(positions_mm),
        "minimum_sampled_distance_mm": witness[1],
        "maximum_sampled_overlap_volume_mm3": maximum_overlap,
        "maximum_between_sample_motion_bound_mm": half_step,
        "continuous_certified_clearance_mm": continuous_clearance,
        "witness_withdrawal_mm": witness[0],
        "method": "occt_brep_distance_with_near_witness_boolean",
        "passed": passed,
    }


def _axial_capture_cam_record(
    robot_plate: BRepComponent, cam_shape: cq.Shape
) -> dict[str, Any]:
    """Certify the coupled axial/recenter route against the complete cam."""

    offsets = _axial_capture_offsets()
    observations: list[tuple[float, float, float, float]] = []
    for preseated_mm in offsets:
        lateral_offset = CAD.positive_lock_cam_capture_lateral_offset_mm(
            preseated_mm
        )
        placed = robot_plate.shape.translate(
            (lateral_offset, 0.0, -preseated_mm)
        ).val()
        distance = float(placed.distance(cam_shape))
        overlap = 0.0
        if distance <= NUMERIC_DISTANCE_TOLERANCE_MM:
            overlap = _intersection_volume_mm3(placed, cam_shape)
        observations.append((preseated_mm, lateral_offset, distance, overlap))
    witness = min(observations, key=lambda item: (item[2], -item[3]))
    interval_motion = [
        math.hypot(
            observations[index + 1][0] - observations[index][0],
            observations[index + 1][1] - observations[index][1],
        )
        for index in range(len(observations) - 1)
    ]
    motion_bound = max(interval_motion) / 2.0
    continuous_clearance = witness[2] - motion_bound
    maximum_overlap = max(item[3] for item in observations)
    passed = bool(
        continuous_clearance + NUMERIC_DISTANCE_TOLERANCE_MM
        >= MANUFACTURING_CLEARANCE_MM
        and maximum_overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
    )
    return {
        "component": robot_plate.name,
        "dock_component": "positive_lock_cam",
        "semantics": "forbidden_cam_clearance_during_coupled_axial_recenter",
        "axis_dock_local": [0.0, 0.0, 1.0],
        "lateral_axis_dock_local": [1.0, 0.0, 0.0],
        "lateral_offset_start_mm": CAD.ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM,
        "lateral_offset_end_mm": 0.0,
        "recenter_start_preseat_mm": (
            CAD.DOCK_CAM_CAPTURE_RECENTER_START_PRESEAT_MM
        ),
        "recenter_end_preseat_mm": (
            CAD.DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM
        ),
        "preseat_start_mm": AXIAL_CAPTURE_PRESEAT_MM,
        "preseat_end_mm": 0.0,
        "sample_count": len(offsets),
        "sample_step_mm": AXIAL_CAPTURE_SWEEP_STEP_MM,
        "sampled_preseat_offsets_sha256": _canonical_sha256(offsets),
        "minimum_sampled_distance_mm": witness[2],
        "maximum_sampled_overlap_volume_mm3": maximum_overlap,
        "maximum_between_sample_motion_bound_mm": motion_bound,
        "continuous_certified_clearance_mm": continuous_clearance,
        "witness_preseat_offset_mm": witness[0],
        "witness_lateral_offset_mm": witness[1],
        "route_samples_sha256": _canonical_sha256(
            [[item[0], item[1]] for item in observations]
        ),
        "method": "occt_brep_distance_with_near_witness_boolean",
        "passed": passed,
    }


def _passive_positive_lock_cam_record() -> dict[str, Any]:
    """Recompute the complete passive axial-open and -Y return sequence."""

    cam = CAD.positive_lock_cam().val()
    lead = CAD.positive_lock_cam_axial_lead().val()
    slider_native = CAD.locking_slider().val()
    robot_components = _robot_side_components()
    tool_components = _tool_side_components()
    robot_plate = next(
        component for component in robot_components if component.name == "robot_plate"
    )
    studs = {
        side: CAD.shoulder_lock_stud().translate((x_value, 0.0, 0.0)).val()
        for side, x_value in (
            ("left", -CAD.LOCK_STUD_X),
            ("right", CAD.LOCK_STUD_X),
        )
    }

    capture_offsets = _axial_capture_offsets()
    capture_samples: list[dict[str, float]] = []
    maximum_slider_cam_overlap = 0.0
    maximum_slider_stud_overlap = 0.0
    for preseated_mm in capture_offsets:
        lateral_mm = CAD.positive_lock_cam_capture_lateral_offset_mm(
            preseated_mm
        )
        q_mm = CAD.positive_lock_cam_capture_q_max_mm(preseated_mm)
        slider = slider_native.translate(
            (
                q_mm + lateral_mm,
                0.0,
                CAD.SLIDER_Z - CAD.PLATE_THICKNESS - preseated_mm,
            )
        )
        slider_cam_distance = float(slider.distance(cam))
        slider_cam_overlap = (
            _intersection_volume_mm3(slider, cam)
            if slider_cam_distance <= NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        stud_distances = {
            side: float(slider.distance(stud)) for side, stud in studs.items()
        }
        stud_overlaps = {
            side: (
                _intersection_volume_mm3(slider, stud)
                if stud_distances[side] <= NUMERIC_DISTANCE_TOLERANCE_MM
                else 0.0
            )
            for side, stud in studs.items()
        }
        maximum_slider_cam_overlap = max(
            maximum_slider_cam_overlap, slider_cam_overlap
        )
        maximum_slider_stud_overlap = max(
            maximum_slider_stud_overlap, *stud_overlaps.values()
        )
        capture_samples.append(
            {
                "preseat_mm": preseated_mm,
                "lateral_mm": lateral_mm,
                "q_mm": q_mm,
                "slider_cam_distance_mm": slider_cam_distance,
                "minimum_slider_stud_distance_mm": min(stud_distances.values()),
                "maximum_slider_cam_overlap_mm3": slider_cam_overlap,
                "maximum_slider_stud_overlap_mm3": max(stud_overlaps.values()),
            }
        )

    # Tighten the critical shoulder/head-entry clearance independently of the
    # coarser full-approach grid.  The slider moves equally in dock X and Z on
    # the ruled lead, so the half-step Euclidean motion is sqrt(2)*step/2.
    stud_step_mm = 0.01
    stud_intervals = round(
        (CAD.DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM + 0.1) / stud_step_mm
    )
    stud_samples: list[tuple[float, float]] = []
    for index in range(stud_intervals + 1):
        preseated_mm = round(index * stud_step_mm, 10)
        lateral_mm = CAD.positive_lock_cam_capture_lateral_offset_mm(
            preseated_mm
        )
        q_mm = CAD.positive_lock_cam_capture_q_max_mm(preseated_mm)
        slider = slider_native.translate(
            (
                q_mm + lateral_mm,
                0.0,
                CAD.SLIDER_Z - CAD.PLATE_THICKNESS - preseated_mm,
            )
        )
        stud_samples.append(
            (
                preseated_mm,
                min(float(slider.distance(stud)) for stud in studs.values()),
            )
        )
    stud_witness = min(stud_samples, key=lambda item: item[1])
    stud_motion_bound = math.sqrt(2.0) * stud_step_mm / 2.0
    continuous_stud_clearance = stud_witness[1] - stud_motion_bound

    head_entry = next(
        sample
        for sample in capture_samples
        if math.isclose(
            sample["preseat_mm"],
            CAD.DOCK_CAM_HEAD_ENTRY_PRESEAT_MM,
            abs_tol=1.0e-12,
        )
    )
    plate_record = _axial_capture_cam_record(robot_plate, cam)

    # Every source component that can coexist with the dock cam is checked.
    # Tool components remain fixed while the robot-side magnets follow the
    # same p/x route.  The physical slider and plate are recorded above.
    component_records: list[dict[str, Any]] = []
    for component in tool_components:
        shape = component.shape.val()
        distance = float(shape.distance(cam))
        overlap = (
            _intersection_volume_mm3(shape, cam)
            if distance <= NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        component_records.append(
            {
                "component": component.name,
                "motion": "fixed_tool_in_dock",
                "minimum_sampled_distance_mm": distance,
                "maximum_overlap_volume_mm3": overlap,
            }
        )
    for component in robot_components:
        if component.name == "robot_plate" or component.role == "positive_lock_slider":
            continue
        observations: list[tuple[float, float]] = []
        for preseated_mm in capture_offsets:
            lateral_mm = CAD.positive_lock_cam_capture_lateral_offset_mm(
                preseated_mm
            )
            placed = component.shape.translate(
                (lateral_mm, 0.0, -preseated_mm)
            ).val()
            distance = float(placed.distance(cam))
            overlap = (
                _intersection_volume_mm3(placed, cam)
                if distance <= NUMERIC_DISTANCE_TOLERANCE_MM
                else 0.0
            )
            observations.append((distance, overlap))
        component_records.append(
            {
                "component": component.name,
                "motion": "coupled_axial_recenter",
                "minimum_sampled_distance_mm": min(item[0] for item in observations),
                "maximum_overlap_volume_mm3": max(item[1] for item in observations),
            }
        )
    component_records.extend(
        [
            {
                "component": "robot_plate",
                "motion": "coupled_axial_recenter",
                "minimum_sampled_distance_mm": plate_record[
                    "minimum_sampled_distance_mm"
                ],
                "maximum_overlap_volume_mm3": plate_record[
                    "maximum_sampled_overlap_volume_mm3"
                ],
            },
            {
                "component": "positive_lock_slider_physical_capture_path",
                "motion": "coupled_p_x_q",
                "minimum_sampled_distance_mm": min(
                    sample["slider_cam_distance_mm"]
                    for sample in capture_samples
                ),
                "maximum_overlap_volume_mm3": maximum_slider_cam_overlap,
            },
        ]
    )
    component_records.sort(key=lambda item: item["component"])

    release_step_mm = 0.1
    release_positions = [
        round(index * release_step_mm, 10)
        for index in range(round(15.0 / release_step_mm) + 1)
    ]
    release_samples: list[dict[str, float]] = []
    maximum_release_overlap = 0.0
    for withdrawal_mm in release_positions:
        q_mm = CAD.positive_lock_cam_release_q_max_mm(withdrawal_mm)
        slider = slider_native.translate(
            (
                q_mm,
                -withdrawal_mm,
                CAD.SLIDER_Z - CAD.PLATE_THICKNESS,
            )
        )
        distance = float(slider.distance(cam))
        overlap = (
            _intersection_volume_mm3(slider, cam)
            if distance <= NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        maximum_release_overlap = max(maximum_release_overlap, overlap)
        release_samples.append(
            {
                "withdrawal_mm": withdrawal_mm,
                "q_mm": q_mm,
                "slider_cam_distance_mm": distance,
                "slider_cam_overlap_mm3": overlap,
            }
        )
    exit_sample = release_samples[-1]

    contract = CAD.positive_lock_cam_contract()
    cam_bounds = _bbox_record(_bbox_tuple(cam))
    lead_bounds = _bbox_record(_bbox_tuple(lead))
    geometry = {
        "cam_is_single_valid_solid": bool(
            cam.isValid() and len(cam.Solids()) == 1
        ),
        "cam_volume_mm3": _shape_volume_mm3(cam),
        "cam_bounds": cam_bounds,
        "lead_volume_mm3": _shape_volume_mm3(lead),
        "lead_bounds": lead_bounds,
        "contract": contract,
    }
    expected = contract["expected_geometry"]
    load = contract["quasistatic_load_envelope"]
    manufacturing = contract["manufacturability"]
    checks = {
        "cam_is_single_valid_solid": geometry["cam_is_single_valid_solid"],
        "cam_volume_matches_contract": math.isclose(
            geometry["cam_volume_mm3"],
            expected["total_volume_mm3"],
            abs_tol=1.0e-9,
        ),
        "cam_bounds_match_contract": bool(
            all(
                math.isclose(
                    cam_bounds[f"{axis}_mm"][endpoint],
                    expected["bounds_mm"][axis][endpoint],
                    abs_tol=1.0e-6,
                )
                for axis in ("x", "y", "z")
                for endpoint in (0, 1)
            )
        ),
        "minimum_feature_exceeds_declared_process_floor": bool(
            manufacturing["minimum_feature_mm"]
            >= manufacturing["declared_process_floor_mm"]
        ),
        "lead_is_45_degree_self_supporting": bool(
            manufacturing["lead_is_self_supporting_at_45_deg"]
        ),
        "quasistatic_loads_are_finite_and_positive": bool(
            all(
                math.isfinite(float(load[key])) and float(load[key]) > 0.0
                for key in (
                    "maximum_spring_force_n",
                    "maximum_axial_reaction_n",
                    "maximum_cam_normal_force_n",
                    "contact_face_area_mm2",
                    "mean_contact_pressure_mpa",
                )
            )
        ),
        "capture_cam_boolean_overlap_is_zero": bool(
            maximum_slider_cam_overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
        ),
        "capture_stud_boolean_overlap_is_zero": bool(
            maximum_slider_stud_overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
        ),
        "head_entry_occurs_after_open_and_recenter": bool(
            head_entry["lateral_mm"] == 0.0
            and head_entry["q_mm"]
            <= CAD.DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM + 1.0e-12
        ),
        "continuous_stud_clearance_meets_manufacturing_floor": bool(
            continuous_stud_clearance + NUMERIC_DISTANCE_TOLERANCE_MM
            >= MANUFACTURING_CLEARANCE_MM
        ),
        "robot_plate_cam_continuous_clearance_passes": bool(
            plate_record["passed"]
        ),
        "all_component_cam_overlaps_are_zero": bool(
            all(
                record["maximum_overlap_volume_mm3"]
                <= OVERLAP_VOLUME_TOLERANCE_MM3
                for record in component_records
            )
        ),
        "release_cam_boolean_overlap_is_zero": bool(
            maximum_release_overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
        ),
        "release_reaches_q3_at_nominal_exit": math.isclose(
            exit_sample["q_mm"], CAD.SLIDER_TRAVEL, abs_tol=1.0e-12
        ),
        "nominal_exit_cam_clearance_meets_floor": bool(
            exit_sample["slider_cam_distance_mm"]
            + NUMERIC_DISTANCE_TOLERANCE_MM
            >= MANUFACTURING_CLEARANCE_MM
        ),
    }
    return {
        "source_contract": contract,
        "geometry": geometry,
        "capture": {
            "sample_step_mm": AXIAL_CAPTURE_SWEEP_STEP_MM,
            "sample_count": len(capture_samples),
            "sample_digest_sha256": _canonical_sha256(capture_samples),
            "ramp_contact_start_preseat_mm": contract["passive_capture"][
                "ramp_contact_start_preseat_mm"
            ],
            "head_entry_sample": head_entry,
            "maximum_slider_cam_overlap_mm3": maximum_slider_cam_overlap,
            "maximum_slider_stud_overlap_mm3": maximum_slider_stud_overlap,
            "tight_stud_clearance": {
                "sample_step_mm": stud_step_mm,
                "sample_count": len(stud_samples),
                "witness_preseat_mm": stud_witness[0],
                "minimum_sampled_distance_mm": stud_witness[1],
                "maximum_between_sample_motion_bound_mm": stud_motion_bound,
                "continuous_certified_clearance_mm": continuous_stud_clearance,
                "sample_digest_sha256": _canonical_sha256(stud_samples),
            },
            "robot_plate_cam": plate_record,
            "component_cam_records": component_records,
        },
        "release": {
            "axis_dock_local": [0.0, -1.0, 0.0],
            "sample_step_mm": release_step_mm,
            "sample_count": len(release_samples),
            "sample_digest_sha256": _canonical_sha256(release_samples),
            "maximum_slider_cam_overlap_mm3": maximum_release_overlap,
            "q3_tangent_withdrawal_mm": contract["passive_release"][
                "q3_tangent_withdrawal_mm"
            ],
            "nominal_exit_sample": exit_sample,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _mechanism_preservation_record() -> dict[str, Any]:
    """Recompute plate structure and released slider/keyhole/stud datums."""

    plate = CAD.robot_plate().val()
    plate_volume = _shape_volume_mm3(plate)
    retained_fraction = (
        plate_volume / ROBOT_PLATE_REFERENCE_VOLUME_MM3
    )
    slider = CAD.locking_slider().val()
    slider_bounds = _bbox_tuple(slider)
    cam = CAD.positive_lock_cam().val()
    unlocked = CAD.locking_slider().translate(
        (0.0, 0.0, CAD.SLIDER_Z - CAD.PLATE_THICKNESS)
    ).val()
    locked = CAD.locking_slider().translate(
        (CAD.SLIDER_TRAVEL, 0.0, CAD.SLIDER_Z - CAD.PLATE_THICKNESS)
    ).val()
    unlocked_cam_gap = float(unlocked.distance(cam))
    locked_cam_overlap = _intersection_volume_mm3(locked, cam)
    plate_bounds = _bbox_tuple(plate)
    positive_lock_travel = _positive_lock_travel_record()
    checks = {
        "robot_plate_is_single_valid_solid": bool(
            plate.isValid() and len(plate.Solids()) == 1
        ),
        "robot_plate_volume_matches_hash_pinned_source_revision": math.isclose(
            plate_volume,
            ROBOT_PLATE_EXPECTED_SOURCE_VOLUME_MM3,
            abs_tol=1.0e-6,
        ),
        "relief_spans_full_printed_plate_thickness": bool(
            CAD.ROBOT_CAM_RELIEF_Z_MIN == 0.0
            and CAD.ROBOT_CAM_RELIEF_Z_MAX == CAD.PLATE_THICKNESS
        ),
        "stud_well_ligament_preserved": bool(
            CAD.ROBOT_CAM_RELIEF_TO_STUD_WELL_LIGAMENT_MM
            >= ROBOT_PLATE_MINIMUM_FUNCTIONAL_LIGAMENT_MM
        ),
        "slider_lobe_ligament_preserved": bool(
            CAD.ROBOT_CAM_RELIEF_TO_SLIDER_LOBE_LIGAMENT_MM
            >= ROBOT_PLATE_MINIMUM_FUNCTIONAL_LIGAMENT_MM
        ),
        "slider_source_volume_matches_swept_neck_release": math.isclose(
            _shape_volume_mm3(slider),
            POSITIVE_LOCK_RELEASE_SLIDER_VOLUME_MM3,
            abs_tol=1.0e-9,
        ),
        "slider_source_bounds_match_swept_neck_release": all(
            math.isclose(slider_bounds[axis][limit], expected, abs_tol=1.0e-12)
            for axis, bounds in enumerate(POSITIVE_LOCK_RELEASE_SLIDER_BOUNDS_MM)
            for limit, expected in enumerate(bounds)
        ),
        "stud_keyhole_full_travel_is_clear_and_retained": bool(
            positive_lock_travel["passed"]
        ),
        "keyhole_neck_clears_shoulder": bool(
            CAD.KEYHOLE_NECK_WIDTH > CAD.LOCK_SHOULDER_DIAMETER
        ),
        "keyhole_neck_retains_head": bool(
            CAD.KEYHOLE_NECK_WIDTH < CAD.LOCK_HEAD_DIAMETER
        ),
        "unlocked_cam_gap_preserved": math.isclose(
            unlocked_cam_gap, 0.05, abs_tol=1.0e-12
        ),
        "locked_cam_engagement_preserved": bool(
            locked_cam_overlap > OVERLAP_VOLUME_TOLERANCE_MM3
        ),
    }
    return {
        "robot_plate_volume_mm3": plate_volume,
        "reference_plate_volume_mm3": ROBOT_PLATE_REFERENCE_VOLUME_MM3,
        "gross_volume_guard_role": (
            "secondary_sanity_only_local_web_ligament_and_clearance_checks_are_authoritative"
        ),
        "retained_volume_fraction": retained_fraction,
        "removed_volume_mm3": (
            ROBOT_PLATE_REFERENCE_VOLUME_MM3 - plate_volume
        ),
        "robot_plate_bbox": _bbox_record(plate_bounds),
        "relief_contract": CAD.robot_cam_relief_contract(),
        "stud_centres_xy_mm": [
            [-CAD.LOCK_STUD_X, 0.0],
            [CAD.LOCK_STUD_X, 0.0],
        ],
        "robot_stud_well_centres_xy_mm": [
            [-CAD.LOCK_STUD_X, 0.0],
            [CAD.LOCK_STUD_X, 0.0],
        ],
        "slider_bbox_native_mm": _bbox_record(slider_bounds),
        "slider_volume_mm3": _shape_volume_mm3(slider),
        "slider_travel_mm": CAD.SLIDER_TRAVEL,
        "positive_lock_keyhole_contract": CAD.positive_lock_keyhole_contract(),
        "positive_lock_travel": positive_lock_travel,
        "keyhole_entry_diameter_mm": CAD.KEYHOLE_ENTRY_DIAMETER,
        "keyhole_neck_width_mm": CAD.KEYHOLE_NECK_WIDTH,
        "stud_shoulder_diameter_mm": CAD.LOCK_SHOULDER_DIAMETER,
        "stud_head_diameter_mm": CAD.LOCK_HEAD_DIAMETER,
        "unlocked_slider_to_cam_gap_mm": unlocked_cam_gap,
        "locked_slider_cam_overlap_volume_mm3": locked_cam_overlap,
        "checks": checks,
        "passed": all(checks.values()),
    }


INTERFACE_AUTHORITY_REQUIREMENTS = (
    (
        "fabrication_process_tolerance_qualified",
        "fabrication_process_tolerance_unqualified",
    ),
    (
        "fixed_pogo_shell_exact_drawing_bound",
        "fixed_pogo_shell_is_illustrative_reference_only",
    ),
    (
        "pogo_mounting_sectional_bore_resolved",
        "pogo_mounting_sectional_bore_unresolved",
    ),
    (
        "ground_first_mate_shoulder_datum_resolved",
        "ground_first_mate_shoulder_datum_unresolved",
    ),
    (
        "ground_first_mate_tolerance_stack_qualified",
        "ground_first_mate_tolerance_stack_unqualified",
    ),
    (
        "magnetic_fastener_seating_and_preload_bound",
        "magnetic_fastener_seating_and_preload_unproven",
    ),
    (
        "moving_interface_pair_route_recomputed",
        "moving_interface_pair_route_not_recomputed",
    ),
    (
        "printed_interface_feature_strength_qualified",
        "printed_interface_feature_strength_unqualified",
    ),
)


def _interface_authority_verdict(
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Recompute authority blockers instead of trusting a declared verdict."""

    expected_blockers = [
        blocker
        for flag, blocker in INTERFACE_AUTHORITY_REQUIREMENTS
        if authority.get(flag) is not True
    ]
    if (
        authority.get("fabrication_process_tolerance_qualified") is True
        and authority.get("qualified_combined_error_limit_mm") is None
    ):
        expected_blockers.append("qualified_combined_error_limit_missing")
    computed_release_ready = not expected_blockers
    return {
        "expected_blockers": expected_blockers,
        "computed_release_ready": computed_release_ready,
        "declaration_consistent": bool(
            authority.get("blockers") == expected_blockers
            and authority.get("release_ready") is computed_release_ready
        ),
    }


def _interface_hardware_fit_record() -> dict[str, Any]:
    """Recompute nominal mating-face fit and preserve unresolved authorities."""

    contract = CAD.interface_hardware_fit_contract()
    required = float(contract["required_clearance_mm"])
    motion_allowance = float(contract["unqualified_local_motion_allowance_mm"])
    robot_native = CAD.robot_plate().val()
    tool_native = CAD.tool_plate(stock_gripper=True).val()
    robot = CAD.robot_plate().translate(
        (0.0, 0.0, -CAD.PLATE_THICKNESS)
    ).val()
    tool = tool_native

    pad_records: list[dict[str, Any]] = []
    for index, (x_value, y_value) in enumerate(CAD.pogo_points(), start=1):
        pad = CAD.contact_pad().translate((x_value, y_value, -0.05)).val()
        distance = float(robot.distance(pad))
        pad_records.append(
            {
                "index": index,
                "centre_xy_mm": [x_value, y_value],
                "distance_mm": distance,
                "overlap_volume_mm3": _intersection_volume_mm3(robot, pad),
                "static_residual_after_unqualified_motion_allowance_mm": (
                    distance - motion_allowance
                ),
            }
        )

    fixed_shell_reference_records: list[dict[str, Any]] = []
    for index, ((x_value, y_value), signal) in enumerate(
        zip(CAD.pogo_points(), CAD.CONTACT_SIGNALS), start=1
    ):
        protrusion = (
            CAD.POGO_GROUND_PROTRUSION
            if signal == "GND"
            else CAD.POGO_STANDARD_PROTRUSION
        )
        installed_z = CAD.PLATE_THICKNESS + protrusion - CAD.POGO_OVERALL_LENGTH
        shell = CAD.pogo_reference_fixed_shell().translate(
            (x_value, y_value, installed_z)
        ).val()
        fixed_shell_reference_records.append(
            {
                "index": index,
                "signal": signal,
                "installed_z_mm": installed_z,
                "reference_overlap_with_printed_plate_mm3": (
                    _intersection_volume_mm3(shell, robot_native)
                ),
                "semantics": (
                    "illustrative_reference_overlap_not_exact_or_conservative_part_authority"
                ),
            }
        )

    magnetic_records: list[dict[str, Any]] = []
    unqualified_target_screw = _countersunk_screw(5.0, 10.0, 2.7, 10.0)
    for index, (x_value, y_value) in enumerate(CAD.magnet_points(), start=1):
        magnet = CAD.screw_on_magnet().translate(
            (
                x_value,
                y_value,
                -CAD.MAGNET_HEIGHT - CAD.MAGNETIC_HARDWARE_FACE_RECESS,
            )
        ).val()
        target = CAD.steel_target().translate(
            (x_value, y_value, CAD.MAGNETIC_HARDWARE_FACE_RECESS)
        ).val()
        target_screw = unqualified_target_screw.translate(
            (x_value, y_value, 0.0)
        ).val()
        magnetic_records.append(
            {
                "index": index,
                "centre_xy_mm": [x_value, y_value],
                "magnet_to_own_pocket_distance_mm": float(
                    magnet.distance(robot)
                ),
                "magnet_to_own_pocket_overlap_mm3": (
                    _intersection_volume_mm3(magnet, robot)
                ),
                "target_to_own_pocket_distance_mm": float(
                    target.distance(tool)
                ),
                "target_to_own_pocket_overlap_mm3": (
                    _intersection_volume_mm3(target, tool)
                ),
                "magnet_to_opposing_plate_distance_mm": float(
                    magnet.distance(tool)
                ),
                "magnet_to_opposing_plate_overlap_mm3": (
                    _intersection_volume_mm3(magnet, tool)
                ),
                "target_to_opposing_plate_distance_mm": float(
                    target.distance(robot)
                ),
                "target_to_opposing_plate_overlap_mm3": (
                    _intersection_volume_mm3(target, robot)
                ),
                "magnet_to_target_air_gap_mm": float(magnet.distance(target)),
                "target_to_unqualified_reference_screw_distance_mm": float(
                    target.distance(target_screw)
                ),
                "target_fastener_semantics": (
                    "reference_transform_only_seating_and_preload_unproven"
                ),
            }
        )

    stud_records: list[dict[str, Any]] = []
    for x_value in (-CAD.LOCK_STUD_X, CAD.LOCK_STUD_X):
        stud = CAD.shoulder_lock_stud().translate((x_value, 0.0, 0.0)).val()
        distance = float(robot.distance(stud))
        stud_records.append(
            {
                "centre_x_mm": x_value,
                "distance_mm": distance,
                "overlap_volume_mm3": _intersection_volume_mm3(robot, stud),
                "static_residual_after_unqualified_motion_allowance_mm": (
                    distance - motion_allowance
                ),
            }
        )

    geometry_checks = {
        "robot_plate_is_one_valid_solid": bool(
            robot_native.isValid() and len(robot_native.Solids()) == 1
        ),
        "stock_tool_plate_is_one_valid_solid": bool(
            tool_native.isValid() and len(tool_native.Solids()) == 1
        ),
        "pogo_target_pads_do_not_overlap_robot_plate": bool(
            pad_records
            and all(
                record["overlap_volume_mm3"] <= OVERLAP_VOLUME_TOLERANCE_MM3
                for record in pad_records
            )
        ),
        "pogo_target_pad_seated_clearance_passes": bool(
            pad_records
            and min(
                record[
                    "static_residual_after_unqualified_motion_allowance_mm"
                ]
                for record in pad_records
            )
            + NUMERIC_DISTANCE_TOLERANCE_MM
            >= required
        ),
        "magnet_and_target_rear_faces_bear_on_own_pocket_floors": bool(
            magnetic_records
            and all(
                abs(record["magnet_to_own_pocket_distance_mm"])
                <= NUMERIC_DISTANCE_TOLERANCE_MM
                and record["magnet_to_own_pocket_overlap_mm3"]
                <= OVERLAP_VOLUME_TOLERANCE_MM3
                and abs(record["target_to_own_pocket_distance_mm"])
                <= NUMERIC_DISTANCE_TOLERANCE_MM
                and record["target_to_own_pocket_overlap_mm3"]
                <= OVERLAP_VOLUME_TOLERANCE_MM3
                for record in magnetic_records
            )
        ),
        "magnetic_hardware_cross_plate_seated_clearance_passes": bool(
            magnetic_records
            and all(
                min(
                    record["magnet_to_opposing_plate_distance_mm"],
                    record["target_to_opposing_plate_distance_mm"],
                )
                - motion_allowance
                + NUMERIC_DISTANCE_TOLERANCE_MM
                >= required
                and record["magnet_to_opposing_plate_overlap_mm3"]
                <= OVERLAP_VOLUME_TOLERANCE_MM3
                and record["target_to_opposing_plate_overlap_mm3"]
                <= OVERLAP_VOLUME_TOLERANCE_MM3
                for record in magnetic_records
            )
        ),
        "magnet_target_recessed_air_gap_matches_source": bool(
            magnetic_records
            and all(
                math.isclose(
                    record["magnet_to_target_air_gap_mm"],
                    2.0 * CAD.MAGNETIC_HARDWARE_FACE_RECESS,
                    abs_tol=1.0e-12,
                )
                for record in magnetic_records
            )
        ),
        "stud_heads_do_not_overlap_fixed_plate": bool(
            stud_records
            and all(
                record["overlap_volume_mm3"] <= OVERLAP_VOLUME_TOLERANCE_MM3
                for record in stud_records
            )
        ),
        "stud_head_seated_clearance_passes": bool(
            stud_records
            and min(
                record[
                    "static_residual_after_unqualified_motion_allowance_mm"
                ]
                for record in stud_records
            )
            + NUMERIC_DISTANCE_TOLERANCE_MM
            >= required
        ),
        "official_plunger_diameter_is_smaller_than_legacy_reference_pilot": bool(
            CAD.POGO_PLUNGER_DIAMETER
            < CAD.POGO_LEGACY_REFERENCE_PILOT_DIAMETER
        ),
    }
    geometry_passed = all(geometry_checks.values())
    release_authority = contract["release_authority"]
    authority_verdict = _interface_authority_verdict(release_authority)
    authority_blockers = list(authority_verdict["expected_blockers"])
    authority_passed = bool(
        authority_verdict["computed_release_ready"]
        and authority_verdict["declaration_consistent"]
    )
    passed = bool(geometry_passed and authority_passed)
    return {
        "source_contract": contract,
        "pogo_target_pad_records": pad_records,
        "fixed_pogo_shell_reference_records": fixed_shell_reference_records,
        "magnetic_hardware_records": magnetic_records,
        "fixed_stud_head_records": stud_records,
        "geometry_checks": geometry_checks,
        "geometry_passed": geometry_passed,
        "authority_verdict": authority_verdict,
        "authority_blockers": authority_blockers,
        "authority_passed": authority_passed,
        "passed": passed,
        "release_ready": passed,
    }


def _positive_lock_travel_record() -> dict[str, Any]:
    """Prove shoulder clearance and head retention across the full lock stroke."""

    sample_count = int(
        round(CAD.SLIDER_TRAVEL / POSITIVE_LOCK_TRAVEL_SWEEP_STEP_MM)
    ) + 1
    offsets = [
        index * CAD.SLIDER_TRAVEL / (sample_count - 1)
        for index in range(sample_count)
    ]
    shoulder_solids = {
        side: CAD.axis_cylinder(
            CAD.LOCK_SHOULDER_DIAMETER,
            CAD.LOCK_SHOULDER_LENGTH,
            (
                x_value,
                0.0,
                CAD.PLATE_THICKNESS - CAD.LOCK_SHOULDER_LENGTH,
            ),
        ).val()
        for side, x_value in (("left", -CAD.LOCK_STUD_X), ("right", CAD.LOCK_STUD_X))
    }
    head_projection_solids = {
        side: CAD.axis_cylinder(
            CAD.LOCK_HEAD_DIAMETER,
            CAD.SLIDER_THICKNESS,
            (x_value, 0.0, CAD.SLIDER_Z),
        ).val()
        for side, x_value in (("left", -CAD.LOCK_STUD_X), ("right", CAD.LOCK_STUD_X))
    }
    observations: list[dict[str, Any]] = []
    for offset in offsets:
        slider = CAD.locking_slider().translate((offset, 0.0, CAD.SLIDER_Z)).val()
        sides: dict[str, Any] = {}
        for side, shoulder in shoulder_solids.items():
            sides[side] = {
                "shoulder_overlap_volume_mm3": _intersection_volume_mm3(
                    slider, shoulder
                ),
                "shoulder_clearance_mm": float(slider.distance(shoulder)),
            }
        observations.append({"slider_offset_mm": offset, "sides": sides})

    shoulder_overlaps = [
        record["sides"][side]["shoulder_overlap_volume_mm3"]
        for record in observations
        for side in ("left", "right")
    ]
    shoulder_clearances = [
        record["sides"][side]["shoulder_clearance_mm"]
        for record in observations
        for side in ("left", "right")
    ]
    unlocked_slider = CAD.locking_slider().translate((0.0, 0.0, CAD.SLIDER_Z)).val()
    locked_slider = CAD.locking_slider().translate(
        (CAD.SLIDER_TRAVEL, 0.0, CAD.SLIDER_Z)
    ).val()
    unlocked_head_overlap = {
        side: _intersection_volume_mm3(unlocked_slider, head)
        for side, head in head_projection_solids.items()
    }
    locked_head_retention = {
        side: _intersection_volume_mm3(locked_slider, head)
        for side, head in head_projection_solids.items()
    }
    contract = CAD.positive_lock_keyhole_contract()
    checks = {
        "all_sampled_shoulder_intersections_are_zero": bool(
            max(shoulder_overlaps) <= OVERLAP_VOLUME_TOLERANCE_MM3
        ),
        "continuous_capsule_radial_clearance_is_positive": bool(
            contract["minimum_radial_shoulder_clearance_mm"]
            > NUMERIC_DISTANCE_TOLERANCE_MM
        ),
        "sampled_clearance_matches_continuous_capsule_bound": bool(
            min(shoulder_clearances) + NUMERIC_DISTANCE_TOLERANCE_MM
            >= contract["minimum_radial_shoulder_clearance_mm"]
        ),
        "unlocked_entry_passes_each_head": bool(
            max(unlocked_head_overlap.values()) <= OVERLAP_VOLUME_TOLERANCE_MM3
        ),
        "locked_neck_retains_each_head": bool(
            contract["minimum_radial_head_retention_overlap_mm"] > 0.0
            and min(locked_head_retention.values()) > OVERLAP_VOLUME_TOLERANCE_MM3
        ),
    }
    return {
        "source_frame": "robot_plate_native_mm",
        "slider_translation_axis": [1.0, 0.0, 0.0],
        "slider_z_bounds_mm": [
            CAD.SLIDER_Z,
            CAD.SLIDER_Z + CAD.SLIDER_THICKNESS,
        ],
        "shoulder_z_bounds_mm": [
            CAD.PLATE_THICKNESS - CAD.LOCK_SHOULDER_LENGTH,
            CAD.PLATE_THICKNESS,
        ],
        "sample_start_mm": 0.0,
        "sample_end_mm": CAD.SLIDER_TRAVEL,
        "sample_step_mm": POSITIVE_LOCK_TRAVEL_SWEEP_STEP_MM,
        "sample_count": sample_count,
        "sampled_offsets_sha256": _canonical_sha256(offsets),
        "maximum_sampled_shoulder_overlap_volume_mm3": max(shoulder_overlaps),
        "minimum_sampled_shoulder_clearance_mm": min(shoulder_clearances),
        "continuous_minimum_shoulder_clearance_mm": contract[
            "minimum_radial_shoulder_clearance_mm"
        ],
        "unlocked_projected_head_overlap_volume_mm3": unlocked_head_overlap,
        "locked_projected_head_retention_volume_mm3": locked_head_retention,
        "observations": observations,
        "method": "occt_brep_boolean_distance_plus_exact_capsule_containment",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _mesh_sweep_record(
    moving: TriangleMesh,
    dock_shape: cq.Shape,
    positions_mm: list[float],
) -> dict[str, Any]:
    mesh_bounds = (
        (float(moving.vertices_mm[:, 0].min()), float(moving.vertices_mm[:, 0].max())),
        (float(moving.vertices_mm[:, 1].min()), float(moving.vertices_mm[:, 1].max())),
        (float(moving.vertices_mm[:, 2].min()), float(moving.vertices_mm[:, 2].max())),
    )
    dock_bounds = _bbox_tuple(dock_shape)
    swept_bounds = (
        mesh_bounds[0],
        (mesh_bounds[1][0] - SWEEP_END_MM, mesh_bounds[1][1] - SWEEP_START_MM),
        mesh_bounds[2],
    )
    continuous_aabb_clearance = _bbox_distance(swept_bounds, dock_bounds)
    witness_position = min(
        positions_mm,
        key=lambda withdrawal: _bbox_distance(
            _translated_bounds(mesh_bounds, (0.0, -withdrawal, 0.0)), dock_bounds
        ),
    )
    screen = _mesh_screen(moving, dock_shape, (0.0, -witness_position, 0.0))
    passed = continuous_aabb_clearance >= MANUFACTURING_CLEARANCE_MM
    return {
        "component": moving.name,
        "dock_component": "full_dock",
        "component_state": moving.state,
        "component_role": "calibrated_source_mesh_payload",
        "semantics": "forbidden_component_continuous_clearance",
        "sample_count": len(positions_mm),
        "sample_step_mm": positions_mm[1] - positions_mm[0],
        "sampled_positions_sha256": _canonical_sha256(positions_mm),
        "continuous_aabb_clearance_mm": continuous_aabb_clearance,
        "witness_withdrawal_mm": witness_position,
        "mesh_screen": screen,
        "method": "continuous_source_mesh_aabb_bound_plus_fcpw_witness",
        "passed": passed,
    }


def _stop_envelope(
    tool_components: list[BRepComponent],
    moving_jaw: TriangleMesh,
    stop_shape: cq.Shape,
) -> dict[str, Any]:
    stock_plate = next(component for component in tool_components if component.name == "stock_tool_plate")
    plate_bounds = _bbox_tuple(stock_plate.shape.val())
    stop_bounds = _bbox_tuple(stop_shape)
    seated_gap = stop_bounds[1][0] - plate_bounds[1][1]
    records: list[dict[str, Any]] = []
    for component in tool_components:
        for forward_mm in (0.0, STOP_FORWARD_ENVELOPE_MM):
            placed = component.shape.translate((0.0, forward_mm, 0.0)).val()
            distance = float(placed.distance(stop_shape))
            overlap = (
                _intersection_volume_mm3(placed, stop_shape)
                if distance <= NUMERIC_DISTANCE_TOLERANCE_MM
                else 0.0
            )
            intentional_plate_stop = (
                component.name == "stock_tool_plate"
                and forward_mm == STOP_FORWARD_ENVELOPE_MM
            )
            passed = (
                overlap <= OVERLAP_VOLUME_TOLERANCE_MM3
                and (
                    intentional_plate_stop
                    or distance + NUMERIC_DISTANCE_TOLERANCE_MM >= MANUFACTURING_CLEARANCE_MM
                )
            )
            records.append(
                {
                    "component": component.name,
                    "forward_mm": forward_mm,
                    "distance_mm": distance,
                    "overlap_volume_mm3": overlap,
                    "semantic": (
                        "intended_plate_stop_tangency"
                        if intentional_plate_stop
                        else "forbidden_stop_clearance"
                    ),
                    "passed": passed,
                }
            )
    jaw_bounds = (
        (float(moving_jaw.vertices_mm[:, 0].min()), float(moving_jaw.vertices_mm[:, 0].max())),
        (float(moving_jaw.vertices_mm[:, 1].min()), float(moving_jaw.vertices_mm[:, 1].max())),
        (float(moving_jaw.vertices_mm[:, 2].min()), float(moving_jaw.vertices_mm[:, 2].max())),
    )
    for forward_mm in (0.0, STOP_FORWARD_ENVELOPE_MM):
        clearance = _bbox_distance(
            _translated_bounds(jaw_bounds, (0.0, forward_mm, 0.0)), stop_bounds
        )
        records.append(
            {
                "component": moving_jaw.name,
                "forward_mm": forward_mm,
                "distance_mm": clearance,
                "overlap_volume_mm3": None,
                "semantic": "forbidden_stop_continuous_aabb_clearance",
                "passed": clearance >= MANUFACTURING_CLEARANCE_MM,
            }
        )
    return {
        "stop_bbox": _bbox_record(stop_bounds),
        "stock_plate_bbox": _bbox_record(plate_bounds),
        "seated_plate_to_stop_y_gap_mm": seated_gap,
        "required_gap_range_mm": [STOP_GAP_MIN_MM, STOP_GAP_MAX_MM],
        "forward_envelope_mm": STOP_FORWARD_ENVELOPE_MM,
        "component_results": records,
        "passed": (
            STOP_GAP_MIN_MM <= seated_gap <= STOP_GAP_MAX_MM
            and all(record["passed"] for record in records)
        ),
    }


def _stud_cam_inequality() -> dict[str, Any]:
    stud_head_x_max = CAD.LOCK_STUD_X + CAD.LOCK_HEAD_DIAMETER / 2.0
    cam_inner_x = CAD.DOCK_CAM_X_INNER
    unlocked_tab_end_x = CAD.SLIDER_TAB_END_X
    locked_tab_end_x = CAD.SLIDER_TAB_END_X + CAD.SLIDER_TRAVEL
    stud_to_cam_margin = cam_inner_x - stud_head_x_max
    unlocked_tab_gap = cam_inner_x - unlocked_tab_end_x
    locked_tab_engagement = locked_tab_end_x - cam_inner_x
    return {
        "stud_center_x_mm": CAD.LOCK_STUD_X,
        "stud_head_radius_mm": CAD.LOCK_HEAD_DIAMETER / 2.0,
        "stud_head_x_max_mm": stud_head_x_max,
        "cam_inner_x_mm": cam_inner_x,
        "slider_unlocked_tab_end_x_mm": unlocked_tab_end_x,
        "slider_locked_tab_end_x_mm": locked_tab_end_x,
        "stud_to_cam_x_margin_mm": stud_to_cam_margin,
        "required_stud_to_cam_x_margin_mm": MIN_STUD_TO_CAM_X_MARGIN_MM,
        "unlocked_tab_to_cam_gap_mm": unlocked_tab_gap,
        "locked_tab_cam_engagement_mm": locked_tab_engagement,
        "inequalities": {
            "cam_misses_stud_head": stud_head_x_max + MIN_STUD_TO_CAM_X_MARGIN_MM <= cam_inner_x,
            "cam_starts_beyond_unlocked_tab": unlocked_tab_end_x <= cam_inner_x,
            "cam_reaches_locked_tab": cam_inner_x <= locked_tab_end_x,
        },
        "passed": (
            stud_head_x_max + MIN_STUD_TO_CAM_X_MARGIN_MM <= cam_inner_x
            and unlocked_tab_end_x <= cam_inner_x <= locked_tab_end_x
        ),
    }


def _mount_composition() -> dict[str, Any]:
    contract = _source_mount_contract()
    source_plate = CAD.tool_plate(stock_gripper=True).val()
    exported_plate = cq.importers.importStep(
        str(CORE_EXPORT_DIR / "so101_stock_gripper_tool_plate.step")
    ).val()
    plate_volume_delta = abs(_shape_volume_mm3(source_plate) - _shape_volume_mm3(exported_plate))

    fixed_step = cq.importers.importStep(str(FIXED_GRIPPER_STEP_PATH)).val()
    fixed_stl_vertices_m, _ = _read_binary_stl(FIXED_GRIPPER_STL_PATH)
    fixed_step_bounds = _bbox_tuple(fixed_step)
    fixed_stl_bounds_mm = tuple(
        (
            float(fixed_stl_vertices_m[:, axis].min() * 1000.0),
            float(fixed_stl_vertices_m[:, axis].max() * 1000.0),
        )
        for axis in range(3)
    )
    bbox_error = max(
        abs(fixed_step_bounds[axis][endpoint] - fixed_stl_bounds_mm[axis][endpoint])
        for axis in range(3)
        for endpoint in range(2)
    )
    calibration = _xml_calibration()
    required_sim_transform = _required_stock_sim_body_transform()
    return {
        **contract,
        "stock_plate_source_vs_export_volume_delta_mm3": plate_volume_delta,
        "stock_plate_volume_tolerance_mm3": 1.0e-6,
        "fixed_step_bbox": _bbox_record(fixed_step_bounds),
        "fixed_stl_scaled_bbox": _bbox_record(fixed_stl_bounds_mm),
        "fixed_step_to_stl_bbox_max_error_mm": bbox_error,
        "fixed_step_to_stl_bbox_tolerance_mm": 0.001,
        "calibrated_xml_transform_chain": {
            key: value.tolist() for key, value in calibration.items()
        },
        "rack_interlock": {
            "required_jaw_state": "closed_limit",
            "closed_angle_rad": float(calibration["jaw_joint_range_rad"][0]),
            "reason": "jaw motion is disabled during docking; open-jaw rack clearance is outside this authority",
        },
        "required_sim_stock_wrapper_transform": required_sim_transform,
        "passed": (
            plate_volume_delta <= 1.0e-6
            and bbox_error <= 0.001
            and required_sim_transform["position_residual_m"] <= 1.0e-12
            and required_sim_transform["rotation_residual_frobenius"] <= 1.0e-12
        ),
    }


def _thresholds_record() -> dict[str, Any]:
    return {
        "manufacturing_clearance_mm": MANUFACTURING_CLEARANCE_MM,
        "numeric_distance_tolerance_mm": NUMERIC_DISTANCE_TOLERANCE_MM,
        "overlap_volume_tolerance_mm3": OVERLAP_VOLUME_TOLERANCE_MM3,
        "stop_gap_range_mm": [STOP_GAP_MIN_MM, STOP_GAP_MAX_MM],
        "minimum_stud_to_cam_x_margin_mm": MIN_STUD_TO_CAM_X_MARGIN_MM,
        "robot_plate_cam_required_clearance_mm": CAD.ROBOT_CAM_CLEARANCE_MM,
        "robot_plate_cam_sample_step_mm": CAM_CLEARANCE_SWEEP_STEP_MM,
        "axial_capture_preseat_mm": AXIAL_CAPTURE_PRESEAT_MM,
        "axial_capture_sample_step_mm": AXIAL_CAPTURE_SWEEP_STEP_MM,
        "axial_capture_guided_offset_mm": CAD.ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM,
        "robot_plate_expected_source_volume_mm3": (
            ROBOT_PLATE_EXPECTED_SOURCE_VOLUME_MM3
        ),
        "robot_plate_retained_volume_guard_role": (
            "secondary_sanity_only_local_web_ligament_and_clearance_checks_are_authoritative"
        ),
        "robot_plate_minimum_functional_ligament_mm": (
            ROBOT_PLATE_MINIMUM_FUNCTIONAL_LIGAMENT_MM
        ),
        "positive_lock_travel_sample_step_mm": (
            POSITIVE_LOCK_TRAVEL_SWEEP_STEP_MM
        ),
        "positive_lock_minimum_radial_shoulder_clearance_mm": (
            CAD.positive_lock_keyhole_contract()[
                "minimum_radial_shoulder_clearance_mm"
            ]
        ),
        "positive_lock_minimum_radial_head_retention_overlap_mm": (
            CAD.positive_lock_keyhole_contract()[
                "minimum_radial_head_retention_overlap_mm"
            ]
        ),
        "passive_cam_open_q_max_mm": CAD.DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM,
        "passive_cam_head_entry_preseat_mm": (
            CAD.DOCK_CAM_HEAD_ENTRY_PRESEAT_MM
        ),
    }


def _expected_core_manifest_contracts() -> dict[str, Any]:
    return {
        "robot_plate_cam_relief": CAD.robot_cam_relief_contract(),
        "interface_hardware_fit": CAD.interface_hardware_fit_contract(),
        "core_dock_stop": CAD.core_dock_stop_spec(),
        "stock_gripper_mount": CAD.stock_gripper_mount_contract(),
        "positive_lock_keyhole": CAD.positive_lock_keyhole_contract(),
        "positive_lock_cam": CAD.positive_lock_cam_contract(),
    }


def validate_core_manifest() -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(CORE_MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"core_manifest_unreadable:{exc}"]
    if manifest.get("schema_version") != "1.0" or manifest.get("units") != "mm":
        errors.append("core_manifest_schema_or_units_mismatch")
    if manifest.get("generator") != _file_record(CAD_GENERATOR_PATH):
        errors.append("core_manifest_generator_record_mismatch")
    if manifest.get("contracts") != _expected_core_manifest_contracts():
        errors.append("core_manifest_contract_mismatch")
    records = manifest.get("files", [])
    if not isinstance(records, list):
        return errors + ["core_manifest_files_not_list"]
    expected_paths = [
        f"QuickChange/SO101_Magnetic/exports/{name}"
        for name in sorted(CAD.CORE_OUTPUT_NAMES)
    ]
    observed_paths = [record.get("path") for record in records]
    if observed_paths != expected_paths:
        errors.append("core_manifest_file_inventory_mismatch")
    expected_records = [
        {
            **_file_record(CORE_EXPORT_DIR / name),
            "role": CAD._artifact_role(name),
        }
        for name in sorted(CAD.CORE_OUTPUT_NAMES)
    ]
    # _file_record emits path/bytes/hash; normalize key order only for clarity.
    expected_records = [
        {
            "path": record["path"],
            "role": record["role"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in expected_records
    ]
    if records != expected_records:
        errors.append("core_manifest_file_record_mismatch")
    if manifest.get("file_count") != len(expected_records):
        errors.append("core_manifest_file_count_mismatch")
    inventory_payload = [
        {key: record[key] for key in ("path", "role", "bytes", "sha256")}
        for record in expected_records
    ]
    if manifest.get("inventory_sha256") != _canonical_sha256(inventory_payload):
        errors.append("core_manifest_inventory_digest_mismatch")
    return errors


def _authorities_record() -> dict[str, Any]:
    return {
        "validator": _file_record(Path(__file__)),
        "cad_generator": _file_record(CAD_GENERATOR_PATH),
        "calibrated_robot_xml": _file_record(ROBOT_XML_PATH),
        "official_fixed_gripper_step": _file_record(FIXED_GRIPPER_STEP_PATH),
        "fixed_gripper_stl_crosscheck": _file_record(FIXED_GRIPPER_STL_PATH),
        "moving_jaw_source_mesh": _file_record(MOVING_JAW_STL_PATH),
        "core_cad_manifest": _file_record(CORE_MANIFEST_PATH),
        "core_exports": [
            _file_record(CORE_EXPORT_DIR / name) for name in EXPECTED_CORE_EXPORTS
        ],
    }


def _expected_inventory_names() -> list[str]:
    return sorted(
        [
            component.name
            for component in _tool_side_components() + _robot_side_components()
        ]
        + ["moving_jaw_closed_calibrated_stl"]
    )


def build_report(step_mm: float = DEFAULT_SWEEP_STEP_MM) -> dict[str, Any]:
    positions = _sweep_positions(step_mm)
    cam_clearance_positions = _sweep_positions(CAM_CLEARANCE_SWEEP_STEP_MM)
    dock = _dock_authority()
    mount = _mount_composition()
    calibration = _xml_calibration()
    jaw_closed = _moving_jaw_tool_mesh(float(calibration["jaw_joint_range_rad"][0]))
    tool_components = _tool_side_components()
    robot_components = _robot_side_components()

    path_results: list[dict[str, Any]] = []
    separated_components = {pair[0] for pair in INTENDED_ZERO_VOLUME_CONTACT_PAIRS}
    for component in tool_components + robot_components:
        if component.name not in separated_components:
            continue
        for feature_name in DOCK_FEATURE_NAMES:
            pair = (component.name, feature_name)
            pair_positions = (
                cam_clearance_positions
                if pair == ("robot_plate", "positive_lock_cam")
                else positions
            )
            path_results.append(
                _brep_sweep_record(
                    component,
                    dock[feature_name].val(),
                    pair_positions,
                    dock_component=feature_name,
                    intended_zero_volume_contact=(
                        pair in INTENDED_ZERO_VOLUME_CONTACT_PAIRS
                    ),
                )
            )

    for component in [
        item
        for item in tool_components + robot_components
        if item.name not in separated_components
    ]:
        forbidden_dock = (
            dock["dock_without_cam"].val()
            if component.role == "positive_lock_slider"
            else dock["full_dock"].val()
        )
        dock_component = (
            "dock_without_positive_lock_cam"
            if component.role == "positive_lock_slider"
            else "full_dock"
        )
        path_results.append(
            _brep_sweep_record(
                component,
                forbidden_dock,
                positions,
                dock_component=dock_component,
                intended_zero_volume_contact=False,
            )
        )
    path_results.append(_mesh_sweep_record(jaw_closed, dock["full_dock"].val(), positions))

    cam_contact_results = []
    for component in robot_components:
        if component.role != "positive_lock_slider":
            continue
        placed = component.shape.val()
        distance = float(placed.distance(dock["positive_lock_cam"].val()))
        overlap = (
            _intersection_volume_mm3(placed, dock["positive_lock_cam"].val())
            if distance <= NUMERIC_DISTANCE_TOLERANCE_MM
            else 0.0
        )
        cam_contact_results.append(
            {
                "slider_state": component.state,
                "distance_mm": distance,
                "overlap_volume_mm3": overlap,
                "semantic": "intended_cam_actuation_envelope",
                "passed": True,
            }
        )

    robot_plate = next(
        component for component in robot_components if component.name == "robot_plate"
    )
    axial_capture = _axial_capture_cam_record(
        robot_plate, dock["positive_lock_cam"].val()
    )
    passive_cam = _passive_positive_lock_cam_record()
    mechanism_preservation = _mechanism_preservation_record()
    interface_hardware_fit = _interface_hardware_fit_record()
    stop = _stop_envelope(
        tool_components,
        jaw_closed,
        dock["seating_stop"].val(),
    )
    stud_cam = _stud_cam_inequality()
    core_manifest_errors = validate_core_manifest()
    inventory = sorted(
        [component.name for component in tool_components + robot_components]
        + [jaw_closed.name]
    )
    blockers = []
    for result in path_results:
        if not result["passed"]:
            blockers.append(f"rack_sweep:{result['component']}:{result.get('component_state', '')}")
    if not mount["passed"]:
        blockers.append("mount_composition")
    if not stop["passed"]:
        blockers.append("stop_envelope")
    if not stud_cam["passed"]:
        blockers.append("stud_cam_inequality")
    if not axial_capture["passed"]:
        blockers.append("axial_capture_cam_clearance")
    if not passive_cam["passed"]:
        blockers.append("passive_positive_lock_cam")
    if not mechanism_preservation["passed"]:
        blockers.append("mechanism_preservation")
    if not interface_hardware_fit["passed"]:
        blockers.append("interface_hardware_fit_authority")
    if core_manifest_errors:
        blockers.append("core_cad_manifest")

    authorities = _authorities_record()
    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "units": "mm",
        "passed": passed,
        "release_ready": passed,
        "blockers": sorted(blockers),
        "authorities": authorities,
        "thresholds": _thresholds_record(),
        "mount_composition": mount,
        "dock_stop_contract": CAD.core_dock_stop_spec(),
        "core_cad_manifest_validation": {
            "manifest": _file_record(CORE_MANIFEST_PATH),
            "errors": core_manifest_errors,
            "passed": not core_manifest_errors,
        },
        "inventory": {
            "component_names": inventory,
            "component_count": len(inventory),
            "canonical_sha256": _canonical_sha256(inventory),
        },
        "intended_zero_volume_contact_pairs": [
            list(pair) for pair in sorted(INTENDED_ZERO_VOLUME_CONTACT_PAIRS)
        ],
        "withdrawal_sweep": {
            "axis": [0.0, -1.0, 0.0],
            "start_mm": SWEEP_START_MM,
            "end_mm": SWEEP_END_MM,
            "sample_step_mm": step_mm,
            "sampled_positions_mm": positions,
            "sampled_positions_sha256": _canonical_sha256(positions),
            "results": path_results,
            "passed": all(result["passed"] for result in path_results),
        },
        "stop_envelope": stop,
        "stud_cam_inequality": stud_cam,
        "axial_capture_sweep": axial_capture,
        "passive_positive_lock_cam": passive_cam,
        "mechanism_preservation": mechanism_preservation,
        "interface_hardware_fit": interface_hardware_fit,
        "cam_actuation_diagnostics": cam_contact_results,
        "validation": {
            "pair_result_count": len(path_results),
            "failed_pair_result_count": sum(not result["passed"] for result in path_results),
            "all_source_hashes_present": True,
            "machine_json_canonical_sha256_without_this_field": None,
        },
    }


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "units",
        "passed",
        "release_ready",
        "blockers",
        "authorities",
        "thresholds",
        "mount_composition",
        "dock_stop_contract",
        "core_cad_manifest_validation",
        "inventory",
        "intended_zero_volume_contact_pairs",
        "withdrawal_sweep",
        "stop_envelope",
        "stud_cam_inequality",
        "axial_capture_sweep",
        "passive_positive_lock_cam",
        "mechanism_preservation",
        "interface_hardware_fit",
        "cam_actuation_diagnostics",
        "validation",
    }
    if not required.issubset(report):
        errors.append(f"missing_top_level_keys:{sorted(required - set(report))}")
        return errors
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if report.get("units") != "mm":
        errors.append("units_mismatch")
    if report.get("thresholds") != _thresholds_record():
        errors.append("thresholds_mismatch")
    expected_contacts = [
        list(pair) for pair in sorted(INTENDED_ZERO_VOLUME_CONTACT_PAIRS)
    ]
    if report.get("intended_zero_volume_contact_pairs") != expected_contacts:
        errors.append("intended_contact_policy_mismatch")
    if report.get("authorities") != _authorities_record():
        errors.append("source_authority_record_mismatch")
    if report.get("mount_composition") != _mount_composition():
        errors.append("mount_composition_recomputation_mismatch")
    if report.get("dock_stop_contract") != CAD.core_dock_stop_spec():
        errors.append("dock_stop_contract_mismatch")
    current_manifest_errors = validate_core_manifest()
    expected_manifest_validation = {
        "manifest": _file_record(CORE_MANIFEST_PATH),
        "errors": current_manifest_errors,
        "passed": not current_manifest_errors,
    }
    if report.get("core_cad_manifest_validation") != expected_manifest_validation:
        errors.append("core_manifest_validation_mismatch")

    positions = report["withdrawal_sweep"].get("sampled_positions_mm", [])
    if not positions or positions[0] != 0.0 or positions[-1] != 80.0:
        errors.append("withdrawal_range_not_exact_0_to_80_mm")
    try:
        expected_positions = _sweep_positions(
            float(report["withdrawal_sweep"].get("sample_step_mm"))
        )
    except (TypeError, ValueError, RuntimeError):
        expected_positions = []
        errors.append("invalid_withdrawal_sample_step")
    if positions != expected_positions:
        errors.append("withdrawal_positions_not_recomputed")
    if report["withdrawal_sweep"].get("sampled_positions_sha256") != _canonical_sha256(positions):
        errors.append("withdrawal_position_digest_mismatch")
    results = report["withdrawal_sweep"].get("results", [])
    if report["validation"].get("pair_result_count") != len(results):
        errors.append("pair_result_count_mismatch")
    seen_result_keys: set[tuple[str, str, str]] = set()
    observed_intended_contacts: set[tuple[str, str]] = set()
    for index, result in enumerate(results):
        component = result.get("component")
        dock_component = result.get("dock_component")
        state = result.get("component_state")
        key = (str(component), str(dock_component), str(state))
        if key in seen_result_keys:
            errors.append(f"duplicate_path_result:{index}:{key}")
        seen_result_keys.add(key)
        semantics = result.get("semantics")
        pair = (component, dock_component)
        try:
            pair_step = float(result.get("sample_step_mm"))
            pair_positions = _sweep_positions(pair_step)
        except (TypeError, ValueError, RuntimeError):
            pair_step = math.nan
            pair_positions = []
            errors.append(f"invalid_pair_sample_step:{index}")
        if result.get("sample_count") != len(pair_positions):
            errors.append(f"pair_sample_count_mismatch:{index}")
        if result.get("sampled_positions_sha256") != _canonical_sha256(pair_positions):
            errors.append(f"pair_sample_digest_mismatch:{index}")
        required_pair_step = (
            CAM_CLEARANCE_SWEEP_STEP_MM
            if pair == ("robot_plate", "positive_lock_cam")
            else float(report["withdrawal_sweep"].get("sample_step_mm"))
        )
        if not math.isclose(pair_step, required_pair_step, abs_tol=1.0e-12):
            errors.append(f"pair_sample_policy_mismatch:{index}")
        if semantics == "intended_stock_plate_keeper_tangency":
            if pair not in INTENDED_ZERO_VOLUME_CONTACT_PAIRS:
                errors.append(f"unnamed_intended_contact:{index}:{pair}")
            else:
                observed_intended_contacts.add(pair)
            recomputed_result_passed = (
                float(result.get("initial_distance_mm", math.inf))
                <= NUMERIC_DISTANCE_TOLERANCE_MM
                and float(
                    result.get("maximum_sampled_overlap_volume_mm3", math.inf)
                )
                <= OVERLAP_VOLUME_TOLERANCE_MM3
            )
            if not math.isclose(
                float(result.get("maximum_between_sample_motion_bound_mm", math.nan)),
                pair_step / 2.0,
                abs_tol=1.0e-12,
            ):
                errors.append(f"pair_motion_bound_mismatch:{index}")
        elif result.get("method") == "occt_brep_distance_with_near_witness_boolean":
            minimum = float(result.get("minimum_sampled_distance_mm", math.nan))
            motion = float(result.get("maximum_between_sample_motion_bound_mm", math.nan))
            continuous = float(result.get("continuous_certified_clearance_mm", math.nan))
            if not math.isclose(motion, pair_step / 2.0, abs_tol=1.0e-12):
                errors.append(f"pair_motion_bound_mismatch:{index}")
            if not math.isclose(continuous, minimum - motion, abs_tol=1.0e-12):
                errors.append(f"continuous_clearance_arithmetic_mismatch:{index}")
            recomputed_result_passed = (
                continuous + NUMERIC_DISTANCE_TOLERANCE_MM
                >= MANUFACTURING_CLEARANCE_MM
                and float(
                    result.get("maximum_sampled_overlap_volume_mm3", math.inf)
                )
                <= OVERLAP_VOLUME_TOLERANCE_MM3
            )
        elif result.get("method") == "continuous_source_mesh_aabb_bound_plus_fcpw_witness":
            recomputed_result_passed = (
                float(result.get("continuous_aabb_clearance_mm", -math.inf))
                >= MANUFACTURING_CLEARANCE_MM
                and result.get("mesh_screen", {}).get("clearance_authority") is False
            )
        else:
            errors.append(f"unknown_path_result_method:{index}")
            recomputed_result_passed = False
        if result.get("passed") is not recomputed_result_passed:
            errors.append(f"path_result_verdict_mismatch:{index}")
    if observed_intended_contacts != set(INTENDED_ZERO_VOLUME_CONTACT_PAIRS):
        errors.append("intended_contact_result_coverage_mismatch")

    failed = sum(not result.get("passed", False) for result in results)
    if report["validation"].get("failed_pair_result_count") != failed:
        errors.append("failed_pair_result_count_mismatch")

    stop = report["stop_envelope"]
    stop_children = stop.get("component_results", [])
    stop_passed = (
        STOP_GAP_MIN_MM
        <= float(stop.get("seated_plate_to_stop_y_gap_mm", math.inf))
        <= STOP_GAP_MAX_MM
        and bool(stop_children)
        and all(child.get("passed") is True for child in stop_children)
    )
    if stop.get("passed") is not stop_passed:
        errors.append("stop_envelope_verdict_mismatch")
    expected_stud_cam = _stud_cam_inequality()
    if report.get("stud_cam_inequality") != expected_stud_cam:
        errors.append("stud_cam_inequality_recomputation_mismatch")
    robot_plate = next(
        component
        for component in _robot_side_components()
        if component.name == "robot_plate"
    )
    expected_axial_capture = _axial_capture_cam_record(
        robot_plate, _dock_authority()["positive_lock_cam"].val()
    )
    if report.get("axial_capture_sweep") != expected_axial_capture:
        errors.append("axial_capture_sweep_recomputation_mismatch")
    expected_passive_cam = _passive_positive_lock_cam_record()
    if report.get("passive_positive_lock_cam") != expected_passive_cam:
        errors.append("passive_positive_lock_cam_recomputation_mismatch")
    expected_mechanism = _mechanism_preservation_record()
    if report.get("mechanism_preservation") != expected_mechanism:
        errors.append("mechanism_preservation_recomputation_mismatch")
    expected_interface_fit = _interface_hardware_fit_record()
    if report.get("interface_hardware_fit") != expected_interface_fit:
        errors.append("interface_hardware_fit_recomputation_mismatch")

    expected_blockers = []
    for result in results:
        if not result.get("passed", False):
            expected_blockers.append(
                f"rack_sweep:{result.get('component')}:{result.get('component_state', '')}"
            )
    if report["mount_composition"].get("passed") is not True:
        expected_blockers.append("mount_composition")
    if not stop_passed:
        expected_blockers.append("stop_envelope")
    if expected_stud_cam.get("passed") is not True:
        expected_blockers.append("stud_cam_inequality")
    if expected_axial_capture.get("passed") is not True:
        expected_blockers.append("axial_capture_cam_clearance")
    if expected_passive_cam.get("passed") is not True:
        expected_blockers.append("passive_positive_lock_cam")
    if expected_mechanism.get("passed") is not True:
        expected_blockers.append("mechanism_preservation")
    if expected_interface_fit.get("passed") is not True:
        expected_blockers.append("interface_hardware_fit_authority")
    if current_manifest_errors:
        expected_blockers.append("core_cad_manifest")
    expected_blockers = sorted(expected_blockers)
    if report.get("blockers") != expected_blockers:
        errors.append("blocker_inventory_mismatch")

    recomputed_passed = (
        failed == 0
        and report["mount_composition"].get("passed") is True
        and stop_passed
        and expected_stud_cam.get("passed") is True
        and expected_axial_capture.get("passed") is True
        and expected_passive_cam.get("passed") is True
        and expected_mechanism.get("passed") is True
        and expected_interface_fit.get("passed") is True
        and not current_manifest_errors
        and not expected_blockers
    )
    if report.get("passed") != recomputed_passed or report.get("release_ready") != recomputed_passed:
        errors.append("top_level_verdict_not_recomputed")
    inventory = report["inventory"].get("component_names", [])
    if report["inventory"].get("component_count") != len(inventory):
        errors.append("component_inventory_count_mismatch")
    if report["inventory"].get("canonical_sha256") != _canonical_sha256(inventory):
        errors.append("component_inventory_digest_mismatch")
    if inventory != _expected_inventory_names():
        errors.append("component_inventory_source_mismatch")

    digest = report["validation"].get(
        "machine_json_canonical_sha256_without_this_field"
    )
    if digest is not None:
        digest_payload = json.loads(json.dumps(report))
        digest_payload["validation"][
            "machine_json_canonical_sha256_without_this_field"
        ] = None
        if digest != _canonical_sha256(digest_payload):
            errors.append("machine_json_digest_mismatch")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="machine JSON output path",
    )
    parser.add_argument(
        "--sweep-step-mm",
        type=float,
        default=DEFAULT_SWEEP_STEP_MM,
        help="bounded withdrawal sample step; must land exactly at 80 mm",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="print the report without writing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args.sweep_step_mm)
    errors = validate_report(report)
    if errors:
        raise SystemExit("invalid CAD clearance report: " + "; ".join(errors))
    digest_payload = json.loads(json.dumps(report))
    digest_payload["validation"]["machine_json_canonical_sha256_without_this_field"] = None
    report["validation"]["machine_json_canonical_sha256_without_this_field"] = _canonical_sha256(
        digest_payload
    )
    if not args.stdout_only:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "release_ready": report["release_ready"],
                "blockers": report["blockers"],
                "component_count": report["inventory"]["component_count"],
                "sample_count": len(report["withdrawal_sweep"]["sampled_positions_mm"]),
                "report": None if args.stdout_only else str(args.report),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
