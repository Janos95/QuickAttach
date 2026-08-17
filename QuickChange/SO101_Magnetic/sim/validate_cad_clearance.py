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
ROBOT_XML_PATH = REPO_ROOT / "Simulation/SO101/so101_new_calib.xml"
FIXED_GRIPPER_STEP_PATH = (
    REPO_ROOT / "STEP/SO101/Follower_Specific/Wrist_Roll_Follower_SO101.step"
)
FIXED_GRIPPER_STL_PATH = (
    REPO_ROOT / "Simulation/SO101/assets/wrist_roll_follower_so101_v1.stl"
)
MOVING_JAW_STL_PATH = REPO_ROOT / "Simulation/SO101/assets/moving_jaw_so101_v1.stl"
REPORT_PATH = HERE / "cad_clearance_report.json"

SCHEMA_VERSION = "1.0"
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

# Generator save_assemblies() places the official STEP at global Z=19.051 mm
# while the stock tool mating face is global Z=9.5 mm.
FIXED_STEP_TO_TOOL_POS_MM = (0.4875, 0.218, 9.551)
FIXED_STEP_TO_TOOL_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

EXPECTED_CORE_EXPORTS = (
    "so101_robot_plate.step",
    "so101_stock_gripper_tool_plate.step",
    "so101_passive_tool_dock.step",
    "so101_stock_gripper_retrofit_assembly.step",
)


def _load_cad_generator():
    spec = importlib.util.spec_from_file_location("core_quick_change_cad", CAD_GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CAD generator {CAD_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAD = _load_cad_generator()


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
    stop = (
        cq.Workplane("XY")
        .box(82.0, 6.0, CAD.PLATE_THICKNESS + 6.0, centered=True)
        .translate((-4.0, 29.0, CAD.PLATE_THICKNESS / 2.0))
    )
    for x_value in (-25.0, 21.0):
        stop = stop.cut(
            CAD.axis_cylinder(
                4.4,
                6.2,
                (x_value, 25.9, CAD.PLATE_THICKNESS / 2.0),
                (0.0, 1.0, 0.0),
            )
        )
    features["seating_stop"] = stop.clean()
    features["positive_lock_cam"] = (
        cq.Workplane("XY")
        .polyline([(28.0, -16.0), (34.0, -16.0), (34.0, 0.0), (24.05, 0.0)])
        .close()
        .extrude(2.2)
        .translate((0.0, 0.0, -4.15))
        .clean()
    )
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
                    CAD.steel_target().translate((x_value, y_value, 0.0)),
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
                    (x_value, y_value, -CAD.MAGNET_HEIGHT)
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
        "cq.Vector(0, 0, PLATE_THICKNESS)",
        "cq.Vector(0.4875, 0.218, 2 * PLATE_THICKNESS + 0.051)",
    )
    missing = [fragment for fragment in required_fragments if fragment not in save_source]
    if missing:
        raise RuntimeError(f"stock-gripper mount composition drifted: {missing}")
    return {
        "source_fragments": list(required_fragments),
        "source_function_sha256": hashlib.sha256(save_source.encode()).hexdigest(),
        "robot_assembly_stock_plate_pos_mm": [0.0, 0.0, CAD.PLATE_THICKNESS],
        "robot_assembly_fixed_step_pos_mm": [
            0.4875,
            0.218,
            2 * CAD.PLATE_THICKNESS + 0.051,
        ],
        "tool_local_fixed_step_pos_mm": list(FIXED_STEP_TO_TOOL_POS_MM),
        "tool_local_fixed_step_quat_wxyz": list(FIXED_STEP_TO_TOOL_QUAT_WXYZ),
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
        "minimum_sampled_distance_mm": witness[1],
        "maximum_sampled_overlap_volume_mm3": maximum_overlap,
        "maximum_between_sample_motion_bound_mm": half_step,
        "continuous_certified_clearance_mm": continuous_clearance,
        "witness_withdrawal_mm": witness[0],
        "method": "occt_brep_distance_with_near_witness_boolean",
        "passed": passed,
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
        "component_state": moving.state,
        "component_role": "calibrated_source_mesh_payload",
        "semantics": "forbidden_component_continuous_clearance",
        "sample_count": len(positions_mm),
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
    cam_inner_x = 24.05
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
        "passed": plate_volume_delta <= 1.0e-6 and bbox_error <= 0.001,
    }


def build_report(step_mm: float = DEFAULT_SWEEP_STEP_MM) -> dict[str, Any]:
    positions = _sweep_positions(step_mm)
    dock = _dock_authority()
    mount = _mount_composition()
    calibration = _xml_calibration()
    jaw_closed = _moving_jaw_tool_mesh(float(calibration["jaw_joint_range_rad"][0]))
    tool_components = _tool_side_components()
    robot_components = _robot_side_components()

    path_results: list[dict[str, Any]] = []
    stock_plate = next(
        component for component in tool_components if component.name == "stock_tool_plate"
    )
    keeper_features = {
        "left_lower_rail",
        "right_lower_rail",
        "left_upper_rail",
        "right_upper_rail",
    }
    for feature_name in (
        "left_lower_rail",
        "right_lower_rail",
        "left_upper_rail",
        "right_upper_rail",
        "left_wall",
        "right_wall",
        "seating_stop",
        "positive_lock_cam",
    ):
        path_results.append(
            _brep_sweep_record(
                stock_plate,
                dock[feature_name].val(),
                positions,
                dock_component=feature_name,
                intended_zero_volume_contact=feature_name in keeper_features,
            )
        )

    for component in [
        item for item in tool_components + robot_components if item.name != "stock_tool_plate"
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

    stop = _stop_envelope(
        tool_components,
        jaw_closed,
        dock["seating_stop"].val(),
    )
    stud_cam = _stud_cam_inequality()
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

    authorities = {
        "validator": _file_record(Path(__file__)),
        "cad_generator": _file_record(CAD_GENERATOR_PATH),
        "calibrated_robot_xml": _file_record(ROBOT_XML_PATH),
        "official_fixed_gripper_step": _file_record(FIXED_GRIPPER_STEP_PATH),
        "fixed_gripper_stl_crosscheck": _file_record(FIXED_GRIPPER_STL_PATH),
        "moving_jaw_source_mesh": _file_record(MOVING_JAW_STL_PATH),
        "core_exports": [
            _file_record(CORE_EXPORT_DIR / name) for name in EXPECTED_CORE_EXPORTS
        ],
    }
    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "units": "mm",
        "passed": passed,
        "release_ready": passed,
        "blockers": sorted(blockers),
        "authorities": authorities,
        "thresholds": {
            "manufacturing_clearance_mm": MANUFACTURING_CLEARANCE_MM,
            "numeric_distance_tolerance_mm": NUMERIC_DISTANCE_TOLERANCE_MM,
            "overlap_volume_tolerance_mm3": OVERLAP_VOLUME_TOLERANCE_MM3,
            "stop_gap_range_mm": [STOP_GAP_MIN_MM, STOP_GAP_MAX_MM],
            "minimum_stud_to_cam_x_margin_mm": MIN_STUD_TO_CAM_X_MARGIN_MM,
        },
        "mount_composition": mount,
        "inventory": {
            "component_names": inventory,
            "component_count": len(inventory),
            "canonical_sha256": _canonical_sha256(inventory),
        },
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
        "mount_composition",
        "inventory",
        "withdrawal_sweep",
        "stop_envelope",
        "stud_cam_inequality",
        "validation",
    }
    if not required.issubset(report):
        errors.append(f"missing_top_level_keys:{sorted(required - set(report))}")
        return errors
    positions = report["withdrawal_sweep"].get("sampled_positions_mm", [])
    if not positions or positions[0] != 0.0 or positions[-1] != 80.0:
        errors.append("withdrawal_range_not_exact_0_to_80_mm")
    if report["withdrawal_sweep"].get("sampled_positions_sha256") != _canonical_sha256(positions):
        errors.append("withdrawal_position_digest_mismatch")
    results = report["withdrawal_sweep"].get("results", [])
    if report["validation"].get("pair_result_count") != len(results):
        errors.append("pair_result_count_mismatch")
    failed = sum(not result.get("passed", False) for result in results)
    if report["validation"].get("failed_pair_result_count") != failed:
        errors.append("failed_pair_result_count_mismatch")
    recomputed_passed = (
        failed == 0
        and report["mount_composition"].get("passed") is True
        and report["stop_envelope"].get("passed") is True
        and report["stud_cam_inequality"].get("passed") is True
        and not report.get("blockers")
    )
    if report.get("passed") != recomputed_passed or report.get("release_ready") != recomputed_passed:
        errors.append("top_level_verdict_not_recomputed")
    inventory = report["inventory"].get("component_names", [])
    if report["inventory"].get("component_count") != len(inventory):
        errors.append("component_inventory_count_mismatch")
    if report["inventory"].get("canonical_sha256") != _canonical_sha256(inventory):
        errors.append("component_inventory_digest_mismatch")
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
