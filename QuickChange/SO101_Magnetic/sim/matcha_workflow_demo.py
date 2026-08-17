#!/usr/bin/env python3
"""Collision-active SO-101 matcha workflow simulation.

This module builds the calibrated upstream robot and reconstructed matcha
workcell entirely in memory.  Runtime contacts are real MuJoCo contacts and
all controller motion advances through ``mj_step``; equality constraints are
used only for named mechanical captures/locks after their physical guards.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

import qc_collision_geometry as qc


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROBOT_XML = REPO_ROOT / "Simulation" / "SO101" / "so101_new_calib.xml"
SCENE_XML = HERE / "matcha_workflow_scene.xml"
CONFIG_PATH = HERE / "matcha_tool_geometry.json"
MATCHA_CAD_EXPORTS = HERE.parent / "matcha_tools" / "exports"
PAYLOAD_MASS_LEDGER_PATHS = {
    "spoon": MATCHA_CAD_EXPORTS / "so101_matcha_spoon_mass_ledger.json",
    "whisk": MATCHA_CAD_EXPORTS / "so101_matcha_whisk_mass_ledger.json",
}
PAYLOAD_MASS_LEDGER_SHA256 = {
    "spoon": "8871e3225cd0ce2b94df96c211d490fd581f83aba555d1cd62f46ad41dd81e37",
    "whisk": "4a66b75393e42f7004db99badc13d9a94898d28c2c1383c33fc0c8e00fc535fd",
}
WHISK_ROTOR_LEDGER_COMPONENT_IDS = (
    "whisk_motor_shaft",
    "whisk_eccentric_rotor",
    "whisk_eccentric_pin",
    "whisk_rotor_counterweight",
)
WHISK_CARRIAGE_LEDGER_COMPONENT_IDS = (
    "whisk_carriage_x",
    "whisk_compliance_carriage",
    "whisk_compliance_spring",
    "whisk_food_grade_bellows",
    "whisk_brush_hub",
    "whisk_bamboo_bristles",
    "whisk_brush_collision_envelope",
)
LEDGER_LOCK_HARDWARE_COMPONENT_IDS = (
    "shoulder_lock_stud_1_McMaster_90318A720",
    "lock_stud_nut_1_DIN934_M3",
    "shoulder_lock_stud_2_McMaster_90318A720",
    "lock_stud_nut_2_DIN934_M3",
)

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

# Deterministic calibrated IK solutions at 5 mm spacing along the mating-frame
# normal.  A dense FK sweep of each adjacent joint-linear interval gives a
# worst lateral deviation of 0.040539 mm and 6.67e-8 rad orientation error,
# comfortably inside the published 0.50 mm fixed-side cam-relief corridor.
# The first implicit point is PRE_CAPTURE_BASE_Q at 55 mm; these rows are the
# 50 mm through seated (0 mm) solutions.  Tool-specific routes substitute only
# the constant shoulder-pan value, preserving the same axial solution.
ALIGNED_CAPTURE_OFFSETS_MM = tuple(range(50, -1, -5))
ALIGNED_CAPTURE_BASE_Q = (
    (-1.05304069762456, 1.11178519885508, -0.0587445012305259),
    (-0.990100705135375, 1.08681965232811, -0.0967189471927332),
    (-0.928950517085393, 1.0602421719393, -0.13129165485391),
    (-0.869604742879709, 1.03215949513191, -0.162554752252201),
    (-0.81204042711556, 1.00266692686841, -0.190626499752852),
    (-0.756205853341443, 0.971848117233138, -0.215642263891695),
    (-0.702028101250179, 0.939775168527349, -0.237747067277171),
    (-0.649419286681867, 0.906508949902112, -0.257089663220245),
    (-0.59828154218571, 0.872099522648475, -0.273817980462766),
    (-0.548510866399194, 0.83658660199164, -0.288075735592447),
    (-0.5, 0.8, -0.3),
)
# Core-CAD-authorized +0.20 mm open-side guide route.  These deterministic
# 1 mm-spaced IK roots hold dock-local X=+0.200 mm and the mating orientation
# from 50 mm pre-seat through zero.  A dense joint-linear FK sweep measures
# 0.201841 mm maximum lateral deviation.  The released full-depth cam relief
# certifies >=0.250 mm continuous clearance at the nominal offset.  A same-Z
# final row establishes the exact five keeper datums.
CORE_GUIDED_CAPTURE_OFFSETS_MM = tuple(range(50, -1, -1))
CORE_GUIDED_CAPTURE_BASE_Q = (
    (-1.05309039863527, 1.11032046570321, -0.0572300670679406),
    (-1.04038114070762, 1.10546950157827, -0.0650883608706501),
    (-1.02774127826731, 1.10055035586343, -0.0728090775961235),
    (-1.01517138886496, 1.09556398173715, -0.080392592872194),
    (-1.00267196498263, 1.09051131632805, -0.0878393513454215),
    (-0.990243417310746, 1.08539328029565, -0.0951498629849015),
    (-0.977886078014275, 1.08021077745593, -0.102324699441657),
    (-0.965600203977752, 1.07496469444963, -0.109364490471876),
    (-0.953385980019733, 1.06965590045103, -0.116269920431296),
    (-0.941243522068057, 1.06428524691521, -0.123041724847152),
    (-0.929172880288271, 1.05885356736148, -0.12968068707321),
    (-0.91717404215838, 1.053361677191, -0.136187635032616),
    (-0.905246935483918, 1.04781037353641, -0.142563438052487),
    (-0.893391431348073, 1.04220043514156, -0.14880900379349),
    (-0.881607346992348, 1.03653262226929, -0.154925275276938),
    (-0.869894448623887, 1.03080767663523, -0.160913228011342),
    (-0.858252454146261, 1.02502632136603, -0.166773867219772),
    (-0.846681035811087, 1.0191892609799, -0.172508225168813),
    (-0.835179822788373, 1.01329718138788, -0.17811735859951),
    (-0.823748403654037, 1.00735074991418, -0.183602346260142),
    (-0.812386328793482, 1.00135061533383, -0.188964286540347),
    (-0.801093112720501, 0.995297407926249, -0.194204295205748),
    (-0.789868236311269, 0.989191739543146, -0.199323503231877),
    (-0.778711148953391, 0.983034203689355, -0.204323054735964),
    (-0.767621270610429, 0.976825375615287, -0.209204105004859),
    (-0.756597993802488, 0.970565812419668, -0.213967818617179),
    (-0.745640685503797, 0.964256053161365, -0.218615367657567),
    (-0.734748688958348, 0.957896618979131, -0.223147930020783),
    (-0.723921325414918, 0.951488013218161, -0.227566687803243),
    (-0.713157895782947, 0.94503072156243, -0.231872825779483),
    (-0.702457682210844, 0.938525212171808, -0.236067529960964),
    (-0.6918199495885, 0.931971935823044, -0.240151986234544),
    (-0.681243946975805, 0.925371326053714, -0.244127379077908),
    (-0.670728908959097, 0.918723799308316, -0.24799489034922),
    (-0.66027405693752, 0.91202975508574, -0.25175569814822),
    (-0.649878600341329, 0.905289576087353, -0.255410975746024),
    (-0.639541737784195, 0.898503628365028, -0.258961890580834),
    (-0.629262658151594, 0.891672261468451, -0.262409603316857),
    (-0.619040541627407, 0.8847958085911, -0.265755266963694),
    (-0.608874560660782, 0.877874586714305, -0.269000026053524),
    (-0.598763880875414, 0.870908896748876, -0.272145015873463),
    (-0.588707661923249, 0.863899023673746, -0.275191361750497),
    (-0.578705058284712, 0.856845236671178, -0.278140178386465),
    (-0.568755220017467, 0.849747789258067, -0.2809925692406),
    (-0.558857293455671, 0.842606919412918, -0.283749625957247),
    (-0.549010421861703, 0.835422849698069, -0.286412427836367),
    (-0.539213746032241, 0.828195787376817, -0.288982041344575),
    (-0.529466404860554, 0.820925924525033, -0.291459519664479),
    (-0.519767535856783, 0.813613438136955, -0.293845902280172),
    (-0.510116275628005, 0.80625849022481, -0.296142214596805),
    (-0.500511760319716, 0.798861227911943, -0.298349467592227),
)
CORE_GUIDED_CAPTURE_LATERAL_OFFSET_M = 0.0002

# Frozen source-coupled gripper capture route.  The embedded table stores
# dock-local preseated distance, source X offset, and five arm joint values in
# canonical little-endian float64 rows.  Keeping the exact binary roster here
# avoids a runtime IK dependency while the public contract below exposes every
# decoded row for independent FK replay.
CORE_CAPTURE_ROUTE_STATE_BYTES_SHA256 = (
    "e91e8699d4ef1d174d73341198e21499cbf615279e4e1a87a6fbe98929f0004c"
)
CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256 = (
    "3fef8469bf8cbddff822d9a6a1a31de9de2872be4bfe75d87575f01c83b99966"
)
CORE_CAPTURE_ROUTE_Q_SHA256 = (
    "107b40015a76fd55f09681164ae75aa12ae738a7385fba05ab4d94f7f29e40bd"
)
CORE_CAPTURE_ROUTE_PHASE_Q_SHA256 = {
    "gripper_capture_axial_open_side": (
        "2f06345e62fb5cbb4230bdee741addffba7810a39cb0d5681bad100e4c4f94ca"
    ),
    "gripper_capture_coupled_recenter": (
        "6df3f9c5e6be3581970694462fa1f2805b121f02db00a3a7d1303ec5ecf27792"
    ),
    "gripper_capture_centered_final": (
        "f6a6a64c60287807be1196735476b8abc81761e62135c2d50807e94aced63b27"
    ),
}
CORE_CAPTURE_ROUTE_ALIGNMENT_Q_SHA256 = (
    "39603d9ace7749f1001a27d04d36a4f33e071aff2367b213396c12f94e188299"
)
_CORE_CAPTURE_ROUTE_STATE_BASE64 = (
    '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'
)


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _decode_core_capture_route_source_states(
) -> tuple[tuple[float, float, tuple[float, ...]], ...]:
    raw = base64.b64decode(_CORE_CAPTURE_ROUTE_STATE_BASE64, validate=True)
    if hashlib.sha256(raw).hexdigest() != CORE_CAPTURE_ROUTE_STATE_BYTES_SHA256:
        raise RuntimeError("core capture route float64 bytes drifted")
    values = np.frombuffer(raw, dtype="<f8")
    if values.size != 276 * 7 or not np.all(np.isfinite(values)):
        raise RuntimeError("core capture route has invalid shape or values")
    rows = values.reshape(276, 7)
    result = tuple(
        (
            float(row[0]),
            float(row[1]),
            tuple(float(value) for value in row[2:]),
        )
        for row in rows
    )
    state_records = [
        {"preseat_mm": p_mm, "source_x_mm": x_mm, "q_rad": list(q_rad)}
        for p_mm, x_mm, q_rad in result
    ]
    if (
        _canonical_json_sha256(state_records)
        != CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256
    ):
        raise RuntimeError("core capture route source-state digest drifted")
    if (
        _canonical_json_sha256([list(record[2]) for record in result])
        != CORE_CAPTURE_ROUTE_Q_SHA256
    ):
        raise RuntimeError("core capture route joint digest drifted")
    return result


CORE_CAPTURE_ROUTE_SOURCE_STATES = _decode_core_capture_route_source_states()
CORE_CAPTURE_ROUTE_PHASE_ROW_RANGES = {
    "gripper_capture_axial_open_side": (0, 243),
    "gripper_capture_coupled_recenter": (243, 259),
    "gripper_capture_centered_final": (259, 275),
}
CORE_CAPTURE_ROUTE_PHASE_TIMING_S = {
    "gripper_capture_lateral_align": (0.25, 1.0),
    "gripper_capture_axial_open_side": (1.60, 3.0),
    "gripper_capture_coupled_recenter": (0.50, 1.5),
    "gripper_capture_centered_final": (0.50, 1.5),
}
CORE_CAPTURE_ROUTE_ENDPOINTS_MM = {
    "gripper_capture_lateral_align": (55.0, 0.20),
    "gripper_capture_axial_open_side": (6.4, 0.20),
    "gripper_capture_coupled_recenter": (3.2, 0.0),
    "gripper_capture_centered_final": (0.0, 0.0),
}
CORE_CAPTURE_ROUTE_ACTION_NAMES = frozenset(
    CORE_CAPTURE_ROUTE_PHASE_TIMING_S
)
CORE_CAPTURE_ROUTE_ENDPOINT_Q_ERROR_RAD = 0.002
CORE_CAPTURE_ROUTE_ENDPOINT_QVEL_RAD_S = 0.02
CORE_CAPTURE_ROUTE_ENDPOINT_POSITION_ERROR_M = 0.00005
CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD = math.radians(0.1)
CORE_CAPTURE_ROUTE_ENDPOINT_DWELL_TICKS = 4
CORE_CAPTURE_SOURCE_CORRIDOR_CONTINUOUS_CLEARANCE_MM = 0.249902439
CORE_CAPTURE_SOURCE_CORRIDOR_MANUFACTURING_CLEARANCE_MM = 0.20
CORE_CAPTURE_SOURCE_CORRIDOR_RESERVE_MM = 0.009902439
CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM = 0.040
CORE_CAPTURE_ROUTE_DENSE_FRACTIONS = tuple(
    index / 100.0 for index in range(101)
)
CORE_CAPTURE_ROUTE_SOURCE_X_OPEN_MM = 0.20
CORE_CAPTURE_ROUTE_RECENTER_START_PRESEAT_MM = 6.4
CORE_CAPTURE_ROUTE_RECENTER_END_PRESEAT_MM = 3.2
CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_DIGEST_PREIMAGE = {
    "source_generator_sha256": qc.POGO_CAD_SOURCE_SHA256,
    "positive_lock_cam_contract_sha256": (
        qc.CORE_DOCK_CAM_CONTRACT_CANONICAL_SHA256
    ),
    "embedded_state_bytes_sha256": CORE_CAPTURE_ROUTE_STATE_BYTES_SHA256,
    "source_state_sha256": CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256,
    "q_roster_sha256": CORE_CAPTURE_ROUTE_Q_SHA256,
    "phase_row_ranges": CORE_CAPTURE_ROUTE_PHASE_ROW_RANGES,
    "phase_timing_s": CORE_CAPTURE_ROUTE_PHASE_TIMING_S,
    "endpoint_guard": {
        "q_error_rad": CORE_CAPTURE_ROUTE_ENDPOINT_Q_ERROR_RAD,
        "qvel_rad_s": CORE_CAPTURE_ROUTE_ENDPOINT_QVEL_RAD_S,
        "position_error_m": CORE_CAPTURE_ROUTE_ENDPOINT_POSITION_ERROR_M,
        "orientation_error_rad": (
            CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD
        ),
        "dwell_ticks": CORE_CAPTURE_ROUTE_ENDPOINT_DWELL_TICKS,
    },
    "live_source_corridor_max_error_mm": (
        CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
    ),
}
CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256 = _canonical_json_sha256(
    CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_DIGEST_PREIMAGE
)


def _core_capture_source_x_mm(preseat_mm: float) -> float:
    if not math.isfinite(preseat_mm) or preseat_mm < 0.0:
        raise ValueError("preseat_mm must be finite and nonnegative")
    if preseat_mm >= CORE_CAPTURE_ROUTE_RECENTER_START_PRESEAT_MM:
        return CORE_CAPTURE_ROUTE_SOURCE_X_OPEN_MM
    if preseat_mm <= CORE_CAPTURE_ROUTE_RECENTER_END_PRESEAT_MM:
        return 0.0
    return CORE_CAPTURE_ROUTE_SOURCE_X_OPEN_MM * (
        (preseat_mm - CORE_CAPTURE_ROUTE_RECENTER_END_PRESEAT_MM)
        / (
            CORE_CAPTURE_ROUTE_RECENTER_START_PRESEAT_MM
            - CORE_CAPTURE_ROUTE_RECENTER_END_PRESEAT_MM
        )
    )


CORE_CAM_TAB_LEADING_GEOM_NAME = "qc_col_lock_slider_tab_part_001"
CORE_CAM_TAB_NONCONTACT_GEOM_NAME = "qc_col_lock_slider_tab_part_000"
CORE_CAM_TAB_MAIN_GEOM_NAME = "dock_gripper_cam_collision"
CORE_CAM_TAB_LEAD_GEOM_NAME = "dock_gripper_cam_axial_lead_collision"
CORE_CAM_TAB_HOLD_GEOM_NAME = "dock_gripper_cam_hold_finger_collision"
CORE_CAM_TAB_ROOT_GEOM_NAMES = (
    "dock_gripper_cam_outer_root_lower_collision",
    "dock_gripper_cam_outer_root_upper_collision",
)
CORE_CAM_TAB_CAPTURE_ACTIONS = (
    "gripper_capture_lateral_align",
    "gripper_capture_axial_open_side",
    "gripper_capture_coupled_recenter",
    "gripper_capture_centered_final",
)
CORE_CAM_TAB_FREE_SPACE_ACTIONS = CORE_CAM_TAB_CAPTURE_ACTIONS[:2]
CORE_CAM_TAB_FUNCTIONAL_ACTIONS = CORE_CAM_TAB_CAPTURE_ACTIONS[2:]
CORE_CAM_TAB_POINT_TOLERANCE_MM = 0.020
CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM = 0.020
CORE_CAM_TAB_MIN_NORMAL_ALIGNMENT = 0.999
CORE_CAM_TAB_PASSIVE_OPEN_Q_MM = 0.05000000000000071
CORE_CAM_TAB_RAMP_CONTACT_START_PRESEAT_MM = 6.346666666666667
CORE_CAM_TAB_RAMP_END_PRESEAT_MM = 3.2
CORE_CAM_TAB_LEAD_SEAM_END_PRESEAT_MM = 1.6
CORE_CAM_TAB_MAIN_EDGE_END_PRESEAT_MM = 0.95
CORE_CAM_TAB_LEAD_PLANE_SUM_MM = 17.65
CORE_CAM_TAB_LEAD_NORMAL_CAM_TO_TAB = tuple(
    value / math.sqrt(2.0) for value in (-1.0, 0.0, -1.0)
)
CORE_CAM_TAB_HOLD_NORMAL_CAM_TO_TAB = (-1.0, 0.0, 0.0)
CORE_CAM_TAB_MAIN_SLOPE = 0.246875
CORE_CAM_TAB_MAIN_SLOPE_NORMAL_CAM_TO_TAB = (
    -0.970852159759157,
    -0.239679126940542,
    0.0,
)
CORE_CAM_TAB_MAIN_TOP_NORMAL_CAM_TO_TAB = (0.0, 1.0, 0.0)
CORE_CAM_TAB_PRECONTACT_X_GAP_MM = 0.050
CORE_CAM_TAB_PRECONTACT_NORMAL_GAP_MM = (
    CORE_CAM_TAB_PRECONTACT_X_GAP_MM / math.sqrt(2.0)
)
CORE_CAM_TAB_PRECONTACT_RETAINED_X_GAP_MM = (
    CORE_CAM_TAB_PRECONTACT_X_GAP_MM
    - CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
)
CORE_CAM_TAB_PRECONTACT_RETAINED_NORMAL_GAP_MM = (
    CORE_CAM_TAB_PRECONTACT_RETAINED_X_GAP_MM / math.sqrt(2.0)
)
CORE_CAM_TAB_STALE_POST_CAPTURE_OVERLAP_MM3 = 9.440000000000005


def _core_cam_tab_source_q_max_mm(preseat_mm: float, source_x_mm: float) -> float:
    """Return the source frictionless capture envelope at a live p/X state."""

    if not (
        math.isfinite(preseat_mm)
        and math.isfinite(source_x_mm)
        and preseat_mm >= 0.0
    ):
        raise ValueError("cam capture p/X must be finite and p nonnegative")
    return max(
        CORE_CAM_TAB_PASSIVE_OPEN_Q_MM,
        min(3.0, preseat_mm - source_x_mm - 3.15),
    )


CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_PREIMAGE = {
    "source_generator_sha256": qc.POGO_CAD_SOURCE_SHA256,
    "positive_lock_cam_contract_sha256": (
        qc.CORE_DOCK_CAM_CONTRACT_CANONICAL_SHA256
    ),
    "positive_lock_slider_step_sha256": qc.POSITIVE_LOCK_SLIDER_STEP_SHA256,
    "capture_route_contract_identity_sha256": (
        CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256
    ),
    "leading_tab_geom": CORE_CAM_TAB_LEADING_GEOM_NAME,
    "cam_geoms": list(qc.CORE_DOCK_CAM_COLLISION_GEOM_NAMES),
    "provisional_point_and_penetration_tolerance_mm": (
        CORE_CAM_TAB_POINT_TOLERANCE_MM
    ),
}
CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_SHA256 = _canonical_json_sha256(
    CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_PREIMAGE
)


def core_cam_tab_contact_runtime_contract() -> dict[str, Any]:
    """Publish the exact capture-only cam/tab envelope, fail closed.

    The 20 um point/depth guard is a provisional simulation safety bound.  It
    is not a friction, load, contact-force, dynamics, or release authority.
    """

    cam_runtime_contract = qc.positive_lock_cam_runtime_contract()
    blockers = [
        "provisional_20um_contact_guard_not_physical_contact_authority",
        "positive_lock_cam_friction_coefficient_unqualified",
        "positive_lock_cam_load_capacity_unqualified",
        "positive_lock_cam_dynamics_unqualified",
        "free_space_servo_tracking_not_yet_closed",
        "post_capture_negative_z_slider_return_authority_stale_after_hold_finger_addition",
    ]
    return {
        "schema_version": "1.0",
        "contract_kind": "capture_only_exact_cam_tab_source_envelope",
        "frame": "dock_gripper",
        "contract_identity_digest_preimage": copy.deepcopy(
            CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_PREIMAGE
        ),
        "contract_identity_sha256": (
            CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_SHA256
        ),
        "source_binding": {
            "generator_file": {
                "path": str(qc.POGO_CAD_SOURCE_PATH.relative_to(REPO_ROOT)),
                "bytes": qc.POGO_CAD_SOURCE_BYTES,
                "sha256": qc.POGO_CAD_SOURCE_SHA256,
            },
            "positive_lock_cam_contract_sha256": (
                qc.CORE_DOCK_CAM_CONTRACT_CANONICAL_SHA256
            ),
            "runtime_cam_geometry_contract_sha256": _canonical_json_sha256(
                cam_runtime_contract
            ),
            "positive_lock_slider_step": {
                "path": str(qc.POSITIVE_LOCK_SLIDER_STEP.relative_to(REPO_ROOT)),
                "bytes": qc.POSITIVE_LOCK_SLIDER_STEP.stat().st_size,
                "sha256": qc.POSITIVE_LOCK_SLIDER_STEP_SHA256,
            },
            "capture_route_contract_identity_sha256": (
                CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256
            ),
        },
        "runtime_inventory": {
            "contact_eligible_leading_tab_geom": (
                CORE_CAM_TAB_LEADING_GEOM_NAME
            ),
            "non_contact_tab_geom": CORE_CAM_TAB_NONCONTACT_GEOM_NAME,
            "main_geom": CORE_CAM_TAB_MAIN_GEOM_NAME,
            "axial_lead_geom": CORE_CAM_TAB_LEAD_GEOM_NAME,
            "hold_finger_geom": CORE_CAM_TAB_HOLD_GEOM_NAME,
            "always_forbidden_root_geoms": list(
                CORE_CAM_TAB_ROOT_GEOM_NAMES
            ),
            "all_cam_geoms": list(qc.CORE_DOCK_CAM_COLLISION_GEOM_NAMES),
        },
        "capture_law": {
            "preseat_formula": "-dock_local_robot_mating_z_mm",
            "lateral_formula": "dock_local_robot_mating_x_mm",
            "passive_q_max_formula_mm": "clamp(p-x-3.15,0.05,3.0)",
            "ramp_contact_start_preseat_mm": (
                CORE_CAM_TAB_RAMP_CONTACT_START_PRESEAT_MM
            ),
            "ramp_end_preseat_mm": CORE_CAM_TAB_RAMP_END_PRESEAT_MM,
            "passive_open_q_mm": CORE_CAM_TAB_PASSIVE_OPEN_Q_MM,
            "maximum_live_source_x_error_mm": (
                CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
            ),
        },
        "phase_policy": {
            "no_cam_contact_actions": list(CORE_CAM_TAB_FREE_SPACE_ACTIONS),
            "functional_contact_actions": list(
                CORE_CAM_TAB_FUNCTIONAL_ACTIONS
            ),
            "dock_hold_must_be_active": True,
            "attach_equality_must_be_inactive": True,
            "slider_return_is_excluded": True,
        },
        "source_surfaces": [
            {
                "surface_role": "functional_axial_lead_ramp",
                "action": "gripper_capture_coupled_recenter",
                "runtime_pair": [
                    CORE_CAM_TAB_LEADING_GEOM_NAME,
                    CORE_CAM_TAB_LEAD_GEOM_NAME,
                ],
                "preseat_bounds_mm": [
                    CORE_CAM_TAB_RAMP_END_PRESEAT_MM,
                    CORE_CAM_TAB_RAMP_CONTACT_START_PRESEAT_MM,
                ],
                "locus": {
                    "plane": "x+z=17.65mm",
                    "y_bounds_mm": [0.0, 2.0],
                    "z_bounds_mm": [-9.6, -6.4],
                },
                "normal_cam_to_tab_dock_local": list(
                    CORE_CAM_TAB_LEAD_NORMAL_CAM_TO_TAB
                ),
                "functional_coverage_required": True,
            },
            {
                "surface_role": "functional_hold_finger_face",
                "action": "gripper_capture_centered_final",
                "runtime_pair": [
                    CORE_CAM_TAB_LEADING_GEOM_NAME,
                    CORE_CAM_TAB_HOLD_GEOM_NAME,
                ],
                "preseat_bounds_mm": [0.0, 3.2],
                "locus": {
                    "plane": "x=24.05mm",
                    "y_bounds_mm": [0.0, 2.0],
                    "z_bounds_mm": [-6.4, -4.15],
                    "tab_z_bounds_formula_mm": ["-4.8-p", "-3.2-p"],
                    "accepted_z_is_interval_intersection": True,
                },
                "normal_cam_to_tab_dock_local": list(
                    CORE_CAM_TAB_HOLD_NORMAL_CAM_TO_TAB
                ),
                "functional_coverage_required": True,
            },
            {
                "surface_role": "lead_hold_partition_seam_nonfunctional",
                "action": "gripper_capture_centered_final",
                "runtime_pair": [
                    CORE_CAM_TAB_LEADING_GEOM_NAME,
                    CORE_CAM_TAB_LEAD_GEOM_NAME,
                ],
                "preseat_bounds_mm": [1.6, 3.2],
                "locus": {
                    "line": "x=24.05mm,z=-6.4mm",
                    "y_bounds_mm": [0.0, 2.0],
                },
                "normal_cone_cam_to_tab": [
                    list(CORE_CAM_TAB_LEAD_NORMAL_CAM_TO_TAB),
                    list(CORE_CAM_TAB_HOLD_NORMAL_CAM_TO_TAB),
                ],
                "closed_top_cap_positive_z_is_forbidden": True,
                "functional_coverage_required": False,
            },
            {
                "surface_role": "main_hold_edge_tangency_nonfunctional",
                "action": "gripper_capture_centered_final",
                "runtime_pair": [
                    CORE_CAM_TAB_LEADING_GEOM_NAME,
                    CORE_CAM_TAB_MAIN_GEOM_NAME,
                ],
                "preseat_bounds_mm": [0.0, 0.95],
                "locus": {
                    "line": "x=24.05mm,y=0mm",
                    "z_lower_mm": -4.15,
                    "z_upper_formula_mm": "-3.2-p",
                },
                "normal_cone_cam_to_tab": [
                    list(CORE_CAM_TAB_MAIN_SLOPE_NORMAL_CAM_TO_TAB),
                    list(CORE_CAM_TAB_MAIN_TOP_NORMAL_CAM_TO_TAB),
                ],
                "functional_coverage_required": False,
            },
        ],
        "provisional_guard": {
            "point_tolerance_mm": CORE_CAM_TAB_POINT_TOLERANCE_MM,
            "maximum_penetration_mm": (
                CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
            ),
            "minimum_normal_alignment": CORE_CAM_TAB_MIN_NORMAL_ALIGNMENT,
            "lead_maximum_q_excess_mm": math.sqrt(2.0)
            * CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM,
            "hold_and_main_maximum_q_mm": (
                CORE_CAM_TAB_PASSIVE_OPEN_Q_MM
                + CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
            ),
            "authority": "provisional_simulation_guard_only",
        },
        "free_space_endpoint_clearance": {
            "axial_action_endpoint_preseat_mm": 6.4,
            "x_axis_gap_mm": CORE_CAM_TAB_PRECONTACT_X_GAP_MM,
            "lead_normal_gap_mm": CORE_CAM_TAB_PRECONTACT_NORMAL_GAP_MM,
            "retained_x_gap_at_40um_corridor_mm": (
                CORE_CAM_TAB_PRECONTACT_RETAINED_X_GAP_MM
            ),
            "retained_lead_normal_gap_at_40um_corridor_mm": (
                CORE_CAM_TAB_PRECONTACT_RETAINED_NORMAL_GAP_MM
            ),
        },
        "post_capture_exclusion": {
            "excluded_actions": [
                "gripper_lock_cam_disengagement",
                "gripper_slider_return_verify",
                "gripper_physical_lock_confirm",
            ],
            "reason": (
                "negative_z_slider_return_clearance_witness_is_stale_after_"
                "hold_finger_addition"
            ),
            "q3_negative_z_1p2mm_complete_cam_overlap_mm3": (
                CORE_CAM_TAB_STALE_POST_CAPTURE_OVERLAP_MM3
            ),
            "legacy_main_wedge_only_clearance_mm": 0.25,
            "must_be_reaudited_before_authorization": True,
            "retired_default_action_kinds": [
                "axial_disengage",
                "slider_return",
                "physical_lock_confirm",
            ],
            "custom_injection_abort_reason": (
                "retired_negative_z_lock_sequence"
            ),
            "default_action_sequence_ends_at": (
                "gripper_dock_release_verify"
            ),
        },
        "evidence_requirements": {
            "audit_frequency": "after_every_mj_step_before_generic_contact_audit",
            "raw_contact_force_torque_width": 6,
            "zero_contact_cannot_pass_functional_coverage": True,
            "functional_roles_required": [
                "functional_axial_lead_ramp",
                "functional_hold_finger_face",
            ],
            "all_candidate_contacts_remain_counted": True,
        },
        "authority_scope": {
            "static_geometry_phase_and_locus_authority": True,
            "provisional_contact_classification_authority": False,
            "friction_coefficient_authority": False,
            "load_capacity_authority": False,
            "contact_force_authority": False,
            "dynamics_authority": False,
            "post_capture_release_authority": False,
            "blockers": blockers,
            "release_ready": False,
        },
        "passed": True,
        "release_ready": False,
    }
ALIGNED_CAPTURE_STATIC_MAX_LATERAL_DEVIATION_M = 0.000205
CAM_RELIEF_CORRIDOR_M = 0.0005
CORE_KEEPER_MAX_PENETRATION_MM = 0.020
CORE_KEEPER_MAX_SEPARATION_MM = 0.020
CORE_KEEPER_MIN_NORMAL_ALIGNMENT = 0.999
CORE_LOCK_RELEASE_AXIS_DOCK_LOCAL = (0.0, 0.0, -1.0)
CORE_LOCK_RELEASE_SOURCE_AXIS = "dock_local_negative_z"
CORE_LOCK_RELEASE_STROKE_MM = 1.20
CORE_LOCK_RELEASE_MIN_STROKE_MM = 1.15
MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM = CORE_LOCK_RELEASE_MIN_STROKE_MM
MAXIMUM_SOURCE_AXIS_WITHDRAWAL_MM = CORE_LOCK_RELEASE_STROKE_MM
LOCKED_SLIDER_POSITION_BAND_M = (0.00295, 0.00305)
LOCKED_SLIDER_SPEED_LIMIT_M_S = 0.001
LOCKED_SLIDER_SETTLED_DWELL_S = 0.050
LOCK_CAM_MANUFACTURING_CLEARANCE_MM = 0.20

# The exact target is the 19.982448% interpolation between the frozen 1 mm and
# 2 mm +X=0.20 mm guided IK roots.  FK places the robot mating site at
# [0.200817, -0.009784, -1.200000] mm in the dock frame, preserving the
# mating orientation to 3 microradians while remaining just inside the
# audited 1.20 mm endpoint.
CORE_LOCK_DISENGAGEMENT_TARGET_Q = (
    -0.72,
    -0.5120448336845652,
    0.8077281888667815,
    -0.2956833551822162,
    0.0,
)

CORE_KEEPER_CONTACT_CONTRACT = (
    {
        "source_pair": ["stock_tool_plate", "left_lower_rail"],
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
    {
        "source_pair": ["stock_tool_plate", "left_upper_rail"],
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
    {
        "source_pair": ["stock_tool_plate", "right_lower_rail"],
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
    {
        "source_pair": ["stock_tool_plate", "right_upper_rail"],
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
    {
        "source_pair": ["robot_plate", "left_lower_rail"],
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
)

CORE_POSITIVE_LOCK_CONTRACT = {
    "slider_body": "qc_positive_lock_slider",
    "slider_body_parent": "robot_plate_frame",
    "slider_body_pos_at_joint_reference_m": list(
        qc.POSITIVE_LOCK_SLIDER_BASE_POS_M
    ),
    "slider_joint": "qc_positive_lock_slider_joint",
    "slider_joint_axis": [1.0, 0.0, 0.0],
    "slider_joint_range_m": list(qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M),
    "slider_joint_reference_m": qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M[1],
    "slider_spring_reference_m": qc.POSITIVE_LOCK_SLIDER_SPRINGREF_M,
    "slider_stiffness_n_m": qc.POSITIVE_LOCK_SLIDER_STIFFNESS_N_M,
    "slider_material": "304 stainless steel",
    "slider_source_density_kg_m3": qc.POSITIVE_LOCK_SLIDER_DENSITY_KG_M3,
    "slider_source_volume_mm3": qc.POSITIVE_LOCK_SLIDER_SOURCE_VOLUME_MM3,
    "slider_source_mass_kg": qc.POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG,
    "slider_source_com_m": list(qc.POSITIVE_LOCK_SLIDER_SOURCE_COM_M),
    "slider_source_full_inertia_kg_m2": list(
        qc.POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2
    ),
    "collision_prisms_contribute_inertia": False,
    "lock_hardware_material": "stainless steel",
    "lock_hardware_density_kg_m3": qc.POSITIVE_LOCK_HARDWARE_DENSITY_KG_M3,
    "lock_hardware_rigid_body_pattern": "tool_<tool>_positive_lock_hardware",
    "lock_hardware_body_parent_pattern": "tool_<tool>",
    "lock_hardware_body_pos_m": [0.0, 0.0, 0.0],
    "lock_hardware_body_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "lock_hardware_has_joint": False,
    "lock_hardware_collision_geoms_contribute_inertia": False,
    "lock_hardware_source_volume_mm3": (
        qc.POSITIVE_LOCK_HARDWARE_SOURCE_VOLUME_MM3
    ),
    "lock_hardware_source_mass_kg": qc.POSITIVE_LOCK_HARDWARE_SOURCE_MASS_KG,
    "lock_hardware_source_com_m": list(qc.POSITIVE_LOCK_HARDWARE_SOURCE_COM_M),
    "lock_hardware_source_full_inertia_kg_m2": list(
        qc.POSITIVE_LOCK_HARDWARE_SOURCE_FULL_INERTIA_KG_M2
    ),
    "lock_hardware_source_components": [
        {
            "component_id": "shoulder_screw_pair",
            "source_artifact": (
                "QuickChange/SO101_Magnetic/exports/"
                "hardware_McMaster_90318A720_shoulder_screw.step"
            ),
            "source_artifact_sha256": (
                qc.POSITIVE_LOCK_SHOULDER_SCREW_STEP_SHA256
            ),
            "placements_tool_root_m": [
                {"pos": [x_value, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
                for x_value in qc.POSITIVE_LOCK_HARDWARE_STACK_X_M
            ],
        },
        {
            "component_id": "holed_din934_m3_nut_pair",
            "source_artifact": (
                "QuickChange/SO101_Magnetic/exports/"
                "hardware_DIN934_M3_lock_stud_nut.step"
            ),
            "source_artifact_sha256": qc.POSITIVE_LOCK_STUD_NUT_STEP_SHA256,
            "placements_tool_root_m": [
                {
                    "pos": [
                        x_value,
                        qc.POSITIVE_LOCK_HARDWARE_NUT_TRANSLATION_M[1],
                        qc.POSITIVE_LOCK_HARDWARE_NUT_TRANSLATION_M[2],
                    ],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
                for x_value in qc.POSITIVE_LOCK_HARDWARE_STACK_X_M
            ],
        },
    ],
    "slider_damping_n_s_m": qc.POSITIVE_LOCK_SLIDER_DAMPING_N_S_M,
    "damping_authority": "critical_damping_2_sqrt_exact_step_mass_times_spring_k",
    "slider_frictionloss_n": 0.0,
    "slider_armature_kg": 0.0,
    "slider_limit_solref": list(qc.POSITIVE_LOCK_SLIDER_LIMIT_SOLREF),
    "slider_limit_solimp": list(qc.POSITIVE_LOCK_SLIDER_LIMIT_SOLIMP),
    "dynamic_settling_release_ready": False,
    "source_artifact": (
        "QuickChange/SO101_Magnetic/exports/so101_positive_lock_slider.step"
    ),
    "source_artifact_sha256": qc.POSITIVE_LOCK_SLIDER_STEP_SHA256,
    "source_absolute_mesh_requested_deflection_mm": (
        qc.POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM
    ),
    "source_angular_deflection_rad": (
        qc.POSITIVE_LOCK_SLIDER_ANGULAR_DEFLECTION_RAD
    ),
    "conservative_void_expansion_mm": (
        qc.POSITIVE_LOCK_SLIDER_VOID_EXPANSION_MM
    ),
    "slider_collision_exact_source_subset": True,
    "slider_collision_continuous_clearance_authority": False,
    "source_prism_z_bounds_m": list(qc.POSITIVE_LOCK_SLIDER_Z_BOUNDS_M),
    "active_geom_prefixes": [
        "qc_col_lock_slider_bridge_part_",
        "qc_col_lock_slider_left_lobe_part_",
        "qc_col_lock_slider_right_lobe_part_",
        "qc_col_lock_slider_tab_part_",
    ],
    "tool_lock_features": {
        "centres_x_m": [-0.012, 0.012],
        "centre_y_m": 0.0,
        "shoulder_diameter_m": 0.004,
        "head_diameter_m": 0.006,
        "tools": ["gripper", "spoon", "whisk"],
    },
    "state_semantics": {
        "unlocked_q_m": 0.0,
        "locked_q_m": 0.003,
        "unlocked_source_translation_x_m": 0.0,
        "locked_source_translation_x_m": 0.003,
    },
    "dynamic_negative_api": None,
    "broad_plate_proxy_contract": {
        "role": "approximate dynamics broadphase",
        "exact_named_critical_lock_voids": True,
        "exact_source_subset": False,
        "continuous_clearance": False,
        "release_ready": False,
    },
    "release_ready": False,
}


def _aligned_capture_waypoints(
    tool: str, *, reverse: bool = False
) -> tuple[tuple[float, ...], ...]:
    """Return frozen FK/IK waypoints for normal approach or withdrawal.

    The core gripper follows the source cam's coupled dock-local p/X law and
    has no same-Z seated recenter row.  The controller executes this roster as
    three separately time-scaled phases so both source-law breakpoints have
    zero commanded endpoint velocity.
    """

    if tool not in DOCK_CAPTURE_Q:
        raise ValueError(f"unsupported aligned capture tool {tool!r}")
    if tool == "gripper":
        forward = tuple(record[2] for record in CORE_CAPTURE_ROUTE_SOURCE_STATES)
    else:
        pan = float(DOCK_CAPTURE_Q[tool][0])
        forward = tuple(
            (pan, lift, elbow, wrist_flex, 0.0)
            for lift, elbow, wrist_flex in ALIGNED_CAPTURE_BASE_Q
        )
    if not reverse:
        return forward
    # The caller starts at the seated row, so omit it and finish at the exact
    # 55 mm pre-capture datum.
    return tuple(reversed((tuple(DOCK_PRE_CAPTURE_Q[tool]), *forward[:-1])))


def _core_lock_disengagement_waypoints() -> tuple[tuple[float, ...], ...]:
    """Return the audited seat-to-1.20 mm dock-local -Z lock stroke."""

    pan = float(DOCK_CAPTURE_Q["gripper"][0])
    seated_open_side = (
        pan,
        *CORE_GUIDED_CAPTURE_BASE_Q[-1],
        0.0,
    )
    one_mm = (
        pan,
        *CORE_GUIDED_CAPTURE_BASE_Q[-2],
        0.0,
    )
    return (
        tuple(float(value) for value in seated_open_side),
        tuple(float(value) for value in one_mm),
        CORE_LOCK_DISENGAGEMENT_TARGET_Q,
    )

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
    joint_waypoints: tuple[tuple[float, ...], ...] = ()

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
        if any(len(waypoint) != len(ARM_JOINTS) for waypoint in self.joint_waypoints):
            raise ValueError(f"wrong waypoint width for {self.name}")
        if self.joint_waypoints and self.target_q is None:
            raise ValueError(f"waypoint route needs a target for {self.name}")
        if self.joint_waypoints and self.joint_waypoints[-1] != self.target_q:
            raise ValueError(f"last waypoint must equal target for {self.name}")


def _core_capture_move_actions() -> tuple[WorkflowAction, ...]:
    """Return four finite, zero-endpoint-velocity source-route actions."""

    q_rows = tuple(record[2] for record in CORE_CAPTURE_ROUTE_SOURCE_STATES)
    align_duration, align_timeout = CORE_CAPTURE_ROUTE_PHASE_TIMING_S[
        "gripper_capture_lateral_align"
    ]
    axial_duration, axial_timeout = CORE_CAPTURE_ROUTE_PHASE_TIMING_S[
        "gripper_capture_axial_open_side"
    ]
    recenter_duration, recenter_timeout = CORE_CAPTURE_ROUTE_PHASE_TIMING_S[
        "gripper_capture_coupled_recenter"
    ]
    final_duration, final_timeout = CORE_CAPTURE_ROUTE_PHASE_TIMING_S[
        "gripper_capture_centered_final"
    ]
    return (
        WorkflowAction(
            name="gripper_capture_lateral_align",
            kind="move",
            tool="gripper",
            target_q=q_rows[0],
            joint_waypoints=(q_rows[0],),
            duration_s=align_duration,
            timeout_s=align_timeout,
        ),
        WorkflowAction(
            name="gripper_capture_axial_open_side",
            kind="move",
            tool="gripper",
            target_q=q_rows[243],
            joint_waypoints=q_rows[1:244],
            duration_s=axial_duration,
            timeout_s=axial_timeout,
        ),
        WorkflowAction(
            name="gripper_capture_coupled_recenter",
            kind="move",
            tool="gripper",
            target_q=q_rows[259],
            joint_waypoints=q_rows[244:260],
            duration_s=recenter_duration,
            timeout_s=recenter_timeout,
        ),
        WorkflowAction(
            name="gripper_capture_centered_final",
            kind="move",
            tool="gripper",
            target_q=q_rows[275],
            joint_waypoints=q_rows[260:276],
            duration_s=final_duration,
            timeout_s=final_timeout,
        ),
    )


def _recovery_controller_actions(
    tool: str = "gripper", *, include_rack_exit: bool = False
) -> tuple[WorkflowAction, ...]:
    if tool not in ALL_TOOL_IDS:
        raise ValueError(f"unsupported recovery tool {tool}")
    if tool == "gripper":
        capture_moves = _core_capture_move_actions()
    else:
        capture_moves = (
            WorkflowAction(
                name=f"{tool}_to_capture",
                kind="move",
                tool=tool,
                target_q=tuple(float(value) for value in DOCK_CAPTURE_Q[tool]),
                joint_waypoints=_aligned_capture_waypoints(tool),
                duration_s=1.5,
                timeout_s=3.5,
            ),
        )
    capture_release = (
        *capture_moves,
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
    if tool != "gripper":
        return capture_release
    # Full rack removal remains a separately audited phase-B milestone.
    # ``include_rack_exit`` is retained as a fail-closed compatibility flag;
    # it does not append the retired unverified 55 mm route.
    if include_rack_exit:
        raise ValueError("full rack exit is not yet a validated controller action")
    # The former -Z disengagement, slider-return and physical-lock suffix is
    # deliberately absent.  The complete authored cam's hold finger overlaps
    # the q=3 slider by 9.44 mm^3 after that 1.2 mm motion; the legacy 0.25 mm
    # scalar measured only the main wedge.  Default production stops honestly
    # after attached dock release with the physical slider still unlocked.
    return capture_release


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
    asset = robot_root.find("asset")
    if asset is None:
        raise RuntimeError("Calibrated robot no longer has an asset container")
    qc.add_robot_quick_change_interface(original, asset)
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


def _parallel_axis_term(mass_kg: float, com_m: np.ndarray) -> np.ndarray:
    return mass_kg * (
        float(com_m @ com_m) * np.eye(3, dtype=np.float64)
        - np.outer(com_m, com_m)
    )


def _validated_mass_properties(
    *, mass_kg: float, com_m: Any, inertia_about_com_kg_m2: Any, label: str
) -> dict[str, Any]:
    com = np.asarray(com_m, dtype=np.float64)
    inertia = np.asarray(inertia_about_com_kg_m2, dtype=np.float64)
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise RuntimeError(f"{label} has invalid mass {mass_kg}")
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise RuntimeError(f"{label} has invalid COM {com}")
    if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise RuntimeError(f"{label} has invalid inertia tensor")
    if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1.0e-15):
        raise RuntimeError(f"{label} inertia tensor is not symmetric")
    inertia = 0.5 * (inertia + inertia.T)
    if float(np.min(np.linalg.eigvalsh(inertia))) <= 0.0:
        raise RuntimeError(f"{label} inertia tensor is not positive definite")
    return {
        "mass_kg": float(mass_kg),
        "com_tool_m": com,
        "inertia_about_com_tool_kg_m2": inertia,
    }


def _aggregate_ledger_components(
    component_records: list[dict[str, Any]], *, label: str
) -> dict[str, Any]:
    positive_records = [
        record for record in component_records if float(record["mass_kg"]) > 0.0
    ]
    if not positive_records:
        raise RuntimeError(f"{label} has no positive-mass ledger components")
    mass_kg = math.fsum(float(record["mass_kg"]) for record in positive_records)
    moment = np.sum(
        [
            float(record["mass_kg"])
            * (0.001 * np.asarray(record["com_mm"], dtype=np.float64))
            for record in positive_records
        ],
        axis=0,
    )
    com_m = moment / mass_kg
    inertia_about_origin = np.zeros((3, 3), dtype=np.float64)
    for record in positive_records:
        component_mass = float(record["mass_kg"])
        component_com = 0.001 * np.asarray(record["com_mm"], dtype=np.float64)
        component_inertia = np.asarray(
            record["inertia_about_com_kg_m2"], dtype=np.float64
        )
        inertia_about_origin += component_inertia + _parallel_axis_term(
            component_mass, component_com
        )
    inertia_about_com = inertia_about_origin - _parallel_axis_term(mass_kg, com_m)
    result = _validated_mass_properties(
        mass_kg=mass_kg,
        com_m=com_m,
        inertia_about_com_kg_m2=inertia_about_com,
        label=label,
    )
    result["component_ids"] = tuple(
        str(record["name"]) for record in component_records
    )
    return result


def _subtract_mass_properties(
    total: dict[str, Any],
    children: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    residual_mass = float(total["mass_kg"])
    residual_moment = residual_mass * np.asarray(total["com_tool_m"])
    residual_inertia_origin = np.asarray(
        total["inertia_about_com_tool_kg_m2"], dtype=np.float64
    ) + _parallel_axis_term(residual_mass, np.asarray(total["com_tool_m"]))
    for child in children:
        child_mass = float(child["mass_kg"])
        child_com = np.asarray(child["com_tool_m"], dtype=np.float64)
        residual_mass -= child_mass
        residual_moment -= child_mass * child_com
        residual_inertia_origin -= np.asarray(
            child["inertia_about_com_tool_kg_m2"], dtype=np.float64
        ) + _parallel_axis_term(child_mass, child_com)
    if residual_mass <= 0.0:
        raise RuntimeError(f"{label} residual mass is not positive")
    residual_com = residual_moment / residual_mass
    residual_inertia = residual_inertia_origin - _parallel_axis_term(
        residual_mass, residual_com
    )
    return _validated_mass_properties(
        mass_kg=residual_mass,
        com_m=residual_com,
        inertia_about_com_kg_m2=residual_inertia,
        label=label,
    )


@cache
def _payload_mass_inertia_authority(tool: str) -> dict[str, Any]:
    """Load one pinned ledger and partition its inertia across runtime bodies."""

    if tool not in PAYLOAD_MASS_LEDGER_PATHS:
        raise ValueError(f"no matcha payload ledger for {tool!r}")
    ledger_path = PAYLOAD_MASS_LEDGER_PATHS[tool]
    source_bytes = ledger_path.read_bytes()
    observed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    expected_sha256 = PAYLOAD_MASS_LEDGER_SHA256[tool]
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{tool} mass-ledger hash mismatch: expected {expected_sha256}, "
            f"got {observed_sha256}"
        )
    ledger = json.loads(source_bytes)
    if ledger.get("tool") != tool or ledger.get("schema_version") != "1.0":
        raise RuntimeError(f"unexpected {tool} mass-ledger identity")
    component_records = list(ledger.get("components", []))
    records_by_name = {str(record["name"]): record for record in component_records}
    if len(records_by_name) != len(component_records):
        raise RuntimeError(f"{tool} mass ledger has duplicate component names")
    missing_hardware = sorted(
        set(LEDGER_LOCK_HARDWARE_COMPONENT_IDS) - records_by_name.keys()
    )
    if missing_hardware:
        raise RuntimeError(f"{tool} ledger omits lock hardware: {missing_hardware}")

    total = _validated_mass_properties(
        mass_kg=float(ledger["total_mass_kg"]),
        com_m=0.001 * np.asarray(ledger["com_mm"], dtype=np.float64),
        inertia_about_com_kg_m2=ledger["inertia_about_com_kg_m2"],
        label=f"{tool} total mass ledger",
    )
    hardware = _validated_mass_properties(
        mass_kg=qc.POSITIVE_LOCK_HARDWARE_SOURCE_MASS_KG,
        com_m=qc.POSITIVE_LOCK_HARDWARE_SOURCE_COM_M,
        inertia_about_com_kg_m2=np.asarray(
            [
                [qc.POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2[0], 0.0, 0.0],
                [0.0, qc.POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2[1], 0.0],
                [0.0, 0.0, qc.POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2[2]],
            ],
            dtype=np.float64,
        ),
        label=f"{tool} exact lock hardware",
    )
    moving: dict[str, dict[str, Any]] = {}
    if tool == "whisk":
        for body_role, component_ids in (
            ("rotor", WHISK_ROTOR_LEDGER_COMPONENT_IDS),
            ("carriage", WHISK_CARRIAGE_LEDGER_COMPONENT_IDS),
        ):
            missing = sorted(set(component_ids) - records_by_name.keys())
            if missing:
                raise RuntimeError(
                    f"whisk ledger omits {body_role} components: {missing}"
                )
            moving[body_role] = _aggregate_ledger_components(
                [records_by_name[name] for name in component_ids],
                label=f"whisk {body_role} ledger partition",
            )
    root = _subtract_mass_properties(
        total,
        [hardware, *moving.values()],
        label=f"{tool} rigid-root ledger residual",
    )

    # Recombine the declared runtime bodies and require exact target closure.
    declared = [root, hardware, *moving.values()]
    combined_mass = math.fsum(float(record["mass_kg"]) for record in declared)
    combined_com = np.sum(
        [float(record["mass_kg"]) * record["com_tool_m"] for record in declared],
        axis=0,
    ) / combined_mass
    combined_inertia_origin = np.sum(
        [
            record["inertia_about_com_tool_kg_m2"]
            + _parallel_axis_term(float(record["mass_kg"]), record["com_tool_m"])
            for record in declared
        ],
        axis=0,
    )
    combined_inertia = combined_inertia_origin - _parallel_axis_term(
        combined_mass, combined_com
    )
    if abs(combined_mass - float(total["mass_kg"])) > 1.0e-15:
        raise RuntimeError(f"{tool} runtime mass partition does not close")
    if not np.allclose(
        combined_com, total["com_tool_m"], rtol=0.0, atol=1.0e-15
    ) or not np.allclose(
        combined_inertia,
        total["inertia_about_com_tool_kg_m2"],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError(f"{tool} runtime inertia partition does not close")
    return {
        "ledger_path": ledger_path,
        "ledger_sha256": expected_sha256,
        "total": total,
        "root": root,
        "hardware": hardware,
        **moving,
    }


def payload_mass_inertia_contract(tool: str) -> dict[str, Any]:
    """Return a JSON-safe public description of one runtime inertia split."""

    authority = _payload_mass_inertia_authority(tool)
    result: dict[str, Any] = {
        "tool": tool,
        "source_ledger": str(authority["ledger_path"].relative_to(REPO_ROOT)),
        "source_ledger_sha256": authority["ledger_sha256"],
        "collision_geoms_contribute_inertia": False,
        "ledger_lock_hardware_component_ids_replaced_by_exact_source_child": list(
            LEDGER_LOCK_HARDWARE_COMPONENT_IDS
        ),
        "runtime_bodies": {},
    }
    role_to_body = {
        "root": f"tool_{tool}",
        "hardware": f"tool_{tool}_positive_lock_hardware",
        "rotor": "whisk_eccentric_rotor",
        "carriage": "whisk_compliance_carriage",
    }
    for role in ("root", "hardware", "rotor", "carriage"):
        if role not in authority:
            continue
        properties = authority[role]
        result["runtime_bodies"][role] = {
            "body": role_to_body[role],
            "mass_kg": float(properties["mass_kg"]),
            "com_tool_m": np.asarray(properties["com_tool_m"]).tolist(),
            "inertia_about_com_tool_kg_m2": np.asarray(
                properties["inertia_about_com_tool_kg_m2"]
            ).tolist(),
            "component_ids": list(properties.get("component_ids", ())),
        }
    return result


def _append_explicit_inertial(
    body: ET.Element,
    properties: dict[str, Any],
    *,
    body_origin_tool_m: tuple[float, float, float],
) -> None:
    if body.find("./inertial") is not None:
        raise RuntimeError(f"body {body.get('name')} already has an inertial")
    com_local = np.asarray(properties["com_tool_m"], dtype=np.float64) - np.asarray(
        body_origin_tool_m, dtype=np.float64
    )
    inertia = np.asarray(
        properties["inertia_about_com_tool_kg_m2"], dtype=np.float64
    )
    full_inertia = (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    )
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": " ".join(f"{value:.17g}" for value in com_local),
            "mass": f"{float(properties['mass_kg']):.17g}",
            "fullinertia": " ".join(f"{value:.17g}" for value in full_inertia),
        },
    )


def _make_descendant_geoms_massless(body: ET.Element) -> None:
    for geom in body.iter("geom"):
        geom.attrib.pop("density", None)
        geom.set("mass", "0")


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


def _add_whisk_payload(
    tool: ET.Element,
    actuator: ET.Element,
    mass_authority: dict[str, Any],
) -> None:
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
    _append_explicit_inertial(
        rotor,
        mass_authority["rotor"],
        body_origin_tool_m=(0.0, 0.0, 0.052),
    )
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
    _append_explicit_inertial(
        carriage,
        mass_authority["carriage"],
        body_origin_tool_m=(0.0, 0.0, 0.060),
    )
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
    asset: ET.Element,
    actuator: ET.Element,
    tool_name: str,
    stock_gripper: ET.Element | None,
) -> ET.Element:
    position, quat = DOCK_POSES[tool_name]
    mass_authority = (
        _payload_mass_inertia_authority(tool_name)
        if tool_name in PAYLOAD_MASS_LEDGER_PATHS
        else None
    )
    tool = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"tool_{tool_name}",
            "pos": " ".join(f"{value:.12g}" for value in position),
            "quat": " ".join(f"{value:.12g}" for value in quat),
        },
    )
    if mass_authority is not None:
        _append_explicit_inertial(
            tool,
            mass_authority["root"],
            body_origin_tool_m=(0.0, 0.0, 0.0),
        )
    ET.SubElement(tool, "freejoint", {"name": f"tool_{tool_name}_free"})
    qc.add_tool_quick_change_interface(tool, asset, tool_name)
    if tool_name == "gripper":
        if stock_gripper is None:
            raise RuntimeError("Stock gripper subtree was not supplied")
        tool.append(stock_gripper)
    elif tool_name == "spoon":
        _add_spoon_payload(tool)
    elif tool_name == "whisk":
        if mass_authority is None:
            raise RuntimeError("whisk payload mass authority is absent")
        _add_whisk_payload(tool, actuator, mass_authority)
    else:
        raise RuntimeError(f"Unsupported tool {tool_name}")
    ET.SubElement(
        tool,
        "site",
        {"name": f"{tool_name}_tool_id_site", "pos": "-0.031 0 0", "size": "0.001"},
    )
    if mass_authority is not None:
        _make_descendant_geoms_massless(tool)
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
                    "geom1": f"qc_col_pogo_{signal}_plunger",
                    "geom2": f"{tool}_pad_{signal}_collision",
                    "condim": "1",
                    "margin": "0",
                    "gap": "0",
                    "solref": "0.0005 1",
                    "solimp": "0.99 0.9999 0.00001",
                },
            )
    # The robot electrical wing runs along the core dock's left-lower keeper
    # during the final source-axis approach.  This is a normal-only guide
    # contact: tangential Coulomb friction would turn the exact source datum
    # into an artificial axial brake.  The second pair hardens the exact
    # robot/tool mating-plane witness at the seated pose.
    for name, geom1, geom2, detection_margin in (
        (
            "core_robot_left_lower_keeper_guide_pair",
            "qc_col_robot_plate_electrical_wing_edge__keeper_land",
            "dock_gripper_keeper_left_lower_collision",
            0.0,
        ),
        (
            "core_robot_tool_wing_mating_pair",
            "qc_col_robot_plate_electrical_wing_edge__keeper_land",
            "matcha_col_gripper_plate_electrical_wing_edge__mating_land__keeper_land",
            0.0,
        ),
        (
            "core_tool_right_upper_keeper_pair",
            "matcha_col_gripper_plate_xpos__mating_land__locator_land__dock_stop_land",
            "dock_gripper_keeper_right_upper_collision",
            0.00002,
        ),
    ):
        ET.SubElement(
            contact,
            "pair",
            {
                "name": name,
                "geom1": geom1,
                "geom2": geom2,
                "condim": "1",
                "margin": f"{detection_margin:.9g}",
                "gap": f"{detection_margin:.9g}",
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
            asset,
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


def isolated_positive_lock_return(
    *,
    spring_enabled: bool = True,
    pin_equality_active: bool = False,
    max_time_s: float = 0.5,
) -> dict[str, Any]:
    """Exercise only the physical slider spring and declared upper limit.

    The unlocked initial state is loaded with ``mj_resetDataKeyframe`` before
    observation.  Every subsequent state transition is a real ``mj_step``;
    no Python write to qpos/qvel occurs after initialization.  Negative hooks
    can remove the spring or pin q=0 without changing the acceptance logic.
    """

    if not math.isfinite(max_time_s) or not 0.05 <= max_time_s <= 5.0:
        raise ValueError("max_time_s must be finite and in [0.05, 5.0]")
    timestep_s = 0.00025
    stiffness_n_m = (
        qc.POSITIVE_LOCK_SLIDER_STIFFNESS_N_M if spring_enabled else 0.0
    )
    # A one-joint equality pins that joint to its compiled reference state.
    # The negative-only model therefore uses qref=0 so the equality genuinely
    # holds the unlocked coordinate; the nominal model retains production's
    # qref=3 mm semantics.
    isolated_joint_reference_m = (
        0.0 if pin_equality_active else qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M[1]
    )
    equality_xml = (
        '<equality><joint name="pin_unlocked" joint1="slider" '
        'polycoef="0 0 0 0 0" active="true" solref="0.00001 1" '
        'solimp="0.9999 0.99999 0.000001 0.5 2"/></equality>'
        if pin_equality_active
        else "<equality/>"
    )
    full_inertia = " ".join(
        f"{value:.17g}"
        for value in qc.POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2
    )
    xml = f"""
<mujoco model="isolated positive lock return">
  <compiler autolimits="true"/>
  <option timestep="{timestep_s:.12g}" integrator="implicitfast"
          gravity="0 0 0" iterations="80" ls_iterations="20"/>
  <worldbody>
    <body name="slider_body" pos="{isolated_joint_reference_m:.17g} 0 0">
      <inertial pos="{qc.POSITIVE_LOCK_SLIDER_SOURCE_COM_M[0]:.17g} 0
                     {qc.POSITIVE_LOCK_SLIDER_SOURCE_COM_M[2]:.17g}"
                mass="{qc.POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG:.17g}"
                fullinertia="{full_inertia}"/>
      <joint name="slider" type="slide" axis="1 0 0" limited="true"
             range="0 0.003" ref="{isolated_joint_reference_m:.17g}"
             stiffness="{stiffness_n_m:.17g}"
             springref="{qc.POSITIVE_LOCK_SLIDER_SPRINGREF_M:.17g}"
             damping="{qc.POSITIVE_LOCK_SLIDER_DAMPING_N_S_M:.17g}"
             frictionloss="0" armature="0"
             solreflimit="{' '.join(f'{value:.12g}' for value in qc.POSITIVE_LOCK_SLIDER_LIMIT_SOLREF)}"
             solimplimit="{' '.join(f'{value:.12g}' for value in qc.POSITIVE_LOCK_SLIDER_LIMIT_SOLIMP)}"/>
    </body>
  </worldbody>
  {equality_xml}
  <keyframe><key name="unlocked" qpos="0"/></keyframe>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    joint_id = int(model.joint("slider").id)
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    initial_q_m = float(data.qpos[qpos_address])
    initial_qvel_m_s = float(data.qvel[dof_address])

    steps = math.ceil(max_time_s / timestep_s)
    q_history = np.empty(steps, dtype=np.float64)
    qvel_history = np.empty(steps, dtype=np.float64)
    for index in range(steps):
        mujoco.mj_step(model, data)
        q_history[index] = float(data.qpos[qpos_address])
        qvel_history[index] = float(data.qvel[dof_address])

    locked_lower_m = 0.00295
    locked_upper_m = 0.00305
    speed_limit_m_s = 0.001
    required_dwell_s = 0.050
    dwell_steps = math.ceil(required_dwell_s / timestep_s)
    final_q = q_history[-dwell_steps:]
    final_qvel = qvel_history[-dwell_steps:]
    reached_locked_band = bool(
        np.any((q_history >= locked_lower_m) & (q_history <= locked_upper_m))
    )
    final_position_in_band = bool(
        np.all((final_q >= locked_lower_m) & (final_q <= locked_upper_m))
    )
    final_speed_bounded = bool(np.all(np.abs(final_qvel) <= speed_limit_m_s))
    low_speed_dwell_verified = final_position_in_band and final_speed_bounded
    range_excursion_verified = bool(
        float(np.min(q_history)) >= -0.00005
        and float(np.max(q_history)) <= locked_upper_m
    )
    passed = bool(
        spring_enabled
        and not pin_equality_active
        and reached_locked_band
        and low_speed_dwell_verified
        and range_excursion_verified
    )
    return {
        "spring_enabled": bool(spring_enabled),
        "pin_equality_active": bool(pin_equality_active),
        "initialization_method": "mj_resetDataKeyframe_before_observation",
        "direct_state_writes_after_initialization": 0,
        "physics_transition_method": "mujoco.mj_step",
        "physics_substep_count": int(steps),
        "timestep_s": timestep_s,
        "elapsed_s": float(steps * timestep_s),
        "initial_q_m": initial_q_m,
        "initial_qvel_m_s": initial_qvel_m_s,
        "isolated_joint_reference_m": isolated_joint_reference_m,
        "spring_stiffness_n_m": float(stiffness_n_m),
        "spring_reference_m": qc.POSITIVE_LOCK_SLIDER_SPRINGREF_M,
        "damping_n_s_m": qc.POSITIVE_LOCK_SLIDER_DAMPING_N_S_M,
        "damping_derivation": "2*sqrt(exact_step_mass_kg*spring_stiffness_n_m)",
        "frictionloss_n": 0.0,
        "armature_kg": 0.0,
        "limit_solref": list(qc.POSITIVE_LOCK_SLIDER_LIMIT_SOLREF),
        "limit_solimp": list(qc.POSITIVE_LOCK_SLIDER_LIMIT_SOLIMP),
        "declared_joint_range_m": list(qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M),
        "locked_dwell_band_m": [locked_lower_m, locked_upper_m],
        "settled_speed_limit_m_s": speed_limit_m_s,
        "required_dwell_s": required_dwell_s,
        "reached_locked_band": reached_locked_band,
        "low_speed_dwell_verified": low_speed_dwell_verified,
        "dwell_s": required_dwell_s if low_speed_dwell_verified else 0.0,
        "q_min_final_m": float(np.min(final_q)),
        "q_max_final_m": float(np.max(final_q)),
        "max_abs_qvel_final_m_s": float(np.max(np.abs(final_qvel))),
        "q_min_trajectory_m": float(np.min(q_history)),
        "q_max_trajectory_m": float(np.max(q_history)),
        "max_abs_qvel_trajectory_m_s": float(np.max(np.abs(qvel_history))),
        "maximum_upper_limit_penetration_m": max(
            0.0,
            float(np.max(q_history)) - qc.POSITIVE_LOCK_SLIDER_JOINT_RANGE_M[1],
        ),
        "range_excursion_verified": range_excursion_verified,
        "passed": passed,
        "release_ready": False,
    }


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


def initialized_active_collision_geometry_sha256(
    model: mujoco.MjModel, data: mujoco.MjData
) -> str:
    """Hash every active geom parameter and its initialized world transform."""

    records: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        if not (
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        ):
            continue
        record: dict[str, Any] = {
            "geom_id": geom_id,
            "name": str(model.geom(geom_id).name),
            "body": str(model.body(int(model.geom_bodyid[geom_id])).name),
            "type": int(model.geom_type[geom_id]),
            "group": int(model.geom_group[geom_id]),
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
            "pos_float_hex": [
                float(value).hex() for value in model.geom_pos[geom_id]
            ],
            "quat_float_hex": [
                float(value).hex() for value in model.geom_quat[geom_id]
            ],
            "size_float_hex": [
                float(value).hex() for value in model.geom_size[geom_id]
            ],
            "initialized_world_pos_float_hex": [
                float(value).hex() for value in data.geom_xpos[geom_id]
            ],
            "initialized_world_xmat_float_hex": [
                float(value).hex() for value in data.geom_xmat[geom_id]
            ],
        }
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = np.ascontiguousarray(
                model.mesh_vert[
                    vertex_start : vertex_start + vertex_count
                ]
            )
            faces = np.ascontiguousarray(
                model.mesh_face[face_start : face_start + face_count]
            )
            record["mesh"] = {
                "vertex_count": vertex_count,
                "face_count": face_count,
                "vertex_dtype": vertices.dtype.str,
                "face_dtype": faces.dtype.str,
                "vertex_bytes_sha256": hashlib.sha256(
                    vertices.tobytes()
                ).hexdigest(),
                "face_bytes_sha256": hashlib.sha256(
                    faces.tobytes()
                ).hexdigest(),
            }
        records.append(record)
    return _canonical_json_sha256(records)


def _forward_scratch_arm_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_qpos_ids: np.ndarray,
    arm_q_rad: np.ndarray,
) -> None:
    """Evaluate scratch FK through MuJoCo's generalized-position API.

    This helper is used only by the static route contract on a private
    ``MjData``.  It never belongs to the controller call graph and performs no
    Python assignment to physical state.  ``mj_differentiatePos`` followed by
    ``mj_integratePos`` is the same topology-aware replay primitive used by
    the exact generalized-position trace contract.
    """

    target_qpos = np.array(data.qpos, dtype=np.float64, copy=True)
    target_qpos[arm_qpos_ids] = np.asarray(arm_q_rad, dtype=np.float64)
    generalized_velocity = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model,
        generalized_velocity,
        1.0,
        data.qpos,
        target_qpos,
    )
    mujoco.mj_integratePos(
        model,
        data.qpos,
        generalized_velocity,
        1.0,
    )
    mujoco.mj_forward(model, data)


def _small_rotation_angle(rotation: np.ndarray) -> float:
    sine_vector = 0.5 * np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(math.atan2(float(np.linalg.norm(sine_vector)), cosine))


def _core_capture_route_dense_fk_evidence(
    model: mujoco.MjModel,
) -> dict[str, Any]:
    """Replay the frozen joint-linear segments at every declared fraction."""

    data = mujoco.MjData(model)
    initialize(model, data)
    arm_qpos = np.asarray(
        [model.joint(name).qposadr[0] for name in ARM_JOINTS], dtype=int
    )
    dock_id = int(model.body("dock_gripper").id)
    mating_id = int(model.site("robot_mating_face").id)
    dock_position = np.asarray(data.xpos[dock_id], dtype=np.float64).copy()
    dock_rotation = np.asarray(
        data.xmat[dock_id], dtype=np.float64
    ).reshape(3, 3).copy()

    rows = CORE_CAPTURE_ROUTE_SOURCE_STATES
    phase_rows = (
        (
            "gripper_capture_lateral_align",
            (
                (
                    55.0,
                    0.0,
                    tuple(float(value) for value in DOCK_PRE_CAPTURE_Q["gripper"]),
                ),
                rows[0],
            ),
            "linear_lateral_alignment_at_p55",
        ),
        (
            "gripper_capture_axial_open_side",
            rows[0:244],
            "source_piecewise_x_law",
        ),
        (
            "gripper_capture_coupled_recenter",
            rows[243:260],
            "source_piecewise_x_law",
        ),
        (
            "gripper_capture_centered_final",
            rows[259:276],
            "source_piecewise_x_law",
        ),
    )
    phase_reports: list[dict[str, Any]] = []
    for phase_name, phase_states, expected_x_kind in phase_rows:
        maximum_preseat_error_mm = 0.0
        maximum_x_error_mm = 0.0
        maximum_abs_y_mm = 0.0
        maximum_orientation_error_rad = 0.0
        sample_count = 0
        sample_hasher = hashlib.sha256()
        sample_hasher.update(phase_name.encode() + b"\0")
        previous_preseat_mm = math.inf
        monotone_nonincreasing_preseat = True
        for interval_index, (start, end) in enumerate(
            zip(phase_states, phase_states[1:])
        ):
            start_p_mm, start_x_mm, start_q = start
            end_p_mm, end_x_mm, end_q = end
            start_q_array = np.asarray(start_q, dtype=np.float64)
            end_q_array = np.asarray(end_q, dtype=np.float64)
            for fraction_index, fraction in enumerate(
                CORE_CAPTURE_ROUTE_DENSE_FRACTIONS
            ):
                q_value = start_q_array + fraction * (
                    end_q_array - start_q_array
                )
                _forward_scratch_arm_configuration(
                    model, data, arm_qpos, q_value
                )
                local_position_mm = (
                    np.asarray(data.site_xpos[mating_id], dtype=np.float64)
                    - dock_position
                ) @ dock_rotation * 1000.0
                observed_preseat_mm = -float(local_position_mm[2])
                expected_preseat_mm = float(
                    start_p_mm + fraction * (end_p_mm - start_p_mm)
                )
                if expected_x_kind == "source_piecewise_x_law":
                    expected_x_mm = _core_capture_source_x_mm(
                        max(0.0, observed_preseat_mm)
                    )
                else:
                    expected_x_mm = float(
                        start_x_mm + fraction * (end_x_mm - start_x_mm)
                    )
                mating_rotation = np.asarray(
                    data.site_xmat[mating_id], dtype=np.float64
                ).reshape(3, 3)
                orientation_error_rad = _small_rotation_angle(
                    dock_rotation.T @ mating_rotation
                )
                maximum_preseat_error_mm = max(
                    maximum_preseat_error_mm,
                    abs(observed_preseat_mm - expected_preseat_mm),
                )
                maximum_x_error_mm = max(
                    maximum_x_error_mm,
                    abs(float(local_position_mm[0]) - expected_x_mm),
                )
                maximum_abs_y_mm = max(
                    maximum_abs_y_mm, abs(float(local_position_mm[1]))
                )
                maximum_orientation_error_rad = max(
                    maximum_orientation_error_rad, orientation_error_rad
                )
                if observed_preseat_mm > previous_preseat_mm + 1.0e-9:
                    monotone_nonincreasing_preseat = False
                previous_preseat_mm = observed_preseat_mm
                sample_hasher.update(
                    struct.pack(
                        "<IIddddd",
                        interval_index,
                        fraction_index,
                        fraction,
                        observed_preseat_mm,
                        float(local_position_mm[0]),
                        float(local_position_mm[1]),
                        orientation_error_rad,
                    )
                )
                sample_count += 1
        if phase_name == "gripper_capture_lateral_align":
            thresholds = {
                "maximum_preseat_error_mm": 0.0002,
                "maximum_source_x_error_mm": 0.0003,
                "maximum_abs_transverse_y_mm": 0.010,
                "maximum_orientation_error_rad": 1.0e-9,
            }
        else:
            thresholds = {
                "maximum_preseat_error_mm": 0.00005,
                "maximum_source_x_error_mm": 0.0001,
                "maximum_abs_transverse_y_mm": 0.010,
                "maximum_orientation_error_rad": 1.0e-9,
            }
        observed = {
            "maximum_preseat_error_mm": maximum_preseat_error_mm,
            "maximum_source_x_error_mm": maximum_x_error_mm,
            "maximum_abs_transverse_y_mm": maximum_abs_y_mm,
            "maximum_orientation_error_rad": maximum_orientation_error_rad,
        }
        preseat_progression_passed = (
            maximum_preseat_error_mm
            <= thresholds["maximum_preseat_error_mm"]
            if phase_name == "gripper_capture_lateral_align"
            else monotone_nonincreasing_preseat
        )
        phase_reports.append(
            {
                "action": phase_name,
                "interval_count": len(phase_states) - 1,
                "sample_count": sample_count,
                "expected_x_kind": expected_x_kind,
                "monotone_nonincreasing_preseat": (
                    monotone_nonincreasing_preseat
                ),
                "preseat_progression_kind": (
                    "constant_within_bound"
                    if phase_name == "gripper_capture_lateral_align"
                    else "monotone_nonincreasing"
                ),
                "preseat_progression_passed": preseat_progression_passed,
                "observed": observed,
                "thresholds": thresholds,
                "sample_sha256": sample_hasher.hexdigest(),
                "passed": preseat_progression_passed
                and all(observed[key] <= value for key, value in thresholds.items()),
            }
        )
    return {
        "sampling_contract": {
            "order": "action_then_interval_then_fraction",
            "fraction_count_per_interval": len(
                CORE_CAPTURE_ROUTE_DENSE_FRACTIONS
            ),
            "fractions": list(CORE_CAPTURE_ROUTE_DENSE_FRACTIONS),
            "endpoints_included_for_every_interval": True,
            "sample_digest_preimage": (
                "action_utf8+nul once, then little-endian "
                "<uint32 interval,uint32 fraction_index,"
                "float64 fraction,p_mm,x_mm,y_mm,orientation_rad>"
            ),
        },
        "phases": phase_reports,
        "passed": all(report["passed"] for report in phase_reports),
    }


def _move_action_command_kinematics(
    action: WorkflowAction,
    start_q: tuple[float, ...] | list[float],
    controller_dt_s: float,
) -> dict[str, Any]:
    """Sample the controller's exact quintic/polyline command schedule."""

    if action.kind != "move" or not action.joint_waypoints:
        raise ValueError("command kinematics requires a waypoint move action")
    times = np.arange(
        0.0,
        action.duration_s + 0.5 * controller_dt_s,
        controller_dt_s,
        dtype=np.float64,
    )
    route = np.asarray(
        (tuple(start_q), *action.joint_waypoints), dtype=np.float64
    )
    commands: list[np.ndarray] = []
    command_hasher = hashlib.sha256()
    for sample_index, time_s in enumerate(times):
        alpha = min(1.0, max(0.0, float(time_s) / action.duration_s))
        smooth = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
        route_position = smooth * (len(route) - 1)
        segment = min(int(math.floor(route_position)), len(route) - 2)
        segment_fraction = route_position - segment
        command = route[segment] + segment_fraction * (
            route[segment + 1] - route[segment]
        )
        commands.append(command)
        command_hasher.update(
            struct.pack("<Id", sample_index, float(time_s))
        )
        command_hasher.update(
            np.asarray(command, dtype="<f8").tobytes()
        )
    command_array = np.asarray(commands, dtype=np.float64)
    velocities = np.diff(command_array, axis=0) / controller_dt_s
    accelerations = np.diff(velocities, axis=0) / controller_dt_s
    maximum_speed = float(np.max(np.abs(velocities)))
    maximum_acceleration = float(np.max(np.abs(accelerations)))
    bounds = {
        "gripper_capture_lateral_align": (0.012, 0.15),
        "gripper_capture_axial_open_side": (0.66, 1.40),
        "gripper_capture_coupled_recenter": (0.12, 0.75),
        "gripper_capture_centered_final": (0.12, 0.75),
    }[action.name]
    return {
        "controller_dt_s": controller_dt_s,
        "time_sample_rule": "arange(0,T+dt/2,dt)",
        "time_sample_count": len(times),
        "velocity_method": "first_forward_difference_over_dt",
        "acceleration_method": "second_forward_difference_over_dt",
        "maximum_abs_joint_speed_rad_s": maximum_speed,
        "maximum_abs_joint_acceleration_rad_s2": maximum_acceleration,
        "maximum_abs_joint_speed_bound_rad_s": bounds[0],
        "maximum_abs_joint_acceleration_bound_rad_s2": bounds[1],
        "command_sample_sha256": command_hasher.hexdigest(),
        "command_sample_digest_preimage": (
            "per sample little-endian <uint32 sample_index,float64 time_s> "
            "then five little-endian float64 q values"
        ),
        "passed": maximum_speed <= bounds[0]
        and maximum_acceleration <= bounds[1],
    }


@cache
def _positive_lock_cam_capture_route_contract_cached() -> dict[str, Any]:
    model = build_model()
    initialized_data = mujoco.MjData(model)
    initialize(model, initialized_data)
    xml_text, _ = _build_xml_and_assets()
    source_binding = {
        "generator_file": {
            "path": str(qc.POGO_CAD_SOURCE_PATH.relative_to(REPO_ROOT)),
            "bytes": qc.POGO_CAD_SOURCE_BYTES,
            "sha256": qc.POGO_CAD_SOURCE_SHA256,
        },
        "positive_lock_cam_contract_sha256": (
            qc.CORE_DOCK_CAM_CONTRACT_CANONICAL_SHA256
        ),
        "route_functions": {
            "lateral_x_mm": "positive_lock_cam_capture_lateral_offset_mm",
            "slider_q_max_mm": "positive_lock_cam_capture_q_max_mm",
        },
    }
    model_binding = {
        "model_xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
        "initialized_active_collision_geometry_sha256": (
            initialized_active_collision_geometry_sha256(
                model, initialized_data
            )
        ),
        "physics_timestep_s": float(model.opt.timestep),
    }
    state_records = [
        {"preseat_mm": p_mm, "source_x_mm": x_mm, "q_rad": list(q_rad)}
        for p_mm, x_mm, q_rad in CORE_CAPTURE_ROUTE_SOURCE_STATES
    ]
    action_objects = _core_capture_move_actions()
    action_records: list[dict[str, Any]] = []
    alignment_q = [
        [float(value) for value in DOCK_PRE_CAPTURE_Q["gripper"]],
        list(CORE_CAPTURE_ROUTE_SOURCE_STATES[0][2]),
    ]
    if (
        _canonical_json_sha256(alignment_q)
        != CORE_CAPTURE_ROUTE_ALIGNMENT_Q_SHA256
    ):
        raise RuntimeError("core capture alignment digest drifted")
    action_start_q_by_name = {
        "gripper_capture_lateral_align": alignment_q[0],
        "gripper_capture_axial_open_side": list(
            CORE_CAPTURE_ROUTE_SOURCE_STATES[0][2]
        ),
        "gripper_capture_coupled_recenter": list(
            CORE_CAPTURE_ROUTE_SOURCE_STATES[243][2]
        ),
        "gripper_capture_centered_final": list(
            CORE_CAPTURE_ROUTE_SOURCE_STATES[259][2]
        ),
    }
    controller_dt_s = (
        float(model.opt.timestep) * PHYSICS_SUBSTEPS_PER_CONTROLLER_STEP
    )
    for action in action_objects:
        if action.name == "gripper_capture_lateral_align":
            row_range: list[int] | None = None
            full_q_roster = alignment_q
            expected_q_sha = CORE_CAPTURE_ROUTE_ALIGNMENT_Q_SHA256
        else:
            start, end = CORE_CAPTURE_ROUTE_PHASE_ROW_RANGES[action.name]
            row_range = [start, end]
            full_q_roster = [
                list(record[2])
                for record in CORE_CAPTURE_ROUTE_SOURCE_STATES[
                    start : end + 1
                ]
            ]
            expected_q_sha = CORE_CAPTURE_ROUTE_PHASE_Q_SHA256[action.name]
        observed_q_sha = _canonical_json_sha256(full_q_roster)
        if observed_q_sha != expected_q_sha:
            raise RuntimeError(f"{action.name} route digest drifted")
        action_records.append(
            {
                "name": action.name,
                "kind": action.kind,
                "tool": action.tool,
                "duration_s": action.duration_s,
                "timeout_s": action.timeout_s,
                "source_row_range_inclusive": row_range,
                "full_endpoint_inclusive_q_count": len(full_q_roster),
                "joint_waypoint_count_excluding_action_start": len(
                    action.joint_waypoints
                ),
                "endpoint_q_rad": list(action.target_q or ()),
                "q_roster_sha256": observed_q_sha,
                "time_scaling": "quintic_10a3_minus_15a4_plus_6a5",
                "zero_commanded_endpoint_velocity": True,
                "command_kinematics": _move_action_command_kinematics(
                    action,
                    action_start_q_by_name[action.name],
                    controller_dt_s,
                ),
            }
        )
    waypoint_digest_preimage = {
        "source_binding": source_binding,
        "model_binding": model_binding,
        "source_states": state_records,
    }
    dense_fk = _core_capture_route_dense_fk_evidence(model)
    old_open_q = (
        -0.72,
        *CORE_GUIDED_CAPTURE_BASE_Q[-1],
        0.0,
    )
    old_data = mujoco.MjData(model)
    initialize(model, old_data)
    arm_qpos = np.asarray(
        [model.joint(name).qposadr[0] for name in ARM_JOINTS], dtype=int
    )
    dock = old_data.body("dock_gripper")
    dock_position = np.asarray(dock.xpos, dtype=np.float64).copy()
    dock_rotation = np.asarray(
        dock.xmat, dtype=np.float64
    ).reshape(3, 3).copy()
    old_positions: list[list[float]] = []
    for q_value in (old_open_q, tuple(DOCK_CAPTURE_Q["gripper"])):
        _forward_scratch_arm_configuration(
            model, old_data, arm_qpos, np.asarray(q_value, dtype=np.float64)
        )
        old_positions.append(
            list(
                (
                    np.asarray(
                        old_data.site("robot_mating_face").xpos,
                        dtype=np.float64,
                    )
                    - dock_position
                )
                @ dock_rotation
                * 1000.0
            )
        )
    retired_same_z = {
        "name": "constant_x_plus_0p20_then_same_z_recenter",
        "endpoint_positions_dock_local_mm": old_positions,
        "preseat_change_mm": (
            -old_positions[1][2] + old_positions[0][2]
        ),
        "lateral_change_mm": old_positions[1][0] - old_positions[0][0],
        "complete_source_cam_overlap_mm3": {
            "slider_q_0p00mm": 0.2382911392405093,
            "slider_q_0p05mm": 0.3369620253164586,
        },
        "overlap_authority": (
            "independent_exact_OCCT_recomputation_required"
        ),
        "retired_single_action_command_kinematics": {
            "duration_s": 1.5,
            "maximum_abs_joint_speed_rad_s": 0.8784800468374154,
            "maximum_abs_joint_acceleration_rad_s2": 138.03108797818098,
            "rejected_for_nonzero_velocity_at_source_law_breakpoints": True,
        },
        "violates_source_piecewise_x_law": True,
        "single_global_action_crosses_velocity_kinks": True,
        "rejected": True,
    }
    endpoint_guard = {
        "maximum_q_error_rad": CORE_CAPTURE_ROUTE_ENDPOINT_Q_ERROR_RAD,
        "maximum_abs_qvel_rad_s": CORE_CAPTURE_ROUTE_ENDPOINT_QVEL_RAD_S,
        "maximum_fk_position_error_m": (
            CORE_CAPTURE_ROUTE_ENDPOINT_POSITION_ERROR_M
        ),
        "maximum_fk_orientation_error_rad": (
            CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD
        ),
        "maximum_absolute_source_x_error_mm": (
            CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
        ),
        "required_contiguous_controller_ticks": (
            CORE_CAPTURE_ROUTE_ENDPOINT_DWELL_TICKS
        ),
        "advance_on_elapsed_time_only": False,
    }
    live_source_corridor_guard = {
        "active_after_action": "gripper_capture_lateral_align",
        "audited_actions": [
            "gripper_capture_axial_open_side",
            "gripper_capture_coupled_recenter",
            "gripper_capture_centered_final",
        ],
        "audit_frequency": "after_every_mj_step",
        "preseat_formula": "-dock_local_robot_mating_z_mm",
        "lateral_x_formula": "dock_local_robot_mating_x_mm",
        "maximum_absolute_source_x_error_mm": (
            CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
        ),
        "bound_provenance_mm": {
            "continuous_plate_cam_clearance": (
                CORE_CAPTURE_SOURCE_CORRIDOR_CONTINUOUS_CLEARANCE_MM
            ),
            "manufacturing_clearance": (
                CORE_CAPTURE_SOURCE_CORRIDOR_MANUFACTURING_CLEARANCE_MM
            ),
            "retained_reserve": CORE_CAPTURE_SOURCE_CORRIDOR_RESERVE_MM,
            "available_tracking_error": (
                CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
            ),
            "formula": "0.249902439 - 0.20 - 0.009902439 = 0.040",
        },
        "violation_abort_reason": "core_capture_source_corridor_violation",
        "pass_requires": {
            "audited_substeps_greater_than_zero": True,
            "all_three_audited_actions_observed": True,
            "all_four_route_endpoint_events_completed": True,
            "maximum_error_within_bound": True,
            "current_abort_absent": True,
        },
        "live_dynamics_authority": False,
    }
    blockers = [
        "cam_contact_policy_not_authorized",
        "live_dynamics_not_validated",
        "closed_loop_source_law_tracking_not_implemented",
        "live_mujoco_route_tracking_not_yet_certified",
        "cam_tab_contact_force_and_depth_not_yet_certified",
        "positive_lock_cam_friction_coefficient_unqualified",
        "positive_lock_cam_load_capacity_unqualified",
        "positive_lock_cam_dynamics_unqualified",
    ]
    report = {
        "schema_version": "1.0",
        "tool": "gripper",
        "frame": "dock_gripper",
        "route_kind": "source_coupled_positive_lock_cam_capture",
        "embedded_state_bytes_sha256": (
            CORE_CAPTURE_ROUTE_STATE_BYTES_SHA256
        ),
        "contract_identity_digest_preimage": copy.deepcopy(
            CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_DIGEST_PREIMAGE
        ),
        "contract_identity_sha256": (
            CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256
        ),
        "source_binding": source_binding,
        "model_binding": model_binding,
        "route_law": {
            "preseat_from_fk": "-dock_local_robot_mating_z_mm",
            "lateral_x_from_fk": "dock_local_robot_mating_x_mm",
            "transverse_from_fk": "dock_local_robot_mating_y_mm",
            "orientation_reference": "seated_dock_frame",
            "x_breakpoints_mm": [
                [55.0, 0.20],
                [6.4, 0.20],
                [3.2, 0.0],
                [0.0, 0.0],
            ],
        },
        "source_states": state_records,
        "source_state_sha256": _canonical_json_sha256(state_records),
        "q_roster_sha256": _canonical_json_sha256(
            [record["q_rad"] for record in state_records]
        ),
        "canonical_waypoint_digest_preimage": waypoint_digest_preimage,
        "canonical_waypoint_sha256": _canonical_json_sha256(
            waypoint_digest_preimage
        ),
        "actions": action_records,
        "endpoint_guard": endpoint_guard,
        "live_source_corridor_guard": live_source_corridor_guard,
        "dense_fk_evidence": dense_fk,
        "retired_route_negative": retired_same_z,
        "state_write_contract": {
            "arm_command_target": "data.ctrl",
            "direct_pogo_qpos_writes_after_initialization": 0,
            "direct_slider_qpos_writes_after_initialization": 0,
            "validation_method": "independent_ast_and_callgraph_required",
        },
        "authority_scope": {
            "static_source_route_and_fk_authority": True,
            "live_tracking_authority": False,
            "contact_force_authority": False,
            "friction_coefficient_authority": False,
            "load_capacity_authority": False,
            "dynamics_authority": False,
            "blockers": blockers,
            "release_ready": False,
        },
        "passed": dense_fk["passed"]
        and all(
            record["command_kinematics"]["passed"]
            for record in action_records
        ),
        "release_ready": False,
    }
    if report["source_state_sha256"] != CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256:
        raise RuntimeError("core capture public source-state digest drifted")
    if report["q_roster_sha256"] != CORE_CAPTURE_ROUTE_Q_SHA256:
        raise RuntimeError("core capture public joint digest drifted")
    return report


def positive_lock_cam_capture_route_contract() -> dict[str, Any]:
    """Return independently replayable static route authority evidence."""

    return copy.deepcopy(_positive_lock_cam_capture_route_contract_cached())


def core_capture_route_runtime_contract() -> dict[str, Any]:
    """Compatibility-free public name for the core runtime route contract."""

    return positive_lock_cam_capture_route_contract()


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
        self.geom_names = tuple(
            str(model.geom(geom_id).name) for geom_id in range(model.ngeom)
        )
        self.robot_mating_land_names = frozenset({
            "qc_col_robot_plate_core__mating_land",
            "qc_col_robot_plate_cam_relief_part_01",
            *(
                str(model.geom(geom_id).name)
                for geom_id in range(model.ngeom)
                if str(model.geom(geom_id).name).startswith(
                    "qc_col_robot_plate_upper_well_partition_"
                )
                and str(model.geom(geom_id).name).endswith("__mating_land")
            ),
        })
        self.dock_stop_names_by_tool = MappingProxyType({
            tool: frozenset(
                name
                for name in self.geom_names
                if qc.is_dock_stop_collision_name(tool, name)
            )
            for tool in ALL_TOOL_IDS
        })
        self.pogo_pair_contract = MappingProxyType({
            frozenset(
                {
                    f"qc_col_pogo_{signal}_plunger",
                    f"{tool}_pad_{signal}_collision",
                }
            ): (tool, signal)
            for tool in ALL_TOOL_IDS
            for signal in qc.SIGNALS
        })
        self.support_contact_pairs = frozenset(
            {
                frozenset(
                    {
                        f"dock_{tool}_support_anchor_collision",
                        f"dock_{tool}_support_collision",
                    }
                )
                for tool in ALL_TOOL_IDS
            }
            | {
                frozenset(
                    {
                        f"dock_{tool}_support_collision",
                        "matcha_floor_collision",
                    }
                )
                for tool in ALL_TOOL_IDS
            }
        )
        self.slider_tab_geom_ids = tuple(
            geom_id
            for geom_id, name in enumerate(self.geom_names)
            if name.startswith("qc_col_lock_slider_tab_part_")
        )
        self.dock_gripper_cam_geom_ids = tuple(
            int(model.geom(name).id)
            for name in qc.positive_lock_cam_collision_geom_names("gripper")
        )
        # Preserve the published legacy scalar as the main-wedge ID while all
        # physical clearance/contact queries consume the complete roster.
        self.dock_gripper_cam_geom_id = self.dock_gripper_cam_geom_ids[0]
        self.core_cam_tab_leading_geom_id = int(
            model.geom(CORE_CAM_TAB_LEADING_GEOM_NAME).id
        )
        self.core_cam_tab_noncontact_geom_id = int(
            model.geom(CORE_CAM_TAB_NONCONTACT_GEOM_NAME).id
        )
        self.core_cam_geom_name_by_id = MappingProxyType(
            {
                int(model.geom(name).id): name
                for name in qc.CORE_DOCK_CAM_COLLISION_GEOM_NAMES
            }
        )
        self.core_cam_tab_allowed_contact_indices: set[int] = set()
        self.core_cam_tab_contact_records: list[dict[str, Any]] = []
        self.core_cam_tab_audited_substeps = 0
        self.core_cam_tab_phase_counts = {
            name: 0 for name in CORE_CAM_TAB_CAPTURE_ACTIONS
        }
        self.core_cam_tab_candidate_contact_count = 0
        self.core_cam_tab_rejected_contact_count = 0
        self.core_cam_tab_functional_role_counts = {
            "functional_axial_lead_ramp": 0,
            "functional_hold_finger_face": 0,
        }
        self.core_capture_free_space_samples: list[dict[str, Any]] = []
        self.core_capture_free_space_phase_counts = {
            name: 0 for name in CORE_CAM_TAB_FREE_SPACE_ACTIONS
        }
        self.current_move_command_smooth = 0.0
        self.action_index = 0
        self.action_started_s = float(data.time)
        self.move_endpoint_dwell_ticks = 0
        self.core_capture_source_corridor_armed = False
        self.core_capture_source_corridor_max_error_mm = 0.0
        self.core_capture_source_corridor_witness: dict[str, Any] | None = None
        self.core_capture_source_corridor_audited_substeps = 0
        self.core_capture_source_corridor_phase_counts = {
            "gripper_capture_axial_open_side": 0,
            "gripper_capture_coupled_recenter": 0,
            "gripper_capture_centered_final": 0,
        }
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
        self.core_keeper_capture_verified = False
        self.core_keeper_capture_report: dict[str, Any] | None = None
        self.attachment_candidate_verified = False
        self.attachment_verified = False
        self.physical_lock_confirmed = False
        self.lock_candidate_verified = False
        self.locked = False
        self.lock_confirmation_phase = "pre_capture"
        self.minimum_source_axis_withdrawal_mm = MINIMUM_SOURCE_AXIS_WITHDRAWAL_MM
        self.capture_pogo_signals: list[str] = []
        self.capture_mating_pose_evidence: dict[str, Any] | None = None
        self.capture_robot_mating_rotation: np.ndarray | None = None
        self.first_physical_lock_true_substep: int | None = None
        self.source_axis_withdrawal_evidence: dict[str, Any] | None = None
        self.slider_return_evidence: dict[str, Any] | None = None
        self.slider_return_samples: list[tuple[int, float, float]] = []
        self.slider_settled_substeps = 0
        self.slider_settled_cam_min_clearance_mm = math.inf
        self.slider_settled_cam_contact_count = 0
        self.lock_stroke_guide_records: dict[tuple[str, str], dict[str, Any]] = {}
        self.capture_live_substeps = 0
        self.lock_live_substeps = 0
        self.release_live_substeps = 0
        self.physics_substep_count = 0
        self.forbidden_contact_count = 0
        self.max_forbidden_penetration_m = 0.0
        self.first_forbidden_pair: tuple[str, str] | None = None
        self.max_tracking_error_rad = 0.0
        self.max_route_lateral_deviation_m = 0.0
        self.max_route_orientation_error_rad = 0.0
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

    def contact_classification_cache(self) -> dict[str, Any]:
        """Expose the immutable, compiled-name-exact audit classifier roster."""

        return {
            "geom_names": self.geom_names,
            "robot_mating_land_names": self.robot_mating_land_names,
            "dock_stop_names_by_tool": tuple(
                (tool, self.dock_stop_names_by_tool[tool])
                for tool in sorted(self.dock_stop_names_by_tool)
            ),
            "pogo_pair_contract": tuple(
                sorted(
                    (
                        tuple(sorted(pair)),
                        tuple(contract),
                    )
                    for pair, contract in self.pogo_pair_contract.items()
                )
            ),
            "support_contact_pairs": tuple(
                sorted(tuple(sorted(pair)) for pair in self.support_contact_pairs)
            ),
            "slider_tab_geom_ids": self.slider_tab_geom_ids,
            "dock_gripper_cam_geom_id": self.dock_gripper_cam_geom_id,
            "core_cam_tab_leading_geom_id": (
                self.core_cam_tab_leading_geom_id
            ),
            "core_cam_tab_noncontact_geom_id": (
                self.core_cam_tab_noncontact_geom_id
            ),
            "core_cam_geom_name_by_id": tuple(
                sorted(self.core_cam_geom_name_by_id.items())
            ),
        }

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

    @staticmethod
    def _quat_wxyz_from_rotation(rotation: np.ndarray) -> list[float]:
        """Return one finite, sign-canonical MuJoCo quaternion."""

        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion, np.asarray(rotation, dtype=np.float64).reshape(9)
        )
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("cannot encode nonfinite mating-frame rotation")
        quaternion /= norm
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        return [float(value) for value in quaternion]

    def _mating_world_pose_evidence(self) -> dict[str, Any]:
        """Snapshot the live robot mating site and immutable dock body pose."""

        robot = self.data.site("robot_mating_face")
        dock = self.data.body("dock_gripper")
        return {
            "robot_mating_position_world_m": [
                float(value) for value in np.asarray(robot.xpos, dtype=float)
            ],
            "robot_mating_quat_wxyz": self._quat_wxyz_from_rotation(
                np.asarray(robot.xmat, dtype=float).reshape(3, 3)
            ),
            "dock_position_world_m": [
                float(value) for value in np.asarray(dock.xpos, dtype=float)
            ],
            "dock_quat_wxyz": self._quat_wxyz_from_rotation(
                np.asarray(dock.xmat, dtype=float).reshape(3, 3)
            ),
        }

    def _record_lock_stroke_contacts(self) -> None:
        """Accumulate exact four-rail sliding-contact depth/normal/force data."""

        dock_rotation = np.asarray(
            self.data.body("dock_gripper").xmat, dtype=float
        ).reshape(3, 3)
        contract_by_pair = {
            frozenset(record["runtime_pair"]): record
            for record in CORE_KEEPER_CONTACT_CONTRACT
            if record["source_pair"][0] == "stock_tool_plate"
        }
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            names = frozenset(
                self.geom_names[int(geom_id)] for geom_id in contact.geom
            )
            contract = contract_by_pair.get(names)
            if contract is None:
                continue
            key = tuple(str(value) for value in contract["source_pair"])
            record = self.lock_stroke_guide_records.setdefault(
                key,
                {
                    "source_pair": list(contract["source_pair"]),
                    "runtime_pair": list(contract["runtime_pair"]),
                    "contact_sample_count": 0,
                    "max_penetration_mm": 0.0,
                    "max_normal_force_n": 0.0,
                    "all_normals_valid": True,
                    "finite_force_verified": True,
                },
            )
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = abs(float(force[0]))
            record["contact_sample_count"] += 1
            record["max_penetration_mm"] = max(
                float(record["max_penetration_mm"]),
                max(0.0, -float(contact.dist)) * 1000.0,
            )
            record["max_normal_force_n"] = max(
                float(record["max_normal_force_n"]), normal_force
            )
            record["finite_force_verified"] = bool(
                record["finite_force_verified"]
                and np.all(np.isfinite(force))
            )
            record["all_normals_valid"] = bool(
                record["all_normals_valid"]
                and self._core_keeper_normal_is_valid(
                    contact, contract, dock_rotation
                )
            )

    def _lock_stroke_report(self) -> dict[str, Any]:
        if (
            self.capture_mating_pose_evidence is None
            or self.capture_robot_mating_rotation is None
        ):
            raise RuntimeError("lock stroke cannot precede physical capture")
        current_pose = self._mating_world_pose_evidence()
        capture_position = np.asarray(
            self.capture_mating_pose_evidence[
                "robot_mating_position_world_m"
            ],
            dtype=float,
        )
        current_position = np.asarray(
            current_pose["robot_mating_position_world_m"], dtype=float
        )
        dock_rotation = np.asarray(
            self.data.body("dock_gripper").xmat, dtype=float
        ).reshape(3, 3)
        displacement_local = dock_rotation.T @ (current_position - capture_position)
        displacement_norm = float(np.linalg.norm(displacement_local))
        withdrawal_mm = -1000.0 * float(displacement_local[2])
        axis_alignment = (
            -float(displacement_local[2]) / displacement_norm
            if displacement_norm > 0.0
            else 0.0
        )
        lateral_deviation_mm = 1000.0 * float(
            np.linalg.norm(displacement_local[:2])
        )
        current_rotation = np.asarray(
            self.data.site("robot_mating_face").xmat, dtype=float
        ).reshape(3, 3)
        cosine = float(
            (
                np.trace(
                    self.capture_robot_mating_rotation.T @ current_rotation
                )
                - 1.0
            )
            / 2.0
        )
        orientation_error_rad = math.acos(max(-1.0, min(1.0, cosine)))
        guide_records: list[dict[str, Any]] = []
        for contract in CORE_KEEPER_CONTACT_CONTRACT:
            if contract["source_pair"][0] != "stock_tool_plate":
                continue
            key = tuple(str(value) for value in contract["source_pair"])
            record = copy.deepcopy(
                self.lock_stroke_guide_records.get(
                    key,
                    {
                        "source_pair": list(contract["source_pair"]),
                        "runtime_pair": list(contract["runtime_pair"]),
                        "contact_sample_count": 0,
                        "max_penetration_mm": 0.0,
                        "max_normal_force_n": 0.0,
                        "all_normals_valid": True,
                        "finite_force_verified": True,
                    },
                )
            )
            record["force_acceptance_semantics"] = (
                "finite_measured_evidence_no_release_limit_declared"
            )
            record["passed"] = bool(
                record["max_penetration_mm"] <= CORE_KEEPER_MAX_PENETRATION_MM
                and record["all_normals_valid"]
                and record["finite_force_verified"]
            )
            guide_records.append(record)
        passed = bool(
            CORE_LOCK_RELEASE_MIN_STROKE_MM <= withdrawal_mm
            <= CORE_LOCK_RELEASE_STROKE_MM + 1.0e-6
            and axis_alignment >= 0.999
            and lateral_deviation_mm <= CAM_RELIEF_CORRIDOR_M * 1000.0
            and orientation_error_rad <= CAPTURE_ORIENTATION_TOLERANCE_RAD
            and not self._equality_active("dock_gripper_hold")
            and self._equality_active("attach_gripper")
            and all(record["passed"] for record in guide_records)
        )
        return {
            "source_axis": CORE_LOCK_RELEASE_SOURCE_AXIS,
            "source_axis_dock_local": list(CORE_LOCK_RELEASE_AXIS_DOCK_LOCAL),
            "withdrawal_mm": withdrawal_mm,
            "minimum_withdrawal_mm": CORE_LOCK_RELEASE_MIN_STROKE_MM,
            "maximum_withdrawal_mm": CORE_LOCK_RELEASE_STROKE_MM,
            "axis_alignment": axis_alignment,
            "lateral_deviation_mm": lateral_deviation_mm,
            "orientation_error_rad": orientation_error_rad,
            "dock_hold_active": self._equality_active("dock_gripper_hold"),
            "attach_equality_active": self._equality_active("attach_gripper"),
            "guide_contact_records": guide_records,
            **current_pose,
            "passed": passed,
        }

    def _record_slider_state(self, action: WorkflowAction) -> None:
        joint = self.model.joint("qc_positive_lock_slider_joint")
        qpos = float(self.data.qpos[int(joint.qposadr[0])])
        qvel = float(self.data.qvel[int(joint.dofadr[0])])
        self.slider_return_samples.append(
            (int(self.physics_substep_count), qpos, qvel)
        )
        in_band = (
            LOCKED_SLIDER_POSITION_BAND_M[0]
            <= qpos
            <= LOCKED_SLIDER_POSITION_BAND_M[1]
        )
        speed_bounded = abs(qvel) <= LOCKED_SLIDER_SPEED_LIMIT_M_S
        if action.kind == "slider_return" and in_band and speed_bounded:
            self.slider_settled_substeps += 1
            runtime_clearance_mm, cam_contacts = self._slider_cam_runtime_clearance()
            self.slider_settled_cam_min_clearance_mm = min(
                self.slider_settled_cam_min_clearance_mm,
                runtime_clearance_mm,
            )
            self.slider_settled_cam_contact_count += int(cam_contacts)
        elif action.kind == "slider_return":
            self.slider_settled_substeps = 0
            self.slider_settled_cam_min_clearance_mm = math.inf
            self.slider_settled_cam_contact_count = 0

    def _slider_cam_runtime_clearance(self) -> tuple[float, int]:
        cam_ids = self.dock_gripper_cam_geom_ids
        tab_ids = self.slider_tab_geom_ids
        if not tab_ids:
            raise RuntimeError("positive-lock slider has no active cam-tab prisms")
        distances: list[float] = []
        for geom_id in tab_ids:
            for cam_id in cam_ids:
                from_to = np.empty(6, dtype=np.float64)
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model, self.data, geom_id, cam_id, 0.1, from_to
                    )
                )
                if not math.isfinite(distance):
                    raise RuntimeError("nonfinite slider/cam runtime distance")
                distances.append(distance)
        cam_contacts = sum(
            1
            for index in range(self.data.ncon)
            if any(
                cam_id in {int(value) for value in self.data.contact[index].geom}
                for cam_id in cam_ids
            )
            and any(
                int(value) in tab_ids for value in self.data.contact[index].geom
            )
        )
        return min(distances) * 1000.0, cam_contacts

    def _core_cam_tab_live_kinematics(self) -> dict[str, float]:
        """Return live dock-frame capture state without mutating physics."""

        dock = self.data.body("dock_gripper")
        mating = self.data.site("robot_mating_face")
        dock_rotation = np.asarray(dock.xmat, dtype=np.float64).reshape(3, 3)
        local_position_mm = dock_rotation.T @ (
            np.asarray(mating.xpos, dtype=np.float64)
            - np.asarray(dock.xpos, dtype=np.float64)
        ) * 1000.0
        mating_rotation = np.asarray(
            mating.xmat, dtype=np.float64
        ).reshape(3, 3)
        relative_rotation = dock_rotation.T @ mating_rotation
        orientation_error_rad = math.acos(
            float(
                np.clip(
                    (float(np.trace(relative_rotation)) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        slider_joint = self.model.joint("qc_positive_lock_slider_joint")
        slider_q_mm = (
            float(self.data.qpos[int(slider_joint.qposadr[0])]) * 1000.0
        )
        preseat_mm = -float(local_position_mm[2])
        source_x_mm = float(local_position_mm[0])
        nonnegative_preseat_mm = max(0.0, preseat_mm)
        return {
            "preseat_mm": preseat_mm,
            "source_x_mm": source_x_mm,
            "transverse_y_mm": float(local_position_mm[1]),
            "orientation_error_rad": orientation_error_rad,
            "slider_q_mm": slider_q_mm,
            "source_q_max_mm": _core_cam_tab_source_q_max_mm(
                nonnegative_preseat_mm, source_x_mm
            ),
            "expected_source_x_mm": _core_capture_source_x_mm(
                nonnegative_preseat_mm
            ),
        }

    @staticmethod
    def _core_cam_tab_interval_error(
        value: float, lower: float, upper: float
    ) -> float:
        if value < lower:
            return lower - value
        if value > upper:
            return value - upper
        return 0.0

    @staticmethod
    def _core_cam_tab_seam_normal_is_valid(normal: np.ndarray) -> bool:
        """Accept only the source-union lead-to-hold outward normal cone."""

        xz_alignment = float(np.linalg.norm(normal[[0, 2]]))
        return bool(
            xz_alignment >= CORE_CAM_TAB_MIN_NORMAL_ALIGNMENT
            and float(normal[0]) <= 1.0e-12
            and float(normal[2]) <= 1.0e-12
            and -float(normal[2]) <= -float(normal[0]) + 1.0e-12
        )

    @staticmethod
    def _core_cam_tab_main_edge_normal_is_valid(normal: np.ndarray) -> bool:
        """Accept the exact main-slope/hold-top convex edge normal cone."""

        xy_alignment = float(np.linalg.norm(normal[:2]))
        return bool(
            xy_alignment >= CORE_CAM_TAB_MIN_NORMAL_ALIGNMENT
            and float(normal[0]) <= 1.0e-12
            and float(normal[1])
            >= CORE_CAM_TAB_MAIN_SLOPE * float(normal[0]) - 1.0e-12
        )

    def _core_cam_tab_contact_record(
        self, contact_index: int, action: WorkflowAction
    ) -> dict[str, Any] | None:
        """Classify and retain one live core-cam contact fail closed."""

        contact = self.data.contact[contact_index]
        geom_ids = (int(contact.geom[0]), int(contact.geom[1]))
        cam_id = next(
            (value for value in geom_ids if value in self.core_cam_geom_name_by_id),
            None,
        )
        if cam_id is None:
            return None
        other_id = geom_ids[1] if geom_ids[0] == cam_id else geom_ids[0]
        cam_name = self.core_cam_geom_name_by_id[cam_id]
        other_name = self.geom_names[other_id]
        pair_eligible = bool(
            other_id == self.core_cam_tab_leading_geom_id
            and cam_name
            in {
                CORE_CAM_TAB_MAIN_GEOM_NAME,
                CORE_CAM_TAB_LEAD_GEOM_NAME,
                CORE_CAM_TAB_HOLD_GEOM_NAME,
            }
        )

        dock = self.data.body("dock_gripper")
        dock_rotation = np.asarray(dock.xmat, dtype=np.float64).reshape(3, 3)
        contact_position_world = np.asarray(contact.pos, dtype=np.float64)
        contact_position_dock_mm = dock_rotation.T @ (
            contact_position_world - np.asarray(dock.xpos, dtype=np.float64)
        ) * 1000.0
        normal_world = np.asarray(contact.frame[:3], dtype=np.float64)
        # MuJoCo's frame normal points geom0 -> geom1.  Canonicalize cam -> tab.
        if geom_ids[0] != cam_id:
            normal_world = -normal_world
        normal_dock = dock_rotation.T @ normal_world
        normal_length = float(np.linalg.norm(normal_dock))
        if math.isfinite(normal_length) and normal_length > 0.0:
            normal_dock = normal_dock / normal_length
        else:
            normal_dock = np.full(3, np.nan, dtype=np.float64)

        live = self._core_cam_tab_live_kinematics()
        preseat_mm = float(live["preseat_mm"])
        source_x_mm = float(live["source_x_mm"])
        slider_q_mm = float(live["slider_q_mm"])
        source_q_max_mm = float(live["source_q_max_mm"])
        point_x, point_y, point_z = (
            float(value) for value in contact_position_dock_mm
        )
        tolerance = CORE_CAM_TAB_POINT_TOLERANCE_MM
        surface_role = "unclassified_or_forbidden_core_cam_contact"
        locus_error_mm = math.inf
        normal_alignment = -math.inf
        locus_valid = False
        normal_valid = False
        q_valid = False
        functional_coverage_role: str | None = None

        if (
            pair_eligible
            and cam_name == CORE_CAM_TAB_LEAD_GEOM_NAME
            and action.name == "gripper_capture_coupled_recenter"
            and CORE_CAM_TAB_RAMP_END_PRESEAT_MM - tolerance
            <= preseat_mm
            <= CORE_CAM_TAB_RAMP_CONTACT_START_PRESEAT_MM + tolerance
        ):
            surface_role = "functional_axial_lead_ramp"
            locus_error_mm = max(
                abs(point_x + point_z - CORE_CAM_TAB_LEAD_PLANE_SUM_MM),
                self._core_cam_tab_interval_error(point_y, 0.0, 2.0),
                self._core_cam_tab_interval_error(point_z, -9.6, -6.4),
            )
            normal_alignment = float(
                normal_dock
                @ np.asarray(
                    CORE_CAM_TAB_LEAD_NORMAL_CAM_TO_TAB, dtype=np.float64
                )
            )
            locus_valid = locus_error_mm <= tolerance
            normal_valid = normal_alignment >= CORE_CAM_TAB_MIN_NORMAL_ALIGNMENT
            q_excess_mm = slider_q_mm - source_q_max_mm
            q_valid = bool(
                -tolerance <= q_excess_mm
                <= math.sqrt(2.0)
                * CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
            )
            functional_coverage_role = surface_role
        elif (
            pair_eligible
            and cam_name == CORE_CAM_TAB_HOLD_GEOM_NAME
            and action.name == "gripper_capture_centered_final"
            and -tolerance <= preseat_mm <= 3.2 + tolerance
        ):
            surface_role = "functional_hold_finger_face"
            tab_lower_z_mm = -4.8 - max(0.0, preseat_mm)
            tab_upper_z_mm = -3.2 - max(0.0, preseat_mm)
            locus_error_mm = max(
                abs(point_x - 24.05),
                self._core_cam_tab_interval_error(point_y, 0.0, 2.0),
                self._core_cam_tab_interval_error(
                    point_z,
                    max(-6.4, tab_lower_z_mm),
                    min(-4.15, tab_upper_z_mm),
                ),
            )
            normal_alignment = float(
                normal_dock
                @ np.asarray(
                    CORE_CAM_TAB_HOLD_NORMAL_CAM_TO_TAB, dtype=np.float64
                )
            )
            locus_valid = locus_error_mm <= tolerance
            normal_valid = normal_alignment >= CORE_CAM_TAB_MIN_NORMAL_ALIGNMENT
            q_valid = bool(
                -tolerance
                <= slider_q_mm
                <= CORE_CAM_TAB_PASSIVE_OPEN_Q_MM
                + CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
            )
            functional_coverage_role = surface_role
        elif (
            pair_eligible
            and cam_name == CORE_CAM_TAB_LEAD_GEOM_NAME
            and action.name == "gripper_capture_centered_final"
            and CORE_CAM_TAB_LEAD_SEAM_END_PRESEAT_MM - tolerance
            <= preseat_mm
            < CORE_CAM_TAB_RAMP_END_PRESEAT_MM
        ):
            surface_role = "lead_hold_partition_seam_nonfunctional"
            locus_error_mm = max(
                abs(point_x - 24.05),
                abs(point_z + 6.4),
                self._core_cam_tab_interval_error(point_y, 0.0, 2.0),
            )
            normal_alignment = float(np.linalg.norm(normal_dock[[0, 2]]))
            locus_valid = locus_error_mm <= tolerance
            normal_valid = self._core_cam_tab_seam_normal_is_valid(normal_dock)
            q_valid = bool(
                -tolerance
                <= slider_q_mm
                <= CORE_CAM_TAB_PASSIVE_OPEN_Q_MM
                + CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
            )
        elif (
            pair_eligible
            and cam_name == CORE_CAM_TAB_MAIN_GEOM_NAME
            and action.name == "gripper_capture_centered_final"
            and -tolerance
            <= preseat_mm
            <= CORE_CAM_TAB_MAIN_EDGE_END_PRESEAT_MM + tolerance
        ):
            surface_role = "main_hold_edge_tangency_nonfunctional"
            upper_z_mm = -3.2 - max(0.0, preseat_mm)
            locus_error_mm = max(
                abs(point_x - 24.05),
                abs(point_y),
                self._core_cam_tab_interval_error(
                    point_z, -4.15, upper_z_mm
                ),
            )
            normal_alignment = float(np.linalg.norm(normal_dock[:2]))
            locus_valid = locus_error_mm <= tolerance
            normal_valid = self._core_cam_tab_main_edge_normal_is_valid(
                normal_dock
            )
            q_valid = bool(
                -tolerance
                <= slider_q_mm
                <= CORE_CAM_TAB_PASSIVE_OPEN_Q_MM
                + CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
            )

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, contact_index, force)
        penetration_mm = max(0.0, -float(contact.dist) * 1000.0)
        force_finite = bool(np.all(np.isfinite(force)))
        dock_hold_active = self._equality_active("dock_gripper_hold")
        attach_equality_active = self._equality_active("attach_gripper")
        phase_state_valid = bool(
            action.name in CORE_CAM_TAB_CAPTURE_ACTIONS
            and dock_hold_active
            and not attach_equality_active
            and abs(
                source_x_mm - float(live["expected_source_x_mm"])
            )
            <= CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
            and abs(float(live["transverse_y_mm"])) <= 0.010
            and float(live["orientation_error_rad"])
            <= CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD
        )
        provisional_passed = bool(
            pair_eligible
            and surface_role != "unclassified_or_forbidden_core_cam_contact"
            and phase_state_valid
            and locus_valid
            and normal_valid
            and q_valid
            and force_finite
            and float(force[0]) >= -1.0e-12
            and penetration_mm
            <= CORE_CAM_TAB_PROVISIONAL_MAX_PENETRATION_MM
        )
        return {
            "physics_substep_count": int(self.physics_substep_count),
            "sim_time_s": float(self.data.time),
            "action": action.name,
            "runtime_pair": [self.geom_names[geom_ids[0]], self.geom_names[geom_ids[1]]],
            "canonical_pair": [other_name, cam_name],
            "cam_geom": cam_name,
            "tab_or_other_geom": other_name,
            "surface_role": surface_role,
            "preseat_mm": preseat_mm,
            "source_x_mm": source_x_mm,
            "transverse_y_mm": float(live["transverse_y_mm"]),
            "orientation_error_rad": float(live["orientation_error_rad"]),
            "slider_q_mm": slider_q_mm,
            "source_q_max_mm": source_q_max_mm,
            "source_x_error_mm": (
                source_x_mm - float(live["expected_source_x_mm"])
            ),
            "dock_hold_active": dock_hold_active,
            "attach_equality_active": attach_equality_active,
            "contact_dist_mm": float(contact.dist) * 1000.0,
            "penetration_mm": penetration_mm,
            "contact_position_world_m": [
                float(value) for value in contact_position_world
            ],
            "contact_position_dock_local_mm": [
                float(value) for value in contact_position_dock_mm
            ],
            "contact_normal_raw_world": [
                float(value) for value in np.asarray(contact.frame[:3], dtype=float)
            ],
            "contact_normal_cam_to_tab_dock_local": [
                float(value) if math.isfinite(float(value)) else None
                for value in normal_dock
            ],
            "contact_frame_3x3": [
                float(value) for value in np.asarray(contact.frame, dtype=float)
            ],
            "contact_friction": [
                float(value) for value in np.asarray(contact.friction, dtype=float)
            ],
            "contact_solref": [
                float(value) for value in np.asarray(contact.solref, dtype=float)
            ],
            "contact_solimp": [
                float(value) for value in np.asarray(contact.solimp, dtype=float)
            ],
            "contact_force_torque_6d": [
                float(value) if math.isfinite(float(value)) else None
                for value in force
            ],
            "locus_error_mm": (
                locus_error_mm if math.isfinite(locus_error_mm) else None
            ),
            "normal_alignment": (
                normal_alignment if math.isfinite(normal_alignment) else None
            ),
            "q_excess_mm": slider_q_mm - source_q_max_mm,
            "pair_eligible": pair_eligible,
            "phase_state_valid": phase_state_valid,
            "locus_valid": locus_valid,
            "normal_valid": normal_valid,
            "q_valid": q_valid,
            "force_finite": force_finite,
            "provisional_classification_passed": provisional_passed,
            "functional_coverage_role": functional_coverage_role,
        }

    def _audit_core_capture_cam_tab_contacts(
        self, action: WorkflowAction
    ) -> None:
        """Count every core-cam contact before the generic contact audit."""

        self.core_cam_tab_allowed_contact_indices.clear()
        if action.name not in CORE_CAM_TAB_CAPTURE_ACTIONS:
            return
        self.core_cam_tab_audited_substeps += 1
        self.core_cam_tab_phase_counts[action.name] += 1
        for contact_index in range(self.data.ncon):
            record = self._core_cam_tab_contact_record(contact_index, action)
            if record is None:
                continue
            self.core_cam_tab_contact_records.append(record)
            if bool(record["provisional_classification_passed"]):
                self.core_cam_tab_candidate_contact_count += 1
                self.core_cam_tab_allowed_contact_indices.add(contact_index)
                role = record["functional_coverage_role"]
                if role is not None:
                    self.core_cam_tab_functional_role_counts[str(role)] += 1
            else:
                self.core_cam_tab_rejected_contact_count += 1

    def _record_core_capture_free_space_tracking(
        self, action: WorkflowAction
    ) -> None:
        """Record every real free-space capture substep before cam contact."""

        if action.name not in CORE_CAM_TAB_FREE_SPACE_ACTIONS:
            return
        live = self._core_cam_tab_live_kinematics()
        if action.name == "gripper_capture_lateral_align":
            expected_preseat_mm = 55.0
            expected_x_mm = 0.20 * self.current_move_command_smooth
        else:
            expected_preseat_mm = 55.0 - 48.6 * self.current_move_command_smooth
            expected_x_mm = 0.20
        q_error = np.asarray(self.data.qpos[self.arm_qpos_ids], dtype=float) - np.asarray(
            self.data.ctrl[self.arm_actuator_ids], dtype=float
        )
        lead_x_gap_mm = (
            float(live["preseat_mm"])
            - float(live["source_x_mm"])
            - 3.15
            - float(live["slider_q_mm"])
        )
        cam_contact_count = sum(
            1
            for index in range(self.data.ncon)
            if any(
                int(geom_id) in self.core_cam_geom_name_by_id
                for geom_id in self.data.contact[index].geom
            )
        )
        sample = {
            "physics_substep_count": int(self.physics_substep_count),
            "sim_time_s": float(self.data.time),
            "action": action.name,
            "command_smooth_fraction": float(self.current_move_command_smooth),
            "commanded_arm_q_rad": [
                float(value) for value in self.data.ctrl[self.arm_actuator_ids]
            ],
            "observed_arm_q_rad": [
                float(value) for value in self.data.qpos[self.arm_qpos_ids]
            ],
            "observed_arm_qvel_rad_s": [
                float(value) for value in self.data.qvel[self.arm_dof_ids]
            ],
            "max_abs_q_tracking_error_rad": float(np.max(np.abs(q_error))),
            "expected_preseat_mm": expected_preseat_mm,
            "observed_preseat_mm": float(live["preseat_mm"]),
            "preseat_error_mm": float(live["preseat_mm"])
            - expected_preseat_mm,
            "expected_x_mm": expected_x_mm,
            "observed_x_mm": float(live["source_x_mm"]),
            "x_error_mm": float(live["source_x_mm"]) - expected_x_mm,
            "transverse_y_mm": float(live["transverse_y_mm"]),
            "orientation_error_rad": float(live["orientation_error_rad"]),
            "slider_q_mm": float(live["slider_q_mm"]),
            "source_q_max_mm": float(live["source_q_max_mm"]),
            "lead_x_gap_mm": lead_x_gap_mm,
            "lead_normal_clearance_mm": lead_x_gap_mm / math.sqrt(2.0),
            "cam_contact_count": cam_contact_count,
            "finite": bool(
                all(
                    math.isfinite(float(value))
                    for value in (
                        *live.values(),
                        expected_preseat_mm,
                        expected_x_mm,
                        lead_x_gap_mm,
                    )
                )
                and np.all(np.isfinite(q_error))
            ),
        }
        self.core_capture_free_space_samples.append(sample)
        self.core_capture_free_space_phase_counts[action.name] += 1

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
        geom_a = self.geom_names[int(contact.geom[0])]
        geom_b = self.geom_names[int(contact.geom[1])]
        expected = {
            f"qc_col_pogo_{signal}_plunger",
            f"{tool}_pad_{signal}_collision",
        }
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
        signed_pad_axial_m = float(offset @ pad_axis)
        expected_signed_pad_axial_m = (
            -float(self.model.geom_size[pad_id, 1]) - float(contact.dist) / 2.0
        )
        contact_normal = np.asarray(contact.frame[:3], dtype=float)
        plunger_id = int(self.model.geom(f"qc_col_pogo_{signal}_plunger").id)
        if int(contact.geom[0]) == plunger_id:
            normal_from_plunger_to_pad = contact_normal
        elif int(contact.geom[1]) == plunger_id:
            normal_from_plunger_to_pad = -contact_normal
        else:
            return False
        plunger_rotation = np.asarray(
            self.data.geom_xmat[plunger_id], dtype=float
        ).reshape(3, 3)
        plunger_positive_z = plunger_rotation[:, 2]
        plunger_center = np.asarray(self.data.geom_xpos[plunger_id], dtype=float)
        plunger_offset = np.asarray(contact.pos, dtype=float) - plunger_center
        plunger_axial = float(plunger_offset @ plunger_positive_z)
        plunger_radial_m = float(
            np.linalg.norm(
                plunger_offset - plunger_axial * plunger_positive_z
            )
        )
        return (
            # The physical bus witness is the exposed underside of the pad,
            # not its centre plane or copper back face.  MuJoCo locates a
            # penetrating contact halfway between the two original surfaces.
            abs(signed_pad_axial_m - expected_signed_pad_axial_m)
            <= CONTACT_NUMERICAL_EPSILON_M
            and plunger_radial_m
            <= qc.POGO_PLUNGER_DIAMETER_M / 2.0 + CONTACT_NUMERICAL_EPSILON_M
            # This explicit pair is directed from the source plunger toward
            # the target pad.  Edge or reversed normals cannot establish an
            # axial bus contact even when identity and point are otherwise
            # correct.
            and np.all(np.isfinite(normal_from_plunger_to_pad))
            and float(np.linalg.norm(normal_from_plunger_to_pad)) >= 0.999
            and float(normal_from_plunger_to_pad @ plunger_positive_z) >= 0.999
        )

    def _pogo_contact_signals(self, tool: str) -> set[str]:
        observed: set[str] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = frozenset(
                self.geom_names[int(geom_id)] for geom_id in contact.geom
            )
            contract = self.pogo_pair_contract.get(pair)
            if contract is None or contract[0] != tool:
                continue
            signal = contract[1]
            if self._matching_pogo_contact_is_valid(contact, tool, signal):
                observed.add(signal)
        return observed

    def _dock_stop_contact_is_valid(self, contact: mujoco.MjContact, tool: str) -> bool:
        geom_a = self.geom_names[int(contact.geom[0])]
        geom_b = self.geom_names[int(contact.geom[1])]
        stop_names = [
            name
            for name in (geom_a, geom_b)
            if name in self.dock_stop_names_by_tool[tool]
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

    def _core_keeper_aligned_approach_is_valid(self) -> bool:
        """Return whether the arm is inside the reviewed final guide corridor."""

        action = self.current_action
        if (
            action is None
            or action.name != "gripper_to_capture"
            or not action.joint_waypoints
            or not self._equality_active("dock_gripper_hold")
            or self._equality_active("attach_gripper")
        ):
            return False
        robot_site = self.data.site("robot_mating_face")
        dock_body = self.data.body("dock_gripper")
        dock_rotation = np.asarray(dock_body.xmat, dtype=float).reshape(3, 3)
        local_offset = dock_rotation.T @ (
            np.asarray(robot_site.xpos, dtype=float)
            - np.asarray(dock_body.xpos, dtype=float)
        )
        robot_rotation = np.asarray(robot_site.xmat, dtype=float).reshape(3, 3)
        cosine = float((np.trace(dock_rotation.T @ robot_rotation) - 1.0) / 2.0)
        orientation_error = math.acos(max(-1.0, min(1.0, cosine)))
        return bool(
            float(np.linalg.norm(local_offset[:2])) <= CAM_RELIEF_CORRIDOR_M
            and orientation_error <= CAPTURE_ORIENTATION_TOLERANCE_RAD
        )

    def _core_keeper_oriented_normal_dock_local(
        self,
        contact: mujoco.MjContact,
        contract: dict[str, Any],
        dock_rotation: np.ndarray,
    ) -> np.ndarray:
        normal_world = np.asarray(contact.frame[:3], dtype=float)
        runtime_pair = list(contract["runtime_pair"])
        contact_geom0 = self.geom_names[int(contact.geom[0])]
        if contact_geom0 != runtime_pair[0]:
            normal_world = -normal_world
        normal_local = dock_rotation.T @ normal_world
        length = float(np.linalg.norm(normal_local))
        if not math.isfinite(length) or length <= 0.0:
            return np.full(3, np.nan, dtype=float)
        return normal_local / length

    @staticmethod
    def _core_keeper_normal_is_valid(
        contact: mujoco.MjContact,
        contract: dict[str, Any],
        dock_rotation: np.ndarray,
    ) -> bool:
        normal_local = dock_rotation.T @ np.asarray(contact.frame[:3], dtype=float)
        length = float(np.linalg.norm(normal_local))
        if not math.isfinite(length) or length <= 0.0:
            return False
        normal_local = normal_local / length
        if contract.get("expected_local_normal_subspace") == "dock_xz_plane":
            if float(np.linalg.norm(normal_local[[0, 2]])) < CORE_KEEPER_MIN_NORMAL_ALIGNMENT:
                return False
            product = float(normal_local[0] * normal_local[2])
            if contract["source_pair"][1] == "left_lower_rail":
                return product >= -1.0e-12
            if contract["source_pair"][1] == "right_lower_rail":
                return product <= 1.0e-12
            return False
        axis_index = 0 if contract["expected_local_normal_axis"] == "x" else 2
        return abs(float(normal_local[axis_index])) >= CORE_KEEPER_MIN_NORMAL_ALIGNMENT

    def _core_keeper_normal_alignment(
        self,
        contact: mujoco.MjContact,
        contract: dict[str, Any],
        dock_rotation: np.ndarray,
    ) -> float:
        normal_local = self._core_keeper_oriented_normal_dock_local(
            contact, contract, dock_rotation
        )
        if not np.all(np.isfinite(normal_local)):
            return 0.0
        if contract.get("expected_local_normal_subspace") == "dock_xz_plane":
            return float(np.linalg.norm(normal_local[[0, 2]]))
        axis_index = 0 if contract["expected_local_normal_axis"] == "x" else 2
        return abs(float(normal_local[axis_index]))

    def _core_keeper_contact_is_valid(self, contact: mujoco.MjContact) -> bool:
        geom_names = {
            self.geom_names[int(contact.geom[0])],
            self.geom_names[int(contact.geom[1])],
        }
        contract = next(
            (
                record
                for record in CORE_KEEPER_CONTACT_CONTRACT
                if geom_names == set(record["runtime_pair"])
            ),
            None,
        )
        if contract is None:
            return False
        if float(contact.dist) < -CORE_KEEPER_MAX_PENETRATION_MM * 1.0e-3:
            return False
        source_component = str(contract["source_pair"][0])
        if source_component == "robot_plate":
            if not (
                self._capture_pose_is_valid("gripper")
                or self._core_keeper_aligned_approach_is_valid()
            ):
                return False
        else:
            dock_separation = float(
                np.linalg.norm(
                    self.data.body("dock_gripper").xpos
                    - self.data.body("tool_gripper").xpos
                )
            )
            if dock_separation > CAPTURE_POSITION_TOLERANCE_M:
                return False
        if not (
            self._equality_active("dock_gripper_hold")
            or self._equality_active("attach_gripper")
        ):
            return False
        dock_rotation = np.asarray(
            self.data.body("dock_gripper").xmat, dtype=float
        ).reshape(3, 3)
        alignment = self._core_keeper_normal_alignment(
            contact, contract, dock_rotation
        )
        return bool(
            alignment >= CORE_KEEPER_MIN_NORMAL_ALIGNMENT
            and self._core_keeper_normal_is_valid(contact, contract, dock_rotation)
        )

    def _core_robot_tool_wing_mating_contact_is_valid(
        self, contact: mujoco.MjContact
    ) -> bool:
        expected = {
            "qc_col_robot_plate_electrical_wing_edge__keeper_land",
            "matcha_col_gripper_plate_electrical_wing_edge__mating_land__keeper_land",
        }
        names = {
            self.geom_names[int(contact.geom[0])],
            self.geom_names[int(contact.geom[1])],
        }
        if names != expected:
            return False
        if not (
            -float(contact.dist) <= CORE_KEEPER_MAX_PENETRATION_MM * 1.0e-3
            and self._capture_pose_is_valid("gripper")
            and (
                self._equality_active("dock_gripper_hold")
                or self._equality_active("attach_gripper")
            )
        ):
            return False
        tool_rotation = np.asarray(
            self.data.body("tool_gripper").xmat, dtype=float
        ).reshape(3, 3)
        return (
            abs(float(np.asarray(contact.frame[:3]) @ tool_rotation[:, 2]))
            >= CORE_KEEPER_MIN_NORMAL_ALIGNMENT
        )

    def _dock_local_point_mm(self, point_world: np.ndarray) -> list[float]:
        dock = self.data.body("dock_gripper")
        rotation = np.asarray(dock.xmat, dtype=float).reshape(3, 3)
        local = rotation.T @ (
            np.asarray(point_world, dtype=float) - np.asarray(dock.xpos, dtype=float)
        )
        return [float(value * 1000.0) for value in local]

    def _runtime_geom_edges_dock_local_mm(self, geom_id: int) -> list[np.ndarray]:
        """Return the live topological edges of one keeper geom in dock space.

        The optional lower-keeper witness is a true edge/edge tangency.  A
        proximity window over vertices is not sufficient here: the rounded
        tool-plate corner has vertices only a few microns from the straight
        source edge, and joining unrelated nearby vertices invents a locus
        that is not on either collision surface.  Preserve the compiled box or
        mesh topology and rank complete edges instead.
        """

        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            vertices_local = np.asarray(
                [
                    (sx * size[0], sy * size[1], sz * size[2])
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                    for sz in (-1.0, 1.0)
                ],
                dtype=float,
            )
            # The comprehension above orders sign triplets lexicographically;
            # box edges differ in exactly one sign bit.
            edge_indices = [
                (index, index ^ bit)
                for index in range(8)
                for bit in (1, 2, 4)
                if index < (index ^ bit)
            ]
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            vertex_start = int(self.model.mesh_vertadr[mesh_id])
            vertex_count = int(self.model.mesh_vertnum[mesh_id])
            vertices_local = np.asarray(
                self.model.mesh_vert[
                    vertex_start : vertex_start + vertex_count
                ],
                dtype=float,
            )
            face_start = int(self.model.mesh_faceadr[mesh_id])
            face_count = int(self.model.mesh_facenum[mesh_id])
            faces = np.asarray(
                self.model.mesh_face[face_start : face_start + face_count],
                dtype=np.int64,
            )
            edge_indices = sorted(
                {
                    tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                    for face in faces
                    for index in range(3)
                    if int(face[index]) != int(face[(index + 1) % 3])
                }
            )
        else:
            raise RuntimeError("keeper line fallback requires box/mesh geometry")
        geom_rotation = np.asarray(
            self.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        vertices_world = (
            np.asarray(self.data.geom_xpos[geom_id], dtype=float)
            + vertices_local @ geom_rotation.T
        )
        dock = self.data.body("dock_gripper")
        dock_rotation = np.asarray(dock.xmat, dtype=float).reshape(3, 3)
        vertices_dock_mm = (
            (vertices_world - np.asarray(dock.xpos, dtype=float)) @ dock_rotation
        ) * 1000.0
        return [vertices_dock_mm[list(edge)] for edge in edge_indices]

    def _analytic_keeper_line_closest_points_mm(
        self, geom_ids: list[int], source_witness: dict[str, Any]
    ) -> list[list[float]]:
        fixed = source_witness["fixed_coordinates_mm"]
        fixed_x = float(fixed["x"])
        fixed_z = float(fixed["z"])
        candidate_edges: list[np.ndarray] = []
        for geom_id in geom_ids:
            edges = [
                edge
                for edge in self._runtime_geom_edges_dock_local_mm(geom_id)
                if abs(float(edge[1, 1] - edge[0, 1])) > 1.0e-12
            ]
            if not edges:
                raise RuntimeError("could not reconstruct keeper line edge")
            # Minimize the worst endpoint error, not a pooled vertex error.
            # This selects the complete straight support edge and cannot join
            # two unrelated vertices from neighboring rounded-corner chords.
            def edge_key(edge: np.ndarray) -> tuple[float, float, float]:
                scores = (edge[:, 0] - fixed_x) ** 2 + (
                    edge[:, 2] - fixed_z
                ) ** 2
                return (
                    float(np.max(scores)),
                    float(np.mean(scores)),
                    -abs(float(edge[1, 1] - edge[0, 1])),
                )

            candidate_edges.append(min(edges, key=edge_key))
        lower = max(
            float(source_witness["line_axis_bounds_mm"][0]),
            *(float(np.min(edge[:, 1])) for edge in candidate_edges),
        )
        upper = min(
            float(source_witness["line_axis_bounds_mm"][1]),
            *(float(np.max(edge[:, 1])) for edge in candidate_edges),
        )
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise RuntimeError("live keeper edges have no source-line overlap")
        witness_y = (lower + upper) / 2.0
        points: list[list[float]] = []
        for edge in candidate_edges:
            low = edge[int(np.argmin(edge[:, 1]))]
            high = edge[int(np.argmax(edge[:, 1]))]
            span = float(high[1] - low[1])
            alpha = 0.5 if abs(span) <= 1.0e-12 else (witness_y - low[1]) / span
            point = low + alpha * (high - low)
            points.append([float(value) for value in point])
        return points

    @staticmethod
    def _source_witness_point_error_mm(
        point_mm: list[float], source_witness: dict[str, Any]
    ) -> float:
        coordinates = {axis: float(point_mm[index]) for index, axis in enumerate("xyz")}
        if source_witness["kind"] == "line_tangency":
            axis = str(source_witness["line_axis"])
            fixed = source_witness["fixed_coordinates_mm"]
            squared_error = sum(
                (coordinates[name] - float(value)) ** 2
                for name, value in fixed.items()
            )
            lower, upper = (float(value) for value in source_witness["line_axis_bounds_mm"])
            axis_error = max(lower - coordinates[axis], 0.0, coordinates[axis] - upper)
            return math.sqrt(squared_error + axis_error * axis_error)
        if source_witness["kind"] != "planar_face_tangency":
            raise ValueError(f"unsupported source witness {source_witness['kind']!r}")
        normal_axis = str(source_witness["normal_axis"])
        normal_error = abs(
            coordinates[normal_axis] - float(source_witness["plane_coordinate_mm"])
        )
        tangent_error_sq = 0.0
        for axis, bounds in source_witness["tangential_bounds_mm"].items():
            lower, upper = (float(value) for value in bounds)
            delta = max(lower - coordinates[axis], 0.0, coordinates[axis] - upper)
            tangent_error_sq += delta * delta
        boundary_error = 0.0
        constraint = source_witness.get("source_boundary_constraint")
        if constraint is not None:
            if constraint.get("kind") != "rounded_rectangle":
                raise ValueError("unsupported keeper source-boundary constraint")
            half_width = float(constraint["half_width_mm"])
            half_height = float(constraint["half_height_mm"])
            radius = float(constraint["corner_radius_mm"])
            qx = abs(coordinates["x"]) - (half_width - radius)
            qy = abs(coordinates["y"]) - (half_height - radius)
            outside = math.hypot(max(qx, 0.0), max(qy, 0.0)) - radius
            boundary_error = max(0.0, outside)
        return math.sqrt(
            normal_error * normal_error
            + tangent_error_sq
            + boundary_error * boundary_error
        )

    def _core_keeper_contact_report(
        self, phase: str = "pre_attach_seated_keeper_capture"
    ) -> dict[str, Any]:
        """Return exact live/analytic evidence for all five core keepers."""

        position_error_m, angle_error_rad = self._tool_pose_error("gripper")
        dock_rotation = np.asarray(
            self.data.body("dock_gripper").xmat, dtype=float
        ).reshape(3, 3)
        records: list[dict[str, Any]] = []
        for contract in CORE_KEEPER_CONTACT_CONTRACT:
            runtime_pair = list(contract["runtime_pair"])
            source_witness = copy.deepcopy(contract["source_witness"])
            geom_ids = [int(self.model.geom(name).id) for name in runtime_pair]
            matching_contacts = [
                self.data.contact[index]
                for index in range(self.data.ncon)
                if {int(value) for value in self.data.contact[index].geom}
                == set(geom_ids)
            ]
            from_to = np.full(6, np.nan, dtype=np.float64)
            signed_distance_m = float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom_ids[0],
                    geom_ids[1],
                    0.01,
                    from_to,
                )
            )
            raw_closest_points = (
                [
                    self._dock_local_point_mm(from_to[:3]),
                    self._dock_local_point_mm(from_to[3:]),
                ]
                if np.all(np.isfinite(from_to))
                else []
            )
            raw_closest_points_valid = bool(
                raw_closest_points
                and all(
                    self._source_witness_point_error_mm(point, source_witness)
                    <= float(source_witness["point_tolerance_mm"])
                    for point in raw_closest_points
                )
            )
            if (
                source_witness["kind"] == "line_tangency"
                and not raw_closest_points_valid
                and not matching_contacts
                and abs(signed_distance_m) <= CONTACT_NUMERICAL_EPSILON_M
            ):
                closest_points_dock_local_mm = (
                    self._analytic_keeper_line_closest_points_mm(
                        geom_ids, source_witness
                    )
                )
                closest_point_method = (
                    "analytic_box_box_line_tangency_from_live_geom_transforms"
                )
                mujoco_from_to_valid = False
            else:
                closest_points_dock_local_mm = raw_closest_points
                closest_point_method = "mujoco_mj_geomDistance"
                mujoco_from_to_valid = raw_closest_points_valid
            contact_points_dock_local_mm = [
                self._dock_local_point_mm(np.asarray(contact.pos, dtype=float))
                for contact in matching_contacts
            ]
            max_penetration_mm = max(
                (
                    max(0.0, -float(contact.dist)) * 1000.0
                    for contact in matching_contacts
                ),
                default=max(0.0, -signed_distance_m) * 1000.0,
            )
            if matching_contacts:
                oriented_normals = [
                    self._core_keeper_oriented_normal_dock_local(
                        contact, contract, dock_rotation
                    )
                    for contact in matching_contacts
                ]
                normals_valid = all(
                    self._core_keeper_normal_is_valid(
                        contact, contract, dock_rotation
                    )
                    for contact in matching_contacts
                )
                alignments = [
                    self._core_keeper_normal_alignment(
                        contact, contract, dock_rotation
                    )
                    for contact in matching_contacts
                ]
                witness_method = "live_mujoco_contact"
            else:
                # The two source lower-rail contacts are exact edge tangencies,
                # which MuJoCo need not place in ncon.  The signed primitive
                # distance plus both live geom frames still proves the same
                # active physical geometry without inventing contact force.
                alignments = [1.0]
                oriented_normals = []
                normals_valid = True
                witness_method = (
                    "live_mujoco_signed_geom_distance_and_source_semantics"
                )
            minimum_alignment = min(alignments)
            source_pair = list(contract["source_pair"])
            live_contact_required = (
                source_pair[1].endswith("upper_rail")
                or source_pair[0] == "robot_plate"
            )
            witness_points = (
                contact_points_dock_local_mm
                if matching_contacts
                else closest_points_dock_local_mm
            )
            maximum_source_witness_error_mm = max(
                (
                    self._source_witness_point_error_mm(point, source_witness)
                    for point in witness_points
                ),
                default=1.0e9,
            )
            passed = bool(
                math.isfinite(signed_distance_m)
                and signed_distance_m * 1000.0 <= CORE_KEEPER_MAX_SEPARATION_MM
                and max_penetration_mm <= CORE_KEEPER_MAX_PENETRATION_MM
                and minimum_alignment >= CORE_KEEPER_MIN_NORMAL_ALIGNMENT
                and normals_valid
                and (matching_contacts or not live_contact_required)
                and maximum_source_witness_error_mm
                <= float(source_witness["point_tolerance_mm"])
            )
            record = {
                "source_pair": source_pair,
                "runtime_pair": runtime_pair,
                "source_witness": source_witness,
                "contact_count": len(matching_contacts),
                "signed_distance_mm": signed_distance_m * 1000.0,
                "max_penetration_mm": max_penetration_mm,
                "closest_points_dock_local_mm": closest_points_dock_local_mm,
                "closest_point_method": closest_point_method,
                "mujoco_from_to_valid": mujoco_from_to_valid,
                "contact_points_dock_local_mm": contact_points_dock_local_mm,
                "contact_normals_from_runtime_pair_0_to_1_dock_local": [
                    [float(value) for value in normal]
                    for normal in oriented_normals
                ],
                "maximum_contact_point_source_witness_error_mm": (
                    maximum_source_witness_error_mm
                ),
                "witness_method": witness_method,
                "passed": passed,
            }
            if "expected_local_normal_subspace" in contract:
                record["expected_local_normal_subspace"] = contract[
                    "expected_local_normal_subspace"
                ]
                record["minimum_normal_subspace_alignment"] = minimum_alignment
            else:
                record["expected_local_normal_axis"] = contract[
                    "expected_local_normal_axis"
                ]
                record["minimum_normal_alignment"] = minimum_alignment
            records.append(record)
        stop_contact_count = sum(
            1
            for index in range(self.data.ncon)
            if any(
                qc.is_dock_stop_collision_name(
                    "gripper", self.geom_names[int(geom_id)]
                )
                for geom_id in self.data.contact[index].geom
            )
        )
        dock_hold_active = self._equality_active("dock_gripper_hold")
        attach_equality_active = self._equality_active("attach_gripper")
        pogo_signals = sorted(self._pogo_contact_signals("gripper"))
        observed_tool_id = self._tool_id_from_compiled_bus("gripper")
        expected_tool_id = ALL_TOOL_IDS["gripper"]
        tool_identity_verified = observed_tool_id == expected_tool_id
        phase_state_valid = bool(
            dock_hold_active
            and (
                not attach_equality_active
                if phase == "pre_attach_seated_keeper_capture"
                else True
            )
        )
        return {
            "passed": bool(
                position_error_m <= CAPTURE_POSITION_TOLERANCE_M
                and angle_error_rad <= CAPTURE_ORIENTATION_TOLERANCE_RAD
                and stop_contact_count == 0
                and phase_state_valid
                and pogo_signals == sorted(qc.SIGNALS)
                and tool_identity_verified
                and all(record["passed"] for record in records)
            ),
            "phase": phase,
            "witness_sim_time_s": float(self.data.time),
            "pose_position_error_mm": position_error_m * 1000.0,
            "pose_angle_error_deg": math.degrees(angle_error_rad),
            "dock_hold_active": dock_hold_active,
            "attach_equality_active": attach_equality_active,
            "stop_contact_count": stop_contact_count,
            "pogo_signals": pogo_signals,
            "observed_tool_id": observed_tool_id,
            "expected_tool_id": expected_tool_id,
            "tool_identity_verified": tool_identity_verified,
            "records": records,
        }

    def _mating_land_contact_is_valid(
        self, contact: mujoco.MjContact, tool: str
    ) -> bool:
        geom_a = self.geom_names[int(contact.geom[0])]
        geom_b = self.geom_names[int(contact.geom[1])]
        matching_robot_lands = self.robot_mating_land_names.intersection(
            {geom_a, geom_b}
        )
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

    def _interface_guard(self, tool: str) -> bool:
        signals = self._pogo_contact_signals(tool)
        id_matches = self._tool_id_from_compiled_bus(tool) == ALL_TOOL_IDS[tool]
        dock_or_attach = (
            (
                self._core_keeper_contact_report(
                    phase=(
                        "attached_dock_hold_keeper_verify"
                        if self._equality_active("attach_gripper")
                        else "pre_attach_seated_keeper_capture"
                    )
                )["passed"]
                if tool == "gripper"
                else self._dock_stop_is_seated(tool)
            )
            if self._equality_active(f"dock_{tool}_hold")
            else self._equality_active(f"attach_{tool}")
        )
        return (
            signals == set(qc.SIGNALS)
            and id_matches
            and dock_or_attach
            and self._capture_pose_is_valid(tool)
        )

    def _allowed_penetrating_contact(
        self,
        contact: mujoco.MjContact,
        contact_index: int | None = None,
    ) -> bool:
        if (
            contact_index is not None
            and contact_index in self.core_cam_tab_allowed_contact_indices
        ):
            return True
        geom_a = self.geom_names[int(contact.geom[0])]
        geom_b = self.geom_names[int(contact.geom[1])]
        pair = frozenset({geom_a, geom_b})
        if pair in self.support_contact_pairs:
            return True
        if self._core_keeper_contact_is_valid(contact):
            return True
        if self._core_robot_tool_wing_mating_contact_is_valid(contact):
            return True
        pogo_contract = self.pogo_pair_contract.get(pair)
        if pogo_contract is not None:
            return self._matching_pogo_contact_is_valid(
                contact, pogo_contract[0], pogo_contract[1]
            )
        for tool in ALL_TOOL_IDS:
            if pair.intersection(self.dock_stop_names_by_tool[tool]) and (
                self._dock_stop_contact_is_valid(contact, tool)
            ):
                return True
            if self._mating_land_contact_is_valid(contact, tool):
                return True
        return False

    def _audit_contacts(self) -> None:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            penetration = -float(contact.dist)
            if penetration <= CONTACT_NUMERICAL_EPSILON_M:
                continue
            if self._allowed_penetrating_contact(contact, contact_index):
                continue
            geom_a = self.geom_names[int(contact.geom[0])]
            geom_b = self.geom_names[int(contact.geom[1])]
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

    def _record_route_alignment(self, action: WorkflowAction) -> None:
        if not action.joint_waypoints or action.tool is None:
            return
        robot_site = self.data.site("robot_mating_face")
        dock_body = self.data.body(f"dock_{action.tool}")
        dock_rotation = np.asarray(dock_body.xmat, dtype=float).reshape(3, 3)
        local_offset = dock_rotation.T @ (
            np.asarray(robot_site.xpos, dtype=float)
            - np.asarray(dock_body.xpos, dtype=float)
        )
        self.max_route_lateral_deviation_m = max(
            self.max_route_lateral_deviation_m,
            float(np.linalg.norm(local_offset[:2])),
        )
        robot_rotation = np.asarray(robot_site.xmat, dtype=float).reshape(3, 3)
        cosine = float((np.trace(dock_rotation.T @ robot_rotation) - 1.0) / 2.0)
        orientation_error = math.acos(max(-1.0, min(1.0, cosine)))
        self.max_route_orientation_error_rad = max(
            self.max_route_orientation_error_rad, orientation_error
        )

    def _audit_core_capture_source_corridor(
        self, action: WorkflowAction
    ) -> None:
        """Fail closed on live FK drift from the source cam p/X law."""

        audited_actions = {
            "gripper_capture_axial_open_side",
            "gripper_capture_coupled_recenter",
            "gripper_capture_centered_final",
        }
        if (
            not self.core_capture_source_corridor_armed
            or action.name not in audited_actions
        ):
            return
        self.core_capture_source_corridor_audited_substeps += 1
        self.core_capture_source_corridor_phase_counts[action.name] += 1
        mating = self.data.site("robot_mating_face")
        dock = self.data.body("dock_gripper")
        dock_rotation = np.asarray(
            dock.xmat, dtype=np.float64
        ).reshape(3, 3)
        local_position_mm = (
            np.asarray(mating.xpos, dtype=np.float64)
            - np.asarray(dock.xpos, dtype=np.float64)
        ) @ dock_rotation * 1000.0
        preseat_mm = -float(local_position_mm[2])
        expected_x_mm = _core_capture_source_x_mm(max(0.0, preseat_mm))
        observed_x_mm = float(local_position_mm[0])
        signed_error_mm = observed_x_mm - expected_x_mm
        absolute_error_mm = abs(signed_error_mm)
        if absolute_error_mm >= self.core_capture_source_corridor_max_error_mm:
            self.core_capture_source_corridor_max_error_mm = absolute_error_mm
            self.core_capture_source_corridor_witness = {
                "action": action.name,
                "sim_time_s": float(self.data.time),
                "physics_substep_count": int(self.physics_substep_count),
                "preseat_mm": preseat_mm,
                "observed_x_mm": observed_x_mm,
                "expected_source_x_mm": expected_x_mm,
                "signed_error_mm": signed_error_mm,
                "absolute_error_mm": absolute_error_mm,
            }
        if absolute_error_mm > CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM:
            self._abort("core_capture_source_corridor_violation")

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
            self._record_route_alignment(action)
            self._record_core_capture_free_space_tracking(action)
            self._audit_core_capture_cam_tab_contacts(action)
            self._audit_core_capture_source_corridor(action)
            if self.abort_reason is not None:
                return
            if action.kind == "axial_disengage":
                self._record_lock_stroke_contacts()
            if (
                self.attached_tool == "gripper"
                and not self._equality_active("dock_gripper_hold")
                and self._equality_active("attach_gripper")
            ):
                self._record_slider_state(action)
            self._audit_contacts()
            self._record_actuator_loads()
            if self.abort_reason is not None:
                return
            if (
                action.kind in {"capture", "lock_verify", "release_verify"}
                and (self.attachment_verified or self.locked)
                and self.attached_tool is not None
                and self._equality_active(f"attach_{self.attached_tool}")
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

    def _advance_action(self, event: str, **evidence: Any) -> None:
        action = self.actions[self.action_index]
        journal_record = {
            "event": event,
            "action": action.name,
            "sim_time_s": float(self.data.time),
            "physics_substep_count": int(self.physics_substep_count),
        }
        journal_record.update(evidence)
        self.journal.append(journal_record)
        if (
            event == "move_complete"
            and action.name == "gripper_capture_lateral_align"
        ):
            self.core_capture_source_corridor_armed = True
        elif (
            event == "move_complete"
            and action.name == "gripper_capture_centered_final"
        ):
            self.core_capture_source_corridor_armed = False
        self.action_index += 1
        if self.action_index >= len(self.actions):
            self.completed = True
            self.motion_stopped = True
            self.success = (
                self.attachment_verified
                and self.attached_tool == "gripper"
                and not self.locked
                and not self.physical_lock_confirmed
                and all(
                    count > 0
                    for count in self.core_cam_tab_functional_role_counts.values()
                )
                and self.core_cam_tab_rejected_contact_count == 0
                and self.forbidden_contact_count == 0
                and self.abort_reason is None
            )
            return
        self.action_started_s = float(self.data.time)
        self.move_endpoint_dwell_ticks = 0
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

    def _core_capture_route_endpoint_evidence(
        self, action: WorkflowAction
    ) -> dict[str, Any]:
        """Return live dock-frame pose evidence at one route endpoint."""

        if action.name not in CORE_CAPTURE_ROUTE_ENDPOINTS_MM:
            raise ValueError(f"not a core capture route action: {action.name}")
        preseat_mm, source_x_mm = CORE_CAPTURE_ROUTE_ENDPOINTS_MM[action.name]
        dock = self.data.body("dock_gripper")
        mating = self.data.site("robot_mating_face")
        dock_rotation = np.asarray(dock.xmat, dtype=np.float64).reshape(3, 3)
        local_position = (
            np.asarray(mating.xpos, dtype=np.float64)
            - np.asarray(dock.xpos, dtype=np.float64)
        ) @ dock_rotation
        expected_position = np.asarray(
            [source_x_mm, 0.0, -preseat_mm], dtype=np.float64
        ) * 0.001
        position_error_m = float(
            np.linalg.norm(local_position - expected_position)
        )
        mating_rotation = np.asarray(
            mating.xmat, dtype=np.float64
        ).reshape(3, 3)
        relative_rotation = dock_rotation.T @ mating_rotation
        orientation_error_rad = float(
            math.acos(
                np.clip(
                    (float(np.trace(relative_rotation)) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        return {
            "action": action.name,
            "target_preseat_mm": preseat_mm,
            "target_source_x_mm": source_x_mm,
            "observed_preseat_mm": -float(local_position[2]) * 1000.0,
            "observed_x_mm": float(local_position[0]) * 1000.0,
            "source_x_error_mm": (
                float(local_position[0]) * 1000.0 - source_x_mm
            ),
            "observed_transverse_y_mm": float(local_position[1]) * 1000.0,
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
            "physics_substep_count": int(self.physics_substep_count),
            "sim_time_s": float(self.data.time),
        }

    def _command_move(self, action: WorkflowAction, elapsed_s: float) -> None:
        if action.target_q is None:
            self._abort("move_missing_target")
            return
        target = np.asarray(action.target_q, dtype=float)
        alpha = min(1.0, max(0.0, elapsed_s / action.duration_s))
        smooth = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
        self.current_move_command_smooth = float(smooth)
        if action.joint_waypoints:
            route = np.asarray(
                (tuple(self.action_start_q), *action.joint_waypoints), dtype=float
            )
            route_position = smooth * (len(route) - 1)
            segment = min(int(math.floor(route_position)), len(route) - 2)
            segment_alpha = route_position - segment
            command = (
                route[segment]
                + segment_alpha * (route[segment + 1] - route[segment])
            )
        else:
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
            if action.name in CORE_CAPTURE_ROUTE_ACTION_NAMES:
                endpoint_evidence = (
                    self._core_capture_route_endpoint_evidence(action)
                )
                position_error_m = float(endpoint_evidence["position_error_m"])
                orientation_error_rad = float(
                    endpoint_evidence["orientation_error_rad"]
                )
                endpoint_valid = (
                    target_error <= CORE_CAPTURE_ROUTE_ENDPOINT_Q_ERROR_RAD
                    and speed <= CORE_CAPTURE_ROUTE_ENDPOINT_QVEL_RAD_S
                    and position_error_m
                    <= CORE_CAPTURE_ROUTE_ENDPOINT_POSITION_ERROR_M
                    and orientation_error_rad
                    <= CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD
                    and abs(float(endpoint_evidence["source_x_error_mm"]))
                    <= CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
                )
                if endpoint_valid:
                    self.move_endpoint_dwell_ticks += 1
                else:
                    self.move_endpoint_dwell_ticks = 0
                if (
                    self.move_endpoint_dwell_ticks
                    >= CORE_CAPTURE_ROUTE_ENDPOINT_DWELL_TICKS
                ):
                    self._advance_action(
                        "move_complete",
                        endpoint_q_error_rad=target_error,
                        endpoint_max_abs_qvel_rad_s=speed,
                        endpoint_fk_position_error_m=position_error_m,
                        endpoint_fk_orientation_error_rad=(
                            orientation_error_rad
                        ),
                        endpoint_dwell_ticks=(
                            CORE_CAPTURE_ROUTE_ENDPOINT_DWELL_TICKS
                        ),
                        route_endpoint_evidence=endpoint_evidence,
                    )
                return
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
        if action.tool == "gripper":
            keeper_report = self._core_keeper_contact_report()
            if not keeper_report["passed"]:
                self._abort("core_keeper_capture_not_verified")
                return
            self.core_keeper_capture_report = copy.deepcopy(keeper_report)
            self.core_keeper_capture_verified = True
        elif not self._dock_stop_is_seated(action.tool):
            self._abort("dock_stop_not_seated")
            return
        self.bus_connected = True
        self.handshake_achieved = True
        self.capture_pogo_signals = sorted(self._pogo_contact_signals(action.tool))
        self.capture_mating_pose_evidence = self._mating_world_pose_evidence()
        self.capture_robot_mating_rotation = np.asarray(
            self.data.site("robot_mating_face").xmat, dtype=float
        ).reshape(3, 3).copy()
        self.data.eq_active[self.model.equality(f"attach_{action.tool}").id] = 1
        self.attached_tool = action.tool
        self.lock_confirmation_phase = "captured_slider_still_unlocked"
        self._advance_action(
            "physical_capture_complete",
            physical_lock_confirmed=False,
            core_keeper_capture_verified=self.core_keeper_capture_verified,
            dock_hold_active=self._equality_active(f"dock_{action.tool}_hold"),
            attach_equality_active=self._equality_active(f"attach_{action.tool}"),
            pogo_signals=list(self.capture_pogo_signals),
            observed_tool_id=self._tool_id_from_compiled_bus(action.tool),
            expected_tool_id=ALL_TOOL_IDS[action.tool],
            tool_identity_verified=(
                self._tool_id_from_compiled_bus(action.tool)
                == ALL_TOOL_IDS[action.tool]
            ),
            **self.capture_mating_pose_evidence,
        )

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
        self.attachment_candidate_verified = bool(
            equality_active
            and self._interface_guard(action.tool)
            and (
                self.core_keeper_capture_verified
                if action.tool == "gripper"
                else self._dock_stop_is_seated(action.tool)
            )
        )
        if not self.attachment_candidate_verified:
            self._abort("physical_attachment_not_confirmed")
            return
        self.data.eq_active[self.model.equality(f"dock_{action.tool}_hold").id] = 0
        self.lock_confirmation_phase = "dock_released_slider_still_unlocked"
        self._advance_action(
            "dock_hold_released",
            physical_lock_confirmed=False,
            core_keeper_capture_verified=self.core_keeper_capture_verified,
            dock_hold_active=False,
            attach_equality_active=self._equality_active(f"attach_{action.tool}"),
        )

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
        self.attachment_verified = bool(
            self.attachment_candidate_verified
            and self._equality_active(f"attach_{action.tool}")
            and self._interface_guard(action.tool)
        )
        if not self.attachment_verified:
            self._abort("post_release_attachment_not_confirmed")
            return
        # The source cam still holds the positive-lock slider open at this
        # seated phase.  Physical lock remains false until source-axis rack
        # withdrawal, cam clearance and spring return are all modeled.
        self.locked = False
        self.physical_lock_confirmed = False
        self.lock_confirmation_phase = "dock_released_slider_still_unlocked"
        self._advance_action(
            "dock_release_verified",
            physical_lock_confirmed=False,
            attachment_verified=True,
        )

    def _command_axial_disengage(
        self, action: WorkflowAction, elapsed_s: float
    ) -> None:
        """Hard-retired source-negative motion; never take a physics step."""

        del action, elapsed_s
        self._abort("retired_negative_z_lock_sequence")

    def _command_slider_return(self, action: WorkflowAction) -> None:
        """Hard-retired source-negative motion; never take a physics step."""

        del action
        self._abort("retired_negative_z_lock_sequence")

    def _command_physical_lock_confirm(self, action: WorkflowAction) -> None:
        """Hard-retired source-negative claim; physical lock stays false."""

        del action
        self._abort("retired_negative_z_lock_sequence")

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
        elif action.kind == "axial_disengage":
            self._command_axial_disengage(action, elapsed_s)
        elif action.kind == "slider_return":
            self._command_slider_return(action)
        elif action.kind == "physical_lock_confirm":
            self._command_physical_lock_confirm(action)
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
        route_endpoint_records = [
            copy.deepcopy(record)
            for record in self.journal
            if record.get("event") == "move_complete"
            and record.get("action") in CORE_CAPTURE_ROUTE_ACTION_NAMES
        ]
        completed_route_endpoint_actions = {
            str(record["action"]) for record in route_endpoint_records
        }
        corridor_observed = (
            self.core_capture_source_corridor_audited_substeps > 0
        )
        corridor_all_phases_observed = all(
            count > 0
            for count in self.core_capture_source_corridor_phase_counts.values()
        )
        route_all_endpoints_completed = (
            completed_route_endpoint_actions
            == CORE_CAPTURE_ROUTE_ACTION_NAMES
        )
        live_source_corridor_passed = bool(
            corridor_observed
            and corridor_all_phases_observed
            and route_all_endpoints_completed
            and self.core_capture_source_corridor_witness is not None
            and self.core_capture_source_corridor_max_error_mm
            <= CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
            and self.abort_reason is None
        )
        cam_contact_records = copy.deepcopy(self.core_cam_tab_contact_records)
        cam_contact_observed = len(cam_contact_records) > 0
        cam_all_phases_observed = all(
            count > 0 for count in self.core_cam_tab_phase_counts.values()
        )
        cam_functional_coverage = all(
            count > 0
            for count in self.core_cam_tab_functional_role_counts.values()
        )
        cam_contact_evidence_passed = bool(
            cam_contact_observed
            and cam_all_phases_observed
            and cam_functional_coverage
            and self.core_cam_tab_rejected_contact_count == 0
            and self.core_cam_tab_candidate_contact_count
            == len(cam_contact_records)
            and route_all_endpoints_completed
            and self.abort_reason is None
        )
        cam_penetrations = [
            float(record["penetration_mm"]) for record in cam_contact_records
        ]
        cam_locus_errors = [
            float(record["locus_error_mm"])
            for record in cam_contact_records
            if record["locus_error_mm"] is not None
        ]
        cam_normal_forces = [
            abs(float(record["contact_force_torque_6d"][0]))
            for record in cam_contact_records
            if record["contact_force_torque_6d"][0] is not None
        ]

        free_space_samples = copy.deepcopy(
            self.core_capture_free_space_samples
        )
        free_space_observed = len(free_space_samples) > 0
        free_space_all_phases_observed = all(
            count > 0
            for count in self.core_capture_free_space_phase_counts.values()
        )
        free_space_endpoint_actions = {
            str(record["action"])
            for record in route_endpoint_records
            if record.get("action") in CORE_CAM_TAB_FREE_SPACE_ACTIONS
        }
        free_space_all_endpoints_completed = free_space_endpoint_actions == set(
            CORE_CAM_TAB_FREE_SPACE_ACTIONS
        )
        maximum_free_q_error = max(
            (
                float(record["max_abs_q_tracking_error_rad"])
                for record in free_space_samples
            ),
            default=math.inf,
        )
        maximum_free_preseat_error = max(
            (abs(float(record["preseat_error_mm"])) for record in free_space_samples),
            default=math.inf,
        )
        maximum_free_x_error = max(
            (abs(float(record["x_error_mm"])) for record in free_space_samples),
            default=math.inf,
        )
        maximum_free_y = max(
            (abs(float(record["transverse_y_mm"])) for record in free_space_samples),
            default=math.inf,
        )
        maximum_free_orientation = max(
            (float(record["orientation_error_rad"]) for record in free_space_samples),
            default=math.inf,
        )
        minimum_free_lead_gap = min(
            (float(record["lead_x_gap_mm"]) for record in free_space_samples),
            default=-math.inf,
        )
        free_space_cam_contacts = sum(
            int(record["cam_contact_count"]) for record in free_space_samples
        )
        free_space_tracking_passed = bool(
            free_space_observed
            and free_space_all_phases_observed
            and free_space_all_endpoints_completed
            and all(bool(record["finite"]) for record in free_space_samples)
            and maximum_free_q_error <= CORE_CAPTURE_ROUTE_ENDPOINT_Q_ERROR_RAD
            and maximum_free_preseat_error <= 0.050
            and maximum_free_x_error <= CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
            and maximum_free_y <= 0.010
            and maximum_free_orientation
            <= CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD
            and minimum_free_lead_gap >= 0.0
            and free_space_cam_contacts == 0
            and self.abort_reason is None
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
            "core_keeper_capture_verified": self.core_keeper_capture_verified,
            "core_keeper_contact_report": self.core_keeper_capture_report,
            "attachment_candidate_verified": self.attachment_candidate_verified,
            "attachment_verified": self.attachment_verified,
            "physical_lock_confirmed": self.physical_lock_confirmed,
            "lock_candidate_verified": self.lock_candidate_verified,
            "locked": self.locked,
            "lock_confirmation_phase": self.lock_confirmation_phase,
            "minimum_source_axis_withdrawal_mm": (
                self.minimum_source_axis_withdrawal_mm
            ),
            "capture_pogo_signals": list(self.capture_pogo_signals),
            "capture_four_signal_bus_verified": (
                self.capture_pogo_signals == sorted(qc.SIGNALS)
            ),
            "source_axis_withdrawal_evidence": copy.deepcopy(
                self.source_axis_withdrawal_evidence
            ),
            "slider_return_evidence": copy.deepcopy(self.slider_return_evidence),
            "first_physical_lock_true_substep": (
                self.first_physical_lock_true_substep
            ),
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
            "route_alignment": {
                "method": (
                    "source_coupled_positive_lock_cam_four_phase_dense_fk_ik_waypoints"
                ),
                "runtime_contract_api": "core_capture_route_runtime_contract",
                "contract_identity_sha256": (
                    CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256
                ),
                "source_state_sha256": (
                    CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256
                ),
                "q_roster_sha256": CORE_CAPTURE_ROUTE_Q_SHA256,
                "embedded_state_bytes_sha256": (
                    CORE_CAPTURE_ROUTE_STATE_BYTES_SHA256
                ),
                "phase_actions": list(CORE_CAPTURE_ROUTE_PHASE_TIMING_S),
                "phase_endpoint_journal_evidence": route_endpoint_records,
                "completed_endpoint_actions": sorted(
                    completed_route_endpoint_actions
                ),
                "all_four_endpoints_completed": (
                    route_all_endpoints_completed
                ),
                "measured_max_lateral_deviation_m": (
                    self.max_route_lateral_deviation_m
                ),
                "measured_max_orientation_error_rad": (
                    self.max_route_orientation_error_rad
                ),
                "cam_relief_corridor_m": CAM_RELIEF_CORRIDOR_M,
                "live_source_corridor": {
                    "armed": self.core_capture_source_corridor_armed,
                    "observed": corridor_observed,
                    "audited_substeps": (
                        self.core_capture_source_corridor_audited_substeps
                    ),
                    "audited_substeps_by_phase": dict(
                        self.core_capture_source_corridor_phase_counts
                    ),
                    "all_three_phases_observed": (
                        corridor_all_phases_observed
                    ),
                    "maximum_absolute_error_mm": (
                        self.core_capture_source_corridor_max_error_mm
                    ),
                    "maximum_allowed_error_mm": (
                        CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
                    ),
                    "witness": copy.deepcopy(
                        self.core_capture_source_corridor_witness
                    ),
                    "violation_abort_reason": (
                        "core_capture_source_corridor_violation"
                    ),
                    "passed": live_source_corridor_passed,
                    "live_dynamics_authority": False,
                },
                "passed": (
                    self.max_route_lateral_deviation_m <= CAM_RELIEF_CORRIDOR_M
                    and live_source_corridor_passed
                ),
            },
            "core_cam_tab_contact_evidence": {
                "schema_version": "1.0",
                "evidence_kind": (
                    "real_mujoco_per_substep_capture_cam_tab_contacts"
                ),
                "runtime_contract_api": (
                    "core_cam_tab_contact_runtime_contract"
                ),
                "contract_identity_sha256": (
                    CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_SHA256
                ),
                "physics_timestep_s": float(self.model.opt.timestep),
                "observed": cam_contact_observed,
                "audited_substeps": self.core_cam_tab_audited_substeps,
                "audited_substeps_by_phase": dict(
                    self.core_cam_tab_phase_counts
                ),
                "all_four_capture_phases_observed": cam_all_phases_observed,
                "raw_contact_records": cam_contact_records,
                "raw_contact_records_sha256": _canonical_json_sha256(
                    cam_contact_records
                ),
                "candidate_contact_count": (
                    self.core_cam_tab_candidate_contact_count
                ),
                "rejected_contact_count": (
                    self.core_cam_tab_rejected_contact_count
                ),
                "functional_role_counts": dict(
                    self.core_cam_tab_functional_role_counts
                ),
                "functional_coverage_observed": cam_functional_coverage,
                "zero_contact_cannot_pass": True,
                "maximum_penetration_mm": max(
                    cam_penetrations, default=None
                ),
                "maximum_locus_error_mm": max(
                    cam_locus_errors, default=None
                ),
                "maximum_abs_normal_force_n": max(
                    cam_normal_forces, default=None
                ),
                "provisional_geometry_classification_passed": (
                    cam_contact_evidence_passed
                ),
                "contact_force_authority": False,
                "friction_coefficient_authority": False,
                "dynamics_authority": False,
                "post_capture_negative_z_and_slider_return_excluded": True,
                "passed": cam_contact_evidence_passed,
                "release_ready": False,
            },
            "core_capture_free_space_tracking_evidence": {
                "schema_version": "1.0",
                "evidence_kind": (
                    "real_mujoco_per_substep_free_space_servo_tracking"
                ),
                "route_contract_identity_sha256": (
                    CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256
                ),
                "cam_contact_contract_identity_sha256": (
                    CORE_CAM_TAB_CONTACT_CONTRACT_IDENTITY_SHA256
                ),
                "physics_timestep_s": float(self.model.opt.timestep),
                "observed": free_space_observed,
                "audited_substeps_by_phase": dict(
                    self.core_capture_free_space_phase_counts
                ),
                "all_free_space_phases_observed": (
                    free_space_all_phases_observed
                ),
                "completed_endpoint_actions": sorted(
                    free_space_endpoint_actions
                ),
                "all_free_space_endpoints_completed": (
                    free_space_all_endpoints_completed
                ),
                "raw_samples": free_space_samples,
                "raw_samples_sha256": _canonical_json_sha256(
                    free_space_samples
                ),
                "maximum_abs_q_tracking_error_rad": (
                    maximum_free_q_error if free_space_observed else None
                ),
                "maximum_abs_preseat_error_mm": (
                    maximum_free_preseat_error if free_space_observed else None
                ),
                "maximum_abs_x_error_mm": (
                    maximum_free_x_error if free_space_observed else None
                ),
                "maximum_abs_transverse_y_mm": (
                    maximum_free_y if free_space_observed else None
                ),
                "maximum_orientation_error_rad": (
                    maximum_free_orientation if free_space_observed else None
                ),
                "minimum_lead_x_gap_mm": (
                    minimum_free_lead_gap if free_space_observed else None
                ),
                "cam_contact_observation_count": free_space_cam_contacts,
                "thresholds": {
                    "maximum_abs_q_tracking_error_rad": (
                        CORE_CAPTURE_ROUTE_ENDPOINT_Q_ERROR_RAD
                    ),
                    "maximum_abs_preseat_error_mm": 0.050,
                    "maximum_abs_x_error_mm": (
                        CORE_CAPTURE_SOURCE_CORRIDOR_MAX_ERROR_MM
                    ),
                    "maximum_abs_transverse_y_mm": 0.010,
                    "maximum_orientation_error_rad": (
                        CORE_CAPTURE_ROUTE_ENDPOINT_ORIENTATION_ERROR_RAD
                    ),
                    "minimum_lead_x_gap_mm": 0.0,
                    "maximum_cam_contact_count": 0,
                },
                "passed": free_space_tracking_passed,
                "live_dynamics_authority": False,
                "release_ready": False,
            },
            "max_actuator_utilization": dict(self.max_actuator_utilization),
            "action_deadlines_s": {
                action.name: action.timeout_s for action in self.actions
            },
            "declared_deadline_sum_s": math.fsum(
                action.timeout_s for action in self.actions
            ),
            "global_safety_margin_s": WORKFLOW_GLOBAL_SAFETY_MARGIN_S,
            "journal": list(self.journal),
            "milestone": "keeper_capture_and_dock_release_slider_unlocked",
            "physical_lock_intentionally_unclaimed": True,
            "release_ready": False,
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
    result["milestone"] = "keeper_capture_and_dock_release_slider_unlocked"
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
