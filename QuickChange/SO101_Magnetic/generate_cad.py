#!/usr/bin/env python3
"""Generate the retrofit SO-101 powered magnetic quick changer.

All dimensions are millimetres.  The magnets capture and preload the faces;
two shoulder studs and a dock-actuated keyhole slider provide the positive
mechanical lock.  A four-contact pogo cartridge carries the stock gripper's
Feetech power and half-duplex TTL bus without a manual cable operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import cadquery as cq


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EXPORT_DIR = HERE / "exports"
CORE_MANIFEST_NAME = "core_cad_manifest.json"
CORE_OUTPUT_NAMES = (
    "bill_of_materials.csv",
    "design_parameters.json",
    "electrical_pinout.csv",
    "engineering_check.json",
    "hardware_CS-Q-12-12-04-N.step",
    "hardware_DIN934_M3_lock_stud_nut.step",
    "hardware_E-GUL4-10_reference.step",
    "hardware_MC-12-12-03.step",
    "hardware_McMaster_90318A720_shoulder_screw.step",
    "hardware_Mill-Max_7983-1-15-20-75-14-11-0_reference.step",
    "so101_passive_tool_dock.step",
    "so101_passive_tool_dock.stl",
    "so101_positive_lock_slider.step",
    "so101_positive_lock_slider.stl",
    "so101_positive_lock_slider_profile.dxf",
    "so101_quick_change_assembly.step",
    "so101_robot_plate.step",
    "so101_robot_plate.stl",
    "so101_stock_gripper_retrofit_assembly.step",
    "so101_stock_gripper_tool_plate.step",
    "so101_stock_gripper_tool_plate.stl",
    "so101_tool_contact_board_reference.step",
    "so101_tool_contact_board_reference.stl",
    "so101_tool_plate.step",
    "so101_tool_plate.stl",
    "tool_contact_board.kicad_pcb",
    "tool_contact_board_fab_drawing.svg",
)

# Existing SO-101 follower interface, measured from the official STEP.  The
# official assembly guide specifies four M3x6 screws at this joint.
HORN_HOLE_PATTERN = 9.9
HORN_CLEARANCE_DIAMETER = 3.3

ROBOT_WIDTH = 48.0
ROBOT_HEIGHT = 48.0
TOOL_WIDTH = 56.0
TOOL_HEIGHT = 50.0
PLATE_THICKNESS = 9.5
CORNER_RADIUS = 4.0

# Left-side electrical wing.  Keeping it outside the positive-lock cavity
# makes the contact cartridge replaceable without disturbing the load path.
ELECTRICAL_WING_X_MIN = -36.0
ELECTRICAL_WING_X_MAX = -24.0
ELECTRICAL_WING_HEIGHT = 24.0
CONTACT_CENTER_X = -30.0
CONTACT_TARGET_PAD_OFFSET_X = -1.0
CONTACT_WIRE_PAD_OFFSET_X = 3.5
CONTACT_PITCH = 5.0
CONTACT_Y = (-7.5, -2.5, 2.5, 7.5)
CONTACT_SIGNALS = ("GND", "+12V", "TTL_DATA", "TOOL_ID_SPARE")

# Supermagnete CS-Q-12-12-04-N (EAN 7640172691830).
MAGNET_PART_NUMBER = "CS-Q-12-12-04-N"
MAGNET_WIDTH = 12.0
MAGNET_HEIGHT = 4.0
MAGNET_HOLE_DIAMETER = 4.5
MAGNET_COUNTERSINK_DIAMETER = 9.46
MAGNET_COUNTERSINK_DEPTH = 2.48
MAGNET_POCKET_WIDTH = 12.60
MAGNET_POCKET_DEPTH = 4.05

# Matching Supermagnete MC-12-12-03 steel target (EAN 7640172691892).
TARGET_PART_NUMBER = "MC-12-12-03"
TARGET_WIDTH = 12.0
TARGET_HEIGHT = 3.0
TARGET_SMALL_HOLE_DIAMETER = 5.7
TARGET_COUNTERSINK_DIAMETER = 11.7
TARGET_POCKET_WIDTH = 12.60
TARGET_POCKET_DEPTH = 3.05

# Both catalog magnetic components bear on their pocket floors, leaving their
# brittle plated faces deliberately below the printed mating lands.  The
# widened pockets retain 0.30 mm nominal lateral clearance to the opposing
# 12 mm component; the 0.05 mm recess prevents the hardware from becoming the
# impact stop.
MAGNETIC_HARDWARE_FACE_RECESS = 0.05

MAGNET_CENTER_Y = 16.0

LOCATOR_X = 20.0
LOCATOR_HEIGHT = 3.5
LOCATOR_BASE_DIAMETER = 5.5
LOCATOR_TIP_DIAMETER = 4.3
SOCKET_CLEARANCE = 0.25
RELIEVED_SOCKET_Y_CLEARANCE = 0.35

# Positive lock.  The slider is inside a printed roof: the shoulder-screw
# heads pull the slider into that roof rather than relying on guide screws.
SLIDER_THICKNESS = 1.6
SLIDER_TRAVEL = 3.0
SLIDER_Z = 4.7
SLIDER_SLOT_BOTTOM = 4.5
SLIDER_SLOT_TOP = 6.5
SLIDER_BRIDGE_HEIGHT = 4.8
SLIDER_LOBE_RADIUS = 4.4
SLIDER_TAB_END_X = 24.0
LOCK_STUD_X = 12.0
LOCK_SHOULDER_DIAMETER = 4.0
LOCK_SHOULDER_LENGTH = 5.0
LOCK_HEAD_DIAMETER = 6.0
LOCK_HEAD_HEIGHT = 1.3
LOCK_THREAD_LENGTH = 4.0
LOCK_NUT_ACROSS_FLATS = 5.5
LOCK_NUT_HEIGHT = 2.4
LOCK_NUT_POCKET_ACROSS_FLATS = 5.7
LOCK_NUT_POCKET_FLOOR = 1.5
KEYHOLE_ENTRY_DIAMETER = 6.5
KEYHOLE_NECK_WIDTH = 4.25
GUIDE_SLOT_WIDTH = 2.4

# Fixed robot-plate head wells are deliberately larger/deeper than the moving
# slider entry holes.  Keeping these datums separate preserves head retention
# in the 6.5 mm keyholes while providing route and fabrication reserve around
# the 6.0 mm stud heads.
ROBOT_STUD_WELL_DIAMETER = 7.2
ROBOT_STUD_WELL_BOTTOM_Z = 2.8

# The 4 mm ENIG targets must never bear on the printed robot face around the
# much smaller press-fit pogo pilots.  A shallow face relief clears each pad
# while retaining the shallow surface webs measured by the fit contract. The
# legacy straight pilot below it is not retention authority; the official
# stepped barb/knurl mounting section remains an explicit release blocker.
POGO_PAD_RELIEF_DIAMETER = 4.6
POGO_PAD_RELIEF_DEPTH = 0.35

# The narrow keyhole is a true swept shoulder path, not merely a rectangle
# between the unlocked and locked centre coordinates.  Its two semicircular
# ends are centred at the same stud datum before and after the complete 3 mm
# slider translation.  This leaves 0.125 mm radial shoulder clearance while
# retaining 0.875 mm radial overlap against the 6 mm head.
KEYHOLE_NECK_OVERALL_LENGTH = SLIDER_TRAVEL + KEYHOLE_NECK_WIDTH
KEYHOLE_NECK_CENTER_OFFSET_X = -SLIDER_TRAVEL / 2.0

# Passive dock cam and the feature-specific robot-plate relief around it.  The
# cam must retain the full 3 mm slider stroke, so its 24.05 mm inner datum does
# not move.  Instead the fixed printed plate is locally recessed with a 0.50
# mm three-dimensional guard band around the cam's complete swept prism.
DOCK_CAM_X_INNER = 24.05
DOCK_CAM_X_OUTER_MIN = 28.0
DOCK_CAM_X_OUTER_MAX = 34.0
DOCK_CAM_Y_MIN = -16.0
DOCK_CAM_Y_MAX = 0.0
DOCK_CAM_Z_MIN = -4.15
DOCK_CAM_THICKNESS = 2.2

# The planar X/Y wedge cannot open the slider during the final axial approach:
# its first contact occurs after the stud heads reach the slider plane.  A
# narrow, integral 45-degree lead acts only on the exposed slider tab.  The
# robot follows the published +0.20 -> 0.00 mm lateral recenter while the lead
# lowers the slider from locked to the 0.05 mm passive cam-clearance state.
# All coordinates remain in the dock-local millimetre frame.
DOCK_CAM_AXIAL_LEAD_X_INNER_LOWER = 27.25
DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER = DOCK_CAM_X_INNER
DOCK_CAM_AXIAL_LEAD_X_OUTER = 29.0
DOCK_CAM_AXIAL_LEAD_Y_MIN = 0.0
DOCK_CAM_AXIAL_LEAD_Y_MAX = 2.0
DOCK_CAM_AXIAL_LEAD_Z_LOWER = -9.6
DOCK_CAM_AXIAL_LEAD_Z_UPPER = -6.4
DOCK_CAM_AXIAL_HOLD_Z_UPPER = DOCK_CAM_Z_MIN
# A 1 mm outer root bridge overlaps both source solids while remaining beyond
# the locked tab's x=27 mm maximum.  This makes the printed cam one OCCT solid
# without delaying or obstructing the -Y passive-return path.
DOCK_CAM_AXIAL_ROOT_X_BOUNDS = (28.0, 29.0)
DOCK_CAM_AXIAL_ROOT_Y_BOUNDS = (-1.0, 1.0)
DOCK_CAM_AXIAL_ROOT_Z_BOUNDS = (-4.65, -3.65)
DOCK_CAM_CAPTURE_RECENTER_START_PRESEAT_MM = 6.4
DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM = 3.2
DOCK_CAM_HEAD_ENTRY_PRESEAT_MM = 3.1
DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM = DOCK_CAM_X_INNER - SLIDER_TAB_END_X
DOCK_CAM_PASSIVE_RELEASE_INITIAL_DWELL_MM = 2.0
ROBOT_CAM_CLEARANCE_MM = 0.50
ROBOT_CAM_RELIEF_X_MAX = ROBOT_WIDTH / 2.0 + 1.0
ROBOT_CAM_RELIEF_X_MIN = DOCK_CAM_X_INNER - ROBOT_CAM_CLEARANCE_MM
ROBOT_CAM_RELIEF_Y_MIN = DOCK_CAM_Y_MIN - ROBOT_CAM_CLEARANCE_MM
# During rack exit the cam's 24.05 mm tip traverses the plate toward +Y.
# Carry the narrow recess through the complete plate edge, plus the same
# radial guard band.  The recess also spans the complete 9.5 mm printed plate
# thickness: limiting it to the cam's seated Z envelope left only the 0.05 mm
# outer-edge cap clearance while the robot approached axially.  The full-depth
# cut preserves the slider/keyhole datums while making the same 0.50 mm radial
# guard authoritative during both axial capture and rack exit.
ROBOT_CAM_RELIEF_Y_MAX = ROBOT_HEIGHT / 2.0 + ROBOT_CAM_CLEARANCE_MM
ROBOT_CAM_RELIEF_Z_MIN = 0.0
ROBOT_CAM_RELIEF_Z_MAX = PLATE_THICKNESS
ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM = 0.20

# Feature-to-relief ligaments are public mechanism-preservation invariants.
# The right stud well is the closest fixed functional cut; the right slider
# lobe is the closest moving lock feature.  Neither datum is moved by the
# relief correction.
ROBOT_STUD_WELL_RADIUS = ROBOT_STUD_WELL_DIAMETER / 2.0
ROBOT_CAM_RELIEF_TO_STUD_WELL_LIGAMENT_MM = ROBOT_CAM_RELIEF_X_MIN - (
    LOCK_STUD_X + ROBOT_STUD_WELL_RADIUS
)
ROBOT_CAM_RELIEF_TO_SLIDER_LOBE_LIGAMENT_MM = ROBOT_CAM_RELIEF_X_MIN - (
    LOCK_STUD_X + SLIDER_LOBE_RADIUS
)

# Released core stock-gripper dock stop.  Spoon and whisk stops belong to the
# separate matcha rack authority and intentionally have different X/Y bounds.
CORE_DOCK_STOP_X_MIN = -45.0
CORE_DOCK_STOP_X_MAX = 37.0
CORE_DOCK_STOP_Y_MIN = 26.0
CORE_DOCK_STOP_Y_MAX = 32.0
CORE_DOCK_STOP_Z_MIN = -3.0
CORE_DOCK_STOP_Z_MAX = 12.5
CORE_DOCK_STOP_HOLE_X = (-25.0, 21.0)
CORE_DOCK_STOP_HOLE_Y_START = 25.9
CORE_DOCK_STOP_HOLE_DIAMETER = 4.4

# The assembly datums are public source authority for downstream simulation.
STOCK_TOOL_PLATE_ASSEMBLY_POS_MM = (0.0, 0.0, PLATE_THICKNESS)
STOCK_FIXED_STEP_ASSEMBLY_POS_MM = (
    0.4875,
    0.218,
    2.0 * PLATE_THICKNESS + 0.051,
)
STOCK_FIXED_STEP_TOOL_LOCAL_POS_MM = (0.4875, 0.218, PLATE_THICKNESS + 0.051)
STOCK_FIXED_STEP_TOOL_LOCAL_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

# MISUMI E-GUL4-10: 304 stainless, 4 mm OD, 10 mm free length,
# 0.98 N/mm, 4 mm permitted deflection.  It returns the slider to locked.
RETURN_SPRING_PART_NUMBER = "E-GUL4-10"
RETURN_SPRING_OD = 4.0
RETURN_SPRING_WIRE = 0.45
RETURN_SPRING_FREE_LENGTH = 10.0
RETURN_SPRING_RATE_N_PER_MM = 0.98
RETURN_SPRING_MAX_DEFLECTION = 4.0
RETURN_SPRING_FIXED_X = -22.8
RETURN_SPRING_CENTER_Z = 4.2

# Mill-Max high-current solder-cup spring contact.  The manufacturer specifies
# an 8 A maximum, 6.2 A derated current, nominal 1.40 +/- 0.13 mm full stroke,
# and a 1.07 mm plunger. Ground is shown 0.2 mm farther forward for first-mate,
# but the shoulder datum needed to manufacture that offset is unresolved. The
# following two solids are visualization envelopes only, not purchased-part or
# dynamic contact authority.
POGO_PART_NUMBER = "7983-1-15-20-75-14-11-0"
POGO_REFERENCE_BODY_NOMINAL_DIAMETER = 2.11
POGO_OFFICIAL_DIAMETER_TOLERANCE = 0.051
# The current straight pilot is retained only as a legacy visualization datum.
# It is not a fabrication prescription: Mill-Max documents mutually exclusive
# barb- and knurl-side installations with different holes and a body
# counterbore/shoulder stop.
POGO_LEGACY_REFERENCE_PILOT_DIAMETER = 1.575
POGO_OVERALL_LENGTH = 9.5
POGO_MID_STROKE = 0.7
POGO_NOMINAL_FULL_STROKE = 1.40
POGO_FULL_STROKE_TOLERANCE = 0.13
POGO_GUARANTEED_MINIMUM_FULL_STROKE = (
    POGO_NOMINAL_FULL_STROKE - POGO_FULL_STROKE_TOLERANCE
)
POGO_STANDARD_PROTRUSION = 0.70
POGO_GROUND_PROTRUSION = 0.90
POGO_REFERENCE_SOLDER_CUP_DIAMETER = 1.64
POGO_REFERENCE_SOLDER_CUP_LENGTH = 3.0
POGO_REFERENCE_FIXED_SLEEVE_LENGTH = 5.1
POGO_PLUNGER_DIAMETER = 1.07
POGO_REFERENCE_PLUNGER_EXPOSURE = (
    POGO_OVERALL_LENGTH
    - POGO_REFERENCE_SOLDER_CUP_LENGTH
    - POGO_REFERENCE_FIXED_SLEEVE_LENGTH
)
CONTACT_BOARD_WIDTH = 10.0
CONTACT_BOARD_HEIGHT = 22.0
CONTACT_BOARD_THICKNESS = 1.0
CONTACT_PAD_DIAMETER = 4.0
CONTACT_WIRE_PAD_DIAMETER = 2.0
CONTACT_WIRE_DRILL_DIAMETER = 1.0

TOOL_MOUNT_X = 21.0
TOOL_MOUNT_Y = 17.0
STOCK_GRIPPER_INSERT_HOLE_DIAMETER = 4.0
STOCK_GRIPPER_INSERT_DEPTH = 5.9

# Official component values used for the interface envelope.  These are not a
# published whole-arm payload rating.
SERVO_RATED_TORQUE_KG_CM = 10.0
SERVO_STALL_TORQUE_KG_CM = 30.0
KG_CM_TO_N_M = 0.0980665
SERVO_RATED_TORQUE_N_M = SERVO_RATED_TORQUE_KG_CM * KG_CM_TO_N_M
SERVO_STALL_TORQUE_N_M = SERVO_STALL_TORQUE_KG_CM * KG_CM_TO_N_M
MAGNET_FORCE_EACH_N = 29.4
MAGNET_DISPLACEMENT_EACH_N = 5.88


def rounded_plate(width: float, height: float, thickness: float, radius: float) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(width, height, thickness, centered=(True, True, False))
        .edges("|Z")
        .fillet(radius)
    )


def magnet_points() -> list[tuple[float, float]]:
    return [(0.0, -MAGNET_CENTER_Y), (0.0, MAGNET_CENTER_Y)]


def horn_points() -> list[tuple[float, float]]:
    offset = HORN_HOLE_PATTERN / 2
    return [(-offset, -offset), (-offset, offset), (offset, -offset), (offset, offset)]


def pogo_points() -> list[tuple[float, float]]:
    return [(CONTACT_CENTER_X + CONTACT_TARGET_PAD_OFFSET_X, y) for y in CONTACT_Y]


def interface_hardware_fit_contract() -> dict[str, object]:
    """Return source dimensions and honest authority gaps at the mating face."""

    rear_wiring_pocket_top = -0.05 + 1.9
    counterbore_bottom = PLATE_THICKNESS - POGO_PAD_RELIEF_DEPTH
    ground_pin_sleeve_top = (
        PLATE_THICKNESS
        + POGO_GROUND_PROTRUSION
        - POGO_OVERALL_LENGTH
        + POGO_REFERENCE_SOLDER_CUP_LENGTH
        + POGO_REFERENCE_FIXED_SLEEVE_LENGTH
    )
    nearest_horn_counterbore_ligament = (
        math.hypot(LOCK_STUD_X - HORN_HOLE_PATTERN / 2.0, HORN_HOLE_PATTERN / 2.0)
        - ROBOT_STUD_WELL_DIAMETER / 2.0
        - 6.2 / 2.0
    )
    return {
        "schema_version": "1.0",
        "frame": "mating_component_native_mm",
        "required_clearance_mm": 0.20,
        # This allowance is arithmetic only.  The moving cross-interface pair
        # route has not yet been sampled/replayed and is a release blocker.
        "unqualified_local_motion_allowance_mm": 0.0501,
        # This is only the exact-CAD arithmetic left after the route interval
        # and required-clearance deductions.  It is not a qualified SLS/FDM
        # or catalog-hardware tolerance and must never be presented as one.
        "unqualified_arithmetic_residual_mm": 0.0499,
        "release_authority": {
            "fabrication_process_tolerance_qualified": False,
            "qualified_combined_error_limit_mm": None,
            "fixed_pogo_shell_exact_drawing_bound": False,
            "pogo_mounting_sectional_bore_resolved": False,
            "ground_first_mate_shoulder_datum_resolved": False,
            "ground_first_mate_tolerance_stack_qualified": False,
            "magnetic_fastener_seating_and_preload_bound": False,
            "moving_interface_pair_route_recomputed": False,
            "printed_interface_feature_strength_qualified": False,
            "release_ready": False,
            "blockers": [
                "fabrication_process_tolerance_unqualified",
                "fixed_pogo_shell_is_illustrative_reference_only",
                "pogo_mounting_sectional_bore_unresolved",
                "ground_first_mate_shoulder_datum_unresolved",
                "ground_first_mate_tolerance_stack_unqualified",
                "magnetic_fastener_seating_and_preload_unproven",
                "moving_interface_pair_route_not_recomputed",
                "printed_interface_feature_strength_unqualified",
            ],
        },
        "magnetic_hardware": {
            "magnet_pocket_width_mm": MAGNET_POCKET_WIDTH,
            "target_pocket_width_mm": TARGET_POCKET_WIDTH,
            "component_width_mm": MAGNET_WIDTH,
            "nominal_lateral_clearance_mm": (
                min(MAGNET_POCKET_WIDTH, TARGET_POCKET_WIDTH) - MAGNET_WIDTH
            )
            / 2.0,
            "face_recess_each_mm": MAGNETIC_HARDWARE_FACE_RECESS,
            "combined_magnet_target_air_gap_mm": (
                2.0 * MAGNETIC_HARDWARE_FACE_RECESS
            ),
            "magnet_native_z_bounds_mm": [
                PLATE_THICKNESS - MAGNET_POCKET_DEPTH,
                PLATE_THICKNESS - MAGNETIC_HARDWARE_FACE_RECESS,
            ],
            "target_tool_local_z_bounds_mm": [
                MAGNETIC_HARDWARE_FACE_RECESS,
                TARGET_POCKET_DEPTH,
            ],
            "own_pocket_support": "named_rear_face_to_pocket_floor_tangency",
        },
        "pogo_target_pad_relief": {
            "manufacturer_source": (
                "https://www.mill-max.com/products/discrete-spring-loaded-pins/"
                "spring-loaded-pin-with-solder-cup-termination/7983/"
                "7983-1-15-20-75-14-11-0"
            ),
            "manufacturer_press_fit_application_note": (
                "https://www.mill-max.com/sites/default/files/external/assets/"
                "2020-10/spring-loaded_solder-cup_pin_2.pdf"
            ),
            "manufacturer_press_fit_application_note_sha256": (
                "bbf4c414a11bd3355cde2bb25624c6736b61942964b2cbb3fc42c67c09e87adf"
            ),
            "manufacturer_dimension_drawing": {
                "url": (
                    "https://www.mill-max.com/sites/default/files/external/"
                    "products/fullsize/2020-09/7983.svg"
                ),
                "sha256": (
                    "c97327d953663a0aa04ea389ee2d2be19372ffa21503f46e5cbbfb0fd2e890e8"
                ),
            },
            "official_profile_nominal_diameters_mm": {
                "plunger": 1.07,
                "upper": 1.73,
                "shoulder": 1.85,
                "barb": 1.94,
                "body": 2.11,
                "knurl": 1.65,
                "solder_cup_shaft": 1.52,
                "solder_cup_bore": 0.97,
            },
            "official_standard_length_tolerance_mm": 0.15,
            "ground_first_mate_tolerance_stack": {
                "nominal_offset_mm": (
                    POGO_GROUND_PROTRUSION - POGO_STANDARD_PROTRUSION
                ),
                "independent_pin_length_error_bound_mm": 0.30,
                "guaranteed_worst_case_lead_mm": (
                    POGO_GROUND_PROTRUSION
                    - POGO_STANDARD_PROTRUSION
                    - 0.30
                ),
                "passed": False,
            },
            "fixed_shell_authority": "illustrative_reference_only_not_conservative",
            "mounting_authority": (
                "unresolved_choose_barb_or_knurl_and_rebuild_sectional_bore"
            ),
            "diameter_mm": POGO_PAD_RELIEF_DIAMETER,
            "depth_mm": POGO_PAD_RELIEF_DEPTH,
            "target_pad_diameter_mm": CONTACT_PAD_DIAMETER,
            "legacy_reference_straight_pilot_diameter_mm": (
                POGO_LEGACY_REFERENCE_PILOT_DIAMETER
            ),
            "official_barb_mounting_hole_mm": 1.92,
            "official_knurl_feature_diameter_mm": 1.65,
            "official_knurl_mounting_hole_mm": 1.58,
            "official_minimum_body_counterbore_diameter_mm": 2.21,
            "fixed_shell_reference_nominal_body_diameter_mm": (
                POGO_REFERENCE_BODY_NOMINAL_DIAMETER
            ),
            "official_diameter_tolerance_mm": POGO_OFFICIAL_DIAMETER_TOLERANCE,
            "official_plunger_diameter_mm": POGO_PLUNGER_DIAMETER,
            "official_nominal_full_stroke_mm": POGO_NOMINAL_FULL_STROKE,
            "official_full_stroke_tolerance_mm": POGO_FULL_STROKE_TOLERANCE,
            "official_guaranteed_minimum_full_stroke_mm": (
                POGO_GUARANTEED_MINIMUM_FULL_STROKE
            ),
            "reference_render_retraction_range_mm": [
                0.0,
                POGO_GROUND_PROTRUSION,
            ],
            "nominal_radial_clearance_mm": (
                POGO_PAD_RELIEF_DIAMETER - CONTACT_PAD_DIAMETER
            )
            / 2.0,
            "nominal_axial_clearance_mm": (
                POGO_PAD_RELIEF_DEPTH - 0.05
            ),
            "minimum_adjacent_surface_web_mm": (
                CONTACT_PITCH - POGO_PAD_RELIEF_DIAMETER
            ),
            "minimum_outer_y_surface_web_mm": (
                ELECTRICAL_WING_HEIGHT / 2.0
                - max(abs(value) for value in CONTACT_Y)
                - POGO_PAD_RELIEF_DIAMETER / 2.0
            ),
            "legacy_reference_pilot_land_mm": (
                counterbore_bottom - rear_wiring_pocket_top
            ),
            "legacy_ground_shell_to_face_relief_offset_mm": (
                counterbore_bottom - ground_pin_sleeve_top
            ),
        },
        "fixed_stud_head_wells": {
            "diameter_mm": ROBOT_STUD_WELL_DIAMETER,
            "bottom_native_z_mm": ROBOT_STUD_WELL_BOTTOM_Z,
            "stud_head_diameter_mm": LOCK_HEAD_DIAMETER,
            "nominal_radial_clearance_mm": (
                ROBOT_STUD_WELL_DIAMETER - LOCK_HEAD_DIAMETER
            )
            / 2.0,
            "nominal_axial_clearance_mm": (
                PLATE_THICKNESS
                - LOCK_SHOULDER_LENGTH
                - LOCK_HEAD_HEIGHT
                - ROBOT_STUD_WELL_BOTTOM_Z
            ),
            "cam_relief_ligament_mm": (
                ROBOT_CAM_RELIEF_TO_STUD_WELL_LIGAMENT_MM
            ),
            "nearest_horn_counterbore_ligament_mm": (
                nearest_horn_counterbore_ligament
            ),
            "slider_lobe_bearing_annulus_mm": (
                SLIDER_LOBE_RADIUS - ROBOT_STUD_WELL_RADIUS
            ),
        },
    }


def square_cutter(width: float, depth: float, x: float, y: float, z: float) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(width, width, depth, centered=(True, True, False))
        .translate((x, y, z))
    )


def box_cutter(
    width: float,
    height: float,
    depth: float,
    x: float,
    y: float,
    z: float,
) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(width, height, depth, centered=(True, True, False))
        .translate((x, y, z))
    )


def hex_cutter(across_flats: float, depth: float, x: float, y: float, z: float = -0.05) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .polygon(6, across_flats / 0.8660254038)
        .extrude(depth)
        .translate((x, y, z))
    )


def axis_cylinder(
    diameter: float,
    length: float,
    start: tuple[float, float, float],
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> cq.Workplane:
    solid = cq.Solid.makeCylinder(
        diameter / 2,
        length,
        cq.Vector(*start),
        cq.Vector(*direction),
    )
    return cq.Workplane(obj=solid)


def cylinder_cutter(
    diameter: float,
    height: float,
    x: float,
    y: float,
    z: float = -0.1,
) -> cq.Workplane:
    return axis_cylinder(diameter, height, (x, y, z))


def screw_on_magnet() -> cq.Workplane:
    magnet = cq.Workplane("XY").box(
        MAGNET_WIDTH, MAGNET_WIDTH, MAGNET_HEIGHT, centered=(True, True, False)
    )
    magnet = magnet.cut(cylinder_cutter(MAGNET_HOLE_DIAMETER, MAGNET_HEIGHT + 0.2, 0, 0))
    countersink = cq.Solid.makeCone(
        MAGNET_COUNTERSINK_DIAMETER / 2,
        MAGNET_HOLE_DIAMETER / 2,
        MAGNET_COUNTERSINK_DEPTH + 0.05,
        cq.Vector(0, 0, MAGNET_HEIGHT + 0.05),
        cq.Vector(0, 0, -1),
    )
    return magnet.cut(cq.Workplane(obj=countersink)).clean()


def steel_target() -> cq.Workplane:
    target = cq.Workplane("XY").box(
        TARGET_WIDTH, TARGET_WIDTH, TARGET_HEIGHT, centered=(True, True, False)
    )
    target = target.cut(cylinder_cutter(TARGET_SMALL_HOLE_DIAMETER, TARGET_HEIGHT + 0.2, 0, 0))
    countersink = cq.Solid.makeCone(
        TARGET_COUNTERSINK_DIAMETER / 2,
        TARGET_SMALL_HOLE_DIAMETER / 2,
        TARGET_HEIGHT + 0.05,
        cq.Vector(0, 0, -0.05),
        cq.Vector(0, 0, 1),
    )
    return target.cut(cq.Workplane(obj=countersink)).clean()


def slider_blank(clearance: float = 0.0) -> cq.Workplane:
    thickness = SLIDER_THICKNESS
    lobe_radius = SLIDER_LOBE_RADIUS + clearance
    bridge = cq.Workplane("XY").box(
        2 * LOCK_STUD_X,
        SLIDER_BRIDGE_HEIGHT + 2 * clearance,
        thickness,
        centered=(True, True, False),
    )
    blank = bridge
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        lobe = cq.Workplane("XY").circle(lobe_radius).extrude(thickness).translate((x, 0, 0))
        blank = blank.union(lobe)

    tab_start = LOCK_STUD_X + SLIDER_LOBE_RADIUS - 0.5
    tab_length = SLIDER_TAB_END_X - tab_start
    tab = cq.Workplane("XY").box(
        tab_length,
        4.0 + 2 * clearance,
        thickness,
        centered=(True, True, False),
    ).translate((tab_start + tab_length / 2, 0, 0))
    return blank.union(tab).clean()


def locking_slider() -> cq.Workplane:
    slider = slider_blank()

    # Unlocked: entry holes align with the two shoulder-screw heads.  The
    # return spring shifts the slider +X by 3 mm, placing each 4 mm shoulder in
    # a 4.25 mm swept neck while the 6 mm head remains below the slider.  The
    # capsule spans the complete shoulder-centre path and includes one neck
    # radius beyond each endpoint; a centre-to-centre rectangle alone would
    # intersect the shoulder over the latter half of the declared travel.
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        slider = slider.cut(
            cylinder_cutter(KEYHOLE_ENTRY_DIAMETER, SLIDER_THICKNESS + 0.2, x, 0)
        )
        shoulder_path = (
            cq.Workplane("XY")
            .center(x + KEYHOLE_NECK_CENTER_OFFSET_X, 0)
            .slot2D(KEYHOLE_NECK_OVERALL_LENGTH, KEYHOLE_NECK_WIDTH, 0)
            .extrude(SLIDER_THICKNESS + 0.2)
            .translate((0, 0, -0.1))
        )
        slider = slider.cut(shoulder_path)

    # A single flush M2 guide screw prevents loss during servicing.  The
    # printed roof, not this screw, reacts tool separation load.
    guide_slot = (
        cq.Workplane("XY")
        .center(-SLIDER_TRAVEL / 2, 0)
        .slot2D(SLIDER_TRAVEL + GUIDE_SLOT_WIDTH, GUIDE_SLOT_WIDTH, 0)
        .extrude(SLIDER_THICKNESS + 0.2)
        .translate((0, 0, -0.1))
    )
    return slider.cut(guide_slot).clean()


def shoulder_lock_stud() -> cq.Workplane:
    # Local Z=0 is the tool mating face.  The M3 thread goes into the tool;
    # the precision 4 mm shoulder and low-profile head project into the robot.
    thread = cq.Workplane("XY").circle(1.5).extrude(LOCK_THREAD_LENGTH)
    shoulder = (
        cq.Workplane("XY")
        .circle(LOCK_SHOULDER_DIAMETER / 2)
        .extrude(LOCK_SHOULDER_LENGTH)
        .translate((0, 0, -LOCK_SHOULDER_LENGTH))
    )
    head = (
        cq.Workplane("XY")
        .circle(LOCK_HEAD_DIAMETER / 2)
        .extrude(LOCK_HEAD_HEIGHT)
        .translate((0, 0, -LOCK_SHOULDER_LENGTH - LOCK_HEAD_HEIGHT))
    )
    return thread.union(shoulder).union(head).clean()


def lock_stud_nut() -> cq.Workplane:
    nut = (
        cq.Workplane("XY")
        .polygon(6, LOCK_NUT_ACROSS_FLATS / 0.8660254038)
        .extrude(LOCK_NUT_HEIGHT)
    )
    return nut.cut(cylinder_cutter(3.2, LOCK_NUT_HEIGHT + 0.2, 0, 0)).clean()


def compression_spring(length: float = RETURN_SPRING_FREE_LENGTH) -> cq.Workplane:
    # Reference envelope/visual model of the catalog spring, axis along +X.
    turns = 7.0
    helix_height = max(length - RETURN_SPRING_WIRE, 0.5)
    pitch = helix_height / turns
    mean_radius = (RETURN_SPRING_OD - RETURN_SPRING_WIRE) / 2
    path = cq.Wire.makeHelix(pitch, helix_height, mean_radius)
    profile = cq.Workplane("XZ").center(mean_radius, 0).circle(RETURN_SPRING_WIRE / 2)
    spring = profile.sweep(path, isFrenet=True)
    return spring.rotate((0, 0, 0), (0, 1, 0), 90).clean()


def pogo_reference_fixed_shell() -> cq.Workplane:
    """Return an illustrative shell, never a conservative clearance bound."""

    solder_cup = (
        cq.Workplane("XY")
        .circle(POGO_REFERENCE_SOLDER_CUP_DIAMETER / 2.0)
        .extrude(POGO_REFERENCE_SOLDER_CUP_LENGTH)
    )
    sleeve = (
        cq.Workplane("XY")
        .circle(POGO_REFERENCE_BODY_NOMINAL_DIAMETER / 2.0)
        .extrude(POGO_REFERENCE_FIXED_SLEEVE_LENGTH)
        .translate((0.0, 0.0, POGO_REFERENCE_SOLDER_CUP_LENGTH))
    )
    return solder_cup.union(sleeve).clean()


def pogo_reference_plunger(retraction_mm: float = 0.0) -> cq.Workplane:
    """Return a visual plunger envelope, never dynamic or retention authority."""

    if (
        not math.isfinite(retraction_mm)
        or retraction_mm < 0.0
        or retraction_mm > POGO_GROUND_PROTRUSION
    ):
        raise ValueError("retraction_mm must be finite and inside reference range")
    exposed_length = POGO_REFERENCE_PLUNGER_EXPOSURE - retraction_mm
    if exposed_length <= 0.0:
        raise ValueError("retraction_mm removes the reference plunger")
    return (
        cq.Workplane("XY")
        .circle(POGO_PLUNGER_DIAMETER / 2.0)
        .extrude(exposed_length)
        .translate(
            (
                0.0,
                0.0,
                POGO_REFERENCE_SOLDER_CUP_LENGTH
                + POGO_REFERENCE_FIXED_SLEEVE_LENGTH,
            )
        )
    )


def pogo_reference_pin(retraction_mm: float = 0.0) -> cq.Workplane:
    """Return the explicitly non-authoritative display envelope."""

    return pogo_reference_fixed_shell().union(
        pogo_reference_plunger(retraction_mm)
    ).clean()


def contact_board() -> cq.Workplane:
    board = cq.Workplane("XY").box(
        CONTACT_BOARD_WIDTH,
        CONTACT_BOARD_HEIGHT,
        CONTACT_BOARD_THICKNESS,
        centered=(True, True, False),
    )
    for y in CONTACT_Y:
        board = board.cut(
            cylinder_cutter(
                CONTACT_WIRE_DRILL_DIAMETER,
                CONTACT_BOARD_THICKNESS + 0.2,
                CONTACT_WIRE_PAD_OFFSET_X,
                y,
            )
        )
    return board.clean()


def contact_pad() -> cq.Workplane:
    return cq.Workplane("XY").circle(CONTACT_PAD_DIAMETER / 2).extrude(0.05)


def slider_track_envelope() -> cq.Workplane:
    depth = SLIDER_SLOT_TOP - SLIDER_SLOT_BOTTOM
    clearance = 0.22
    diameter = 2 * (SLIDER_LOBE_RADIUS + clearance)
    envelope = None
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        swept_lobe = (
            cq.Workplane("XY")
            .center(x + SLIDER_TRAVEL / 2, 0)
            .slot2D(diameter + SLIDER_TRAVEL, diameter, 0)
            .extrude(depth)
        )
        envelope = swept_lobe if envelope is None else envelope.union(swept_lobe)

    bridge = cq.Workplane("XY").box(
        2 * LOCK_STUD_X + SLIDER_TRAVEL,
        SLIDER_BRIDGE_HEIGHT + 2 * clearance,
        depth,
        centered=(True, True, False),
    ).translate((SLIDER_TRAVEL / 2, 0, 0))
    envelope = envelope.union(bridge)

    tab_start = LOCK_STUD_X + SLIDER_LOBE_RADIUS - 0.5
    tab_end = SLIDER_TAB_END_X + SLIDER_TRAVEL + 0.25
    tab = cq.Workplane("XY").box(
        tab_end - tab_start,
        4.0 + 2 * clearance,
        depth,
        centered=(True, True, False),
    ).translate(((tab_start + tab_end) / 2, 0, 0))
    return envelope.union(tab).translate((0, 0, SLIDER_SLOT_BOTTOM)).clean()


def core_dock_stop_spec() -> dict[str, object]:
    """Return the exact released core stop contract in the dock-local frame."""

    return {
        "dock": "core_stock_gripper",
        "bounds_mm": {
            "x": [CORE_DOCK_STOP_X_MIN, CORE_DOCK_STOP_X_MAX],
            "y": [CORE_DOCK_STOP_Y_MIN, CORE_DOCK_STOP_Y_MAX],
            "z": [CORE_DOCK_STOP_Z_MIN, CORE_DOCK_STOP_Z_MAX],
        },
        "center_mm": [
            (CORE_DOCK_STOP_X_MIN + CORE_DOCK_STOP_X_MAX) / 2.0,
            (CORE_DOCK_STOP_Y_MIN + CORE_DOCK_STOP_Y_MAX) / 2.0,
            (CORE_DOCK_STOP_Z_MIN + CORE_DOCK_STOP_Z_MAX) / 2.0,
        ],
        "size_mm": [
            CORE_DOCK_STOP_X_MAX - CORE_DOCK_STOP_X_MIN,
            CORE_DOCK_STOP_Y_MAX - CORE_DOCK_STOP_Y_MIN,
            CORE_DOCK_STOP_Z_MAX - CORE_DOCK_STOP_Z_MIN,
        ],
        "through_holes": [
            {
                "axis": [0.0, 1.0, 0.0],
                "x_mm": x_value,
                "y_start_mm": CORE_DOCK_STOP_HOLE_Y_START,
                "z_mm": (CORE_DOCK_STOP_Z_MIN + CORE_DOCK_STOP_Z_MAX) / 2.0,
                "diameter_mm": CORE_DOCK_STOP_HOLE_DIAMETER,
            }
            for x_value in CORE_DOCK_STOP_HOLE_X
        ],
    }


def stock_gripper_mount_contract() -> dict[str, object]:
    """Return the exact fixed STEP placement relative to the coupling face."""

    return {
        "tool_plate_assembly_pos_mm": list(STOCK_TOOL_PLATE_ASSEMBLY_POS_MM),
        "fixed_step_assembly_pos_mm": list(STOCK_FIXED_STEP_ASSEMBLY_POS_MM),
        "fixed_step_tool_local_pos_mm": list(STOCK_FIXED_STEP_TOOL_LOCAL_POS_MM),
        "fixed_step_tool_local_quat_wxyz": list(
            STOCK_FIXED_STEP_TOOL_LOCAL_QUAT_WXYZ
        ),
    }


def robot_cam_relief_contract() -> dict[str, object]:
    """Return the complete fixed-plate/cam clearance source contract."""

    return {
        "required_clearance_mm": ROBOT_CAM_CLEARANCE_MM,
        "bounds_native_mm": {
            "x": [ROBOT_CAM_RELIEF_X_MIN, ROBOT_CAM_RELIEF_X_MAX],
            "y": [ROBOT_CAM_RELIEF_Y_MIN, ROBOT_CAM_RELIEF_Y_MAX],
            "z": [ROBOT_CAM_RELIEF_Z_MIN, ROBOT_CAM_RELIEF_Z_MAX],
        },
        "through_full_printed_plate_thickness": (
            ROBOT_CAM_RELIEF_Z_MIN == 0.0
            and ROBOT_CAM_RELIEF_Z_MAX == PLATE_THICKNESS
        ),
        "guided_axial_approach_offset_mm": ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM,
        "cam_inner_x_mm": DOCK_CAM_X_INNER,
        "slider_travel_mm": SLIDER_TRAVEL,
        "stud_centres_x_mm": [-LOCK_STUD_X, LOCK_STUD_X],
        "stud_well_radius_mm": ROBOT_STUD_WELL_RADIUS,
        "minimum_relief_to_stud_well_ligament_mm": (
            ROBOT_CAM_RELIEF_TO_STUD_WELL_LIGAMENT_MM
        ),
        "minimum_relief_to_slider_lobe_ligament_mm": (
            ROBOT_CAM_RELIEF_TO_SLIDER_LOBE_LIGAMENT_MM
        ),
    }


def positive_lock_cam_capture_lateral_offset_mm(preseat_mm: float) -> float:
    """Return the source route's dock-local +X offset at a preseated gap."""

    if not math.isfinite(preseat_mm) or preseat_mm < 0.0:
        raise ValueError("preseat_mm must be finite and nonnegative")
    start = DOCK_CAM_CAPTURE_RECENTER_START_PRESEAT_MM
    end = DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM
    if preseat_mm >= start:
        return ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM
    if preseat_mm <= end:
        return 0.0
    return ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM * (
        (preseat_mm - end) / (start - end)
    )


def positive_lock_cam_capture_q_max_mm(preseat_mm: float) -> float:
    """Return the frictionless cam-contact upper bound on slider travel.

    ``q=0`` is the nominal unlocked datum.  The cam's intentional 0.05 mm
    tab clearance lets the passive equilibrium sit at ``q=0.05`` while still
    remaining inside the released keyhole-entry clearance.
    """

    lateral = positive_lock_cam_capture_lateral_offset_mm(preseat_mm)
    if preseat_mm >= DOCK_CAM_CAPTURE_RECENTER_START_PRESEAT_MM:
        return SLIDER_TRAVEL
    if preseat_mm <= DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM:
        return DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM
    return max(
        DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM,
        min(SLIDER_TRAVEL, preseat_mm - 3.15 - lateral),
    )


def positive_lock_cam_release_q_max_mm(withdrawal_mm: float) -> float:
    """Return the seated X/Y wedge's passive q envelope during -Y exit."""

    if not math.isfinite(withdrawal_mm) or withdrawal_mm < 0.0:
        raise ValueError("withdrawal_mm must be finite and nonnegative")
    wedge_slope = (DOCK_CAM_X_OUTER_MIN - DOCK_CAM_X_INNER) / (
        DOCK_CAM_Y_MAX - DOCK_CAM_Y_MIN
    )
    return max(
        DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM,
        min(
            SLIDER_TRAVEL,
            DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM
            + wedge_slope
            * (withdrawal_mm - DOCK_CAM_PASSIVE_RELEASE_INITIAL_DWELL_MM),
        ),
    )


def positive_lock_cam_contract() -> dict[str, object]:
    """Return the complete, executable passive-cam source contract."""

    lead_run = (
        DOCK_CAM_AXIAL_LEAD_X_INNER_LOWER
        - DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER
    )
    lead_rise = DOCK_CAM_AXIAL_LEAD_Z_UPPER - DOCK_CAM_AXIAL_LEAD_Z_LOWER
    land_width = DOCK_CAM_AXIAL_LEAD_Y_MAX - DOCK_CAM_AXIAL_LEAD_Y_MIN
    lower_wall = (
        DOCK_CAM_AXIAL_LEAD_X_OUTER
        - DOCK_CAM_AXIAL_LEAD_X_INNER_LOWER
    )
    upper_wall = (
        DOCK_CAM_AXIAL_LEAD_X_OUTER
        - DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER
    )
    lead_volume = land_width * lead_rise * (lower_wall + upper_wall) / 2.0
    hold_volume = (
        land_width
        * upper_wall
        * (DOCK_CAM_AXIAL_HOLD_Z_UPPER - DOCK_CAM_AXIAL_LEAD_Z_UPPER)
    )
    root_widths = [
        bounds[1] - bounds[0]
        for bounds in (
            DOCK_CAM_AXIAL_ROOT_X_BOUNDS,
            DOCK_CAM_AXIAL_ROOT_Y_BOUNDS,
            DOCK_CAM_AXIAL_ROOT_Z_BOUNDS,
        )
    ]
    root_gross_volume = math.prod(root_widths)
    # The symmetric bridge overlaps the hold below z=-4.15/y>0 and the main
    # wedge above z=-4.15/y<0 by equal 0.5 mm3 volumes.
    root_overlap_each = 0.5
    root_net_added_volume = root_gross_volume - 2.0 * root_overlap_each
    lead_angle_deg = math.degrees(math.atan2(lead_rise, lead_run))
    spring_locked_length = (
        -LOCK_STUD_X
        - SLIDER_LOBE_RADIUS
        + SLIDER_TRAVEL
        - RETURN_SPRING_FIXED_X
    )
    spring_unlocked_length = spring_locked_length - SLIDER_TRAVEL
    maximum_spring_force = (
        RETURN_SPRING_FREE_LENGTH - spring_unlocked_length
    ) * RETURN_SPRING_RATE_N_PER_MM
    maximum_cam_normal_force = maximum_spring_force / math.cos(
        math.radians(lead_angle_deg)
    )
    contact_face_area = land_width * math.hypot(lead_run, lead_rise)
    wedge_slope = (DOCK_CAM_X_OUTER_MIN - DOCK_CAM_X_INNER) / (
        DOCK_CAM_Y_MAX - DOCK_CAM_Y_MIN
    )
    q3_release_tangent = DOCK_CAM_PASSIVE_RELEASE_INITIAL_DWELL_MM + (
        SLIDER_TRAVEL - DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM
    ) / wedge_slope
    ramp_contact_start = (
        SLIDER_TRAVEL + 2.95
    ) / (1.0 - ROBOT_CAM_GUIDED_APPROACH_OFFSET_MM / lead_rise)
    return {
        "schema_version": "1.0",
        "frame": "dock_local_mm",
        "construction": (
            "union_main_xy_wedge_ruled_axial_lead_hold_finger"
        ),
        "main_xy_wedge": {
            "polygon_xy_mm": [
                [DOCK_CAM_X_OUTER_MIN, DOCK_CAM_Y_MIN],
                [DOCK_CAM_X_OUTER_MAX, DOCK_CAM_Y_MIN],
                [DOCK_CAM_X_OUTER_MAX, DOCK_CAM_Y_MAX],
                [DOCK_CAM_X_INNER, DOCK_CAM_Y_MAX],
            ],
            "z_bounds_mm": [
                DOCK_CAM_Z_MIN,
                DOCK_CAM_Z_MIN + DOCK_CAM_THICKNESS,
            ],
        },
        "axial_lead": {
            "kind": "ruled_loft_between_rectangles",
            "lower_rectangle_mm": {
                "x": [
                    DOCK_CAM_AXIAL_LEAD_X_INNER_LOWER,
                    DOCK_CAM_AXIAL_LEAD_X_OUTER,
                ],
                "y": [DOCK_CAM_AXIAL_LEAD_Y_MIN, DOCK_CAM_AXIAL_LEAD_Y_MAX],
                "z": DOCK_CAM_AXIAL_LEAD_Z_LOWER,
            },
            "upper_rectangle_mm": {
                "x": [
                    DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER,
                    DOCK_CAM_AXIAL_LEAD_X_OUTER,
                ],
                "y": [DOCK_CAM_AXIAL_LEAD_Y_MIN, DOCK_CAM_AXIAL_LEAD_Y_MAX],
                "z": DOCK_CAM_AXIAL_LEAD_Z_UPPER,
            },
            "run_mm": lead_run,
            "rise_mm": lead_rise,
            "contact_angle_deg": lead_angle_deg,
            "minimum_wall_mm": lower_wall,
            "contact_land_width_mm": land_width,
            "volume_mm3": lead_volume,
        },
        "hold_finger": {
            "kind": "axis_aligned_box",
            "bounds_mm": {
                "x": [
                    DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER,
                    DOCK_CAM_AXIAL_LEAD_X_OUTER,
                ],
                "y": [DOCK_CAM_AXIAL_LEAD_Y_MIN, DOCK_CAM_AXIAL_LEAD_Y_MAX],
                "z": [
                    DOCK_CAM_AXIAL_LEAD_Z_UPPER,
                    DOCK_CAM_AXIAL_HOLD_Z_UPPER,
                ],
            },
            "volume_mm3": hold_volume,
        },
        "outer_root_bridge": {
            "kind": "axis_aligned_box",
            "bounds_mm": {
                "x": list(DOCK_CAM_AXIAL_ROOT_X_BOUNDS),
                "y": list(DOCK_CAM_AXIAL_ROOT_Y_BOUNDS),
                "z": list(DOCK_CAM_AXIAL_ROOT_Z_BOUNDS),
            },
            "gross_volume_mm3": root_gross_volume,
            "overlap_with_hold_mm3": root_overlap_each,
            "overlap_with_main_wedge_mm3": root_overlap_each,
            "net_added_volume_mm3": root_net_added_volume,
            "outside_locked_tab_swept_x": True,
        },
        "expected_geometry": {
            "added_volume_mm3": (
                lead_volume + hold_volume + root_net_added_volume
            ),
            "total_volume_mm3": 325.435,
            "bounds_mm": {
                "x": [DOCK_CAM_X_INNER, DOCK_CAM_X_OUTER_MAX],
                "y": [DOCK_CAM_Y_MIN, DOCK_CAM_AXIAL_LEAD_Y_MAX],
                "z": [
                    DOCK_CAM_AXIAL_LEAD_Z_LOWER,
                    DOCK_CAM_Z_MIN + DOCK_CAM_THICKNESS,
                ],
            },
        },
        "passive_capture": {
            "preseat_definition": (
                "robot_source_translation_dock_local_z_mm=-preseat_mm"
            ),
            "lateral_axis": "dock_local_positive_x",
            "recenter_start_preseat_mm": (
                DOCK_CAM_CAPTURE_RECENTER_START_PRESEAT_MM
            ),
            "recenter_end_preseat_mm": (
                DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM
            ),
            "lateral_offset_breakpoints_mm": [
                [DOCK_CAM_CAPTURE_RECENTER_START_PRESEAT_MM, 0.20],
                [DOCK_CAM_CAPTURE_RECENTER_END_PRESEAT_MM, 0.0],
                [0.0, 0.0],
            ],
            "ramp_contact_start_preseat_mm": ramp_contact_start,
            "head_entry_tangent_preseat_mm": DOCK_CAM_HEAD_ENTRY_PRESEAT_MM,
            "nominal_unlocked_q_mm": 0.0,
            "passive_open_q_max_mm": DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM,
            "ramp_q_affine_coefficients": {
                "preseat": 1.0,
                "lateral_offset": -1.0,
                "constant": -3.15,
                "clamp_mm": [DOCK_CAM_PASSIVE_OPEN_Q_MAX_MM, SLIDER_TRAVEL],
            },
        },
        "passive_release": {
            "axis": "dock_local_negative_y",
            "initial_q_hold_withdrawal_mm": (
                DOCK_CAM_PASSIVE_RELEASE_INITIAL_DWELL_MM
            ),
            "q_per_withdrawal_slope": wedge_slope,
            "q3_tangent_withdrawal_mm": q3_release_tangent,
            "nominal_exit_withdrawal_mm": 15.0,
            "required_exit_clearance_mm": 0.20,
        },
        "manufacturability": {
            "minimum_feature_mm": min(
                lower_wall, land_width, *root_widths
            ),
            "declared_process_floor_mm": 0.20,
            "lead_is_self_supporting_at_45_deg": lead_angle_deg == 45.0,
        },
        "quasistatic_load_envelope": {
            "return_spring_rate_n_per_mm": RETURN_SPRING_RATE_N_PER_MM,
            "maximum_spring_deflection_mm": (
                RETURN_SPRING_FREE_LENGTH - spring_unlocked_length
            ),
            "maximum_spring_force_n": maximum_spring_force,
            "maximum_axial_reaction_n": maximum_spring_force
            * math.tan(math.radians(lead_angle_deg)),
            "maximum_cam_normal_force_n": maximum_cam_normal_force,
            "contact_face_area_mm2": contact_face_area,
            "mean_contact_pressure_mpa": (
                maximum_cam_normal_force / contact_face_area
            ),
            "semantics": "frictionless_quasistatic_source_envelope",
        },
    }


def positive_lock_keyhole_contract() -> dict[str, object]:
    """Return the released shoulder-clearance and head-retention geometry."""

    neck_radius = KEYHOLE_NECK_WIDTH / 2.0
    return {
        "slider_travel_mm": SLIDER_TRAVEL,
        "slider_locked_translation_mm": [SLIDER_TRAVEL, 0.0, 0.0],
        "stud_centres_x_mm": [-LOCK_STUD_X, LOCK_STUD_X],
        "keyhole_entry_diameter_mm": KEYHOLE_ENTRY_DIAMETER,
        "shoulder_path_kind": "capsule",
        "shoulder_path_centerline_x_offsets_mm": [-SLIDER_TRAVEL, 0.0],
        "shoulder_path_overall_x_offsets_mm": [
            -SLIDER_TRAVEL - neck_radius,
            neck_radius,
        ],
        "shoulder_path_overall_length_mm": KEYHOLE_NECK_OVERALL_LENGTH,
        "neck_width_mm": KEYHOLE_NECK_WIDTH,
        "neck_radius_mm": neck_radius,
        "stud_shoulder_diameter_mm": LOCK_SHOULDER_DIAMETER,
        "minimum_radial_shoulder_clearance_mm": (
            neck_radius - LOCK_SHOULDER_DIAMETER / 2.0
        ),
        "stud_head_diameter_mm": LOCK_HEAD_DIAMETER,
        "minimum_radial_head_retention_overlap_mm": (
            LOCK_HEAD_DIAMETER / 2.0 - neck_radius
        ),
        "head_to_slider_axial_gap_mm": (
            SLIDER_Z - (PLATE_THICKNESS - LOCK_SHOULDER_LENGTH)
        ),
    }


def robot_cam_relief() -> cq.Workplane:
    """Return the guarded local recess in the fixed robot plate."""

    return box_cutter(
        ROBOT_CAM_RELIEF_X_MAX - ROBOT_CAM_RELIEF_X_MIN,
        ROBOT_CAM_RELIEF_Y_MAX - ROBOT_CAM_RELIEF_Y_MIN,
        ROBOT_CAM_RELIEF_Z_MAX - ROBOT_CAM_RELIEF_Z_MIN,
        (ROBOT_CAM_RELIEF_X_MIN + ROBOT_CAM_RELIEF_X_MAX) / 2.0,
        (ROBOT_CAM_RELIEF_Y_MIN + ROBOT_CAM_RELIEF_Y_MAX) / 2.0,
        ROBOT_CAM_RELIEF_Z_MIN,
    )


def core_dock_stop() -> cq.Workplane:
    """Build the physical core stop from its public collision contract."""

    spec = core_dock_stop_spec()
    center = spec["center_mm"]
    size = spec["size_mm"]
    stop = (
        cq.Workplane("XY")
        .box(*size, centered=True)
        .translate(tuple(center))
    )
    for hole in spec["through_holes"]:
        stop = stop.cut(
            axis_cylinder(
                hole["diameter_mm"],
                size[1] + 0.2,
                (hole["x_mm"], hole["y_start_mm"], hole["z_mm"]),
                tuple(hole["axis"]),
            )
        )
    return stop.clean()


def _positive_lock_cam_main_wedge() -> cq.Workplane:
    """Build the seated/rack-exit X/Y wedge without the axial lead."""

    return (
        cq.Workplane("XY")
        .polyline(
            [
                (DOCK_CAM_X_OUTER_MIN, DOCK_CAM_Y_MIN),
                (DOCK_CAM_X_OUTER_MAX, DOCK_CAM_Y_MIN),
                (DOCK_CAM_X_OUTER_MAX, DOCK_CAM_Y_MAX),
                (DOCK_CAM_X_INNER, DOCK_CAM_Y_MAX),
            ]
        )
        .close()
        .extrude(DOCK_CAM_THICKNESS)
        .translate((0.0, 0.0, DOCK_CAM_Z_MIN))
        .clean()
    )


def _rectangle_wire_at_z(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_value: float,
) -> cq.Wire:
    """Return a deterministic closed rectangle wire in the dock frame."""

    return cq.Wire.makePolygon(
        [
            cq.Vector(x_min, y_min, z_value),
            cq.Vector(x_max, y_min, z_value),
            cq.Vector(x_max, y_max, z_value),
            cq.Vector(x_min, y_max, z_value),
        ],
        close=True,
    )


def positive_lock_cam_axial_lead() -> cq.Workplane:
    """Build the narrow ruled lead and vertical hold finger.

    The lower and upper rectangles share their outer X and Y bounds.  Their
    inner X values differ by exactly the 3.2 mm axial run, producing the
    source-authoritative 45-degree tab-contact plane without a faceted or
    mesh-derived approximation.
    """

    lower = _rectangle_wire_at_z(
        DOCK_CAM_AXIAL_LEAD_X_INNER_LOWER,
        DOCK_CAM_AXIAL_LEAD_X_OUTER,
        DOCK_CAM_AXIAL_LEAD_Y_MIN,
        DOCK_CAM_AXIAL_LEAD_Y_MAX,
        DOCK_CAM_AXIAL_LEAD_Z_LOWER,
    )
    upper = _rectangle_wire_at_z(
        DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER,
        DOCK_CAM_AXIAL_LEAD_X_OUTER,
        DOCK_CAM_AXIAL_LEAD_Y_MIN,
        DOCK_CAM_AXIAL_LEAD_Y_MAX,
        DOCK_CAM_AXIAL_LEAD_Z_UPPER,
    )
    ruled_lead = cq.Workplane(obj=cq.Solid.makeLoft([lower, upper], ruled=True))
    hold_finger = (
        cq.Workplane("XY")
        .box(
            DOCK_CAM_AXIAL_LEAD_X_OUTER
            - DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER,
            DOCK_CAM_AXIAL_LEAD_Y_MAX - DOCK_CAM_AXIAL_LEAD_Y_MIN,
            DOCK_CAM_AXIAL_HOLD_Z_UPPER - DOCK_CAM_AXIAL_LEAD_Z_UPPER,
            centered=True,
        )
        .translate(
            (
                (
                    DOCK_CAM_AXIAL_LEAD_X_INNER_UPPER
                    + DOCK_CAM_AXIAL_LEAD_X_OUTER
                )
                / 2.0,
                (DOCK_CAM_AXIAL_LEAD_Y_MIN + DOCK_CAM_AXIAL_LEAD_Y_MAX)
                / 2.0,
                (DOCK_CAM_AXIAL_LEAD_Z_UPPER + DOCK_CAM_AXIAL_HOLD_Z_UPPER)
                / 2.0,
            )
        )
    )
    root_bridge = (
        cq.Workplane("XY")
        .box(
            DOCK_CAM_AXIAL_ROOT_X_BOUNDS[1]
            - DOCK_CAM_AXIAL_ROOT_X_BOUNDS[0],
            DOCK_CAM_AXIAL_ROOT_Y_BOUNDS[1]
            - DOCK_CAM_AXIAL_ROOT_Y_BOUNDS[0],
            DOCK_CAM_AXIAL_ROOT_Z_BOUNDS[1]
            - DOCK_CAM_AXIAL_ROOT_Z_BOUNDS[0],
            centered=True,
        )
        .translate(
            tuple(
                (bounds[0] + bounds[1]) / 2.0
                for bounds in (
                    DOCK_CAM_AXIAL_ROOT_X_BOUNDS,
                    DOCK_CAM_AXIAL_ROOT_Y_BOUNDS,
                    DOCK_CAM_AXIAL_ROOT_Z_BOUNDS,
                )
            )
        )
    )
    return ruled_lead.union(hold_finger).union(root_bridge).clean()


def positive_lock_cam() -> cq.Workplane:
    """Build the complete passive X/Y wedge plus axial lead authority."""

    return _positive_lock_cam_main_wedge().union(
        positive_lock_cam_axial_lead()
    ).clean()


def robot_plate() -> cq.Workplane:
    plate = rounded_plate(ROBOT_WIDTH, ROBOT_HEIGHT, PLATE_THICKNESS, CORNER_RADIUS)
    wing = cq.Workplane("XY").box(
        ELECTRICAL_WING_X_MAX - ELECTRICAL_WING_X_MIN,
        ELECTRICAL_WING_HEIGHT,
        PLATE_THICKNESS,
        centered=(True, True, False),
    ).translate(((ELECTRICAL_WING_X_MIN + ELECTRICAL_WING_X_MAX) / 2, 0, 0))
    plate = plate.union(wing)

    # Screws install from the coupling face into the existing wrist horn.  The
    # counterbores end at the internal-lock roof, so they do not block travel.
    plate = (
        plate.faces(">Z")
        .workplane()
        .pushPoints(horn_points())
        .cboreHole(HORN_CLEARANCE_DIAMETER, 6.2, 3.0)
    )
    plate = plate.faces("<Z").workplane().hole(7.0, depth=2.2)

    for x, y in magnet_points():
        plate = plate.cut(
            square_cutter(
                MAGNET_POCKET_WIDTH,
                MAGNET_POCKET_DEPTH,
                x,
                y,
                PLATE_THICKNESS - MAGNET_POCKET_DEPTH,
            )
        )
        plate = plate.cut(cylinder_cutter(3.4, PLATE_THICKNESS + 0.2, x, y))
        plate = plate.cut(hex_cutter(5.7, 2.6, x, y))

    plate = plate.cut(slider_track_envelope())

    # The dock cam retains its full 3 mm unlock datum.  A local fixed-plate
    # relief supplies printable clearance without altering the slider, keyhole
    # alignment, spring stroke or cam engagement.
    plate = plate.cut(robot_cam_relief())

    # Fixed entry/head-clearance wells for the two shoulder screws.
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        plate = plate.cut(
            cylinder_cutter(
                ROBOT_STUD_WELL_DIAMETER,
                PLATE_THICKNESS - ROBOT_STUD_WELL_BOTTOM_Z + 0.1,
                x,
                0,
                ROBOT_STUD_WELL_BOTTOM_Z,
            )
        )

    # Side-bore for the catalog return spring.  At lock the spring is 9.4 mm;
    # at unlock it is 6.4 mm, retaining 0.4 mm margin to its catalog limit.
    spring_channel_length = 10.4
    plate = plate.cut(
        axis_cylinder(
            RETURN_SPRING_OD + 0.35,
            spring_channel_length,
            (RETURN_SPRING_FIXED_X, 0, RETURN_SPRING_CENTER_Z),
            (1, 0, 0),
        )
    )

    # Flush M2x6 guide/retaining screw: clearance above, pilot only in the base.
    plate = plate.cut(cylinder_cutter(2.3, PLATE_THICKNESS - SLIDER_SLOT_TOP + 0.1, 0, 0, SLIDER_SLOT_TOP))
    plate = plate.cut(cylinder_cutter(4.2, 1.25, 0, 0, PLATE_THICKNESS - 1.2))
    plate = plate.cut(cylinder_cutter(1.7, 2.0, 0, 0, SLIDER_SLOT_BOTTOM - 2.0))

    # Legacy visual pilot only. The official barb/knurl sectional mounting bore
    # remains a release blocker and this cut is not a fabrication prescription.
    for x, y in pogo_points():
        plate = plate.cut(
            cylinder_cutter(
                POGO_LEGACY_REFERENCE_PILOT_DIAMETER,
                PLATE_THICKNESS + 0.2,
                x,
                y,
            )
        )
        plate = plate.cut(
            cylinder_cutter(
                POGO_PAD_RELIEF_DIAMETER,
                POGO_PAD_RELIEF_DEPTH + 0.1,
                x,
                y,
                PLATE_THICKNESS - POGO_PAD_RELIEF_DEPTH,
            )
        )
    rear_wiring_pocket = box_cutter(8.0, 20.0, 1.9, CONTACT_CENTER_X, 0, -0.05)
    plate = plate.cut(rear_wiring_pocket)

    for x in (-LOCATOR_X, LOCATOR_X):
        pin = cq.Solid.makeCone(
            LOCATOR_BASE_DIAMETER / 2,
            LOCATOR_TIP_DIAMETER / 2,
            LOCATOR_HEIGHT,
            cq.Vector(x, 0, PLATE_THICKNESS),
            cq.Vector(0, 0, 1),
        )
        plate = plate.union(cq.Workplane(obj=pin))
    return plate.clean()


def locator_socket(x: float, y_offset: float = 0.0) -> cq.Workplane:
    cone = cq.Solid.makeCone(
        LOCATOR_BASE_DIAMETER / 2 + SOCKET_CLEARANCE,
        LOCATOR_TIP_DIAMETER / 2 + SOCKET_CLEARANCE,
        LOCATOR_HEIGHT + 0.35,
        cq.Vector(x, y_offset, -0.05),
        cq.Vector(0, 0, 1),
    )
    return cq.Workplane(obj=cone)


def base_tool_plate() -> cq.Workplane:
    plate = rounded_plate(TOOL_WIDTH, TOOL_HEIGHT, PLATE_THICKNESS, CORNER_RADIUS)
    wing = cq.Workplane("XY").box(
        -28.0 - ELECTRICAL_WING_X_MIN,
        ELECTRICAL_WING_HEIGHT,
        PLATE_THICKNESS,
        centered=(True, True, False),
    ).translate(((ELECTRICAL_WING_X_MIN - 28.0) / 2, 0, 0))
    return plate.union(wing).clean()


def tool_plate(stock_gripper: bool = False) -> cq.Workplane:
    plate = base_tool_plate()

    for x, y in magnet_points():
        plate = plate.cut(square_cutter(TARGET_POCKET_WIDTH, TARGET_POCKET_DEPTH, x, y, 0))
        plate = plate.cut(cylinder_cutter(5.8, PLATE_THICKNESS + 0.2, x, y))
        # Rear-loaded DIN 934 M5 nut for an ISO 10642 M5x10 countersunk
        # screw.  The attached tool body keeps the nut captive.
        plate = plate.cut(
            hex_cutter(8.3, 4.1, x, y, PLATE_THICKNESS - 4.05)
        )

    plate = plate.cut(locator_socket(-LOCATOR_X))
    plate = plate.cut(locator_socket(LOCATOR_X, -RELIEVED_SOCKET_Y_CLEARANCE))
    plate = plate.cut(locator_socket(LOCATOR_X, RELIEVED_SOCKET_Y_CLEARANCE))

    # The selected shoulder screw has a 4 mm thread.  Rear-loaded DIN 934 M3
    # nuts drop to a 1.5 mm printed floor, placing their full 2.4 mm thickness
    # within that thread reach.  The attached tool body retains the nuts.
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        plate = plate.cut(cylinder_cutter(3.2, PLATE_THICKNESS + 0.2, x, 0))
        plate = plate.cut(
            hex_cutter(
                LOCK_NUT_POCKET_ACROSS_FLATS,
                PLATE_THICKNESS - LOCK_NUT_POCKET_FLOOR + 0.05,
                x,
                0,
                LOCK_NUT_POCKET_FLOOR,
            )
        )

    # Recess for a 1.0 mm target PCB.  A narrow rear window aligns with four
    # plated wire holes, while the opposite side remains a retaining ledge.
    plate = plate.cut(
        box_cutter(
            CONTACT_BOARD_WIDTH + 0.25,
            CONTACT_BOARD_HEIGHT + 0.25,
            CONTACT_BOARD_THICKNESS + 0.05,
            CONTACT_CENTER_X,
            0,
            -0.05,
        )
    )
    plate = plate.cut(
        box_cutter(
            3.0,
            18.0,
            PLATE_THICKNESS - CONTACT_BOARD_THICKNESS + 0.1,
            CONTACT_CENTER_X + CONTACT_WIRE_PAD_OFFSET_X,
            0,
            CONTACT_BOARD_THICKNESS,
        )
    )

    if stock_gripper:
        plate = (
            plate.faces(">Z")
            .workplane()
            .pushPoints(horn_points())
            .hole(STOCK_GRIPPER_INSERT_HOLE_DIAMETER, depth=STOCK_GRIPPER_INSERT_DEPTH)
        )
    else:
        tool_points = [
            (-TOOL_MOUNT_X, -TOOL_MOUNT_Y),
            (-TOOL_MOUNT_X, TOOL_MOUNT_Y),
            (TOOL_MOUNT_X, -TOOL_MOUNT_Y),
            (TOOL_MOUNT_X, TOOL_MOUNT_Y),
        ]
        plate = plate.faces(">Z").workplane().pushPoints(tool_points).hole(3.3)
        plate = plate.faces(">Z").workplane().hole(8.0)
    return plate.clean()


def tool_dock() -> cq.Workplane:
    rail_length = 76.0
    rail_center_y = -2.0
    dock = None
    rail_data = (
        (-1, -39.5, -37.0, -43.0),
        (1, 31.5, 29.0, 35.0),
    )
    for _, lower_x, upper_x, wall_x in rail_data:
        lower = cq.Workplane("XY").box(7.0, rail_length, 3.0, centered=True).translate(
            (lower_x, rail_center_y, -1.5)
        )
        upper = cq.Workplane("XY").box(8.0, rail_length, 3.0, centered=True).translate(
            (upper_x, rail_center_y, PLATE_THICKNESS + 1.5)
        )
        wall = cq.Workplane("XY").box(
            4.0, rail_length, PLATE_THICKNESS + 6.0, centered=True
        ).translate((wall_x, rail_center_y, PLATE_THICKNESS / 2))
        side = lower.union(upper).union(wall)
        dock = side if dock is None else dock.union(side)

    dock = dock.union(core_dock_stop())

    # Passive wedge: while the coupled tool slides the final 12 mm into the
    # rack, this surface pushes the protruding slider tab 3 mm left to unlock.
    # On withdrawal the catalog spring returns the slider to positive lock.
    dock = dock.union(positive_lock_cam())
    return dock.clean()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_step_header(path: Path) -> None:
    """Remove OCCT's wall-clock timestamp from one generated STEP file."""

    text = path.read_text()
    canonical, replacements = re.subn(
        r"(FILE_NAME\('[^']*',)'[^']+'",
        r"\1'1970-01-01T00:00:00'",
        text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"could not canonicalize STEP header timestamp: {path}")
    path.write_text(canonical)


def _canonicalize_dxf_metadata(path: Path) -> None:
    """Remove ezdxf's random metadata and unordered class emission.

    The drawing entities are deterministic, but ezdxf intentionally assigns
    fresh document GUIDs and timestamps on every write.  Its required-class
    registry can also emit otherwise identical CLASS records in hash-order.
    None of those values affect fabrication geometry, but each would otherwise
    prevent hash-closed regeneration.
    """

    text = path.read_text()
    substitutions = (
        (
            r"(\$FINGERPRINTGUID\s*\n\s*2\s*\n)\{[^}\n]+\}",
            r"\1{00000000-0000-0000-0000-000000000001}",
        ),
        (
            r"(\$VERSIONGUID\s*\n\s*2\s*\n)\{[^}\n]+\}",
            r"\1{00000000-0000-0000-0000-000000000002}",
        ),
        (
            r"(1\.4\.4 @ )[0-9T:+.-]+",
            r"\g<1>1970-01-01T00:00:00+00:00",
        ),
        (
            r"(\$TDCREATE\s*\n\s*40\s*\n)[0-9.]+",
            r"\g<1>2440587.5",
        ),
        (
            r"(\$TDUPDATE\s*\n\s*40\s*\n)[0-9.]+",
            r"\g<1>2440587.5",
        ),
    )
    expected_counts = (1, 1, 2, 1, 1)
    for (pattern, replacement), expected in zip(substitutions, expected_counts):
        text, replacements = re.subn(pattern, replacement, text)
        if replacements != expected:
            raise RuntimeError(
                f"could not canonicalize DXF metadata ({replacements} != {expected}): "
                f"{path}"
            )

    classes_match = re.search(
        r"(  0\nSECTION\n  2\nCLASSES\n)(.*?)(?=  0\nENDSEC)",
        text,
        flags=re.DOTALL,
    )
    if classes_match is None:
        raise RuntimeError(f"DXF CLASSES section is absent: {path}")
    classes_body = classes_match.group(2)
    class_blocks = re.findall(
        r"  0\nCLASS\n.*?(?=  0\nCLASS\n|\Z)",
        classes_body,
        flags=re.DOTALL,
    )
    if len(class_blocks) < 2 or "".join(class_blocks) != classes_body:
        raise RuntimeError(f"DXF CLASS records could not be partitioned: {path}")

    def class_name(block: str) -> str:
        match = re.search(r"\n  1\n([^\n]+)", block)
        if match is None:
            raise RuntimeError(f"DXF CLASS name is absent: {path}")
        return match.group(1)

    canonical_classes = "".join(sorted(class_blocks, key=class_name))
    text = (
        text[: classes_match.start(2)]
        + canonical_classes
        + text[classes_match.end(2) :]
    )
    path.write_text(text)


def _artifact_role(name: str) -> str:
    if name.endswith("_assembly.step"):
        return "complete_assembly_step"
    if name.endswith(".step"):
        return "exact_cad_step"
    if name.endswith(".stl"):
        return "tessellated_mesh_export"
    if name.endswith(".dxf"):
        return "fabrication_profile_dxf"
    if name.endswith(".kicad_pcb"):
        return "editable_pcb_source"
    if name.endswith(".svg"):
        return "pcb_fabrication_drawing"
    if name == "bill_of_materials.csv":
        return "bill_of_materials"
    if name == "electrical_pinout.csv":
        return "electrical_pinout"
    if name == "design_parameters.json":
        return "design_parameters"
    if name == "engineering_check.json":
        return "engineering_check"
    raise RuntimeError(f"unclassified core CAD artifact: {name}")


def write_core_manifest(output_dir: Path = EXPORT_DIR) -> dict[str, object]:
    """Hash-close every generated core artifact without self-reference."""

    missing = [name for name in CORE_OUTPUT_NAMES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"core CAD generation did not emit required files: {missing}")
    files = [
        {
            "path": f"QuickChange/SO101_Magnetic/exports/{name}",
            "role": _artifact_role(name),
            "bytes": (output_dir / name).stat().st_size,
            "sha256": _sha256(output_dir / name),
        }
        for name in sorted(CORE_OUTPUT_NAMES)
    ]
    inventory_payload = [
        {key: record[key] for key in ("path", "role", "bytes", "sha256")}
        for record in files
    ]
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "units": "mm",
        "generator": {
            "path": "QuickChange/SO101_Magnetic/generate_cad.py",
            "bytes": Path(__file__).stat().st_size,
            "sha256": _sha256(Path(__file__)),
        },
        "contracts": {
            "robot_plate_cam_relief": robot_cam_relief_contract(),
            "interface_hardware_fit": interface_hardware_fit_contract(),
            "core_dock_stop": core_dock_stop_spec(),
            "stock_gripper_mount": stock_gripper_mount_contract(),
            "positive_lock_keyhole": positive_lock_keyhole_contract(),
            "positive_lock_cam": positive_lock_cam_contract(),
        },
        "files": files,
        "file_count": len(files),
        "inventory_sha256": _canonical_json_sha256(inventory_payload),
    }
    (output_dir / CORE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest


def export_part(shape: cq.Workplane, stem: str) -> None:
    step_path = EXPORT_DIR / f"{stem}.step"
    cq.exporters.export(shape, str(step_path))
    _canonicalize_step_header(step_path)
    cq.exporters.export(
        shape,
        str(EXPORT_DIR / f"{stem}.stl"),
        tolerance=0.08,
        angularTolerance=0.12,
    )


def export_reference_step(shape: cq.Workplane, stem: str) -> None:
    """Export exact reference hardware without an untracked mesh derivative."""

    step_path = EXPORT_DIR / f"{stem}.step"
    cq.exporters.export(shape, str(step_path))
    _canonicalize_step_header(step_path)


def _strip_assembly_colors(assembly: cq.Assembly) -> None:
    """Keep named assembly structure while avoiding unstable STEP styles.

    OCCT's geometric/name section is deterministic, while its colored STEP
    presentation entity ordering is not.  Release assemblies therefore retain
    names and placements but deliberately omit decorative per-part colors.
    """

    for component in assembly.objects.values():
        component.color = None


def add_hardware(assembly: cq.Assembly, include_studs: bool = True) -> None:
    magnet = screw_on_magnet()
    target = steel_target()
    for index, (x, y) in enumerate(magnet_points(), start=1):
        assembly.add(
            magnet,
            name=f"magnet_{index}_{MAGNET_PART_NUMBER}",
            loc=cq.Location(
                cq.Vector(
                    x,
                    y,
                    PLATE_THICKNESS
                    - MAGNET_HEIGHT
                    - MAGNETIC_HARDWARE_FACE_RECESS,
                )
            ),
            color=cq.Color(0.48, 0.12, 0.62),
        )
        assembly.add(
            target,
            name=f"target_{index}_{TARGET_PART_NUMBER}",
            loc=cq.Location(
                cq.Vector(
                    x,
                    y,
                    PLATE_THICKNESS + MAGNETIC_HARDWARE_FACE_RECESS,
                )
            ),
            color=cq.Color(0.72, 0.75, 0.78),
        )

    assembly.add(
        locking_slider(),
        name="positive_lock_slider_locked",
        loc=cq.Location(cq.Vector(SLIDER_TRAVEL, 0, SLIDER_Z)),
        color=cq.Color(0.10, 0.75, 0.34),
    )

    locked_spring_length = (
        -LOCK_STUD_X - SLIDER_LOBE_RADIUS + SLIDER_TRAVEL - RETURN_SPRING_FIXED_X
    )
    assembly.add(
        compression_spring(locked_spring_length),
        name=f"return_spring_{RETURN_SPRING_PART_NUMBER}",
        loc=cq.Location(cq.Vector(RETURN_SPRING_FIXED_X, 0, RETURN_SPRING_CENTER_Z)),
        color=cq.Color(0.78, 0.80, 0.83),
    )

    if include_studs:
        for index, x in enumerate((-LOCK_STUD_X, LOCK_STUD_X), start=1):
            assembly.add(
                shoulder_lock_stud(),
                name=f"shoulder_lock_stud_{index}_McMaster_90318A720",
                loc=cq.Location(cq.Vector(x, 0, PLATE_THICKNESS)),
                color=cq.Color(0.75, 0.77, 0.80),
            )
            assembly.add(
                lock_stud_nut(),
                name=f"lock_stud_nut_{index}_DIN934_M3",
                loc=cq.Location(
                    cq.Vector(x, 0, PLATE_THICKNESS + LOCK_NUT_POCKET_FLOOR)
                ),
                color=cq.Color(0.69, 0.71, 0.74),
            )

    pin = pogo_reference_pin()
    for index, ((x, y), signal) in enumerate(zip(pogo_points(), CONTACT_SIGNALS), start=1):
        protrusion = POGO_GROUND_PROTRUSION if signal == "GND" else POGO_STANDARD_PROTRUSION
        z = PLATE_THICKNESS + protrusion - POGO_OVERALL_LENGTH
        assembly.add(
            pin,
            name=f"P{index}_{signal}_{POGO_PART_NUMBER}",
            loc=cq.Location(cq.Vector(x, y, z)),
            color=cq.Color(0.91, 0.68, 0.14),
        )

    assembly.add(
        contact_board(),
        name="tool_contact_board_FR4",
        loc=cq.Location(cq.Vector(CONTACT_CENTER_X, 0, PLATE_THICKNESS)),
        color=cq.Color(0.05, 0.42, 0.20),
    )
    pad = contact_pad()
    for index, ((x, y), signal) in enumerate(zip(pogo_points(), CONTACT_SIGNALS), start=1):
        assembly.add(
            pad,
            name=f"target_pad_P{index}_{signal}",
            loc=cq.Location(cq.Vector(x, y, PLATE_THICKNESS - 0.05)),
            color=cq.Color(0.95, 0.72, 0.08),
        )


def save_assemblies(robot: cq.Workplane, generic: cq.Workplane, stock: cq.Workplane) -> None:
    assembly = cq.Assembly(name="so101_powered_magnetic_quick_change_v0_2")
    assembly.add(robot, name="robot_plate", color=cq.Color(0.95, 0.70, 0.10))
    assembly.add(
        generic,
        name="generic_tool_plate",
        loc=cq.Location(cq.Vector(0, 0, PLATE_THICKNESS)),
        color=cq.Color(0.12, 0.42, 0.90),
    )
    add_hardware(assembly)
    _strip_assembly_colors(assembly)
    quick_change_path = EXPORT_DIR / "so101_quick_change_assembly.step"
    assembly.save(str(quick_change_path))
    _canonicalize_step_header(quick_change_path)

    retrofit = cq.Assembly(name="so101_stock_gripper_powered_retrofit_v0_2")
    retrofit.add(robot, name="robot_plate", color=cq.Color(0.95, 0.70, 0.10))
    retrofit.add(
        stock,
        name="stock_gripper_tool_plate",
        loc=cq.Location(cq.Vector(*STOCK_TOOL_PLATE_ASSEMBLY_POS_MM)),
        color=cq.Color(0.12, 0.42, 0.90),
    )
    add_hardware(retrofit)
    stock_gripper_path = (
        REPOSITORY_ROOT / "STEP/SO101/Follower_Specific/Wrist_Roll_Follower_SO101.step"
    )
    if stock_gripper_path.exists():
        stock_gripper = cq.importers.importStep(str(stock_gripper_path))
        retrofit.add(
            stock_gripper,
            name="official_stock_gripper_body_reference",
            loc=cq.Location(cq.Vector(*STOCK_FIXED_STEP_ASSEMBLY_POS_MM)),
            color=cq.Color(0.90, 0.74, 0.12),
        )
    _strip_assembly_colors(retrofit)
    retrofit_path = EXPORT_DIR / "so101_stock_gripper_retrofit_assembly.step"
    retrofit.save(str(retrofit_path))
    _canonicalize_step_header(retrofit_path)


def write_contact_board_files() -> None:
    pinout_path = EXPORT_DIR / "electrical_pinout.csv"
    with pinout_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pin",
                "target_x_mm",
                "wire_x_mm",
                "y_mm",
                "signal",
                "stock_gripper_use",
                "notes",
            ]
        )
        notes = (
            "first-mate/last-break; official Feetech pin 1",
            "official Feetech pin 2",
            "official Feetech pin 3, half-duplex TTL",
            "optional tool ID/detect; unused by stock three-wire gripper",
        )
        for index, (y, signal, note) in enumerate(zip(CONTACT_Y, CONTACT_SIGNALS, notes), start=1):
            writer.writerow(
                [
                    index,
                    CONTACT_TARGET_PAD_OFFSET_X,
                    CONTACT_WIRE_PAD_OFFSET_X,
                    y,
                    signal,
                    "yes" if index <= 3 else "no",
                    note,
                ]
            )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="120mm" height="120mm" viewBox="-20 -28 40 56">
  <style>text {{ font-family: sans-serif; font-size: 1.55px; }} .dim {{ stroke:#555; stroke-width:.18; fill:none; }} .cu {{ fill:#d9a514; stroke:#6e5100; stroke-width:.18; }} .trace {{ stroke:#d9a514; stroke-width:1.0; }}</style>
  <rect x="{-CONTACT_BOARD_WIDTH/2}" y="{-CONTACT_BOARD_HEIGHT/2}" width="{CONTACT_BOARD_WIDTH}" height="{CONTACT_BOARD_HEIGHT}" rx="0.7" fill="#1b7f4b" stroke="#111" stroke-width="0.25"/>
  <text x="0" y="{-CONTACT_BOARD_HEIGHT/2-2.4}" text-anchor="middle">SO-101 tool target PCB • 1.0 mm FR-4 • ENIG • 2 oz Cu</text>
'''
    for index, (y, signal) in enumerate(zip(CONTACT_Y, CONTACT_SIGNALS), start=1):
        svg += (
            f'  <line class="trace" x1="{CONTACT_TARGET_PAD_OFFSET_X}" y1="{-y}" '
            f'x2="{CONTACT_WIRE_PAD_OFFSET_X}" y2="{-y}"/>'
            f'<circle class="cu" cx="{CONTACT_TARGET_PAD_OFFSET_X}" cy="{-y}" '
            f'r="{CONTACT_PAD_DIAMETER/2}"/>'
            f'<circle class="cu" cx="{CONTACT_WIRE_PAD_OFFSET_X}" cy="{-y}" '
            f'r="{CONTACT_WIRE_PAD_DIAMETER/2}"/>'
            f'<circle cx="{CONTACT_WIRE_PAD_OFFSET_X}" cy="{-y}" '
            f'r="{CONTACT_WIRE_DRILL_DIAMETER/2}" fill="#f7f7f7" stroke="#333" stroke-width=".12"/>'
            f'<text x="6.2" y="{-y+0.55}">P{index} {signal}</text>\n'
        )
    svg += f'''  <path class="dim" d="M {-CONTACT_BOARD_WIDTH/2} 15 v3 M {CONTACT_BOARD_WIDTH/2} 15 v3 M {-CONTACT_BOARD_WIDTH/2} 17 h{CONTACT_BOARD_WIDTH}"/>
  <text x="0" y="20" text-anchor="middle">{CONTACT_BOARD_WIDTH:.1f} mm</text>
  <path class="dim" d="M -7 {-CONTACT_BOARD_HEIGHT/2} h-3 M -7 {CONTACT_BOARD_HEIGHT/2} h-3 M -9 {-CONTACT_BOARD_HEIGHT/2} v{CONTACT_BOARD_HEIGHT}"/>
  <text x="-11" y="0" transform="rotate(-90 -11 0)" text-anchor="middle">{CONTACT_BOARD_HEIGHT:.1f} mm</text>
  <text x="0" y="24" text-anchor="middle">Target Ø{CONTACT_PAD_DIAMETER:.1f} • wire pad Ø{CONTACT_WIRE_PAD_DIAMETER:.1f}/drill Ø{CONTACT_WIRE_DRILL_DIAMETER:.1f} • pitch {CONTACT_PITCH:.1f} mm</text>
  <text x="0" y="26.5" text-anchor="middle">Robot-side view • wires solder through plated holes from rear</text>
</svg>
'''
    (EXPORT_DIR / "tool_contact_board_fab_drawing.svg").write_text(svg)

    # Functional editable KiCad source.  Each ENIG contact target is routed to
    # a rear-accessible plated hole for direct soldering of the tool harness.
    board = '''(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.0))
  (paper "A4")
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (36 "B.SilkS" user "b.silkscreen") (37 "F.SilkS" user "f.silkscreen") (44 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0))
  (net 0 "")
'''
    for index, signal in enumerate(CONTACT_SIGNALS, start=1):
        board += f'  (net {index} "{signal}")\n'
    board += f'''  (gr_rect (start {-CONTACT_BOARD_WIDTH/2} {-CONTACT_BOARD_HEIGHT/2}) (end {CONTACT_BOARD_WIDTH/2} {CONTACT_BOARD_HEIGHT/2}) (stroke (width 0.15) (type default)) (fill none) (layer "Edge.Cuts"))
  (footprint "SO101_TARGET_PADS" (layer "F.Cu") (at 0 0)
    (property "Reference" "J1" (at 0 -13 0) (layer "F.SilkS"))
    (property "Value" "SO101_TOOL_CONTACTS" (at 0 13 0) (layer "F.SilkS") hide)
'''
    for index, (y, signal) in enumerate(zip(CONTACT_Y, CONTACT_SIGNALS), start=1):
        board += (
            f'    (pad "{index}" smd circle (at {CONTACT_TARGET_PAD_OFFSET_X} {y}) '
            f'(size {CONTACT_PAD_DIAMETER} {CONTACT_PAD_DIAMETER}) '
            f'(layers "F.Cu" "F.Mask") (net {index} "{signal}"))\n'
        )
        board += (
            f'    (pad "{index}W" thru_hole circle (at {CONTACT_WIRE_PAD_OFFSET_X} {y}) '
            f'(size {CONTACT_WIRE_PAD_DIAMETER} {CONTACT_WIRE_PAD_DIAMETER}) '
            f'(drill {CONTACT_WIRE_DRILL_DIAMETER}) (layers "*.Cu" "*.Mask") '
            f'(net {index} "{signal}"))\n'
        )
        board += f'    (fp_text user "P{index} {signal}" (at 0 {y + 2.7}) (layer "B.SilkS") (effects (font (size 0.65 0.65) (thickness 0.11)) (justify mirror)))\n'
    board += "  )\n"
    for index, y in enumerate(CONTACT_Y, start=1):
        width = 1.2 if index <= 2 else 0.6
        board += (
            f'  (segment (start {CONTACT_TARGET_PAD_OFFSET_X} {y}) '
            f'(end {CONTACT_WIRE_PAD_OFFSET_X} {y}) (width {width}) '
            f'(layer "F.Cu") (net {index}))\n'
        )
    board += ")\n"
    (EXPORT_DIR / "tool_contact_board.kicad_pcb").write_text(board)


def write_bom() -> None:
    rows = [
        ("robot", 1, "so101_robot_plate", "printed PA12/PA-CF part", "exports/so101_robot_plate.step", "new"),
        ("robot", 2, MAGNET_PART_NUMBER, "12x12x4 mm N35 screw-on magnet", "https://www.supermagnete.de/eng/screw-on-neodymium-magnets/screw-on-block-magnet-12-x-12-x-4mm_CS-Q-12-12-04-N", "new"),
        ("robot", 2, "ISO 10642 M3x10", "countersunk screw for magnet", "standard fastener", "new"),
        ("robot", 2, "DIN 934 M3", "nut for magnet screw", "standard fastener", "new"),
        ("robot", 4, "M3x10 socket-head", "robot plate to existing wrist horn", "standard fastener", "new; verify engagement/no bottoming"),
        ("robot", 1, "304 stainless lock slider", "1.5-1.6 mm sheet", "exports/so101_positive_lock_slider_profile.dxf", "new; STL is fit-check only"),
        ("robot", 1, RETURN_SPRING_PART_NUMBER, "OD4 x L10 mm, 0.98 N/mm return spring", "https://us.misumi-ec.com/vona2/detail/110310903689/?HissuCode=E-GUL4-10", "new"),
        ("robot", 1, "M2x6 flat-head", "slider guide/retainer", "standard fastener", "new"),
        ("robot", 4, POGO_PART_NUMBER, "high-current solder-cup spring pin", "https://www.mill-max.com/products/discrete-spring-loaded-pins/spring-loaded-pin-with-solder-cup-termination/7983/7983-1-15-20-75-14-11-0", "reference only; sectional mounting bore and first-mate shoulder datums unresolved"),
        ("tool", 1, "so101_stock_gripper_tool_plate", "printed PA12/PA-CF part", "exports/so101_stock_gripper_tool_plate.step", "new"),
        ("tool", 2, TARGET_PART_NUMBER, "12x12x3 mm Q235 steel magnet target", "https://www.supermagnete.de/eng/magnet-counterparts-to-screw-on/metal-plates-with-countersunk-hole-12-x-12-x-3mm_MC-12-12-03", "new"),
        ("tool", 2, "ISO 10642 M5x10", "countersunk screw for steel target", "standard fastener", "new"),
        ("tool", 2, "DIN 934 M5", "rear captive nut for steel target", "standard fastener", "new"),
        ("tool", 2, "McMaster 90318A720", "M3; 4 mm dia x 5 mm shoulder pull stud", "https://www.mcmaster.com/90318A720/", "new"),
        ("tool", 2, "DIN 934 M3", "rear captive pull-stud nut, 5.5 AF x 2.4", "https://accu-components.com/us/hexagon-nuts/7888-HPN-M3-A2", "new"),
        ("tool", 1, "SO101 target PCB", "10x22x1.0 mm FR-4, 2 oz, ENIG", "exports/tool_contact_board.kicad_pcb", "fabricate"),
        ("gripper retrofit", 4, "Ruthex RX-M3x5.7", "heat-set insert", "https://www.ruthex.de/en/products/ruthex-gewindeeinsatz-m3-100-stuck-rx-m3x5-7-messing-gewindebuchsen", "new"),
        ("gripper retrofit", 4, "M3x6", "stock gripper to detachable tool plate", "standard fastener", "reuse original"),
        ("gripper retrofit", 1, "3-wire adapter harness", "GND/+12V/TTL; mating stock connectors", "electrical_pinout.csv", "new; do not cut stock arm harness"),
        ("dock", 1, "so101_passive_tool_dock", "printed PA12/PA-CF part", "exports/so101_passive_tool_dock.step", "new"),
    ]
    with (EXPORT_DIR / "bill_of_materials.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["assembly", "quantity", "part", "specification", "source", "status_or_note"])
        writer.writerows(rows)


def engineering_values() -> dict[str, float]:
    magnet_pair_force = 2 * MAGNET_FORCE_EACH_N
    ideal_magnet_pry_moment = magnet_pair_force * (MAGNET_CENTER_Y / 1000)
    edge_reaction_at_stall = SERVO_STALL_TORQUE_N_M / (ROBOT_WIDTH / 2000)
    locked_spring_length = -LOCK_STUD_X - SLIDER_LOBE_RADIUS + SLIDER_TRAVEL - RETURN_SPRING_FIXED_X
    unlocked_spring_length = locked_spring_length - SLIDER_TRAVEL
    return {
        "servo_rated_torque_Nm": round(SERVO_RATED_TORQUE_N_M, 4),
        "servo_peak_stall_torque_Nm": round(SERVO_STALL_TORQUE_N_M, 4),
        "proof_to_rated_ratio": round(SERVO_STALL_TORQUE_N_M / SERVO_RATED_TORQUE_N_M, 1),
        "magnet_pair_catalog_axial_force_N": round(magnet_pair_force, 2),
        "magnet_only_ideal_pry_moment_Nm": round(ideal_magnet_pry_moment, 4),
        "edge_reaction_at_stall_envelope_N": round(edge_reaction_at_stall, 1),
        "approx_reaction_per_lock_stud_N": round(edge_reaction_at_stall / 2, 1),
        "return_spring_locked_length_mm": round(locked_spring_length, 2),
        "return_spring_unlocked_length_mm": round(unlocked_spring_length, 2),
        "return_spring_max_design_deflection_mm": round(RETURN_SPRING_FREE_LENGTH - unlocked_spring_length, 2),
        "return_spring_max_design_force_N": round((RETURN_SPRING_FREE_LENGTH - unlocked_spring_length) * RETURN_SPRING_RATE_N_PER_MM, 2),
        "pogo_force_four_contacts_at_midstroke_N": round(4 * 0.060 * 9.80665, 2),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic core SO-101 quick-change CAD package."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPORT_DIR,
        help="artifact directory; defaults to QuickChange/SO101_Magnetic/exports",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    global EXPORT_DIR
    args = _argument_parser().parse_args(argv)
    EXPORT_DIR = args.output_dir.resolve()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    robot = robot_plate()
    generic = tool_plate(stock_gripper=False)
    stock = tool_plate(stock_gripper=True)
    dock = tool_dock()
    slider = locking_slider()
    magnet = screw_on_magnet()
    target = steel_target()
    shoulder = shoulder_lock_stud()
    lock_nut = lock_stud_nut()
    spring = compression_spring()
    pogo = pogo_reference_pin()
    board = contact_board()

    export_part(robot, "so101_robot_plate")
    export_part(generic, "so101_tool_plate")
    export_part(stock, "so101_stock_gripper_tool_plate")
    export_part(dock, "so101_passive_tool_dock")
    export_part(slider, "so101_positive_lock_slider")
    slider_profile_path = EXPORT_DIR / "so101_positive_lock_slider_profile.dxf"
    cq.exporters.export(slider.faces(">Z"), str(slider_profile_path))
    _canonicalize_dxf_metadata(slider_profile_path)
    export_reference_step(magnet, f"hardware_{MAGNET_PART_NUMBER}")
    export_reference_step(target, f"hardware_{TARGET_PART_NUMBER}")
    export_reference_step(shoulder, "hardware_McMaster_90318A720_shoulder_screw")
    export_reference_step(lock_nut, "hardware_DIN934_M3_lock_stud_nut")
    export_reference_step(spring, f"hardware_{RETURN_SPRING_PART_NUMBER}_reference")
    export_reference_step(pogo, f"hardware_Mill-Max_{POGO_PART_NUMBER}_reference")
    export_part(board, "so101_tool_contact_board_reference")
    save_assemblies(robot, generic, stock)
    write_contact_board_files()
    write_bom()

    checks = engineering_values()
    parameters = {
        "version": "0.2-powered-positive-lock",
        "units": "mm",
        "so101_retrofit": {
            "horn_pattern_square": HORN_HOLE_PATTERN,
            "robot_plate_screws": "4 x M3x10 socket-head",
            "official_stock_gripper_joint_screws": "4 x M3x6",
            "uses_existing_arm_holes": True,
            "permanent_arm_modification": False,
        },
        "robot_plate": [ROBOT_WIDTH, ROBOT_HEIGHT, PLATE_THICKNESS],
        "tool_plate": [TOOL_WIDTH, TOOL_HEIGHT, PLATE_THICKNESS],
        "collision_geometry_contract": {
            "core_dock_stop": core_dock_stop_spec(),
            "stock_gripper_mount": stock_gripper_mount_contract(),
            "interface_hardware_fit": interface_hardware_fit_contract(),
            "positive_lock_keyhole": positive_lock_keyhole_contract(),
            "positive_lock_cam": {
                **positive_lock_cam_contract(),
                "robot_plate_required_clearance_mm": ROBOT_CAM_CLEARANCE_MM,
                "robot_plate_relief_bounds_native_mm": (
                    robot_cam_relief_contract()["bounds_native_mm"]
                ),
                "robot_plate_relief": robot_cam_relief_contract(),
            },
        },
        "magnet": {
            "manufacturer": "Webcraft GmbH / supermagnete",
            "part_number": MAGNET_PART_NUMBER,
            "ean": "7640172691830",
            "quantity_per_robot": 2,
            "centres_xy": magnet_points(),
            "dimensions": [MAGNET_WIDTH, MAGNET_WIDTH, MAGNET_HEIGHT],
            "fastener_each": "ISO 10642 M3x10 countersunk screw + DIN 934 M3 nut",
            "catalog_axial_force_each_N": MAGNET_FORCE_EACH_N,
            "catalog_displacement_force_each_N": MAGNET_DISPLACEMENT_EACH_N,
        },
        "steel_target": {
            "manufacturer": "Webcraft GmbH / supermagnete",
            "part_number": TARGET_PART_NUMBER,
            "ean": "7640172691892",
            "quantity_per_tool": 2,
            "dimensions": [TARGET_WIDTH, TARGET_WIDTH, TARGET_HEIGHT],
            "fastener_each": "ISO 10642 M5x10 countersunk screw + DIN 934 M5 nut",
        },
        "positive_lock": {
            "type": "internal keyhole slider, spring locked, passively cammed open by dock",
            "travel": SLIDER_TRAVEL,
            "keyhole_contract": positive_lock_keyhole_contract(),
            "studs": "2 x McMaster 90318A720, M3, 4 mm shoulder dia, 5 mm shoulder length",
            "stud_retention": "2 x rear-loaded DIN 934 M3 nuts, 5.5 mm AF x 2.4 mm",
            "slider_material": "1.5-1.6 mm 304 stainless steel; printed copy for fit checks only",
            "return_spring": RETURN_SPRING_PART_NUMBER,
            "fails_locked_on_power_loss": True,
        },
        "electrical": {
            "contacts": 4,
            "signals": list(CONTACT_SIGNALS),
            "pogo_pin": POGO_PART_NUMBER,
            "pogo_catalog_max_current_A": 8.0,
            "pogo_catalog_derated_current_A": 6.2,
            "ground_first_mate_offset": POGO_GROUND_PROTRUSION - POGO_STANDARD_PROTRUSION,
            "stock_gripper_uses": ["GND", "+12V", "TTL_DATA"],
            "tool_board": [CONTACT_BOARD_WIDTH, CONTACT_BOARD_HEIGHT, CONTACT_BOARD_THICKNESS],
            "tool_board_target_pad_x_from_board_center": CONTACT_TARGET_PAD_OFFSET_X,
            "tool_board_wire_pad_x_from_board_center": CONTACT_WIRE_PAD_OFFSET_X,
            "wire_pad_diameter_drill": [CONTACT_WIRE_PAD_DIAMETER, CONTACT_WIRE_DRILL_DIAMETER],
            "tool_board_connection": "direct solder to four plated through holes from rear",
        },
        "stock_gripper_adapter": {
            "insert": "4 x Ruthex RX-M3x5.7 heat-set inserts for the stock gripper",
            "lock_stud_nuts": "2 x DIN 934 M3 captive nuts",
            "uses_original_gripper_holes": True,
            "control": "Feetech motor ID 6 remains on the original half-duplex TTL bus through P1-P3",
        },
        "official_component_limits_not_arm_payload": {
            "STS3215_rated_torque_kg_cm": SERVO_RATED_TORQUE_KG_CM,
            "STS3215_peak_stall_torque_kg_cm": SERVO_STALL_TORQUE_KG_CM,
            "STS3215_peak_stall_current_A": 2.7,
            "whole_arm_payload_published_by_official_SO101_docs": False,
        },
        "engineering_envelope": checks,
        "estimated_solid_volumes_cm3": {
            "robot_plate": round(robot.val().Volume() / 1000, 2),
            "generic_tool_plate": round(generic.val().Volume() / 1000, 2),
            "stock_gripper_tool_plate": round(stock.val().Volume() / 1000, 2),
            "passive_dock": round(dock.val().Volume() / 1000, 2),
            "lock_slider": round(slider.val().Volume() / 1000, 2),
        },
    }
    (EXPORT_DIR / "design_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (EXPORT_DIR / "engineering_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_core_manifest(EXPORT_DIR)
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
