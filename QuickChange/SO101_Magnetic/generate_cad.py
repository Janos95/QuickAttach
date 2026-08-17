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
    "so101_core_dock_support_bracket.step",
    "so101_core_dock_support_bracket.stl",
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
# much smaller press-fit pogo mounts.  A shallow face relief clears each pad
# while retaining the surface webs measured by the fit contract.  Below it,
# each pin gets the selected Ø1.58 knurl land, a separate Ø2.31 body
# counterbore, and a signal-specific shoulder hard-stop.  Geometry is resolved;
# process tolerance, pull-out, and installed reliability remain blockers.
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
CORE_DOCK_STOP_COUNTERSINK_DIAMETER = 8.2
CORE_DOCK_STOP_COUNTERSINK_DEPTH = 1.9

# Floor-closing support for the rolled core dock.  This is deliberately a
# separate printed BRep: the two M4 joints into the stop and four M6 floor
# anchors are explicit load-path fasteners, never a coincident-face union.
CORE_DOCK_SUPPORT_FLOOR_Y_MM = 193.9154579377553
CORE_DOCK_SUPPORT_TOP_HOLE_X_MM = CORE_DOCK_STOP_HOLE_X
CORE_DOCK_SUPPORT_TOP_HOLE_Z_MM = 4.75
CORE_DOCK_SUPPORT_BASE_HOLE_X_MM = (-40.0, 36.0)
CORE_DOCK_SUPPORT_BASE_HOLE_Z_MM = (-23.25, 32.75)
CORE_DOCK_SUPPORT_M4_CLEARANCE_DIAMETER_MM = 4.4
CORE_DOCK_SUPPORT_M4_NUT_POCKET_RADIUS_MM = 4.75
CORE_DOCK_SUPPORT_M4_TAIL_RADIUS_MM = 2.3
CORE_DOCK_SUPPORT_M6_CLEARANCE_DIAMETER_MM = 6.6
CORE_DOCK_SUPPORT_M6_COUNTERSINK_DIAMETER_MM = 12.0
CORE_DOCK_SUPPORT_M6_COUNTERSINK_DEPTH_MM = 2.7
CORE_DOCK_SUPPORT_EXPECTED_VOLUME_MM3 = 162415.4180526403
CORE_DOCK_SUPPORT_SUPERSEDED_ESTIMATE_VOLUME_MM3 = 162308.50715898623
CORE_DOCK_WITH_SUPPORT_EXPECTED_VOLUME_MM3 = 184159.32283760287

# Five-DOF tool-view roll.  The world placement is R_old * Rz(+87.210869...)
# in matrix-column convention; equivalently it is a -87.210869... tool-view
# roll about the mating normal.  The resulting source -Y axis is world +Z.
CORE_DOCK_TOOL_VIEW_ROLL_DEG = -87.21086925015224
CORE_DOCK_WORLD_POS_M = (
    0.19082795371216685,
    0.1330713713445051,
    0.1939154579377553,
)
CORE_DOCK_WORLD_QUAT_WXYZ = (
    0.6440855284765126,
    -0.6440855284765125,
    0.2918112952014223,
    -0.2918112952014225,
)
CORE_DOCK_WORLD_AXES = (
    (0.6593846719714732, -0.7518057291408950, 0.0),
    (0.0, 0.0, -1.0),
    (0.7518057291408950, 0.6593846719714733, 0.0),
)

# Canonical <=0.5 mm continuation from the exact seated state.  These rows
# are source authority rather than a regenerated runtime artifact.
CORE_DOCK_RELEASE_ROSTER = (
    (0.0, (-0.72, -0.5, 0.8, -0.3, -1.522116811941435)),
    (0.5, (-0.7200000000000001, -0.5012713525907627, 0.7971523199971643, -0.29588096740640185, -1.5221168119414348)),
    (1.0, (-0.7199999999999999, -0.5025209619924711, 0.7942909079367223, -0.29176994594425143, -1.522116811941435)),
    (1.5, (-0.72, -0.5037489159236884, 0.7914158067505378, -0.2876668908268499, -1.5221168119414346)),
    (2.0, (-0.7199999999999999, -0.5049553020734726, 0.7885270586980251, -0.28357175662455236, -1.5221168119414348)),
    (2.5, (-0.7199999999999998, -0.5061402080808916, 0.785624705362152, -0.27948449728126057, -1.5221168119414346)),
    (3.0, (-0.7199999999999999, -0.5073037215149596, 0.7827087876455384, -0.2754050661305792, -1.5221168119414348)),
    (3.5, (-0.72, -0.5084459298549687, 0.7797793457666375, -0.271333415911669, -1.5221168119414348)),
    (4.0, (-0.7200000000000001, -0.5095669204712283, 0.7768364192560011, -0.26726949878477324, -1.522116811941435)),
    (4.5, (-0.7199999999999999, -0.5106667806061898, 0.7738800469526174, -0.2632132663464279, -1.5221168119414348)),
    (5.0, (-0.72, -0.5117455973559675, 0.7709102670003262, -0.2591646696443586, -1.5221168119414346)),
    (5.5, (-0.7200000000000001, -0.5128034576522418, 0.7679271168443034, -0.25512365919206204, -1.5221168119414348)),
    (6.0, (-0.7200000000000001, -0.5138404482445372, 0.7649306332276101, -0.2510901849830732, -1.5221168119414348)),
    (6.5, (-0.7199999999999999, -0.5148566556828839, 0.7619208521878057, -0.24706419650492184, -1.5221168119414343)),
    (7.0, (-0.7199999999999998, -0.5158521663008359, 0.7588978090536203, -0.24304564275278437, -1.5221168119414348)),
    (7.5, (-0.72, -0.5168270661988639, 0.7558615384416828, -0.2390344722428192, -1.5221168119414348)),
    (8.0, (-0.72, -0.5177814412280972, 0.7528120742532991, -0.23503063302520222, -1.5221168119414348)),
    (8.5, (-0.7200000000000005, -0.5187153769744209, 0.7497494496712797, -0.23103407269685902, -1.5221168119414346)),
    (9.0, (-0.7199999999999999, -0.519628958742918, 0.7466736971568104, -0.22704473841389222, -1.5221168119414343)),
    (9.5, (-0.7199999999999999, -0.5205222715426575, 0.7435848484463662, -0.22306257690370881, -1.5221168119414343)),
    (10.0, (-0.7199999999999999, -0.5213954000718088, 0.7404829345486624, -0.2190875344768539, -1.5221168119414348)),
    (10.5, (-0.7200000000000001, -0.5222484287030923, 0.7373679857416372, -0.21511955703854524, -1.5221168119414343)),
    (11.0, (-0.72, -0.5230814414695496, 0.7342400315694654, -0.21115859009991603, -1.5221168119414346)),
    (11.5, (-0.7200000000000001, -0.5238945220506328, 0.7310991008396011, -0.20720457878896845, -1.5221168119414348)),
    (12.0, (-0.7200000000000001, -0.5246877537586018, 0.7279452216198397, -0.20325746786123822, -1.5221168119414348)),
    (12.5, (-0.7199999999999999, -0.5254612195252387, 0.7247784212354064, -0.19931720171016784, -1.522116811941435)),
    (13.0, (-0.7199999999999998, -0.5262150018888452, 0.7215987262660535, -0.19538372437720883, -1.5221168119414346)),
    (13.5, (-0.7200000000000001, -0.526949182981541, 0.7184061625431684, -0.19145697956162747, -1.522116811941435)),
    (14.0, (-0.7200000000000001, -0.5276638445168658, 0.7152007551469122, -0.18753691063004685, -1.5221168119414343)),
    (14.5, (-0.7199999999999998, -0.5283590677776248, 0.7119825284033249, -0.18362346062570054, -1.522116811941435)),
    (15.0, (-0.7200000000000001, -0.5290349336040562, 0.708751505881477, -0.17971657227742088, -1.5221168119414346)),
)
CORE_DOCK_RELEASE_ROSTER_SHA256 = (
    "f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293"
)

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

# Mill-Max high-current solder-cup spring contact.  A compact derived authority
# ledger pins the official dimensioned SVG and press-fit application note by
# URL, byte count, and SHA-256.  The manufacturer files are intentionally not
# redistributed because their redistribution terms were not established.
# Dimensions use the authoritative inch callouts converted to millimetres
# rather than the rounded bracketed values printed on the drawing.
POGO_PART_NUMBER = "7983-1-15-20-75-14-11-0"
POGO_SOURCE_AUTHORITY_DIR = HERE / "source_authority" / "millmax_7983"
POGO_AUTHORITY_LEDGER_PATH = POGO_SOURCE_AUTHORITY_DIR / "authority_ledger.json"
POGO_DIMENSION_DRAWING_SHA256 = (
    "c97327d953663a0aa04ea389ee2d2be19372ffa21503f46e5cbbfb0fd2e890e8"
)
POGO_PRESS_FIT_NOTE_SHA256 = (
    "bbf4c414a11bd3355cde2bb25624c6736b61942964b2cbb3fc42c67c09e87adf"
)

INCH_TO_MM = 25.4
POGO_STANDARD_LENGTH_TOLERANCE = 0.006 * INCH_TO_MM
POGO_OFFICIAL_DIAMETER_TOLERANCE = 0.002 * INCH_TO_MM
POGO_OFFICIAL_ANGLE_TOLERANCE_DEG = 2.0
POGO_MAX_EXPOSED_PLUNGER = 0.062 * INCH_TO_MM
POGO_SHELL_TOP_TO_SHOULDER_BOTTOM = 0.132 * INCH_TO_MM
POGO_SHOULDER_BOTTOM_FROM_BASE = 0.180 * INCH_TO_MM
POGO_FIXED_SHELL_LENGTH = (
    POGO_SHOULDER_BOTTOM_FROM_BASE + POGO_SHELL_TOP_TO_SHOULDER_BOTTOM
)
POGO_OVERALL_REFERENCE_LENGTH = 0.374 * INCH_TO_MM
POGO_OVERALL_LENGTH = POGO_FIXED_SHELL_LENGTH + POGO_MAX_EXPOSED_PLUNGER
if not math.isclose(
    POGO_OVERALL_LENGTH,
    POGO_OVERALL_REFERENCE_LENGTH,
    abs_tol=1.0e-12,
):
    raise RuntimeError("Mill-Max 7983 reference overall dimension chain mismatch")
POGO_SHOULDER_THICKNESS = 0.028 * INCH_TO_MM
POGO_KNURL_LENGTH = 0.030 * INCH_TO_MM
POGO_CUP_REFERENCE_LENGTH = 0.145 * INCH_TO_MM
POGO_CUP_BORE_LENGTH = 0.100 * INCH_TO_MM
POGO_BARB_AXIAL_REFERENCE = 0.025 * INCH_TO_MM

POGO_PLUNGER_DIAMETER = 0.042 * INCH_TO_MM
POGO_UPPER_GUIDE_DIAMETER = 0.068 * INCH_TO_MM
POGO_UPPER_SHELL_DIAMETER = 0.073 * INCH_TO_MM
POGO_BARB_DIAMETER = 0.0765 * INCH_TO_MM
POGO_SHOULDER_DIAMETER = 0.083 * INCH_TO_MM
POGO_KNURL_DIAMETER = 0.065 * INCH_TO_MM
POGO_SOLDER_CUP_DIAMETER = 0.060 * INCH_TO_MM
POGO_SOLDER_CUP_BORE_DIAMETER = 0.038 * INCH_TO_MM

POGO_MID_STROKE = 0.0275 * INCH_TO_MM
POGO_NOMINAL_FULL_STROKE = 0.055 * INCH_TO_MM
POGO_FULL_STROKE_TOLERANCE = 0.005 * INCH_TO_MM
POGO_GUARANTEED_MINIMUM_FULL_STROKE = (
    POGO_NOMINAL_FULL_STROKE - POGO_FULL_STROKE_TOLERANCE
)
POGO_MAXIMUM_FULL_STROKE = (
    POGO_NOMINAL_FULL_STROKE + POGO_FULL_STROKE_TOLERANCE
)
POGO_STANDARD_PROTRUSION = 0.70
POGO_GROUND_PROTRUSION = 0.90

# The selected installation is the anti-rotation, solder-cup-first knurl mode
# from the Mill-Max application note.  The 1.58 mm land is not enlarged to the
# old 1.575 mm generic pilot; a separate body counterbore terminates at the
# shoulder hard-stop.  Print/process tolerances and pull-out force remain
# explicit release blockers.
POGO_SELECTED_MOUNTING_MODE = "knurl_solder_cup_first"
POGO_KNURL_RETENTION_LAND_DIAMETER = 1.58
POGO_BODY_COUNTERBORE_MINIMUM_DIAMETER = 0.087 * INCH_TO_MM
POGO_BODY_COUNTERBORE_DIAMETER = 2.31
POGO_BARB_RECOMMENDED_HOLE_DIAMETER = 0.0755 * INCH_TO_MM
CONTACT_BOARD_WIDTH = 10.0
CONTACT_BOARD_HEIGHT = 22.0
CONTACT_BOARD_THICKNESS = 1.0
CONTACT_PAD_DIAMETER = 4.0
CONTACT_PAD_THICKNESS = 0.05
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


def _pogo_authority_ledger() -> tuple[dict[str, object], dict[str, object]]:
    """Load and independently cross-check the derived manufacturer ledger."""

    if not POGO_AUTHORITY_LEDGER_PATH.is_file():
        raise FileNotFoundError(
            f"missing Mill-Max authority ledger: {POGO_AUTHORITY_LEDGER_PATH}"
        )
    payload = POGO_AUTHORITY_LEDGER_PATH.read_bytes()
    ledger = json.loads(payload)
    expected_dimension_facts = {
        "maximum_exposed_plunger": 0.062,
        "shell_top_to_shoulder_bottom": 0.132,
        "shoulder_bottom_from_base": 0.180,
        "shoulder_thickness": 0.028,
        "knurl_length": 0.030,
        "cup_reference_length": 0.145,
        "cup_bore_length": 0.100,
        "barb_axial_reference": 0.025,
        "mid_stroke": 0.0275,
        "full_stroke": 0.055,
        "full_stroke_tolerance": 0.005,
    }
    expected_diameters = {
        "moving_plunger": 0.042,
        "upper_guide": 0.068,
        "upper_shell": 0.073,
        "barb": 0.0765,
        "shoulder": 0.083,
        "knurl": 0.065,
        "solder_cup_outer": 0.060,
        "solder_cup_bore": 0.038,
    }
    expected_standard_tolerances = {
        "length_in": 0.006,
        "diameter_in": 0.002,
        "angle_deg": 2.0,
    }
    expected_press_fit_dimensions = {
        "barb_recommended_hole": 0.0755,
        "knurl_recommended_hole": 0.062,
        "body_counterbore_minimum": 0.087,
    }
    if ledger.get("part_number") != POGO_PART_NUMBER:
        raise RuntimeError("Mill-Max authority ledger part-number mismatch")
    if ledger.get("schema_version") != "1.0" or ledger.get("drawing_units") != "inch":
        raise RuntimeError("Mill-Max authority ledger schema or units drifted")
    if ledger.get("axial_dimensions_in") != expected_dimension_facts:
        raise RuntimeError("Mill-Max authority ledger axial dimensions drifted")
    if ledger.get("reference_dimensions_in") != {
        "overall_parenthesized": 0.374
    }:
        raise RuntimeError("Mill-Max authority ledger reference dimensions drifted")
    if ledger.get("diameters_in") != expected_diameters:
        raise RuntimeError("Mill-Max authority ledger diameters drifted")
    if ledger.get("standard_tolerances") != expected_standard_tolerances:
        raise RuntimeError("Mill-Max authority ledger tolerances drifted")
    if (
        ledger.get("press_fit_note_dimensions_in")
        != expected_press_fit_dimensions
    ):
        raise RuntimeError("Mill-Max press-fit dimensions drifted")
    provenance = ledger.get("provenance", {})
    drawing = provenance.get("dimension_drawing", {})
    press_fit_note = provenance.get("press_fit_application_note", {})
    expected_drawing = {
        "url": (
            "https://www.mill-max.com/sites/default/files/external/products/"
            "fullsize/2020-09/7983.svg"
        ),
        "media_type": "image/svg+xml",
        "bytes": 175611,
        "sha256": POGO_DIMENSION_DRAWING_SHA256,
    }
    expected_press_fit_note = {
        "url": (
            "https://www.mill-max.com/sites/default/files/external/assets/"
            "2020-10/spring-loaded_solder-cup_pin_2.pdf"
        ),
        "media_type": "application/pdf",
        "bytes": 509252,
        "sha256": POGO_PRESS_FIT_NOTE_SHA256,
    }
    if drawing != expected_drawing:
        raise RuntimeError("Mill-Max drawing digest drifted")
    if press_fit_note != expected_press_fit_note:
        raise RuntimeError("Mill-Max press-fit note digest drifted")
    if provenance.get("retrieval_timestamp_available") is not False:
        raise RuntimeError("Mill-Max retrieval provenance declaration drifted")
    redistribution = provenance.get("redistribution", {})
    if (
        redistribution.get("manufacturer_file_license_confirmed") is not False
        or redistribution.get("manufacturer_files_vendored") is not False
    ):
        raise RuntimeError("Mill-Max redistribution declaration drifted")
    record = {
        "path": POGO_AUTHORITY_LEDGER_PATH.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return ledger, record


def pogo_installed_datum(signal: str) -> dict[str, object]:
    """Return the selected knurl-mount datum for one installed pin."""

    if signal not in CONTACT_SIGNALS:
        raise ValueError(f"unsupported pogo signal: {signal}")
    centre_xy = pogo_points()[CONTACT_SIGNALS.index(signal)]
    protrusion = (
        POGO_GROUND_PROTRUSION
        if signal == "GND"
        else POGO_STANDARD_PROTRUSION
    )
    shoulder_to_full_extension_tip = (
        POGO_SHELL_TOP_TO_SHOULDER_BOTTOM + POGO_MAX_EXPOSED_PLUNGER
    )
    shoulder_stop_z = (
        PLATE_THICKNESS + protrusion - shoulder_to_full_extension_tip
    )
    base_z = shoulder_stop_z - POGO_SHOULDER_BOTTOM_FROM_BASE
    shell_top_z = base_z + POGO_FIXED_SHELL_LENGTH
    full_extension_tip_z = base_z + POGO_OVERALL_LENGTH
    target_pad_exposed_contact_plane_z = PLATE_THICKNESS - CONTACT_PAD_THICKNESS
    mated_compression = full_extension_tip_z - target_pad_exposed_contact_plane_z
    knurl_bottom_z = shoulder_stop_z - POGO_KNURL_LENGTH
    rear_wiring_pocket_top_z = -0.05 + 1.9
    return {
        "signal": signal,
        "centre_xy_mm": list(centre_xy),
        "installation_mode": POGO_SELECTED_MOUNTING_MODE,
        "insertion_direction": "mating_face_toward_rear_negative_z",
        "base_z_mm": base_z,
        "fixed_shell_top_z_mm": shell_top_z,
        "shoulder_stop_plane_z_mm": shoulder_stop_z,
        "shoulder_z_bounds_mm": [
            shoulder_stop_z,
            shoulder_stop_z + POGO_SHOULDER_THICKNESS,
        ],
        "knurl_z_bounds_mm": [knurl_bottom_z, shoulder_stop_z],
        "retention_land_z_bounds_mm": [
            rear_wiring_pocket_top_z,
            shoulder_stop_z,
        ],
        "body_counterbore_z_bounds_mm": [shoulder_stop_z, PLATE_THICKNESS],
        "full_extension_tip_z_mm": full_extension_tip_z,
        "nominal_face_protrusion_mm": protrusion,
        "target_pad_exposed_contact_plane_z_mm": (
            target_pad_exposed_contact_plane_z
        ),
        "mated_compression_mm": mated_compression,
        "mated_tip_z_mm": target_pad_exposed_contact_plane_z,
        "nominal_design_remaining_against_catalog_minimum_stroke_mm": (
            POGO_GUARANTEED_MINIMUM_FULL_STROKE - mated_compression
        ),
        "remaining_stroke_semantics": (
            "nominal installation arithmetic only; part, bore, target, and "
            "fabrication tolerances are not included"
        ),
    }


def pogo_interface_authority_contract() -> dict[str, object]:
    """Return the hash-pinned 7983 profile and selected sectional mount."""

    ledger, ledger_record = _pogo_authority_ledger()
    provenance = ledger["provenance"]
    installed = [pogo_installed_datum(signal) for signal in CONTACT_SIGNALS]
    first_mate_nominal = POGO_GROUND_PROTRUSION - POGO_STANDARD_PROTRUSION
    # Shoulder-to-tip uses two separately toleranced drawing dimensions
    # (.132 fixed span + .062 exposed plunger).  Comparing two independent
    # pins therefore accumulates four standard length-tolerance terms.
    independent_length_error = 4.0 * POGO_STANDARD_LENGTH_TOLERANCE
    knurl_min = POGO_KNURL_DIAMETER - POGO_OFFICIAL_DIAMETER_TOLERANCE
    knurl_max = POGO_KNURL_DIAMETER + POGO_OFFICIAL_DIAMETER_TOLERANCE
    cup_max = POGO_SOLDER_CUP_DIAMETER + POGO_OFFICIAL_DIAMETER_TOLERANCE
    shoulder_min = POGO_SHOULDER_DIAMETER - POGO_OFFICIAL_DIAMETER_TOLERANCE
    shoulder_max = POGO_SHOULDER_DIAMETER + POGO_OFFICIAL_DIAMETER_TOLERANCE
    authority = {
        "official_sources_hash_pinned": True,
        "fixed_shell_drawing_envelope_reconstructed": True,
        "moving_plunger_split_from_fixed_shell": True,
        "knurl_mounting_mode_selected": True,
        "sectional_bore_and_shoulder_stop_resolved": True,
        "ground_first_mate_shoulder_datum_resolved": True,
        "ground_first_mate_tolerance_stack_qualified": False,
        "knurl_press_fit_process_and_pullout_qualified": False,
        "installed_electrical_cycle_reliability_qualified": False,
        "release_ready": False,
        "blockers": [
            "ground_first_mate_tolerance_stack_unqualified",
            "knurl_press_fit_process_and_pullout_unqualified",
            "installed_electrical_cycle_reliability_unqualified",
        ],
    }
    return {
        "schema_version": "1.0",
        "part_number": POGO_PART_NUMBER,
        "units": "mm",
        "pin_source_frame": "solder_cup_end_z0_axis_positive_toward_plunger",
        "official_sources": {
            "derived_authority_ledger": ledger_record,
            "dimension_drawing_svg": provenance["dimension_drawing"],
            "press_fit_application_note_pdf": provenance[
                "press_fit_application_note"
            ],
            "redistribution": provenance["redistribution"],
            "offline_manufacturer_byte_revalidation_available": False,
            "hash_pin_semantics": (
                "records the recovered manufacturer byte digests; cached "
                "manufacturer artwork is not redistributed"
            ),
            "product_page": (
                "https://www.mill-max.com/products/discrete-spring-loaded-pins/"
                "spring-loaded-pin-with-solder-cup-termination/7983/"
                "7983-1-15-20-75-14-11-0"
            ),
        },
        "fixed_shell_envelope_authority": {
            "kind": "official-drawing-derived_conservative_nominal_exterior",
            "manufacturer_3d_cad": False,
            "transition_spans_enlarged_to_adjacent_maximum_diameter": True,
            "manufacturing_diameter_tolerance_included": False,
            "mass_or_internal_material_authority": False,
        },
        "drawing_tolerances": {
            "standard_length_mm": POGO_STANDARD_LENGTH_TOLERANCE,
            "standard_diameter_mm": POGO_OFFICIAL_DIAMETER_TOLERANCE,
            "standard_angle_deg": POGO_OFFICIAL_ANGLE_TOLERANCE_DEG,
        },
        "dimensioned_profile": {
            "overall_reference_length_mm": POGO_OVERALL_REFERENCE_LENGTH,
            "overall_reference_semantics": (
                "parenthesized drawing reference; independently checked "
                "against the dimensioned .180+.132+.062 chain"
            ),
            "fixed_shell_length_mm": POGO_FIXED_SHELL_LENGTH,
            "maximum_exposed_plunger_mm": POGO_MAX_EXPOSED_PLUNGER,
            "shell_top_to_shoulder_bottom_mm": (
                POGO_SHELL_TOP_TO_SHOULDER_BOTTOM
            ),
            "shoulder_bottom_from_base_mm": POGO_SHOULDER_BOTTOM_FROM_BASE,
            "shoulder_thickness_mm": POGO_SHOULDER_THICKNESS,
            "knurl_length_mm": POGO_KNURL_LENGTH,
            "cup_reference_length_mm": POGO_CUP_REFERENCE_LENGTH,
            "cup_bore_length_mm": POGO_CUP_BORE_LENGTH,
            "barb_axial_reference_mm": POGO_BARB_AXIAL_REFERENCE,
            "nominal_diameters_mm": {
                "moving_plunger": POGO_PLUNGER_DIAMETER,
                "upper_guide": POGO_UPPER_GUIDE_DIAMETER,
                "upper_shell": POGO_UPPER_SHELL_DIAMETER,
                "barb": POGO_BARB_DIAMETER,
                "shoulder": POGO_SHOULDER_DIAMETER,
                "knurl": POGO_KNURL_DIAMETER,
                "solder_cup_outer": POGO_SOLDER_CUP_DIAMETER,
                "solder_cup_bore": POGO_SOLDER_CUP_BORE_DIAMETER,
            },
            "fixed_shell_collision_envelope_segments": [
                {
                    "name": "solder_cup",
                    "z_bounds_mm": [
                        0.0,
                        POGO_CUP_REFERENCE_LENGTH,
                    ],
                    "outer_diameter_mm": POGO_SOLDER_CUP_DIAMETER,
                    "semantics": "drawing_nominal_outer_envelope",
                },
                {
                    "name": "cup_to_knurl_transition_bound",
                    "z_bounds_mm": [
                        POGO_CUP_REFERENCE_LENGTH,
                        POGO_SHOULDER_BOTTOM_FROM_BASE - POGO_KNURL_LENGTH,
                    ],
                    "outer_diameter_mm": max(
                        POGO_SOLDER_CUP_DIAMETER,
                        POGO_KNURL_DIAMETER,
                    ),
                    "semantics": (
                        "conservative_largest_adjacent_dimensioned_diameter"
                    ),
                },
                {
                    "name": "knurl",
                    "z_bounds_mm": [
                        POGO_SHOULDER_BOTTOM_FROM_BASE - POGO_KNURL_LENGTH,
                        POGO_SHOULDER_BOTTOM_FROM_BASE,
                    ],
                    "outer_diameter_mm": POGO_KNURL_DIAMETER,
                    "semantics": "dimensioned_press_fit_feature",
                },
                {
                    "name": "shoulder",
                    "z_bounds_mm": [
                        POGO_SHOULDER_BOTTOM_FROM_BASE,
                        POGO_SHOULDER_BOTTOM_FROM_BASE
                        + POGO_SHOULDER_THICKNESS,
                    ],
                    "outer_diameter_mm": POGO_SHOULDER_DIAMETER,
                    "semantics": "dimensioned_hard_stop_feature",
                },
                {
                    "name": "plunger_side_fixed_features",
                    "z_bounds_mm": [
                        POGO_SHOULDER_BOTTOM_FROM_BASE
                        + POGO_SHOULDER_THICKNESS,
                        POGO_FIXED_SHELL_LENGTH,
                    ],
                    "outer_diameter_mm": POGO_BARB_DIAMETER,
                    "semantics": (
                        "conservative_full_span_envelope_of_dimensioned_"
                        "upper_guide_upper_shell_and_barb_diameters"
                    ),
                },
            ],
            "moving_plunger": {
                "outer_diameter_mm": POGO_PLUNGER_DIAMETER,
                "maximum_exposed_length_mm": POGO_MAX_EXPOSED_PLUNGER,
                "motion_kind": "prismatic",
                "motion_axis": [0.0, 0.0, -1.0],
                "compression_range_mm": [0.0, POGO_MAXIMUM_FULL_STROKE],
                "collision_shape_semantics": (
                    "official-drawing-derived conservative cylindrical "
                    "envelope of round tip; not manufacturer 3D CAD"
                ),
            },
        },
        "stroke": {
            "mid_stroke_nominal_mm": POGO_MID_STROKE,
            "full_stroke_nominal_mm": POGO_NOMINAL_FULL_STROKE,
            "full_stroke_tolerance_mm": POGO_FULL_STROKE_TOLERANCE,
            "guaranteed_minimum_full_stroke_mm": (
                POGO_GUARANTEED_MINIMUM_FULL_STROKE
            ),
            "maximum_full_stroke_mm": POGO_MAXIMUM_FULL_STROKE,
        },
        "selected_mounting_design": {
            "mode": POGO_SELECTED_MOUNTING_MODE,
            "application_note_knurl_feature_label_mm": 1.65,
            "drawing_knurl_nominal_diameter_mm": POGO_KNURL_DIAMETER,
            "application_note_hole_exact_inch_conversion_mm": (
                0.062 * INCH_TO_MM
            ),
            "application_note_hole_rounded_label_mm": 1.58,
            "retention_land_diameter_mm": (
                POGO_KNURL_RETENTION_LAND_DIAMETER
            ),
            "body_counterbore_minimum_diameter_mm": (
                POGO_BODY_COUNTERBORE_MINIMUM_DIAMETER
            ),
            "body_counterbore_design_diameter_mm": (
                POGO_BODY_COUNTERBORE_DIAMETER
            ),
            "barb_alternative_recommended_hole_diameter_mm": (
                POGO_BARB_RECOMMENDED_HOLE_DIAMETER
            ),
            "installed_datums": installed,
            "nominal_and_part_tolerance_only_fit": {
                "knurl_diameter_range_mm": [knurl_min, knurl_max],
                "knurl_diametral_interference_range_mm": [
                    knurl_min - POGO_KNURL_RETENTION_LAND_DIAMETER,
                    knurl_max - POGO_KNURL_RETENTION_LAND_DIAMETER,
                ],
                "solder_cup_max_diameter_mm": cup_max,
                "solder_cup_minimum_diametral_passage_mm": (
                    POGO_KNURL_RETENTION_LAND_DIAMETER - cup_max
                ),
                "shoulder_diameter_range_mm": [shoulder_min, shoulder_max],
                "minimum_shoulder_bearing_radial_overlap_mm": (
                    shoulder_min - POGO_KNURL_RETENTION_LAND_DIAMETER
                )
                / 2.0,
                "body_counterbore_minimum_radial_clearance_mm": (
                    POGO_BODY_COUNTERBORE_DIAMETER - shoulder_max
                )
                / 2.0,
                "fabrication_hole_tolerance_included": False,
                "pullout_force_bound_included": False,
            },
        },
        "first_mate_tolerance_stack": {
            "nominal_ground_lead_mm": first_mate_nominal,
            "shoulder_to_tip_dimension_terms_per_pin": [
                ".132_in_fixed_span",
                ".062_in_maximum_exposed_plunger",
            ],
            "independent_standard_length_tolerance_term_count": 4,
            "independent_pin_pair_error_bound_mm": independent_length_error,
            "guaranteed_worst_case_ground_lead_mm": (
                first_mate_nominal - independent_length_error
            ),
            "passed": False,
        },
        "release_authority": authority,
    }


def interface_hardware_fit_contract() -> dict[str, object]:
    """Return source dimensions and honest authority gaps at the mating face."""

    pogo_authority = pogo_interface_authority_contract()
    rear_wiring_pocket_top = -0.05 + 1.9
    counterbore_bottom = PLATE_THICKNESS - POGO_PAD_RELIEF_DEPTH
    ground_pin_shell_top = float(
        pogo_installed_datum("GND")["fixed_shell_top_z_mm"]
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
            "fixed_pogo_shell_exact_drawing_bound": bool(
                pogo_authority["release_authority"]
                ["fixed_shell_drawing_envelope_reconstructed"]
            ),
            "fixed_pogo_shell_bound_semantics": (
                "official-drawing-derived conservative nominal exterior; "
                "not manufacturer 3D CAD or manufacturing-tolerance authority"
            ),
            "pogo_mounting_sectional_bore_resolved": bool(
                pogo_authority["release_authority"]
                ["sectional_bore_and_shoulder_stop_resolved"]
            ),
            "ground_first_mate_shoulder_datum_resolved": bool(
                pogo_authority["release_authority"]
                ["ground_first_mate_shoulder_datum_resolved"]
            ),
            "ground_first_mate_tolerance_stack_qualified": False,
            "knurl_press_fit_process_and_pullout_qualified": False,
            "installed_electrical_cycle_reliability_qualified": False,
            "magnetic_fastener_seating_and_preload_bound": False,
            "moving_interface_pair_route_recomputed": False,
            "printed_interface_feature_strength_qualified": False,
            "release_ready": False,
            "blockers": [
                "fabrication_process_tolerance_unqualified",
                "ground_first_mate_tolerance_stack_unqualified",
                "knurl_press_fit_process_and_pullout_unqualified",
                "installed_electrical_cycle_reliability_unqualified",
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
            "pogo_interface_authority": pogo_authority,
            "diameter_mm": POGO_PAD_RELIEF_DIAMETER,
            "depth_mm": POGO_PAD_RELIEF_DEPTH,
            "target_pad_diameter_mm": CONTACT_PAD_DIAMETER,
            "selected_sectional_mount": {
                "mode": POGO_SELECTED_MOUNTING_MODE,
                "retention_land_diameter_mm": (
                    POGO_KNURL_RETENTION_LAND_DIAMETER
                ),
                "body_counterbore_diameter_mm": (
                    POGO_BODY_COUNTERBORE_DIAMETER
                ),
                "installed_datums": pogo_authority["selected_mounting_design"]
                ["installed_datums"],
                "rear_wiring_pocket_top_z_mm": rear_wiring_pocket_top,
                "minimum_retention_land_height_mm": min(
                    float(datum["shoulder_stop_plane_z_mm"])
                    - rear_wiring_pocket_top
                    for datum in pogo_authority["selected_mounting_design"]
                    ["installed_datums"]
                ),
                "minimum_body_counterbore_depth_mm": min(
                    PLATE_THICKNESS
                    - float(datum["shoulder_stop_plane_z_mm"])
                    for datum in pogo_authority["selected_mounting_design"]
                    ["installed_datums"]
                ),
                "minimum_adjacent_counterbore_web_mm": (
                    CONTACT_PITCH - POGO_BODY_COUNTERBORE_DIAMETER
                ),
                "minimum_outer_y_counterbore_web_mm": (
                    ELECTRICAL_WING_HEIGHT / 2.0
                    - max(abs(value) for value in CONTACT_Y)
                    - POGO_BODY_COUNTERBORE_DIAMETER / 2.0
                ),
                "ground_shell_top_to_face_relief_floor_mm": (
                    counterbore_bottom - ground_pin_shell_top
                ),
                "fabrication_hole_tolerance_included": False,
                "pullout_force_bound_included": False,
            },
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


def axis_cone(
    start_diameter: float,
    end_diameter: float,
    length: float,
    start: tuple[float, float, float],
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> cq.Workplane:
    """Return an axis-directed conical-frustum cutter."""

    solid = cq.Solid.makeCone(
        start_diameter / 2.0,
        end_diameter / 2.0,
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


def _pogo_cylinder(outer_diameter: float, z_min: float, z_max: float) -> cq.Workplane:
    """Build one positive-Z axisymmetric profile segment."""

    if not all(math.isfinite(value) for value in (outer_diameter, z_min, z_max)):
        raise ValueError("pogo profile dimensions must be finite")
    if outer_diameter <= 0.0 or z_max <= z_min:
        raise ValueError("pogo profile segment must have positive size")
    return (
        cq.Workplane("XY")
        .circle(outer_diameter / 2.0)
        .extrude(z_max - z_min)
        .translate((0.0, 0.0, z_min))
    )


def pogo_official_fixed_feature_solids() -> dict[str, cq.Workplane]:
    """Return drawing-bounded fixed 7983 feature envelopes.

    The source drawing does not dimension every small transition.  The two
    transition spans therefore use the largest adjacent dimensioned diameter;
    this deliberately adds material and is conservative for collision checks.
    The solder-cup bore is not subtracted because this API is an exterior
    collision envelope, not a manufacturing solid or mass authority.
    """

    knurl_bottom = POGO_SHOULDER_BOTTOM_FROM_BASE - POGO_KNURL_LENGTH
    shoulder_top = POGO_SHOULDER_BOTTOM_FROM_BASE + POGO_SHOULDER_THICKNESS
    return {
        "solder_cup": _pogo_cylinder(
            POGO_SOLDER_CUP_DIAMETER,
            0.0,
            POGO_CUP_REFERENCE_LENGTH,
        ),
        "cup_to_knurl_transition_bound": _pogo_cylinder(
            max(POGO_SOLDER_CUP_DIAMETER, POGO_KNURL_DIAMETER),
            POGO_CUP_REFERENCE_LENGTH,
            knurl_bottom,
        ),
        "knurl": _pogo_cylinder(
            POGO_KNURL_DIAMETER,
            knurl_bottom,
            POGO_SHOULDER_BOTTOM_FROM_BASE,
        ),
        "shoulder": _pogo_cylinder(
            POGO_SHOULDER_DIAMETER,
            POGO_SHOULDER_BOTTOM_FROM_BASE,
            shoulder_top,
        ),
        "plunger_side_fixed_features": _pogo_cylinder(
            max(
                POGO_UPPER_GUIDE_DIAMETER,
                POGO_UPPER_SHELL_DIAMETER,
                POGO_BARB_DIAMETER,
            ),
            shoulder_top,
            POGO_FIXED_SHELL_LENGTH,
        ),
    }


def pogo_official_fixed_shell() -> cq.Workplane:
    """Return the conservative exterior envelope of the fixed 7983 shell."""

    segments = list(pogo_official_fixed_feature_solids().values())
    shell = segments[0]
    for segment in segments[1:]:
        shell = shell.union(segment)
    return shell.clean()


def pogo_official_plunger(compression_mm: float = 0.0) -> cq.Workplane:
    """Return the independently moving official Ø1.0668 mm plunger envelope."""

    if (
        not math.isfinite(compression_mm)
        or compression_mm < 0.0
        or compression_mm > POGO_MAXIMUM_FULL_STROKE
    ):
        raise ValueError(
            "compression_mm must be finite and inside the official full-stroke range"
        )
    exposed_length = POGO_MAX_EXPOSED_PLUNGER - compression_mm
    if exposed_length <= 0.0:
        raise ValueError("compression_mm removes the exposed plunger envelope")
    return _pogo_cylinder(
        POGO_PLUNGER_DIAMETER,
        POGO_FIXED_SHELL_LENGTH,
        POGO_FIXED_SHELL_LENGTH + exposed_length,
    )


def pogo_official_pin(compression_mm: float = 0.0) -> cq.Workplane:
    """Return fixed-shell plus prismatic-plunger collision envelopes."""

    return pogo_official_fixed_shell().union(
        pogo_official_plunger(compression_mm)
    ).clean()


# Backward-compatible API names retained for downstream export callers.  They
# now resolve to the hash-pinned official envelope rather than the old visual
# approximation.
def pogo_reference_fixed_shell() -> cq.Workplane:
    return pogo_official_fixed_shell()


def pogo_reference_plunger(retraction_mm: float = 0.0) -> cq.Workplane:
    return pogo_official_plunger(retraction_mm)


def pogo_reference_pin(retraction_mm: float = 0.0) -> cq.Workplane:
    return pogo_official_pin(retraction_mm)


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
    return (
        cq.Workplane("XY")
        .circle(CONTACT_PAD_DIAMETER / 2)
        .extrude(CONTACT_PAD_THICKNESS)
    )


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
        "front_countersinks": [
            {
                "axis": [0.0, 1.0, 0.0],
                "x_mm": x_value,
                "y_bounds_mm": [
                    CORE_DOCK_STOP_Y_MIN,
                    CORE_DOCK_STOP_Y_MIN + CORE_DOCK_STOP_COUNTERSINK_DEPTH,
                ],
                "z_mm": (CORE_DOCK_STOP_Z_MIN + CORE_DOCK_STOP_Z_MAX) / 2.0,
                "major_diameter_mm": CORE_DOCK_STOP_COUNTERSINK_DIAMETER,
                "minor_diameter_mm": CORE_DOCK_STOP_HOLE_DIAMETER,
                "included_angle_deg": 90.0,
                "target_head_recess_mm": [0.05, 0.15],
            }
            for x_value in CORE_DOCK_STOP_HOLE_X
        ],
        "expected_volume_mm3": 7379.269784962569,
        "countersink_removed_volume_mm3": 64.26651371693515,
        "remaining_ligaments_mm": {
            "rear_y_wall": 4.1,
            "minimum_z": 3.65,
            "minimum_x": 11.9,
        },
    }


def core_dock_release_roster() -> list[dict[str, object]]:
    """Return the canonical 31-row rolled-frame IK continuation."""

    return [
        {"withdrawal_mm": withdrawal, "q_rad": list(q_rad)}
        for withdrawal, q_rad in CORE_DOCK_RELEASE_ROSTER
    ]


def core_dock_support_contract() -> dict[str, object]:
    """Return the fail-closed rolled-dock installation source contract.

    The geometric fields are executable dimensions.  Audit-only FK and
    collision results are pinned separately and never turn the release green:
    runtime adoption, material allowables, substrate authority, and physical
    contact dynamics remain explicit blockers.
    """

    roster = core_dock_release_roster()
    roster_digest = hashlib.sha256(
        json.dumps(
            roster,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    maximum_joint_step_deg = max(
        math.degrees(
            max(
                abs(current - previous)
                for current, previous in zip(
                    roster[index]["q_rad"], roster[index - 1]["q_rad"]
                )
            )
        )
        for index in range(1, len(roster))
    )
    post_length = 187.9154579377553 - 40.0
    section_area = 43.0**2 - 35.0**2
    section_second_moment = (43.0**4 - 35.0**4) / 12.0
    section_modulus = section_second_moment / (43.0 / 2.0)
    reverse_cam_force = 4.989345448 + 3.528
    load_moment = (
        2.942
        + 9.80665 * 0.100
        + reverse_cam_force * 0.03875
    )
    elastic_modulus_mpa = 1500.0
    bending_stress = load_moment * 1000.0 / section_modulus
    tip_deflection = (
        load_moment
        * 1000.0
        * post_length**2
        / (2.0 * elastic_modulus_mpa * section_second_moment)
    )
    tip_rotation_deg = math.degrees(
        load_moment
        * 1000.0
        * post_length
        / (elastic_modulus_mpa * section_second_moment)
    )
    compression_force = 140.923995
    euler_load = (
        math.pi**2
        * elastic_modulus_mpa
        * section_second_moment
        / (4.0 * post_length**2)
    )
    blockers = [
        "runtime_placements_and_matcha_base_authority_are_stale",
        "vendor_or_normative_source_missing_for_selected_M4_and_M6_fasteners",
        "floor_fixture_substrate_and_M6_thread_authority_missing",
        "PA12_modulus_strength_creep_and_process_allowables_unqualified",
        "printed_dimensional_tolerance_and_anchor_strength_unqualified",
        "cam_contact_friction_reverse_insertion_and_capture_dynamics_unvalidated",
        "full_compiled_arm_collision_screen_not_yet_regenerated_from_this_source",
    ]
    return {
        "schema_version": "1.0-source-checkpoint",
        "frame": {
            "position_m": list(CORE_DOCK_WORLD_POS_M),
            "quat_wxyz": list(CORE_DOCK_WORLD_QUAT_WXYZ),
            "tool_view_roll_deg": CORE_DOCK_TOOL_VIEW_ROLL_DEG,
            "matrix_composition": "R_world_new=R_world_old*Rz(+87.21086925015224deg)",
            "dock_local_axes_world_columns": [
                list(axis) for axis in CORE_DOCK_WORLD_AXES
            ],
            "source_negative_y_world": [0.0, 0.0, 1.0],
            "source_negative_y_is_world_up": True,
            "rotation_scope": "core_interface_and_dock_datum_only",
        },
        "release_roster": {
            "rows": roster,
            "row_count": len(roster),
            "step_mm": 0.5,
            "withdrawal_bounds_mm": [0.0, 15.0],
            "canonical_sha256": roster_digest,
            "expected_canonical_sha256": CORE_DOCK_RELEASE_ROSTER_SHA256,
            "maximum_joint_step_deg": maximum_joint_step_deg,
            "endpoint_q_deg": [math.degrees(q) for q in roster[-1]["q_rad"]],
            "solver_audit": {
                "method": "deterministic_continuation_from_exact_seated_state",
                "maximum_fk_position_error_mm": 1.4152622167509189e-13,
                "maximum_fk_orientation_error_deg": 3.6184477995385756e-14,
                "runtime_recomputation_pending": True,
            },
        },
        "passive_release": {
            "axis": "dock_local_negative_y",
            "slider_q_mm": {
                "0_to_2_mm": 0.05,
                "2_to_15_mm_formula": "min(3,0.05+0.246875*(withdrawal_mm-2))",
                "slope": 0.246875,
                "q3_withdrawal_mm": 13.949367088607595,
            },
            "full_cam_clearance_at_15_mm": 0.251814779,
        },
        "printed_brep": {
            "component_inventory": [
                {
                    "name": "core_passive_tool_dock",
                    "solid_count": 1,
                    "expected_volume_mm3": 21743.904784962568,
                    "connection": "two_explicit_M4_stop_fasteners",
                },
                {
                    "name": "core_dock_floor_support_bracket",
                    "solid_count": 1,
                    "expected_volume_mm3": CORE_DOCK_SUPPORT_EXPECTED_VOLUME_MM3,
                    "bounds_mm": {
                        "x": [-52.0, 48.0],
                        "y": [32.0, CORE_DOCK_SUPPORT_FLOOR_Y_MM],
                        "z": [-35.25, 44.75],
                    },
                    "connection": "four_explicit_M6_floor_fasteners",
                },
            ],
            "installed_printed_volume_mm3": (
                CORE_DOCK_WITH_SUPPORT_EXPECTED_VOLUME_MM3
            ),
            "superseded_non_brep_estimate_mm3": (
                CORE_DOCK_SUPPORT_SUPERSEDED_ESTIMATE_VOLUME_MM3
            ),
            "mass_claim": None,
            "mass_blocker": "printed_material_density_and_condition_not_selected",
            "boolean_primitives": {
                "head_main_box_bounds_mm": {
                    "x": [-32.0, 24.0], "y": [32.0, 40.0], "z": [0.5, 9.0]
                },
                "right_reinforcement_box_bounds_mm": {
                    "x": [23.0, 27.0], "y": [32.0, 40.0], "z": [1.0, 8.5]
                },
                "right_reinforcement_overlap_mm3": 60.0,
                "post_outer_box_bounds_mm": {
                    "x": [-23.5, 19.5],
                    "y": [38.0, 187.9154579377553],
                    "z": [-16.75, 26.25],
                },
                "post_inner_passage_box_bounds_mm": {
                    "x": [-19.5, 15.5],
                    "y": [50.0, CORE_DOCK_SUPPORT_FLOOR_Y_MM],
                    "z": [-12.75, 22.25],
                },
                "base_box_bounds_mm": {
                    "x": [-52.0, 48.0],
                    "y": [185.9154579377553, CORE_DOCK_SUPPORT_FLOOR_Y_MM],
                    "z": [-35.25, 44.75],
                },
                "base_passage_box_bounds_mm": {
                    "x": [-19.5, 15.5],
                    "y": [185.9154579377553, CORE_DOCK_SUPPORT_FLOOR_Y_MM],
                    "z": [-12.75, 22.25],
                },
                "head_post_positive_overlap_mm3_before_pockets": 731.0,
                "head_post_positive_overlap_mm3_after_pockets": 716.8886804667261,
                "post_base_positive_overlap_mm3": 1248.0,
                "m4_clearance_holes": {
                    "axis": [0.0, 1.0, 0.0],
                    "centres_xz_mm": [[-25.0, 4.75], [21.0, 4.75]],
                    "diameter_mm": CORE_DOCK_SUPPORT_M4_CLEARANCE_DIAMETER_MM,
                    "y_bounds_mm": [31.9, 40.1],
                },
                "m4_nut_pockets": {
                    "radius_mm": CORE_DOCK_SUPPORT_M4_NUT_POCKET_RADIUS_MM,
                    "y_bounds_mm": [39.8, 46.5],
                },
                "m4_thread_tail_pockets": {
                    "radius_mm": CORE_DOCK_SUPPORT_M4_TAIL_RADIUS_MM,
                    "y_bounds_mm": [46.5, 51.2],
                },
                "m6_floor_holes": {
                    "axis": [0.0, 1.0, 0.0],
                    "centres_xz_mm": [
                        [x_value, z_value]
                        for x_value in CORE_DOCK_SUPPORT_BASE_HOLE_X_MM
                        for z_value in CORE_DOCK_SUPPORT_BASE_HOLE_Z_MM
                    ],
                    "diameter_mm": CORE_DOCK_SUPPORT_M6_CLEARANCE_DIAMETER_MM,
                    "through_y_bounds_mm": [185.8154579377553, 194.0154579377553],
                    "countersink_major_diameter_mm": (
                        CORE_DOCK_SUPPORT_M6_COUNTERSINK_DIAMETER_MM
                    ),
                    "countersink_depth_mm": CORE_DOCK_SUPPORT_M6_COUNTERSINK_DEPTH_MM,
                    "countersink_included_angle_deg": 90.0,
                },
            },
        },
        "fasteners": {
            "upper": {
                "quantity": 2,
                "nominal_selection": "ISO_10642_M4x25_A2",
                "washer": "DIN_125_M4_OD9x0.8",
                "nut": "DIN_985_M4_AF7x5",
                "joint_grip_mm": 14.0,
                "washer_plus_nut_mm": 5.8,
                "nominal_thread_protrusion_mm": 5.2,
                "analytic_envelope_volume_mm3": 1071.4870,
                "source_authority": False,
            },
            "lower": {
                "quantity": 4,
                "nominal_selection": "ISO_10642_M6x20",
                "required_tapped_engagement_below_base_mm": 12.0,
                "pitch_mm": 1.0,
                "analytic_envelope_volume_mm3": 2714.3361,
                "substrate_authority": False,
            },
            "total_analytic_envelope_volume_mm3": 3785.823076873815,
        },
        "geometry_audit": {
            "minimum_fixed_dock_distances_mm": {
                "left_lower_rail": 4.031128874,
                "right_lower_rail": 1.414213562,
                "left_upper_rail": 1.118033989,
                "right_upper_rail": 1.0,
                "left_wall": 9.0,
                "right_wall": 6.0,
                "seating_stop": 0.0,
                "positive_lock_cam": 30.35827729,
            },
            "seating_stop_zero_distance_semantic": "explicit_two_M4_bolted_joint",
            "positive_overlap_with_forbidden_dock_features_mm3": 0.0,
            "moving_source_minimum_distances_mm": {
                "stock_tool_plate": 7.0,
                "official_fixed_gripper_body": 7.632880447,
                "robot_plate": 8.015609771,
                "moving_jaw_continuous_aabb": 7.9002132,
            },
            "runtime_physical_brep_count_per_state": 21,
            "runtime_brep_breakdown": {
                "tool_side": 17,
                "robot_plate": 1,
                "magnets": 2,
                "selected_slider_state": 1,
            },
            "moving_jaw_mesh_count": 1,
            "legacy_report_count_with_both_mutually_exclusive_sliders_and_jaw": 23,
            "full_arm_screen": {
                "compiled_arm_geometry_count": 14,
                "continuation_substeps_per_interval": 10,
                "evaluated_transform_states": 3062,
                "minimum_sampled_outer_aabb_lower_bound_mm": 14.717707794,
                "compiled_chain_radius_bound_m": 0.547662487,
                "maximum_subsample_sum_abs_dq_rad": 0.000823806519,
                "between_sample_motion_bound_mm": 0.451445972,
                "continuous_clearance_lower_bound_mm": 14.266261822,
                "nearest_static_fixture_distance_mm": 27.832,
                "runtime_recomputation_pending": True,
            },
        },
        "tolerance_budget": {
            "terms_mm": {
                "dock_print": 0.20,
                "support_print": 0.20,
                "bolt_float": 0.15,
                "seating_and_roll": 0.10,
                "numeric": 0.05,
            },
            "total_mm": 0.70,
            "minimum_nominal_fixed_clearance_mm": 1.0,
            "residual_mm": 0.30,
            "required_residual_mm": 0.20,
            "dimensionally_qualified": False,
        },
        "ligaments_mm": {
            "support_wall": 4.0,
            "top_hole_left_x": 4.8,
            "top_hole_right_x": 3.8,
            "top_hole_minimum_z": 2.05,
            "base_countersink_minimum_edge": 6.0,
        },
        "load_proxy": {
            "scope": "screening_only_not_material_or_joint_certification",
            "section_area_mm2": section_area,
            "second_moment_mm4": section_second_moment,
            "section_modulus_mm3": section_modulus,
            "cantilever_length_mm": post_length,
            "assumed_elastic_modulus_mpa": elastic_modulus_mpa,
            "combined_moment_Nm": load_moment,
            "bending_stress_mpa": bending_stress,
            "tip_deflection_mm": tip_deflection,
            "tip_rotation_deg": tip_rotation_deg,
            "compression_force_N": compression_force,
            "compression_stress_mpa": compression_force / section_area,
            "cantilever_euler_load_kN": euler_load / 1000.0,
            "euler_load_ratio": euler_load / compression_force,
            "joint_force_proxies_N": {
                "top_over_46mm": 92.450,
                "top_over_15.5mm": 274.369,
                "floor_over_56mm": 75.941,
                "floor_over_76mm": 55.957,
            },
            "proof_target": "4.253_Nm_plus_reverse_insertion_load",
        },
        "blockers": blockers,
        "release_ready": False,
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


def core_dock_support_primitives() -> dict[str, cq.Workplane]:
    """Build every positive/cutting primitive in the floor-support BRep."""

    floor_y = CORE_DOCK_SUPPORT_FLOOR_Y_MM
    head_main = (
        cq.Workplane("XY")
        .box(56.0, 8.0, 8.5, centered=True)
        .translate((-4.0, 36.0, 4.75))
    )
    right_reinforcement = (
        cq.Workplane("XY")
        .box(4.0, 8.0, 7.5, centered=True)
        .translate((25.0, 36.0, 4.75))
    )
    post_outer = (
        cq.Workplane("XY")
        .box(43.0, 149.9154579377553, 43.0, centered=True)
        .translate((-2.0, 112.95772896887765, 4.75))
    )
    post_inner_passage = (
        cq.Workplane("XY")
        .box(35.0, floor_y - 50.0, 35.0, centered=True)
        .translate((-2.0, (50.0 + floor_y) / 2.0, 4.75))
    )
    hollow_post = post_outer.cut(post_inner_passage).clean()
    base_outer = (
        cq.Workplane("XY")
        .box(100.0, 8.0, 80.0, centered=True)
        .translate((-2.0, floor_y - 4.0, 4.75))
    )
    base_passage = (
        cq.Workplane("XY")
        .box(35.0, 8.2, 35.0, centered=True)
        .translate((-2.0, floor_y - 4.0, 4.75))
    )
    hollow_base = base_outer.cut(base_passage).clean()
    return {
        "head_main": head_main,
        "right_reinforcement": right_reinforcement,
        "post_outer": post_outer,
        "post_inner_passage": post_inner_passage,
        "hollow_post": hollow_post,
        "base_outer": base_outer,
        "base_passage": base_passage,
        "hollow_base": hollow_base,
    }


def core_dock_support_bracket() -> cq.Workplane:
    """Build the separate, positively fastened floor-support pedestal."""

    primitives = core_dock_support_primitives()
    support = (
        primitives["head_main"]
        .union(primitives["right_reinforcement"])
        .union(primitives["hollow_post"])
        .union(primitives["hollow_base"])
        .clean()
    )
    for x_value in CORE_DOCK_SUPPORT_TOP_HOLE_X_MM:
        support = support.cut(
            axis_cylinder(
                CORE_DOCK_SUPPORT_M4_CLEARANCE_DIAMETER_MM,
                8.2,
                (x_value, 31.9, CORE_DOCK_SUPPORT_TOP_HOLE_Z_MM),
                (0.0, 1.0, 0.0),
            )
        )
        support = support.cut(
            axis_cylinder(
                2.0 * CORE_DOCK_SUPPORT_M4_NUT_POCKET_RADIUS_MM,
                6.7,
                (x_value, 39.8, CORE_DOCK_SUPPORT_TOP_HOLE_Z_MM),
                (0.0, 1.0, 0.0),
            )
        )
        support = support.cut(
            axis_cylinder(
                2.0 * CORE_DOCK_SUPPORT_M4_TAIL_RADIUS_MM,
                4.7,
                (x_value, 46.5, CORE_DOCK_SUPPORT_TOP_HOLE_Z_MM),
                (0.0, 1.0, 0.0),
            )
        )
    for x_value in CORE_DOCK_SUPPORT_BASE_HOLE_X_MM:
        for z_value in CORE_DOCK_SUPPORT_BASE_HOLE_Z_MM:
            support = support.cut(
                axis_cylinder(
                    CORE_DOCK_SUPPORT_M6_CLEARANCE_DIAMETER_MM,
                    8.2,
                    (x_value, CORE_DOCK_SUPPORT_FLOOR_Y_MM - 8.1, z_value),
                    (0.0, 1.0, 0.0),
                )
            )
            support = support.cut(
                axis_cone(
                    CORE_DOCK_SUPPORT_M6_COUNTERSINK_DIAMETER_MM,
                    CORE_DOCK_SUPPORT_M6_CLEARANCE_DIAMETER_MM,
                    CORE_DOCK_SUPPORT_M6_COUNTERSINK_DEPTH_MM,
                    (x_value, CORE_DOCK_SUPPORT_FLOOR_Y_MM - 8.0, z_value),
                    (0.0, 1.0, 0.0),
                )
            )
    return support.clean()


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
    for countersink in spec["front_countersinks"]:
        stop = stop.cut(
            axis_cone(
                countersink["major_diameter_mm"],
                countersink["minor_diameter_mm"],
                countersink["y_bounds_mm"][1] - countersink["y_bounds_mm"][0],
                (
                    countersink["x_mm"],
                    countersink["y_bounds_mm"][0],
                    countersink["z_mm"],
                ),
                tuple(countersink["axis"]),
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

    # Selected Mill-Max knurl/solder-cup-first sectional mount.  The Ø1.58
    # retention land grips only the knurl; the separate Ø2.31 body
    # counterbore terminates at the shoulder hard-stop.  Its per-signal depth
    # preserves the nominal 0.20 mm ground-first offset.  Print-process fit,
    # pull-out force, and the first-mate tolerance stack remain release-red.
    for (x, y), signal in zip(pogo_points(), CONTACT_SIGNALS):
        datum = pogo_installed_datum(signal)
        shoulder_stop_z = float(datum["shoulder_stop_plane_z_mm"])
        plate = plate.cut(
            cylinder_cutter(
                POGO_KNURL_RETENTION_LAND_DIAMETER,
                shoulder_stop_z + 0.05,
                x,
                y,
                -0.05,
            )
        )
        plate = plate.cut(
            cylinder_cutter(
                POGO_BODY_COUNTERBORE_DIAMETER,
                PLATE_THICKNESS - shoulder_stop_z + 0.1,
                x,
                y,
                shoulder_stop_z,
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
            "core_dock_floor_support": core_dock_support_contract(),
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

    pin = pogo_official_pin()
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
            loc=cq.Location(
                cq.Vector(x, y, PLATE_THICKNESS - CONTACT_PAD_THICKNESS)
            ),
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
            (
                "nominal first-mate/last-break datum only; tolerance stack "
                "unqualified; official Feetech pin 1"
            ),
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
        ("robot", 4, POGO_PART_NUMBER, "high-current solder-cup spring pin", "https://www.mill-max.com/products/discrete-spring-loaded-pins/spring-loaded-pin-with-solder-cup-termination/7983/7983-1-15-20-75-14-11-0", "knurl/solder-cup-first sectional mount; fit, pull-out, first-mate tolerance, and cycle reliability unqualified"),
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
        ("dock", 1, "so101_core_dock_support_bracket", "printed PA12/PA-CF floor pedestal", "exports/so101_core_dock_support_bracket.step", "source checkpoint; do not fabricate before blockers close"),
        ("dock", 2, "ISO 10642 M4x25 A2", "stop-to-support countersunk screw", "standard/vendor authority pending", "blocked"),
        ("dock", 2, "DIN 125 M4 + DIN 985 M4", "washer and locknut for support head", "standard/vendor authority pending", "blocked"),
        ("dock", 4, "ISO 10642 M6x20", "support-to-tapped-fixture countersunk screw", "standard/vendor and fixture-substrate authority pending", "blocked"),
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
    support = core_dock_support_bracket()
    slider = locking_slider()
    magnet = screw_on_magnet()
    target = steel_target()
    shoulder = shoulder_lock_stud()
    lock_nut = lock_stud_nut()
    spring = compression_spring()
    pogo = pogo_official_pin()
    board = contact_board()

    export_part(robot, "so101_robot_plate")
    export_part(generic, "so101_tool_plate")
    export_part(stock, "so101_stock_gripper_tool_plate")
    export_part(dock, "so101_passive_tool_dock")
    export_part(support, "so101_core_dock_support_bracket")
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
            "core_dock_floor_support": core_dock_support_contract(),
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
            "ground_first_mate_nominal_reference_offset_mm": (
                POGO_GROUND_PROTRUSION - POGO_STANDARD_PROTRUSION
            ),
            "ground_first_mate_guaranteed_worst_case_lead_mm": (
                pogo_interface_authority_contract()["first_mate_tolerance_stack"]
                ["guaranteed_worst_case_ground_lead_mm"]
            ),
            "ground_first_mate_release_qualified": False,
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
            "core_dock_support_bracket": round(support.val().Volume() / 1000, 2),
            "lock_slider": round(slider.val().Volume() / 1000, 2),
        },
    }
    (EXPORT_DIR / "design_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (EXPORT_DIR / "engineering_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_core_manifest(EXPORT_DIR)
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
