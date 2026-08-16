#!/usr/bin/env python3
"""Generate the retrofit SO-101 powered magnetic quick changer.

All dimensions are millimetres.  The magnets capture and preload the faces;
two shoulder studs and a dock-actuated keyhole slider provide the positive
mechanical lock.  A four-contact pogo cartridge carries the stock gripper's
Feetech power and half-duplex TTL bus without a manual cable operation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cadquery as cq


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EXPORT_DIR = HERE / "exports"

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
MAGNET_POCKET_WIDTH = 12.25
MAGNET_POCKET_DEPTH = 4.05

# Matching Supermagnete MC-12-12-03 steel target (EAN 7640172691892).
TARGET_PART_NUMBER = "MC-12-12-03"
TARGET_WIDTH = 12.0
TARGET_HEIGHT = 3.0
TARGET_SMALL_HOLE_DIAMETER = 5.7
TARGET_COUNTERSINK_DIAMETER = 11.7
TARGET_POCKET_WIDTH = 12.25
TARGET_POCKET_DEPTH = 3.05

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
# an 8 A maximum, 6.4 A derated current, 1.397 mm full stroke and 0.7 mm
# mid-stroke.  Ground is installed 0.2 mm farther forward for first-mate.
POGO_PART_NUMBER = "7983-1-15-20-75-14-11-0"
POGO_MAX_DIAMETER = 2.1
POGO_PRESS_FIT_HOLE = 1.575
POGO_OVERALL_LENGTH = 9.5
POGO_MID_STROKE = 0.7
POGO_FULL_STROKE = 1.397
POGO_STANDARD_PROTRUSION = 0.70
POGO_GROUND_PROTRUSION = 0.90
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
    # a 4.25 mm neck while the 6 mm head remains below the slider.
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        slider = slider.cut(
            cylinder_cutter(KEYHOLE_ENTRY_DIAMETER, SLIDER_THICKNESS + 0.2, x, 0)
        )
        neck_min = x - SLIDER_TRAVEL - 0.15
        neck_max = x + 0.15
        slider = slider.cut(
            box_cutter(
                neck_max - neck_min,
                KEYHOLE_NECK_WIDTH,
                SLIDER_THICKNESS + 0.2,
                (neck_min + neck_max) / 2,
                0,
                -0.1,
            )
        )

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


def pogo_pin() -> cq.Workplane:
    # Simplified but dimensionally bounded reference for the Mill-Max part.
    solder_cup = cq.Workplane("XY").circle(0.82).extrude(3.0)
    sleeve = cq.Workplane("XY").circle(POGO_MAX_DIAMETER / 2).extrude(5.1).translate((0, 0, 3.0))
    plunger = cq.Workplane("XY").circle(0.84).extrude(1.4).translate((0, 0, 8.1))
    return solder_cup.union(sleeve).union(plunger).clean()


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

    # Fixed entry/head-clearance wells for the two shoulder screws.
    for x in (-LOCK_STUD_X, LOCK_STUD_X):
        plate = plate.cut(
            cylinder_cutter(
                KEYHOLE_ENTRY_DIAMETER + 0.15,
                PLATE_THICKNESS - 2.9,
                x,
                0,
                2.9,
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

    # Pogo cartridge: ream these printed pilots to the Mill-Max press-fit size.
    for x, y in pogo_points():
        plate = plate.cut(cylinder_cutter(POGO_PRESS_FIT_HOLE, PLATE_THICKNESS + 0.2, x, y))
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

    stop = cq.Workplane("XY").box(
        82.0, 6.0, PLATE_THICKNESS + 6.0, centered=True
    ).translate((-4.0, 29.0, PLATE_THICKNESS / 2))
    dock = dock.union(stop)

    # Passive wedge: while the coupled tool slides the final 12 mm into the
    # rack, this surface pushes the protruding slider tab 3 mm left to unlock.
    # On withdrawal the catalog spring returns the slider to positive lock.
    cam = (
        cq.Workplane("XY")
        .polyline([(28.0, -16.0), (34.0, -16.0), (34.0, 0.0), (24.05, 0.0)])
        .close()
        .extrude(2.2)
        .translate((0, 0, -4.15))
    )
    dock = dock.union(cam)

    for x in (-25.0, 21.0):
        dock = dock.cut(
            axis_cylinder(4.4, 6.2, (x, 25.9, PLATE_THICKNESS / 2), (0, 1, 0))
        )
    return dock.clean()


def export_part(shape: cq.Workplane, stem: str) -> None:
    cq.exporters.export(shape, str(EXPORT_DIR / f"{stem}.step"))
    cq.exporters.export(
        shape,
        str(EXPORT_DIR / f"{stem}.stl"),
        tolerance=0.08,
        angularTolerance=0.12,
    )


def add_hardware(assembly: cq.Assembly, include_studs: bool = True) -> None:
    magnet = screw_on_magnet()
    target = steel_target()
    for index, (x, y) in enumerate(magnet_points(), start=1):
        assembly.add(
            magnet,
            name=f"magnet_{index}_{MAGNET_PART_NUMBER}",
            loc=cq.Location(cq.Vector(x, y, PLATE_THICKNESS - MAGNET_HEIGHT)),
            color=cq.Color(0.48, 0.12, 0.62),
        )
        assembly.add(
            target,
            name=f"target_{index}_{TARGET_PART_NUMBER}",
            loc=cq.Location(cq.Vector(x, y, PLATE_THICKNESS)),
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

    pin = pogo_pin()
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
    assembly.save(str(EXPORT_DIR / "so101_quick_change_assembly.step"))

    retrofit = cq.Assembly(name="so101_stock_gripper_powered_retrofit_v0_2")
    retrofit.add(robot, name="robot_plate", color=cq.Color(0.95, 0.70, 0.10))
    retrofit.add(
        stock,
        name="stock_gripper_tool_plate",
        loc=cq.Location(cq.Vector(0, 0, PLATE_THICKNESS)),
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
            loc=cq.Location(cq.Vector(0.4875, 0.218, 2 * PLATE_THICKNESS + 0.051)),
            color=cq.Color(0.90, 0.74, 0.12),
        )
    retrofit.save(str(EXPORT_DIR / "so101_stock_gripper_retrofit_assembly.step"))


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
        ("robot", 4, POGO_PART_NUMBER, "high-current solder-cup spring pin", "https://www.mill-max.com/products/new/high-current-small-scale-spring-loaded-pins", "new"),
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


def main() -> None:
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
    pogo = pogo_pin()
    board = contact_board()

    export_part(robot, "so101_robot_plate")
    export_part(generic, "so101_tool_plate")
    export_part(stock, "so101_stock_gripper_tool_plate")
    export_part(dock, "so101_passive_tool_dock")
    export_part(slider, "so101_positive_lock_slider")
    cq.exporters.export(slider.faces(">Z"), str(EXPORT_DIR / "so101_positive_lock_slider_profile.dxf"))
    export_part(magnet, f"hardware_{MAGNET_PART_NUMBER}")
    export_part(target, f"hardware_{TARGET_PART_NUMBER}")
    export_part(shoulder, "hardware_McMaster_90318A720_shoulder_screw")
    export_part(lock_nut, "hardware_DIN934_M3_lock_stud_nut")
    export_part(spring, f"hardware_{RETURN_SPRING_PART_NUMBER}_reference")
    export_part(pogo, f"hardware_Mill-Max_{POGO_PART_NUMBER}_reference")
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
            "pogo_catalog_derated_current_A": 6.4,
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
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
