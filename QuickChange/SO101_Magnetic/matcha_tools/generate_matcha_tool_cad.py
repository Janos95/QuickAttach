#!/usr/bin/env python3
"""Generate the two matcha tools and their two-bay passive rack.

All authored dimensions are millimetres.  The coupling plate and every
coupling-side datum come directly from ``../generate_cad.py``; this module does
not carry a second approximation of the magnetic quick-change interface.

The generator deliberately keeps every rigid item as a named component.  That
lets the MuJoCo asset builder consume collision geometry for screws, nuts,
targets and electronics rather than silently replacing an assembly with a
single decorative mesh.  Collision-only envelopes are separately labelled and
have zero mass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Iterable

import cadquery as cq


PACKAGE_DIR = Path(__file__).resolve().parent
QUICK_CHANGE_DIR = PACKAGE_DIR.parent
EXPORT_DIR = PACKAGE_DIR / "exports"
BASE_GENERATOR = QUICK_CHANGE_DIR / "generate_cad.py"

SCHEMA_VERSION = "1.0"
SPOON_TOOL_ID = 21
WHISK_TOOL_ID = 22
WHISK_BUS_ADDRESS = 7

# Common fabrication assumptions.  They are inputs to the mass ledger, not
# claims about a future print or purchased part.
MATERIAL_DENSITY_KG_M3 = {
    "pa12_sls_assumption": 1010.0,
    "stainless_304": 8000.0,
    "steel": 7850.0,
    "aluminium_6061": 2700.0,
    "fr4": 1850.0,
    "copper": 8960.0,
    "silicone": 1100.0,
    "bamboo_dry_assumption": 700.0,
    "smd_effective": 3200.0,
    "collision_only": 0.0,
}

# Tool-axis dimensions.  Coupling face is Z=0 and payload points along +Z.
SPOON_STEM_DIAMETER = 3.2
SPOON_STEM_START_Z = 12.0
SPOON_BOWL_CENTER_Z = 112.0
SPOON_BOWL_OUTER_RADII_YZ = (7.0, 10.0)
SPOON_BOWL_DEPTH = 5.0
SPOON_BOWL_WALL = 0.8

WHISK_MOTOR_DIAMETER = 25.0
WHISK_MOTOR_Z = (17.0, 49.5)
WHISK_ECCENTRIC_MM = 4.0
WHISK_COMPLIANCE_TRAVEL_MM = 5.10
WHISK_COMPLIANCE_LIMITS_MM = (-5.05, 0.05)
WHISK_BELLOWS_RADII_MM = (17.2, 19.0)
WHISK_BELLOWS_Z_MM = (49.5, 91.0)

# The rack is deliberately wider than the complete rigid tool envelope.  The
# plate is retained at its two outer edge lands; payloads remain inside them.
RACK_INSERTION_START_Y = -80.0
RACK_SEATED_Y = 0.0
RACK_WALL_CLEARANCE = 0.50
RACK_REAR_CLEARANCE = 0.30
RACK_PCB_LOWER_RAIL_CLEARANCE = 0.30
RACK_BAY_PITCH = 96.0
RACK_BAY_NAMES = ("spoon", "whisk")
MATCHA_DOCK_STOP_X_MIN = -41.0
MATCHA_DOCK_STOP_X_MAX = 33.0
MATCHA_DOCK_STOP_Y_MIN = 25.0
MATCHA_DOCK_STOP_Y_MAX = 31.0
MATCHA_DOCK_STOP_Z_MIN = -3.0
MATCHA_DOCK_STOP_Z_MAX = 12.5


def _load_interface_module():
    spec = importlib.util.spec_from_file_location(
        "so101_magnetic_interface_authority", BASE_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load interface authority: {BASE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "TOOL_WIDTH": 56.0,
        "TOOL_HEIGHT": 50.0,
        "PLATE_THICKNESS": 9.5,
        "TOOL_MOUNT_X": 21.0,
        "TOOL_MOUNT_Y": 17.0,
    }
    drift = {
        name: (getattr(module, name, None), value)
        for name, value in expected.items()
        if getattr(module, name, None) != value
    }
    if drift:
        raise RuntimeError(f"quick-change interface authority drifted: {drift}")
    return module


INTERFACE = _load_interface_module()


@dataclass(frozen=True)
class Component:
    """A named, homogeneous rigid part or an explicitly conservative envelope."""

    name: str
    shape: cq.Workplane
    material: str
    role: str
    source: str
    mass_override_kg: float | None = None
    fabrication: bool = True

    @property
    def density_kg_m3(self) -> float:
        return MATERIAL_DENSITY_KG_M3[self.material]


def _wp(shape: cq.Shape) -> cq.Workplane:
    return cq.Workplane(obj=shape)


def _translated(shape: cq.Workplane, xyz: tuple[float, float, float]) -> cq.Workplane:
    return shape.translate(xyz)


def _axis_cylinder(
    diameter: float,
    length: float,
    start: tuple[float, float, float],
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> cq.Workplane:
    solid = cq.Solid.makeCylinder(
        diameter / 2.0,
        length,
        cq.Vector(*start),
        cq.Vector(*direction),
    )
    return _wp(solid)


def _hex_nut(across_flats: float, height: float, bore: float) -> cq.Workplane:
    outer = cq.Workplane("XY").polygon(6, across_flats / math.cos(math.pi / 6)).extrude(height)
    return outer.cut(_axis_cylinder(bore, height + 0.2, (0.0, 0.0, -0.1))).clean()


def _countersunk_screw(
    shaft_diameter: float,
    head_diameter: float,
    head_height: float,
    overall_length: float,
) -> cq.Workplane:
    """Simplified ISO 10642 envelope, local head face at Z=0."""

    head = cq.Solid.makeCone(
        head_diameter / 2.0,
        shaft_diameter / 2.0,
        head_height,
        cq.Vector(0, 0, 0),
        cq.Vector(0, 0, 1),
    )
    shaft = cq.Solid.makeCylinder(
        shaft_diameter / 2.0,
        max(overall_length - head_height, 0.1),
        cq.Vector(0, 0, head_height),
        cq.Vector(0, 0, 1),
    )
    return _wp(head.fuse(shaft))


def _socket_set_screw(length: float = 6.0) -> cq.Workplane:
    return _axis_cylinder(3.0, length, (0.0, 0.0, 0.0))


def common_tool_plate() -> cq.Workplane:
    """Return the unmodified generic quick-change tool plate."""

    return INTERFACE.tool_plate(stock_gripper=False)


def tool_interface_hardware_shapes() -> dict[str, cq.Workplane]:
    """Complete rigid tool-side coupling hardware in tool-local coordinates."""

    shapes: dict[str, cq.Workplane] = {}
    m5_screw = _countersunk_screw(5.0, 10.0, 2.7, 10.0)
    m5_nut = _hex_nut(8.0, 4.0, 5.0)
    for index, (x, y) in enumerate(INTERFACE.magnet_points(), start=1):
        shapes[f"target_{index}_MC-12-12-03"] = _translated(
            INTERFACE.steel_target(), (x, y, 0.0)
        )
        shapes[f"target_screw_{index}_ISO10642_M5x10"] = _translated(
            m5_screw, (x, y, 0.0)
        )
        shapes[f"target_nut_{index}_DIN934_M5"] = _translated(
            m5_nut, (x, y, INTERFACE.PLATE_THICKNESS - 4.0)
        )

    for index, x in enumerate((-INTERFACE.LOCK_STUD_X, INTERFACE.LOCK_STUD_X), start=1):
        shapes[f"shoulder_lock_stud_{index}_McMaster_90318A720"] = _translated(
            INTERFACE.shoulder_lock_stud(), (x, 0.0, 0.0)
        )
        shapes[f"lock_stud_nut_{index}_DIN934_M3"] = _translated(
            INTERFACE.lock_stud_nut(), (x, 0.0, INTERFACE.LOCK_NUT_POCKET_FLOOR)
        )

    shapes["tool_contact_board_FR4"] = _translated(
        INTERFACE.contact_board(), (INTERFACE.CONTACT_CENTER_X, 0.0, 0.0)
    )
    for index, ((x, y), signal) in enumerate(
        zip(INTERFACE.pogo_points(), INTERFACE.CONTACT_SIGNALS), start=1
    ):
        shapes[f"target_pad_P{index}_{signal}"] = _translated(
            INTERFACE.contact_pad(), (x, y, -0.05)
        )
    return shapes


def _interface_components() -> list[Component]:
    components: list[Component] = [
        Component(
            "common_tool_plate",
            common_tool_plate(),
            "pa12_sls_assumption",
            "printed_structure",
            "../generate_cad.py:tool_plate(stock_gripper=False)",
        )
    ]
    for name, shape in tool_interface_hardware_shapes().items():
        if "board" in name:
            material = "fr4"
        elif "pad_" in name:
            material = "copper"
        else:
            material = "steel"
        components.append(
            Component(name, shape, material, "interface_hardware", "reconstructed_catalog_envelope")
        )
    return components


def _identity_resistor(tool_id: int) -> Component:
    # 0603 envelope on the protected rear side of the target PCB.  Its value is
    # assigned during electrical design review; the CAD records only the unique
    # tool-ID mapping and occupied volume.
    shape = (
        cq.Workplane("XY")
        .box(1.6, 0.8, 0.55, centered=(True, True, False))
        .translate((INTERFACE.CONTACT_CENTER_X - 2.4, 8.8, 1.0))
    )
    return Component(
        f"tool_id_{tool_id}_resistor_0603",
        shape,
        "smd_effective",
        "identity_electronics",
        "reconstructed_0603_envelope",
    )


def _carrier_blank() -> cq.Workplane:
    base = INTERFACE.rounded_plate(46.0, 38.0, 8.0, 3.0).translate(
        (0.0, 0.0, INTERFACE.PLATE_THICKNESS)
    )
    # Central service opening and a right-side pocket reserved for the computed
    # steel balance slug.  Both tools use the same datum and retention cap.
    base = base.cut(_axis_cylinder(27.0, 8.2, (0.0, 0.0, INTERFACE.PLATE_THICKNESS - 0.1)))
    base = base.cut(_axis_cylinder(9.1, 8.2, (18.4, 0.0, INTERFACE.PLATE_THICKNESS - 0.1)))
    bridge_front = (
        cq.Workplane("XY")
        .box(46.0, 5.0, 8.0, centered=(True, True, False))
        .translate((0.0, -16.5, INTERFACE.PLATE_THICKNESS))
    )
    bridge_rear = bridge_front.translate((0.0, 33.0, 0.0))
    return base.union(bridge_front).union(bridge_rear).clean()


def spoon_payload_shapes() -> dict[str, cq.Workplane]:
    carrier = _carrier_blank()
    collar = (
        cq.Workplane("XY")
        .circle(6.0)
        .extrude(18.0)
        .translate((0.0, 0.0, INTERFACE.PLATE_THICKNESS))
        .cut(
            _axis_cylinder(
                SPOON_STEM_DIAMETER + 0.25,
                18.2,
                (0.0, 0.0, INTERFACE.PLATE_THICKNESS - 0.1),
            )
        )
    )
    # Opposed radial set-screw bores keep the carrier balanced and give a
    # positive mechanical retention path for the stainless stem.
    for direction in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)):
        start = (-7.0 * direction[0], 0.0, 21.5)
        collar = collar.cut(_axis_cylinder(3.2, 14.0, start, direction))
    carrier = carrier.union(collar).clean()

    stem_end = SPOON_BOWL_CENTER_Z - SPOON_BOWL_OUTER_RADII_YZ[1] + 1.0
    stem = _axis_cylinder(
        SPOON_STEM_DIAMETER,
        stem_end - SPOON_STEM_START_Z,
        (0.0, 0.0, SPOON_STEM_START_Z),
    )

    ry, rz = SPOON_BOWL_OUTER_RADII_YZ
    outer = (
        cq.Workplane("YZ")
        .ellipse(ry, rz)
        .extrude(SPOON_BOWL_DEPTH)
        .translate((-SPOON_BOWL_DEPTH / 2.0, 0.0, SPOON_BOWL_CENTER_Z))
    )
    inner = (
        cq.Workplane("YZ")
        .ellipse(ry - SPOON_BOWL_WALL, rz - SPOON_BOWL_WALL)
        .extrude(SPOON_BOWL_DEPTH - SPOON_BOWL_WALL)
        .translate(
            (
                -SPOON_BOWL_DEPTH / 2.0 + SPOON_BOWL_WALL,
                0.0,
                SPOON_BOWL_CENTER_Z,
            )
        )
    )
    bowl = outer.cut(inner).clean()
    spoon = stem.union(bowl).clean()

    set_screw_x_minus = _socket_set_screw(7.0).rotate((0, 0, 0), (0, 1, 0), 90).translate(
        (-7.0, 0.0, 21.5)
    )
    set_screw_x_plus = _socket_set_screw(7.0).rotate((0, 0, 0), (0, 1, 0), -90).translate(
        (7.0, 0.0, 21.5)
    )
    return {
        "spoon_printed_carrier": carrier,
        "spoon_stainless_insert": spoon,
        "spoon_set_screw_x_minus_M3x6": set_screw_x_minus,
        "spoon_set_screw_x_plus_M3x6": set_screw_x_plus,
    }


def _spring_envelope(outer_diameter: float, inner_diameter: float, z0: float, z1: float) -> cq.Workplane:
    outer = _axis_cylinder(outer_diameter, z1 - z0, (0.0, 0.0, z0))
    inner = _axis_cylinder(inner_diameter, z1 - z0 + 0.2, (0.0, 0.0, z0 - 0.1))
    return outer.cut(inner).clean()


def _whisk_bristles() -> cq.Workplane:
    bristles: cq.Workplane | None = None
    z0 = 97.0
    z1 = 137.0
    for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        start = (4.0 * math.cos(angle), 4.0 * math.sin(angle), z0)
        end = (7.5 * math.cos(angle), 7.5 * math.sin(angle), z1)
        delta = tuple(end[i] - start[i] for i in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        direction = tuple(value / length for value in delta)
        strand = _axis_cylinder(0.9, length, start, direction)
        bristles = strand if bristles is None else bristles.union(strand)
    assert bristles is not None
    return bristles.clean()


def whisk_payload_shapes() -> dict[str, cq.Workplane]:
    carrier = _carrier_blank()
    motor_tower = (
        _axis_cylinder(18.0, 32.5, (0.0, 0.0, 17.0))
        .cut(_axis_cylinder(13.2, 32.7, (0.0, 0.0, 16.9)))
    )
    carrier = carrier.union(motor_tower).clean()

    motor = _axis_cylinder(
        WHISK_MOTOR_DIAMETER,
        WHISK_MOTOR_Z[1] - WHISK_MOTOR_Z[0],
        (0.0, 0.0, WHISK_MOTOR_Z[0]),
    )
    motor_shaft = _axis_cylinder(4.0, 6.0, (0.0, 0.0, WHISK_MOTOR_Z[1]))
    rotor = _axis_cylinder(20.0, 4.0, (0.0, 0.0, 53.5))
    eccentric_pin = _axis_cylinder(4.0, 7.0, (WHISK_ECCENTRIC_MM, 0.0, 57.5))
    counterweight = (
        cq.Workplane("XY")
        .box(7.0, 6.0, 4.0, centered=(True, True, False))
        .translate((-5.5, 0.0, 53.5))
    )
    carriage_x = (
        cq.Workplane("XY")
        .box(26.0, 13.0, 5.0, centered=(True, True, False))
        .translate((0.0, 0.0, 61.0))
        .cut(
            cq.Workplane("XY")
            .center(0.0, 0.0)
            .slot2D(2.0 * WHISK_ECCENTRIC_MM + 4.4, 4.4, 90)
            .extrude(5.2)
            .translate((0.0, 0.0, 60.9))
        )
    )
    compliance_carriage = (
        _axis_cylinder(14.0, 20.0, (0.0, 0.0, 66.0))
        .cut(_axis_cylinder(8.2, 20.2, (0.0, 0.0, 65.9)))
    )
    compliance_spring = _spring_envelope(10.0, 8.4, 67.0, 82.0)
    bellows = _spring_envelope(
        2.0 * WHISK_BELLOWS_RADII_MM[1],
        2.0 * WHISK_BELLOWS_RADII_MM[0],
        WHISK_BELLOWS_Z_MM[0],
        WHISK_BELLOWS_Z_MM[1],
    )
    brush_hub = _axis_cylinder(14.0, 12.0, (0.0, 0.0, 86.0))
    brush_bristles = _whisk_bristles()
    brush_collision_envelope = cq.Workplane(obj=cq.Solid.makeCone(
        7.0,
        8.5,
        40.0,
        cq.Vector(0.0, 0.0, 97.0),
        cq.Vector(0.0, 0.0, 1.0),
    ))
    driver_board = (
        cq.Workplane("XY")
        .box(22.0, 16.0, 1.6, centered=(True, True, False))
        .translate((0.0, 0.0, 13.0))
    )
    return {
        "whisk_printed_carrier": carrier,
        "whisk_motor_12V_envelope": motor,
        "whisk_motor_shaft": motor_shaft,
        "whisk_eccentric_rotor": rotor,
        "whisk_eccentric_pin": eccentric_pin,
        "whisk_rotor_counterweight": counterweight,
        "whisk_carriage_x": carriage_x,
        "whisk_compliance_carriage": compliance_carriage,
        "whisk_compliance_spring": compliance_spring,
        "whisk_food_grade_bellows": bellows,
        "whisk_brush_hub": brush_hub,
        "whisk_bamboo_bristles": brush_bristles,
        "whisk_brush_collision_envelope": brush_collision_envelope,
        "whisk_bus_driver_board": driver_board,
    }


def _payload_components(tool: str) -> list[Component]:
    if tool == "spoon":
        components = []
        for name, shape in spoon_payload_shapes().items():
            if "printed" in name:
                material, role = "pa12_sls_assumption", "printed_structure"
            else:
                material, role = "stainless_304", "food_contact_hardware"
            components.append(Component(name, shape, material, role, "reconstructed_dimensioned_geometry"))
        components.append(_identity_resistor(SPOON_TOOL_ID))
        return components
    if tool != "whisk":
        raise ValueError(f"unknown tool {tool!r}")

    result: list[Component] = []
    for name, shape in whisk_payload_shapes().items():
        if name == "whisk_printed_carrier":
            material, role, mass_override, fabrication = (
                "pa12_sls_assumption",
                "printed_structure",
                None,
                True,
            )
        elif name == "whisk_motor_12V_envelope":
            material, role, mass_override, fabrication = (
                "steel",
                "actuator_hardware",
                0.072,
                True,
            )
        elif name == "whisk_bus_driver_board":
            material, role, mass_override, fabrication = (
                "fr4",
                "actuator_electronics",
                0.008,
                True,
            )
        elif name == "whisk_food_grade_bellows":
            material, role, mass_override, fabrication = (
                "silicone",
                "food_splash_guard",
                None,
                True,
            )
        elif name == "whisk_bamboo_bristles":
            material, role, mass_override, fabrication = (
                "bamboo_dry_assumption",
                "food_contact_hardware",
                None,
                True,
            )
        elif name == "whisk_brush_collision_envelope":
            material, role, mass_override, fabrication = (
                "collision_only",
                "conservative_collision_envelope",
                None,
                False,
            )
        elif "spring" in name:
            material, role, mass_override, fabrication = (
                "stainless_304",
                "compliance_hardware",
                None,
                True,
            )
        elif "carriage" in name or "hub" in name:
            material, role, mass_override, fabrication = (
                "aluminium_6061",
                "mechanism_hardware",
                None,
                True,
            )
        else:
            material, role, mass_override, fabrication = (
                "steel",
                "mechanism_hardware",
                None,
                True,
            )
        result.append(
            Component(
                name,
                shape,
                material,
                role,
                "reconstructed_dimensioned_geometry",
                mass_override,
                fabrication,
            )
        )
    result.append(_identity_resistor(WHISK_TOOL_ID))
    return result


def _mass_properties(component: Component) -> dict[str, object]:
    solid = component.shape.val()
    volume_mm3 = float(solid.Volume())
    center = cq.Shape.centerOfMass(solid)
    geometric_mass_kg = volume_mm3 * component.density_kg_m3 * 1.0e-9
    mass_kg = component.mass_override_kg if component.mass_override_kg is not None else geometric_mass_kg
    scale = mass_kg / volume_mm3 if volume_mm3 > 0.0 else 0.0
    inertia_mm2_kg = cq.Shape.matrixOfInertia(solid)
    inertia_kg_m2 = [
        [float(value) * scale * 1.0e-6 for value in row]
        for row in inertia_mm2_kg
    ]
    return {
        "name": component.name,
        "role": component.role,
        "material": component.material,
        "density_kg_m3": component.density_kg_m3,
        "source": component.source,
        "fabrication": component.fabrication,
        "volume_mm3": volume_mm3,
        "mass_kg": mass_kg,
        "mass_override": component.mass_override_kg is not None,
        "com_mm": [float(center.x), float(center.y), float(center.z)],
        "inertia_about_com_kg_m2": inertia_kg_m2,
    }


def _counterbalance_component(components: Iterable[Component], tool: str) -> Component:
    props = [_mass_properties(component) for component in components if component.fabrication]
    moment_x_kg_mm = sum(float(p["mass_kg"]) * float(p["com_mm"][0]) for p in props)
    slug_x = 18.4
    required_mass_kg = -moment_x_kg_mm / slug_x
    if not 0.0005 <= required_mass_kg <= 0.0080:
        raise RuntimeError(
            f"{tool} counterbalance solution {required_mass_kg * 1000:.3f} g is outside its 0.5-8 g pocket"
        )
    radius = 4.45
    volume_mm3 = required_mass_kg / (MATERIAL_DENSITY_KG_M3["steel"] * 1.0e-9)
    height = volume_mm3 / (math.pi * radius * radius)
    if height > 7.8:
        raise RuntimeError(f"{tool} balance slug height {height:.3f} mm exceeds the carrier pocket")
    shape = _axis_cylinder(2.0 * radius, height, (slug_x, 0.0, INTERFACE.PLATE_THICKNESS + 0.1))
    return Component(
        f"{tool}_balance_slug_steel",
        shape,
        "steel",
        "mass_balance_hardware",
        "computed_from_component_mass_ledger",
    )


def build_tool(tool: str, *, include_collision_envelopes: bool = True) -> list[Component]:
    """Build a complete named tool component roster.

    ``tool`` is ``"spoon"`` or ``"whisk"``.  The returned list includes the
    exact common plate, all coupling hardware, payload hardware, electronics,
    and a computed steel counterbalance.  Collision-only shapes are retained
    by default and are unmistakably labelled with zero density.
    """

    components = _interface_components() + _payload_components(tool)
    components.append(_counterbalance_component(components, tool))
    if not include_collision_envelopes:
        components = [component for component in components if component.fabrication]
    names = [component.name for component in components]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate component names in {tool}: {names}")
    return components


def mass_ledger(tool: str) -> dict[str, object]:
    entries = [_mass_properties(component) for component in build_tool(tool)]
    physical = [entry for entry in entries if entry["fabrication"]]
    total_mass = sum(float(entry["mass_kg"]) for entry in physical)
    com = [
        sum(float(entry["mass_kg"]) * float(entry["com_mm"][axis]) for entry in physical)
        / total_mass
        for axis in range(3)
    ]
    # Parallel-axis theorem, all terms expressed in SI.
    inertia = [[0.0] * 3 for _ in range(3)]
    for entry in physical:
        mass = float(entry["mass_kg"])
        delta_m = [(float(entry["com_mm"][i]) - com[i]) * 1.0e-3 for i in range(3)]
        local = entry["inertia_about_com_kg_m2"]
        r2 = sum(value * value for value in delta_m)
        for i in range(3):
            for j in range(3):
                shift = mass * ((r2 if i == j else 0.0) - delta_m[i] * delta_m[j])
                inertia[i][j] += float(local[i][j]) + shift
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "tool_id": SPOON_TOOL_ID if tool == "spoon" else WHISK_TOOL_ID,
        "bus_address": None if tool == "spoon" else WHISK_BUS_ADDRESS,
        "density_contract": "declared design assumptions; weigh completed hardware before control use",
        "components": entries,
        "total_mass_kg": total_mass,
        "com_mm": com,
        "inertia_about_com_kg_m2": inertia,
        "balance_acceptance": {
            "x_abs_max_mm": 0.05,
            "y_abs_max_mm": 0.05,
            "passed": abs(com[0]) <= 0.05 and abs(com[1]) <= 0.05,
        },
    }


def matcha_dock_stop_spec(bay_name: str | None = None) -> dict[str, object]:
    """Return the exact two-bay stop contract in local or rack coordinates."""

    if bay_name is not None and bay_name not in RACK_BAY_NAMES:
        raise ValueError(f"unknown matcha rack bay {bay_name!r}")
    x_shift = 0.0 if bay_name is None else rack_bay_x(bay_name)
    return {
        "dock": "matcha_local" if bay_name is None else f"matcha_{bay_name}",
        "bay_x_mm": x_shift,
        "bounds_mm": {
            "x": [MATCHA_DOCK_STOP_X_MIN + x_shift, MATCHA_DOCK_STOP_X_MAX + x_shift],
            "y": [MATCHA_DOCK_STOP_Y_MIN, MATCHA_DOCK_STOP_Y_MAX],
            "z": [MATCHA_DOCK_STOP_Z_MIN, MATCHA_DOCK_STOP_Z_MAX],
        },
        "center_mm": [
            (MATCHA_DOCK_STOP_X_MIN + MATCHA_DOCK_STOP_X_MAX) / 2.0 + x_shift,
            (MATCHA_DOCK_STOP_Y_MIN + MATCHA_DOCK_STOP_Y_MAX) / 2.0,
            (MATCHA_DOCK_STOP_Z_MIN + MATCHA_DOCK_STOP_Z_MAX) / 2.0,
        ],
        "size_mm": [
            MATCHA_DOCK_STOP_X_MAX - MATCHA_DOCK_STOP_X_MIN,
            MATCHA_DOCK_STOP_Y_MAX - MATCHA_DOCK_STOP_Y_MIN,
            MATCHA_DOCK_STOP_Z_MAX - MATCHA_DOCK_STOP_Z_MIN,
        ],
    }


def _dock_parts() -> dict[str, cq.Workplane]:
    x_min = -36.0
    x_max = 28.0
    rail_y = -6.0
    rail_length = 86.0
    z_top = INTERFACE.PLATE_THICKNESS
    parts: dict[str, cq.Workplane] = {}
    # Outer walls leave a measured 0.50 mm air gap from the plate outline.
    parts["left_wall"] = (
        cq.Workplane("XY")
        .box(5.0, rail_length, z_top + 6.0, centered=True)
        .translate((x_min - RACK_WALL_CLEARANCE - 2.5, rail_y, z_top / 2.0))
    )
    parts["right_wall"] = (
        cq.Workplane("XY")
        .box(5.0, rail_length, z_top + 6.0, centered=True)
        .translate((x_max + RACK_WALL_CLEARANCE + 2.5, rail_y, z_top / 2.0))
    )
    # Two-millimetre edge engagement; lower ledges are tangent at Z=0 and
    # upper ledges retain the plate with 0.30 mm rear clearance.
    # The target PCB begins at X=-35 mm.  Its FR-4 edge must not share the
    # plate's support contact, so the left lower ledge stops 0.30 mm outboard
    # of the PCB and bears only on the printed electrical wing.
    left_lower_x_min = x_min - RACK_WALL_CLEARANCE - 5.0
    left_lower_x_max = (
        INTERFACE.CONTACT_CENTER_X
        - INTERFACE.CONTACT_BOARD_WIDTH / 2.0
        - RACK_PCB_LOWER_RAIL_CLEARANCE
    )
    parts["left_lower_ledge"] = (
        cq.Workplane("XY")
        .box(left_lower_x_max - left_lower_x_min, rail_length, 3.0, centered=True)
        .translate(((left_lower_x_min + left_lower_x_max) / 2.0, rail_y, -1.5))
    )
    parts["right_lower_ledge"] = (
        cq.Workplane("XY")
        .box(7.5, rail_length, 3.0, centered=True)
        .translate((x_max + 1.75, rail_y, -1.5))
    )
    parts["left_upper_ledge"] = (
        cq.Workplane("XY")
        .box(7.5, rail_length, 3.0, centered=True)
        .translate((x_min - 1.75, rail_y, z_top + RACK_REAR_CLEARANCE + 1.5))
    )
    parts["right_upper_ledge"] = (
        cq.Workplane("XY")
        .box(7.5, rail_length, 3.0, centered=True)
        .translate((x_max + 1.75, rail_y, z_top + RACK_REAR_CLEARANCE + 1.5))
    )
    stop_spec = matcha_dock_stop_spec()
    parts["seating_stop"] = (
        cq.Workplane("XY")
        .box(*stop_spec["size_mm"], centered=True)
        .translate(tuple(stop_spec["center_mm"]))
    )
    # The core generator is the sole cam authority.  Matcha bays reuse the
    # complete X/Y wedge plus axial lead; carrying a locally shifted polygon
    # here would silently change passive-open timing between docks.
    parts["positive_lock_cam"] = INTERFACE.positive_lock_cam()
    return {name: shape.clean() for name, shape in parts.items()}


def build_rack() -> dict[str, cq.Workplane]:
    """Return a named two-bay matcha rack with a common beam and two feet.

    The stock gripper remains on the separate, existing core quick-change dock;
    it is deliberately outside this rack's collision and release authority.
    """

    result: dict[str, cq.Workplane] = {}
    for bay_index, bay_name in enumerate(RACK_BAY_NAMES):
        x_shift = rack_bay_x(bay_name)
        for part_name, shape in _dock_parts().items():
            result[f"dock_{bay_name}_{part_name}"] = shape.translate((x_shift, 0.0, 0.0))
    beam_width = (len(RACK_BAY_NAMES) - 1) * RACK_BAY_PITCH + 92.0
    result["rack_cross_beam"] = (
        cq.Workplane("XY")
        .box(beam_width, 10.0, 18.0, centered=(True, True, False))
        .translate((0.0, 36.0, -8.0))
    )
    for side, x in (("left", -beam_width / 2.0 + 8.0), ("right", beam_width / 2.0 - 8.0)):
        result[f"rack_{side}_foot"] = (
            cq.Workplane("XY")
            .box(24.0, 54.0, 8.0, centered=(True, True, False))
            .translate((x, 17.0, -16.0))
        )
    return result


def rack_bay_x(bay_name: str) -> float:
    """Return the centred rack X datum for a named matcha bay."""

    bay_index = RACK_BAY_NAMES.index(bay_name)
    return (bay_index - (len(RACK_BAY_NAMES) - 1) / 2.0) * RACK_BAY_PITCH


def _shape_compound(components: Iterable[Component]) -> cq.Workplane:
    solids: list[cq.Shape] = []
    for component in components:
        solids.extend(component.shape.vals())
    return _wp(cq.Compound.makeCompound(solids))


def _assembly(name: str, components: Iterable[Component]) -> cq.Assembly:
    assembly = cq.Assembly(name=name)
    for component in components:
        if component.fabrication:
            assembly.add(component.shape, name=component.name)
    return assembly


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonicalize_step_header(path: Path) -> None:
    """Remove OCCT's wall-clock timestamp so equivalent STEP bytes reproduce."""

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


def _export_workplane(shape: cq.Workplane, path: Path) -> None:
    if path.suffix == ".step":
        cq.exporters.export(shape, str(path))
        _canonicalize_step_header(path)
    elif path.suffix == ".stl":
        cq.exporters.export(shape, str(path), tolerance=0.05, angularTolerance=0.10)
    else:
        raise ValueError(path)


def generate_exports(output_dir: Path = EXPORT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    emitted: list[Path] = []
    ledgers: dict[str, object] = {}
    for tool in ("spoon", "whisk"):
        components = build_tool(tool)
        printable = next(component for component in components if component.name == f"{tool}_printed_carrier")
        plate = next(component for component in components if component.name == "common_tool_plate")
        printed_body = plate.shape.union(printable.shape).clean()
        for suffix in ("step", "stl"):
            path = output_dir / f"so101_matcha_{tool}_printed_body.{suffix}"
            _export_workplane(printed_body, path)
            emitted.append(path)
        assembly_path = output_dir / f"so101_matcha_{tool}_tool_assembly.step"
        _assembly(f"so101_matcha_{tool}_tool", components).save(str(assembly_path))
        _canonicalize_step_header(assembly_path)
        emitted.append(assembly_path)
        ledgers[tool] = mass_ledger(tool)
        ledger_path = output_dir / f"so101_matcha_{tool}_mass_ledger.json"
        ledger_path.write_text(json.dumps(ledgers[tool], indent=2, sort_keys=True) + "\n")
        emitted.append(ledger_path)

    rack_parts = build_rack()
    rack_compound = _wp(cq.Compound.makeCompound([shape.val() for shape in rack_parts.values()]))
    for suffix in ("step", "stl"):
        path = output_dir / f"so101_matcha_two_bay_rack.{suffix}"
        _export_workplane(rack_compound, path)
        emitted.append(path)

    def artifact_role(path: Path) -> str:
        name = path.name
        if "mass_ledger" in name:
            return "mass_ledger"
        if "tool_assembly" in name:
            return "complete_rigid_assembly_step"
        if "printed_body.step" in name:
            return "printable_body_step"
        if "printed_body.stl" in name:
            return "printable_body_stl"
        if "two_bay_rack.step" in name:
            return "rack_step"
        if "two_bay_rack.stl" in name:
            return "rack_stl"
        raise RuntimeError(f"unclassified generated artifact: {path}")

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "units": "mm",
        "generator": {
            "path": Path(__file__).name,
            "sha256": _sha256(Path(__file__)),
        },
        "interface_authority": {
            "path": str(BASE_GENERATOR.relative_to(QUICK_CHANGE_DIR.parent.parent)),
            "sha256": _sha256(BASE_GENERATOR),
            "construction": "unmodified tool_plate(stock_gripper=False)",
        },
        "tools": {
            "matcha_spoon": {
                "tool_id": SPOON_TOOL_ID,
                "bus_address": None,
                "component_count": len(build_tool("spoon")),
                "assembly_path": "exports/so101_matcha_spoon_tool_assembly.step",
                "mass_ledger_path": "exports/so101_matcha_spoon_mass_ledger.json",
            },
            "matcha_whisk": {
                "tool_id": WHISK_TOOL_ID,
                "bus_address": WHISK_BUS_ADDRESS,
                "component_count": len(build_tool("whisk")),
                "assembly_path": "exports/so101_matcha_whisk_tool_assembly.step",
                "mass_ledger_path": "exports/so101_matcha_whisk_mass_ledger.json",
            },
        },
        "whisk_mechanism": {
            "eccentric_mm": WHISK_ECCENTRIC_MM,
            "compliance_travel_mm": WHISK_COMPLIANCE_TRAVEL_MM,
            "compliance_limits_mm": list(WHISK_COMPLIANCE_LIMITS_MM),
        },
        "rack": {
            "bay_names": list(RACK_BAY_NAMES),
            "bay_pitch_mm": RACK_BAY_PITCH,
            "wall_clearance_mm": RACK_WALL_CLEARANCE,
            "rear_clearance_mm": RACK_REAR_CLEARANCE,
            "pcb_lower_rail_clearance_mm": RACK_PCB_LOWER_RAIL_CLEARANCE,
            "insertion_y_mm": [RACK_INSERTION_START_Y, RACK_SEATED_Y],
            "stop_contracts": {
                bay_name: matcha_dock_stop_spec(bay_name)
                for bay_name in RACK_BAY_NAMES
            },
            "positive_lock_cam": INTERFACE.positive_lock_cam_contract(),
            "stock_gripper_scope": "separate core quick-change dock; not part of this rack",
        },
        "files": [],
    }
    for path in sorted(emitted):
        manifest["files"].append({
            "path": f"exports/{path.name}",
            "role": artifact_role(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    manifest_path = output_dir / "matcha_tool_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the SO-101 spoon, actuated whisk, and two-bay matcha rack CAD package."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="build both tools and print mass/identity summaries without writing exports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPORT_DIR,
        help="artifact directory (default: matcha_tools/exports)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    if args.check_only:
        print(
            json.dumps(
                {
                    tool: {
                        "tool_id": SPOON_TOOL_ID if tool == "spoon" else WHISK_TOOL_ID,
                        "bus_address": None if tool == "spoon" else WHISK_BUS_ADDRESS,
                        "component_count": len(build_tool(tool)),
                        "total_mass_kg": mass_ledger(tool)["total_mass_kg"],
                        "com_mm": mass_ledger(tool)["com_mm"],
                    }
                    for tool in ("spoon", "whisk")
                },
                indent=2,
            )
        )
        return
    manifest = generate_exports(args.output_dir)
    print(
        json.dumps(
            {
                "generated": len(manifest["files"]),
                "spoon_tool_id": SPOON_TOOL_ID,
                "whisk_tool_id": WHISK_TOOL_ID,
                "whisk_bus_address": WHISK_BUS_ADDRESS,
                "manifest": str(args.output_dir / "matcha_tool_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
