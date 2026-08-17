#!/usr/bin/env python3
"""Focused, seconds-scale checks for the matcha CAD source contract."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import cadquery as cq


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_matcha_tool_cad as cad  # noqa: E402


class MatchaToolCadTests(unittest.TestCase):
    def test_fixed_tool_and_bus_identity(self) -> None:
        self.assertEqual(cad.SPOON_TOOL_ID, 21)
        self.assertEqual(cad.WHISK_TOOL_ID, 22)
        self.assertEqual(cad.WHISK_BUS_ADDRESS, 7)
        self.assertEqual(cad.WHISK_COMPLIANCE_LIMITS_MM, (-5.05, 0.05))
        self.assertAlmostEqual(cad.WHISK_COMPLIANCE_TRAVEL_MM, 5.10)

    def test_common_plate_is_exact_authority_shape(self) -> None:
        actual = cad.common_tool_plate().val()
        expected = cad.INTERFACE.tool_plate(stock_gripper=False).val()
        common = actual.intersect(expected)
        self.assertAlmostEqual(actual.Volume(), expected.Volume(), places=8)
        self.assertAlmostEqual(common.Volume(), expected.Volume(), places=8)
        actual_box = actual.BoundingBox()
        self.assertEqual(
            (actual_box.xmin, actual_box.xmax, actual_box.ymin, actual_box.ymax),
            (-36.0, 28.0, -25.0, 25.0),
        )

    def test_interface_hardware_inventory_is_complete(self) -> None:
        hardware = cad.tool_interface_hardware_shapes()
        self.assertEqual(len(hardware), 15)
        required_families = {
            "target_1_MC-12-12-03",
            "target_2_MC-12-12-03",
            "target_screw_1_ISO10642_M5x10",
            "target_screw_2_ISO10642_M5x10",
            "target_nut_1_DIN934_M5",
            "target_nut_2_DIN934_M5",
            "shoulder_lock_stud_1_McMaster_90318A720",
            "shoulder_lock_stud_2_McMaster_90318A720",
            "lock_stud_nut_1_DIN934_M3",
            "lock_stud_nut_2_DIN934_M3",
            "tool_contact_board_FR4",
            "target_pad_P1_GND",
            "target_pad_P2_+12V",
            "target_pad_P3_TTL_DATA",
            "target_pad_P4_TOOL_ID_SPARE",
        }
        self.assertEqual(set(hardware), required_families)
        self.assertTrue(all(shape.val().Volume() > 0.0 for shape in hardware.values()))

    def test_tool_rosters_have_no_hidden_or_duplicate_rigid_parts(self) -> None:
        for tool, expected_count in (("spoon", 22), ("whisk", 32)):
            components = cad.build_tool(tool)
            names = [component.name for component in components]
            self.assertEqual(len(names), expected_count)
            self.assertEqual(len(names), len(set(names)))
            self.assertIn("common_tool_plate", names)
            self.assertIn(f"{tool}_balance_slug_steel", names)
            self.assertTrue(all(component.shape.val().Volume() > 0.0 for component in components))

        whisk = {component.name: component for component in cad.build_tool("whisk")}
        self.assertEqual(
            {name for name, component in whisk.items() if not component.fabrication},
            {"whisk_brush_collision_envelope"},
        )
        self.assertEqual(whisk["whisk_brush_collision_envelope"].density_kg_m3, 0.0)

    def test_mass_ledgers_are_balanced_and_have_positive_inertia(self) -> None:
        for tool, tool_id, bus in (
            ("spoon", 21, None),
            ("whisk", 22, 7),
        ):
            ledger = cad.mass_ledger(tool)
            self.assertEqual(ledger["tool_id"], tool_id)
            self.assertEqual(ledger["bus_address"], bus)
            self.assertTrue(ledger["balance_acceptance"]["passed"])
            self.assertLessEqual(abs(ledger["com_mm"][0]), 0.05)
            self.assertLessEqual(abs(ledger["com_mm"][1]), 0.05)
            self.assertGreater(ledger["total_mass_kg"], 0.0)
            inertia = ledger["inertia_about_com_kg_m2"]
            self.assertTrue(all(inertia[i][i] > 0.0 for i in range(3)))

    def test_spoon_capacity_and_tool_envelopes_are_declared(self) -> None:
        ry, rz = cad.SPOON_BOWL_OUTER_RADII_YZ
        cavity_mm3 = (
            math.pi
            * (ry - cad.SPOON_BOWL_WALL)
            * (rz - cad.SPOON_BOWL_WALL)
            * (cad.SPOON_BOWL_DEPTH - cad.SPOON_BOWL_WALL)
        )
        self.assertGreater(cavity_mm3 / 1000.0, 0.70)
        self.assertLess(cavity_mm3 / 1000.0, 0.85)
        for tool in ("spoon", "whisk"):
            bounds = cad._shape_compound(cad.build_tool(tool)).val().BoundingBox()
            self.assertGreaterEqual(bounds.xmin, -36.000001)
            self.assertLessEqual(bounds.xmax, 28.000001)
            self.assertGreater(bounds.zmax, 120.0)

    def test_three_bay_rack_source_roster(self) -> None:
        rack = cad.build_rack()
        self.assertEqual(len(rack), 27)
        for bay in cad.RACK_BAY_NAMES:
            names = {name for name in rack if name.startswith(f"dock_{bay}_")}
            self.assertEqual(len(names), 8)
        self.assertEqual(cad.RACK_WALL_CLEARANCE, 0.50)
        self.assertEqual(cad.RACK_REAR_CLEARANCE, 0.30)
        board = cad.tool_interface_hardware_shapes()["tool_contact_board_FR4"].val().BoundingBox()
        left_rail = rack["dock_spoon_left_lower_ledge"].val().BoundingBox()
        # Convert the spoon bay rail back into tool-local X before comparing.
        spoon_bay_x = (
            cad.RACK_BAY_NAMES.index("spoon") - 1
        ) * cad.RACK_BAY_PITCH
        clearance = board.xmin - (left_rail.xmax - spoon_bay_x)
        self.assertAlmostEqual(clearance, cad.RACK_PCB_LOWER_RAIL_CLEARANCE, places=8)
        self.assertGreaterEqual(clearance + 1.0e-9, 0.30)


if __name__ == "__main__":
    unittest.main()
