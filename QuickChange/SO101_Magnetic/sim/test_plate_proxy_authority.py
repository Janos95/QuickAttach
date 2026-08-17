#!/usr/bin/env python3
"""Focused, bounded tests for the standalone STEP/FCPW plate authority."""

from __future__ import annotations

import copy
import unittest

import numpy as np

import generate_plate_proxy_authority as generator
import validate_plate_proxy_authority as validator


class PlateProxySourceAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = generator.plate_source_specs()

    def test_released_step_hashes_and_absolute_meshes_are_closed(self) -> None:
        expected = {
            "robot_plate": (10_570, 10),
            "generic_tool_plate": (5_482, 8),
        }
        for component_id, (triangle_count, void_count) in expected.items():
            with self.subTest(component_id=component_id):
                preflight = generator.preflight_component(self.specs[component_id])
                self.assertTrue(preflight["passed"])
                self.assertEqual(preflight["source_triangle_count"], triangle_count)
                self.assertEqual(preflight["functional_void_family_count"], void_count)
                certificate = preflight["tessellation_certificate"]
                self.assertFalse(certificate["relative_mode"])
                self.assertEqual(
                    certificate["requested_absolute_linear_deflection_mm"], 0.005
                )
                self.assertEqual(
                    certificate["source_to_faceted_surface_bound_mm"], 0.010
                )
                self.assertTrue(certificate["topology"]["watertight"])
                self.assertTrue(certificate["topology"]["orientation_consistent"])
                self.assertTrue(certificate["topology"]["positive_volume"])

    def test_functional_void_rosters_close_exact_named_families(self) -> None:
        for component_id, expected in generator.EXPECTED_VOID_FAMILIES.items():
            roster = generator.functional_void_roster(component_id)
            self.assertEqual({record["family"] for record in roster}, expected)
            self.assertTrue(all(record["probe_count"] > 0 for record in roster))
            self.assertTrue(all(len(record["probe_sha256"]) == 64 for record in roster))

    def test_source_frame_rejects_even_self_consistent_rotation(self) -> None:
        spec = self.specs["robot_plate"]
        frame = generator._frame_payload(spec)
        self.assertTrue(validator.validate_source_frame(frame, spec))
        mutated = copy.deepcopy(frame)
        mutated["body_local_quat_wxyz"] = [2**-0.5, 0.0, 0.0, 2**-0.5]
        core = {key: value for key, value in mutated.items() if key != "frame_sha256"}
        mutated["frame_sha256"] = validator._canonical_sha256(core)
        self.assertFalse(validator.validate_source_frame(mutated, spec))


class PlateProxyConstructionKernelTests(unittest.TestCase):
    def test_greedy_merge_preserves_exact_cell_union(self) -> None:
        records = np.asarray(
            [
                [0, 1, 0, 1, 0, 1],
                [1, 2, 0, 1, 0, 1],
                [0, 1, 1, 2, 0, 1],
                [1, 2, 1, 2, 0, 1],
            ],
            dtype=np.int32,
        )
        margins = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        merged, merged_margins, record = generator.greedy_merge_boxes(records, margins)
        np.testing.assert_array_equal(merged, np.asarray([[0, 2, 0, 2, 0, 1]]))
        np.testing.assert_allclose(merged_margins, [0.1])
        self.assertTrue(record["union_preserved_exactly"])
        self.assertFalse(record["material_added"])

    def test_signed_distance_lower_bound_never_exceeds_exact_box_distance(self) -> None:
        source = generator._box_triangles(
            np.asarray([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        )
        index = generator.SignedFcpwMesh(source)
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [-2.5, 0.0, 0.0],
            ]
        )
        inside, lower, upper = index.query(points)
        np.testing.assert_array_equal(inside, [True, True, False, False])
        exact = np.asarray([2.0, 1.0, 1.0, 0.5])
        self.assertTrue(np.all(lower <= exact + 1.0e-12))
        self.assertTrue(np.all(upper >= exact - 1.0e-12))

    def test_small_octree_boxes_are_whole_cell_subsets_and_compact(self) -> None:
        source = generator._box_triangles(
            np.asarray([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        )
        parameters = generator.OctreeParameters(
            maximum_depth=6,
            boundary_threshold_mm=0.35,
            source_faceting_bound_mm=0.01,
            source_witness_covering_radius_mm=0.12,
            max_proxy_boxes_after_merge=5_000,
        )
        boxes, result, _ = generator.build_adaptive_subset_boxes(source, parameters)
        self.assertTrue(result["exact_subset"]["passed"])
        self.assertGreater(result["exact_subset"]["minimum_exact_subset_margin_mm"], 0.0)
        self.assertLessEqual(
            result["octree"]["unresolved_boundary_maximum_mm"], 0.35
        )
        recomputed = validator.recompute_exact_subset(source, boxes, parameters)
        self.assertTrue(recomputed["passed"])
        self.assertLess(len(boxes), result["exact_subset"]["accepted_cell_count_before_merge"])

    def test_piece_inventory_is_hash_closed_and_derived_values_recomputed(self) -> None:
        boxes = np.asarray(
            [
                [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0]],
                [[0.0, -1.0, 0.0], [1.0, 1.0, 1.0]],
            ]
        )
        inventory = generator._boxes_inventory(boxes)
        parsed = validator.boxes_from_inventory(inventory)
        np.testing.assert_allclose(parsed, boxes)
        inventory["pieces"][0]["center_mm"][0] += 0.25
        with self.assertRaises(ValueError):
            validator.boxes_from_inventory(inventory)


class PlateProxyAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = generator.plate_source_specs()["robot_plate"]
        cls.source_triangles = generator._box_triangles(
            np.asarray([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        )
        cls.frame = generator._frame_payload(spec)
        cls.boxes = np.asarray([[[-1.92, -1.92, -1.92], [1.92, 1.92, 1.92]]])
        cls.parameters = generator.OctreeParameters(
            maximum_depth=8,
            boundary_threshold_mm=0.35,
            source_witness_covering_radius_mm=0.12,
        )
        if not np.all(cls.boxes[:, 0] >= -2.0) or not np.all(cls.boxes[:, 1] <= 2.0):
            raise RuntimeError("adversarial fixture is not an exact analytic subset")
        boundary = generator.certify_source_to_proxy_surface(
            cls.source_triangles, cls.boxes, cls.parameters
        )
        if not boundary["passed"]:
            raise RuntimeError("adversarial fixture is not a passing boundary proxy")

    def test_all_five_geometry_and_frame_mutations_are_rejected(self) -> None:
        result = validator.run_adversarial_suite(
            "robot_plate",
            self.source_triangles,
            self.boxes,
            self.parameters,
            self.frame,
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"], result)
        self.assertEqual(
            {record["case"] for record in result["cases"]},
            {
                "translated_plus_200mm",
                "filled_functional_bore",
                "deleted_wall",
                "injected_outside_cell",
                "rotated_source_frame",
            },
        )


if __name__ == "__main__":
    unittest.main()
