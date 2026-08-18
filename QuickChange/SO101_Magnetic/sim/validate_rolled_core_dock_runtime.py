#!/usr/bin/env python3
"""Generate and validate the rolled core-dock runtime checkpoint.

The checkpoint consumes the public source/runtime contracts, independently
enumerates the compiled arm and dock geometry, replays the off-default 31-row
release continuation, derives a topology motion bound, and computes
conservative outer-AABB clearance lower bounds.  A green report grants only
hash-bound geometric authority.  It never grants material, fastener,
substrate, contact, dynamics, physical-lock, or release authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = HERE / "rolled_core_dock_runtime_report.json"
VALIDATOR_RELATIVE_PATH = (
    "QuickChange/SO101_Magnetic/sim/validate_rolled_core_dock_runtime.py"
)
REPORT_RELATIVE_PATH = (
    "QuickChange/SO101_Magnetic/sim/rolled_core_dock_runtime_report.json"
)
SCHEMA_VERSION = "1.0-rolled-core-dock-runtime-authority"

EXPECTED_DIRECT_SOURCE_RECORDS = (
    (
        "runtime_qc_implementation",
        "QuickChange/SO101_Magnetic/sim/qc_collision_geometry.py",
        119_264,
        "0739de8ba54af81b576596360dfa7b9eb34618d0d71182fe31662644e562c0ba",
    ),
    (
        "runtime_workflow_implementation",
        "QuickChange/SO101_Magnetic/sim/matcha_workflow_demo.py",
        462_659,
        "42414b1acdffe39c2affef94f6dd1df28f125670851036e54e6b4910d208ca7c",
    ),
    (
        "runtime_geometry_source",
        "QuickChange/SO101_Magnetic/sim/matcha_tool_geometry.json",
        6_359,
        "ee411b37e867d93a7fa21b00aeda47172579bd9db48d6ea963ec2345d013a065",
    ),
    (
        "runtime_scene_source",
        "QuickChange/SO101_Magnetic/sim/matcha_workflow_scene.xml",
        1_994,
        "756fc91d01ec5ed2bacda41035b3494f637d9dc6241189cfdb1d1540cd10749d",
    ),
    (
        "upstream_robot_source",
        "Simulation/SO101/so101_new_calib.xml",
        13_921,
        "d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580",
    ),
)
EXPECTED_QC_SOURCE_FILES_CANONICAL_SHA256 = (
    "d59f906102c74512a08d22928737c37a0987276f9719d062120ec80e9340c157"
)
EXPECTED_TRIPLE_CONTRACT_LOCATIONS_CANONICAL_SHA256 = (
    "f4379718a4c2d08165366108805952d65fbca62c66b8891449a2f9a5b0a0e60d"
)
EXPECTED_SUPPORT_COMPONENTS_CANONICAL_SHA256 = (
    "146f1b40c3fe304ce873f6fe7e20a1605e98f46707abaf477cc60115055dfc3d"
)
EXPECTED_RUNTIME_FRAME_CANONICAL_SHA256 = (
    "c22819ccda90572adf075db35f54692d91be8f2201f87c0329ec05a21c3308a1"
)
EXPECTED_RUNTIME_SUPPORT_CONTRACT_SHA256 = (
    "addb9021b2420ebd6314d656c68923e12632b5ae1df9b80995e4934b4cec991c"
)
EXPECTED_SOURCE_CONTRACT_SHA256 = (
    "1befdd739c9eea2645408178317b93c720d61d11c4fd31802761626ac1b456ee"
)
EXPECTED_SUPPORT_SOURCE_VOLUME_MM3 = 162_415.4180526403
EXPECTED_SUPPORT_PROXY_VOLUME_MM3 = 177_797.24575315934
EXPECTED_SUPPORT_PROXY_EXCESS_VOLUME_MM3 = 15_381.827700519032
EXPECTED_SUPPORT_SOURCE_MISSING_VOLUME_MM3 = 0.0
EXPECTED_SUPPORT_PASSAGE_WITNESS_M = (-0.002, 0.100, 0.00475)
EXPECTED_ROLL_DEG = -87.21086925015224

EXPECTED_CAPTURE_ROUTE = {
    "row_count": 276,
    "f8le_shape": [276, 7],
    "state_bytes_sha256": "3df49a196375ee9ec2f2aab2a8750c7ab0d8f4f8bde79144630cf406c7c008cc",
    "source_state_sha256": "000c6323497fe2b99d03d19f831524d0dc02a2ef6bdbac41eaa8da8d4520719b",
    "q_roster_sha256": "141701b473e207995738331a6324f8c1b9f65af1472a98a5d7e3138eeff1891e",
    "wrist_roll_q_sha256": "e3fb0426f9defe378ee7d245a30c216321cc0335c12f1bd555e2614696796854",
    "wrist_roll_f8le_sha256": "1aa8b9c2d5348864b0b7ad4ee3e1d999e4778572289fcfd3aa461a5862a1aff4",
    "phase_contract_sha256": "eda9370110ed58ac4d625fa6d5015398c8c71f1501b2498cf274e0510e7cad77",
    "alignment_q_sha256": "692cc966e266841bbe864d180a90083410089bd85f327c3f579276625a66e22c",
    "contract_identity_sha256": "1db1e8a2325a4afdc451330c2b464ee9d85930a956ca22b8f073bdfa16105088",
    "endpoint_fk_sha256": "f209f29fb26d24e3acd32e198340fcfb96d97022b1b2d5f681f90fdfda654b75",
    "endpoint_fk_maxima_sha256": "4d6672e5955d670e2a608bbfc41f01542877d3aebade080204cd83751a65066b",
    "dense_fk_sha256": "703f7fc4b77b1fa3c97fd499b99b4c7911660429e762ce359adad021be4a804b",
}
EXPECTED_GRAVITY_FF = {
    "identity_sha256": "a039042d0c32263fe9565000f8eee7e442d6be06f48b3834d77861f26b36c4e2",
    "formula_sha256": "a84c10e16c890b5e1ee4e4479c0d15d7e07a75f2afae17c62e639e8adc55cc27",
    "guard_thresholds_sha256": "a30d3c871580b36a8f18eca6f07b6d4e5eee34cbecf50093aaca7c4cb1d3ee40",
    "move_action_roster_sha256": "3a3301d0adf98e1e617766891ebf7be4129d94d0440d7f4fcb694f273313a3ba",
}
EXPECTED_MODEL_IDENTITIES = {
    "assembled_xml_sha256": "d919728e7108061f7ede7bd74991c2b5e42fa0985d5e731c30c95e0a660a953a",
    "compiled_model_xml_equivalent_sha256": "edfc58afb55d83901f3e35f7e3426d5ffeef696257ef55e28955b400249480d0",
    "initialized_active_collision_geometry_sha256": "401cbab95925ec5688d54b875c7d9e2e3788ebaa717cc2eb92fe94b1600b2083",
    "binary_asset_records_canonical_sha256": "7beb0a1c6868c03385d91ebdcb3a4861d367f69485657882714069e1b8b8a0ee",
}
EXPECTED_AUDIT_STREAMS = {
    "arm_geom_names_sha256": "bdd87258161b28f976a6f5db11bccb4ad044ec656f5ef84f20dda066db12024b",
    "arm_compiled_geom_records_sha256": "ffd5c33711ec1322092267d6f2dda24011b7284909214319d407b85ce60269c7",
    "support_geom_names_sha256": "370894d7664ca36ea30914208ce074c0a86d44fee2547da12b41c85ed5a26277",
    "dock_target_geom_names_sha256": "5177a7d663660482e3b21343ab244f301d2d443752289917c50609a9f53e6b55",
    "dock_target_compiled_geom_records_sha256": "bc9f846cab565aaf4724364f7c03571f6e191415526e149c3ed5f03cc84abc24",
    "sample_q_sha256": "9a7752fd420b897c74fdc6fc159fd6644a7914bb2eec78a2e8717a742bd0667a",
    "distance_stream_sha256": "75b6a8ad31dcbd390e3434f9dde6c16c9a47e551d364eceba3a6bb966ead93db",
    "topology_radius_records_sha256": "cb90b85ece50da441459d5e0820fbe133aebfedc67a2a73db9d6c133bcb72969",
    "row_fk_records_sha256": "d65bb2b78a6d02f2edb20d9ab439c9bbefb4cdb7d57dd8b8cb47e40530ab37f4",
    "subsample_fk_stream_sha256": "578291420b2cc64baf40389cb18b48527b441dfa2e73ff3277eb29f30808b760",
    "startup_contact_records_sha256": "55b39fe064803cfd99938c8fd9ed37b2b84304051d106efbe5a39fe2b3af90a1",
}
EXPECTED_MINIMUM_SAMPLED_CLEARANCE_MM = 4.3592095380000915
EXPECTED_MAXIMUM_TOPOLOGY_MOTION_BOUND_MM = 0.13870802188957512
EXPECTED_CONTINUOUS_CLEARANCE_LOWER_BOUND_MM = 4.220501516110517
EXPECTED_SUPPORT_MINIMUM_SAMPLED_CLEARANCE_MM = 16.34024798273126
EXPECTED_SUPPORT_CONTINUOUS_CLEARANCE_LOWER_BOUND_MM = 16.201539960841686
EXPECTED_DEFAULT_ACTION_NAMES = (
    "gripper_capture_lateral_align",
    "gripper_capture_axial_open_side",
    "gripper_capture_coupled_recenter",
    "gripper_capture_centered_final",
    "gripper_physical_capture",
    "gripper_lock_verify",
    "gripper_dock_release_verify",
)
CHECKPOINT_BLOCKERS = (
    "printed_material_and_process_allowables_unqualified",
    "M4_M6_fastener_and_joint_authority_missing",
    "floor_substrate_and_thread_authority_missing",
    "contact_force_and_friction_authority_missing",
    "capture_and_release_dynamics_not_validated",
    "physical_lock_and_reverse_insertion_not_validated",
)

EXPECTED_ARM_NAMES_BY_BODY = {
    "base": tuple(f"robot_col_base_{index:02d}" for index in range(4)),
    "shoulder": tuple(f"robot_col_shoulder_{index:02d}" for index in range(3)),
    "upper_arm": tuple(f"robot_col_upper_arm_{index:02d}" for index in range(2)),
    "lower_arm": tuple(f"robot_col_lower_arm_{index:02d}" for index in range(3)),
    "wrist": tuple(f"robot_col_wrist_{index:02d}" for index in range(2)),
}
EXPECTED_ARM_COUNT = 14
EXPECTED_SUPPORT_COUNT = 11
EXPECTED_CORE_DOCK_STOP_PART_COUNT = 68
EXPECTED_DOCK_TARGET_COUNT = 90
EXPECTED_SUPPORT_TOPOLOGY_TANGENCY_COUNT = 15
EXPECTED_MODEL_COUNTS = {
    "nq": 37,
    "nbody": 40,
    "ngeom": 1726,
    "nmesh": 1318,
    "ncontact_pair": 15,
}
EXPECTED_RELEASE_ROWS = 31
EXPECTED_RELEASE_SHA256 = (
    "f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293"
)
EXPECTED_RELEASE_F8LE_SHA256 = (
    "bd7e32c0197644d4277966e6eeb129603560c261aa9fa38d43fdb224d4acd492"
)
EXPECTED_RELEASE_Q_ROSTER_SHA256 = (
    "030ed1acc312636cf85a1bdf580e2a9a17c7b5e0e7acbc9f0f482cc077ff8652"
)
EXPECTED_RELEASE_SLIDER_ROWS_SHA256 = (
    "690179a9a521d4a37cb9ec7ee8de4c501160e9ea1a741953b1c1e908726e2bf1"
)
EXPECTED_STATIC_RELEASE_CONTRACT_SHA256 = (
    "765d7ce132cef99774d23c6887ca2fe834a0bd7491ebf1d88ea776bc3672ab47"
)
EXPECTED_CORE_DOCK_POSITION_M = (
    0.19082795371216685,
    0.1330713713445051,
    0.1939154579377553,
)
EXPECTED_CORE_DOCK_QUAT_WXYZ = (
    0.6440855284765126,
    -0.6440855284765125,
    0.2918112952014223,
    -0.2918112952014225,
)
EXPECTED_SPOON_POSE = (
    (0.24084947630993864, -0.0001778089765695206, 0.1939154579377552),
    (-0.017209108230571146, 0.7068973380865915, -0.01720910823057085, 0.706897338086591),
)
EXPECTED_WHISK_POSE = (
    (0.19059347652281455, -0.13333873141492703, 0.19391545793775508),
    (0.23291576804733463, 0.6555206479743557, -0.26512766750997674, 0.6676452987889),
)
EXPECTED_RUNTIME_BLOCKERS = (
    "vendor_or_normative_source_missing_for_selected_M4_and_M6_fasteners",
    "floor_fixture_substrate_and_M6_thread_authority_missing",
    "PA12_modulus_strength_creep_and_process_allowables_unqualified",
    "printed_dimensional_tolerance_and_anchor_strength_unqualified",
    "cam_contact_friction_reverse_insertion_and_capture_dynamics_unvalidated",
    "runtime_support_proxy_fills_M4_M6_clearances_and_countersinks",
    "runtime_support_proxy_head_envelope_overcovers_source_for_nonoverlap",
    "runtime_support_proxy_is_not_physical_or_load_path_authority",
)
SUBSTEPS_PER_INTERVAL = 10
REQUIRED_CONTINUOUS_CLEARANCE_MM = 0.20
SUBSAMPLE_PATH_POSITION_TOLERANCE_MM = 0.001
SUBSAMPLE_PATH_ORIENTATION_TOLERANCE_RAD = math.radians(0.1)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file_record(repo: Path, role: str, relative_path: str) -> dict[str, Any]:
    path = repo / relative_path
    payload = path.read_bytes()
    return {
        "role": role,
        "path": relative_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def rotation_angle(rotation: np.ndarray) -> float:
    sine = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) * 0.5
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return float(math.atan2(float(np.linalg.norm(sine)), cosine))


def active_geom(model: Any, geom_id: int) -> bool:
    return bool(
        int(model.geom_contype[geom_id])
        or int(model.geom_conaffinity[geom_id])
    )


def geom_local_vertices(model: Any, mujoco: Any, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.asarray(
            [
                (sx * size[0], sy * size[1], sz * size[2])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
    raise RuntimeError(
        f"unsupported audit geom type {geom_type} for {model.geom(geom_id).name}"
    )


def geom_world_vertices(
    model: Any, data: Any, mujoco: Any, geom_id: int
) -> np.ndarray:
    local = geom_local_vertices(model, mujoco, geom_id)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    return position + local @ rotation.T


def compiled_geom_record(model: Any, mujoco: Any, geom_id: int) -> dict[str, Any]:
    """Return the exact compiled inputs used for one audit geometry."""

    vertices = np.asarray(geom_local_vertices(model, mujoco, geom_id), dtype="<f8")
    return {
        "name": str(model.geom(geom_id).name),
        "body": str(model.body(int(model.geom_bodyid[geom_id])).name),
        "type": int(model.geom_type[geom_id]),
        "data_id": int(model.geom_dataid[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
        "pos_m": np.asarray(model.geom_pos[geom_id], dtype=np.float64).tolist(),
        "quat_wxyz": np.asarray(
            model.geom_quat[geom_id], dtype=np.float64
        ).tolist(),
        "size": np.asarray(model.geom_size[geom_id], dtype=np.float64).tolist(),
        "local_vertex_count": int(vertices.shape[0]),
        "local_vertices_f8le_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
    }


def bounds(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def aabb_distance(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> float:
    left_min, left_max = left
    right_min, right_max = right
    gap = np.maximum(0.0, np.maximum(right_min - left_max, left_min - right_max))
    return float(np.linalg.norm(gap))


def body_chain(model: Any, ancestor: int, descendant: int) -> list[int]:
    reverse = [descendant]
    current = descendant
    while current != ancestor:
        current = int(model.body_parentid[current])
        if current == 0 and ancestor != 0:
            raise RuntimeError(
                f"body {model.body(ancestor).name} is not an ancestor of "
                f"{model.body(descendant).name}"
            )
        reverse.append(current)
    return list(reversed(reverse))


def topology_radius_bounds(
    model: Any,
    mujoco: Any,
    arm_geom_ids: list[int],
    arm_joint_ids: list[int],
) -> tuple[dict[int, dict[int, float]], float, list[dict[str, Any]]]:
    """Return rigorous point-to-ancestor-joint radius upper bounds.

    Each descendant path is bounded by the triangle inequality over fixed
    body offsets plus the exact maximum compiled geom-vertex radius.  Rotations
    preserve every term, so the result is configuration independent.
    """

    joint_body = {joint_id: int(model.jnt_bodyid[joint_id]) for joint_id in arm_joint_ids}
    per_geom: dict[int, dict[int, float]] = {}
    records: list[dict[str, Any]] = []
    global_bound = 0.0
    for geom_id in arm_geom_ids:
        geom_body = int(model.geom_bodyid[geom_id])
        geom_rotation = np.asarray(model.geom_quat[geom_id], dtype=np.float64)
        # Convert the compiled local quaternion through MuJoCo itself by using
        # mju_quat2Mat, avoiding a second convention implementation.
        matrix_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix_flat, geom_rotation)
        rotation = matrix_flat.reshape(3, 3)
        local_vertices = geom_local_vertices(model, mujoco, geom_id)
        body_points = (
            np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            + local_vertices @ rotation.T
        )
        geom_body_radius = float(np.max(np.linalg.norm(body_points, axis=1)))
        bounds_by_joint: dict[int, float] = {}
        for joint_id in arm_joint_ids:
            ancestor_body = joint_body[joint_id]
            current = geom_body
            ancestors: set[int] = set()
            while current != 0:
                ancestors.add(current)
                current = int(model.body_parentid[current])
            if ancestor_body not in ancestors:
                continue
            chain = body_chain(model, ancestor_body, geom_body)
            joint_pos = np.asarray(model.jnt_pos[joint_id], dtype=np.float64)
            if len(chain) == 1:
                radius = float(
                    np.max(np.linalg.norm(body_points - joint_pos, axis=1))
                )
            else:
                first_offset = np.asarray(model.body_pos[chain[1]], dtype=np.float64)
                radius = float(np.linalg.norm(first_offset - joint_pos))
                for body_id in chain[2:]:
                    radius += float(
                        np.linalg.norm(
                            np.asarray(model.body_pos[body_id], dtype=np.float64)
                        )
                    )
                radius += geom_body_radius
            bounds_by_joint[joint_id] = radius
            global_bound = max(global_bound, radius)
        per_geom[geom_id] = bounds_by_joint
        records.append(
            {
                "geom": str(model.geom(geom_id).name),
                "body": str(model.body(geom_body).name),
                "geom_body_radius_m": geom_body_radius,
                "joint_radius_bounds_m": {
                    str(model.joint(joint_id).name): radius
                    for joint_id, radius in bounds_by_joint.items()
                },
            }
        )
    return per_geom, global_bound, records


def build_rolled_core_dock_runtime_report(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Recompute and return the deterministic rolled runtime report."""

    repo = Path(repository_root).resolve()
    sim_dir = repo / "QuickChange" / "SO101_Magnetic" / "sim"
    consumed_file_specs = tuple(
        (role, relative_path)
        for role, relative_path, _bytes, _sha256 in EXPECTED_DIRECT_SOURCE_RECORDS
    )
    consumed_files_before = [
        source_file_record(repo, role, relative_path)
        for role, relative_path in consumed_file_specs
    ]
    sys.path.insert(0, str(sim_dir))

    import matcha_workflow_demo as demo

    mujoco = demo.mujoco
    qc = demo.qc
    support_contract = qc.core_dock_floor_support_runtime_contract()
    capture_contract = demo.core_capture_route_runtime_contract()
    static_release_contract = demo.core_dock_static_release_route_contract()
    xml_text, assets = demo._build_xml_and_assets()
    asset_records = [
        {
            "name": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(assets.items())
    ]
    model = demo.build_model()
    data = mujoco.MjData(model)
    demo.initialize(model, data)

    failures: list[str] = []
    expected_direct_records = [
        {
            "role": role,
            "path": relative_path,
            "bytes": byte_count,
            "sha256": digest,
        }
        for role, relative_path, byte_count, digest in EXPECTED_DIRECT_SOURCE_RECORDS
    ]
    if consumed_files_before != expected_direct_records:
        failures.append("direct_source_identity")
    observed_model_counts = {
        "nq": int(model.nq),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "ncontact_pair": int(model.npair),
    }
    if observed_model_counts != EXPECTED_MODEL_COUNTS:
        failures.append(f"compiled_model_counts:{observed_model_counts!r}")
    expected_core_dock_pose = (
        EXPECTED_CORE_DOCK_POSITION_M,
        EXPECTED_CORE_DOCK_QUAT_WXYZ,
    )
    if tuple(qc.CORE_DOCK_WORLD_POS_M) != EXPECTED_CORE_DOCK_POSITION_M:
        failures.append("core_dock_source_position_changed")
    if tuple(qc.CORE_DOCK_WORLD_QUAT_WXYZ) != EXPECTED_CORE_DOCK_QUAT_WXYZ:
        failures.append("core_dock_source_quaternion_changed")
    if demo.DOCK_POSES["gripper"] != expected_core_dock_pose:
        failures.append("workflow_core_dock_pose_changed")
    if support_contract["runtime_frame"]["position_m"] != list(
        EXPECTED_CORE_DOCK_POSITION_M
    ):
        failures.append("contract_core_dock_position_changed")
    if support_contract["runtime_frame"]["quat_wxyz"] != list(
        EXPECTED_CORE_DOCK_QUAT_WXYZ
    ):
        failures.append("contract_core_dock_quaternion_changed")
    required_false_contract_fields = {
        "release.physical_release_authority": support_contract["release"][
            "physical_release_authority"
        ],
        "support_proxy.exact_source_brep_boundary_authority": support_contract[
            "support_proxy"
        ]["exact_source_brep_boundary_authority"],
        "support_proxy.physical_geometry_authority": support_contract[
            "support_proxy"
        ]["physical_geometry_authority"],
        "support_proxy.mass_authority": support_contract["support_proxy"][
            "mass_authority"
        ],
        "support_proxy.fastener_authority": support_contract["support_proxy"][
            "fastener_authority"
        ],
        "support_proxy.substrate_authority": support_contract["support_proxy"][
            "substrate_authority"
        ],
        "support_proxy.load_path_authority": support_contract["support_proxy"][
            "load_path_authority"
        ],
        "support_proxy.tolerance_authority": support_contract["support_proxy"][
            "tolerance_authority"
        ],
        "authority_scope.capture_dynamics_authority": support_contract[
            "authority_scope"
        ]["capture_dynamics_authority"],
        "authority_scope.physical_release_authority": support_contract[
            "authority_scope"
        ]["physical_release_authority"],
        "authority_scope.release_ready": support_contract["authority_scope"][
            "release_ready"
        ],
        "top_level.release_ready": support_contract["release_ready"],
    }
    for field, value in required_false_contract_fields.items():
        if value is not False:
            failures.append(f"runtime_contract_false_promotion:{field}")
    observed_runtime_blockers = tuple(
        support_contract["authority_scope"]["blockers"]
    )
    if observed_runtime_blockers != EXPECTED_RUNTIME_BLOCKERS:
        failures.append("runtime_contract_blocker_inventory")
    if support_contract["contract_integrity_passed"] is not True:
        failures.append("runtime_contract_integrity")

    source_binding = support_contract["source_binding"]
    if canonical_sha256(source_binding["files"]) != (
        EXPECTED_QC_SOURCE_FILES_CANONICAL_SHA256
    ):
        failures.append("qc_source_file_roster")
    if canonical_sha256(source_binding["triple_contract_locations"]) != (
        EXPECTED_TRIPLE_CONTRACT_LOCATIONS_CANONICAL_SHA256
    ):
        failures.append("triple_contract_locations")
    if source_binding["triple_contract_equal"] is not True:
        failures.append("triple_contract_not_equal")
    if source_binding["canonical_sha256"] != EXPECTED_SOURCE_CONTRACT_SHA256:
        failures.append("source_contract_identity")
    if canonical_sha256(support_contract["runtime_frame"]) != (
        EXPECTED_RUNTIME_FRAME_CANONICAL_SHA256
    ):
        failures.append("runtime_frame_identity")

    support_components = support_contract["support_proxy"]["components"]
    if canonical_sha256(support_components) != (
        EXPECTED_SUPPORT_COMPONENTS_CANONICAL_SHA256
    ):
        failures.append("support_component_identity")
    analytic_volume_mm3 = math.fsum(
        math.prod(float(axis[1]) - float(axis[0]) for axis in record["bounds_m"])
        * 1.0e9
        for record in support_components
    )
    positive_overlap_count = 0
    for index, left in enumerate(support_components):
        for right in support_components[index + 1 :]:
            overlap_m3 = math.prod(
                max(
                    0.0,
                    min(float(left["bounds_m"][axis][1]), float(right["bounds_m"][axis][1]))
                    - max(float(left["bounds_m"][axis][0]), float(right["bounds_m"][axis][0])),
                )
                for axis in range(3)
            )
            positive_overlap_count += int(overlap_m3 > 0.0)

    def point_inside_support_proxy(point: tuple[float, float, float]) -> bool:
        return any(
            all(
                float(axis[0]) < coordinate < float(axis[1])
                for coordinate, axis in zip(point, record["bounds_m"], strict=True)
            )
            for record in support_components
        )

    passage_witness_inside = point_inside_support_proxy(
        EXPECTED_SUPPORT_PASSAGE_WITNESS_M
    )
    hole_witnesses = [
        list(value) for value in qc.CORE_DOCK_SUPPORT_PROXY_FILLED_HOLE_WITNESSES_M
    ]
    hole_witness_inside = [
        point_inside_support_proxy(tuple(value)) for value in hole_witnesses
    ]
    source_missing_volume_mm3 = float(
        qc.CORE_DOCK_SUPPORT_PROXY_SOURCE_MISSING_VOLUME_MM3
    )
    proxy_excess_volume_mm3 = analytic_volume_mm3 - float(
        qc.CORE_DOCK_SUPPORT_SOURCE_VOLUME_MM3
    )
    if not math.isclose(
        analytic_volume_mm3,
        EXPECTED_SUPPORT_PROXY_VOLUME_MM3,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        failures.append("support_analytic_volume")
    if positive_overlap_count != 0:
        failures.append("support_positive_overlap")
    if passage_witness_inside:
        failures.append("support_passage_filled")
    if not all(hole_witness_inside):
        failures.append("support_hole_not_filled")
    if source_missing_volume_mm3 != EXPECTED_SUPPORT_SOURCE_MISSING_VOLUME_MM3:
        failures.append("support_source_missing_volume")
    if not math.isclose(
        proxy_excess_volume_mm3,
        EXPECTED_SUPPORT_PROXY_EXCESS_VOLUME_MM3,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        failures.append("support_excess_volume")

    capture_rows = [
        [float(preseat_mm), float(source_x_mm), *map(float, q_rad)]
        for preseat_mm, source_x_mm, q_rad in demo.CORE_CAPTURE_ROUTE_SOURCE_STATES
    ]
    capture_f8le = np.asarray(capture_rows, dtype="<f8")
    capture_route_observed = {
        "row_count": len(capture_rows),
        "f8le_shape": list(capture_f8le.shape),
        "state_bytes_sha256": hashlib.sha256(capture_f8le.tobytes()).hexdigest(),
        "source_state_sha256": demo.CORE_CAPTURE_ROUTE_SOURCE_STATE_SHA256,
        "q_roster_sha256": demo.CORE_CAPTURE_ROUTE_Q_SHA256,
        "wrist_roll_q_sha256": demo.CORE_CAPTURE_ROUTE_WRIST_ROLL_Q_SHA256,
        "wrist_roll_f8le_sha256": hashlib.sha256(
            np.asarray(capture_f8le[:, -1], dtype="<f8").tobytes()
        ).hexdigest(),
        "phase_contract_sha256": demo.CORE_CAPTURE_ROUTE_PHASE_CONTRACT_SHA256,
        "alignment_q_sha256": demo.CORE_CAPTURE_ROUTE_ALIGNMENT_Q_SHA256,
        "contract_identity_sha256": demo.CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256,
        "endpoint_fk_sha256": demo.CORE_CAPTURE_ROUTE_ENDPOINT_FK_AUTHORITY_SHA256,
        "endpoint_fk_maxima_sha256": demo.CORE_CAPTURE_ROUTE_ENDPOINT_FK_MAXIMA_AUTHORITY_SHA256,
        "dense_fk_sha256": demo.CORE_CAPTURE_ROUTE_DENSE_20X_FK_AUTHORITY_SHA256,
    }
    if capture_route_observed != EXPECTED_CAPTURE_ROUTE:
        failures.append("capture_route_identity")
    expected_preseat = [55.0 - 0.2 * index for index in range(276)]
    if not np.allclose(
        capture_f8le[:, 0], expected_preseat, rtol=0.0, atol=1.0e-12
    ):
        failures.append("capture_route_preseat_grid")
    expected_source_x = [
        0.2
        if value >= 6.4
        else 0.2 * (value - 3.2) / 3.2
        if value >= 3.2
        else 0.0
        for value in expected_preseat
    ]
    if not np.allclose(
        capture_f8le[:, 1], expected_source_x, rtol=0.0, atol=1.0e-12
    ):
        failures.append("capture_route_source_x_law")
    orientation_bound_rad = math.radians(0.1)
    endpoint_orientation_max_rad = float(
        capture_contract["endpoint_and_joint_evidence"]["endpoint_fk_maxima"][
            "maximum_orientation_error_rad"
        ]
    )
    dense_orientation_max_rad = max(
        float(record["observed"]["maximum_orientation_error_rad"])
        for record in capture_contract["dense_fk_evidence"]["phases"]
    )
    if (
        capture_contract["passed"] is not True
        or capture_contract["endpoint_guard"][
            "maximum_fk_orientation_error_rad"
        ]
        != orientation_bound_rad
        or endpoint_orientation_max_rad > orientation_bound_rad
        or dense_orientation_max_rad > orientation_bound_rad
        or capture_contract["route_law"]["exact_quaternion_required_off_seat"]
        is not False
    ):
        failures.append("capture_route_rolled_orientation_rule")

    gravity_ff_observed = {
        "identity_sha256": demo.CORE_CAPTURE_GRAVITY_BIAS_CONTRACT_IDENTITY_SHA256,
        "formula_sha256": demo.CORE_CAPTURE_GRAVITY_BIAS_FORMULA_SHA256,
        "guard_thresholds_sha256": demo.CORE_CAPTURE_GRAVITY_BIAS_GUARD_THRESHOLDS_SHA256,
        "move_action_roster_sha256": demo.CORE_CAPTURE_MOVE_ACTION_ROSTER_SHA256,
    }
    if gravity_ff_observed != EXPECTED_GRAVITY_FF:
        failures.append("capture_gravity_ff_identity")
    default_action_names = [
        action.name for action in demo._recovery_controller_actions()
    ]
    if tuple(default_action_names) != EXPECTED_DEFAULT_ACTION_NAMES:
        failures.append("default_action_roster")
    rack_exit_rejection = None
    try:
        demo._recovery_controller_actions(include_rack_exit=True)
    except ValueError as exc:
        rack_exit_rejection = str(exc)
    if rack_exit_rejection != "full rack exit is not yet a validated controller action":
        failures.append("rack_exit_not_rejected")
    if (
        static_release_contract["included_in_default_controller_actions"]
        is not False
        or static_release_contract["physical_release_action_implemented"]
        is not False
        or static_release_contract["release_ready"] is not False
    ):
        failures.append("static_release_promoted")
    startup = demo.initial_contact_report(model, data)
    if startup.get("passed") is not True or startup.get("penetration_count") != 0:
        failures.append("startup_penetration")
    startup_contact_records = sorted(
        [
            {
                "geom_a": str(model.geom(int(data.contact[index].geom[0])).name),
                "geom_b": str(model.geom(int(data.contact[index].geom[1])).name),
                "signed_distance_m": float(data.contact[index].dist),
            }
            for index in range(data.ncon)
        ],
        key=lambda record: (
            record["geom_a"],
            record["geom_b"],
            record["signed_distance_m"],
        ),
    )
    if len(startup_contact_records) != int(startup["contact_count"]):
        failures.append("startup_contact_inventory")

    arm_geom_ids: list[int] = []
    arm_by_body: dict[str, list[str]] = {
        name: [] for name in EXPECTED_ARM_NAMES_BY_BODY
    }
    for geom_id in range(model.ngeom):
        name = str(model.geom(geom_id).name)
        body_name = str(model.body(int(model.geom_bodyid[geom_id])).name)
        if body_name in EXPECTED_ARM_NAMES_BY_BODY and active_geom(model, geom_id):
            if name not in EXPECTED_ARM_NAMES_BY_BODY[body_name]:
                failures.append(f"unexpected_active_arm_geom:{body_name}:{name}")
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                failures.append(f"non_mesh_arm_geom:{name}")
            arm_geom_ids.append(geom_id)
            arm_by_body[body_name].append(name)
    arm_geom_ids.sort(key=lambda geom_id: str(model.geom(geom_id).name))
    for values in arm_by_body.values():
        values.sort()
    if len(arm_geom_ids) != EXPECTED_ARM_COUNT:
        failures.append(f"arm_geom_count:{len(arm_geom_ids)}")
    for body_name, expected_names in EXPECTED_ARM_NAMES_BY_BODY.items():
        observed_names = tuple(arm_by_body[body_name])
        if observed_names != expected_names:
            failures.append(
                f"arm_body_inventory:{body_name}:{observed_names!r}"
            )

    support_names = list(qc.CORE_DOCK_SUPPORT_PROXY_GEOM_NAMES)
    if len(support_names) != EXPECTED_SUPPORT_COUNT:
        failures.append(f"support_contract_count:{len(support_names)}")
    support_geom_ids: list[int] = []
    for name in support_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            failures.append(f"missing_support_geom:{name}")
            continue
        support_geom_ids.append(int(geom_id))
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            failures.append(f"support_not_box:{name}")

    removed_names = support_contract["support_proxy"]["removed_legacy_geom_names"]
    for name in removed_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0:
            failures.append(f"legacy_geom_present:{name}")
    for name in support_contract["support_proxy"]["removed_legacy_body_names"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            failures.append(f"legacy_body_present:{name}")

    dock_body_id = int(model.body("dock_gripper").id)
    dock_geom_ids = sorted(
        [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == dock_body_id
            and active_geom(model, geom_id)
        ],
        key=lambda geom_id: str(model.geom(geom_id).name),
    )
    if set(support_geom_ids) - set(dock_geom_ids):
        failures.append("support_not_owned_by_dock")
    dock_geom_names = [str(model.geom(geom_id).name) for geom_id in dock_geom_ids]
    expected_dock_names = sorted(
        [
            *support_names,
            *(
                "dock_gripper_qc_col_dock_stop_part_"
                f"{index:03d}__dock_stop_land"
                for index in range(EXPECTED_CORE_DOCK_STOP_PART_COUNT)
            ),
            "dock_gripper_keeper_left_lower_collision",
            "dock_gripper_keeper_left_upper_collision",
            "dock_gripper_keeper_right_lower_collision",
            "dock_gripper_keeper_right_upper_collision",
            "dock_gripper_wall_left_collision",
            "dock_gripper_wall_right_collision",
            *qc.CORE_DOCK_CAM_COLLISION_GEOM_NAMES,
        ]
    )
    if len(dock_geom_ids) != EXPECTED_DOCK_TARGET_COUNT:
        failures.append(f"dock_target_geom_count:{len(dock_geom_ids)}")
    if dock_geom_names != expected_dock_names:
        failures.append("dock_target_geom_inventory")

    arm_geom_records = [
        compiled_geom_record(model, mujoco, geom_id) for geom_id in arm_geom_ids
    ]
    dock_geom_records = [
        compiled_geom_record(model, mujoco, geom_id) for geom_id in dock_geom_ids
    ]

    # Static target AABBs are evaluated once from the actual compiled model.
    static_bounds = {
        geom_id: bounds(geom_world_vertices(model, data, mujoco, geom_id))
        for geom_id in dock_geom_ids
    }

    tangency_pairs: list[list[str]] = []
    tangency_pairs.extend(support_contract["support_proxy"]["face_tangencies"])
    tangency_pairs.extend(
        [name, "matcha_floor_collision"]
        for name in support_contract["support_proxy"][
            "floor_contact_geom_names"
        ]
    )
    chain = support_contract["support_proxy"]["declared_floor_support_chain"]
    tangency_pairs.extend([list(pair) for pair in zip(chain, chain[1:])])
    seen_pairs: set[tuple[str, str]] = set()
    tangency_records: list[dict[str, Any]] = []
    for first, second in tangency_pairs:
        key = tuple(sorted((first, second)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, first)
        second_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, second)
        if first_id < 0 or second_id < 0:
            failures.append(f"missing_tangency_geom:{first}:{second}")
            continue
        witness = np.zeros(6, dtype=np.float64)
        distance_m = float(
            mujoco.mj_geomDistance(
                model, data, first_id, second_id, 0.05, witness
            )
        )
        tangency_records.append(
            {"first": first, "second": second, "distance_m": distance_m}
        )
        if abs(distance_m) > 1.0e-7:
            failures.append(f"tangency_distance:{first}:{second}:{distance_m}")
    head_name = "dock_gripper_floor_support_head_collision"
    neck_name = "dock_gripper_floor_support_neck_collision"
    expected_tangency_keys = {
        tuple(
            sorted(
                (
                    head_name,
                    "dock_gripper_floor_support_reinforcement_collision",
                )
            )
        ),
        tuple(sorted((head_name, neck_name))),
        tuple(
            sorted(
                (
                    "dock_gripper_qc_col_dock_stop_part_000__dock_stop_land",
                    head_name,
                )
            )
        ),
    }
    for axis in ("x_min", "x_max", "z_min", "z_max"):
        post_name = f"dock_gripper_floor_support_post_{axis}_wall_collision"
        base_name = f"dock_gripper_floor_support_base_{axis}_wall_collision"
        expected_tangency_keys.add(tuple(sorted((neck_name, post_name))))
        expected_tangency_keys.add(tuple(sorted((post_name, base_name))))
        expected_tangency_keys.add(
            tuple(sorted((base_name, "matcha_floor_collision")))
        )
    if (
        len(tangency_records) != EXPECTED_SUPPORT_TOPOLOGY_TANGENCY_COUNT
        or seen_pairs != expected_tangency_keys
    ):
        failures.append("support_tangency_inventory")

    spoon_whisk: dict[str, Any] = {}
    for tool, expected in (
        ("spoon", EXPECTED_SPOON_POSE),
        ("whisk", EXPECTED_WHISK_POSE),
    ):
        declared = demo.DOCK_POSES[tool]
        if declared != expected:
            failures.append(f"declared_{tool}_pose_changed")
        body_id = int(model.body(f"dock_{tool}").id)
        observed_position = np.asarray(model.body_pos[body_id], dtype=np.float64)
        observed_quaternion = np.asarray(model.body_quat[body_id], dtype=np.float64)
        # The unchanged matcha docks intentionally retain their historical
        # ``.9g`` MJCF position serialization.  Compare against those exact
        # serialized doubles, while separately pinning the full declarations.
        serialized_position = np.asarray(
            [float(f"{value:.9g}") for value in expected[0]], dtype=np.float64
        )
        if not np.array_equal(observed_position, serialized_position):
            failures.append(f"compiled_{tool}_position_changed")
        # The compiler normalizes the source quaternion.
        expected_quaternion = np.asarray(expected[1], dtype=np.float64)
        expected_quaternion /= np.linalg.norm(expected_quaternion)
        if not np.allclose(
            observed_quaternion, expected_quaternion, rtol=0.0, atol=1.0e-11
        ):
            failures.append(f"compiled_{tool}_quaternion_changed")
        spoon_whisk[tool] = {
            "declared_position_m": list(declared[0]),
            "declared_quat_wxyz": list(declared[1]),
            "expected_serialized_position_m": serialized_position.tolist(),
            "compiled_position_m": observed_position.tolist(),
            "compiled_quat_wxyz": observed_quaternion.tolist(),
        }

    roster = qc.core_dock_release_roster()
    roster_sha = canonical_sha256(roster)
    if len(roster) != EXPECTED_RELEASE_ROWS:
        failures.append(f"release_row_count:{len(roster)}")
    if roster_sha != EXPECTED_RELEASE_SHA256:
        failures.append(f"release_roster_sha256:{roster_sha}")
    if roster_sha != qc.CORE_DOCK_RELEASE_ROSTER_CANONICAL_SHA256:
        failures.append("release_roster_public_digest_mismatch")
    if roster != support_contract["release"]["roster"]:
        failures.append("release_roster_contract_mismatch")
    if roster != [
        {"withdrawal_mm": withdrawal, "q_rad": list(q_rad)}
        for withdrawal, q_rad in qc.CORE_DOCK_RELEASE_ROSTER
    ]:
        failures.append("release_roster_constant_mismatch")
    expected_withdrawals = [0.5 * index for index in range(EXPECTED_RELEASE_ROWS)]
    if [row["withdrawal_mm"] for row in roster] != expected_withdrawals:
        failures.append("release_withdrawal_grid")
    roster_f8le = np.asarray(
        [
            [float(row["withdrawal_mm"]), *map(float, row["q_rad"])]
            for row in roster
        ],
        dtype="<f8",
    )
    slider_rows: list[dict[str, float]] = []
    for withdrawal in expected_withdrawals:
        independently_expected_q_mm = (
            0.05
            if withdrawal <= 2.0
            else min(3.0, 0.05 + 0.246875 * (withdrawal - 2.0))
        )
        observed_q_mm = float(qc.core_dock_release_slider_q_mm(withdrawal))
        if observed_q_mm != independently_expected_q_mm:
            failures.append(
                f"release_slider_q_law:{withdrawal}:{observed_q_mm}"
            )
        slider_rows.append(
            {
                "withdrawal_mm": withdrawal,
                "slider_q_mm": observed_q_mm,
            }
        )

    arm_joint_ids = [int(model.joint(name).id) for name in demo.ARM_JOINTS]
    arm_qpos_ids = np.asarray(
        [int(model.joint(name).qposadr[0]) for name in demo.ARM_JOINTS],
        dtype=int,
    )
    radius_by_geom, global_radius, radius_records = topology_radius_bounds(
        model, mujoco, arm_geom_ids, arm_joint_ids
    )

    dock_position = np.asarray(data.body("dock_gripper").xpos, dtype=np.float64).copy()
    dock_rotation = np.asarray(
        data.body("dock_gripper").xmat, dtype=np.float64
    ).reshape(3, 3).copy()
    dock_quaternion = np.asarray(EXPECTED_CORE_DOCK_QUAT_WXYZ, dtype=np.float64)
    if not np.allclose(
        dock_position,
        np.asarray(EXPECTED_CORE_DOCK_POSITION_M, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        failures.append("compiled_dock_position")
    if not math.isclose(float(np.linalg.norm(dock_quaternion)), 1.0, abs_tol=1.0e-12):
        failures.append("source_dock_quaternion_norm")
    dock_source_matrix_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(dock_source_matrix_flat, dock_quaternion)
    dock_source_rotation = dock_source_matrix_flat.reshape(3, 3)
    if not np.allclose(
        dock_rotation, dock_source_rotation, rtol=0.0, atol=3.0e-12
    ):
        failures.append("compiled_dock_quaternion")
    release_axis_world = -dock_rotation[:, 1]
    if not np.allclose(release_axis_world, (0.0, 0.0, 1.0), rtol=0.0, atol=3.0e-12):
        failures.append("release_axis_not_world_up")

    row_fk_records: list[dict[str, Any]] = []
    seated_position: np.ndarray | None = None
    seated_rotation: np.ndarray | None = None
    max_row_position_error_mm = 0.0
    max_row_orientation_error_rad = 0.0
    seated_frame_position_error_mm = math.inf
    seated_frame_orientation_error_rad = math.inf
    for row in roster:
        q_value = np.asarray(row["q_rad"], dtype=np.float64)
        data.qpos[arm_qpos_ids] = q_value
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        mating = data.site("robot_mating_face")
        mating_position = np.asarray(mating.xpos, dtype=np.float64).copy()
        mating_rotation = np.asarray(mating.xmat, dtype=np.float64).reshape(3, 3).copy()
        if seated_position is None:
            seated_position = mating_position.copy()
            seated_rotation = mating_rotation.copy()
            seated_frame_position_error_mm = 1000.0 * float(
                np.linalg.norm(mating_position - dock_position)
            )
            seated_frame_orientation_error_rad = rotation_angle(
                dock_rotation.T @ mating_rotation
            )
        assert seated_rotation is not None
        displacement_local_mm = dock_rotation.T @ (
            mating_position - seated_position
        ) * 1000.0
        expected_local_mm = np.asarray(
            (0.0, -float(row["withdrawal_mm"]), 0.0), dtype=np.float64
        )
        position_error_mm = float(
            np.linalg.norm(displacement_local_mm - expected_local_mm)
        )
        orientation_error_rad = rotation_angle(seated_rotation.T @ mating_rotation)
        max_row_position_error_mm = max(max_row_position_error_mm, position_error_mm)
        max_row_orientation_error_rad = max(
            max_row_orientation_error_rad, orientation_error_rad
        )
        row_fk_records.append(
            {
                "withdrawal_mm": float(row["withdrawal_mm"]),
                "displacement_dock_local_mm": displacement_local_mm.tolist(),
                "position_error_mm": position_error_mm,
                "orientation_error_rad": orientation_error_rad,
                "world_displacement_m": (mating_position - seated_position).tolist(),
            }
        )
    if max_row_position_error_mm > 1.0e-8:
        failures.append(f"release_fk_position:{max_row_position_error_mm}")
    if max_row_orientation_error_rad > 1.0e-8:
        failures.append(f"release_fk_orientation:{max_row_orientation_error_rad}")
    if seated_frame_position_error_mm > 1.0e-8:
        failures.append(
            f"release_seated_frame_position:{seated_frame_position_error_mm}"
        )
    if seated_frame_orientation_error_rad > 1.0e-8:
        failures.append(
            "release_seated_frame_orientation:"
            f"{seated_frame_orientation_error_rad}"
        )

    samples: list[tuple[float, np.ndarray, int, int]] = []
    for interval_index in range(len(roster) - 1):
        first = roster[interval_index]
        second = roster[interval_index + 1]
        first_q = np.asarray(first["q_rad"], dtype=np.float64)
        second_q = np.asarray(second["q_rad"], dtype=np.float64)
        for fraction_index in range(SUBSTEPS_PER_INTERVAL):
            fraction = fraction_index / SUBSTEPS_PER_INTERVAL
            withdrawal = float(first["withdrawal_mm"]) + fraction * (
                float(second["withdrawal_mm"]) - float(first["withdrawal_mm"])
            )
            q_value = first_q + fraction * (second_q - first_q)
            samples.append((withdrawal, q_value, interval_index, fraction_index))
    samples.append(
        (
            float(roster[-1]["withdrawal_mm"]),
            np.asarray(roster[-1]["q_rad"], dtype=np.float64),
            len(roster) - 2,
            SUBSTEPS_PER_INTERVAL,
        )
    )
    expected_state_count = (len(roster) - 1) * SUBSTEPS_PER_INTERVAL + 1
    if len(samples) != expected_state_count:
        failures.append(f"sample_state_count:{len(samples)}")

    sample_hasher = hashlib.sha256()
    distance_hasher = hashlib.sha256()
    sample_fk_hasher = hashlib.sha256()
    maximum_sample_path_position_error_mm = 0.0
    maximum_sample_path_orientation_error_rad = 0.0
    maximum_sample_world_axis_error_mm = 0.0
    minimum_distance_m = math.inf
    minimum_witness: dict[str, Any] | None = None
    support_geom_id_set = set(support_geom_ids)
    minimum_by_target_family_m = {
        "floor_support_proxy": math.inf,
        "core_dock_fixture": math.inf,
    }
    minimum_witness_by_target_family: dict[str, dict[str, Any] | None] = {
        "floor_support_proxy": None,
        "core_dock_fixture": None,
    }

    def make_distance_witness(
        *,
        sample_index: int,
        interval_index: int,
        fraction_index: int,
        withdrawal: float,
        arm_geom_id: int,
        target_geom_id: int,
        distance_m: float,
        arm_bounds: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, Any]:
        return {
            "sample_index": sample_index,
            "interval_index": interval_index,
            "fraction_index": fraction_index,
            "withdrawal_mm": withdrawal,
            "arm_geom": str(model.geom(arm_geom_id).name),
            "target_geom": str(model.geom(target_geom_id).name),
            "distance_mm": 1000.0 * distance_m,
            "arm_aabb_world_m": [
                arm_bounds[0].tolist(),
                arm_bounds[1].tolist(),
            ],
            "target_aabb_world_m": [
                static_bounds[target_geom_id][0].tolist(),
                static_bounds[target_geom_id][1].tolist(),
            ],
        }

    assert seated_position is not None
    assert seated_rotation is not None
    for sample_index, (withdrawal, q_value, interval_index, fraction_index) in enumerate(samples):
        sample_hasher.update(
            struct.pack("<IId", sample_index, interval_index, withdrawal)
        )
        sample_hasher.update(np.asarray(q_value, dtype="<f8").tobytes())
        data.qpos[arm_qpos_ids] = q_value
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        sample_mating = data.site("robot_mating_face")
        sample_position = np.asarray(
            sample_mating.xpos, dtype=np.float64
        ).copy()
        sample_rotation = np.asarray(
            sample_mating.xmat, dtype=np.float64
        ).reshape(3, 3).copy()
        sample_local_mm = dock_rotation.T @ (
            sample_position - seated_position
        ) * 1000.0
        sample_expected_local_mm = np.asarray(
            (0.0, -withdrawal, 0.0), dtype=np.float64
        )
        sample_path_error_mm = float(
            np.linalg.norm(sample_local_mm - sample_expected_local_mm)
        )
        sample_orientation_error_rad = rotation_angle(
            seated_rotation.T @ sample_rotation
        )
        sample_expected_world_m = (
            0.001 * withdrawal * release_axis_world
        )
        sample_world_axis_error_mm = 1000.0 * float(
            np.linalg.norm(
                sample_position - seated_position - sample_expected_world_m
            )
        )
        maximum_sample_path_position_error_mm = max(
            maximum_sample_path_position_error_mm, sample_path_error_mm
        )
        maximum_sample_path_orientation_error_rad = max(
            maximum_sample_path_orientation_error_rad,
            sample_orientation_error_rad,
        )
        maximum_sample_world_axis_error_mm = max(
            maximum_sample_world_axis_error_mm, sample_world_axis_error_mm
        )
        sample_fk_hasher.update(
            struct.pack(
                "<Idddd",
                sample_index,
                sample_path_error_mm,
                sample_orientation_error_rad,
                sample_world_axis_error_mm,
                withdrawal,
            )
        )
        for arm_geom_id in arm_geom_ids:
            arm_bounds = bounds(
                geom_world_vertices(model, data, mujoco, arm_geom_id)
            )
            for target_geom_id in dock_geom_ids:
                distance_m = aabb_distance(
                    arm_bounds, static_bounds[target_geom_id]
                )
                distance_hasher.update(
                    struct.pack(
                        "<IIIId",
                        sample_index,
                        arm_geom_id,
                        target_geom_id,
                        fraction_index,
                        distance_m,
                    )
                )
                target_family = (
                    "floor_support_proxy"
                    if target_geom_id in support_geom_id_set
                    else "core_dock_fixture"
                )
                needs_global_witness = distance_m < minimum_distance_m
                needs_family_witness = (
                    distance_m < minimum_by_target_family_m[target_family]
                )
                if needs_global_witness or needs_family_witness:
                    witness_record = make_distance_witness(
                        sample_index=sample_index,
                        interval_index=interval_index,
                        fraction_index=fraction_index,
                        withdrawal=withdrawal,
                        arm_geom_id=arm_geom_id,
                        target_geom_id=target_geom_id,
                        distance_m=distance_m,
                        arm_bounds=arm_bounds,
                    )
                if needs_global_witness:
                    minimum_distance_m = distance_m
                    minimum_witness = witness_record
                if needs_family_witness:
                    minimum_by_target_family_m[target_family] = distance_m
                    minimum_witness_by_target_family[target_family] = witness_record

    if (
        maximum_sample_path_position_error_mm
        > SUBSAMPLE_PATH_POSITION_TOLERANCE_MM
    ):
        failures.append(
            "release_subsample_path_position:"
            f"{maximum_sample_path_position_error_mm}"
        )
    if (
        maximum_sample_path_orientation_error_rad
        > SUBSAMPLE_PATH_ORIENTATION_TOLERANCE_RAD
    ):
        failures.append(
            "release_subsample_path_orientation:"
            f"{maximum_sample_path_orientation_error_rad}"
        )

    maximum_substep_sum_abs_dq = 0.0
    maximum_topology_motion_m = 0.0
    motion_witness: dict[str, Any] | None = None
    for interval_index in range(len(roster) - 1):
        delta = (
            np.asarray(roster[interval_index + 1]["q_rad"], dtype=np.float64)
            - np.asarray(roster[interval_index]["q_rad"], dtype=np.float64)
        ) / SUBSTEPS_PER_INTERVAL
        maximum_substep_sum_abs_dq = max(
            maximum_substep_sum_abs_dq, float(np.sum(np.abs(delta)))
        )
        for geom_id in arm_geom_ids:
            radius_map = radius_by_geom[geom_id]
            bound_m = math.fsum(
                radius_map.get(joint_id, 0.0) * abs(float(delta[index]))
                for index, joint_id in enumerate(arm_joint_ids)
            )
            if bound_m > maximum_topology_motion_m:
                maximum_topology_motion_m = bound_m
                motion_witness = {
                    "interval_index": interval_index,
                    "arm_geom": str(model.geom(geom_id).name),
                    "substep_delta_q_rad": delta.tolist(),
                    "sum_abs_delta_q_rad": float(np.sum(np.abs(delta))),
                    "motion_bound_mm": 1000.0 * bound_m,
                }
    coarse_global_motion_m = global_radius * maximum_substep_sum_abs_dq
    if maximum_topology_motion_m > coarse_global_motion_m + 1.0e-15:
        failures.append("topology_motion_exceeds_global_bound")
    continuous_lower_bound_mm = 1000.0 * (
        minimum_distance_m - maximum_topology_motion_m
    )
    clearance_passed = bool(
        math.isfinite(continuous_lower_bound_mm)
        and continuous_lower_bound_mm >= REQUIRED_CONTINUOUS_CLEARANCE_MM
    )
    if not clearance_passed:
        failures.append(f"continuous_clearance:{continuous_lower_bound_mm}")
    target_family_clearance: dict[str, dict[str, Any]] = {}
    for family, family_minimum_m in minimum_by_target_family_m.items():
        family_continuous_lower_bound_mm = 1000.0 * (
            family_minimum_m - maximum_topology_motion_m
        )
        family_passed = bool(
            math.isfinite(family_continuous_lower_bound_mm)
            and family_continuous_lower_bound_mm
            >= REQUIRED_CONTINUOUS_CLEARANCE_MM
        )
        target_geom_count = (
            len(support_geom_ids)
            if family == "floor_support_proxy"
            else len(dock_geom_ids) - len(support_geom_ids)
        )
        target_family_clearance[family] = {
            "target_geom_count": target_geom_count,
            "distance_evaluation_count": (
                len(samples) * len(arm_geom_ids) * target_geom_count
            ),
            "minimum_sampled_outer_aabb_lower_bound_mm": (
                1000.0 * family_minimum_m
            ),
            "minimum_witness": minimum_witness_by_target_family[family],
            "continuous_clearance_lower_bound_mm": (
                family_continuous_lower_bound_mm
            ),
            "required_clearance_mm": REQUIRED_CONTINUOUS_CLEARANCE_MM,
            "passed": family_passed,
        }
        if not family_passed:
            failures.append(
                f"continuous_clearance_{family}:"
                f"{family_continuous_lower_bound_mm}"
            )

    xml_sha = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()
    compiled_sha = demo.compiled_model_xml_equivalent_sha256(model)
    fresh = mujoco.MjData(model)
    demo.initialize(model, fresh)
    active_sha = demo.initialized_active_collision_geometry_sha256(model, fresh)
    observed_model_identities = {
        "assembled_xml_sha256": xml_sha,
        "compiled_model_xml_equivalent_sha256": compiled_sha,
        "initialized_active_collision_geometry_sha256": active_sha,
        "binary_asset_records_canonical_sha256": canonical_sha256(asset_records),
    }
    if observed_model_identities != EXPECTED_MODEL_IDENTITIES:
        failures.append("compiled_model_identity")
    source_files = support_contract["source_binding"]["files"]
    source_file_check = []
    for record in source_files:
        path = repo / str(record["path"])
        source_file_check.append(
            {
                **record,
                "observed_bytes": path.stat().st_size,
                "observed_sha256": sha256_file(path),
                "matches": bool(
                    path.stat().st_size == int(record["bytes"])
                    and sha256_file(path) == record["sha256"]
                ),
            }
        )
        if source_file_check[-1]["matches"] is not True:
            failures.append(f"source_file:{record['role']}")
    consumed_files_after = [
        source_file_record(repo, role, relative_path)
        for role, relative_path in consumed_file_specs
    ]
    if consumed_files_after != consumed_files_before:
        failures.append("consumed_source_files_changed_during_audit")

    script_path = repo / VALIDATOR_RELATIVE_PATH
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "actual_compiled_mesh_outer_AABB_to_actual_compiled_static_dock_"
            "geom_outer_AABB_with_topology_joint_motion_bound"
        ),
        "source_binding": {
            "runtime_contract_schema": support_contract["schema_version"],
            "runtime_contract_canonical_sha256": canonical_sha256(
                support_contract
            ),
            "release_roster_canonical_sha256": roster_sha,
            "files": source_file_check,
            "directly_consumed_runtime_files": consumed_files_after,
            "directly_consumed_runtime_files_canonical_sha256": (
                canonical_sha256(consumed_files_after)
            ),
            "audit_script": {
                "path": VALIDATOR_RELATIVE_PATH,
                "bytes": script_path.stat().st_size,
                "sha256": sha256_file(script_path),
            },
            "qc_source_files_canonical_sha256": canonical_sha256(
                source_binding["files"]
            ),
            "triple_contract_locations": source_binding[
                "triple_contract_locations"
            ],
            "triple_contract_locations_canonical_sha256": canonical_sha256(
                source_binding["triple_contract_locations"]
            ),
            "triple_contract_equal": source_binding["triple_contract_equal"],
            "source_contract_canonical_sha256": source_binding[
                "canonical_sha256"
            ],
        },
        "model_binding": {
            "assembled_xml_sha256": xml_sha,
            "binary_asset_count": len(asset_records),
            "binary_asset_records": asset_records,
            "binary_asset_records_canonical_sha256": canonical_sha256(
                asset_records
            ),
            "compiled_model_xml_equivalent_sha256": compiled_sha,
            "initialized_active_collision_geometry_sha256": active_sha,
            "physics_timestep_s": float(model.opt.timestep),
            "expected_counts": EXPECTED_MODEL_COUNTS,
            **observed_model_counts,
            "nv": int(model.nv),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "mujoco_version": str(mujoco.__version__),
        },
        "rolled_frame": {
            "position_m": list(qc.CORE_DOCK_WORLD_POS_M),
            "quat_wxyz": list(qc.CORE_DOCK_WORLD_QUAT_WXYZ),
            "tool_view_roll_deg": support_contract["runtime_frame"][
                "tool_view_roll_deg"
            ],
            "dock_local_axes_world_columns": support_contract["runtime_frame"][
                "dock_local_axes_world_columns"
            ],
            "release_axis": "dock_local_negative_y",
            "release_axis_world": support_contract["runtime_frame"][
                "release_axis_world"
            ],
        },
        "support_proxy": {
            "frame": "dock_gripper",
            "components": support_components,
            "components_canonical_sha256": canonical_sha256(
                support_components
            ),
            "component_count": len(support_components),
            "pairwise_positive_overlap_count": positive_overlap_count,
            "source_volume_mm3": float(qc.CORE_DOCK_SUPPORT_SOURCE_VOLUME_MM3),
            "analytic_box_union_volume_mm3": analytic_volume_mm3,
            "source_missing_volume_mm3": source_missing_volume_mm3,
            "analytic_excess_volume_mm3": proxy_excess_volume_mm3,
            "covers_positive_source_primitives": support_contract[
                "support_proxy"
            ]["covers_positive_source_primitives"],
            "passage_witness_m": list(EXPECTED_SUPPORT_PASSAGE_WITNESS_M),
            "passage_witness_inside_proxy": passage_witness_inside,
            "filled_hole_witnesses_m": hole_witnesses,
            "filled_hole_witnesses_inside_proxy": hole_witness_inside,
            "conservative_hole_fill": support_contract["support_proxy"][
                "conservative_hole_fill"
            ],
            "removed_legacy_geom_names": support_contract["support_proxy"][
                "removed_legacy_geom_names"
            ],
            "removed_legacy_body_names": support_contract["support_proxy"][
                "removed_legacy_body_names"
            ],
            "exact_source_brep_boundary_authority": False,
            "physical_geometry_authority": False,
            "mass_authority": False,
            "fastener_authority": False,
            "substrate_authority": False,
            "load_path_authority": False,
            "tolerance_authority": False,
        },
        "capture_route": {
            **capture_route_observed,
            "preseat_bounds_mm": [55.0, 0.0],
            "preseat_step_mm": -0.2,
            "source_x_breakpoints_mm": [[55.0, 0.2], [6.4, 0.2], [3.2, 0.0], [0.0, 0.0]],
            "phase_row_ranges_inclusive": {
                name: list(value)
                for name, value in demo.CORE_CAPTURE_ROUTE_PHASE_ROW_RANGES.items()
            },
            "phase_timing_s_duration_timeout": {
                name: list(value)
                for name, value in demo.CORE_CAPTURE_ROUTE_PHASE_TIMING_S.items()
            },
            "orientation_rule": (
                "fixed_rolled_frame_best_attainable_5DOF_error_at_most_0.1deg"
            ),
            "orientation_bound_rad": orientation_bound_rad,
            "maximum_endpoint_orientation_error_rad": endpoint_orientation_max_rad,
            "maximum_dense_orientation_error_rad": dense_orientation_max_rad,
            "exact_quaternion_required_off_seat": False,
            "passed": capture_contract["passed"],
        },
        "gravity_feedforward": {
            **gravity_ff_observed,
            "capture_route_identity_sha256": demo.CORE_CAPTURE_ROUTE_CONTRACT_IDENTITY_SHA256,
            "static_only": True,
            "dynamics_authority": False,
            "passed": True,
        },
        "default_actions": {
            "names": default_action_names,
            "static_release_continuation_included": False,
            "rack_exit_flag_rejected": True,
            "rack_exit_rejection": rack_exit_rejection,
        },
        "release_route": {
            "row_count": len(roster),
            "step_mm": 0.5,
            "withdrawal_bounds_mm": [0.0, 15.0],
            "canonical_sha256": roster_sha,
            "roster": roster,
            "f8le_shape": list(roster_f8le.shape),
            "f8le_sha256": hashlib.sha256(roster_f8le.tobytes()).hexdigest(),
            "q_roster_canonical_sha256": static_release_contract[
                "q_roster_canonical_sha256"
            ],
            "static_release_contract_sha256": demo.CORE_DOCK_STATIC_RELEASE_CONTRACT_SHA256,
            "included_in_default_controller_actions": False,
            "physical_release_action_implemented": False,
            "slider_q_law": "0.05 if s<=2 else min(3,0.05+0.246875*(s-2))",
            "slider_rows": slider_rows,
            "slider_rows_canonical_sha256": canonical_sha256(slider_rows),
            "seated_frame_position_error_mm": seated_frame_position_error_mm,
            "seated_frame_orientation_error_rad": (
                seated_frame_orientation_error_rad
            ),
            "maximum_row_fk_position_error_mm": max_row_position_error_mm,
            "maximum_row_fk_orientation_error_rad": max_row_orientation_error_rad,
            "maximum_subsample_path_position_error_mm": (
                maximum_sample_path_position_error_mm
            ),
            "maximum_subsample_world_axis_error_mm": (
                maximum_sample_world_axis_error_mm
            ),
            "subsample_path_position_tolerance_mm": (
                SUBSAMPLE_PATH_POSITION_TOLERANCE_MM
            ),
            "maximum_subsample_path_orientation_error_rad": (
                maximum_sample_path_orientation_error_rad
            ),
            "subsample_path_orientation_tolerance_rad": (
                SUBSAMPLE_PATH_ORIENTATION_TOLERANCE_RAD
            ),
            "subsample_fk_stream_sha256": sample_fk_hasher.hexdigest(),
            "row_fk_records_sha256": canonical_sha256(row_fk_records),
            "physical_release_authority": False,
        },
        "inventory": {
            "arm_geom_count": len(arm_geom_ids),
            "arm_by_body": arm_by_body,
            "arm_geom_names": [str(model.geom(value).name) for value in arm_geom_ids],
            "arm_geom_names_sha256": canonical_sha256(
                [str(model.geom(value).name) for value in arm_geom_ids]
            ),
            "arm_compiled_geom_records_sha256": canonical_sha256(
                arm_geom_records
            ),
            "arm_compiled_geom_records": arm_geom_records,
            "support_geom_count": len(support_geom_ids),
            "support_geom_names": support_names,
            "support_geom_names_sha256": canonical_sha256(support_names),
            "dock_target_geom_count": len(dock_geom_ids),
            "dock_target_geom_names": dock_geom_names,
            "dock_target_geom_names_sha256": canonical_sha256(dock_geom_names),
            "dock_target_compiled_geom_records_sha256": canonical_sha256(
                dock_geom_records
            ),
            "dock_target_compiled_geom_records": dock_geom_records,
        },
        "sampling": {
            "interval_count": len(roster) - 1,
            "joint_linear_substeps_per_interval": SUBSTEPS_PER_INTERVAL,
            "unique_state_count": len(samples),
            "expected_unique_state_count": expected_state_count,
            "distance_evaluation_count": (
                len(samples) * len(arm_geom_ids) * len(dock_geom_ids)
            ),
            "sample_q_sha256": sample_hasher.hexdigest(),
            "distance_stream_sha256": distance_hasher.hexdigest(),
        },
        "continuous_clearance": {
            "minimum_sampled_outer_aabb_lower_bound_mm": (
                1000.0 * minimum_distance_m
            ),
            "minimum_witness": minimum_witness,
            "target_family_clearance": target_family_clearance,
            "topology_global_chain_radius_bound_m": global_radius,
            "topology_radius_records": radius_records,
            "topology_radius_records_sha256": canonical_sha256(radius_records),
            "maximum_substep_sum_abs_dq_rad": maximum_substep_sum_abs_dq,
            "maximum_pair_specific_topology_motion_bound_mm": (
                1000.0 * maximum_topology_motion_m
            ),
            "coarse_global_motion_bound_mm": 1000.0 * coarse_global_motion_m,
            "motion_witness": motion_witness,
            "continuous_clearance_lower_bound_mm": continuous_lower_bound_mm,
            "required_clearance_mm": REQUIRED_CONTINUOUS_CLEARANCE_MM,
            "passed": clearance_passed,
        },
        "startup": {
            **startup,
            "contact_records": startup_contact_records,
            "contact_records_sha256": canonical_sha256(
                startup_contact_records
            ),
        },
        "support_topology": {
            "tangency_count": len(tangency_records),
            "tangencies": tangency_records,
            "passed": not any("tangency" in item for item in failures),
        },
        "unchanged_matcha_docks": spoon_whisk,
        "authority_scope": {
            "continuous_geometric_clearance_authority": True,
            "default_action_roster_identity_authority": True,
            "continuous_geometric_clearance_scope": (
                "only_the_exact_hash_bound_31_row_joint_linear_release_"
                "continuation_and_compiled_collision_inventory"
            ),
            "exact_source_brep_boundary_authority": False,
            "material_authority": False,
            "mass_authority": False,
            "fastener_authority": False,
            "substrate_authority": False,
            "load_path_authority": False,
            "tolerance_authority": False,
            "contact_force_authority": False,
            "friction_coefficient_authority": False,
            "capture_dynamics_authority": False,
            "contact_dynamics_authority": False,
            "physical_lock_authority": False,
            "physical_release_authority": False,
            "default_runtime_action_authority": False,
            "release_ready": False,
            "runtime_contract_required_false_fields": (
                required_false_contract_fields
            ),
            "runtime_contract_blockers": list(observed_runtime_blockers),
            "checkpoint_blockers": list(CHECKPOINT_BLOCKERS),
        },
        "failures": failures,
        "geometry_passed": not failures,
        "passed": not failures,
        "release_ready": False,
        "canonical_sha256_without_this_field": None,
    }
    evidence["canonical_sha256_without_this_field"] = canonical_sha256(evidence)
    return evidence


def rolled_core_dock_runtime_report_errors(
    report: object,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    recompute_runtime: bool = False,
) -> list[str]:
    """Return fail-closed semantic errors for a checkpoint report.

    The default path is intentionally pure and fast enough for coherent
    adversarial mutation tests.  ``recompute_runtime=True`` additionally
    rebuilds the compiled model and all 379,260 arm/target/state distances.
    """

    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report:not_object"]
    repo = Path(repository_root).resolve()

    def value(path: str) -> Any:
        current: Any = report
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                errors.append(f"missing:{path}")
                return None
            current = current[part]
        return current

    def expect(path: str, expected: object) -> None:
        observed = value(path)
        if observed != expected:
            errors.append(f"value:{path}")

    expected_top_keys = {
        "schema_version",
        "method",
        "source_binding",
        "model_binding",
        "rolled_frame",
        "support_proxy",
        "capture_route",
        "gravity_feedforward",
        "default_actions",
        "release_route",
        "inventory",
        "sampling",
        "continuous_clearance",
        "startup",
        "support_topology",
        "unchanged_matcha_docks",
        "authority_scope",
        "failures",
        "geometry_passed",
        "passed",
        "release_ready",
        "canonical_sha256_without_this_field",
    }
    if set(report) != expected_top_keys:
        errors.append("schema:top_level_keys")
    expect("schema_version", SCHEMA_VERSION)
    expect("failures", [])
    expect("geometry_passed", True)
    expect("passed", True)
    expect("release_ready", False)
    claimed_seal = report.get("canonical_sha256_without_this_field")
    seal_preimage = dict(report)
    seal_preimage["canonical_sha256_without_this_field"] = None
    if claimed_seal != canonical_sha256(seal_preimage):
        errors.append("integrity:canonical_sha256")

    direct_records = value("source_binding.directly_consumed_runtime_files")
    expected_direct_records = [
        {
            "role": role,
            "path": relative_path,
            "bytes": byte_count,
            "sha256": digest,
        }
        for role, relative_path, byte_count, digest in EXPECTED_DIRECT_SOURCE_RECORDS
    ]
    if direct_records != expected_direct_records:
        errors.append("source:direct_record_identity")
    if isinstance(direct_records, list):
        if value("source_binding.directly_consumed_runtime_files_canonical_sha256") != canonical_sha256(direct_records):
            errors.append("source:direct_record_digest")
        for record in direct_records:
            if not isinstance(record, dict):
                errors.append("source:direct_record_type")
                continue
            relative_path = record.get("path")
            if not isinstance(relative_path, str):
                errors.append("source:direct_record_path")
                continue
            path = repo / relative_path
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                errors.append(f"source:file_identity:{relative_path}")

    expect(
        "source_binding.qc_source_files_canonical_sha256",
        EXPECTED_QC_SOURCE_FILES_CANONICAL_SHA256,
    )
    expect(
        "source_binding.triple_contract_locations_canonical_sha256",
        EXPECTED_TRIPLE_CONTRACT_LOCATIONS_CANONICAL_SHA256,
    )
    triple_locations = value("source_binding.triple_contract_locations")
    if isinstance(triple_locations, list) and canonical_sha256(triple_locations) != EXPECTED_TRIPLE_CONTRACT_LOCATIONS_CANONICAL_SHA256:
        errors.append("source:triple_location_digest")
    expect("source_binding.triple_contract_equal", True)
    expect(
        "source_binding.source_contract_canonical_sha256",
        EXPECTED_SOURCE_CONTRACT_SHA256,
    )
    expect(
        "source_binding.runtime_contract_canonical_sha256",
        EXPECTED_RUNTIME_SUPPORT_CONTRACT_SHA256,
    )
    expect("source_binding.release_roster_canonical_sha256", EXPECTED_RELEASE_SHA256)
    audit_script = value("source_binding.audit_script")
    validator_path = repo / VALIDATOR_RELATIVE_PATH
    expected_validator_record = {
        "path": VALIDATOR_RELATIVE_PATH,
        "bytes": validator_path.stat().st_size if validator_path.is_file() else -1,
        "sha256": sha256_file(validator_path) if validator_path.is_file() else "",
    }
    if audit_script != expected_validator_record:
        errors.append("source:validator_identity")

    for key, expected in EXPECTED_MODEL_IDENTITIES.items():
        expect(f"model_binding.{key}", expected)
    expect("model_binding.binary_asset_count", 13)
    expect("model_binding.expected_counts", EXPECTED_MODEL_COUNTS)
    for key, expected in EXPECTED_MODEL_COUNTS.items():
        expect(f"model_binding.{key}", expected)
    expect("model_binding.nv", 34)
    expect("model_binding.physics_timestep_s", 0.00025)
    assets = value("model_binding.binary_asset_records")
    if isinstance(assets, list) and canonical_sha256(assets) != EXPECTED_MODEL_IDENTITIES["binary_asset_records_canonical_sha256"]:
        errors.append("model:binary_asset_record_digest")

    expect("rolled_frame.position_m", list(EXPECTED_CORE_DOCK_POSITION_M))
    expect("rolled_frame.quat_wxyz", list(EXPECTED_CORE_DOCK_QUAT_WXYZ))
    expect("rolled_frame.tool_view_roll_deg", EXPECTED_ROLL_DEG)
    expect("rolled_frame.release_axis", "dock_local_negative_y")
    expect("rolled_frame.release_axis_world", [0.0, 0.0, 1.0])

    components = value("support_proxy.components")
    if not isinstance(components, list):
        components = []
    expect("support_proxy.component_count", EXPECTED_SUPPORT_COUNT)
    if canonical_sha256(components) != EXPECTED_SUPPORT_COMPONENTS_CANONICAL_SHA256:
        errors.append("support:component_digest")
    if value("support_proxy.components_canonical_sha256") != canonical_sha256(components):
        errors.append("support:component_cross_digest")
    analytic_volume_mm3 = math.fsum(
        math.prod(float(axis[1]) - float(axis[0]) for axis in record.get("bounds_m", ()))
        * 1.0e9
        for record in components
        if isinstance(record, dict)
    )
    if not math.isclose(analytic_volume_mm3, EXPECTED_SUPPORT_PROXY_VOLUME_MM3, rel_tol=0.0, abs_tol=1.0e-9):
        errors.append("support:analytic_volume")
    expect("support_proxy.source_volume_mm3", EXPECTED_SUPPORT_SOURCE_VOLUME_MM3)
    expect("support_proxy.analytic_box_union_volume_mm3", EXPECTED_SUPPORT_PROXY_VOLUME_MM3)
    expect("support_proxy.source_missing_volume_mm3", EXPECTED_SUPPORT_SOURCE_MISSING_VOLUME_MM3)
    expect("support_proxy.analytic_excess_volume_mm3", EXPECTED_SUPPORT_PROXY_EXCESS_VOLUME_MM3)
    expect("support_proxy.pairwise_positive_overlap_count", 0)
    expect("support_proxy.covers_positive_source_primitives", True)
    expect("support_proxy.passage_witness_m", list(EXPECTED_SUPPORT_PASSAGE_WITNESS_M))
    expect("support_proxy.passage_witness_inside_proxy", False)
    expect("support_proxy.filled_hole_witnesses_inside_proxy", [True] * 6)
    expect("support_proxy.conservative_hole_fill", True)
    expect(
        "support_proxy.removed_legacy_geom_names",
        ["dock_gripper_support_collision", "dock_gripper_support_anchor_collision"],
    )
    expect("support_proxy.removed_legacy_body_names", ["dock_gripper_support"])
    for field in (
        "exact_source_brep_boundary_authority",
        "physical_geometry_authority",
        "mass_authority",
        "fastener_authority",
        "substrate_authority",
        "load_path_authority",
        "tolerance_authority",
    ):
        expect(f"support_proxy.{field}", False)

    for key, expected in EXPECTED_CAPTURE_ROUTE.items():
        expect(f"capture_route.{key}", expected)
    expect("capture_route.preseat_bounds_mm", [55.0, 0.0])
    expect("capture_route.preseat_step_mm", -0.2)
    expect(
        "capture_route.source_x_breakpoints_mm",
        [[55.0, 0.2], [6.4, 0.2], [3.2, 0.0], [0.0, 0.0]],
    )
    expect("capture_route.orientation_bound_rad", math.radians(0.1))
    expect("capture_route.exact_quaternion_required_off_seat", False)
    expect("capture_route.passed", True)
    for path in (
        "capture_route.maximum_endpoint_orientation_error_rad",
        "capture_route.maximum_dense_orientation_error_rad",
    ):
        observed = value(path)
        if not isinstance(observed, (int, float)) or observed > math.radians(0.1):
            errors.append(f"capture:orientation_bound:{path}")
    for key, expected in EXPECTED_GRAVITY_FF.items():
        expect(f"gravity_feedforward.{key}", expected)
    expect(
        "gravity_feedforward.capture_route_identity_sha256",
        EXPECTED_CAPTURE_ROUTE["contract_identity_sha256"],
    )
    expect("gravity_feedforward.static_only", True)
    expect("gravity_feedforward.dynamics_authority", False)
    expect("gravity_feedforward.passed", True)

    expect("default_actions.names", list(EXPECTED_DEFAULT_ACTION_NAMES))
    expect("default_actions.static_release_continuation_included", False)
    expect("default_actions.rack_exit_flag_rejected", True)
    expect(
        "default_actions.rack_exit_rejection",
        "full rack exit is not yet a validated controller action",
    )

    roster = value("release_route.roster")
    if not isinstance(roster, list):
        roster = []
    expect("release_route.row_count", EXPECTED_RELEASE_ROWS)
    expect("release_route.step_mm", 0.5)
    expect("release_route.withdrawal_bounds_mm", [0.0, 15.0])
    if canonical_sha256(roster) != EXPECTED_RELEASE_SHA256:
        errors.append("release:roster_digest")
    expect("release_route.canonical_sha256", EXPECTED_RELEASE_SHA256)
    try:
        roster_f8le = np.asarray(
            [[row["withdrawal_mm"], *row["q_rad"]] for row in roster], dtype="<f8"
        )
    except (KeyError, TypeError, ValueError):
        roster_f8le = np.empty((0, 0), dtype="<f8")
        errors.append("release:roster_shape")
    if list(roster_f8le.shape) != [31, 6]:
        errors.append("release:f8le_shape")
    if hashlib.sha256(roster_f8le.tobytes()).hexdigest() != EXPECTED_RELEASE_F8LE_SHA256:
        errors.append("release:f8le_digest")
    expect("release_route.f8le_shape", [31, 6])
    expect("release_route.f8le_sha256", EXPECTED_RELEASE_F8LE_SHA256)
    expect("release_route.q_roster_canonical_sha256", EXPECTED_RELEASE_Q_ROSTER_SHA256)
    expect("release_route.static_release_contract_sha256", EXPECTED_STATIC_RELEASE_CONTRACT_SHA256)
    expect("release_route.included_in_default_controller_actions", False)
    expect("release_route.physical_release_action_implemented", False)
    expect("release_route.physical_release_authority", False)
    slider_rows = value("release_route.slider_rows")
    if not isinstance(slider_rows, list):
        slider_rows = []
    expected_slider_rows = []
    for index in range(31):
        withdrawal = 0.5 * index
        q_mm = 0.05 if withdrawal <= 2.0 else min(3.0, 0.05 + 0.246875 * (withdrawal - 2.0))
        expected_slider_rows.append({"withdrawal_mm": withdrawal, "slider_q_mm": q_mm})
    if slider_rows != expected_slider_rows:
        errors.append("release:slider_law")
    expect("release_route.slider_rows_canonical_sha256", EXPECTED_RELEASE_SLIDER_ROWS_SHA256)
    expect("release_route.row_fk_records_sha256", EXPECTED_AUDIT_STREAMS["row_fk_records_sha256"])
    expect("release_route.subsample_fk_stream_sha256", EXPECTED_AUDIT_STREAMS["subsample_fk_stream_sha256"])
    if value("release_route.maximum_row_fk_position_error_mm") not in (None,) and float(value("release_route.maximum_row_fk_position_error_mm")) > SUBSAMPLE_PATH_POSITION_TOLERANCE_MM:
        errors.append("release:row_fk_position")
    if value("release_route.maximum_row_fk_orientation_error_rad") not in (None,) and float(value("release_route.maximum_row_fk_orientation_error_rad")) > SUBSAMPLE_PATH_ORIENTATION_TOLERANCE_RAD:
        errors.append("release:row_fk_orientation")

    inventory = value("inventory")
    if isinstance(inventory, dict):
        expect("inventory.arm_geom_count", EXPECTED_ARM_COUNT)
        expect("inventory.support_geom_count", EXPECTED_SUPPORT_COUNT)
        expect("inventory.dock_target_geom_count", EXPECTED_DOCK_TARGET_COUNT)
        for field in (
            "arm_geom_names",
            "arm_compiled_geom_records",
            "support_geom_names",
            "dock_target_geom_names",
            "dock_target_compiled_geom_records",
        ):
            records = inventory.get(field)
            digest_field = {
                "arm_geom_names": "arm_geom_names_sha256",
                "arm_compiled_geom_records": "arm_compiled_geom_records_sha256",
                "support_geom_names": "support_geom_names_sha256",
                "dock_target_geom_names": "dock_target_geom_names_sha256",
                "dock_target_compiled_geom_records": "dock_target_compiled_geom_records_sha256",
            }[field]
            if not isinstance(records, list) or canonical_sha256(records) != EXPECTED_AUDIT_STREAMS[digest_field]:
                errors.append(f"inventory:{field}")
            if inventory.get(digest_field) != EXPECTED_AUDIT_STREAMS[digest_field]:
                errors.append(f"inventory:{digest_field}")
        dock_names = inventory.get("dock_target_geom_names", [])
        if any(
            name in dock_names
            for name in (
                "dock_gripper_support_collision",
                "dock_gripper_support_anchor_collision",
            )
        ):
            errors.append("inventory:legacy_support_resurrected")

    expect("sampling.interval_count", 30)
    expect("sampling.joint_linear_substeps_per_interval", SUBSTEPS_PER_INTERVAL)
    expect("sampling.unique_state_count", 301)
    expect("sampling.expected_unique_state_count", 301)
    expect("sampling.distance_evaluation_count", 379_260)
    expect("sampling.sample_q_sha256", EXPECTED_AUDIT_STREAMS["sample_q_sha256"])
    expect("sampling.distance_stream_sha256", EXPECTED_AUDIT_STREAMS["distance_stream_sha256"])

    clearance = value("continuous_clearance")
    if isinstance(clearance, dict):
        expect(
            "continuous_clearance.minimum_sampled_outer_aabb_lower_bound_mm",
            EXPECTED_MINIMUM_SAMPLED_CLEARANCE_MM,
        )
        expect(
            "continuous_clearance.maximum_pair_specific_topology_motion_bound_mm",
            EXPECTED_MAXIMUM_TOPOLOGY_MOTION_BOUND_MM,
        )
        expect(
            "continuous_clearance.continuous_clearance_lower_bound_mm",
            EXPECTED_CONTINUOUS_CLEARANCE_LOWER_BOUND_MM,
        )
        expect("continuous_clearance.required_clearance_mm", REQUIRED_CONTINUOUS_CLEARANCE_MM)
        expect("continuous_clearance.passed", True)
        if not math.isclose(
            float(clearance.get("minimum_sampled_outer_aabb_lower_bound_mm", math.nan))
            - float(clearance.get("maximum_pair_specific_topology_motion_bound_mm", math.nan)),
            float(clearance.get("continuous_clearance_lower_bound_mm", math.nan)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            errors.append("clearance:motion_bound_algebra")
        radius_records = clearance.get("topology_radius_records")
        if not isinstance(radius_records, list) or canonical_sha256(radius_records) != EXPECTED_AUDIT_STREAMS["topology_radius_records_sha256"]:
            errors.append("clearance:topology_radius_records")
        if clearance.get("topology_radius_records_sha256") != EXPECTED_AUDIT_STREAMS["topology_radius_records_sha256"]:
            errors.append("clearance:topology_radius_digest")
        families = clearance.get("target_family_clearance")
        if not isinstance(families, dict) or set(families) != {"floor_support_proxy", "core_dock_fixture"}:
            errors.append("clearance:target_families")
        else:
            support_family = families["floor_support_proxy"]
            fixture_family = families["core_dock_fixture"]
            if support_family.get("target_geom_count") != 11 or fixture_family.get("target_geom_count") != 79:
                errors.append("clearance:target_family_counts")
            if support_family.get("minimum_sampled_outer_aabb_lower_bound_mm") != EXPECTED_SUPPORT_MINIMUM_SAMPLED_CLEARANCE_MM:
                errors.append("clearance:support_sampled_minimum")
            if support_family.get("continuous_clearance_lower_bound_mm") != EXPECTED_SUPPORT_CONTINUOUS_CLEARANCE_LOWER_BOUND_MM:
                errors.append("clearance:support_continuous_minimum")
            if support_family.get("passed") is not True or fixture_family.get("passed") is not True:
                errors.append("clearance:target_family_pass")

    expect("startup.passed", True)
    expect("startup.penetration_count", 0)
    expect("startup.contact_count", 65)
    contacts = value("startup.contact_records")
    if not isinstance(contacts, list) or canonical_sha256(contacts) != EXPECTED_AUDIT_STREAMS["startup_contact_records_sha256"]:
        errors.append("startup:contact_records")
    expect("startup.contact_records_sha256", EXPECTED_AUDIT_STREAMS["startup_contact_records_sha256"])
    expect("support_topology.tangency_count", EXPECTED_SUPPORT_TOPOLOGY_TANGENCY_COUNT)
    expect("support_topology.passed", True)
    tangencies = value("support_topology.tangencies")
    if not isinstance(tangencies, list) or len(tangencies) != EXPECTED_SUPPORT_TOPOLOGY_TANGENCY_COUNT:
        errors.append("support_topology:tangency_inventory")
    elif any(abs(float(record.get("distance_m", math.inf))) > 1.0e-7 for record in tangencies if isinstance(record, dict)):
        errors.append("support_topology:tangency_distance")

    for tool, expected in (("spoon", EXPECTED_SPOON_POSE), ("whisk", EXPECTED_WHISK_POSE)):
        expect(f"unchanged_matcha_docks.{tool}.declared_position_m", list(expected[0]))
        expect(f"unchanged_matcha_docks.{tool}.declared_quat_wxyz", list(expected[1]))

    expect("authority_scope.continuous_geometric_clearance_authority", True)
    expect("authority_scope.default_action_roster_identity_authority", True)
    for field in (
        "exact_source_brep_boundary_authority",
        "material_authority",
        "mass_authority",
        "fastener_authority",
        "substrate_authority",
        "load_path_authority",
        "tolerance_authority",
        "contact_force_authority",
        "friction_coefficient_authority",
        "capture_dynamics_authority",
        "contact_dynamics_authority",
        "physical_lock_authority",
        "physical_release_authority",
        "default_runtime_action_authority",
        "release_ready",
    ):
        expect(f"authority_scope.{field}", False)
    expect("authority_scope.runtime_contract_blockers", list(EXPECTED_RUNTIME_BLOCKERS))
    expect("authority_scope.checkpoint_blockers", list(CHECKPOINT_BLOCKERS))

    if recompute_runtime and not errors:
        rebuilt = build_rolled_core_dock_runtime_report(repo)
        if rebuilt != report:
            errors.append("runtime:recomputed_report_mismatch")
    return sorted(set(errors))


def validate_rolled_core_dock_runtime_report(
    report: object,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    recompute_runtime: bool = False,
) -> None:
    """Raise ``ValueError`` unless ``report`` is the exact fail-closed checkpoint."""

    errors = rolled_core_dock_runtime_report_errors(
        report,
        repository_root=repository_root,
        recompute_runtime=recompute_runtime,
    )
    if errors:
        raise ValueError("rolled core-dock runtime report invalid: " + ", ".join(errors))


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--check",
        type=Path,
        help="validate an existing report instead of generating one",
    )
    parser.add_argument(
        "--recompute-runtime",
        action="store_true",
        help="with --check, rebuild the full model and require byte-equivalent data",
    )
    args = parser.parse_args(argv)
    if args.check is not None:
        report = json.loads(args.check.read_text(encoding="utf-8"))
        errors = rolled_core_dock_runtime_report_errors(
            report,
            repository_root=args.repo,
            recompute_runtime=args.recompute_runtime,
        )
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True))
        return int(bool(errors))

    report = build_rolled_core_dock_runtime_report(args.repo)
    errors = rolled_core_dock_runtime_report_errors(
        report, repository_root=args.repo
    )
    if errors:
        report["failures"] = sorted(set([*report["failures"], *errors]))
        report["geometry_passed"] = False
        report["passed"] = False
        report["canonical_sha256_without_this_field"] = None
        report["canonical_sha256_without_this_field"] = canonical_sha256(report)
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "continuous_lower_bound_mm": report["continuous_clearance"][
                    "continuous_clearance_lower_bound_mm"
                ],
                "errors": errors,
                "geometry_passed": report["geometry_passed"],
                "output": str(args.output),
                "release_ready": False,
                "report_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return int(bool(errors or report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
