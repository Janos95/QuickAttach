#!/usr/bin/env python3
"""Run the bounded independent matcha development-validation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEST_MODULE = "test_matcha_workflow_validation"
HARD_TIMEOUT_SECONDS = 300
PROCESS_TIMEOUT_SECONDS = 285


def qualified(class_name: str, method_name: str) -> str:
    return f"{TEST_MODULE}.{class_name}.{method_name}"


DEVELOPMENT_TESTS = (
    qualified(
        "MatchaIdentityAndCadContractTests",
        "test_tool_and_device_ids_are_exact_unique_and_non_aliasing",
    ),
    qualified(
        "MatchaIdentityAndCadContractTests",
        "test_cad_manifest_closes_hashes_masses_com_and_interface",
    ),
    qualified(
        "MatchaIdentityAndCadContractTests",
        "test_simulation_config_hash_pins_cad_and_collision_authorities",
    ),
    qualified(
        "MatchaIdentityAndCadContractTests",
        "test_recovery_config_is_fail_closed_and_declares_occt_free_fast_gate",
    ),
    qualified(
        "ControllerSourceSafetyTests",
        "test_controller_call_graph_never_teleports_physical_state",
    ),
    qualified(
        "ControllerSourceSafetyTests", "test_controller_advances_only_through_mujoco_step"
    ),
    qualified(
        "ControllerSourceSafetyTests",
        "test_module_helpers_cannot_hide_post_initialize_state_writes",
    ),
    qualified(
        "ControllerSourceSafetyTests",
        "test_capture_path_cannot_claim_a_returned_physical_lock",
    ),
    qualified(
        "RenderedCollisionInventoryTests",
        "test_every_rendered_rigid_body_has_direct_active_collision_geometry",
    ),
    qualified(
        "RenderedCollisionInventoryTests",
        "test_runtime_collision_coverage_api_is_complete",
    ),
    qualified(
        "RenderedCollisionInventoryTests",
        "test_initialized_scene_has_no_unreviewed_penetration",
    ),
    qualified(
        "RenderedCollisionInventoryTests",
        "test_declared_fixture_support_chains_exist_and_are_exactly_tangent",
    ),
    qualified(
        "RenderedCollisionInventoryTests",
        "test_payload_report_active_geom_inventory_matches_compiled_model",
    ),
    qualified(
        "SimCadPlacementContractTests",
        "test_stock_gripper_wrapper_composes_to_published_step_mount",
    ),
    qualified(
        "SimCadPlacementContractTests",
        "test_each_dock_stop_matches_its_exact_source_bounds_and_core_holes",
    ),
    qualified(
        "SimCadPlacementContractTests",
        "test_core_keeper_contract_is_exact_and_excludes_the_air_gap_stop",
    ),
    qualified(
        "SimCadPlacementContractTests",
        "test_matcha_docks_retain_stop_contact_but_core_uses_only_keepers",
    ),
    qualified(
        "SimCadPlacementContractTests",
        "test_line_keeper_normal_cones_are_oriented_not_any_xz_vector",
    ),
    qualified(
        "SimCadPlacementContractTests",
        "test_withdrawal_threshold_is_exactly_safe_against_source_cam",
    ),
    qualified(
        "BoundedDynamicSmokeTests",
        "test_real_substep_capture_lock_and_release_is_collision_safe",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_fcpw_version_and_candidate_input_guards_are_fail_closed",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_fcpw_candidate_is_replayed_on_original_float64_triangle",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_fcpw_batched_candidates_are_deterministic_conservative_replays",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_fast_gate_declares_signed_occupancy_tolerance_and_step_mesh_error",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_bidirectional_gate_rejects_added_and_dropped_components",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_bidirectional_gate_rejects_a_filled_functional_hole",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_public_certificate_is_bidirectional_signed_and_never_release_ready",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_public_certificate_rejects_invalid_thresholds_and_error_bounds",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_topology_and_sign_gate_rejects_open_or_flipped_meshes",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_topology_sign_is_order_invariant_for_disconnected_components",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_topology_accepts_a_correctly_oriented_nested_cavity",
    ),
    qualified(
        "FcpwFastGateContractTests",
        "test_validator_recomputes_and_rejects_fabricated_evidence",
    ),
)

PRODUCTION_INPUTS = (
    HERE / "matcha_workflow_demo.py",
    HERE / "matcha_workflow_scene.xml",
    HERE / "matcha_tool_geometry.json",
    HERE / "generate_matcha_payload_proxy_report.py",
    HERE / "validate_matcha_payload_proxy_report.py",
    HERE / "matcha_payload_proxy_report.json",
    HERE.parent / "matcha_tools" / "exports" / "matcha_tool_manifest.json",
)

FINGERPRINT_INPUTS = (
    Path(__file__).resolve(),
    HERE / "test_matcha_workflow_validation.py",
    *PRODUCTION_INPUTS,
)


def file_digest(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint() -> str:
    records = [
        {
            "path": path.relative_to(HERE).as_posix()
            if path.is_relative_to(HERE)
            else str(path),
            "sha256": file_digest(path),
        }
        for path in FINGERPRINT_INPUTS
    ]
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def list_payload() -> dict[str, object]:
    missing = [str(path) for path in PRODUCTION_INPUTS if not path.is_file()]
    return {
        "tier": "development",
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "test_count": len(DEVELOPMENT_TESTS),
        "tests": list(DEVELOPMENT_TESTS),
        "production_inputs_ready": not missing,
        "missing_production_inputs": missing,
        "input_fingerprint_sha256": input_fingerprint(),
        "development_pass": None,
        "release_ready": False,
    }


def run_development() -> int:
    started = time.monotonic()
    command = [sys.executable, "-m", "unittest", "-v", *DEVELOPMENT_TESTS]
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    elapsed = time.monotonic() - started
    base = list_payload()
    process_pass = returncode == 0 and not timed_out
    ready = bool(base["production_inputs_ready"])
    payload = {
        **base,
        "wall_seconds": elapsed,
        "within_wall_budget": elapsed <= HARD_TIMEOUT_SECONDS,
        "test_process_pass": process_pass,
        # Missing authorities may skip cleanly, but they never become a green
        # development claim merely because unittest reported no failures.
        "development_pass": bool(
            process_pass and ready and elapsed <= HARD_TIMEOUT_SECONDS
        ),
        "release_ready": False,
        "subprocess_returncode": returncode,
        "subprocess_stdout": stdout,
        "subprocess_stderr": stderr,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if process_pass else returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("development",), default="development")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(list_payload(), sort_keys=True, separators=(",", ":")))
        return 0
    return run_development()


if __name__ == "__main__":
    raise SystemExit(main())
