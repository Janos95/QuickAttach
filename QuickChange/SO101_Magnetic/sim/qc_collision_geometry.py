#!/usr/bin/env python3
"""Compact feature-aware collision installation for the SO-101 quick-change.

The runtime model intentionally keeps the calibrated upstream link meshes and
adds small analytic solids only for functional quick-change features.  It does
not load an undifferentiated hundreds-of-parts decomposition.  Exact CAD and
continuous clearance remain separate validation authorities.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from functools import cache
from pathlib import Path


SIGNALS = ("ground", "power", "data", "id")
SIGNAL_Y_M = {
    "ground": -0.0075,
    "power": -0.0025,
    "data": 0.0025,
    "id": 0.0075,
}
SIGNAL_TO_BORE = {"ground": 1, "power": 2, "data": 3, "id": 4}

# Hash-pinned Mill-Max 7983 runtime envelope.  Geometry and installed datums
# are authoritative for nominal collision placement; mass, spring force, and
# damping are deliberately only a simulation model because the recovered
# manufacturer authority does not contain those properties.
POGO_SOURCE_SIGNAL_BY_RUNTIME = {
    "ground": "GND",
    "power": "+12V",
    "data": "TTL_DATA",
    "id": "TOOL_ID_SPARE",
}
POGO_CAD_SOURCE_PATH = Path(__file__).resolve().parent.parent / "generate_cad.py"
POGO_CAD_SOURCE_SHA256 = (
    "f16352a11128bb93599b83cb51fabdf47520d379f14cdfdb5ae4d5f404e90818"
)
POGO_CAD_SOURCE_BYTES = 140_090
POGO_AUTHORITY_LEDGER_PATH = (
    Path(__file__).resolve().parent.parent
    / "source_authority"
    / "millmax_7983"
    / "authority_ledger.json"
)
POGO_AUTHORITY_LEDGER_SHA256 = (
    "120fda06cb2e889800e5d18ce556b320ed8ba8c1de97b6a48e556d0433353c05"
)
POGO_AUTHORITY_LEDGER_BYTES = 2_031
POGO_SOURCE_CONTRACT_CANONICAL_SHA256 = (
    "71f6b8a2aaeac8f6ebbe140534a0ee3926dca4c078c18b60d5de0abcda6ccc69"
)
POGO_FIXED_SHELL_LENGTH_M = 0.0079248
POGO_PLUNGER_DIAMETER_M = 0.0010668
POGO_PLUNGER_MAX_EXPOSED_LENGTH_M = 0.0015748
POGO_PLUNGER_COMPRESSION_RANGE_M = (0.0, 0.001524)
POGO_FIXED_SHELL_SEGMENTS_M = (
    (
        "solder_cup",
        (0.0, 0.003683),
        0.001524,
        "fixed_shell_solder_cup",
    ),
    (
        "cup_to_knurl_transition_bound",
        (0.003683, 0.00381),
        0.001651,
        "fixed_shell_cup_to_knurl_transition_bound",
    ),
    (
        "knurl",
        (0.00381, 0.004572),
        0.001651,
        "fixed_shell_knurl",
    ),
    (
        "shoulder",
        (0.004572, 0.0052832),
        0.0021082,
        "shoulder_stop",
    ),
    (
        "plunger_side_fixed_features",
        (0.0052832, POGO_FIXED_SHELL_LENGTH_M),
        0.0019431,
        "fixed_shell_plunger_side_fixed_features",
    ),
)
_POGO_REMAINING_STROKE_SEMANTICS = (
    "nominal installation arithmetic only; part, bore, target, and "
    "fabrication tolerances are not included"
)
POGO_INSTALLED_DATUMS_MM = {
    signal: {
        "signal": POGO_SOURCE_SIGNAL_BY_RUNTIME[signal],
        "centre_xy_mm": [-31.0, SIGNAL_Y_M[signal] * 1000.0],
        "installation_mode": "knurl_solder_cup_first",
        "insertion_direction": "mating_face_toward_rear_negative_z",
        "base_z_mm": 0.9004000000000012 if signal == "ground" else 0.7004,
        "fixed_shell_top_z_mm": 8.8252 if signal == "ground" else 8.6252,
        "shoulder_stop_plane_z_mm": (
            5.4724 if signal == "ground" else 5.2724
        ),
        "shoulder_z_bounds_mm": (
            [5.4724, 6.1836]
            if signal == "ground"
            else [5.2724, 5.9836]
        ),
        "knurl_z_bounds_mm": (
            [4.710400000000001, 5.4724]
            if signal == "ground"
            else [4.5104, 5.2724]
        ),
        "retention_land_z_bounds_mm": (
            [1.8499999999999999, 5.4724]
            if signal == "ground"
            else [1.8499999999999999, 5.2724]
        ),
        "body_counterbore_z_bounds_mm": (
            [5.4724, 9.5] if signal == "ground" else [5.2724, 9.5]
        ),
        "full_extension_tip_z_mm": 10.4 if signal == "ground" else 10.2,
        "nominal_face_protrusion_mm": 0.9 if signal == "ground" else 0.7,
        "target_pad_exposed_contact_plane_z_mm": 9.45,
        "mated_compression_mm": (
            0.9500000000000011 if signal == "ground" else 0.75
        ),
        "mated_tip_z_mm": 9.45,
        "nominal_design_remaining_against_catalog_minimum_stroke_mm": (
            0.31999999999999895 if signal == "ground" else 0.52
        ),
        "remaining_stroke_semantics": _POGO_REMAINING_STROKE_SEMANTICS,
    }
    for signal in SIGNALS
}
POGO_INSTALLED_DATUMS_M = {
    signal: {
        "source_signal": datum["signal"],
        "centre_xy_m": tuple(value / 1000.0 for value in datum["centre_xy_mm"]),
        "base_z_m": datum["base_z_mm"] / 1000.0,
    }
    for signal, datum in POGO_INSTALLED_DATUMS_MM.items()
}

# These values are explicitly not manufacturer authority.  They retain a
# bounded, deterministic nominal MuJoCo response until mass and force curves
# are independently sourced.  Critical damping is derived from this declared
# simulation-only cylinder-density model, not inherited from an arm joint.
POGO_SIMULATION_ONLY_ENVELOPE_DENSITY_KG_M3 = 1000.0
POGO_SIMULATION_ONLY_STIFFNESS_N_M = 300.0
POGO_SIMULATION_ONLY_PLUNGER_MASS_KG = (
    POGO_SIMULATION_ONLY_ENVELOPE_DENSITY_KG_M3
    * math.pi
    * (POGO_PLUNGER_DIAMETER_M / 2.0) ** 2
    * POGO_PLUNGER_MAX_EXPOSED_LENGTH_M
)
POGO_SIMULATION_ONLY_DAMPING_N_S_M = 2.0 * math.sqrt(
    POGO_SIMULATION_ONLY_STIFFNESS_N_M
    * POGO_SIMULATION_ONLY_PLUNGER_MASS_KG
)

# Frozen source contracts.  The stock-gripper dock is the released core
# quick-change design; spoon and whisk use the separate two-bay matcha rack.
# Keeping these datums distinct prevents the convenient recovered rack box
# from silently becoming collision authority for three different parts.
CORE_DOCK_STOP_BOUNDS_M = (
    (-0.045, 0.037),
    (0.026, 0.032),
    (-0.003, 0.0125),
)
CORE_DOCK_STOP_HOLES_M = (
    (-0.025, 0.00475, 0.0022),
    (0.021, 0.00475, 0.0022),
)

# Hash-pinned rolled core-dock installation authority.  The generator embeds
# the same ``core_dock_floor_support`` object in design_parameters.json and
# core_cad_manifest.json; the independent clearance report carries an exact
# third copy.  Runtime construction refuses to proceed unless all three are
# equal, their canonical digest is unchanged, and the authored release roster
# still hashes to its public source value.
CORE_DOCK_DESIGN_PARAMETERS_PATH = (
    Path(__file__).resolve().parent.parent / "exports" / "design_parameters.json"
)
CORE_DOCK_CAD_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "exports" / "core_cad_manifest.json"
)
CORE_DOCK_CLEARANCE_REPORT_PATH = (
    Path(__file__).resolve().parent / "cad_clearance_report.json"
)
CORE_DOCK_STEP_PATH = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "so101_passive_tool_dock.step"
)
CORE_DOCK_STL_PATH = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "so101_passive_tool_dock.stl"
)
CORE_DOCK_SUPPORT_STEP_PATH = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "so101_core_dock_support_bracket.step"
)
CORE_DOCK_SUPPORT_STL_PATH = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "so101_core_dock_support_bracket.stl"
)
CORE_DOCK_RUNTIME_SOURCE_FILES = (
    (
        "generator",
        POGO_CAD_SOURCE_PATH,
        140_090,
        "f16352a11128bb93599b83cb51fabdf47520d379f14cdfdb5ae4d5f404e90818",
    ),
    (
        "design_parameters",
        CORE_DOCK_DESIGN_PARAMETERS_PATH,
        58_919,
        "74da4648a0115746efb6e1db26faf722c70bfa298049d4db9c064e226de9c05b",
    ),
    (
        "core_cad_manifest",
        CORE_DOCK_CAD_MANIFEST_PATH,
        61_171,
        "0f0138a95977f43379990b0eb07d83cf316ae0e3631f47328e856ac43c960454",
    ),
    (
        "cad_clearance_report",
        CORE_DOCK_CLEARANCE_REPORT_PATH,
        193_283,
        "a581bc046b934d168ba4c45b3ccd2fec892d93be68b02b46bff5fe54d7f655bf",
    ),
    (
        "core_passive_tool_dock_step",
        CORE_DOCK_STEP_PATH,
        180_034,
        "90aaf1b3c887b4d182358ee1e9037f4159d5959d5822263830dd14854749c389",
    ),
    (
        "core_passive_tool_dock_stl",
        CORE_DOCK_STL_PATH,
        104_084,
        "6337352213b600efcd7d1b8fc02fc4834b04ea12976898399c889950eece84eb",
    ),
    (
        "core_dock_floor_support_step",
        CORE_DOCK_SUPPORT_STEP_PATH,
        170_428,
        "239134fb6933cfd63e50117ec61b1a73873f7e396dbcf6da3c94916dc949b4de",
    ),
    (
        "core_dock_floor_support_stl",
        CORE_DOCK_SUPPORT_STL_PATH,
        281_884,
        "c4928349dbaba21da41e584eab25eee08cb0b82243080800469efb10a5be0c7b",
    ),
)
CORE_DOCK_FLOOR_SUPPORT_SOURCE_CONTRACT_CANONICAL_SHA256 = (
    "1befdd739c9eea2645408178317b93c720d61d11c4fd31802761626ac1b456ee"
)
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
CORE_DOCK_TOOL_VIEW_ROLL_DEG = -87.21086925015224
CORE_DOCK_LOCAL_AXES_WORLD_COLUMNS = (
    (0.6593846719714732, -0.751805729140895, 0.0),
    (0.0, 0.0, -1.0),
    (0.751805729140895, 0.6593846719714733, 0.0),
)
CORE_DOCK_SOURCE_NEGATIVE_Y_WORLD = (0.0, 0.0, 1.0)
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
CORE_DOCK_RELEASE_ROSTER_CANONICAL_SHA256 = (
    "f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293"
)
CORE_DOCK_RELEASE_WITHDRAWAL_BOUNDS_MM = (0.0, 15.0)
CORE_DOCK_RELEASE_STEP_MM = 0.5
CORE_DOCK_RELEASE_SLIDER_PRELOAD_MM = 0.05
CORE_DOCK_RELEASE_SLIDER_RAMP_START_MM = 2.0
CORE_DOCK_RELEASE_SLIDER_SLOPE = 0.246875
CORE_DOCK_RELEASE_SLIDER_LIMIT_MM = 3.0
CORE_DOCK_RELEASE_SLIDER_LIMIT_WITHDRAWAL_MM = 13.949367088607595
CORE_DOCK_FULL_CAM_CLEARANCE_AT_15_MM = 0.251814779
CORE_DOCK_FLOOR_SUPPORT_SOURCE_BLOCKERS = (
    "vendor_or_normative_source_missing_for_selected_M4_and_M6_fasteners",
    "floor_fixture_substrate_and_M6_thread_authority_missing",
    "PA12_modulus_strength_creep_and_process_allowables_unqualified",
    "printed_dimensional_tolerance_and_anchor_strength_unqualified",
    "cam_contact_friction_reverse_insertion_and_capture_dynamics_unvalidated",
)

# The runtime support is a deterministic, pairwise-nonoverlapping box union in
# the rolled dock frame.  It conservatively covers every positive source BRep
# primitive.  Its head is deliberately expanded through the post Z envelope
# so the authored head/post overlap can be represented without duplicate
# solver contacts.  M4/M6 clearances and countersinks are deliberately filled;
# therefore this proxy is collision broadphase only, never physical, mass,
# fastener, substrate, load-path, tolerance, or release authority.
CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M = (
    (
        "head",
        "dock_gripper_floor_support_head_collision",
        ((-0.032, 0.024), (0.032, 0.040), (-0.01675, 0.02625)),
    ),
    (
        "reinforcement",
        "dock_gripper_floor_support_reinforcement_collision",
        ((0.024, 0.027), (0.032, 0.040), (-0.01675, 0.02625)),
    ),
    (
        "neck",
        "dock_gripper_floor_support_neck_collision",
        ((-0.0235, 0.0195), (0.040, 0.050), (-0.01675, 0.02625)),
    ),
    (
        "post_x_min_wall",
        "dock_gripper_floor_support_post_x_min_wall_collision",
        ((-0.0235, -0.0195), (0.050, 0.1859154579377553), (-0.01675, 0.02625)),
    ),
    (
        "post_x_max_wall",
        "dock_gripper_floor_support_post_x_max_wall_collision",
        ((0.0155, 0.0195), (0.050, 0.1859154579377553), (-0.01675, 0.02625)),
    ),
    (
        "post_z_min_wall",
        "dock_gripper_floor_support_post_z_min_wall_collision",
        ((-0.0195, 0.0155), (0.050, 0.1859154579377553), (-0.01675, -0.01275)),
    ),
    (
        "post_z_max_wall",
        "dock_gripper_floor_support_post_z_max_wall_collision",
        ((-0.0195, 0.0155), (0.050, 0.1859154579377553), (0.02225, 0.02625)),
    ),
    (
        "base_x_min_wall",
        "dock_gripper_floor_support_base_x_min_wall_collision",
        ((-0.052, -0.0195), (0.1859154579377553, 0.1939154579377553), (-0.03525, 0.04475)),
    ),
    (
        "base_x_max_wall",
        "dock_gripper_floor_support_base_x_max_wall_collision",
        ((0.0155, 0.048), (0.1859154579377553, 0.1939154579377553), (-0.03525, 0.04475)),
    ),
    (
        "base_z_min_wall",
        "dock_gripper_floor_support_base_z_min_wall_collision",
        ((-0.0195, 0.0155), (0.1859154579377553, 0.1939154579377553), (-0.03525, -0.01275)),
    ),
    (
        "base_z_max_wall",
        "dock_gripper_floor_support_base_z_max_wall_collision",
        ((-0.0195, 0.0155), (0.1859154579377553, 0.1939154579377553), (0.02225, 0.04475)),
    ),
)
CORE_DOCK_SUPPORT_PROXY_GEOM_NAMES = tuple(
    name for _role, name, _bounds in CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M
)
CORE_DOCK_SUPPORT_PROXY_FACE_TANGENCIES = (
    (
        "dock_gripper_floor_support_head_collision",
        "dock_gripper_floor_support_reinforcement_collision",
    ),
    (
        "dock_gripper_floor_support_head_collision",
        "dock_gripper_floor_support_neck_collision",
    ),
    *tuple(
        (
            "dock_gripper_floor_support_neck_collision",
            f"dock_gripper_floor_support_post_{axis}_wall_collision",
        )
        for axis in ("x_min", "x_max", "z_min", "z_max")
    ),
    *tuple(
        (
            f"dock_gripper_floor_support_post_{axis}_wall_collision",
            f"dock_gripper_floor_support_base_{axis}_wall_collision",
        )
        for axis in ("x_min", "x_max", "z_min", "z_max")
    ),
)
CORE_DOCK_SUPPORT_PROXY_FLOOR_CONTACT_GEOM_NAMES = tuple(
    f"dock_gripper_floor_support_base_{axis}_wall_collision"
    for axis in ("x_min", "x_max", "z_min", "z_max")
)
CORE_DOCK_SUPPORT_PROXY_DECLARED_CHAIN = (
    "dock_gripper_qc_col_dock_stop_part_000__dock_stop_land",
    "dock_gripper_floor_support_head_collision",
    "dock_gripper_floor_support_neck_collision",
    "dock_gripper_floor_support_post_x_min_wall_collision",
    "dock_gripper_floor_support_base_x_min_wall_collision",
    "matcha_floor_collision",
)
CORE_DOCK_SUPPORT_SOURCE_VOLUME_MM3 = 162_415.4180526403
CORE_DOCK_SUPPORT_PROXY_ANALYTIC_VOLUME_MM3 = 177_797.24575315934
CORE_DOCK_SUPPORT_PROXY_EXCESS_VOLUME_MM3 = 15_381.827700519032
CORE_DOCK_SUPPORT_PROXY_SOURCE_MISSING_VOLUME_MM3 = 0.0
CORE_DOCK_SUPPORT_PROXY_PASSAGE_WITNESS_M = (-0.002, 0.100, 0.00475)
CORE_DOCK_SUPPORT_PROXY_FILLED_HOLE_WITNESSES_M = (
    (-0.025, 0.036, 0.00475),
    (0.021, 0.036, 0.00475),
    (-0.040, 0.190, -0.02325),
    (-0.040, 0.190, 0.03275),
    (0.036, 0.190, -0.02325),
    (0.036, 0.190, 0.03275),
)
CORE_DOCK_SUPPORT_PROXY_BLOCKERS = (
    "runtime_support_proxy_fills_M4_M6_clearances_and_countersinks",
    "runtime_support_proxy_head_envelope_overcovers_source_for_nonoverlap",
    "runtime_support_proxy_is_not_physical_or_load_path_authority",
)
MATCHA_DOCK_STOP_BOUNDS_M = (
    (-0.041, 0.033),
    (0.025, 0.031),
    (-0.003, 0.0125),
)
CORE_DOCK_CAM_POLYGON_M = (
    (0.028, -0.016),
    (0.034, -0.016),
    (0.034, 0.0),
    (0.02405, 0.0),
)
CORE_DOCK_CAM_Z_BOUNDS_M = (-0.00415, -0.00195)
CORE_DOCK_CAM_AXIAL_LEAD_LOWER_RECTANGLE_M = {
    "x_bounds": (0.02725, 0.029),
    "y_bounds": (0.0, 0.002),
    "z": -0.0096,
}
CORE_DOCK_CAM_AXIAL_LEAD_UPPER_RECTANGLE_M = {
    "x_bounds": (0.02405, 0.029),
    "y_bounds": (0.0, 0.002),
    "z": -0.0064,
}
CORE_DOCK_CAM_HOLD_FINGER_BOUNDS_M = (
    (0.02405, 0.029),
    (0.0, 0.002),
    (-0.0064, -0.00415),
)
CORE_DOCK_CAM_OUTER_ROOT_BRIDGE_BOUNDS_M = (
    (0.028, 0.029),
    (-0.001, 0.001),
    (-0.00465, -0.00365),
)
CORE_DOCK_CAM_OUTER_ROOT_REMAINDER_BOUNDS_M = (
    ((0.028, 0.029), (-0.001, 0.0), (-0.00465, -0.00415)),
    ((0.028, 0.029), (0.0, 0.001), (-0.00415, -0.00365)),
)
CORE_DOCK_CAM_CONTRACT_CANONICAL_SHA256 = (
    "ce86f014833452eedcbdccc04e9f6c0e182718c293c718be5d02dab673a6c633"
)
def positive_lock_cam_collision_geom_names(tool: str) -> tuple[str, ...]:
    if tool not in {"gripper", "spoon", "whisk"}:
        raise ValueError(f"unsupported positive-lock cam tool {tool!r}")
    return (
        f"dock_{tool}_cam_collision",
        f"dock_{tool}_cam_axial_lead_collision",
        f"dock_{tool}_cam_hold_finger_collision",
        f"dock_{tool}_cam_outer_root_lower_collision",
        f"dock_{tool}_cam_outer_root_upper_collision",
    )


CORE_DOCK_CAM_COLLISION_GEOM_NAMES = positive_lock_cam_collision_geom_names(
    "gripper"
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


def core_dock_release_roster() -> list[dict[str, object]]:
    """Return a fresh copy of the exact source-authored rolled release roster."""

    return [
        {"withdrawal_mm": withdrawal_mm, "q_rad": list(q_rad)}
        for withdrawal_mm, q_rad in CORE_DOCK_RELEASE_ROSTER
    ]


def core_dock_release_slider_q_mm(withdrawal_mm: float) -> float:
    """Return the source-authored passive slider coordinate for withdrawal.

    The function is deliberately limited to the published 0--15 mm release
    roster.  Extrapolation would create a new, unauthorised motion claim.
    """

    withdrawal = float(withdrawal_mm)
    lower, upper = CORE_DOCK_RELEASE_WITHDRAWAL_BOUNDS_MM
    if not math.isfinite(withdrawal) or not lower <= withdrawal <= upper:
        raise ValueError(
            "core dock release withdrawal must be finite and inside "
            f"[{lower}, {upper}] mm, got {withdrawal_mm!r}"
        )
    if withdrawal <= CORE_DOCK_RELEASE_SLIDER_RAMP_START_MM:
        return CORE_DOCK_RELEASE_SLIDER_PRELOAD_MM
    return min(
        CORE_DOCK_RELEASE_SLIDER_LIMIT_MM,
        CORE_DOCK_RELEASE_SLIDER_PRELOAD_MM
        + CORE_DOCK_RELEASE_SLIDER_SLOPE
        * (withdrawal - CORE_DOCK_RELEASE_SLIDER_RAMP_START_MM),
    )


def _core_dock_source_contract_copies() -> tuple[dict[str, object], ...]:
    try:
        design_parameters = json.loads(
            CORE_DOCK_DESIGN_PARAMETERS_PATH.read_text(encoding="utf-8")
        )
        manifest = json.loads(
            CORE_DOCK_CAD_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        clearance_report = json.loads(
            CORE_DOCK_CLEARANCE_REPORT_PATH.read_text(encoding="utf-8")
        )
        return (
            design_parameters["collision_geometry_contract"][
                "core_dock_floor_support"
            ],
            manifest["contracts"]["core_dock_floor_support"],
            clearance_report["core_dock_floor_support"]["contract"],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "rolled core dock source contract is missing or malformed"
        ) from exc


def _positive_box_overlap_volume_m3(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> float:
    overlap_lengths = [
        max(0.0, min(left_hi, right_hi) - max(left_lo, right_lo))
        for (left_lo, left_hi), (right_lo, right_hi) in zip(
            left, right, strict=True
        )
    ]
    return math.prod(overlap_lengths)


@cache
def _require_core_dock_runtime_sources() -> None:
    """Fail closed unless every rolled-dock source and contract is exact."""

    for role, path, expected_bytes, expected_sha256 in (
        CORE_DOCK_RUNTIME_SOURCE_FILES
    ):
        try:
            observed_bytes = path.stat().st_size
            observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                f"rolled core dock source is unavailable for {role}: {path}"
            ) from exc
        if observed_bytes != expected_bytes:
            raise RuntimeError(
                f"rolled core dock source byte-count mismatch for {role}: "
                f"expected {expected_bytes}, got {observed_bytes}"
            )
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"rolled core dock source hash mismatch for {role}: "
                f"expected {expected_sha256}, got {observed_sha256}"
            )

    design_contract, manifest_contract, report_contract = (
        _core_dock_source_contract_copies()
    )
    if not design_contract == manifest_contract == report_contract:
        raise RuntimeError(
            "core_dock_floor_support differs across design, manifest, and report"
        )
    canonical = json.dumps(
        design_contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(canonical) != 12_107:
        raise RuntimeError(
            "rolled core dock source contract canonical byte-count mismatch: "
            f"expected 12107, got {len(canonical)}"
        )
    observed_contract_digest = hashlib.sha256(canonical).hexdigest()
    if (
        observed_contract_digest
        != CORE_DOCK_FLOOR_SUPPORT_SOURCE_CONTRACT_CANONICAL_SHA256
    ):
        raise RuntimeError(
            "rolled core dock source contract digest mismatch: expected "
            f"{CORE_DOCK_FLOOR_SUPPORT_SOURCE_CONTRACT_CANONICAL_SHA256}, got "
            f"{observed_contract_digest}"
        )

    expected_roster = core_dock_release_roster()
    source_roster = design_contract.get("release_roster")
    if not isinstance(source_roster, dict):
        raise RuntimeError("rolled core dock release roster is missing")
    if source_roster.get("rows") != expected_roster:
        raise RuntimeError("rolled core dock release roster rows drifted")
    roster_digest = _canonical_json_sha256(expected_roster)
    if roster_digest != CORE_DOCK_RELEASE_ROSTER_CANONICAL_SHA256:
        raise RuntimeError(
            "embedded rolled core dock release roster digest mismatch: "
            f"expected {CORE_DOCK_RELEASE_ROSTER_CANONICAL_SHA256}, got "
            f"{roster_digest}"
        )
    if (
        source_roster.get("canonical_sha256") != roster_digest
        or source_roster.get("expected_canonical_sha256") != roster_digest
        or source_roster.get("row_count") != len(expected_roster)
        or source_roster.get("step_mm") != CORE_DOCK_RELEASE_STEP_MM
        or source_roster.get("withdrawal_bounds_mm")
        != list(CORE_DOCK_RELEASE_WITHDRAWAL_BOUNDS_MM)
    ):
        raise RuntimeError("rolled core dock release roster metadata drifted")

    frame = design_contract.get("frame")
    if not isinstance(frame, dict) or (
        frame.get("position_m") != list(CORE_DOCK_WORLD_POS_M)
        or frame.get("quat_wxyz") != list(CORE_DOCK_WORLD_QUAT_WXYZ)
        or frame.get("tool_view_roll_deg") != CORE_DOCK_TOOL_VIEW_ROLL_DEG
        or frame.get("dock_local_axes_world_columns")
        != [list(axis) for axis in CORE_DOCK_LOCAL_AXES_WORLD_COLUMNS]
        or frame.get("source_negative_y_world")
        != list(CORE_DOCK_SOURCE_NEGATIVE_Y_WORLD)
        or frame.get("source_negative_y_is_world_up") is not True
    ):
        raise RuntimeError("rolled core dock frame contract drifted")

    passive_release = design_contract.get("passive_release")
    expected_slider_law = {
        "0_to_2_mm": CORE_DOCK_RELEASE_SLIDER_PRELOAD_MM,
        "2_to_15_mm_formula": "min(3,0.05+0.246875*(withdrawal_mm-2))",
        "slope": CORE_DOCK_RELEASE_SLIDER_SLOPE,
        "q3_withdrawal_mm": CORE_DOCK_RELEASE_SLIDER_LIMIT_WITHDRAWAL_MM,
    }
    if not isinstance(passive_release, dict) or (
        passive_release.get("axis") != "dock_local_negative_y"
        or passive_release.get("slider_q_mm") != expected_slider_law
        or passive_release.get("full_cam_clearance_at_15_mm")
        != CORE_DOCK_FULL_CAM_CLEARANCE_AT_15_MM
    ):
        raise RuntimeError("rolled core dock passive-release law drifted")
    if (
        design_contract.get("blockers")
        != list(CORE_DOCK_FLOOR_SUPPORT_SOURCE_BLOCKERS)
        or design_contract.get("release_ready") is not False
    ):
        raise RuntimeError("rolled core dock release blocker contract drifted")

    printed_brep = design_contract.get("printed_brep")
    if not isinstance(printed_brep, dict) or (
        printed_brep.get("mass_claim") is not None
        or printed_brep.get("mass_blocker")
        != "printed_material_density_and_condition_not_selected"
    ):
        raise RuntimeError("rolled core dock mass-authority contract drifted")
    component_inventory = printed_brep.get("component_inventory")
    if not isinstance(component_inventory, list):
        raise RuntimeError("rolled core dock printed BRep inventory is missing")
    support_records = [
        record
        for record in component_inventory
        if isinstance(record, dict)
        and record.get("name") == "core_dock_floor_support_bracket"
    ]
    if len(support_records) != 1 or support_records[0].get(
        "expected_volume_mm3"
    ) != CORE_DOCK_SUPPORT_SOURCE_VOLUME_MM3:
        raise RuntimeError("rolled core dock support source volume drifted")

    for index, (_role, _name, bounds) in enumerate(
        CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M
    ):
        if len(bounds) != 3 or any(upper <= lower for lower, upper in bounds):
            raise RuntimeError("rolled core dock support proxy has invalid bounds")
        for _other_role, other_name, other_bounds in (
            CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M[index + 1 :]
        ):
            overlap = _positive_box_overlap_volume_m3(bounds, other_bounds)
            if overlap > 0.0:
                raise RuntimeError(
                    "rolled core dock support proxies overlap: "
                    f"{CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M[index][1]} and "
                    f"{other_name} by {overlap} m^3"
                )
    bounds_by_name = {
        name: bounds
        for _role, name, bounds in CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M
    }
    for left_name, right_name in CORE_DOCK_SUPPORT_PROXY_FACE_TANGENCIES:
        try:
            left_bounds = bounds_by_name[left_name]
            right_bounds = bounds_by_name[right_name]
        except KeyError as exc:
            raise RuntimeError(
                "rolled core dock support tangency names an unknown proxy"
            ) from exc
        signed_intersections = [
            min(left_hi, right_hi) - max(left_lo, right_lo)
            for (left_lo, left_hi), (right_lo, right_hi) in zip(
                left_bounds, right_bounds, strict=True
            )
        ]
        if (
            sum(value == 0.0 for value in signed_intersections) != 1
            or sum(value > 0.0 for value in signed_intersections) != 2
        ):
            raise RuntimeError(
                "rolled core dock support proxies are not face-tangent: "
                f"{left_name}, {right_name}, {signed_intersections}"
            )
    for floor_name in CORE_DOCK_SUPPORT_PROXY_FLOOR_CONTACT_GEOM_NAMES:
        if bounds_by_name[floor_name][1][1] != CORE_DOCK_WORLD_POS_M[2]:
            raise RuntimeError(
                f"rolled core dock base proxy does not close to floor: {floor_name}"
            )
    proxy_volume_mm3 = sum(
        math.prod(upper - lower for lower, upper in bounds) * 1.0e9
        for _role, _name, bounds in CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M
    )
    if not math.isclose(
        proxy_volume_mm3,
        CORE_DOCK_SUPPORT_PROXY_ANALYTIC_VOLUME_MM3,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("rolled core dock support proxy volume drifted")

    def point_is_strictly_inside_proxy(point: tuple[float, float, float]) -> bool:
        return any(
            all(lower < coordinate < upper for coordinate, (lower, upper) in zip(
                point, bounds, strict=True
            ))
            for _role, _name, bounds in CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M
        )

    if point_is_strictly_inside_proxy(
        CORE_DOCK_SUPPORT_PROXY_PASSAGE_WITNESS_M
    ):
        raise RuntimeError("rolled core dock support proxy filled its open passage")
    if not all(
        point_is_strictly_inside_proxy(witness)
        for witness in CORE_DOCK_SUPPORT_PROXY_FILLED_HOLE_WITNESSES_M
    ):
        raise RuntimeError(
            "rolled core dock support proxy no longer conservatively fills a hole"
        )


def core_dock_floor_support_runtime_contract() -> dict[str, object]:
    """Return the exact source binding and declared collision-only proxy.

    A green ``contract_integrity_passed`` means only that the source files,
    three embedded source contracts, rolled frame, release roster/law, and
    emitted analytic boxes are internally exact.  Physical and release
    authority remain explicitly false.
    """

    _require_core_dock_runtime_sources()
    source_contract = _core_dock_source_contract_copies()[0]
    repository_root = Path(__file__).resolve().parents[3]
    source_files = [
        {
            "role": role,
            "path": str(path.relative_to(repository_root)),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
        for role, path, expected_bytes, expected_sha256 in (
            CORE_DOCK_RUNTIME_SOURCE_FILES
        )
    ]
    proxy_components = []
    for role, name, bounds in CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M:
        proxy_components.append(
            {
                "role": role,
                "name": name,
                "geom_type": "box",
                "bounds_m": [list(axis_bounds) for axis_bounds in bounds],
                "local_pos_m": [
                    (lower + upper) / 2.0 for lower, upper in bounds
                ],
                "size_m": [
                    (upper - lower) / 2.0 for lower, upper in bounds
                ],
                "mass_kg": 0.0,
            }
        )
    blockers = list(
        dict.fromkeys(
            (*CORE_DOCK_FLOOR_SUPPORT_SOURCE_BLOCKERS, *CORE_DOCK_SUPPORT_PROXY_BLOCKERS)
        )
    )
    return {
        "schema_version": "1.0-rolled-runtime-checkpoint",
        "source_binding": {
            "files": source_files,
            "triple_contract_locations": [
                "design_parameters.collision_geometry_contract.core_dock_floor_support",
                "core_cad_manifest.contracts.core_dock_floor_support",
                "cad_clearance_report.core_dock_floor_support.contract",
            ],
            "triple_contract_equal": True,
            "canonical_bytes": 10_779,
            "canonical_sha256": (
                CORE_DOCK_FLOOR_SUPPORT_SOURCE_CONTRACT_CANONICAL_SHA256
            ),
            "source_contract": copy.deepcopy(source_contract),
        },
        "runtime_frame": {
            "body_name": "dock_gripper",
            "position_m": list(CORE_DOCK_WORLD_POS_M),
            "quat_wxyz": list(CORE_DOCK_WORLD_QUAT_WXYZ),
            "tool_view_roll_deg": CORE_DOCK_TOOL_VIEW_ROLL_DEG,
            "dock_local_axes_world_columns": [
                list(axis) for axis in CORE_DOCK_LOCAL_AXES_WORLD_COLUMNS
            ],
            "release_axis": "dock_local_negative_y",
            "release_axis_world": list(CORE_DOCK_SOURCE_NEGATIVE_Y_WORLD),
        },
        "release": {
            "roster": core_dock_release_roster(),
            "roster_row_count": len(CORE_DOCK_RELEASE_ROSTER),
            "roster_step_mm": CORE_DOCK_RELEASE_STEP_MM,
            "roster_canonical_sha256": (
                CORE_DOCK_RELEASE_ROSTER_CANONICAL_SHA256
            ),
            "slider_q_mm": {
                "0_to_2_mm": CORE_DOCK_RELEASE_SLIDER_PRELOAD_MM,
                "2_to_15_mm_formula": (
                    "min(3,0.05+0.246875*(withdrawal_mm-2))"
                ),
                "slope": CORE_DOCK_RELEASE_SLIDER_SLOPE,
                "q3_withdrawal_mm": (
                    CORE_DOCK_RELEASE_SLIDER_LIMIT_WITHDRAWAL_MM
                ),
            },
            "full_cam_clearance_at_15_mm": (
                CORE_DOCK_FULL_CAM_CLEARANCE_AT_15_MM
            ),
            "physical_release_authority": False,
        },
        "support_proxy": {
            "frame": "dock_gripper",
            "components": proxy_components,
            "component_count": len(proxy_components),
            "geom_names": list(CORE_DOCK_SUPPORT_PROXY_GEOM_NAMES),
            "pairwise_positive_overlap_count": 0,
            "face_tangencies": [
                list(pair) for pair in CORE_DOCK_SUPPORT_PROXY_FACE_TANGENCIES
            ],
            "floor_contact_geom_names": list(
                CORE_DOCK_SUPPORT_PROXY_FLOOR_CONTACT_GEOM_NAMES
            ),
            "declared_floor_support_chain": list(
                CORE_DOCK_SUPPORT_PROXY_DECLARED_CHAIN
            ),
            "removed_legacy_geom_names": [
                "dock_gripper_support_collision",
                "dock_gripper_support_anchor_collision",
            ],
            "removed_legacy_body_names": ["dock_gripper_support"],
            "mass_kg_per_geom": 0.0,
            "collision_role": "conservative_static_broadphase_proxy",
            "covers_positive_source_primitives": True,
            "M4_M6_holes_and_countersinks_represented": False,
            "conservative_hole_fill": True,
            "exact_source_brep_boundary_authority": False,
            "physical_geometry_authority": False,
            "mass_authority": False,
            "fastener_authority": False,
            "substrate_authority": False,
            "load_path_authority": False,
            "tolerance_authority": False,
        },
        "authority_scope": {
            "source_file_and_contract_identity_authority": True,
            "runtime_pose_and_proxy_primitive_identity_authority": True,
            "physical_geometry_authority": False,
            "mass_authority": False,
            "fastener_authority": False,
            "substrate_authority": False,
            "load_capacity_authority": False,
            "capture_dynamics_authority": False,
            "physical_release_authority": False,
            "blockers": blockers,
            "release_ready": False,
        },
        "contract_integrity_passed": True,
        "release_ready": False,
    }


def positive_lock_cam_runtime_contract() -> dict[str, object]:
    """Return the source-bound nominal runtime cam geometry contract.

    The four source roles reproduce the authored set union exactly.  The
    runtime keeps only the root's two nonoverlapping remainder boxes, avoiding
    duplicate solver contacts while retaining the authored 0.5 mm3 overlap
    provenance.  No contact-force or dynamics authority is claimed.
    """

    component_inputs = (
        (
            "main_xy_wedge",
            "single_convex_prism_mesh",
            {
                "polygon_xy": [list(point) for point in CORE_DOCK_CAM_POLYGON_M],
                "z_bounds": list(CORE_DOCK_CAM_Z_BOUNDS_M),
            },
            (CORE_DOCK_CAM_COLLISION_GEOM_NAMES[0],),
            280.72,
            280.72,
        ),
        (
            "axial_lead",
            "single_convex_ruled_loft_mesh",
            {
                "lower_rectangle": {
                    "x_bounds": list(
                        CORE_DOCK_CAM_AXIAL_LEAD_LOWER_RECTANGLE_M["x_bounds"]
                    ),
                    "y_bounds": list(
                        CORE_DOCK_CAM_AXIAL_LEAD_LOWER_RECTANGLE_M["y_bounds"]
                    ),
                    "z": CORE_DOCK_CAM_AXIAL_LEAD_LOWER_RECTANGLE_M["z"],
                },
                "upper_rectangle": {
                    "x_bounds": list(
                        CORE_DOCK_CAM_AXIAL_LEAD_UPPER_RECTANGLE_M["x_bounds"]
                    ),
                    "y_bounds": list(
                        CORE_DOCK_CAM_AXIAL_LEAD_UPPER_RECTANGLE_M["y_bounds"]
                    ),
                    "z": CORE_DOCK_CAM_AXIAL_LEAD_UPPER_RECTANGLE_M["z"],
                },
            },
            (CORE_DOCK_CAM_COLLISION_GEOM_NAMES[1],),
            21.439999999999994,
            21.439999999999994,
        ),
        (
            "hold_finger",
            "analytic_axis_aligned_box",
            {
                "bounds": [
                    list(bounds) for bounds in CORE_DOCK_CAM_HOLD_FINGER_BOUNDS_M
                ]
            },
            (CORE_DOCK_CAM_COLLISION_GEOM_NAMES[2],),
            22.275,
            22.275,
        ),
        (
            "outer_root_bridge",
            "two_nonoverlapping_analytic_boxes_exact_union_remainder",
            {
                "authored_bounds": [
                    list(bounds)
                    for bounds in CORE_DOCK_CAM_OUTER_ROOT_BRIDGE_BOUNDS_M
                ],
                "runtime_remainder_bounds": [
                    [list(bounds) for bounds in remainder]
                    for remainder in CORE_DOCK_CAM_OUTER_ROOT_REMAINDER_BOUNDS_M
                ],
            },
            CORE_DOCK_CAM_COLLISION_GEOM_NAMES[3:5],
            2.000000000000001,
            1.0,
        ),
    )
    components: list[dict[str, object]] = []
    for (
        source_component,
        representation,
        source_geometry_m,
        geom_names,
        source_volume,
        runtime_volume,
    ) in component_inputs:
        digest_preimage = {
            "source_component": source_component,
            "representation": representation,
            "source_geometry_m": source_geometry_m,
        }
        components.append(
            {
                **digest_preimage,
                "runtime_geom_names": list(geom_names),
                "source_volume_mm3": source_volume,
                "runtime_volume_mm3": runtime_volume,
                "canonical_geometry_sha256": _canonical_json_sha256(
                    digest_preimage
                ),
            }
        )

    repository_root = Path(__file__).resolve().parents[3]
    matcha_bays = {}
    for tool in ("spoon", "whisk"):
        matcha_bays[tool] = {
            "frame": f"dock_{tool}",
            "source_function": "INTERFACE.positive_lock_cam",
            "runtime_geom_names": list(
                positive_lock_cam_collision_geom_names(tool)
            ),
            "uses_core_canonical_geometry": True,
            "geometry_and_placement_authority": True,
        }
    blockers = [
        "positive_lock_cam_friction_coefficient_unqualified",
        "positive_lock_cam_load_capacity_unqualified",
        "positive_lock_cam_dynamics_unqualified",
    ]
    return {
        "schema_version": "1.0",
        "source_binding": {
            "generator_file": {
                "path": str(POGO_CAD_SOURCE_PATH.relative_to(repository_root)),
                "bytes": POGO_CAD_SOURCE_BYTES,
                "sha256": POGO_CAD_SOURCE_SHA256,
            },
            "positive_lock_cam_contract_sha256": (
                CORE_DOCK_CAM_CONTRACT_CANONICAL_SHA256
            ),
        },
        "authority_scope": {
            "geometry_and_placement_authority": True,
            "friction_coefficient_authority": False,
            "load_capacity_authority": False,
            "dynamics_authority": False,
            "overlapping_component_contact_force_authority": False,
            "blockers": blockers,
            "release_ready": False,
        },
        "core_gripper": {
            "frame": "dock_gripper",
            "source_function": "positive_lock_cam",
            "runtime_geom_names": list(CORE_DOCK_CAM_COLLISION_GEOM_NAMES),
            "components": components,
            "expected_union": {
                "bounds_m": [
                    [0.02405, 0.034],
                    [-0.016, 0.002],
                    [-0.0096, -0.00195],
                ],
                "component_volume_sum_mm3": 326.435,
                "runtime_component_volume_sum_mm3": 325.435,
                "authored_pair_overlaps": [
                    {
                        "components": ["main_xy_wedge", "outer_root_bridge"],
                        "volume_mm3": 0.5,
                    },
                    {
                        "components": ["hold_finger", "outer_root_bridge"],
                        "volume_mm3": 0.5,
                    },
                ],
                "authored_pair_overlap_total_mm3": 1.0,
                "runtime_pairwise_overlap_total_mm3": 0.0,
                "source_volume_mm3": 325.435,
            },
        },
        "matcha_bays": matcha_bays,
        "passed": True,
        "release_ready": False,
    }

# The released core plate carries a local fixed-side recess around the full
# passive-cam sweep.  Values are the published 0.50 mm guarded source cutter;
# plate partitions below naturally clip it to finite plate material.
ROBOT_CAM_RELIEF_BOUNDS_M = (
    (0.02355, 0.025),
    (-0.0165, 0.0245),
    (0.0, 0.0095),
)

# A 0.20 mm Z partition leaves a conservative staircase around each round
# core-stop passage.  Its maximum planar miss is below the 0.35 mm runtime
# proxy release limit, while no box ever fills source hole material.
CORE_STOP_HOLE_PARTITION_STEP_M = 0.0002

# The lock slider is a thin, non-convex sheet with two keyholes and a guide
# slot.  One MuJoCo mesh geom would convexify it and silently fill those voids,
# so the source top face is decomposed into convex triangular prisms.  The STEP
# hash and absolute OCCT meshing controls are part of the runtime contract.
POSITIVE_LOCK_SLIDER_STEP = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "so101_positive_lock_slider.step"
)
POSITIVE_LOCK_SLIDER_STEP_BYTES = 94_504
POSITIVE_LOCK_SLIDER_STEP_SHA256 = (
    "b9853fe1fcbd7dff91129d7b37c3e1be87d48189486ad91c4c8ab6d57edcbad1"
)
POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM = 0.005
POSITIVE_LOCK_SLIDER_ANGULAR_DEFLECTION_RAD = 0.3
POSITIVE_LOCK_SLIDER_VOID_EXPANSION_MM = 0.010
POSITIVE_LOCK_SLIDER_Z_BOUNDS_M = (0.0, 0.0016)
# The MJCF joint reference is the physical locked q=3 mm state.  Placing the
# body at +3 mm makes q=3 retain locked source geometry while q=0 applies the
# expected -3 mm displacement back to the unlocked source profile.
POSITIVE_LOCK_SLIDER_BASE_POS_M = (0.003, 0.0, 0.0047)
POSITIVE_LOCK_SLIDER_JOINT_RANGE_M = (0.0, 0.003)
POSITIVE_LOCK_SLIDER_SPRINGREF_M = 0.0036
POSITIVE_LOCK_SLIDER_STIFFNESS_N_M = 980.0
POSITIVE_LOCK_SLIDER_DENSITY_KG_M3 = 8000.0
POSITIVE_LOCK_HARDWARE_DENSITY_KG_M3 = 8000.0
POSITIVE_LOCK_SLIDER_SOURCE_VOLUME_MM3 = 220.12468083955645
POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG = 0.0017609974467164514
POSITIVE_LOCK_SLIDER_SOURCE_COM_M = (0.00482206920910109, 0.0, 0.0008)
POSITIVE_LOCK_SLIDER_SOURCE_INERTIA_KG_M2 = (
    7.815117222854e-9,
    2.266197667455e-7,
    2.336835250577e-7,
)
POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2 = (
    *POSITIVE_LOCK_SLIDER_SOURCE_INERTIA_KG_M2,
    0.0,
    0.0,
    0.0,
)
POSITIVE_LOCK_SLIDER_DAMPING_N_S_M = 2.0 * math.sqrt(
    POSITIVE_LOCK_SLIDER_STIFFNESS_N_M * POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG
)
POSITIVE_LOCK_SLIDER_LIMIT_SOLREF = (0.0005, 1.0)
POSITIVE_LOCK_SLIDER_LIMIT_SOLIMP = (0.99, 0.9999, 0.00001, 0.5, 2.0)

# The positive-lock hardware is a rigid, tool-owned source assembly.  Its
# dynamics are derived from the two hash-pinned shoulder-screw and holed-nut
# STEP sources in the tool-root frame, never from overlapping collision
# primitives.  Each nut is translated +1.5 mm along tool-local Z; the two
# screw/nut stacks are translated to X=+/-12 mm, Y=0.
POSITIVE_LOCK_SHOULDER_SCREW_STEP = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "hardware_McMaster_90318A720_shoulder_screw.step"
)
POSITIVE_LOCK_SHOULDER_SCREW_STEP_BYTES = 13_596
POSITIVE_LOCK_SHOULDER_SCREW_STEP_SHA256 = (
    "c2612e972d5af7ae9b9ebd1ec78b8e2b563cd536173ad30230a7f60b8d844f2b"
)
POSITIVE_LOCK_STUD_NUT_STEP = (
    Path(__file__).resolve().parent.parent
    / "exports"
    / "hardware_DIN934_M3_lock_stud_nut.step"
)
POSITIVE_LOCK_STUD_NUT_STEP_BYTES = 27_470
POSITIVE_LOCK_STUD_NUT_STEP_SHA256 = (
    "2682fb17a7a369998b89cb9fe5f5b3b1fe3708ed2ac2a1a67ebb22cd3ace8261"
)
POSITIVE_LOCK_HARDWARE_STACK_X_M = (-0.012, 0.012)
POSITIVE_LOCK_HARDWARE_NUT_TRANSLATION_M = (0.0, 0.0, 0.0015)
POSITIVE_LOCK_HARDWARE_SOURCE_VOLUME_MM3 = 342.8686400999023
POSITIVE_LOCK_HARDWARE_SOURCE_MASS_KG = 0.0027429491207992185
POSITIVE_LOCK_HARDWARE_SOURCE_COM_M = (0.0, 0.0, -0.0011115796404887526)
POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2 = (
    3.6173174131513016e-8,
    4.31157847526606e-7,
    4.0398189716095705e-7,
)
POSITIVE_LOCK_HARDWARE_SOURCE_FULL_INERTIA_KG_M2 = (
    *POSITIVE_LOCK_HARDWARE_SOURCE_INERTIA_KG_M2,
    0.0,
    0.0,
    0.0,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "unnamed"


def _rotate_by_quat(
    vector: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate one vector by a normalized MuJoCo wxyz quaternion."""

    vx, vy, vz = vector
    w_value, qx, qy, qz = quat
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w_value * tx + (qy * tz - qz * ty),
        vy + w_value * ty + (qz * tx - qx * tz),
        vz + w_value * tz + (qx * ty - qy * tx),
    )


def _geom(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, ...] | None,
    rgba: str,
    quat: str | None = None,
    collision: bool = True,
    group: int = 3,
    **attributes: str,
) -> ET.Element:
    record = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(f"{value:.12g}" for value in pos),
        "rgba": rgba,
        "group": str(group),
        "contype": "1" if collision else "0",
        "conaffinity": "1" if collision else "0",
    }
    if size is not None:
        record["size"] = " ".join(f"{value:.12g}" for value in size)
    if quat is not None:
        record["quat"] = quat
    record.update(attributes)
    return ET.SubElement(parent, "geom", record)


def _add_convex_prism_mesh(
    asset: ET.Element,
    *,
    name: str,
    polygon_xy: tuple[tuple[float, float], ...],
    z_bounds: tuple[float, float],
) -> str:
    """Install one deterministic convex source-contract prism mesh."""

    if len(polygon_xy) < 3:
        raise ValueError("a prism needs at least three polygon vertices")
    existing = asset.find(f"./mesh[@name='{name}']")
    if existing is not None:
        return name
    z_min, z_max = z_bounds
    vertices = [(*point, z_min) for point in polygon_xy]
    vertices.extend((*point, z_max) for point in polygon_xy)
    count = len(polygon_xy)
    faces: list[tuple[int, int, int]] = []
    # Input polygons are counter-clockwise.  Reverse the lower cap, retain the
    # upper cap and triangulate each side with deterministic vertex ordering.
    for index in range(1, count - 1):
        faces.append((0, index + 1, index))
        faces.append((count, count + index, count + index + 1))
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following))
        faces.append((index, count + following, count + index))
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": name,
            "vertex": " ".join(
                f"{coordinate:.12g}" for vertex in vertices for coordinate in vertex
            ),
            "face": " ".join(str(item) for face in faces for item in face),
        },
    )
    return name


def _add_convex_polyhedron_mesh(
    asset: ET.Element,
    *,
    name: str,
    vertices: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, int, int], ...],
) -> str:
    """Install one closed, outward-oriented deterministic convex mesh."""

    if len(vertices) < 4 or len(faces) < 4:
        raise ValueError(f"polyhedron {name} is incomplete")
    if any(not math.isfinite(value) for vertex in vertices for value in vertex):
        raise ValueError(f"polyhedron {name} has a nonfinite vertex")
    edge_counts: dict[tuple[int, int], int] = {}
    signed_volume = 0.0
    for face in faces:
        if len(set(face)) != 3 or any(index < 0 or index >= len(vertices) for index in face):
            raise ValueError(f"polyhedron {name} has an invalid face")
        first, second, third = (vertices[index] for index in face)
        edge_a = tuple(second[axis] - first[axis] for axis in range(3))
        edge_b = tuple(third[axis] - first[axis] for axis in range(3))
        cross = (
            edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
            edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
            edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
        )
        if math.sqrt(sum(value * value for value in cross)) <= 1.0e-18:
            raise ValueError(f"polyhedron {name} has a degenerate face")
        signed_volume += sum(first[axis] * cross[axis] for axis in range(3)) / 6.0
        for edge_index in range(3):
            edge = tuple(
                sorted((face[edge_index], face[(edge_index + 1) % 3]))
            )
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError(f"polyhedron {name} is not closed")
    if signed_volume <= 1.0e-18:
        raise ValueError(f"polyhedron {name} is not outward-oriented")
    existing = asset.find(f"./mesh[@name='{name}']")
    if existing is not None:
        return name
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": name,
            "vertex": " ".join(
                f"{coordinate:.12g}"
                for vertex in vertices
                for coordinate in vertex
            ),
            "face": " ".join(str(index) for face in faces for index in face),
        },
    )
    return name


def _core_cam_axial_lead_mesh_geometry() -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    """Return the exact eight-vertex ruled source lead."""

    lower = CORE_DOCK_CAM_AXIAL_LEAD_LOWER_RECTANGLE_M
    upper = CORE_DOCK_CAM_AXIAL_LEAD_UPPER_RECTANGLE_M
    lower_x_min, lower_x_max = lower["x_bounds"]
    lower_y_min, lower_y_max = lower["y_bounds"]
    upper_x_min, upper_x_max = upper["x_bounds"]
    upper_y_min, upper_y_max = upper["y_bounds"]
    vertices = (
        (lower_x_min, lower_y_min, lower["z"]),
        (lower_x_max, lower_y_min, lower["z"]),
        (lower_x_max, lower_y_max, lower["z"]),
        (lower_x_min, lower_y_max, lower["z"]),
        (upper_x_min, upper_y_min, upper["z"]),
        (upper_x_max, upper_y_min, upper["z"]),
        (upper_x_max, upper_y_max, upper["z"]),
        (upper_x_min, upper_y_max, upper["z"]),
    )
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (0, 4, 7),
        (0, 7, 3),
    )
    return vertices, faces


def _expand_slider_void_boundary_vertex_mm(
    x_value: float, y_value: float
) -> tuple[float, float]:
    """Move one known STEP void-boundary node conservatively into material.

    OCCT places polygon nodes on exact circular hole boundaries.  Their chords
    lie inside the void and would therefore add a few microns of false runtime
    material.  Expanding only identified inner-boundary nodes by 10 um makes
    every chord conservative; the expansion plus OCCT's reported faceting
    error remains below the 20 um functional-interface budget.
    """

    margin = POSITIVE_LOCK_SLIDER_VOID_EXPANSION_MM
    boundary_tolerance = 1.0e-6

    # Two 6.5 mm entry holes.  Only exposed circular arcs occur on the final
    # face boundary; straight neck portions are handled below.
    for center_x in (-12.0, 12.0):
        dx = x_value - center_x
        radius = math.hypot(dx, y_value)
        if abs(radius - 3.25) <= boundary_tolerance:
            scale = (3.25 + margin) / radius
            return center_x + dx * scale, y_value * scale

    # M2 guide slot: overall length 5.4 mm, width 2.4 mm, centered at x=-1.5.
    # CadQuery's slot has cap centers x=-3 and x=0.
    for center_x, cap_side in ((-3.0, "left"), (0.0, "right")):
        dx = x_value - center_x
        radius = math.hypot(dx, y_value)
        on_exposed_cap = (
            cap_side == "left" and x_value <= center_x + boundary_tolerance
        ) or (cap_side == "right" and x_value >= center_x - boundary_tolerance)
        if on_exposed_cap and abs(radius - 1.2) <= boundary_tolerance:
            scale = (1.2 + margin) / radius
            return center_x + dx * scale, y_value * scale
    if -3.0 - boundary_tolerance <= x_value <= boundary_tolerance and abs(
        abs(y_value) - 1.2
    ) <= boundary_tolerance:
        return x_value, math.copysign(1.2 + margin, y_value)

    # Each released keyhole neck is a 7.25 x 4.25 mm capsule whose centreline
    # runs from stud_x-3 mm (locked shoulder centre) to stud_x (unlocked
    # shoulder centre).  The entry-circle test above owns the larger R3.25
    # boundary around the unlocked endpoint.  Expand the exposed R2.125 left
    # cap and straight flanks into source material; boundary-node filtering in
    # the caller ensures covered/internal capsule arcs are never adjusted.
    neck_radius = 2.125
    for stud_x in (-12.0, 12.0):
        locked_center_x = stud_x - 3.0
        dx = x_value - locked_center_x
        radius = math.hypot(dx, y_value)
        if (
            x_value <= locked_center_x + boundary_tolerance
            and abs(radius - neck_radius) <= boundary_tolerance
        ):
            scale = (neck_radius + margin) / radius
            return locked_center_x + dx * scale, y_value * scale
        if (
            locked_center_x - boundary_tolerance
            <= x_value
            <= stud_x + boundary_tolerance
            and abs(abs(y_value) - neck_radius) <= boundary_tolerance
        ):
            return x_value, math.copysign(neck_radius + margin, y_value)
    return x_value, y_value


@cache
def _require_exact_core_export_step(
    path: Path, expected_bytes: int, expected_sha256: str
) -> bytes:
    """Bind one runtime STEP to both its file and pinned manifest record."""

    _require_core_dock_runtime_sources()
    source_bytes = path.read_bytes()
    if len(source_bytes) != expected_bytes:
        raise RuntimeError(
            f"core runtime STEP byte-count mismatch for {path.name}: "
            f"expected {expected_bytes}, got {len(source_bytes)}"
        )
    observed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"core runtime STEP hash mismatch for {path.name}: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    repository_root = Path(__file__).resolve().parents[3]
    expected_record = {
        "bytes": expected_bytes,
        "path": path.relative_to(repository_root).as_posix(),
        "role": "exact_cad_step",
        "sha256": expected_sha256,
    }
    try:
        manifest = json.loads(
            CORE_DOCK_CAD_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        matches = [
            record
            for record in manifest["files"]
            if isinstance(record, dict)
            and record.get("path") == expected_record["path"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("core CAD manifest file inventory is malformed") from exc
    if matches != [expected_record]:
        raise RuntimeError(
            f"core CAD manifest record mismatch for {path.name}: "
            f"expected {expected_record}, got {matches}"
        )
    return source_bytes


@cache
def positive_lock_slider_profile_triangles_m() -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
]:
    """Return a deterministic conservative triangle partition of the slider.

    STEP import is the only B-rep operation in this runtime builder.  Collision
    is performed by MuJoCo against the returned convex prisms; no OCCT Boolean
    result is used as a runtime contact decision.
    """

    _require_exact_core_export_step(
        POSITIVE_LOCK_SLIDER_STEP,
        POSITIVE_LOCK_SLIDER_STEP_BYTES,
        POSITIVE_LOCK_SLIDER_STEP_SHA256,
    )

    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    shape = cq.importers.importStep(str(POSITIVE_LOCK_SLIDER_STEP)).val().wrapped
    BRepTools.Clean_s(shape)
    mesher = BRepMesh_IncrementalMesh(
        shape,
        POSITIVE_LOCK_SLIDER_ABSOLUTE_DEFLECTION_MM,
        False,
        POSITIVE_LOCK_SLIDER_ANGULAR_DEFLECTION_RAD,
        False,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("absolute OCCT slider tessellation did not complete")

    top_nodes_mm: list[tuple[float, float, float]] | None = None
    top_faces: list[tuple[int, int, int]] | None = None
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            nodes = [
                triangulation.Node(index).Transformed(transform)
                for index in range(1, triangulation.NbNodes() + 1)
            ]
            if nodes and min(float(point.Z()) for point in nodes) >= 1.6 - 1.0e-8:
                if top_nodes_mm is not None:
                    raise RuntimeError("slider STEP has more than one top face")
                top_nodes_mm = [
                    (float(point.X()), float(point.Y()), float(point.Z()))
                    for point in nodes
                ]
                top_faces = [
                    tuple(int(value) - 1 for value in triangulation.Triangle(index).Get())
                    for index in range(1, triangulation.NbTriangles() + 1)
                ]
        explorer.Next()
    if top_nodes_mm is None or top_faces is None:
        raise RuntimeError("slider STEP top-face triangulation is absent")

    edge_counts: dict[tuple[int, int], int] = {}
    for face in top_faces:
        for index in range(3):
            edge = tuple(sorted((face[index], face[(index + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_nodes = {
        node_index
        for edge, count in edge_counts.items()
        if count == 1
        for node_index in edge
    }
    adjusted_xy_mm = [(x_value, y_value) for x_value, y_value, _ in top_nodes_mm]
    for index in boundary_nodes:
        adjusted_xy_mm[index] = _expand_slider_void_boundary_vertex_mm(
            *adjusted_xy_mm[index]
        )

    triangles: list[tuple[tuple[float, float], ...]] = []
    for face in top_faces:
        polygon_mm = [adjusted_xy_mm[index] for index in face]
        signed_twice_area = sum(
            polygon_mm[index][0] * polygon_mm[(index + 1) % 3][1]
            - polygon_mm[(index + 1) % 3][0] * polygon_mm[index][1]
            for index in range(3)
        )
        if abs(signed_twice_area) <= 1.0e-12:
            raise RuntimeError("slider source triangulation contains a degenerate face")
        if signed_twice_area < 0.0:
            polygon_mm.reverse()
        polygon_m = tuple(
            (float(x_value) * 1.0e-3, float(y_value) * 1.0e-3)
            for x_value, y_value in polygon_mm
        )
        first = min(range(3), key=lambda index: polygon_m[index])
        canonical = tuple(polygon_m[(first + offset) % 3] for offset in range(3))
        triangles.append(canonical)
    triangles.sort()
    if len(triangles) < 100 or len(set(triangles)) != len(triangles):
        raise RuntimeError("slider triangle partition is incomplete or duplicated")
    return tuple(triangles)


def add_positive_lock_slider(frame: ET.Element, asset: ET.Element) -> list[str]:
    """Add the spring-return slider as a moving union of convex prisms."""

    body = ET.SubElement(
        frame,
        "body",
        {
            "name": "qc_positive_lock_slider",
            "pos": " ".join(f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_BASE_POS_M),
        },
    )
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": " ".join(
                f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_SOURCE_COM_M
            ),
            "mass": f"{POSITIVE_LOCK_SLIDER_SOURCE_MASS_KG:.15g}",
            "fullinertia": " ".join(
                f"{value:.15g}"
                for value in POSITIVE_LOCK_SLIDER_SOURCE_FULL_INERTIA_KG_M2
            ),
        },
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": "qc_positive_lock_slider_joint",
            "type": "slide",
            "axis": "1 0 0",
            "range": " ".join(
                f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_JOINT_RANGE_M
            ),
            "limited": "true",
            "ref": f"{POSITIVE_LOCK_SLIDER_JOINT_RANGE_M[1]:.12g}",
            "stiffness": f"{POSITIVE_LOCK_SLIDER_STIFFNESS_N_M:.12g}",
            "springref": f"{POSITIVE_LOCK_SLIDER_SPRINGREF_M:.12g}",
            # The calibrated SO-101 default joint class is a servo-joint
            # model and is not authority for this passive sheet-metal slide.
            # Damping is the analytically derived critical value 2*sqrt(k*m)
            # for the exact STEP mass; Coulomb friction and virtual armature
            # remain explicitly absent.  Limit impedance is frozen separately
            # so the physical 3 mm travel is not governed by MuJoCo defaults.
            "damping": f"{POSITIVE_LOCK_SLIDER_DAMPING_N_S_M:.15g}",
            "frictionloss": "0",
            "armature": "0",
            "solreflimit": " ".join(
                f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_LIMIT_SOLREF
            ),
            "solimplimit": " ".join(
                f"{value:.12g}" for value in POSITIVE_LOCK_SLIDER_LIMIT_SOLIMP
            ),
        },
    )
    names: list[str] = []
    role_counts = {"bridge": 0, "left_lobe": 0, "right_lobe": 0, "tab": 0}
    for triangle in positive_lock_slider_profile_triangles_m():
        centroid_x = sum(point[0] for point in triangle) / 3.0
        centroid_y = sum(point[1] for point in triangle) / 3.0
        if centroid_x >= 0.0159 and abs(centroid_y) <= 0.00201:
            role = "tab"
        elif centroid_x < -0.012 or abs(centroid_y) > 0.0024:
            role = "left_lobe" if centroid_x < 0.0 else "right_lobe"
        else:
            role = "bridge"
        role_index = role_counts[role]
        role_counts[role] += 1
        name = f"qc_col_lock_slider_{role}_part_{role_index:03d}"
        mesh_name = _add_convex_prism_mesh(
            asset,
            name=f"qc_lock_slider_{role}_source_prism_{role_index:03d}",
            polygon_xy=triangle,
            z_bounds=POSITIVE_LOCK_SLIDER_Z_BOUNDS_M,
        )
        _geom(
            body,
            name=name,
            geom_type="mesh",
            size=None,
            mesh=mesh_name,
            rgba="0.10 0.75 0.34 0.82",
            contype="64",
            conaffinity="31",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(name)
    if any(count == 0 for count in role_counts.values()):
        raise RuntimeError(f"slider role partition is incomplete: {role_counts}")
    return names


def _add_box_from_bounds(
    parent: ET.Element,
    *,
    name: str,
    bounds: tuple[tuple[float, float], ...],
    rgba: str,
    **attributes: str,
) -> ET.Element:
    if len(bounds) != 3 or any(hi <= lo for lo, hi in bounds):
        raise ValueError(f"invalid box bounds for {name}: {bounds}")
    return _geom(
        parent,
        name=name,
        geom_type="box",
        pos=tuple((lo + hi) / 2.0 for lo, hi in bounds),
        size=tuple((hi - lo) / 2.0 for lo, hi in bounds),
        rgba=rgba,
        **attributes,
    )


def _add_triangle_prism_geom(
    body: ET.Element,
    asset: ET.Element,
    *,
    name: str,
    triangle_xy: tuple[tuple[float, float], ...],
    z_bounds: tuple[float, float],
    rgba: str,
    **attributes: str,
) -> str:
    signed_twice_area = sum(
        triangle_xy[index][0] * triangle_xy[(index + 1) % 3][1]
        - triangle_xy[(index + 1) % 3][0] * triangle_xy[index][1]
        for index in range(3)
    )
    if abs(signed_twice_area) <= 1.0e-18:
        raise ValueError(f"degenerate triangle for {name}")
    polygon = triangle_xy if signed_twice_area > 0.0 else tuple(reversed(triangle_xy))
    mesh_name = _add_convex_prism_mesh(
        asset,
        name=f"{name}_source_mesh",
        polygon_xy=polygon,
        z_bounds=z_bounds,
    )
    _geom(
        body,
        name=name,
        geom_type="mesh",
        size=None,
        mesh=mesh_name,
        rgba=rgba,
        **attributes,
    )
    return name


def _add_square_minus_regular_void_partition(
    body: ET.Element,
    asset: ET.Element,
    *,
    name_prefix: str,
    center_xy: tuple[float, float],
    outer_half_size: float,
    void_vertex_radius: float,
    sides: int,
    z_bounds: tuple[float, float],
    rgba: str,
    semantic_suffix: str = "",
    **attributes: str,
) -> list[str]:
    """Partition a square-minus-convex-void ring into convex prisms."""

    if sides < 6 or outer_half_size <= void_vertex_radius:
        raise ValueError("invalid regular-void partition")
    center_x, center_y = center_xy
    names: list[str] = []
    inner: list[tuple[float, float]] = []
    outer: list[tuple[float, float]] = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        cosine = math.cos(angle)
        sine = math.sin(angle)
        inner.append(
            (
                center_x + void_vertex_radius * cosine,
                center_y + void_vertex_radius * sine,
            )
        )
        ray_scale = outer_half_size / max(abs(cosine), abs(sine))
        outer.append(
            (center_x + ray_scale * cosine, center_y + ray_scale * sine)
        )
    triangle_index = 0
    for index in range(sides):
        following = (index + 1) % sides
        for triangle in (
            (inner[index], outer[index], outer[following]),
            (inner[index], outer[following], inner[following]),
        ):
            name = f"{name_prefix}_{triangle_index:03d}{semantic_suffix}"
            names.append(
                _add_triangle_prism_geom(
                    body,
                    asset,
                    name=name,
                    triangle_xy=triangle,
                    z_bounds=z_bounds,
                    rgba=rgba,
                    **attributes,
                )
            )
            triangle_index += 1
    return names


def _add_two_well_plate_layer(
    body: ET.Element,
    asset: ET.Element,
    *,
    name_prefix: str,
    plate_bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    z_bounds: tuple[float, float],
    well_x_values: tuple[float, float],
    source_well_radius: float,
    semantic_suffix: str,
    rgba: str,
    first_box_name: str | None = None,
    **attributes: str,
) -> list[str]:
    """Add a rectangular layer with two conservative circular head wells."""

    sides = 40
    void_radius = source_well_radius / math.cos(math.pi / sides) + 1.0e-6
    outer_half = void_radius + 1.0e-5
    x_min, x_max = plate_bounds_xy[0]
    y_min, y_max = plate_bounds_xy[1]
    left_x, right_x = well_x_values
    strips = (
        ((x_min, left_x - outer_half), (y_min, y_max)),
        ((left_x + outer_half, right_x - outer_half), (y_min, y_max)),
        ((right_x + outer_half, x_max), (y_min, y_max)),
        ((left_x - outer_half, left_x + outer_half), (y_min, -outer_half)),
        ((left_x - outer_half, left_x + outer_half), (outer_half, y_max)),
        ((right_x - outer_half, right_x + outer_half), (y_min, -outer_half)),
        ((right_x - outer_half, right_x + outer_half), (outer_half, y_max)),
    )
    names: list[str] = []
    for index, bounds_xy in enumerate(strips):
        name = (
            first_box_name
            if index == 0 and first_box_name is not None
            else f"{name_prefix}_box_{index:02d}{semantic_suffix}"
        )
        _add_box_from_bounds(
            body,
            name=name,
            bounds=(*bounds_xy, z_bounds),
            rgba=rgba,
            **attributes,
        )
        names.append(name)
    for side, center_x in (("left", left_x), ("right", right_x)):
        names.extend(
            _add_square_minus_regular_void_partition(
                body,
                asset,
                name_prefix=f"{name_prefix}_{side}_well_part",
                center_xy=(center_x, 0.0),
                outer_half_size=outer_half,
                void_vertex_radius=void_radius,
                sides=sides,
                z_bounds=z_bounds,
                rgba=rgba,
                semantic_suffix=semantic_suffix,
                **attributes,
            )
        )
    return names


def _slider_track_upper_boundary_m() -> tuple[tuple[float, float], ...]:
    """Return an x-monotone conservative upper boundary of the track void."""

    margin = 1.0e-5
    radius = 0.0044 + 0.00022 + margin
    bridge_y = 0.0024 + 0.00022 + margin
    tab_y = 0.0020 + 0.00022 + margin
    points: list[tuple[float, float]] = []

    def append_arc(center_x: float, start: float, end: float) -> None:
        count = max(1, int(math.ceil(abs(end - start) / 0.08)))
        for index in range(count + 1):
            if points and index == 0:
                continue
            angle = start + (end - start) * index / count
            points.append(
                (center_x + radius * math.cos(angle), radius * math.sin(angle))
            )

    bridge_angle = math.asin(bridge_y / radius)
    tab_angle = math.asin(tab_y / radius)
    append_arc(-0.012, math.pi, math.pi / 2.0)
    points.append((-0.009, radius))
    append_arc(-0.009, math.pi / 2.0, bridge_angle)
    points.append((0.012 - radius * math.cos(bridge_angle), bridge_y))
    append_arc(0.012, math.pi - bridge_angle, math.pi / 2.0)
    points.append((0.015, radius))
    append_arc(0.015, math.pi / 2.0, tab_angle)
    points.append((0.02725 + margin, tab_y))

    ordered: list[tuple[float, float]] = []
    for point in sorted(points):
        if ordered and abs(point[0] - ordered[-1][0]) <= 1.0e-12:
            ordered[-1] = (point[0], max(point[1], ordered[-1][1]))
        else:
            ordered.append(point)
    if any(ordered[index][0] >= ordered[index + 1][0] for index in range(len(ordered) - 1)):
        raise RuntimeError("slider track boundary is not x-monotone")
    return tuple(ordered)


def _add_slider_track_plate_layer(
    body: ET.Element,
    asset: ET.Element,
    *,
    name_prefix: str,
    plate_bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    z_bounds: tuple[float, float],
    rgba: str,
    **attributes: str,
) -> list[str]:
    """Partition a plate layer around the conservative swept slider void."""

    x_min, x_max = plate_bounds_xy[0]
    y_min, y_max = plate_bounds_xy[1]
    source_points = _slider_track_upper_boundary_m()
    points = [point for point in source_points if x_min < point[0] < x_max]
    # The tab leaves through the relieved +X edge; clip its flat upper edge at
    # the remaining plate-material boundary.
    if points[-1][0] < x_max:
        points.append((x_max, points[-1][1]))
    elif points[-1][0] > x_max:
        previous = points[-2]
        following = points[-1]
        alpha = (x_max - previous[0]) / (following[0] - previous[0])
        points[-1] = (
            x_max,
            previous[1] + alpha * (following[1] - previous[1]),
        )

    names: list[str] = []
    left_void_x = points[0][0]
    if left_void_x > x_min:
        name = f"{name_prefix}_left_box_000"
        _add_box_from_bounds(
            body,
            name=name,
            bounds=((x_min, left_void_x), (y_min, y_max), z_bounds),
            rgba=rgba,
            **attributes,
        )
        names.append(name)
    triangle_index = 0
    for lower, upper in zip(points, points[1:]):
        for triangle in (
            ((lower[0], lower[1]), (upper[0], upper[1]), (upper[0], y_max)),
            ((lower[0], lower[1]), (upper[0], y_max), (lower[0], y_max)),
            ((lower[0], -lower[1]), (upper[0], y_min), (upper[0], -upper[1])),
            ((lower[0], -lower[1]), (lower[0], y_min), (upper[0], y_min)),
        ):
            name = f"{name_prefix}_part_{triangle_index:03d}"
            names.append(
                _add_triangle_prism_geom(
                    body,
                    asset,
                    name=name,
                    triangle_xy=triangle,
                    z_bounds=z_bounds,
                    rgba=rgba,
                    **attributes,
                )
            )
            triangle_index += 1
    return names


def activate_upstream_robot_collisions(robot_root: ET.Element) -> list[str]:
    """Name and activate every calibrated upstream collision mesh.

    The stock XML already carries one collision mesh for each moving component
    but leaves the base visuals non-colliding.  Base visual meshes are copied as
    collision twins so the complete stationary link also participates.
    """

    active: list[str] = []
    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Robot XML has no worldbody")
    link_order = {
        "base": 0,
        "shoulder": 1,
        "upper_arm": 2,
        "lower_arm": 3,
        "wrist": 4,
        "gripper": 5,
        "moving_jaw_so101_v1": 6,
    }
    all_link_bits = sum(1 << (index + 1) for index in link_order.values())
    for body in worldbody.iter("body"):
        body_name = _safe_name(body.get("name", "body"))
        link_index = link_order.get(body.get("name", ""))
        collision_bit = 1 << ((link_index if link_index is not None else 7) + 1)
        mask = 1  # environment/tools use bit 1
        if link_index is not None:
            for other_index in link_order.values():
                if abs(other_index - link_index) > 1:
                    mask |= 1 << (other_index + 1)
        else:
            mask |= all_link_bits
        collision_index = 0
        for geom in body.findall("./geom"):
            if geom.get("class") != "collision":
                continue
            name = f"robot_col_{body_name}_{collision_index:02d}"
            collision_index += 1
            geom.set("name", name)
            geom.set("contype", str(collision_bit))
            geom.set("conaffinity", str(mask))
            geom.set("group", "3")
            active.append(name)
        if body.get("name") != "base" or collision_index:
            continue
        for visual_index, visual in enumerate(body.findall("./geom")):
            if visual.get("class") != "visual":
                continue
            twin = copy.deepcopy(visual)
            name = f"robot_col_base_{visual_index:02d}"
            twin.set("name", name)
            twin.set("class", "collision")
            twin.set("material", visual.get("material", ""))
            twin.set("rgba", "0.8 0.55 0.08 0.12")
            twin.set("contype", str(collision_bit))
            twin.set("conaffinity", str(mask))
            twin.set("group", "3")
            body.append(twin)
            active.append(name)
    if not active:
        raise RuntimeError("No upstream collision geometry was activated")
    return active


def pogo_runtime_geometry_contract() -> dict[str, object]:
    """Return exact nominal geometry plus explicitly unqualified dynamics."""

    signals: dict[str, object] = {}
    for signal in SIGNALS:
        datum = POGO_INSTALLED_DATUMS_M[signal]
        fixed_body_name = f"qc_pogo_{signal}_fixed_shell_body"
        fixed_segments = []
        for source_name, z_bounds, diameter, runtime_suffix in (
            POGO_FIXED_SHELL_SEGMENTS_M
        ):
            z_min, z_max = z_bounds
            fixed_segments.append(
                {
                    "source_segment": source_name,
                    "name": f"qc_col_pogo_{signal}_{runtime_suffix}",
                    "geom_type": "cylinder",
                    "local_pos_m": [0.0, 0.0, (z_min + z_max) / 2.0],
                    "size_m": [diameter / 2.0, (z_max - z_min) / 2.0],
                    "bus_contact_eligible": False,
                }
            )
        signals[signal] = {
            "source_signal": datum["source_signal"],
            "fixed_body": {
                "name": fixed_body_name,
                "parent": "robot_plate_frame",
                "pos_m": [*datum["centre_xy_m"], datum["base_z_m"]],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "fixed_segments": fixed_segments,
            "plunger": {
                "body_name": f"qc_pogo_{signal}_plunger_body",
                "parent": fixed_body_name,
                "local_pos_m": [0.0, 0.0, POGO_FIXED_SHELL_LENGTH_M],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "joint_name": f"qc_pogo_{signal}_plunger",
                "joint_type": "slide",
                "axis": [0.0, 0.0, -1.0],
                "range_m": list(POGO_PLUNGER_COMPRESSION_RANGE_M),
                "geom_name": f"qc_col_pogo_{signal}_plunger",
                "geom_type": "cylinder",
                "geom_local_pos_m": [
                    0.0,
                    0.0,
                    POGO_PLUNGER_MAX_EXPOSED_LENGTH_M / 2.0,
                ],
                "geom_size_m": [
                    POGO_PLUNGER_DIAMETER_M / 2.0,
                    POGO_PLUNGER_MAX_EXPOSED_LENGTH_M / 2.0,
                ],
                "bus_contact_eligible": True,
            },
            "installed_datum": copy.deepcopy(POGO_INSTALLED_DATUMS_MM[signal]),
        }
    blockers = [
        "ground_first_mate_tolerance_stack_unqualified",
        "knurl_press_fit_process_and_pullout_unqualified",
        "installed_electrical_cycle_reliability_unqualified",
        "pogo_mass_properties_unqualified",
        "pogo_spring_force_curve_unqualified",
        "pogo_damping_unqualified",
    ]
    repository_root = Path(__file__).resolve().parents[3]
    return {
        "schema_version": "1.0",
        "source_binding": {
            "ledger_file": {
                "path": str(POGO_AUTHORITY_LEDGER_PATH.relative_to(repository_root)),
                "bytes": POGO_AUTHORITY_LEDGER_BYTES,
                "sha256": POGO_AUTHORITY_LEDGER_SHA256,
            },
            "generator_file": {
                "path": str(POGO_CAD_SOURCE_PATH.relative_to(repository_root)),
                "bytes": POGO_CAD_SOURCE_BYTES,
                "sha256": POGO_CAD_SOURCE_SHA256,
            },
            "canonical_contract_sha256": POGO_SOURCE_CONTRACT_CANONICAL_SHA256,
        },
        "runtime_to_source_signal": dict(POGO_SOURCE_SIGNAL_BY_RUNTIME),
        "signals": signals,
        "dynamics_authority": {
            "geometry_and_datum_authority": True,
            "mass_properties_authority": False,
            "spring_force_curve_authority": False,
            "damping_authority": False,
            "ground_first_mate_tolerance_stack_qualified": False,
            "blockers": blockers,
            "release_ready": False,
        },
        "passed": True,
        "release_ready": False,
    }


@cache
def _require_pogo_runtime_sources() -> None:
    """Fail closed if either source pinned by the runtime geometry drifts."""

    for path, expected_bytes, expected_sha256 in (
        (POGO_CAD_SOURCE_PATH, POGO_CAD_SOURCE_BYTES, POGO_CAD_SOURCE_SHA256),
        (
            POGO_AUTHORITY_LEDGER_PATH,
            POGO_AUTHORITY_LEDGER_BYTES,
            POGO_AUTHORITY_LEDGER_SHA256,
        ),
    ):
        observed_bytes = path.stat().st_size
        if observed_bytes != expected_bytes:
            raise RuntimeError(
                f"pogo runtime source byte-count mismatch for {path.name}: "
                f"expected {expected_bytes}, got {observed_bytes}"
            )
        observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"pogo runtime source hash mismatch for {path.name}: "
                f"expected {expected_sha256}, got {observed_sha256}"
            )


def _add_runtime_pogo_assemblies(frame: ET.Element) -> list[str]:
    """Add four source-derived fixed shells and prismatic plungers."""

    _require_pogo_runtime_sources()
    radius = POGO_PLUNGER_DIAMETER_M / 2.0
    length = POGO_PLUNGER_MAX_EXPOSED_LENGTH_M
    mass = POGO_SIMULATION_ONLY_PLUNGER_MASS_KG
    transverse_inertia = mass * (3.0 * radius**2 + length**2) / 12.0
    axial_inertia = mass * radius**2 / 2.0
    names: list[str] = []
    for signal in SIGNALS:
        datum = POGO_INSTALLED_DATUMS_M[signal]
        fixed_body = ET.SubElement(
            frame,
            "body",
            {
                "name": f"qc_pogo_{signal}_fixed_shell_body",
                "pos": " ".join(
                    f"{value:.12g}"
                    for value in (*datum["centre_xy_m"], datum["base_z_m"])
                ),
            },
        )
        for _, z_bounds, diameter, runtime_suffix in POGO_FIXED_SHELL_SEGMENTS_M:
            z_min, z_max = z_bounds
            name = f"qc_col_pogo_{signal}_{runtime_suffix}"
            _geom(
                fixed_body,
                name=name,
                geom_type="cylinder",
                pos=(0.0, 0.0, (z_min + z_max) / 2.0),
                size=(diameter / 2.0, (z_max - z_min) / 2.0),
                rgba="0.78 0.58 0.16 0.62",
                contype="64",
                conaffinity="31",
                mass="0",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(name)

        plunger_body = ET.SubElement(
            fixed_body,
            "body",
            {
                "name": f"qc_pogo_{signal}_plunger_body",
                "pos": f"0 0 {POGO_FIXED_SHELL_LENGTH_M:.12g}",
            },
        )
        ET.SubElement(
            plunger_body,
            "inertial",
            {
                "pos": f"0 0 {length / 2.0:.12g}",
                "mass": f"{mass:.15g}",
                "fullinertia": " ".join(
                    f"{value:.15g}"
                    for value in (
                        transverse_inertia,
                        transverse_inertia,
                        axial_inertia,
                        0.0,
                        0.0,
                        0.0,
                    )
                ),
            },
        )
        ET.SubElement(
            plunger_body,
            "joint",
            {
                "name": f"qc_pogo_{signal}_plunger",
                "type": "slide",
                "axis": "0 0 -1",
                "range": " ".join(
                    f"{value:.12g}"
                    for value in POGO_PLUNGER_COMPRESSION_RANGE_M
                ),
                "limited": "true",
                "ref": "0",
                "springref": "0",
                "stiffness": f"{POGO_SIMULATION_ONLY_STIFFNESS_N_M:.12g}",
                "damping": f"{POGO_SIMULATION_ONLY_DAMPING_N_S_M:.15g}",
                "frictionloss": "0",
                "armature": "0",
                "solreflimit": "0.0005 1",
                "solimplimit": "0.99 0.9999 0.00001 0.5 2",
            },
        )
        plunger_name = f"qc_col_pogo_{signal}_plunger"
        _geom(
            plunger_body,
            name=plunger_name,
            geom_type="cylinder",
            pos=(0.0, 0.0, length / 2.0),
            size=(radius, length / 2.0),
            rgba="1 0.62 0.05 1",
            contype="64",
            conaffinity="31",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(plunger_name)
    return names


def add_robot_quick_change_interface(
    wrist_output: ET.Element, asset: ET.Element
) -> ET.Element:
    """Install the compact powered robot-side coupling on the bare wrist."""

    frame = ET.SubElement(
        wrist_output,
        "body",
        {"name": "robot_plate_frame", "quat": "0 1 0 0"},
    )
    plate_attributes = {
        "contype": "64",
        "conaffinity": "31",
        "solref": "0.0005 1",
        "solimp": "0.99 0.9999 0.00001",
    }
    plate_rgba = "0.93 0.62 0.04 0.28"
    main_xy = ((-0.024, 0.02355), (-0.024, 0.024))
    # Below the fixed head wells the source is continuous material.
    _add_box_from_bounds(
        frame,
        name="qc_col_robot_plate_lower_base_part_00",
        bounds=(*main_xy, (0.0, 0.0029)),
        rgba=plate_rgba,
        **plate_attributes,
    )
    # Above z=2.9 mm, both Ø6.65 entry wells are real voids.  The two layers
    # around the swept slider track use circumscribed 40-gon voids (<18 um
    # source miss) and no additive well geometry.
    _add_two_well_plate_layer(
        frame,
        asset,
        name_prefix="qc_col_robot_plate_lower_well_partition",
        plate_bounds_xy=main_xy,
        z_bounds=(0.0029, 0.0045),
        well_x_values=(-0.012, 0.012),
        source_well_radius=0.003325,
        semantic_suffix="",
        rgba=plate_rgba,
        **plate_attributes,
    )
    _add_slider_track_plate_layer(
        frame,
        asset,
        name_prefix="qc_col_robot_plate_slider_track_partition",
        plate_bounds_xy=main_xy,
        z_bounds=(0.0045, 0.0065),
        rgba=plate_rgba,
        **plate_attributes,
    )
    _add_two_well_plate_layer(
        frame,
        asset,
        name_prefix="qc_col_robot_plate_upper_well_partition",
        plate_bounds_xy=main_xy,
        z_bounds=(0.0065, 0.0095),
        well_x_values=(-0.012, 0.012),
        source_well_radius=0.003325,
        semantic_suffix="__mating_land",
        first_box_name="qc_col_robot_plate_core__mating_land",
        rgba=plate_rgba,
        **plate_attributes,
    )
    # The full-depth +X cam relief leaves only this exact -Y strip of source
    # plate material outside x=23.55 mm.
    _add_box_from_bounds(
        frame,
        name="qc_col_robot_plate_cam_relief_part_01",
        bounds=((0.02355, 0.024), (-0.024, -0.0165), (0.0, 0.0095)),
        rgba=plate_rgba,
        **plate_attributes,
    )
    # The source electrical wing reaches the core dock's left lower keeper.
    # This narrow edge strip lies outside the PCB pocket and is an exact
    # source-material subset; it supplies the fifth published seated tangency.
    _add_box_from_bounds(
        frame,
        name="qc_col_robot_plate_electrical_wing_edge__keeper_land",
        bounds=((-0.036, -0.035125), (-0.012, 0.012), (0.0, 0.0095)),
        rgba="0.93 0.62 0.04 0.28",
        contype="64",
        conaffinity="31",
        solref="0.0005 1",
        solimp="0.99 0.9999 0.00001",
    )
    # The head-entry wells and slider track are source voids, never additive
    # robot-side collision geoms.  The only moving lock material is the
    # hash-pinned, hole-preserving slider partition below.
    add_positive_lock_slider(frame, asset)
    # The four catalog pins are not generic rounded contacts.  Each one has a
    # distinct installed shoulder datum, a five-segment fixed shell and an
    # independently moving official-diameter plunger.
    _add_runtime_pogo_assemblies(frame)
    ET.SubElement(
        frame,
        "site",
        {
            "name": "robot_mating_face",
            "pos": "0 0 0.0095",
            "size": "0.0015",
            "rgba": "1 0.75 0.1 1",
        },
    )
    return frame


def _add_partitioned_contact_board(body: ET.Element, tool: str) -> list[str]:
    names: list[str] = []
    boxes = (
        ("left", -0.0340, 0.0, 0.000999, 0.011),
        ("right", -0.0270, 0.0, 0.001999, 0.011),
        ("row0", -0.0310, -0.01025, 0.002001, 0.00075),
        ("row1", -0.0310, -0.0050, 0.002001, 0.0005),
        ("row2", -0.0310, 0.0, 0.002001, 0.0005),
        ("row3", -0.0310, 0.0050, 0.002001, 0.0005),
        ("row4", -0.0310, 0.01025, 0.002001, 0.00075),
    )
    for suffix, x_value, y_value, x_half, y_half in boxes:
        name = f"{tool}_contact_board_{suffix}_collision"
        _geom(
            body,
            name=name,
            geom_type="box",
            pos=(x_value, y_value, 0.0005),
            size=(x_half, y_half, 0.0005),
            rgba="0.08 0.42 0.18 0.75",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(name)
    return names


def _add_target_with_bore(body: ET.Element, tool: str, side: str, y_value: float) -> list[str]:
    names: list[str] = []
    # Four boxes form an exact conservative square-minus-square proxy around
    # the M5 countersunk passage.  The surface authority measures the remaining
    # square-vs-round difference; the central bore is never filled.
    for suffix, x, y, sx, sy in (
        ("xneg", -0.00425, y_value, 0.00125, 0.0055),
        ("xpos", 0.00425, y_value, 0.00125, 0.0055),
        ("yneg", 0.0, y_value - 0.00425, 0.0030, 0.00125),
        ("ypos", 0.0, y_value + 0.00425, 0.0030, 0.00125),
    ):
        name = f"{tool}_target_{side}_{suffix}_collision__mating_land"
        _geom(
            body,
            name=name,
            geom_type="box",
            pos=(x, y, 0.0015),
            size=(sx, sy, 0.0015),
            rgba="0.62 0.65 0.68 1",
        )
        names.append(name)
    _geom(
        body,
        name=f"{tool}_m5_screw_{side}_collision__mating_land",
        geom_type="cylinder",
        pos=(0.0, y_value, 0.0045),
        size=(0.00245, 0.0045),
        rgba="0.35 0.37 0.40 1",
    )
    _geom(
        body,
        name=f"{tool}_m5_nut_{side}_collision",
        geom_type="cylinder",
        pos=(0.0, y_value, 0.0080),
        size=(0.0040, 0.0020),
        rgba="0.28 0.30 0.33 1",
    )
    names.extend(
        [
            f"{tool}_m5_screw_{side}_collision__mating_land",
            f"{tool}_m5_nut_{side}_collision",
        ]
    )
    return names


@cache
def _require_positive_lock_hardware_sources() -> None:
    """Fail closed if either source used by the rigid hardware inertia drifts."""

    for path, expected_bytes, expected_sha256 in (
        (
            POSITIVE_LOCK_SHOULDER_SCREW_STEP,
            POSITIVE_LOCK_SHOULDER_SCREW_STEP_BYTES,
            POSITIVE_LOCK_SHOULDER_SCREW_STEP_SHA256,
        ),
        (
            POSITIVE_LOCK_STUD_NUT_STEP,
            POSITIVE_LOCK_STUD_NUT_STEP_BYTES,
            POSITIVE_LOCK_STUD_NUT_STEP_SHA256,
        ),
    ):
        _require_exact_core_export_step(path, expected_bytes, expected_sha256)


def _add_positive_lock_hardware(
    tool_body: ET.Element, asset: ET.Element, tool: str
) -> list[str]:
    """Add one rigid, source-inertial shoulder-screw/nut assembly.

    The collision pieces deliberately contribute zero mass.  Their overlaps
    and faceting therefore cannot perturb the exact pair inertia recomputed
    from the two pinned STEP sources and declared placements.
    """

    _require_positive_lock_hardware_sources()
    hardware_body = ET.SubElement(
        tool_body,
        "body",
        {
            "name": f"tool_{tool}_positive_lock_hardware",
            "pos": "0 0 0",
        },
    )
    ET.SubElement(
        hardware_body,
        "inertial",
        {
            "pos": " ".join(
                f"{value:.15g}" for value in POSITIVE_LOCK_HARDWARE_SOURCE_COM_M
            ),
            "mass": f"{POSITIVE_LOCK_HARDWARE_SOURCE_MASS_KG:.15g}",
            "fullinertia": " ".join(
                f"{value:.15g}"
                for value in POSITIVE_LOCK_HARDWARE_SOURCE_FULL_INERTIA_KG_M2
            ),
        },
    )
    nut_radius = 0.0055 / math.sqrt(3.0)
    nut_polygon = tuple(
        (
            nut_radius * math.cos(index * math.pi / 3.0),
            nut_radius * math.sin(index * math.pi / 3.0),
        )
        for index in range(6)
    )
    nut_mesh = _add_convex_prism_mesh(
        asset,
        name=f"{tool}_positive_lock_nut_source_mesh",
        polygon_xy=nut_polygon,
        z_bounds=(0.0015, 0.0039),
    )
    names: list[str] = []
    for side, x_value in zip(
        ("left", "right"), POSITIVE_LOCK_HARDWARE_STACK_X_M, strict=True
    ):
        _geom(
            hardware_body,
            name=f"{tool}_lock_stud_{side}_shoulder_collision",
            geom_type="cylinder",
            pos=(x_value, 0.0, -0.0025),
            size=(0.0020, 0.0025),
            rgba="0.72 0.75 0.78 1",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        _geom(
            hardware_body,
            name=f"{tool}_lock_stud_{side}_head_collision",
            geom_type="cylinder",
            pos=(x_value, 0.0, -0.00565),
            size=(0.0030, 0.00065),
            rgba="0.72 0.75 0.78 1",
            mass="0",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        _geom(
            hardware_body,
            name=f"{tool}_positive_lock_thread_{side}_collision",
            geom_type="cylinder",
            pos=(x_value, 0.0, 0.0020),
            size=(0.0015, 0.0020),
            rgba="0.42 0.44 0.47 1",
            mass="0",
        )
        _geom(
            hardware_body,
            name=f"{tool}_positive_lock_nut_{side}_collision",
            geom_type="mesh",
            size=None,
            pos=(x_value, 0.0, 0.0),
            mesh=nut_mesh,
            rgba="0.42 0.44 0.47 1",
            mass="0",
        )
        names.extend(
            [
                f"{tool}_lock_stud_{side}_shoulder_collision",
                f"{tool}_lock_stud_{side}_head_collision",
                f"{tool}_positive_lock_thread_{side}_collision",
                f"{tool}_positive_lock_nut_{side}_collision",
            ]
        )
    return names


def add_tool_quick_change_interface(
    body: ET.Element, asset: ET.Element, tool: str
) -> list[str]:
    """Install one complete tool-side interface with semantic collision names."""

    names: list[str] = []
    # Plate pieces retain the released outer mating datums.  The positive-lock
    # studs are tool-owned at x=+/-12 mm, y=0; no robot-side duplicate exists.
    plate_pieces = (
        ("center_yneg", 0.0, -0.0195, 0.004, 0.0055, False),
        ("center_middle", 0.0, 0.0, 0.004, 0.006, False),
        ("center_ypos", 0.0, 0.0195, 0.004, 0.0055, True),
    )
    for suffix, x_value, y_value, x_half, y_half, stop_land in plate_pieces:
        semantics = "__mating_land__locator_land"
        if stop_land or suffix in {"xneg", "xpos"}:
            semantics += "__dock_stop_land"
        plate_name = f"matcha_col_{tool}_plate_{suffix}{semantics}"
        _geom(
            body,
            name=plate_name,
            geom_type="box",
            pos=(x_value, y_value, 0.00475),
            size=(x_half, y_half, 0.00475),
            rgba="0.08 0.35 0.9 0.34",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(plate_name)
    outer_semantics = "__mating_land__locator_land__dock_stop_land"
    lock_void_outer_half = 0.00331
    if tool == "gripper":
        # The core keeper witness reaches the released R4 mm right corner.  A
        # box would create false contact at (x=+/-28,y=+/-25) mm, outside the
        # rounded source boundary.  Sixteen chords per quarter keep the
        # inscribed curve's sagitta below 4.82 um inside the 20 um witness
        # budget.  Matcha rack tools retain their analytic stop boxes below.
        arc_steps = 16
        right_outline: list[tuple[float, float]] = [
            (0.012 + lock_void_outer_half, -0.025),
            (0.024, -0.025),
        ]
        right_outline.extend(
            (
                0.024
                + 0.004
                * math.cos(-math.pi / 2.0 + index * math.pi / (2 * arc_steps)),
                -0.021
                + 0.004
                * math.sin(-math.pi / 2.0 + index * math.pi / (2 * arc_steps)),
            )
            for index in range(1, arc_steps + 1)
        )
        right_outline.append((0.028, 0.021))
        right_outline.extend(
            (
                0.024 + 0.004 * math.cos(index * math.pi / (2 * arc_steps)),
                0.021 + 0.004 * math.sin(index * math.pi / (2 * arc_steps)),
            )
            for index in range(1, arc_steps + 1)
        )
        right_outline.append((0.012 + lock_void_outer_half, 0.025))
        left_outline = [(-x_value, -y_value) for x_value, y_value in right_outline]
        for suffix, outline in (("xneg", left_outline), ("xpos", right_outline)):
            plate_name = f"matcha_col_{tool}_plate_{suffix}{outer_semantics}"
            mesh_name = _add_convex_prism_mesh(
                asset,
                name=f"matcha_{tool}_plate_{suffix}_rounded_source_mesh",
                polygon_xy=tuple(outline),
                z_bounds=(0.0, 0.0095),
            )
            _geom(
                body,
                name=plate_name,
                geom_type="mesh",
                size=None,
                mesh=mesh_name,
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(plate_name)
    else:
        for suffix, x_bounds in (
            ("xneg", (-0.028, -0.012 - lock_void_outer_half)),
            ("xpos", (0.012 + lock_void_outer_half, 0.028)),
        ):
            plate_name = f"matcha_col_{tool}_plate_{suffix}{outer_semantics}"
            _add_box_from_bounds(
                body,
                name=plate_name,
                bounds=(x_bounds, (-0.025, 0.025), (0.0, 0.0095)),
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(plate_name)

    # Partition the material surrounding each x=+/-12 mm lock-stud feature.
    # The lower 1.5 mm layer preserves the Ø3.2 through-bore as a circumscribed
    # 24-gon (<15 um miss); the upper layer preserves the AF5.7 nut pocket as
    # an outward-offset exact hexagon.  No plate proxy overlaps the thread or
    # captive nut hardware installed below.
    inner_semantics = "__mating_land__locator_land"
    for side, center_x in (("left", -0.012), ("right", 0.012)):
        if center_x < 0.0:
            inner_x_bounds = (-0.012 + lock_void_outer_half, -0.004)
        else:
            inner_x_bounds = (0.004, 0.012 - lock_void_outer_half)
        inner_name = (
            f"matcha_col_{tool}_plate_{side}_lock_inner"
            f"{inner_semantics}__dock_stop_land"
        )
        _add_box_from_bounds(
            body,
            name=inner_name,
            bounds=(inner_x_bounds, (-0.025, 0.025), (0.0, 0.0095)),
            rgba="0.08 0.35 0.9 0.34",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(inner_name)
        for y_side, y_bounds in (
            ("yneg", (-0.025, -lock_void_outer_half)),
            ("ypos", (lock_void_outer_half, 0.025)),
        ):
            stop_semantic = "__dock_stop_land" if y_side == "ypos" else ""
            strip_name = (
                f"matcha_col_{tool}_plate_{side}_lock_{y_side}"
                f"{inner_semantics}{stop_semantic}"
            )
            _add_box_from_bounds(
                body,
                name=strip_name,
                bounds=(
                    (center_x - lock_void_outer_half, center_x + lock_void_outer_half),
                    y_bounds,
                    (0.0, 0.0095),
                ),
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
            names.append(strip_name)
        bore_sides = 24
        bore_void_radius = 0.0016 / math.cos(math.pi / bore_sides) + 1.0e-6
        names.extend(
            _add_square_minus_regular_void_partition(
                body,
                asset,
                name_prefix=f"matcha_col_{tool}_plate_{side}_lock_bore_part",
                center_xy=(center_x, 0.0),
                outer_half_size=lock_void_outer_half,
                void_vertex_radius=bore_void_radius,
                sides=bore_sides,
                z_bounds=(0.0, 0.0015),
                rgba="0.08 0.35 0.9 0.34",
                semantic_suffix=inner_semantics,
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        )
        nut_void_radius = (0.0057 / 2.0 + 1.0e-5) / math.cos(math.pi / 6.0)
        names.extend(
            _add_square_minus_regular_void_partition(
                body,
                asset,
                name_prefix=f"matcha_col_{tool}_plate_{side}_lock_nut_part",
                center_xy=(center_x, 0.0),
                outer_half_size=lock_void_outer_half,
                void_vertex_radius=nut_void_radius,
                sides=6,
                z_bounds=(0.0015, 0.0095),
                rgba="0.08 0.35 0.9 0.34",
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        )
    # Exact source-material edge outside the recessed contact-board pocket.
    # On the stock-gripper adapter it bears on both left keeper rails; the
    # same source feature is present on every generic tool plate.
    wing_name = (
        f"matcha_col_{tool}_plate_electrical_wing_edge"
        "__mating_land__keeper_land"
    )
    _add_box_from_bounds(
        body,
        name=wing_name,
        bounds=((-0.036, -0.035125), (-0.012, 0.012), (0.0, 0.0095)),
        rgba="0.08 0.35 0.9 0.34",
        solref="0.0005 1",
        solimp="0.99 0.9999 0.00001",
    )
    names.append(wing_name)
    for signal in SIGNALS:
        pad_name = f"{tool}_pad_{signal}_collision"
        _geom(
            body,
            name=pad_name,
            geom_type="cylinder",
            # The copper disk spans tool-local z=[-0.05, 0] mm.  Its exposed
            # face therefore lands at robot-source z=9.45 mm when attached.
            pos=(-0.031, SIGNAL_Y_M[signal], -0.000025),
            size=(0.0020, 0.000025),
            rgba="0.95 0.55 0.05 1",
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
        names.append(pad_name)
    names.extend(_add_partitioned_contact_board(body, tool))
    # The two M5 target stacks sit outside the lock-stud bores; their
    # 5.5 mm outer envelope remains inside the 25 mm plate datum.
    for side, y_value in (("neg", -0.0190), ("pos", 0.0190)):
        names.extend(_add_target_with_bore(body, tool, side, y_value))
    names.extend(_add_positive_lock_hardware(body, asset, tool))
    ET.SubElement(
        body,
        "site",
        {
            "name": f"{tool}_mating_face",
            "pos": "0 0 0",
            "size": "0.0015",
            "rgba": "0.2 0.65 1 1",
        },
    )
    return names


@cache
def _require_core_cam_runtime_source() -> None:
    observed_bytes = POGO_CAD_SOURCE_PATH.stat().st_size
    if observed_bytes != POGO_CAD_SOURCE_BYTES:
        raise RuntimeError(
            "core cam source byte-count mismatch: "
            f"expected {POGO_CAD_SOURCE_BYTES}, got {observed_bytes}"
        )
    observed_sha256 = hashlib.sha256(POGO_CAD_SOURCE_PATH.read_bytes()).hexdigest()
    if observed_sha256 != POGO_CAD_SOURCE_SHA256:
        raise RuntimeError(
            "core cam source hash mismatch: "
            f"expected {POGO_CAD_SOURCE_SHA256}, got {observed_sha256}"
        )


def _add_positive_lock_cam_geoms(
    dock: ET.Element,
    asset: ET.Element,
    *,
    tool: str,
    rgba: str,
) -> list[str]:
    """Install one dock-local exact nonoverlapping source cam partition."""

    _require_core_cam_runtime_source()
    geom_names = positive_lock_cam_collision_geom_names(tool)
    names: list[str] = []
    main_mesh_name = _add_convex_prism_mesh(
        asset,
        name=f"dock_{tool}_positive_lock_cam_source_mesh",
        polygon_xy=CORE_DOCK_CAM_POLYGON_M,
        z_bounds=CORE_DOCK_CAM_Z_BOUNDS_M,
    )
    _geom(
        dock,
        name=geom_names[0],
        geom_type="mesh",
        size=None,
        mesh=main_mesh_name,
        rgba=rgba,
        mass="0",
    )
    names.append(geom_names[0])

    lead_vertices, lead_faces = _core_cam_axial_lead_mesh_geometry()
    lead_mesh_name = _add_convex_polyhedron_mesh(
        asset,
        name=f"dock_{tool}_positive_lock_cam_axial_lead_source_mesh",
        vertices=lead_vertices,
        faces=lead_faces,
    )
    _geom(
        dock,
        name=geom_names[1],
        geom_type="mesh",
        size=None,
        mesh=lead_mesh_name,
        rgba=rgba,
        mass="0",
    )
    names.append(geom_names[1])

    _add_box_from_bounds(
        dock,
        name=geom_names[2],
        bounds=CORE_DOCK_CAM_HOLD_FINGER_BOUNDS_M,
        rgba=rgba,
        mass="0",
    )
    names.append(geom_names[2])
    for name, bounds in zip(
        geom_names[3:],
        CORE_DOCK_CAM_OUTER_ROOT_REMAINDER_BOUNDS_M,
        strict=True,
    ):
        _add_box_from_bounds(
            dock,
            name=name,
            bounds=bounds,
            rgba=rgba,
            mass="0",
        )
        names.append(name)
    if tuple(names) != geom_names:
        raise RuntimeError(f"{tool} cam runtime component inventory drifted")
    return names


def _add_core_dock_floor_support_proxy(dock: ET.Element) -> list[str]:
    """Install the source-bound, nonoverlapping, massless support boxes."""

    _require_core_dock_runtime_sources()
    names: list[str] = []
    for _role, name, bounds in CORE_DOCK_SUPPORT_PROXY_COMPONENTS_M:
        _add_box_from_bounds(
            dock,
            name=name,
            bounds=bounds,
            rgba="0.16 0.18 0.21 1",
            mass="0",
        )
        names.append(name)
    if tuple(names) != CORE_DOCK_SUPPORT_PROXY_GEOM_NAMES:
        raise RuntimeError("rolled core dock support proxy inventory drifted")
    return names


def add_supported_dock(
    worldbody: ET.Element,
    asset: ET.Element,
    tool: str,
    *,
    position: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    rgba: str,
) -> ET.Element:
    """Add a floor-connected passive rack with narrow functional contacts."""

    if tool not in {"gripper", "spoon", "whisk"}:
        raise ValueError(f"unsupported dock source contract for {tool!r}")
    x_value, y_value, z_value = position
    if tool == "gripper":
        _require_core_dock_runtime_sources()
        if tuple(position) != CORE_DOCK_WORLD_POS_M:
            raise RuntimeError(
                "core dock position must equal the hash-bound rolled source "
                f"pose {CORE_DOCK_WORLD_POS_M}, got {position}"
            )
        if tuple(quat) != CORE_DOCK_WORLD_QUAT_WXYZ:
            raise RuntimeError(
                "core dock quaternion must equal the hash-bound rolled source "
                f"pose {CORE_DOCK_WORLD_QUAT_WXYZ}, got {quat}"
            )
    else:
        # The separate matcha rack retains its existing recovered post and
        # rear-anchor proxy.  Its spoon/whisk source datums are intentionally
        # unaffected by the core-gripper rolled-dock checkpoint.
        anchor_corners: list[tuple[float, float, float]] = []
        for local_x in (-0.020, 0.020):
            for local_y in (-0.075, -0.055):
                for local_z in (-0.005, 0.005):
                    rotated = _rotate_by_quat((local_x, local_y, local_z), quat)
                    anchor_corners.append(
                        (
                            x_value + rotated[0],
                            y_value + rotated[1],
                            z_value + rotated[2],
                        )
                    )
        anchor_world = min(
            anchor_corners, key=lambda point: (point[2], point[0], point[1])
        )
        support_height = max(0.002, anchor_world[2])
        support = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"dock_{tool}_support",
                "pos": f"{anchor_world[0]:.9g} {anchor_world[1]:.9g} 0",
            },
        )
        _geom(
            support,
            name=f"dock_{tool}_support_collision",
            geom_type="cylinder",
            pos=(0.0, 0.0, support_height / 2.0),
            size=(0.012, support_height / 2.0),
            rgba="0.16 0.18 0.21 1",
        )
    dock_position = (
        " ".join(f"{value:.17g}" for value in position)
        if tool == "gripper"
        else f"{x_value:.9g} {y_value:.9g} {z_value:.9g}"
    )
    dock_quat = " ".join(
        f"{value:.17g}" if tool == "gripper" else f"{value:.12g}"
        for value in quat
    )
    dock = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"dock_{tool}",
            "pos": dock_position,
            "quat": dock_quat,
        },
    )
    if tool == "gripper":
        _add_core_dock_floor_support_proxy(dock)
    else:
        _geom(
            dock,
            name=f"dock_{tool}_support_anchor_collision",
            geom_type="box",
            pos=(0.0, -0.065, 0.0),
            size=(0.020, 0.010, 0.005),
            rgba="0.16 0.18 0.21 1",
        )
    if tool == "gripper":
        x_bounds, y_bounds, z_bounds = CORE_DOCK_STOP_BOUNDS_M
        hole_z = CORE_DOCK_STOP_HOLES_M[0][1]
        radius = CORE_DOCK_STOP_HOLES_M[0][2]
        middle_min = hole_z - radius
        middle_max = hole_z + radius
        stop_bounds: list[tuple[tuple[float, float], ...]] = [
            (x_bounds, y_bounds, (z_bounds[0], middle_min)),
        ]
        slice_count = round(
            (middle_max - middle_min) / CORE_STOP_HOLE_PARTITION_STEP_M
        )
        for slice_index in range(slice_count):
            z_min = middle_min + slice_index * CORE_STOP_HOLE_PARTITION_STEP_M
            z_max = min(
                middle_max, z_min + CORE_STOP_HOLE_PARTITION_STEP_M
            )
            nearest_delta = (
                0.0
                if z_min <= hole_z <= z_max
                else min(abs(z_min - hole_z), abs(z_max - hole_z))
            )
            hole_half_width = math.sqrt(
                max(0.0, radius * radius - nearest_delta * nearest_delta)
            )
            left_x, right_x = (hole[0] for hole in CORE_DOCK_STOP_HOLES_M)
            for interval in (
                (x_bounds[0], left_x - hole_half_width),
                (left_x + hole_half_width, right_x - hole_half_width),
                (right_x + hole_half_width, x_bounds[1]),
            ):
                stop_bounds.append((interval, y_bounds, (z_min, z_max)))
        stop_bounds.append((x_bounds, y_bounds, (middle_max, z_bounds[1])))
        for index, bounds in enumerate(stop_bounds):
            _add_box_from_bounds(
                dock,
                name=(
                    f"dock_{tool}_qc_col_dock_stop_part_{index:03d}"
                    "__dock_stop_land"
                ),
                bounds=bounds,
                rgba=rgba,
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
    elif tool in {"spoon", "whisk"}:
        _add_box_from_bounds(
            dock,
            name=f"dock_{tool}_qc_col_dock_stop",
            bounds=MATCHA_DOCK_STOP_BOUNDS_M,
            rgba=rgba,
            solref="0.0005 1",
            solimp="0.99 0.9999 0.00001",
        )
    if tool == "gripper":
        # Exact core-dock keeper solids.  At the seated pose the stock plate
        # has four zero-volume tangencies (two lower X lands, two upper Z
        # lands); the robot electrical wing adds the fifth left-lower contact.
        keeper_bounds = {
            "left_lower": ((-0.043, -0.036), (-0.040, 0.036), (-0.003, 0.0)),
            "left_upper": ((-0.041, -0.033), (-0.040, 0.036), (0.0095, 0.0125)),
            "right_lower": ((0.028, 0.035), (-0.040, 0.036), (-0.003, 0.0)),
            "right_upper": ((0.025, 0.033), (-0.040, 0.036), (0.0095, 0.0125)),
        }
        for keeper_name, bounds in keeper_bounds.items():
            _add_box_from_bounds(
                dock,
                name=f"dock_gripper_keeper_{keeper_name}_collision",
                bounds=bounds,
                rgba=rgba,
                solref="0.0005 1",
                solimp="0.99 0.9999 0.00001",
            )
        for side, bounds in (
            ("left", ((-0.045, -0.041), (-0.040, 0.036), (-0.003, 0.0125))),
            ("right", ((0.033, 0.037), (-0.040, 0.036), (-0.003, 0.0125))),
        ):
            _add_box_from_bounds(
                dock,
                name=f"dock_gripper_wall_{side}_collision",
                bounds=bounds,
                rgba=rgba,
            )
    else:
        # The recovered spoon/whisk guide cheeks remain isolated from the core
        # witness; their source-faithful rack replacement belongs with the
        # pending CAD-derived payload/rack checkpoint.
        for side, y_rail in (("left", -0.033), ("right", 0.033)):
            _geom(
                dock,
                name=f"dock_{tool}_rail_{side}_collision",
                geom_type="box",
                pos=(0.0, y_rail, 0.010),
                size=(0.034, 0.003, 0.010),
                rgba=rgba,
            )
    _add_positive_lock_cam_geoms(
        dock,
        asset,
        tool=tool,
        rgba="0.12 0.75 0.35 1",
    )
    ET.SubElement(
        dock,
        "site",
        {
            "name": f"dock_{tool}_target",
            "pos": "0 0 0",
            "size": "0.002",
            "rgba": rgba,
        },
    )
    return dock


def is_dock_stop_collision_name(tool: str, name: str) -> bool:
    """Return whether *name* is an exact member of one dock's stop family."""

    if tool == "gripper":
        return re.fullmatch(
            r"dock_gripper_qc_col_dock_stop_part_[0-9]{3}__dock_stop_land",
            name,
        ) is not None
    if tool in {"spoon", "whisk"}:
        return name == f"dock_{tool}_qc_col_dock_stop"
    return False


def collision_geom_names(root: ET.Element) -> list[str]:
    return sorted(
        geom.get("name", "")
        for geom in root.iter("geom")
        if geom.get("contype", "1") != "0" and geom.get("name")
    )


def require_unique_names(names: Iterable[str]) -> None:
    values = list(names)
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate collision geom names: {duplicates[:8]}")
