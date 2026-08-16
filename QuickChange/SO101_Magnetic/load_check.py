#!/usr/bin/env python3
"""Print the transparent first-order load envelope for the quick changer.

This is a sizing check, not a strength certification.  Printed-material
properties, fastener installation, wear, impact, and arm configuration still
require physical proof testing.
"""

from __future__ import annotations

import json


KG_CM_TO_N_M = 0.0980665
RATED_TORQUE_KG_CM = 10.0
STALL_TORQUE_KG_CM = 30.0
MAGNET_FORCE_EACH_N = 29.4
MAGNET_LEVER_ARM_M = 0.016
INTERFACE_EDGE_RADIUS_M = 0.024


def main() -> None:
    rated = RATED_TORQUE_KG_CM * KG_CM_TO_N_M
    stall = STALL_TORQUE_KG_CM * KG_CM_TO_N_M
    pair_pull = 2 * MAGNET_FORCE_EACH_N
    ideal_magnet_moment = pair_pull * MAGNET_LEVER_ARM_M
    edge_reaction = stall / INTERFACE_EDGE_RADIUS_M
    result = {
        "official_STS3215_rated_torque_Nm": round(rated, 4),
        "official_STS3215_peak_stall_torque_Nm": round(stall, 4),
        "selected_proof_moment_Nm": round(stall, 4),
        "proof_over_rated_ratio": round(stall / rated, 1),
        "two_magnets_catalog_pull_N_ideal_zero_gap": round(pair_pull, 1),
        "magnet_only_ideal_pry_moment_Nm": round(ideal_magnet_moment, 4),
        "magnet_only_passes_rated_torque_check": ideal_magnet_moment >= rated,
        "positive_lock_edge_reaction_at_proof_moment_N": round(edge_reaction, 1),
        "approx_reaction_per_two_shoulder_studs_N": round(edge_reaction / 2, 1),
        "interpretation": (
            "Magnets alone are below even the servo rated-torque reference, so "
            "the positive lock is required. Bench proof-test the assembled, "
            "printed interface to 2.94 N.m in both bending axes before robot use."
        ),
        "payload_warning": (
            "The official SO-101 documentation does not publish a whole-arm "
            "payload rating; servo torque is not an arm payload rating."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
