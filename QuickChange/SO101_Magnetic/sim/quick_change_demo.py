#!/usr/bin/env python3
"""Validate powered capture, positive lock, rack unlock, and release in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
SCENE = HERE / "quick_change_scene.xml"
COUPLED_Z = -0.0095


def smoothstep(value: float) -> float:
    value = np.clip(value, 0.0, 1.0)
    return float(value * value * (3 - 2 * value))


def lerp(start: np.ndarray, end: np.ndarray, value: float) -> np.ndarray:
    return start + (end - start) * smoothstep(value)


def relative_pose(data: mujoco.MjData, body1: int, body2: int) -> np.ndarray:
    rotation1 = data.xmat[body1].reshape(3, 3)
    relative_position = rotation1.T @ (data.xpos[body2] - data.xpos[body1])
    quat1 = data.xquat[body1]
    quat2 = data.xquat[body2]
    inverse1 = np.array([quat1[0], -quat1[1], -quat1[2], -quat1[3]])
    relative_quat = np.empty(4)
    mujoco.mju_mulQuat(relative_quat, inverse1, quat2)
    return np.concatenate((relative_position, relative_quat))


def set_weld(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    active: bool,
    update_pose: bool = False,
) -> None:
    equality_id = model.equality(name).id
    if active and update_pose:
        body1 = model.eq_obj1id[equality_id]
        body2 = model.eq_obj2id[equality_id]
        model.eq_data[equality_id, 3:10] = relative_pose(data, body1, body2)
    data.eq_active[equality_id] = active
    mujoco.mj_forward(model, data)


def robot_position(sim_time: float) -> tuple[np.ndarray, str]:
    far = np.array([0.0, 0.0, -0.055])
    coupled = np.array([0.0, 0.0, COUPLED_Z])
    outside = np.array([0.0, -0.075, COUPLED_Z])
    working = np.array([0.025, -0.095, 0.040])

    if sim_time < 1.5:
        return lerp(far, coupled, sim_time / 1.5), "straight approach"
    if sim_time < 2.0:
        return coupled, "magnetic capture + pogo contact"
    if sim_time < 3.0:
        return lerp(coupled, outside, sim_time - 2.0), "leave rack; spring closes positive lock"
    if sim_time < 4.0:
        return lerp(outside, working, sim_time - 3.0), "powered gripper motion"
    if sim_time < 5.0:
        return lerp(working, outside, sim_time - 4.0), "gripper torque off; return to rack"
    if sim_time < 6.0:
        return lerp(outside, coupled, sim_time - 5.0), "insert in rack; cam opens lock"
    if sim_time < 6.5:
        return coupled, "rack captures tool"
    if sim_time < 8.0:
        return lerp(coupled, far, (sim_time - 6.5) / 1.5), "contacts and magnets release"
    return far, "complete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot_mocap = model.body("robot_plate").mocapid[0]
    magnetic_attached = False
    positive_locked = False
    docked = True
    bus_connected = False
    gripper_torque_enabled = False
    last_label = None

    while data.time < 8.3:
        tick = time.perf_counter()
        position, label = robot_position(data.time)
        data.mocap_pos[robot_mocap] = position
        data.mocap_quat[robot_mocap] = np.array([1.0, 0.0, 0.0, 0.0])

        if label != last_label:
            print(f"[{data.time:4.1f}s] {label}")
            last_label = label

        if 1.5 <= data.time < 2.15 and not magnetic_attached and not positive_locked:
            set_weld(model, data, "magnetic_attach", True, update_pose=True)
            set_weld(model, data, "tool_in_dock", False)
            magnetic_attached, docked = True, False
            bus_connected = True
            gripper_torque_enabled = False
        if 2.15 <= data.time < 5.82 and magnetic_attached and not positive_locked:
            set_weld(model, data, "positive_lock", True, update_pose=True)
            set_weld(model, data, "magnetic_attach", False)
            magnetic_attached, positive_locked = False, True
            print(f"[{data.time:4.1f}s] positive lock engaged (magnets remain as face preload)")
        if 3.0 <= data.time < 4.0 and bus_connected and positive_locked:
            gripper_torque_enabled = True
        if data.time >= 4.0:
            gripper_torque_enabled = False
        if 5.82 <= data.time < 6.5 and positive_locked:
            set_weld(model, data, "magnetic_attach", True, update_pose=True)
            set_weld(model, data, "positive_lock", False)
            magnetic_attached, positive_locked = True, False
            print(f"[{data.time:4.1f}s] dock cam opened positive lock")
        if data.time >= 6.5 and not docked:
            set_weld(model, data, "tool_in_dock", True, update_pose=False)
            set_weld(model, data, "magnetic_attach", False)
            magnetic_attached, docked = False, True
            bus_connected = False

        mujoco.mj_step(model, data)
        if args.realtime:
            remaining = model.opt.timestep - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)

    tool_position = data.body("tool_plate").xpos.copy()
    success = (
        np.linalg.norm(tool_position) < 0.002
        and not data.eq_active[model.equality("magnetic_attach").id]
        and not data.eq_active[model.equality("positive_lock").id]
        and data.eq_active[model.equality("tool_in_dock").id]
        and not bus_connected
    )
    result = {
        "success": bool(success),
        "tool_position": np.round(tool_position, 6).tolist(),
        "magnetic_attach_active": bool(data.eq_active[model.equality("magnetic_attach").id]),
        "positive_lock_active": bool(data.eq_active[model.equality("positive_lock").id]),
        "tool_in_dock_active": bool(data.eq_active[model.equality("tool_in_dock").id]),
        "bus_connected": bus_connected,
        "gripper_torque_enabled": gripper_torque_enabled,
        "sim_time": round(float(data.time), 3),
    }
    print(json.dumps(result, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
