#!/usr/bin/env python3
"""Scripted SO-101 two-box stacking demo for MuJoCo.

The controller uses damped least-squares position IK for the gripper site. A
temporary weld is activated after each close command to make this first task
prototype deterministic while we iterate on gripper and tool-changer geometry.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
SCENE_PATH = HERE / "box_stack_scene.xml"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
OPEN_GRIPPER = 1.25
CLOSED_GRIPPER = -0.05
BOX_HALF_SIZE = 0.026
WORKTOP_Z = 0.030
TARGET_XY = np.array([0.27, 0.105])


@dataclass(frozen=True)
class Phase:
    name: str
    target: np.ndarray | None
    gripper: float
    dwell: float = 0.20
    max_time: float = 3.0
    attach: str | None = None
    detach: str | None = None
    home: bool = False


class BoxStackController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.site_id = model.site("gripperframe").id
        self.arm_dofs = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self.arm_qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self.arm_actuators = np.array([model.actuator(name).id for name in ARM_JOINTS])
        self.gripper_actuator = model.actuator("gripper").id
        self.home_q = np.array([0.0, -0.35, 0.75, -0.30, 0.0])
        self.phase_index = 0
        self.phase_started = data.time
        self.action_done = False
        self.done = False
        self.phases = self._make_phases()
        self._enter_phase(self.phases[0])

    @staticmethod
    def _make_phases() -> list[Phase]:
        bottom_z = WORKTOP_Z + BOX_HALF_SIZE
        top_z = WORKTOP_Z + 3 * BOX_HALF_SIZE
        red_xy = np.array([0.22, -0.105])
        blue_xy = np.array([0.33, -0.095])

        def xyz(xy: np.ndarray, z: float) -> np.ndarray:
            return np.array([xy[0], xy[1], z], dtype=float)

        phases: list[Phase] = [Phase("home", None, OPEN_GRIPPER, home=True, dwell=0.35)]
        for color, pick_xy, place_z in (
            ("red", red_xy, bottom_z),
            ("blue", blue_xy, top_z),
        ):
            phases.extend(
                [
                    Phase(f"above_{color}", xyz(pick_xy, 0.17), OPEN_GRIPPER),
                    Phase(f"reach_{color}", xyz(pick_xy, bottom_z + 0.008), OPEN_GRIPPER, dwell=0.15),
                    Phase(
                        f"grasp_{color}",
                        xyz(pick_xy, bottom_z + 0.008),
                        CLOSED_GRIPPER,
                        dwell=0.40,
                        attach=color,
                    ),
                    Phase(f"lift_{color}", xyz(pick_xy, 0.18), CLOSED_GRIPPER),
                    Phase(f"carry_{color}", xyz(TARGET_XY, 0.19), CLOSED_GRIPPER),
                    Phase(f"place_{color}", xyz(TARGET_XY, place_z + 0.008), CLOSED_GRIPPER, dwell=0.30),
                    Phase(
                        f"release_{color}",
                        xyz(TARGET_XY, place_z + 0.008),
                        OPEN_GRIPPER,
                        dwell=0.45,
                        detach=color,
                    ),
                    Phase(f"retreat_{color}", xyz(TARGET_XY, 0.19), OPEN_GRIPPER),
                ]
            )
        phases.append(Phase("finished", None, OPEN_GRIPPER, home=True, dwell=0.75, max_time=4.0))
        return phases

    @property
    def phase(self) -> Phase:
        return self.phases[self.phase_index]

    def _enter_phase(self, phase: Phase) -> None:
        self.phase_started = self.data.time
        self.action_done = False
        print(f"[{self.data.time:6.2f}s] {phase.name}")

    def _relative_pose(self, body1: int, body2: int) -> np.ndarray:
        rotation1 = self.data.xmat[body1].reshape(3, 3)
        relative_position = rotation1.T @ (self.data.xpos[body2] - self.data.xpos[body1])
        quat1 = self.data.xquat[body1]
        quat2 = self.data.xquat[body2]
        inverse1 = np.array([quat1[0], -quat1[1], -quat1[2], -quat1[3]])
        relative_quat = np.empty(4)
        mujoco.mju_mulQuat(relative_quat, inverse1, quat2)
        return np.concatenate((relative_position, relative_quat))

    def _set_grasp(self, color: str, active: bool) -> None:
        equality_id = self.model.equality(f"hold_{color}_box").id
        if active:
            gripper_body = self.model.body("gripper").id
            box_body = self.model.body(f"{color}_box").id
            # Weld equality data stores a 3D anchor first, followed by the
            # seven-value relative pose and the torque scale.
            self.model.eq_data[equality_id, 3:10] = self._relative_pose(gripper_body, box_body)
        self.data.eq_active[equality_id] = int(active)
        mujoco.mj_forward(self.model, self.data)

    def _position_ik_command(self, target: np.ndarray) -> float:
        error = target - self.data.site_xpos[self.site_id]
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.site_id)
        jacobian = jacobian[:, self.arm_dofs]

        damping = 0.035
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3), error
        )
        step = np.clip(step, -0.045, 0.045)
        command = self.data.qpos[self.arm_qpos] + step

        # A weak posture bias keeps the position-only IK away from folded and
        # joint-limit configurations without fighting the Cartesian target.
        command += 0.002 * (self.home_q - command)
        for index, qpos_address in enumerate(self.arm_qpos):
            joint_id = self.model.dof_jntid[self.arm_dofs[index]]
            command[index] = np.clip(command[index], *self.model.jnt_range[joint_id])
        self.data.ctrl[self.arm_actuators] = command
        return float(np.linalg.norm(error))

    def _home_command(self) -> float:
        self.data.ctrl[self.arm_actuators] = self.home_q
        return float(np.max(np.abs(self.data.qpos[self.arm_qpos] - self.home_q)))

    def update(self) -> None:
        if self.done:
            return

        phase = self.phase
        elapsed = self.data.time - self.phase_started
        error = self._home_command() if phase.home else self._position_ik_command(phase.target)
        self.data.ctrl[self.gripper_actuator] = phase.gripper

        if phase.attach and not self.action_done and elapsed >= 0.22:
            distance = np.linalg.norm(
                self.data.site_xpos[self.site_id] - self.data.body(f"{phase.attach}_box").xpos
            )
            if distance < 0.065:
                self._set_grasp(phase.attach, True)
                self.action_done = True

        if phase.detach and not self.action_done and elapsed >= 0.16:
            self._set_grasp(phase.detach, False)
            self.action_done = True

        settled = error < (0.012 if phase.target is not None else 0.035)
        action_ready = (not phase.attach and not phase.detach) or self.action_done
        if (settled and action_ready and elapsed >= phase.dwell) or elapsed >= phase.max_time:
            if self.phase_index == len(self.phases) - 1:
                self.done = True
                return
            self.phase_index += 1
            self._enter_phase(self.phase)


def stack_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    red = data.body("red_box").xpos.copy()
    blue = data.body("blue_box").xpos.copy()
    xy_tolerance = 0.045
    red_target_z = WORKTOP_Z + BOX_HALF_SIZE
    blue_target_z = WORKTOP_Z + 3 * BOX_HALF_SIZE
    success = (
        np.linalg.norm(red[:2] - TARGET_XY) < xy_tolerance
        and np.linalg.norm(blue[:2] - TARGET_XY) < xy_tolerance
        and abs(red[2] - red_target_z) < 0.025
        and abs(blue[2] - blue_target_z) < 0.035
        and blue[2] > red[2] + 0.035
    )
    return {
        "success": bool(success),
        "red_box_position": np.round(red, 4).tolist(),
        "blue_box_position": np.round(blue, 4).tolist(),
        "target_xy": TARGET_XY.tolist(),
        "sim_time": round(float(data.time), 3),
    }


def save_preview(model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> None:
    renderer = mujoco.Renderer(model, height=540, width=720)
    renderer.update_scene(data, camera="overview")
    pixels = renderer.render()
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to use --save-preview") from exc
    Image.fromarray(pixels).save(path)
    renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--realtime", action="store_true", help="Pace a headless run at wall-clock speed")
    parser.add_argument("--max-seconds", type=float, default=45.0, help="Maximum simulated run time")
    parser.add_argument("--save-preview", type=Path, help="Write a PNG of the final state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = data.qpos[: model.nu]
    mujoco.mj_forward(model, data)
    controller = BoxStackController(model, data)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    try:
        while data.time < args.max_seconds and not controller.done:
            tick = time.perf_counter()
            controller.update()
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            if args.realtime or viewer is not None:
                remaining = model.opt.timestep - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)

        # Let released boxes settle before evaluating the stack.
        for _ in range(400):
            controller.update()
            mujoco.mj_step(model, data)
            if viewer is not None and viewer.is_running():
                viewer.sync()

        metrics = stack_metrics(model, data)
        print(json.dumps(metrics, indent=2))
        if args.save_preview:
            save_preview(model, data, args.save_preview)
            print(f"Saved preview to {args.save_preview}")
        return 0 if metrics["success"] else 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
