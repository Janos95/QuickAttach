#!/usr/bin/env python3
"""Render the complete Matcha tool-change story from one fixed scene camera.

The video uses the compiled ``matcha_workflow_demo`` MuJoCo geometry and the
deterministic timeline in ``matcha_workflow_showcase``.  A CPU z-buffer keeps
the result reproducible on headless machines.  The overlay pitchers and bowl
contents are presentation-only task props; the quick-change, arm, tools,
docks, sieve, stations, and powered whisk are the production compiled model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import matcha_workflow_demo as demo
from matcha_workflow_showcase import (
    SHOWCASE_CAMERA,
    ShowcaseState,
    showcase_duration_s,
    showcase_segments,
    showcase_state_at,
    showcase_summary,
)
from render_gripper_change_video import (
    _font,
    build_render_geometry,
    render_view,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE.parent / "exports" / "so101_matcha_complete_workflow.mp4"
)
DEFAULT_REPORT = (
    HERE.parent / "exports" / "so101_matcha_complete_workflow_report.json"
)
PITCHER_HOME_POSITIONS = {
    "hot_water": (-0.28, -0.08, 0.115),
    "milk": (-0.28, 0.20, 0.115),
}
RECIPE_VISIBLE_BODY = {
    "empty_bowl": None,
    "hot_water_added": "showcase_hot_water",
    "matcha_dosed": "showcase_matcha",
    "whisked": "showcase_matcha",
    "milk_added": "showcase_foam",
    "complete": "showcase_foam",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_pitcher(
    worldbody: ET.Element,
    name: str,
    position: tuple[float, float, float],
    rgba: str,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": f"showcase_{name}_pitcher", "pos": " ".join(map(str, position))},
    )
    ET.SubElement(body, "freejoint", {"name": f"showcase_{name}_pitcher_free"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"showcase_{name}_pitcher_body",
            "type": "cylinder",
            "pos": "0 0 0.025",
            "size": "0.026 0.045",
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    for suffix, pos, size in (
        ("handle_top", "0.039 0 0.062", "0.018 0.006 0.005"),
        ("handle_side", "0.054 0 0.035", "0.005 0.006 0.025"),
        ("handle_bottom", "0.039 0 0.010", "0.018 0.006 0.005"),
        ("spout", "-0.032 0 0.063", "0.016 0.014 0.006"),
    ):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"showcase_{name}_{suffix}",
                "type": "box",
                "pos": pos,
                "size": size,
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )


def _add_recipe_visual(
    worldbody: ET.Element,
    name: str,
    z: float,
    rgba: str,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": name, "pos": f"-0.14 0.08 {z:.6f}"},
    )
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_surface",
            "type": "cylinder",
            "size": "0.044 0.0015",
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )


def build_showcase_model() -> mujoco.MjModel:
    xml_text, assets = demo._build_xml_and_assets()
    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("showcase model has no worldbody")
    _add_pitcher(
        worldbody,
        "hot_water",
        PITCHER_HOME_POSITIONS["hot_water"],
        "0.88 0.32 0.12 0.92",
    )
    _add_pitcher(
        worldbody,
        "milk",
        PITCHER_HOME_POSITIONS["milk"],
        "0.92 0.92 0.86 0.94",
    )
    _add_recipe_visual(worldbody, "showcase_hot_water", 0.0810, "0.38 0.67 0.84 0.75")
    _add_recipe_visual(worldbody, "showcase_matcha", 0.0830, "0.22 0.60 0.18 0.90")
    _add_recipe_visual(worldbody, "showcase_foam", 0.0850, "0.68 0.82 0.52 0.96")
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets
    )
    cameras = [
        str(model.camera(camera_id).name)
        for camera_id in range(model.ncam)
    ]
    if cameras != [SHOWCASE_CAMERA]:
        raise RuntimeError(f"single-camera contract drifted: {cameras}")
    return model


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation).reshape(-1))
    return quaternion


class ShowcaseKinematics:
    """Apply timeline samples to free tools and presentation props."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.arm_qpos = np.asarray(
            [model.joint(name).qposadr[0] for name in demo.ARM_JOINTS],
            dtype=int,
        )
        self.robot_body_id = int(model.body("robot_plate_frame").id)
        self.tool_qpos = {
            tool: int(model.joint(f"tool_{tool}_free").qposadr[0])
            for tool in demo.ALL_TOOL_IDS
        }
        self.tool_dock_qpos = {
            tool: np.asarray(
                data.qpos[address : address + 7], dtype=np.float64
            ).copy()
            for tool, address in self.tool_qpos.items()
        }
        self.pitcher_qpos = {
            name: int(model.joint(f"showcase_{name}_pitcher_free").qposadr[0])
            for name in PITCHER_HOME_POSITIONS
        }
        self.pitcher_home_qpos = {
            name: np.asarray(
                data.qpos[address : address + 7], dtype=np.float64
            ).copy()
            for name, address in self.pitcher_qpos.items()
        }
        self.recipe_qpos = {
            name: int(model.joint(f"{name}_free").qposadr[0])
            for name in ("showcase_hot_water", "showcase_matcha", "showcase_foam")
        }
        self.recipe_home_qpos = {
            name: np.asarray(
                data.qpos[address : address + 7], dtype=np.float64
            ).copy()
            for name, address in self.recipe_qpos.items()
        }
        self.gripper_qpos = int(model.joint("gripper").qposadr[0])
        self.slider_qpos = int(
            model.joint("qc_positive_lock_slider_joint").qposadr[0]
        )
        self.whisk_rotor_qpos = int(
            model.joint("whisk_rotor_joint").qposadr[0]
        )
        self.relative_tool_pose = self._capture_relative_tool_poses()

    def _capture_relative_tool_poses(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        original_arm = np.asarray(self.data.qpos[self.arm_qpos]).copy()
        for tool in demo.ALL_TOOL_IDS:
            self.data.qpos[self.arm_qpos] = demo.DOCK_CAPTURE_Q[tool]
            mujoco.mj_kinematics(self.model, self.data)
            robot_position = np.asarray(
                self.data.xpos[self.robot_body_id], dtype=np.float64
            ).copy()
            robot_rotation = np.asarray(
                self.data.xmat[self.robot_body_id], dtype=np.float64
            ).reshape(3, 3).copy()
            tool_position = np.asarray(
                self.data.body(f"tool_{tool}").xpos, dtype=np.float64
            ).copy()
            tool_rotation = np.asarray(
                self.data.body(f"tool_{tool}").xmat, dtype=np.float64
            ).reshape(3, 3).copy()
            result[tool] = (
                robot_rotation.T @ (tool_position - robot_position),
                robot_rotation.T @ tool_rotation,
            )
        self.data.qpos[self.arm_qpos] = original_arm
        mujoco.mj_kinematics(self.model, self.data)
        return result

    def _attach_tool(self, tool: str) -> None:
        robot_position = np.asarray(
            self.data.xpos[self.robot_body_id], dtype=np.float64
        )
        robot_rotation = np.asarray(
            self.data.xmat[self.robot_body_id], dtype=np.float64
        ).reshape(3, 3)
        relative_position, relative_rotation = self.relative_tool_pose[tool]
        address = self.tool_qpos[tool]
        self.data.qpos[address : address + 3] = (
            robot_position + robot_rotation @ relative_position
        )
        self.data.qpos[address + 3 : address + 7] = _matrix_to_quaternion(
            robot_rotation @ relative_rotation
        )

    def _apply_pitcher(self, state: ShowcaseState) -> None:
        if not state.pitcher_held or state.pitcher is None:
            return
        address = self.pitcher_qpos[state.pitcher]
        grasp = np.asarray(self.data.site("gripperframe").xpos, dtype=np.float64)
        angle = math.radians(72.0) * state.pour_fraction
        quaternion = np.asarray(
            [math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0],
            dtype=np.float64,
        )
        self.data.qpos[address : address + 3] = grasp + np.asarray(
            [0.0, 0.0, -0.058], dtype=np.float64
        )
        self.data.qpos[address + 3 : address + 7] = quaternion

    def apply(self, state: ShowcaseState) -> None:
        for tool, address in self.tool_qpos.items():
            self.data.qpos[address : address + 7] = self.tool_dock_qpos[tool]
        for name, address in self.pitcher_qpos.items():
            self.data.qpos[address : address + 7] = self.pitcher_home_qpos[name]
        for name, address in self.recipe_qpos.items():
            hidden = self.recipe_home_qpos[name].copy()
            hidden[2] = -0.20
            self.data.qpos[address : address + 7] = hidden
        visible = RECIPE_VISIBLE_BODY[state.recipe_stage]
        if visible is not None:
            address = self.recipe_qpos[visible]
            self.data.qpos[address : address + 7] = self.recipe_home_qpos[visible]
        self.data.qpos[self.arm_qpos] = np.asarray(state.arm_q)
        self.data.qpos[self.slider_qpos] = state.slider_q_m
        self.data.qpos[self.gripper_qpos] = state.gripper_q_rad
        self.data.qpos[self.whisk_rotor_qpos] = state.whisk_rotor_rad
        mujoco.mj_kinematics(self.model, self.data)
        if state.attached_tool is not None:
            self._attach_tool(state.attached_tool)
            mujoco.mj_kinematics(self.model, self.data)
        self._apply_pitcher(state)
        mujoco.mj_kinematics(self.model, self.data)


def annotate(frame: np.ndarray, state: ShowcaseState) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title = _font(28, bold=True)
    label = _font(18, bold=True)
    small = _font(15)
    draw.rectangle((0, 0, image.width, 74), fill=(3, 7, 10, 225))
    draw.text((24, 10), "SO-101 QUICKATTACH • MATCHA", font=title, fill=(244, 205, 67, 255))
    draw.text(
        (24, 45),
        "One fixed scene camera • compiled MuJoCo geometry",
        font=small,
        fill=(210, 222, 232, 255),
    )
    panel_top = image.height - 104
    draw.rectangle((0, panel_top, image.width, image.height), fill=(3, 7, 10, 225))
    draw.text(
        (24, panel_top + 12),
        f"{state.elapsed_s:04.1f}s  •  {state.caption}",
        font=label,
        fill=(255, 255, 255, 255),
    )
    chips = (
        ("GRIPPER", state.attached_tool == "gripper"),
        ("SPOON", state.attached_tool == "spoon"),
        ("WHISK", state.attached_tool == "whisk"),
        ("LOCK", state.locked),
        ("BUS", state.bus_connected),
        ("MOTOR", state.whisk_motor),
    )
    x = 24
    for text, active in chips:
        color = (38, 184, 105, 255) if active else (70, 80, 92, 255)
        draw.rounded_rectangle((x, panel_top + 49, x + 92, panel_top + 79), 7, fill=color)
        box = draw.textbbox((0, 0), text, font=small)
        draw.text(
            (x + (92 - (box[2] - box[0])) / 2, panel_top + 55),
            text,
            font=small,
            fill=(255, 255, 255, 255),
        )
        x += 100
    recipe = state.recipe_stage.replace("_", " ").upper()
    draw.text(
        (image.width - 225, panel_top + 56),
        recipe,
        font=small,
        fill=(203, 235, 162, 255),
    )
    progress_y = panel_top - 5
    draw.rectangle((0, progress_y, image.width, progress_y + 5), fill=(45, 53, 62, 255))
    draw.rectangle(
        (0, progress_y, int(image.width * state.progress), progress_y + 5),
        fill=(87, 190, 90, 255),
    )
    return np.asarray(image)


def _ffprobe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def record(
    output: Path,
    report_path: Path,
    fps: int,
    width: int,
    height: int,
    mesh_cell_mm: float,
) -> dict[str, object]:
    if fps <= 0 or width <= 0 or height <= 0 or mesh_cell_mm <= 0.0:
        raise ValueError("video dimensions, fps, and mesh cell must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_showcase_model()
    data = mujoco.MjData(model)
    demo.initialize(model, data)
    kinematics = ShowcaseKinematics(model, data)
    geometry = build_render_geometry(model, mesh_cell_mm / 1000.0)
    duration = showcase_duration_s(showcase_segments())
    frame_count = int(math.ceil(duration * fps)) + 1
    camera_position = np.asarray(model.cam_pos[model.camera(SHOWCASE_CAMERA).id]).copy()
    camera_target = np.asarray([0.0, 0.0, 0.205], dtype=np.float64)
    fov = float(model.cam_fovy[model.camera(SHOWCASE_CAMERA).id])
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not expose stdin")
    try:
        for frame_index in range(frame_count):
            time_s = min(duration, frame_index / fps)
            state = showcase_state_at(time_s)
            kinematics.apply(state)
            frame = render_view(
                model,
                data,
                geometry,
                (width, height),
                camera_position,
                camera_target,
                fov,
            )
            process.stdin.write(annotate(frame, state).tobytes())
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with status {return_code}")
    report: dict[str, object] = {
        **showcase_summary(),
        "video": {
            "path": str(
                output.resolve().relative_to(HERE.parents[2].resolve())
                if output.resolve().is_relative_to(HERE.parents[2].resolve())
                else output
            ),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count_requested": frame_count,
            "mesh_cell_mm": mesh_cell_mm,
            "ffprobe": _ffprobe(output),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mesh-cell-mm", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = record(
        args.output,
        args.report,
        args.fps,
        args.width,
        args.height,
        args.mesh_cell_mm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
