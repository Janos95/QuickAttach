#!/usr/bin/env python3
"""Record the full-arm QuickAttach demo with a deterministic CPU z-buffer.

The MuJoCo state, kinematics, equality constraints, and gripper actuator are the
same ones used by ``so101_gripper_change_demo.py``.  Only rendering is replaced:
this script rasterizes MuJoCo's compiled geometry on the CPU so it works on
headless machines without EGL, OSMesa, X11, or a GPU.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

from so101_gripper_change_demo import (
    QuickChangeController,
    build_model,
    initialize,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "exports" / "so101_mujoco_quickattach_demo.mp4"
VIDEO_SECONDS = 8.0


def _box_triangles(size: np.ndarray) -> np.ndarray:
    x, y, z = size
    vertices = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices[faces]


def _cylinder_triangles(radius: float, half_height: float, segments: int = 18) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring_low = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.full(segments, -half_height))
    )
    ring_high = ring_low.copy()
    ring_high[:, 2] = half_height
    vertices = np.vstack((ring_low, ring_high, [[0, 0, -half_height], [0, 0, half_height]])).astype(np.float32)
    low_center = 2 * segments
    high_center = low_center + 1
    faces: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(
            (
                [index, nxt, segments + nxt],
                [index, segments + nxt, segments + index],
                [low_center, nxt, index],
                [high_center, segments + index, segments + nxt],
            )
        )
    return vertices[np.asarray(faces, dtype=np.int32)]


def _plane_triangles(size: np.ndarray) -> np.ndarray:
    x = max(float(size[0]), 0.7)
    y = max(float(size[1]), 0.7)
    vertices = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]], dtype=np.float32)
    return vertices[np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)]


def _simplified_mesh(model: mujoco.MjModel, mesh_id: int, cell: float) -> np.ndarray:
    vert_adr = model.mesh_vertadr[mesh_id]
    face_adr = model.mesh_faceadr[mesh_id]
    vertices = np.asarray(
        model.mesh_vert[vert_adr : vert_adr + model.mesh_vertnum[mesh_id]], dtype=np.float32
    )
    faces = np.asarray(
        model.mesh_face[face_adr : face_adr + model.mesh_facenum[mesh_id]], dtype=np.int32
    )
    quantized = np.round(vertices / cell).astype(np.int32)
    unique, first, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    del unique
    simplified_vertices = vertices[first]
    simplified_faces = inverse[faces]
    keep = (
        (simplified_faces[:, 0] != simplified_faces[:, 1])
        & (simplified_faces[:, 1] != simplified_faces[:, 2])
        & (simplified_faces[:, 0] != simplified_faces[:, 2])
    )
    simplified_faces = simplified_faces[keep]
    keys = np.sort(simplified_faces, axis=1)
    _, first_face = np.unique(keys, axis=0, return_index=True)
    simplified_faces = simplified_faces[np.sort(first_face)]
    return simplified_vertices[simplified_faces].astype(np.float32)


def build_render_geometry(model: mujoco.MjModel, cell: float) -> list[tuple[np.ndarray, np.ndarray]]:
    mesh_cache: dict[int, np.ndarray] = {}
    geometry: list[tuple[np.ndarray, np.ndarray]] = []
    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id not in mesh_cache:
                mesh_cache[mesh_id] = _simplified_mesh(model, mesh_id, cell)
            triangles = mesh_cache[mesh_id]
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            triangles = _box_triangles(model.geom_size[geom_id])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            triangles = _cylinder_triangles(
                float(model.geom_size[geom_id, 0]), float(model.geom_size[geom_id, 1])
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            triangles = _plane_triangles(model.geom_size[geom_id])
        else:
            continue
        material_id = int(model.geom_matid[geom_id])
        rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id]
        color = np.clip(rgba[:3] * 255.0, 0, 255).astype(np.uint8)
        if model.geom(geom_id).name == "qc_floor":
            color = np.array([35, 40, 47], dtype=np.uint8)
        geometry.append((triangles, color))
    return geometry


def _look_at(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - position
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.vstack((right, up, forward)).astype(np.float32)
    return position.astype(np.float32), rotation


def _scene_triangles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: list[tuple[np.ndarray, np.ndarray]],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    width: int,
    height: int,
    vertical_fov_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    projected: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    focal = 0.5 * height / np.tan(np.deg2rad(vertical_fov_degrees) * 0.5)
    for geom_id, (local_triangles, base_color) in enumerate(geometry):
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world = local_triangles @ rotation.T + data.geom_xpos[geom_id]
        camera = (world - camera_position) @ camera_rotation.T
        z = camera[:, :, 2]
        valid = np.all(z > 0.025, axis=1)
        if not np.any(valid):
            continue
        camera = camera[valid]
        z = camera[:, :, 2]
        screen = np.empty_like(camera, dtype=np.float32)
        screen[:, :, 0] = width * 0.5 + focal * camera[:, :, 0] / z
        screen[:, :, 1] = height * 0.5 - focal * camera[:, :, 1] / z
        screen[:, :, 2] = z
        in_frame = (
            (np.max(screen[:, :, 0], axis=1) >= 0)
            & (np.min(screen[:, :, 0], axis=1) < width)
            & (np.max(screen[:, :, 1], axis=1) >= 0)
            & (np.min(screen[:, :, 1], axis=1) < height)
        )
        screen = screen[in_frame]
        world_valid = world[valid][in_frame]
        if len(screen) == 0:
            continue
        normals = np.cross(world_valid[:, 1] - world_valid[:, 0], world_valid[:, 2] - world_valid[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-8)
        light = np.array([-0.35, -0.55, 0.76])
        light /= np.linalg.norm(light)
        shade = 0.34 + 0.66 * np.abs(normals @ light)
        face_colors = np.clip(base_color[None, :] * shade[:, None], 0, 255).astype(np.uint8)
        projected.append(screen)
        colors.append(face_colors)
    return np.concatenate(projected), np.concatenate(colors)


@njit(cache=True)
def _rasterize(triangles: np.ndarray, colors: np.ndarray, image: np.ndarray, depth: np.ndarray) -> None:
    height, width, _ = image.shape
    for index in range(triangles.shape[0]):
        x0, y0, z0 = triangles[index, 0]
        x1, y1, z1 = triangles[index, 1]
        x2, y2, z2 = triangles[index, 2]
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 0.02:
            continue
        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        if min_x > max_x or min_y > max_y:
            continue
        inv_area = 1.0 / area
        for pixel_y in range(min_y, max_y + 1):
            py = pixel_y + 0.5
            for pixel_x in range(min_x, max_x + 1):
                px = pixel_x + 0.5
                w0 = ((x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)) * inv_area
                w1 = ((x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)) * inv_area
                w2 = 1.0 - w0 - w1
                if w0 < -1e-5 or w1 < -1e-5 or w2 < -1e-5:
                    continue
                reciprocal_depth = w0 / z0 + w1 / z1 + w2 / z2
                if reciprocal_depth <= 0:
                    continue
                value = 1.0 / reciprocal_depth
                if value < depth[pixel_y, pixel_x]:
                    depth[pixel_y, pixel_x] = value
                    image[pixel_y, pixel_x, 0] = colors[index, 0]
                    image[pixel_y, pixel_x, 1] = colors[index, 1]
                    image[pixel_y, pixel_x, 2] = colors[index, 2]


def _background(width: int, height: int) -> np.ndarray:
    top = np.array([30, 42, 58], dtype=np.float32)
    bottom = np.array([7, 10, 15], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rows = top[None, :] * (1.0 - blend) + bottom[None, :] * blend
    return np.repeat(rows[:, None, :], width, axis=1).astype(np.uint8)


def render_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: list[tuple[np.ndarray, np.ndarray]],
    size: tuple[int, int],
    camera_position: np.ndarray,
    camera_target: np.ndarray,
    fov: float,
) -> np.ndarray:
    width, height = size
    position, rotation = _look_at(camera_position, camera_target)
    triangles, colors = _scene_triangles(
        model, data, geometry, position, rotation, width, height, fov
    )
    image = _background(width, height)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    _rasterize(triangles, colors, image, depth)
    return image


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts")):
        candidate = root / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def annotate(
    frame: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: QuickChangeController,
    divider_x: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(28, bold=True)
    label_font = _font(18, bold=True)
    body_font = _font(17)
    small_font = _font(15)
    draw.rectangle((0, 0, image.width, 68), fill=(3, 6, 10, 220))
    draw.text((24, 12), "SO-101 QUICKATTACH", font=title_font, fill=(255, 205, 57, 255))
    draw.text((24, 43), "Actual MuJoCo state • deterministic CPU z-buffer", font=small_font, fill=(205, 215, 228, 255))
    draw.text((divider_x + 18, 20), "COUPLER CLOSE-UP", font=label_font, fill=(237, 242, 250, 255))
    draw.line((divider_x, 68, divider_x, image.height), fill=(225, 232, 240, 170), width=2)

    phase = controller._phase(float(data.time))
    panel_top = image.height - 105
    draw.rectangle((0, panel_top, image.width, image.height), fill=(3, 6, 10, 218))
    draw.text((24, panel_top + 14), f"{data.time:04.1f} s  •  {phase}", font=label_font, fill=(255, 255, 255, 255))

    state_specs = (
        ("DOCK", bool(data.eq_active[model.equality("tool_in_dock").id])),
        ("MAGNET", bool(data.eq_active[model.equality("magnetic_capture").id])),
        ("LOCK", bool(data.eq_active[model.equality("positive_lock").id])),
        ("ID-6 BUS", controller.simulated_id6_handshake),
    )
    x = 24
    for label, active in state_specs:
        color = (42, 204, 112, 255) if active else (83, 93, 108, 255)
        draw.rounded_rectangle((x, panel_top + 52, x + 122, panel_top + 86), radius=8, fill=color)
        text_box = draw.textbbox((0, 0), label, font=small_font)
        text_width = text_box[2] - text_box[0]
        draw.text((x + (122 - text_width) / 2, panel_top + 60), label, font=small_font, fill=(255, 255, 255, 255))
        x += 132

    draw.text(
        (divider_x + 18, panel_top + 51),
        "Yellow: robot-side plate\nBlue: detachable tool plate\nBlack: passive dock\nCapture/lock: switched constraints",
        font=small_font,
        fill=(220, 228, 238, 255),
        spacing=1,
    )
    return np.asarray(image)


def record(output: Path, fps: int, width: int, height: int, cell_mm: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    data = mujoco.MjData(model)
    initialize(model, data)
    controller = QuickChangeController(model, data)
    geometry = build_render_geometry(model, cell_mm / 1000.0)

    divider_x = int(width * 0.64)
    wide_size = (divider_x, height)
    close_size = (width - divider_x, height)
    wide_position = np.array([0.53, -0.46, 0.36])
    wide_target = np.array([0.20, 0.0, 0.16])

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not expose stdin")

    next_frame = 0.0
    frame_period = 1.0 / fps
    try:
        while data.time < VIDEO_SECONDS:
            controller.update()
            mujoco.mj_step(model, data)
            if data.time + 1e-9 < next_frame:
                continue

            robot_face = data.site_xpos[model.site("robot_mating_face").id].copy()
            tool_face = data.body("tool_plate").xpos.copy()
            close_target = 0.5 * (robot_face + tool_face)
            close_position = close_target + np.array([0.20, -0.27, 0.15])
            wide = render_view(
                model, data, geometry, wide_size, wide_position, wide_target, 36.0
            )
            close = render_view(
                model, data, geometry, close_size, close_position, close_target, 38.0
            )
            frame = np.concatenate((wide, close), axis=1)
            frame = annotate(frame, model, data, controller, divider_x)
            process.stdin.write(frame.tobytes())
            next_frame += frame_period
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with status {return_code}")
    print(f"Saved {output} ({width}x{height}, {fps} fps, {VIDEO_SECONDS:.1f} s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mesh-cell-mm", type=float, default=2.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record(args.output, args.fps, args.width, args.height, args.mesh_cell_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
