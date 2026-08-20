#!/usr/bin/env python3
"""Capture the canonical QuickAttach Three.js scene through Playwright and FFmpeg."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


PARKED = [-0.07, -1.65, 0.02, 1.35, 0.0]
HOME = [0.0, -0.5, 0.8, -0.3, 0.0]
DOCK = {
    "gripper": [-0.72, -0.5, 0.8, -0.3, -1.522116811941435],
    "spoon": [0.0, -0.5, 0.8, -0.3, 0.0],
    "whisk": [0.72, -0.5, 0.8, -0.3, 0.0],
}
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def state(joints, tool=None, *, whisk_angle=0.0, gripper=0.15):
    return {
        "joints": dict(zip(ARM_JOINTS, joints, strict=True)),
        "selectedTool": tool,
        "attachedTool": tool,
        "whiskAngle": whisk_angle,
        "gripper": gripper,
    }


TIMELINE = (
    (0.0, state(PARKED)),
    (1.5, state(DOCK["gripper"])),
    (2.0, state(DOCK["gripper"], "gripper")),
    (4.5, state(HOME, "gripper", gripper=0.55)),
    (6.5, state(DOCK["gripper"], "gripper", gripper=0.15)),
    (7.0, state(DOCK["gripper"])),
    (8.5, state(DOCK["spoon"])),
    (9.0, state(DOCK["spoon"], "spoon")),
    (11.5, state(HOME, "spoon")),
    (13.5, state(DOCK["spoon"], "spoon")),
    (14.0, state(DOCK["spoon"])),
    (15.5, state(DOCK["whisk"])),
    (16.0, state(DOCK["whisk"], "whisk")),
    (18.5, state(HOME, "whisk", whisk_angle=7.5)),
    (21.0, state(HOME, "whisk", whisk_angle=22.0)),
    (23.0, state(DOCK["whisk"], "whisk", whisk_angle=26.0)),
    (23.5, state(DOCK["whisk"])),
    (26.0, state(PARKED)),
)


def headless_url(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["headless"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def lerp(first: float, second: float, fraction: float) -> float:
    return first + (second - first) * fraction


def timeline_state(time_s: float) -> dict:
    before_time, before = TIMELINE[0]
    after_time, after = TIMELINE[-1]
    for index in range(1, len(TIMELINE)):
        if time_s <= TIMELINE[index][0]:
            before_time, before = TIMELINE[index - 1]
            after_time, after = TIMELINE[index]
            break
    fraction = 1.0 if after_time <= before_time else max(
        0.0, min(1.0, (time_s - before_time) / (after_time - before_time))
    )
    result = dict(before)
    result["joints"] = {
        name: lerp(before["joints"][name], after["joints"][name], fraction)
        for name in ARM_JOINTS
    }
    result["gripper"] = lerp(before["gripper"], after["gripper"], fraction)
    result["whiskAngle"] = lerp(before["whiskAngle"], after["whiskAngle"], fraction)
    if fraction >= 1.0:
        result["selectedTool"] = after["selectedTool"]
        result["attachedTool"] = after["attachedTool"]
    return result


def load_states(path: Path | None, frame_count: int, fps: float) -> list[dict]:
    if path is None:
        return [timeline_state(index / fps) for index in range(frame_count)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("--states must contain a JSON array of QuickAttach state objects")
    payload = payload or [{}]
    if frame_count == 1:
        return [payload[0]]
    return [
        payload[min(len(payload) - 1, round(index * (len(payload) - 1) / (frame_count - 1)))]
        for index in range(frame_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the shared Three.js QuickAttach viewport into an H.264 MP4."
    )
    parser.add_argument("--url", default="http://127.0.0.1:4173/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=Path, help="JSON array; one state per frame (last repeats)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--keyframe-fps",
        type=float,
        default=1.0,
        help="Browser-rendered keyframes per second; FFmpeg interpolates to --fps",
    )
    parser.add_argument("--seconds", type=float, default=26.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.fps <= 0 or args.keyframe_fps <= 0 or args.seconds <= 0:
        raise SystemExit("--fps, --keyframe-fps, and --seconds must be positive")

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required on PATH")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Python Playwright is required: pip install playwright && playwright install chromium"
        ) from exc

    keyframe_count = max(2, round(args.keyframe_fps * args.seconds) + 1)
    states = load_states(args.states, keyframe_count, args.keyframe_fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="quickattach-threejs-") as temp_name:
        frame_dir = Path(temp_name)
        with sync_playwright() as playwright:
            executable_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
            chromium_args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl"]
            if extra_args := os.environ.get("PLAYWRIGHT_CHROMIUM_ARGS"):
                chromium_args.extend(json.loads(extra_args))
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=executable_path,
                args=chromium_args,
            )
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.goto(headless_url(args.url), wait_until="load")
            page.wait_for_function(
                "window.quickAttach && window.quickAttach.ready === true",
                timeout=120_000,
            )
            page.locator(".loading-card").wait_for(state="detached", timeout=30_000)
            page.evaluate("() => window.quickAttach.showScene()")
            page.evaluate("() => window.quickAttach.resetCamera()")
            page.evaluate("() => window.quickAttach.setCaptureMode(true)")
            page.wait_for_timeout(250)
            canvas = page.locator("canvas")
            for frame_index, frame_state in enumerate(states):
                page.evaluate(
                    "state => window.quickAttach.setState(state)",
                    frame_state,
                )
                data_url = canvas.evaluate(
                    "surface => surface.toDataURL('image/jpeg', 0.92)"
                )
                frame_data = data_url.partition(",")[2]
                (frame_dir / f"frame-{frame_index:06d}.jpg").write_bytes(
                    base64.b64decode(frame_data)
                )
            browser.close()

        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", f"{args.keyframe_fps:.6f}",
                "-i", str(frame_dir / "frame-%06d.jpg"),
                "-vf",
                (
                    "tpad=stop_mode=clone:stop_duration=2,"
                    f"minterpolate=fps={args.fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
                    f"trim=duration={args.seconds},setpts=PTS-STARTPTS"
                ),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(args.output),
            ],
            check=True,
        )
    print(f"Rendered {keyframe_count} browser keyframes from the shared scene to {args.output}")


if __name__ == "__main__":
    main()
