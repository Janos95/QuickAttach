#!/usr/bin/env python3
"""Deterministic end-to-end Matcha showcase choreography.

This module is deliberately separate from ``matcha_workflow_demo``.  The
production controller remains fail-closed around unqualified contact,
friction, material, and reverse-insertion authority.  The showcase consumes
the compiled production model and its published rack trajectories, then
provides a deterministic *visualization* timeline for the complete user story:
the stock gripper handles both pitchers, the spoon doses Matcha, and the
powered whisk mixes it.  It does not promote a rendered sequence into physical
release evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

import matcha_workflow_demo as demo


SHOWCASE_SCHEMA_VERSION = "1.0-matcha-single-camera-showcase"
SHOWCASE_CAMERA = demo.CAMERA_NAME
SHOWCASE_TOOLS = ("gripper", "spoon", "whisk")
SHOWCASE_RECIPE_STAGES = (
    "empty_bowl",
    "hot_water_added",
    "matcha_dosed",
    "whisked",
    "milk_added",
    "complete",
)

NEUTRAL_Q = (0.0, -0.92, 1.18, -0.26, 0.0)
GRIPPER_HOT_PICK_Q = (
    -0.23141183,
    -1.74489890,
    -0.36418979,
    -1.32828810,
    2.49789501,
)
GRIPPER_HOT_POUR_Q = (-0.20, -1.00, 0.80, -0.60, 1.20)
GRIPPER_MILK_PICK_Q = (
    0.50,
    -1.60,
    -0.20,
    -1.00,
    -2.00,
)
GRIPPER_MILK_POUR_Q = (0.20, -1.00, 0.80, -0.60, -1.20)
SPOON_POWDER_Q = (
    -0.78157542,
    -1.50887199,
    -0.80009178,
    -0.35266322,
    0.41159969,
)
SPOON_SIEVE_Q = (
    0.00337061,
    -0.81717745,
    -1.65883972,
    -1.03960185,
    0.57648251,
)
WHISK_BOWL_Q = (
    0.42141243,
    -0.89679577,
    -1.69000000,
    -1.65806285,
    -0.19068022,
)


def _q(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != len(demo.ARM_JOINTS):
        raise ValueError("showcase arm posture has the wrong width")
    return result


@dataclass(frozen=True)
class ShowcaseSegment:
    """One continuous, labelled visual state interval."""

    name: str
    caption: str
    duration_s: float
    start_q: tuple[float, ...]
    end_q: tuple[float, ...]
    attached_tool: str | None = None
    locked: bool = False
    bus_connected: bool = False
    slider_start_m: float = 0.0
    slider_end_m: float = 0.0
    gripper_start_rad: float = 0.90
    gripper_end_rad: float = 0.90
    whisk_motor: bool = False
    pitcher: str | None = None
    pitcher_held: bool = False
    pour_fraction_start: float = 0.0
    pour_fraction_end: float = 0.0
    recipe_stage: str = "empty_bowl"
    recipe_stage_end: str | None = None
    event: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.caption:
            raise ValueError("showcase segment name/caption must be non-empty")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError(f"invalid showcase duration for {self.name}")
        _q(self.start_q)
        _q(self.end_q)
        if self.attached_tool is not None and self.attached_tool not in SHOWCASE_TOOLS:
            raise ValueError(f"unknown showcase tool {self.attached_tool}")
        if self.pitcher not in {None, "hot_water", "milk"}:
            raise ValueError(f"unknown showcase pitcher {self.pitcher}")
        if self.pitcher_held and self.attached_tool != "gripper":
            raise ValueError("only the stock gripper may hold a pitcher")
        if self.whisk_motor and self.attached_tool != "whisk":
            raise ValueError("whisk motor requires the attached whisk")
        if self.recipe_stage not in SHOWCASE_RECIPE_STAGES:
            raise ValueError(f"unknown recipe stage {self.recipe_stage}")
        if (
            self.recipe_stage_end is not None
            and self.recipe_stage_end not in SHOWCASE_RECIPE_STAGES
        ):
            raise ValueError(f"unknown ending recipe stage {self.recipe_stage_end}")
        for slider in (self.slider_start_m, self.slider_end_m):
            if slider < 0.0 or slider > 0.003 + 1.0e-12:
                raise ValueError("positive-lock slider left its published range")


def _segment(
    name: str,
    caption: str,
    duration_s: float,
    start_q: Iterable[float],
    end_q: Iterable[float],
    **kwargs: object,
) -> ShowcaseSegment:
    return ShowcaseSegment(
        name=name,
        caption=caption,
        duration_s=duration_s,
        start_q=_q(start_q),
        end_q=_q(end_q),
        **kwargs,
    )


def _rack_exit_q() -> tuple[float, ...]:
    contract = demo.core_dock_static_release_route_contract()
    return _q(contract["roster"][-1]["q_rad"])


def showcase_segments() -> tuple[ShowcaseSegment, ...]:
    """Return the complete deterministic tool-change and recipe timeline."""

    gripper_capture = _q(demo.DOCK_CAPTURE_Q["gripper"])
    gripper_pre = _q(demo.DOCK_PRE_CAPTURE_Q["gripper"])
    gripper_exit = _rack_exit_q()
    spoon_capture = _q(demo.DOCK_CAPTURE_Q["spoon"])
    spoon_pre = _q(demo.DOCK_PRE_CAPTURE_Q["spoon"])
    whisk_capture = _q(demo.DOCK_CAPTURE_Q["whisk"])
    whisk_pre = _q(demo.DOCK_PRE_CAPTURE_Q["whisk"])

    segments = (
        _segment(
            "overview",
            "Three passive tools • one SO-101",
            1.4,
            NEUTRAL_Q,
            NEUTRAL_Q,
        ),
        _segment(
            "gripper_approach",
            "Approach stock gripper",
            1.2,
            NEUTRAL_Q,
            gripper_pre,
        ),
        _segment(
            "gripper_guided_capture",
            "Cam-guided insertion",
            1.8,
            gripper_pre,
            gripper_capture,
            event="gripper_capture",
        ),
        _segment(
            "gripper_bus_verify",
            "Magnets seated • ID-6 bus verified",
            0.7,
            gripper_capture,
            gripper_capture,
            attached_tool="gripper",
            bus_connected=True,
        ),
        _segment(
            "gripper_rack_exit",
            "Rack exit • spring lock closes",
            1.2,
            gripper_capture,
            gripper_exit,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.0,
            slider_end_m=0.003,
            event="gripper_lock",
        ),
        _segment(
            "hot_water_pick",
            "Stock gripper takes hot-water pitcher",
            1.5,
            gripper_exit,
            GRIPPER_HOT_PICK_Q,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            pitcher="hot_water",
            gripper_start_rad=0.90,
            gripper_end_rad=0.10,
        ),
        _segment(
            "hot_water_pour",
            "Dose hot water",
            1.7,
            GRIPPER_HOT_PICK_Q,
            GRIPPER_HOT_POUR_Q,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            pitcher="hot_water",
            pitcher_held=True,
            gripper_start_rad=0.10,
            gripper_end_rad=0.10,
            pour_fraction_end=1.0,
            recipe_stage_end="hot_water_added",
            event="hot_water_added",
        ),
        _segment(
            "hot_water_return",
            "Return hot-water pitcher",
            1.3,
            GRIPPER_HOT_POUR_Q,
            GRIPPER_HOT_PICK_Q,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            pitcher="hot_water",
            pitcher_held=True,
            gripper_start_rad=0.10,
            gripper_end_rad=0.90,
            recipe_stage="hot_water_added",
        ),
        _segment(
            "gripper_return",
            "Return gripper to rack",
            1.5,
            GRIPPER_HOT_PICK_Q,
            gripper_exit,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            recipe_stage="hot_water_added",
        ),
        _segment(
            "gripper_reverse_insert",
            "Rack cam reopens lock",
            1.2,
            gripper_exit,
            gripper_capture,
            attached_tool="gripper",
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.0,
            recipe_stage="hot_water_added",
            event="gripper_unlock",
        ),
        _segment(
            "gripper_release",
            "Gripper retained • bare wrist withdraws",
            1.1,
            gripper_capture,
            gripper_pre,
            recipe_stage="hot_water_added",
            event="gripper_release",
        ),
        _segment(
            "spoon_approach",
            "Approach Matcha spoon",
            1.1,
            gripper_pre,
            spoon_pre,
            recipe_stage="hot_water_added",
        ),
        _segment(
            "spoon_capture",
            "Capture spoon • verify tool ID 21",
            1.5,
            spoon_pre,
            spoon_capture,
            recipe_stage="hot_water_added",
            event="spoon_capture",
        ),
        _segment(
            "spoon_exit",
            "Lock spoon and leave rack",
            1.0,
            spoon_capture,
            spoon_pre,
            attached_tool="spoon",
            locked=True,
            bus_connected=True,
            slider_start_m=0.0,
            slider_end_m=0.003,
            recipe_stage="hot_water_added",
            event="spoon_lock",
        ),
        _segment(
            "spoon_scoop",
            "Scoop measured Matcha",
            1.5,
            spoon_pre,
            SPOON_POWDER_Q,
            attached_tool="spoon",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            recipe_stage="hot_water_added",
        ),
        _segment(
            "spoon_dose",
            "Dose Matcha through sieve",
            1.7,
            SPOON_POWDER_Q,
            SPOON_SIEVE_Q,
            attached_tool="spoon",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            recipe_stage="hot_water_added",
            recipe_stage_end="matcha_dosed",
            event="matcha_dosed",
        ),
        _segment(
            "spoon_return",
            "Return spoon",
            1.5,
            SPOON_SIEVE_Q,
            spoon_pre,
            attached_tool="spoon",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            recipe_stage="matcha_dosed",
        ),
        _segment(
            "spoon_reverse_insert",
            "Rack cam unlocks spoon",
            1.0,
            spoon_pre,
            spoon_capture,
            attached_tool="spoon",
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.0,
            recipe_stage="matcha_dosed",
        ),
        _segment(
            "spoon_release",
            "Spoon retained in rack",
            0.9,
            spoon_capture,
            spoon_pre,
            recipe_stage="matcha_dosed",
            event="spoon_release",
        ),
        _segment(
            "whisk_approach",
            "Approach powered whisk",
            1.1,
            spoon_pre,
            whisk_pre,
            recipe_stage="matcha_dosed",
        ),
        _segment(
            "whisk_capture",
            "Capture whisk • verify tool ID 22",
            1.5,
            whisk_pre,
            whisk_capture,
            recipe_stage="matcha_dosed",
            event="whisk_capture",
        ),
        _segment(
            "whisk_exit",
            "Lock whisk and leave rack",
            1.0,
            whisk_capture,
            whisk_pre,
            attached_tool="whisk",
            locked=True,
            bus_connected=True,
            slider_start_m=0.0,
            slider_end_m=0.003,
            recipe_stage="matcha_dosed",
            event="whisk_lock",
        ),
        _segment(
            "whisk_mix",
            "Powered chasen mixes Matcha",
            2.3,
            whisk_pre,
            WHISK_BOWL_Q,
            attached_tool="whisk",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            whisk_motor=True,
            recipe_stage="matcha_dosed",
            recipe_stage_end="whisked",
            event="matcha_whisked",
        ),
        _segment(
            "whisk_return",
            "Return powered whisk",
            1.5,
            WHISK_BOWL_Q,
            whisk_pre,
            attached_tool="whisk",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            recipe_stage="whisked",
        ),
        _segment(
            "whisk_reverse_insert",
            "Rack cam unlocks whisk",
            1.0,
            whisk_pre,
            whisk_capture,
            attached_tool="whisk",
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.0,
            recipe_stage="whisked",
        ),
        _segment(
            "whisk_release",
            "Whisk retained in rack",
            0.9,
            whisk_capture,
            whisk_pre,
            recipe_stage="whisked",
            event="whisk_release",
        ),
        _segment(
            "milk_gripper_approach",
            "Reacquire stock gripper",
            1.1,
            whisk_pre,
            gripper_pre,
            recipe_stage="whisked",
        ),
        _segment(
            "milk_gripper_capture",
            "Capture gripper • verify ID-6",
            1.4,
            gripper_pre,
            gripper_capture,
            recipe_stage="whisked",
            event="gripper_recapture",
        ),
        _segment(
            "milk_gripper_exit",
            "Lock gripper and leave rack",
            1.1,
            gripper_capture,
            gripper_exit,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.0,
            slider_end_m=0.003,
            recipe_stage="whisked",
        ),
        _segment(
            "milk_pick",
            "Stock gripper takes milk pitcher",
            1.5,
            gripper_exit,
            GRIPPER_MILK_PICK_Q,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            pitcher="milk",
            gripper_start_rad=0.90,
            gripper_end_rad=0.10,
            recipe_stage="whisked",
        ),
        _segment(
            "milk_pour",
            "Finish with steamed milk",
            1.8,
            GRIPPER_MILK_PICK_Q,
            GRIPPER_MILK_POUR_Q,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            pitcher="milk",
            pitcher_held=True,
            gripper_start_rad=0.10,
            gripper_end_rad=0.10,
            pour_fraction_end=1.0,
            recipe_stage="whisked",
            recipe_stage_end="milk_added",
            event="milk_added",
        ),
        _segment(
            "milk_return",
            "Return milk pitcher",
            1.3,
            GRIPPER_MILK_POUR_Q,
            GRIPPER_MILK_PICK_Q,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            pitcher="milk",
            pitcher_held=True,
            gripper_start_rad=0.10,
            gripper_end_rad=0.90,
            recipe_stage="milk_added",
        ),
        _segment(
            "final_gripper_return",
            "Return gripper to rack",
            1.5,
            GRIPPER_MILK_PICK_Q,
            gripper_exit,
            attached_tool="gripper",
            locked=True,
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.003,
            recipe_stage="milk_added",
        ),
        _segment(
            "final_gripper_insert",
            "Passive cam opens lock",
            1.1,
            gripper_exit,
            gripper_capture,
            attached_tool="gripper",
            bus_connected=True,
            slider_start_m=0.003,
            slider_end_m=0.0,
            recipe_stage="milk_added",
        ),
        _segment(
            "final_release",
            "All tools returned • Matcha complete",
            1.5,
            gripper_capture,
            NEUTRAL_Q,
            recipe_stage="complete",
            event="workflow_complete",
        ),
        _segment(
            "final_hold",
            "Complete tool-change Matcha workflow",
            1.8,
            NEUTRAL_Q,
            NEUTRAL_Q,
            recipe_stage="complete",
        ),
    )
    errors = showcase_contract_errors(segments)
    if errors:
        raise RuntimeError("invalid showcase timeline: " + ", ".join(errors))
    return segments


@dataclass(frozen=True)
class ShowcaseState:
    segment_index: int
    segment_name: str
    caption: str
    elapsed_s: float
    total_time_s: float
    progress: float
    arm_q: tuple[float, ...]
    attached_tool: str | None
    locked: bool
    bus_connected: bool
    slider_q_m: float
    gripper_q_rad: float
    whisk_rotor_rad: float
    whisk_motor: bool
    pitcher: str | None
    pitcher_held: bool
    pour_fraction: float
    recipe_stage: str


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def showcase_duration_s(segments: tuple[ShowcaseSegment, ...] | None = None) -> float:
    rows = showcase_segments() if segments is None else segments
    return math.fsum(segment.duration_s for segment in rows)


def showcase_state_at(
    time_s: float,
    segments: tuple[ShowcaseSegment, ...] | None = None,
) -> ShowcaseState:
    """Sample the deterministic timeline, including its exact final state."""

    rows = showcase_segments() if segments is None else segments
    if not math.isfinite(time_s):
        raise ValueError("showcase time must be finite")
    total = showcase_duration_s(rows)
    clamped = min(total, max(0.0, float(time_s)))
    cursor = 0.0
    selected_index = len(rows) - 1
    local = rows[-1].duration_s
    for index, segment in enumerate(rows):
        end = cursor + segment.duration_s
        if clamped < end or index == len(rows) - 1:
            selected_index = index
            local = clamped - cursor
            break
        cursor = end
    segment = rows[selected_index]
    raw = min(1.0, max(0.0, local / segment.duration_s))
    smooth = _smoothstep(raw)
    start = np.asarray(segment.start_q, dtype=np.float64)
    end = np.asarray(segment.end_q, dtype=np.float64)
    arm_q = start + smooth * (end - start)
    recipe_stage = segment.recipe_stage
    if segment.recipe_stage_end is not None and raw >= 0.58:
        recipe_stage = segment.recipe_stage_end
    slider = segment.slider_start_m + smooth * (
        segment.slider_end_m - segment.slider_start_m
    )
    gripper = segment.gripper_start_rad + smooth * (
        segment.gripper_end_rad - segment.gripper_start_rad
    )
    pour = segment.pour_fraction_start + smooth * (
        segment.pour_fraction_end - segment.pour_fraction_start
    )
    return ShowcaseState(
        segment_index=selected_index,
        segment_name=segment.name,
        caption=segment.caption,
        elapsed_s=clamped,
        total_time_s=total,
        progress=(clamped / total if total else 1.0),
        arm_q=tuple(float(value) for value in arm_q),
        attached_tool=segment.attached_tool,
        locked=segment.locked,
        bus_connected=segment.bus_connected,
        slider_q_m=float(slider),
        gripper_q_rad=float(gripper),
        whisk_rotor_rad=(24.0 * local if segment.whisk_motor else 0.0),
        whisk_motor=segment.whisk_motor,
        pitcher=segment.pitcher,
        pitcher_held=segment.pitcher_held,
        pour_fraction=float(pour),
        recipe_stage=recipe_stage,
    )


def showcase_contract_errors(
    segments: tuple[ShowcaseSegment, ...] | None = None,
) -> list[str]:
    """Return fail-closed errors for the visual story contract."""

    rows = showcase_segments() if segments is None else segments
    errors: list[str] = []
    if not rows:
        return ["timeline_empty"]
    names = [segment.name for segment in rows]
    if len(names) != len(set(names)):
        errors.append("segment_names_not_unique")
    events = [segment.event for segment in rows if segment.event is not None]
    required_events = (
        "gripper_capture",
        "gripper_lock",
        "hot_water_added",
        "gripper_release",
        "spoon_capture",
        "matcha_dosed",
        "spoon_release",
        "whisk_capture",
        "matcha_whisked",
        "whisk_release",
        "gripper_recapture",
        "milk_added",
        "workflow_complete",
    )
    positions = []
    for event in required_events:
        if event not in events:
            errors.append(f"missing_event:{event}")
        else:
            positions.append(events.index(event))
    if len(positions) == len(required_events) and positions != sorted(positions):
        errors.append("event_order")
    attached = {segment.attached_tool for segment in rows}
    if not set(SHOWCASE_TOOLS).issubset(attached):
        errors.append("all_tools_not_attached")
    if not any(
        segment.pitcher == "hot_water" and segment.attached_tool == "gripper"
        for segment in rows
    ):
        errors.append("gripper_hot_water_missing")
    if not any(
        segment.pitcher == "milk" and segment.attached_tool == "gripper"
        for segment in rows
    ):
        errors.append("gripper_milk_missing")
    if not any(segment.whisk_motor for segment in rows):
        errors.append("powered_whisk_missing")
    if rows[-1].recipe_stage != "complete":
        errors.append("final_recipe_stage")
    if rows[-1].attached_tool is not None:
        errors.append("final_tool_not_returned")
    if SHOWCASE_CAMERA != "matcha_scene_camera":
        errors.append("single_scene_camera_identity")
    for index, segment in enumerate(rows[1:], start=1):
        previous = rows[index - 1]
        if not np.allclose(
            np.asarray(previous.end_q),
            np.asarray(segment.start_q),
            rtol=0.0,
            atol=1.0e-12,
        ):
            errors.append(f"arm_discontinuity:{previous.name}:{segment.name}")
    return errors


def showcase_summary() -> dict[str, object]:
    """Return concise machine-readable video acceptance evidence."""

    rows = showcase_segments()
    events = [segment.event for segment in rows if segment.event is not None]
    return {
        "schema_version": SHOWCASE_SCHEMA_VERSION,
        "camera": SHOWCASE_CAMERA,
        "camera_count": 1,
        "duration_s": showcase_duration_s(rows),
        "segment_count": len(rows),
        "tools": list(SHOWCASE_TOOLS),
        "events": events,
        "gripper_pitchers": ["hot_water", "milk"],
        "spoon_task": "dose_matcha_through_sieve",
        "whisk_task": "powered_mix_in_bowl",
        "all_tools_returned": rows[-1].attached_tool is None,
        "final_recipe_stage": rows[-1].recipe_stage,
        "visualization_only": True,
        "physical_release_authority": False,
        "release_ready": False,
        "errors": showcase_contract_errors(rows),
        "passed": not showcase_contract_errors(rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(showcase_summary(), indent=2, sort_keys=True))
