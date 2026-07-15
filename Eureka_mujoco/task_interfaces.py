from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskInterface:
    interface_type: str
    score_display_name: str
    score_description: str
    obs_context_relpath: str
    uses_success_semantics: bool


TASK_INTERFACES = {
    "Ant": TaskInterface(
        interface_type="ant",
        score_display_name="forward_score",
        score_description="forward_score is the ground-truth locomotion metric based on forward velocity along the x-axis. Higher is better.",
        obs_context_relpath="envs/mujoco/ant_v4.py",
        uses_success_semantics=False,
    ),
    "Walker": TaskInterface(
        interface_type="walker",
        score_display_name="forward_score",
        score_description="forward_score is the ground-truth locomotion metric based on forward velocity along the x-axis while staying upright. Higher is better.",
        obs_context_relpath="envs/mujoco/walker2d_v4.py",
        uses_success_semantics=False,
    ),
    "Reacher": TaskInterface(
        interface_type="reach",
        score_display_name="success_rate",
        score_description="success_rate is the ground-truth manipulation metric. Higher is better, with 1 meaning the task succeeded on that step.",
        obs_context_relpath="envs/metaworld/sawyer_reach_v3.py",
        uses_success_semantics=True,
    ),
    "Pick": TaskInterface(
        interface_type="pick",
        score_display_name="success_rate",
        score_description="success_rate is the ground-truth manipulation metric. Higher is better, with 1 meaning the task succeeded on that step.",
        obs_context_relpath="envs/metaworld/sawyer_pick_place_v3.py",
        uses_success_semantics=True,
    ),
    "Door": TaskInterface(
        interface_type="door",
        score_display_name="success_rate",
        score_description="success_rate is the ground-truth manipulation metric. Higher is better, with 1 meaning the task succeeded on that step.",
        obs_context_relpath="envs/metaworld/sawyer_door_v3.py",
        uses_success_semantics=True,
    ),
}


def get_task_interface(task_name: str) -> TaskInterface:
    base_name = task_name[:-3] if task_name.endswith("GPT") else task_name
    if base_name not in TASK_INTERFACES:
        raise KeyError(f"unknown task interface for {task_name!r}")
    return TASK_INTERFACES[base_name]
