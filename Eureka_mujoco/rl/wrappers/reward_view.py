from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RewardView:
    x_velocity: float
    y_velocity: float
    forward_reward: float
    healthy_reward: float
    control_cost: float
    control_cost_weight: float
    contact_cost: float
    contact_cost_weight: float
    env_reward: float
    terminated: bool
    is_healthy: bool
    x_position: float
    y_position: float
    z_position: float
    distance_from_origin: float
    dt: float
    healthy_z_min: float
    healthy_z_max: float
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class WalkerRewardView:
    x_velocity: float
    forward_reward: float
    healthy_reward: float
    control_cost: float
    control_cost_weight: float
    env_reward: float
    terminated: bool
    is_healthy: bool
    x_position: float
    z_position: float
    torso_angle: float
    dt: float
    healthy_z_min: float
    healthy_z_max: float
    healthy_angle_min: float
    healthy_angle_max: float
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class AntRewardView:
    x_velocity: float
    y_velocity: float
    forward_reward: float
    healthy_reward: float
    control_cost: float
    control_cost_weight: float
    contact_cost: float
    contact_cost_weight: float
    env_reward: float
    terminated: bool
    is_healthy: bool
    x_position: float
    y_position: float
    z_position: float
    distance_from_origin: float
    dt: float
    healthy_z_min: float
    healthy_z_max: float
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class GeometricControlRewardView:
    fingertip_pos: np.ndarray
    target_pos: np.ndarray
    fingertip_to_target_vec: np.ndarray
    target_distance: float
    reward_dist: float
    reward_ctrl: float
    env_reward: float
    dt: float
    terminated: bool
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class ManipulationRewardView:
    tcp_center: np.ndarray
    gripper_opening: float
    left_pad_pos: np.ndarray
    right_pad_pos: np.ndarray
    init_tcp: np.ndarray
    init_left_pad: np.ndarray
    init_right_pad: np.ndarray
    obj_pos: np.ndarray
    target_pos: np.ndarray
    obj_init_pos: np.ndarray
    obj_height: float
    height_target: float
    touching_main_object: bool
    dt: float
    terminated: bool
    qpos: np.ndarray
    qvel: np.ndarray


def build_reward_view(env, obs, action, info, env_reward, terminated) -> RewardView:
    qpos = np.array(env.data.qpos.flat.copy(), dtype=np.float64)
    qvel = np.array(env.data.qvel.flat.copy(), dtype=np.float64)
    healthy_z_range = getattr(env, "_healthy_z_range", (float("-inf"), float("inf")))
    control_cost = float(env.control_cost(action)) if hasattr(env, "control_cost") else 0.0
    contact_cost = float(getattr(env, "contact_cost", 0.0))
    return RewardView(
        x_velocity=float(info.get("x_velocity", qvel[0] if len(qvel) > 0 else 0.0)),
        y_velocity=float(info.get("y_velocity", 0.0)),
        forward_reward=float(info.get("reward_forward", info.get("forward_reward", 0.0))),
        healthy_reward=float(getattr(env, "healthy_reward", 0.0)),
        control_cost=control_cost,
        control_cost_weight=float(getattr(env, "_ctrl_cost_weight", 0.0)),
        contact_cost=contact_cost,
        contact_cost_weight=float(getattr(env, "_contact_cost_weight", 0.0)),
        env_reward=float(env_reward),
        terminated=bool(terminated),
        is_healthy=bool(getattr(env, "is_healthy", not terminated)),
        x_position=float(info.get("x_position", qpos[0] if len(qpos) > 0 else 0.0)),
        y_position=float(info.get("y_position", qpos[1] if len(qpos) > 1 else 0.0)),
        z_position=float(qpos[2] if len(qpos) > 2 else 0.0),
        distance_from_origin=float(info.get("distance_from_origin", 0.0)),
        dt=float(getattr(env, "dt", 0.0)),
        healthy_z_min=float(healthy_z_range[0]),
        healthy_z_max=float(healthy_z_range[1]),
        qpos=qpos,
        qvel=qvel,
    )


def build_walker_reward_view(env, obs, action, info, env_reward, terminated) -> WalkerRewardView:
    qpos = np.array(env.data.qpos.flat.copy(), dtype=np.float64)
    qvel = np.array(env.data.qvel.flat.copy(), dtype=np.float64)
    healthy_z_range = getattr(env, "_healthy_z_range", (float("-inf"), float("inf")))
    healthy_angle_range = getattr(env, "_healthy_angle_range", (float("-inf"), float("inf")))
    control_cost = float(env.control_cost(action)) if hasattr(env, "control_cost") else 0.0
    return WalkerRewardView(
        x_velocity=float(info.get("x_velocity", qvel[0] if len(qvel) > 0 else 0.0)),
        forward_reward=float(info.get("reward_forward", info.get("forward_reward", 0.0))),
        healthy_reward=float(getattr(env, "healthy_reward", 0.0)),
        control_cost=control_cost,
        control_cost_weight=float(getattr(env, "_ctrl_cost_weight", 0.0)),
        env_reward=float(env_reward),
        terminated=bool(terminated),
        is_healthy=bool(getattr(env, "is_healthy", not terminated)),
        x_position=float(info.get("x_position", qpos[0] if len(qpos) > 0 else 0.0)),
        z_position=float(qpos[1] if len(qpos) > 1 else 0.0),
        torso_angle=float(qpos[2] if len(qpos) > 2 else 0.0),
        dt=float(getattr(env, "dt", 0.0)),
        healthy_z_min=float(healthy_z_range[0]),
        healthy_z_max=float(healthy_z_range[1]),
        healthy_angle_min=float(healthy_angle_range[0]),
        healthy_angle_max=float(healthy_angle_range[1]),
        qpos=qpos,
        qvel=qvel,
    )


def build_ant_reward_view(env, obs, action, info, env_reward, terminated) -> AntRewardView:
    qpos = np.array(env.data.qpos.flat.copy(), dtype=np.float64)
    qvel = np.array(env.data.qvel.flat.copy(), dtype=np.float64)
    healthy_z_range = getattr(env, "_healthy_z_range", (float("-inf"), float("inf")))
    control_cost = float(env.control_cost(action)) if hasattr(env, "control_cost") else 0.0
    contact_cost = float(getattr(env, "contact_cost", 0.0))
    return AntRewardView(
        x_velocity=float(info.get("x_velocity", 0.0)),
        y_velocity=float(info.get("y_velocity", 0.0)),
        forward_reward=float(info.get("reward_forward", info.get("forward_reward", 0.0))),
        healthy_reward=float(getattr(env, "healthy_reward", 0.0)),
        control_cost=control_cost,
        control_cost_weight=float(getattr(env, "_ctrl_cost_weight", 0.0)),
        contact_cost=contact_cost,
        contact_cost_weight=float(getattr(env, "_contact_cost_weight", 0.0)),
        env_reward=float(env_reward),
        terminated=bool(terminated),
        is_healthy=bool(getattr(env, "is_healthy", not terminated)),
        x_position=float(info.get("x_position", qpos[0] if len(qpos) > 0 else 0.0)),
        y_position=float(info.get("y_position", qpos[1] if len(qpos) > 1 else 0.0)),
        z_position=float(qpos[2] if len(qpos) > 2 else 0.0),
        distance_from_origin=float(info.get("distance_from_origin", 0.0)),
        dt=float(getattr(env, "dt", 0.0)),
        healthy_z_min=float(healthy_z_range[0]),
        healthy_z_max=float(healthy_z_range[1]),
        qpos=qpos,
        qvel=qvel,
    )


def build_geometric_control_reward_view(env, obs, action, info, env_reward, terminated) -> GeometricControlRewardView:
    qpos = np.array(env.data.qpos.flat.copy(), dtype=np.float64)
    qvel = np.array(env.data.qvel.flat.copy(), dtype=np.float64)
    fingertip_pos = np.array(env.get_body_com("fingertip"), dtype=np.float64)
    target_pos = np.array(env.get_body_com("target"), dtype=np.float64)
    fingertip_to_target_vec = fingertip_pos - target_pos
    target_distance = float(np.linalg.norm(fingertip_to_target_vec))
    reward_dist = float(info.get("reward_dist", -target_distance))
    reward_ctrl = float(info.get("reward_ctrl", -env.control_cost(action) if hasattr(env, "control_cost") else 0.0))
    return GeometricControlRewardView(
        fingertip_pos=fingertip_pos,
        target_pos=target_pos,
        fingertip_to_target_vec=fingertip_to_target_vec,
        target_distance=target_distance,
        reward_dist=reward_dist,
        reward_ctrl=reward_ctrl,
        env_reward=float(env_reward),
        dt=float(getattr(env, "dt", 0.0)),
        terminated=bool(terminated),
        qpos=qpos,
        qvel=qvel,
    )


def _safe_body_com(env, name: str, fallback: np.ndarray) -> np.ndarray:
    try:
        return np.array(env.get_body_com(name), dtype=np.float64)
    except Exception:
        return fallback


def _safe_touching_main_object(env) -> bool:
    try:
        return bool(getattr(env, "touching_main_object", False))
    except Exception:
        return False


def build_manipulation_reward_view(env, obs, action, info, env_reward, terminated) -> ManipulationRewardView:
    obs_arr = np.asarray(obs, dtype=np.float64)
    obj_pos = obs_arr[4:7] if obs_arr.shape[0] >= 7 else np.zeros(3, dtype=np.float64)
    gripper_opening = float(obs_arr[3]) if obs_arr.shape[0] >= 4 else 0.0

    zero3 = np.zeros(3, dtype=np.float64)
    tcp_center = np.array(getattr(env, "tcp_center", zero3), dtype=np.float64)
    init_tcp = np.array(getattr(env, "init_tcp", tcp_center), dtype=np.float64)
    init_left_pad = np.array(getattr(env, "init_left_pad", zero3), dtype=np.float64)
    init_right_pad = np.array(getattr(env, "init_right_pad", zero3), dtype=np.float64)
    left_pad_pos = _safe_body_com(env, "leftpad", init_left_pad)
    right_pad_pos = _safe_body_com(env, "rightpad", init_right_pad)

    target_pos = np.array(getattr(env, "_target_pos", zero3) if getattr(env, "_target_pos", None) is not None else zero3, dtype=np.float64)
    obj_init_pos = np.array(getattr(env, "obj_init_pos", zero3) if getattr(env, "obj_init_pos", None) is not None else zero3, dtype=np.float64)

    qpos = np.array(env.data.qpos.flat.copy(), dtype=np.float64) if hasattr(env, "data") else np.zeros(0, dtype=np.float64)
    qvel = np.array(env.data.qvel.flat.copy(), dtype=np.float64) if hasattr(env, "data") else np.zeros(0, dtype=np.float64)

    return ManipulationRewardView(
        tcp_center=tcp_center,
        gripper_opening=gripper_opening,
        left_pad_pos=left_pad_pos,
        right_pad_pos=right_pad_pos,
        init_tcp=init_tcp,
        init_left_pad=init_left_pad,
        init_right_pad=init_right_pad,
        obj_pos=obj_pos,
        target_pos=target_pos,
        obj_init_pos=obj_init_pos,
        obj_height=float(getattr(env, "objHeight", obj_init_pos[2] if obj_init_pos.shape[0] >= 3 else 0.0)),
        height_target=float(getattr(env, "heightTarget", (obj_init_pos[2] if obj_init_pos.shape[0] >= 3 else 0.0) + 0.04)),
        touching_main_object=_safe_touching_main_object(env),
        dt=float(getattr(env, "dt", 0.0)),
        terminated=bool(terminated),
        qpos=qpos,
        qvel=qvel,
    )
