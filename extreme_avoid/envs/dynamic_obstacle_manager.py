"""
Dynamic Obstacle Manager for habitat-sim.

Manages a collection of rigid objects with kinematic motion models.
Each obstacle maintains (position, velocity) state updated per simulation step
via rigid object handle translation updates.

Motion patterns (selectable per-obstacle):
  - "constant_velocity": Linear motion at fixed velocity.
  - "sinusoidal": Oscillating motion along user-defined axis.
  - "trajectory_replay": Pre-recorded waypoint sequence.

CRITICAL (per skill.md §3.7):
  - Ground truth (manager attributes) and estimated values (from perception)
    MUST use different variable names — never mix them.
  - For num_envs>1, performance is unverified; validate with num_envs=1 first.
"""

import torch as th
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum

try:
    import habitat_sim
    from habitat_sim.physics import ManagedRigidObject, MotionType as HabitatMotionType
    try:
        import magnum as mn
    except ImportError:
        mn = None
    _HABITAT_AVAILABLE = True
except ImportError:
    _HABITAT_AVAILABLE = False
    ManagedRigidObject = None
    HabitatMotionType = None
    mn = None


class MotionPattern(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    SINUSOIDAL = "sinusoidal"
    TRAJECTORY_REPLAY = "trajectory_replay"


class DynamicObstacleManager:
    """
    Manages dynamic obstacles in a habitat-sim scene.

    Each obstacle:
      - Is a rigid object obtained via scene.get_rigid_object_manager()
      - Has a motion pattern (constant_velocity, sinusoidal, trajectory_replay)
      - Is stepped each simulation cycle to update its translation.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: th.device = th.device("cpu"),
        scene_manager=None,  # habitat-sim SceneManager-like handle, set after init
    ):
        self.num_envs = num_envs
        self.device = device
        self._scene_manager = scene_manager

        # Obstacle state buffers — GROUND TRUTH (for reward supervision only)
        # Shape: (num_envs, max_obstacles, ...)
        self._num_obstacles = th.zeros(num_envs, dtype=th.long, device=device)
        self._obstacle_positions: List[List[th.Tensor]] = [
            [] for _ in range(num_envs)
        ]  # per-env list of (3,) tensors
        self._obstacle_velocities: List[List[th.Tensor]] = [
            [] for _ in range(num_envs)
        ]
        self._obstacle_handles: List[List[Any]] = [
            [] for _ in range(num_envs)
        ]  # ManagedRigidObject references (not IDs)

        # Motion parameters per obstacle: (pattern, params_dict, phase)
        self._motion_configs: List[List[Dict]] = [[] for _ in range(num_envs)]

        # Bounding box radius per obstacle (for collision approximation)
        self._obstacle_radii: List[List[float]] = [[] for _ in range(num_envs)]

        self._t = 0.0  # simulation time accumulator

    def set_scene_manager(self, scene_manager):
        """Attach the habitat-sim scene manager after construction."""
        self._scene_manager = scene_manager

    def add_obstacle(
        self,
        env_id: int,
        initial_position: Union[th.Tensor, np.ndarray, List[float]],
        initial_velocity: Union[th.Tensor, np.ndarray, List[float]],
        motion_pattern: Union[str, MotionPattern] = MotionPattern.CONSTANT_VELOCITY,
        motion_params: Optional[Dict] = None,
        obstacle_radius: float = 0.3,
        rigid_object: Any = None,  # ManagedRigidObject — if None, caller must spawn
    ):
        """
        Register a dynamic obstacle for environment `env_id`.

        Args:
            env_id: Which parallel env this obstacle belongs to.
            initial_position: World-frame starting position (3,).
            initial_velocity: World-frame initial velocity (3,).
            motion_pattern: Motion model type.
            motion_params: Pattern-specific parameters.
            obstacle_radius: Approximate radius for collision checks.
            rigid_object: habitat_sim.physics.ManagedRigidObject reference.
                CRITICAL: Caller must spawn this BEFORE calling add_obstacle.
                The object should have MotionType.KINEMATIC set.
        """
        if isinstance(motion_pattern, str):
            motion_pattern = MotionPattern(motion_pattern)
        motion_params = motion_params or {}

        init_pos = th.as_tensor(
            initial_position, dtype=th.float32, device=self.device
        )
        init_vel = th.as_tensor(
            initial_velocity, dtype=th.float32, device=self.device
        )

        self._obstacle_positions[env_id].append(init_pos)
        self._obstacle_velocities[env_id].append(init_vel)
        self._obstacle_radii[env_id].append(obstacle_radius)
        self._obstacle_handles[env_id].append(rigid_object)
        self._motion_configs[env_id].append({
            "pattern": motion_pattern,
            "params": motion_params,
            "phase": 0.0,
            "init_position": init_pos.clone(),
            "init_velocity": init_vel.clone(),
        })
        self._num_obstacles[env_id] += 1

        # Set initial position on the actual rigid object
        if rigid_object is not None:
            try:
                pos_np = init_pos.detach().cpu().numpy()
                rigid_object.translation = mn.Vector3(pos_np) if mn is not None else pos_np
            except Exception:
                pass

    def step(self, dt: float):
        """
        Advance all dynamic obstacles by one simulation step.

        Called from DynamicAvoidanceEnv.step() / reset() alongside drone dynamics.

        Args:
            dt: Simulation timestep (same as env.dynamics.ctrl_dt).
        """
        self._t += dt

        for env_id in range(self.num_envs):
            for obs_idx in range(self._num_obstacles[env_id].item()):
                cfg = self._motion_configs[env_id][obs_idx]
                new_pos, new_vel = self._compute_motion(
                    cfg, self._t, dt
                )
                self._obstacle_positions[env_id][obs_idx] = new_pos
                self._obstacle_velocities[env_id][obs_idx] = new_vel

                # Update rigid object in simulation via stored ManagedRigidObject
                rigid_obj = self._obstacle_handles[env_id][obs_idx]
                if rigid_obj is not None:
                    pos_np = new_pos.detach().cpu().numpy()
                    try:
                        rigid_obj.translation = mn.Vector3(pos_np) if mn is not None else pos_np
                    except Exception:
                        pass

    def _compute_motion(
        self,
        config: Dict,
        t: float,
        dt: float,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute new position and velocity for one obstacle based on its motion pattern.

        Returns:
            (new_position, new_velocity) — both (3,) tensors.
        """
        pattern = config["pattern"]
        params = config["params"]
        init_pos = config["init_position"]
        init_vel = config["init_velocity"]

        if pattern == MotionPattern.CONSTANT_VELOCITY:
            # Simple linear extrapolation
            new_pos = init_pos + init_vel * t
            new_vel = init_vel.clone()

        elif pattern == MotionPattern.SINUSOIDAL:
            axis = th.as_tensor(
                params.get("axis", [1.0, 0.0, 0.0]),
                dtype=th.float32, device=self.device
            )
            amplitude = params.get("amplitude", 1.0)
            frequency = params.get("frequency", 0.5)
            axis = axis / (axis.norm() + 1e-8)

            t_tensor = th.tensor(t, dtype=th.float32, device=self.device)
            displacement = amplitude * th.sin(2 * np.pi * frequency * t_tensor)
            new_pos = init_pos + axis * displacement
            new_vel = axis * (amplitude * 2 * np.pi * frequency * th.cos(2 * np.pi * frequency * t_tensor))

        elif pattern == MotionPattern.TRAJECTORY_REPLAY:
            waypoints = th.as_tensor(
                params["waypoints"], dtype=th.float32, device=self.device
            )  # (T, 3)
            loop = params.get("loop", True)
            speed = params.get("speed", 1.0)

            # Linear interpolation between waypoints
            num_wp = waypoints.shape[0]
            if loop:
                # Extend waypoints cyclically for looping
                wp_ext = th.cat([waypoints, waypoints[:1]], dim=0)
            else:
                wp_ext = waypoints

            # Compute segment lengths and cumulative distances
            seg_vectors = wp_ext[1:] - wp_ext[:-1]
            seg_lengths = seg_vectors.norm(dim=1)  # (T,)
            cum_lengths = th.cat([
                th.zeros(1, device=self.device),
                th.cumsum(seg_lengths, dim=0)
            ])
            total_length = cum_lengths[-1]

            if total_length < 1e-8:
                new_pos = init_pos
                new_vel = th.zeros(3, device=self.device)
            else:
                traveled = (speed * t) % total_length if loop else min(speed * t, total_length)
                # Find which segment we're in
                seg_idx = th.searchsorted(cum_lengths[1:], traveled, right=False)
                seg_idx = min(seg_idx, num_wp - 1)
                local_t = (traveled - cum_lengths[seg_idx]) / (seg_lengths[seg_idx] + 1e-8)
                local_t = local_t.clamp(0.0, 1.0)
                new_pos = wp_ext[seg_idx] + local_t * seg_vectors[seg_idx]
                new_vel = (seg_vectors[seg_idx] / (seg_lengths[seg_idx] + 1e-8)) * speed

        else:
            new_pos = init_pos
            new_vel = init_vel

        return new_pos, new_vel

    # --- Ground Truth Accessors (for reward supervision, NEVER for policy input) ---

    def get_obstacle_positions_gt(self, env_id: int = 0) -> th.Tensor:
        """
        Ground truth obstacle positions for env_id.
        Shape: (num_obstacles, 3).

        NAMING: "gt" suffix explicitly marks this as ground truth.
        Policy must NEVER receive this as observation input.
        """
        if self._num_obstacles[env_id] == 0:
            return th.zeros((0, 3), device=self.device)
        return th.stack(self._obstacle_positions[env_id], dim=0)

    def get_obstacle_velocities_gt(self, env_id: int = 0) -> th.Tensor:
        """Ground truth obstacle velocities. Shape: (num_obstacles, 3)."""
        if self._num_obstacles[env_id] == 0:
            return th.zeros((0, 3), device=self.device)
        return th.stack(self._obstacle_velocities[env_id], dim=0)

    def get_obstacle_radii(self, env_id: int = 0) -> th.Tensor:
        """Obstacle bounding radii. Shape: (num_obstacles,)."""
        if self._num_obstacles[env_id] == 0:
            return th.zeros(0, device=self.device)
        return th.tensor(
            self._obstacle_radii[env_id], dtype=th.float32, device=self.device
        )

    def get_all_positions_gt(self) -> List[th.Tensor]:
        """Returns list of (num_obstacles, 3) tensors, one per env."""
        return [self.get_obstacle_positions_gt(i) for i in range(self.num_envs)]

    def get_all_velocities_gt(self) -> List[th.Tensor]:
        """Returns list of (num_obstacles, 3) tensors, one per env."""
        return [self.get_obstacle_velocities_gt(i) for i in range(self.num_envs)]

    def num_obstacles(self, env_id: int = 0) -> int:
        return self._num_obstacles[env_id].item()
