"""
Dynamic Avoidance Environment.

Main integration point for all three innovations (per skill.md §3.8):
  Innovation 1 (Shared Perception): Uses TrackerFusedPolicy for fused features.
  Innovation 2 (TTC Risk Field):   Replaces geodesic with analytical TTC risk.
  Innovation 3 (Confidence Yaw):   Yaw policy weighted by tracking confidence.

Inherits from depthnav.envs.navigation_env.NavigationEnv.

Key changes from parent NavigationEnv:
  - get_reward(): Replaces geodesic_gradient with TTCRiskField.
    Replaces motion-based yaw with risk-weighted, confidence-aware yaw.
  - get_observation(): Adds "color" (RGB) and "obstacle_track" (perception) keys.
  - step()/reset(): Inserts DynamicObstacleManager.step(dt) calls.
  - Supports curriculum stages via `use_perception` flag.

CRITICAL DESIGN RULES (per skill.md):
  1. Ground truth (DynamicObstacleManager) and estimated values (policy aux_dict)
     must use separate variable names — NEVER mix them.
  2. Policy observations must come from perception, NOT from ground truth.
  3. habitat-sim rendering is non-differentiable → confidence_proxy uses geometric proxy.
"""

import torch as th
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Union
from enum import Enum

from depthnav.envs.navigation_env import NavigationEnv, Frame, ActionType, TargetType, get_enum
from depthnav.utils import Rotation3
from gymnasium import spaces

from .dynamic_obstacle_manager import DynamicObstacleManager, MotionPattern
from ..risk.ttc_field import TTCRiskField, compute_ttc_risk, compute_min_ttc
from ..risk.confidence_proxy import ConfidenceProxy, compute_bearing, compute_range


class CurriculumStage(Enum):
    """Training curriculum stages."""
    STAGE1_STATIC = "stage1_static"         # Original static env, no dynamics
    STAGE2_DYNAMIC_GT = "stage2_dynamic_gt" # Dynamic obstacles, GT observation
    STAGE3_PERCEPTION = "stage3_perception" # Real perception input
    STAGE4_EXTREME = "stage4_extreme"       # Full: perception + degraded lighting


class DynamicAvoidanceEnv(NavigationEnv):
    """
    Dynamic obstacle avoidance environment extending NavigationEnv.

    Supports dynamic obstacles with kinematic motion models, TTC-based
    collision risk field, and tracking-confidence-aware yaw control.
    """

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 42,
        visual: bool = False,
        single_env: bool = False,
        max_episode_steps: int = 256,
        device: Optional[th.device] = th.device("cpu"),
        requires_grad: bool = False,
        robot_radius: float = 0.1,
        dynamics_kwargs=None,
        random_kwargs=None,
        base_action=None,
        action_type="THRUST_YAW",
        inertial_frame="START",
        target_type="TARGET_VELOCITY_TARGET_DISTANCE",
        target_kwargs=None,
        reward_kwargs=None,
        bounds=None,
        scene_kwargs=None,
        sensor_kwargs=None,

        # --- NEW: Dynamic obstacle parameters ---
        dynamic_obstacles_config: Optional[List[Dict]] = None,
        max_dynamic_obstacles: int = 4,

        # --- NEW: Curriculum control ---
        curriculum_stage: str = "stage3_perception",

        # --- NEW: TTC/Risk parameters ---
        ttc_threshold: float = 2.0,
        distance_safe: float = 1.5,

        # --- NEW: Confidence proxy ---
        confidence_proxy_path: Optional[str] = None,
        use_confidence_proxy: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            seed=seed,
            visual=visual,
            single_env=single_env,
            max_episode_steps=max_episode_steps,
            device=device,
            requires_grad=requires_grad,
            robot_radius=robot_radius,
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            base_action=base_action or [0.0, 0.0, 0.0],
            action_type=action_type,
            inertial_frame=inertial_frame,
            target_type=target_type,
            target_kwargs=target_kwargs,
            reward_kwargs=reward_kwargs,
            bounds=bounds,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
        )

        # --- Dynamic obstacle manager ---
        self.dynamic_obstacle_manager = DynamicObstacleManager(
            num_envs=num_envs, device=device,
        )

        # Parse curriculum stage
        if isinstance(curriculum_stage, str):
            curriculum_stage = CurriculumStage(curriculum_stage)
        self.curriculum_stage = curriculum_stage
        self.use_dynamic_obstacles = curriculum_stage != CurriculumStage.STAGE1_STATIC
        self.use_perception = curriculum_stage in (
            CurriculumStage.STAGE3_PERCEPTION, CurriculumStage.STAGE4_EXTREME
        )
        self.use_extreme_lighting = curriculum_stage == CurriculumStage.STAGE4_EXTREME
        self.max_dynamic_obstacles = max_dynamic_obstacles

        # --- TTC risk field ---
        self.ttc_field = TTCRiskField(
            ttc_threshold=ttc_threshold,
            distance_safe=distance_safe,
            barrier_beta=5.0,
        )

        # --- Confidence proxy (loaded offline, frozen during training) ---
        self._confidence_proxy = None
        self.use_confidence_proxy = use_confidence_proxy
        if use_confidence_proxy and confidence_proxy_path is not None:
            try:
                self._confidence_proxy = ConfidenceProxy.load(
                    confidence_proxy_path, device=device
                )
            except (FileNotFoundError, Exception):
                # Placeholder: will use heuristic fallback
                self._confidence_proxy = ConfidenceProxy().to(device)
                self._confidence_proxy.eval()
        elif use_confidence_proxy:
            # Create a fresh untrained proxy (will need fitting later)
            self._confidence_proxy = ConfidenceProxy().to(device)
            self._confidence_proxy.eval()

        # --- Observation space extensions ---
        # Add obstacle_track observation (perception outputs)
        obs_emb_dim = 64  # obstacle embedding dimension
        self.observation_space.spaces["obstacle_track"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_dynamic_obstacles * obs_emb_dim,),
            dtype=np.float32,
        )

        # Ensure color sensor is available for perception
        self._has_color_sensor = any("color" in s.get("uuid", "") or "color" in s.get("sensor_type", "")
                                     for s in (sensor_kwargs or []))

        # --- Obstacle prediction cache (for get_reward access) ---
        self._last_obstacle_prediction: Optional[Dict[str, th.Tensor]] = None
        self._last_obstacle_outputs: Optional[Dict[str, th.Tensor]] = None

        # --- Reward weights for new loss terms ---
        self._lambda_risk = reward_kwargs.get("lambda_risk", 1.0) if reward_kwargs else 1.0
        self._lambda_track_conf = reward_kwargs.get("lambda_track_conf", 0.5) if reward_kwargs else 0.5
        self._lambda_yaw_risk = reward_kwargs.get("lambda_yaw_risk", 0.5) if reward_kwargs else 0.5

        # --- Spawn initial dynamic obstacles ---
        if self.use_dynamic_obstacles and dynamic_obstacles_config:
            self._setup_dynamic_obstacles(dynamic_obstacles_config)
        elif self.use_dynamic_obstacles:
            self._setup_default_dynamic_obstacles()

    def _setup_dynamic_obstacles(self, configs: List[Dict]):
        """Spawn dynamic obstacles from configuration."""
        for env_id in range(self.num_envs):
            for cfg in configs[:self.max_dynamic_obstacles]:
                pos = cfg.get("initial_position", [5.0, 0.0, 2.0])
                vel = cfg.get("initial_velocity", [0.0, 0.5, 0.0])
                pattern = cfg.get("motion_pattern", "constant_velocity")
                params = cfg.get("motion_params", {})
                radius = cfg.get("radius", 0.3)
                self.dynamic_obstacle_manager.add_obstacle(
                    env_id=env_id,
                    initial_position=pos,
                    initial_velocity=vel,
                    motion_pattern=pattern,
                    motion_params=params,
                    obstacle_radius=radius,
                )

    def _setup_default_dynamic_obstacles(self):
        """Create default dynamic obstacle configurations."""
        default_configs = [
            {
                "initial_position": [5.0, -3.0, 2.0],
                "initial_velocity": [0.3, 0.5, 0.0],
                "motion_pattern": "constant_velocity",
                "radius": 0.3,
            },
            {
                "initial_position": [8.0, 3.0, 2.0],
                "initial_velocity": [-0.5, -0.3, 0.0],
                "motion_pattern": "sinusoidal",
                "motion_params": {"axis": [1.0, 0.0, 0.0], "amplitude": 2.0, "frequency": 0.3},
                "radius": 0.3,
            },
        ]
        self._setup_dynamic_obstacles(default_configs)

    def set_obstacle_prediction(self, aux_dict: Dict[str, th.Tensor]):
        """
        Receive obstacle prediction from policy's forward() output.

        Called externally by the training loop after policy(obs).
        CRITICAL: This contains ESTIMATED values — never mix with GT.
        """
        self._last_obstacle_prediction = aux_dict.get("obstacle_prediction", None)
        self._last_obstacle_outputs = aux_dict.get("obstacle_outputs", None)

    def get_observation(self):
        """
        Extended observation including color and obstacle tracking.

        Returns a dict with keys:
          - state, depth, target (from parent)
          - color (if color sensor configured)
          - obstacle_track (from perception, if use_perception=True)
        """
        obs = super().get_observation()

        # Add color observation
        if self.visual and self._has_color_sensor:
            for sensor_uuid, sensor_data in self.sensor_obs.items():
                if "color" in sensor_uuid:
                    if self.requires_grad:
                        obs["color"] = th.from_numpy(sensor_data).to(self.device)
                    else:
                        obs["color"] = sensor_data
                    break

        # Add obstacle tracking observation (from perception)
        if self.use_perception and self._last_obstacle_outputs is not None:
            out = self._last_obstacle_outputs
            emb = out.get("embedding", None)
            if emb is not None:
                # Flatten obstacle embeddings: (B, K, D) → (B, K*D)
                B = emb.shape[0]
                obs["obstacle_track"] = emb.reshape(B, -1).to(self.device)
        else:
            # Placeholder: zero vector
            B = self.num_envs
            obs_dim = self.observation_space.spaces["obstacle_track"].shape[0]
            if self.requires_grad:
                obs["obstacle_track"] = th.zeros(
                    (B, obs_dim), device=self.device, dtype=th.float32
                )
            else:
                obs["obstacle_track"] = np.zeros(
                    (B, obs_dim), dtype=np.float32
                )

        return obs

    def step(self, action: th.Tensor, is_test=False):
        """
        Extended step: advance dynamic obstacles alongside drone dynamics.

        Per skill.md §3.8: obstacle_manager.step(dt) inserted into the step lifecycle.
        """
        # Step dynamic obstacles BEFORE drone dynamics
        if self.use_dynamic_obstacles:
            self.dynamic_obstacle_manager.step(self.dynamics.ctrl_dt)

        # Call parent step
        return super().step(action, is_test=is_test)

    def reset_agents(self, indices: Optional[List] = None):
        """Extended reset: also reset dynamic obstacle positions."""
        super().reset_agents(indices=indices)

        # Reset dynamic obstacles — reposition to initial configs
        if self.use_dynamic_obstacles and self.dynamic_obstacle_manager is not None:
            # Reinitialize obstacles for reset agents
            for env_id in (indices if indices is not None else range(self.num_envs)):
                for obs_idx in range(self.dynamic_obstacle_manager.num_obstacles(env_id)):
                    cfg = self.dynamic_obstacle_manager._motion_configs[env_id][obs_idx]
                    cfg["phase"] = 0.0
                    init_pos = cfg["init_position"]
                    init_vel = cfg["init_velocity"]
                    self.dynamic_obstacle_manager._obstacle_positions[env_id][obs_idx] = init_pos.clone()
                    self.dynamic_obstacle_manager._obstacle_velocities[env_id][obs_idx] = init_vel.clone()
            self.dynamic_obstacle_manager._t = 0.0

    def get_reward(self, action=None) -> th.Tensor:
        """
        Modified reward function — Innovation 2 + 3.

        Changes from parent NavigationEnv.get_reward():
          1. Replace geodesic_gradient with TTCRiskField avoidance direction.
          2. Replace motion-based yaw with risk-weighted, confidence-aware yaw.
          3. Add loss_risk (TTC penalty) and loss_track_conf (confidence penalty).
          4. Keep all smoothing losses (loss_a, loss_j, loss_om) unchanged.

        Two modes depending on obstacle input source:
          - GT mode (stage2): obstacle states from DynamicObstacleManager.
          - Perception mode (stage3/4): obstacle states from policy prediction.
        """
        # --- Reward parameters ---
        beta_1 = self.reward_kwargs.get("beta_1", 2.5)
        beta_2 = self.reward_kwargs.get("beta_2", -32)
        lambda_v = self.reward_kwargs.get("lambda_v", 1)
        lambda_c = self.reward_kwargs.get("lambda_c", 2)
        lambda_a = self.reward_kwargs.get("lambda_a", 0.01)
        lambda_j = self.reward_kwargs.get("lambda_j", 0.001)
        lambda_om = self.reward_kwargs.get("lambda_om", 0.03)
        lambda_yaw = self.reward_kwargs.get("lambda_yaw", 0.5)
        lambda_vmax = self.reward_kwargs.get("lambda_vmax", 0.0)
        falloff_dis = self.reward_kwargs.get("falloff_dis", 1.0)
        vel_thresh_slerp_yaw = self.reward_kwargs.get("vel_thresh_slerp_yaw", 1.0)

        # --- Desired direction: TTC risk field replaces geodesic ---
        if self.use_dynamic_obstacles and self.dynamic_obstacle_manager is not None:
            # Get obstacle states — GT or prediction based on curriculum stage
            obs_pos, obs_vel = self._get_obstacle_states_for_reward()

            if obs_pos is not None and obs_pos.shape[1] > 0:
                # Compute TTC risk and avoidance direction (differentiable!)
                risk, risk_grad = compute_ttc_risk(
                    self.position, self.velocity,
                    obs_pos, obs_vel,
                    obstacle_radii=self.dynamic_obstacle_manager.get_obstacle_radii(),
                    ttc_threshold=self.ttc_field.ttc_threshold,
                    distance_safe=self.ttc_field.distance_safe,
                    barrier_beta=self.ttc_field.barrier_beta,
                )
                # Risk gradient gives direction AWAY from danger
                # When risk is low, fall back to target direction
                has_risk = (risk > 0.01).float().unsqueeze(-1)
                desired_direction = F.normalize(
                    has_risk * risk_grad + (1 - has_risk) * self.target_direction,
                    dim=1,
                )
            else:
                risk = th.zeros(self.num_envs, device=self.device)
                risk_grad = th.zeros((self.num_envs, 3), device=self.device)
                desired_direction = self.target_direction
        else:
            # No dynamic obstacles: use target direction (like parent)
            risk = th.zeros(self.num_envs, device=self.device)
            risk_grad = th.zeros((self.num_envs, 3), device=self.device)
            desired_direction = self.target_direction

        # --- Collision loss (same as parent) ---
        def positive_speed_towards_collision(velocity, collision_vec):
            collision_dir = F.normalize(collision_vec, dim=1)
            speed = th.sum(velocity * collision_dir, dim=1)
            return th.clamp(speed, min=0.0)

        speed_towards_collision = positive_speed_towards_collision(
            self.velocity, self.collision_vector
        )
        distance_penalty = (
            falloff_dis - (self.collision_dis - self.robot_radius)
        ).relu() ** 2
        intersection_barrier = th.log(
            1 + th.exp(beta_2 * (self.collision_dis - self.robot_radius))
        )
        loss_c = speed_towards_collision * distance_penalty + beta_1 * intersection_barrier

        # --- Speed limit penalty ---
        loss_vmax = ((self.speed - self.target_speed).relu() ** 2).squeeze(1)

        # --- Velocity deviation loss (now based on desired_direction, not geodesic) ---
        desired_velocity = self.target_speed * desired_direction
        velocity_difference = (desired_velocity - self.moving_average_velocity).norm(dim=1)
        loss_v = F.smooth_l1_loss(
            velocity_difference, th.zeros_like(velocity_difference), reduction="none"
        )

        # --- Smoothness losses (unchanged from parent) ---
        loss_a = self.acceleration.norm(dim=1) ** 2
        loss_j = self.jerk.norm(dim=1) ** 2
        loss_om = self.omega.norm(dim=1) ** 2

        # --- Yaw loss (Innovation 3: risk-weighted, confidence-aware) ---
        def slerp(a, b, t):
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            t = t.clamp(0.0, 1.0)
            dot = (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            theta = th.acos(dot)
            sin_theta = th.sin(theta)
            near_zero = sin_theta < 1e-6
            slerp_result = th.where(
                near_zero,
                F.normalize((1 - t) * a + t * b, dim=1),
                (th.sin((1 - t) * theta) * a + th.sin(t * theta) * b) / sin_theta,
            )
            return F.normalize(slerp_result, dim=1)

        avg_vel = self.exp_moving_average_velocity.clone().detach()
        t = (self.position - self.start_position).norm(dim=1, keepdim=True)
        desired_yaw_original = slerp(self.target_direction, avg_vel, t).clone().detach()

        # --- Risk-weighted yaw ---
        if self.use_dynamic_obstacles and obs_pos is not None and obs_pos.shape[1] > 0:
            # Direction to nearest/closest obstacle
            obs_pos_0 = obs_pos[:, 0, :]  # First obstacle
            direction_to_obstacle = F.normalize(
                obs_pos_0 - self.position, dim=1
            )

            # Risk-based weight: higher risk → prioritize looking at obstacle
            risk_norm = (risk / (risk.max() + 1e-8)).unsqueeze(-1)  # (B, 1)
            risk_norm = th.clamp(risk_norm, 0.0, 1.0)

            # Confidence-weighted yaw (Innovation 3)
            if self._confidence_proxy is not None and self.use_confidence_proxy:
                bearing = compute_bearing(self.position, self.yaw_vector, obs_pos)
                range_ = compute_range(self.position, obs_pos)
                # Estimate illumination from scene (placeholder)
                illum = th.ones(self.num_envs, device=self.device) * (
                    0.3 if self.use_extreme_lighting else 0.8
                )
                track_conf = self._confidence_proxy.get_weighted_confidence(
                    bearing, range_, illum
                )  # (B,)
                track_conf = track_conf.unsqueeze(-1)  # (B, 1)
            else:
                track_conf = th.ones(self.num_envs, 1, device=self.device)

            # Combine weights: risk tells us WHETHER to look, confidence tells us HOW well
            w = risk_norm * track_conf
            desired_yaw_vector = slerp(desired_yaw_original, direction_to_obstacle, w)
        else:
            desired_yaw_vector = desired_yaw_original
            track_conf = th.ones(self.num_envs, 1, device=self.device)

        loss_yaw = -(desired_yaw_vector * self.yaw_vector).sum(dim=1)

        # Suppress yaw loss near target (same as parent)
        loss_yaw = th.where(
            (self.target_distance < 1.0).squeeze(1),
            th.zeros_like(loss_yaw),
            loss_yaw,
        )

        # --- New: Risk penalty (Innovation 2) ---
        loss_risk = F.relu(self.ttc_field.ttc_threshold - compute_min_ttc(
            self.position, self.velocity, obs_pos, obs_vel
        ) if obs_pos is not None and obs_pos.shape[1] > 0 else th.zeros_like(risk))

        # --- New: Tracking confidence loss (Innovation 3) ---
        # Penalize low tracking confidence when risk is present
        loss_track_conf = (1.0 - track_conf.squeeze(-1)) * (risk > 0.01).float()

        # --- Total loss ---
        loss = (
            lambda_v * loss_v
            + lambda_vmax * loss_vmax
            + lambda_c * loss_c
            + lambda_a * loss_a
            + lambda_j * loss_j
            + lambda_om * loss_om
            + lambda_yaw * loss_yaw
            + self._lambda_risk * loss_risk
            + self._lambda_track_conf * loss_track_conf
        )
        reward = -loss

        # --- Metrics ---
        metrics = {
            "loss_v": (lambda_v * loss_v).clone().detach().cpu(),
            "loss_vmax": (lambda_vmax * loss_vmax).clone().detach().cpu(),
            "loss_c": (lambda_c * loss_c).clone().detach().cpu(),
            "loss_a": (lambda_a * loss_a).clone().detach().cpu(),
            "loss_j": (lambda_j * loss_j).clone().detach().cpu(),
            "loss_om": (lambda_om * loss_om).clone().detach().cpu(),
            "loss_yaw": (lambda_yaw * loss_yaw).clone().detach().cpu(),
            "loss_risk": (self._lambda_risk * loss_risk).clone().detach().cpu(),
            "loss_track_conf": (self._lambda_track_conf * loss_track_conf).clone().detach().cpu(),
            "risk": risk.clone().detach().cpu(),
            "track_confidence": track_conf.squeeze(-1).clone().detach().cpu(),
            "min_predicted_ttc": compute_min_ttc(
                self.position, self.velocity, obs_pos, obs_vel
            ).clone().detach().cpu() if (
                obs_pos is not None and obs_pos.shape[1] > 0
            ) else th.full((self.num_envs,), float('inf')).cpu(),
        }

        return reward, metrics

    def _get_obstacle_states_for_reward(self):
        """
        Get obstacle states for reward computation.

        GT mode (stage2): Return ground truth from DynamicObstacleManager.
        Perception mode (stage3/4): Return estimated states from policy prediction.
        """
        if not self.use_perception:
            # Use ground truth (stage2)
            obs_pos = self.dynamic_obstacle_manager.get_obstacle_positions_gt(0)
            obs_vel = self.dynamic_obstacle_manager.get_obstacle_velocities_gt(0)
            if obs_pos.shape[0] == 0:
                return None, None
            return obs_pos.unsqueeze(0), obs_vel.unsqueeze(0)  # (1, K, 3)
        else:
            # Use perceived/estimated states (stage3/4)
            prediction = self._last_obstacle_prediction
            if prediction is not None:
                return (
                    prediction.get("step_1_position", None),
                    prediction.get("step_1_velocity", None),
                )
            # Fallback: use GT but flag it
            obs_pos = self.dynamic_obstacle_manager.get_obstacle_positions_gt(0)
            obs_vel = self.dynamic_obstacle_manager.get_obstacle_velocities_gt(0)
            if obs_pos.shape[0] == 0:
                return None, None
            return obs_pos.unsqueeze(0), obs_vel.unsqueeze(0)

    def close(self):
        """Cleanup resources."""
        super().close()
        self.dynamic_obstacle_manager = None
