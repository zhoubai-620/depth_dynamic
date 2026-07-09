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
import os
from typing import Optional, Dict, List, Union, Any
from enum import Enum

from extreme_avoid.vendor.depthnav.envs.navigation_env import NavigationEnv, Frame, ActionType, TargetType, get_enum
from extreme_avoid.vendor.depthnav.utils import Rotation3
from gymnasium import spaces

from .dynamic_obstacle_manager import DynamicObstacleManager, MotionPattern
from ..risk.ttc_field import TTCRiskField, compute_ttc_risk, compute_min_ttc
from ..risk.confidence_proxy import ConfidenceProxy, compute_bearing, compute_range

try:
    import habitat_sim
    from habitat_sim.physics import MotionType as HabitatMotionType
    _HABITAT_AVAILABLE = True
except ImportError:
    _HABITAT_AVAILABLE = False
    HabitatMotionType = None


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

        self.scene_kwargs = scene_kwargs or {}

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
        """Spawn dynamic obstacles as habitat-sim ManagedRigidObject instances."""
        import logging
        _logger = logging.getLogger(__name__)

        if not self.visual or self.scene_manager is None:
            for env_id in range(self.num_envs):
                for cfg in configs[:self.max_dynamic_obstacles]:
                    self.dynamic_obstacle_manager.add_obstacle(
                        env_id=env_id,
                        initial_position=cfg.get("initial_position", [5.0, 0.0, 2.0]),
                        initial_velocity=cfg.get("initial_velocity", [0.0, 0.5, 0.0]),
                        motion_pattern=cfg.get("motion_pattern", "constant_velocity"),
                        motion_params=cfg.get("motion_params", {}),
                        obstacle_radius=cfg.get("radius", 0.3),
                        rigid_object=None,
                    )
            return

        # --- habitat-sim spawning path ---
        # Read object template path from scene_kwargs (same convention as static obstacles)
        template_path = (self.scene_kwargs or {}).get(
            "obstacle_object_config_path", None
        )
        try:
            template_ids = []
            for env_id in range(self.num_envs):
                scene_id = env_id // self.scene_manager.num_agent_per_scene
                sim = self.scene_manager.scenes[scene_id]
                template_mgr = sim.get_object_template_manager()
                rigid_mgr = sim.get_rigid_object_manager()

                if not template_ids and template_path is not None:
                    if hasattr(template_mgr, 'load_configs') and os.path.exists(template_path):
                        template_ids = template_mgr.load_configs(template_path)
                        if not template_ids:
                            _logger.warning(
                                f"load_configs({template_path}) returned empty list"
                            )
                    else:
                        _logger.warning(
                            f"Object template path not found or load_configs unavailable: {template_path}"
                        )

                for cfg in configs[:self.max_dynamic_obstacles]:
                    pos = cfg.get("initial_position", [5.0, 0.0, 2.0])
                    vel = cfg.get("initial_velocity", [0.0, 0.5, 0.0])
                    pattern = cfg.get("motion_pattern", "constant_velocity")
                    params = cfg.get("motion_params", {})
                    radius = cfg.get("radius", 0.3)

                    rigid_obj = None
                    if template_ids:
                        try:
                            rigid_obj = rigid_mgr.add_object_by_template_id(template_ids[0])
                            # Wrap in magnum.Vector3 per habitat-sim convention
                            # (scene_manager.py uses mn.Vector3 for all translation assignments)
                            try:
                                import magnum as mn
                                rigid_obj.translation = mn.Vector3(pos)
                            except ImportError:
                                rigid_obj.translation = pos
                            rigid_obj.motion_type = HabitatMotionType.KINEMATIC
                        except Exception as e:
                            _logger.warning(
                                f"Failed to spawn dynamic obstacle at {pos}: {e}"
                            )

                    self.dynamic_obstacle_manager.add_obstacle(
                        env_id=env_id,
                        initial_position=pos,
                        initial_velocity=vel,
                        motion_pattern=pattern,
                        motion_params=params,
                        obstacle_radius=radius,
                        rigid_object=rigid_obj,
                    )

                if template_ids:
                    try:
                        sim.recompute_mesh_kdtree()
                    except Exception as e:
                        _logger.warning(f"recompute_mesh_kdtree failed: {e}")

        except (AttributeError, IndexError) as e:
            _logger.warning(f"Cannot access habitat-sim scene for dynamic obstacles: {e}")

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
            obs_pos, obs_vel, obs_mask = self._get_obstacle_states_for_reward()

            if obs_pos is not None and obs_pos.shape[1] > 0:
                # Build per-env radii — pad to match max_k if needed
                K = obs_pos.shape[1]
                all_radii = []
                for env_id in range(self.num_envs):
                    radii = self.dynamic_obstacle_manager.get_obstacle_radii(env_id)
                    k = radii.shape[0]
                    if k < K:
                        radii = th.cat([radii, th.zeros(K - k, device=self.device)])
                    all_radii.append(radii)
                obs_radii = th.stack(all_radii, dim=0)

                # Compute TTC risk and avoidance direction (differentiable!)
                risk, risk_grad = self.ttc_field(
                    self.position, self.velocity,
                    obs_pos, obs_vel,
                    obstacle_radii=obs_radii,
                    obstacle_mask=obs_mask,
                )
                # Risk gradient gives direction AWAY from danger
                # When risk is low, fall back to target direction
                has_risk = (risk > 0.01).float().unsqueeze(-1)
                desired_direction = F.normalize(
                    has_risk * risk_grad + (1 - has_risk) * self.target_direction,
                    dim=1,
                )
            else:
                obs_radii = None
                risk = th.zeros(self.num_envs, device=self.device)
                risk_grad = th.zeros((self.num_envs, 3), device=self.device)
                desired_direction = self.target_direction
        else:
            # No dynamic obstacles: use target direction (like parent)
            obs_pos = obs_vel = obs_mask = obs_radii = None
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
            # Nearest obstacle (smallest distance to drone, excluding phantoms)
            distances = th.norm(obs_pos - self.position.unsqueeze(1), dim=2)  # (B, K)
            if obs_mask is not None:
                distances = th.where(obs_mask.bool(), distances, th.full_like(distances, float('inf')))
            nearest_idx = distances.argmin(dim=1)  # (B,)
            nearest_pos = th.gather(obs_pos, 1, nearest_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)).squeeze(1)  # (B, 3)
            direction_to_obstacle = F.normalize(
                nearest_pos - self.position, dim=1
            )

            # Risk-based weight: higher risk → prioritize looking at obstacle
            risk_norm = th.clamp(risk / (self.ttc_field.ttc_threshold + 1e-8), 0.0, 1.0).unsqueeze(-1)

            # Confidence-weighted yaw (Innovation 3)
            if self._confidence_proxy is not None and self.use_confidence_proxy:
                bearing = compute_bearing(self.position, self.yaw_vector, obs_pos)
                range_ = compute_range(self.position, obs_pos)
                # Estimate illumination from color sensor (mean pixel intensity, normalized)
                if hasattr(self, 'sensor_obs'):
                    color_key = next((k for k in self.sensor_obs if 'color' in k), None)
                    if color_key is not None and isinstance(self.sensor_obs[color_key], np.ndarray):
                        color_data = self.sensor_obs[color_key]  # (num_envs, C, H, W)
                        per_env_mean = color_data.reshape(color_data.shape[0], -1).astype(np.float32).mean(axis=1)
                        illum = th.as_tensor(
                            np.clip(per_env_mean / 255.0, 0.01, 1.0), device=self.device, dtype=th.float32
                        )
                    else:
                        illum = th.full((self.num_envs,),
                                        0.3 if self.use_extreme_lighting else 0.8,
                                        device=self.device)
                else:
                    illum = th.full((self.num_envs,),
                                    0.3 if self.use_extreme_lighting else 0.8,
                                    device=self.device)
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
            self.position, self.velocity, obs_pos, obs_vel,
            obstacle_mask=obs_mask,
        ) if obs_pos is not None and obs_pos.shape[1] > 0 else th.zeros_like(risk)).pow(2)

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
                self.position, self.velocity, obs_pos, obs_vel,
                obstacle_mask=obs_mask,
            ).clone().detach().cpu() if (
                obs_pos is not None and obs_pos.shape[1] > 0
            ) else th.full((self.num_envs,), float('inf')).cpu(),
            "dynamic_collision": self._compute_dynamic_collision_flag(obs_pos, obs_radii).clone().detach().cpu(),
        }

        return reward, metrics

    def _get_obstacle_states_for_reward(self):
        if not self.use_perception:
            # Use ground truth (stage2): collect from all envs, pad to same K
            all_pos = []
            all_vel = []
            all_radii = []
            max_k = 0
            for env_id in range(self.num_envs):
                pos = self.dynamic_obstacle_manager.get_obstacle_positions_gt(env_id)
                vel = self.dynamic_obstacle_manager.get_obstacle_velocities_gt(env_id)
                all_pos.append(pos)
                all_vel.append(vel)
                k = pos.shape[0]
                max_k = max(max_k, k)

            if max_k == 0:
                return None, None, None

            # Pad to uniform K with zeros; construct per-env mask
            padded_pos = []
            padded_vel = []
            masks = []
            for env_id in range(self.num_envs):
                k = all_pos[env_id].shape[0]
                if k < max_k:
                    pad_pos = th.cat([
                        all_pos[env_id],
                        th.zeros((max_k - k, 3), device=self.device)
                    ], dim=0)
                    pad_vel = th.cat([
                        all_vel[env_id],
                        th.zeros((max_k - k, 3), device=self.device)
                    ], dim=0)
                else:
                    pad_pos = all_pos[env_id]
                    pad_vel = all_vel[env_id]
                padded_pos.append(pad_pos)
                padded_vel.append(pad_vel)
                mask = th.cat([
                    th.ones(k, device=self.device),
                    th.zeros(max_k - k, device=self.device)
                ], dim=0)
                masks.append(mask)

            pos_stacked = th.stack(padded_pos, dim=0)      # (B, max_k, 3)
            vel_stacked = th.stack(padded_vel, dim=0)      # (B, max_k, 3)
            mask_stacked = th.stack(masks, dim=0)           # (B, max_k)
            return pos_stacked, vel_stacked, mask_stacked
        else:
            # Use perceived/estimated states (stage3/4)
            prediction = self._last_obstacle_prediction
            if prediction is not None:
                pos = prediction.get("step_1_position", None)
                vel = prediction.get("step_1_velocity", None)
                if pos is not None and vel is not None:
                    # motion_head outputs (B, 1, 3) after P0-3 fix
                    B = pos.shape[0]
                    mask = th.ones(B, 1, device=self.device)
                    return pos, vel, mask
            return None, None, None

    def _compute_dynamic_collision_flag(
        self, obs_pos: th.Tensor, obs_radii: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """Per-step check: has the robot body actually collided with the
        nearest dynamic obstacle this step?

        This must use the same physical criterion as the parent's static
        collision check (`self._collision_dis < self.robot_radius` in
        base_env.py): a collision happens when the *surface* distance
        between the robot and the obstacle drops below the robot's radius.

        Previous (buggy) version compared `min_dist_to_dyn` — the raw
        center-to-center distance to the nearest dynamic obstacle, which
        does not subtract the obstacle's own radius — against
        `self.collision_dis`, which is the robot's distance to the nearest
        *static* surface point. Those two quantities are not the same
        physical measurement (center-to-center vs. center-to-surface), and
        `self.collision_dis` is an observation, not a collision threshold —
        `self.robot_radius` is. That mismatch made this flag systematically
        under-report dynamic-obstacle collisions whenever the obstacle had a
        non-trivial radius, and it wasn't even measuring "collision" at all
        (it was measuring "which is nearer").

        Args:
            obs_pos: (B, K, 3) obstacle center positions (GT or predicted).
            obs_radii: (B, K) obstacle radii. If None, obstacles are treated
                as point masses (radius 0), matching `ttc_field`'s default.

        Returns:
            (B,) float tensor; 1.0 if the robot body currently overlaps the
            nearest dynamic obstacle's surface, else 0.0.
        """
        if obs_pos is None or obs_pos.shape[1] == 0:
            return th.zeros(self.num_envs, device=self.device)

        center_dist = (self.position.unsqueeze(1) - obs_pos).norm(dim=2)  # (B, K)
        if obs_radii is not None:
            surface_dist = center_dist - obs_radii
        else:
            surface_dist = center_dist
        min_surface_dist = surface_dist.min(dim=1).values  # (B,)

        return (min_surface_dist < self.robot_radius).float()

    def close(self):
        """Cleanup resources."""
        super().close()
        self.dynamic_obstacle_manager = None