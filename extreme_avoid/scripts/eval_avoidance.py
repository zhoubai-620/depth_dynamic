#!/usr/bin/env python3
"""
Evaluation Script for Dynamic Avoidance.

Inherits from depthnav/scripts/eval_logger.py Evaluate class and adds
dynamic-obstacle-specific metrics (per skill.md §3.13):

  - Dynamic obstacle collision rate (separate from static geometry)
  - Minimum predicted TTC distribution
  - Tracking continuity (ID switch rate, identity failure count)
  - Risk and confidence metrics over time

These metrics run across all curriculum stages for ablation comparison.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch as th
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from extreme_avoid.registry import env_aliases, policy_aliases
from depthnav.scripts.eval_logger import Evaluate
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.common import observation_to_device
from depthnav.utils import rgba2rgb


class EvaluateDynamicAvoidance(Evaluate):
    """
    Extended evaluator with dynamic obstacle metrics.

    Adds to parent's metrics:
      - dyn_collision_rate: collisions with dynamic obstacles
      - min_ttc_mean/median: minimum TTC distribution
      - id_switch_rate: obstacle identity continuity failures
      - avg_risk: average TTC risk over rollout
      - avg_track_conf: average tracking confidence
    """

    def __init__(self, env, policy):
        super().__init__(env, policy)
        self._dynamic_collisions = []
        self._min_ttcs = []
        self._id_switches = []
        self._risks = []
        self._track_confs = []

    def run_rollouts(
        self,
        num_rollouts: int = 10,
        run_name: Optional[int] = None,
        render: bool = False,
    ) -> pd.DataFrame:
        """Run evaluation rollouts with extended metrics."""
        base_df = super().run_rollouts(num_rollouts, run_name, render)

        base_df["dyn_collision_rate"] = np.mean(self._dynamic_collisions) if self._dynamic_collisions else 0.0
        base_df["min_ttc_mean"] = np.mean(self._min_ttcs) if self._min_ttcs else float('inf')
        base_df["min_ttc_median"] = np.median(self._min_ttcs) if self._min_ttcs else float('inf')
        base_df["id_switch_rate"] = np.mean(self._id_switches) if self._id_switches else 0.0
        base_df["avg_risk"] = np.mean(self._risks) if self._risks else 0.0
        base_df["avg_track_conf"] = np.mean(self._track_confs) if self._track_confs else 0.0

        return base_df

    @th.no_grad()
    @th.no_grad()
    def single_rollout(self, render=False):
        """
        Override parent's single_rollout to fix type(...) == MultiInputPolicy
        to isinstance(...), and fix return type: returns dict-of-list (batch_stats)
        matching the parent's expected structure.
        """
        agent_logs = [
            {
                "position": [],
                "velocity": [],
                "speed": [],
                "acceleration": [],
                "jerk": [],
                "yaw_rate": [],
                "obstacle_distance": [],
                "success": 0,
                "collision": 0,
                "timeout": 0,
                "avg_reward": 0.0,
                "duration": 0.0,
                "steps": 0.0,
                "path_length": 0.0,
                "avg_control_effort": 0.0,
                "last_action_x": 0.0,
                "last_action_y": 0.0,
                "last_action_z": 0.0,
                "last_action_yaw": 0.0,
                "last_position_x": 0.0,
                "last_position_y": 0.0,
                "last_position_z": 0.0,
                "last_velocity_x": 0.0,
                "last_velocity_y": 0.0,
                "last_velocity_z": 0.0,
            }
            for _ in range(self.env.num_envs)
        ]

        eval_info_id_list = [i for i in range(self.env.num_envs)]

        latent_state = th.zeros(
            (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
        )
        while True:
            obs = observation_to_device(self.env.get_observation(), self.policy.device)
            # FIXED: isinstance instead of type()== to support TrackerFusedPolicy
            if isinstance(self.policy, MultiInputPolicy):
                if self.policy.is_recurrent:
                    action, latent_state = self.policy(obs, latent_state)
                else:
                    action = self.policy(obs)
            else:
                action = self.policy(obs["state"])
            obs, reward, terminated, infos = self.env.step(action, is_test=True)

            if render:
                self.render_kwargs["points"] = th.cat(
                    [self.env.position.unsqueeze(1), self.env.target.unsqueeze(1)],
                    dim=1,
                )
                B, C, H, W = obs["depth"].shape
                obs_grid = (
                    obs["depth"].permute(2, 0, 3, 1).reshape(H, B * W, C).cpu().numpy()
                )
                obs_grid = (obs_grid - obs_grid.min()) / (
                    obs_grid.max() - obs_grid.min()
                )
                cv2.imshow("agent cams", obs_grid)

                render_obs = rgba2rgb(
                    self.env.scene_manager.render(**self.render_kwargs)
                )
                render_grid = np.hstack(render_obs)
                cv2.imshow("render cams", render_grid)
                cv2.waitKey(1)

            for index in reversed(eval_info_id_list):
                if not terminated[index]:
                    agent_logs[index]["speed"].append(
                        self.env.speed[index].item()
                    )
                    agent_logs[index]["acceleration"].append(
                        self.env.acceleration[index].norm().item()
                    )
                    agent_logs[index]["jerk"].append(
                        self.env.jerk[index].norm().item()
                    )
                    agent_logs[index]["yaw_rate"].append(
                        self.env.omega[index][2].item()
                    )
                    agent_logs[index]["obstacle_distance"].append(
                        self.env.collision_dis[index].item()
                    )
                    agent_logs[index]["position"].append(self.env.position[index])

                    # Dynamic-obstacle-specific step metrics
                    if isinstance(infos, list) and index < len(infos):
                        self._process_step_metrics(infos[index])
                    elif isinstance(infos, dict):
                        self._process_step_metrics(infos)
                else:
                    eval_info_id_list.remove(index)

                    # Collision: from env attribute (same as original eval_logger.py)
                    agent_logs[index]["collision"] = (
                        self.env.is_collision[index].int().item()
                    )
                    # Success: from infos (key is "is_success" in base_env.py)
                    agent_logs[index]["success"] = int(infos[index]["is_success"])
                    # Timeout: derived (neither collision nor success)
                    agent_logs[index]["timeout"] = int(
                        not agent_logs[index]["collision"]
                        and not agent_logs[index]["success"]
                    )
                    agent_logs[index]["avg_reward"] = infos[index][
                        "episode_avg_step_reward"
                    ].item()
                    agent_logs[index]["duration"] = infos[index][
                        "episode_duration"
                    ].item()
                    agent_logs[index]["steps"] = float(
                        infos[index]["episode_length"].item()
                    )

                    # Last state/action
                    action_cpu = action.cpu()
                    agent_logs[index]["last_action_x"] = action_cpu[index][0].item()
                    agent_logs[index]["last_action_y"] = action_cpu[index][1].item()
                    agent_logs[index]["last_action_z"] = action_cpu[index][2].item()
                    agent_logs[index]["last_action_yaw"] = (
                        action_cpu[index][3].item() if action_cpu.shape[1] >= 4 else 0.0
                    )
                    agent_logs[index]["last_position_x"] = self.env.position[index][0].item()
                    agent_logs[index]["last_position_y"] = self.env.position[index][1].item()
                    agent_logs[index]["last_position_z"] = self.env.position[index][2].item()
                    agent_logs[index]["last_velocity_x"] = self.env.velocity[index][0].item()
                    agent_logs[index]["last_velocity_y"] = self.env.velocity[index][1].item()
                    agent_logs[index]["last_velocity_z"] = self.env.velocity[index][2].item()

                    # Path length (from accumulated positions)
                    points = th.stack(agent_logs[index]["position"])
                    agent_logs[index]["path_length"] = (
                        (points[1:] - points[:-1]).norm(dim=1).sum().item()
                    )

                    # Control effort
                    jerk_t = th.tensor(agent_logs[index]["jerk"])
                    total_ce = (jerk_t ** 2).sum() * self.env.dynamics.ctrl_dt
                    agent_logs[index]["avg_control_effort"] = (
                        (total_ce / len(jerk_t)).item() if len(jerk_t) > 0 else 0.0
                    )

            if len(eval_info_id_list) == 0:
                break

        batch_stats = {
            "avg_speed": [
                th.tensor(agent["speed"]).mean().item() for agent in agent_logs
            ],
            "max_speed": [
                th.tensor(agent["speed"]).max().item() for agent in agent_logs
            ],
            "max_acceleration": [
                th.tensor(agent["acceleration"]).max().item() for agent in agent_logs
            ],
            "avg_yaw_rate": [
                th.tensor(agent["yaw_rate"]).mean().item() for agent in agent_logs
            ],
            "max_yaw_rate": [
                th.tensor(agent["yaw_rate"]).max().item() for agent in agent_logs
            ],
            "avg_min_obstacle_distance": [
                th.tensor(agent["obstacle_distance"]).min().item()
                for agent in agent_logs
            ],
            "collision_count": [agent["collision"] for agent in agent_logs],
            "success_count": [agent["success"] for agent in agent_logs],
            "timeout_count": [agent["timeout"] for agent in agent_logs],
            "avg_reward": [agent["avg_reward"] for agent in agent_logs],
            "duration": [agent["duration"] for agent in agent_logs],
            "steps": [agent["steps"] for agent in agent_logs],
            "path_length": [agent["path_length"] for agent in agent_logs],
            "avg_control_effort": [agent["avg_control_effort"] for agent in agent_logs],
            "last_action_x": [agent["last_action_x"] for agent in agent_logs],
            "last_action_y": [agent["last_action_y"] for agent in agent_logs],
            "last_action_z": [agent["last_action_z"] for agent in agent_logs],
            "last_action_yaw": [agent["last_action_yaw"] for agent in agent_logs],
            "last_position_x": [agent["last_position_x"] for agent in agent_logs],
            "last_position_y": [agent["last_position_y"] for agent in agent_logs],
            "last_position_z": [agent["last_position_z"] for agent in agent_logs],
            "last_velocity_x": [agent["last_velocity_x"] for agent in agent_logs],
            "last_velocity_y": [agent["last_velocity_y"] for agent in agent_logs],
            "last_velocity_z": [agent["last_velocity_z"] for agent in agent_logs],
        }
        return batch_stats

    def _process_step_metrics(self, info: Dict):
        """Extract dynamic-obstacle-specific metrics from env info dict."""
        for key in ["risk", "track_confidence", "min_predicted_ttc"]:
            if key in info.get("loss_metrics", {}):
                val = info["loss_metrics"][key]
                if isinstance(val, th.Tensor):
                    val = val.item()
                if key == "risk":
                    self._risks.append(float(val))
                elif key == "track_confidence":
                    self._track_confs.append(float(val))
                elif key == "min_predicted_ttc":
                    if val < float('inf'):
                        self._min_ttcs.append(float(val))


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic avoidance policy")
    parser.add_argument("--weight", type=str, required=True, help="Policy checkpoint path")
    parser.add_argument("--eval_config", type=str,
                        default="extreme_avoid/configs/eval_cfg/eval_dynamic.yaml",
                        help="Eval environment config YAML")
    parser.add_argument("--policy_cfg_file", type=str, default=None,
                        help="Policy config YAML (e.g. configs/policy_cfg/tracker_fused_yaw.yaml)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--csv_output", type=str, default=None,
                        help="CSV file to append evaluation stats")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml
    if os.path.exists(args.eval_config):
        with open(args.eval_config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "env_class": "dynamic_avoidance_env",
            "env": {
                "num_envs": args.num_envs,
                "visual": True,
                "single_env": True,
                "requires_grad": False,
            }
        }

    env_class = env_aliases[config["env_class"]]
    env = env_class(**config["env"])

    # Load policy with correct kwargs from config
    if not os.path.exists(args.weight):
        print(f"Checkpoint not found: {args.weight}")
        sys.exit(1)

    checkpoint = th.load(args.weight, map_location="cpu", weights_only=False)

    # Load policy kwargs from YAML config (like original eval_logger.py)
    policy_kwargs = {}
    if args.policy_cfg_file and os.path.exists(args.policy_cfg_file):
        with open(args.policy_cfg_file, "r") as f:
            policy_cfg = yaml.safe_load(f)
        policy_kwargs = policy_cfg.get("policy", {})
    elif "policy" in config:
        policy_kwargs = config.get("policy", {})

    policy_class_name = config.get("policy_class", "TrackerFusedPolicy")
    policy_class = policy_aliases.get(policy_class_name, MultiInputPolicy)

    try:
        policy = policy_class(env.observation_space, **policy_kwargs)
        policy.load(args.weight)
    except (TypeError, Exception) as e:
        print(f"Warning: Could not construct policy with config: {e}")
        print("Falling back to MultiInputPolicy...")
        policy = MultiInputPolicy(env.observation_space)
        policy.load(args.weight)

    evaluator = EvaluateDynamicAvoidance(env, policy)
    df = evaluator.run_rollouts(
        num_rollouts=args.num_rollouts,
        render=args.render,
    )

    csv_path = args.csv_output or args.weight.replace(".pth", "_eval_dynamic.csv")
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, float_format="%.3f", mode="a", header=write_header)

    print("\n=== Evaluation Results ===")
    print(df.to_string())
    print(f"\nResults saved to: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
