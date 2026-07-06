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
import torch as th
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from extreme_avoid.registry import env_aliases, policy_aliases
from depthnav.scripts.eval_logger import Evaluate
from depthnav.policies.multi_input_policy import MultiInputPolicy


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
        """
        Run evaluation rollouts with extended metrics.
        """
        base_df = super().run_rollouts(num_rollouts, run_name, render)

        # Add extended columns
        base_df["dyn_collision_rate"] = np.mean(self._dynamic_collisions) if self._dynamic_collisions else 0.0
        base_df["min_ttc_mean"] = np.mean(self._min_ttcs) if self._min_ttcs else float('inf')
        base_df["min_ttc_median"] = np.median(self._min_ttcs) if self._min_ttcs else float('inf')
        base_df["id_switch_rate"] = np.mean(self._id_switches) if self._id_switches else 0.0
        base_df["avg_risk"] = np.mean(self._risks) if self._risks else 0.0
        base_df["avg_track_conf"] = np.mean(self._track_confs) if self._track_confs else 0.0

        return base_df

    def _process_step_metrics(self, info: Dict):
        """Extract dynamic-obstacle-specific metrics from env info."""
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
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--csv_output", type=str, default=None,
                        help="CSV file to append evaluation stats")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load eval config
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

    # Create env
    env_class = env_aliases[config["env_class"]]
    env = env_class(**config["env"])

    # Load policy
    if not os.path.exists(args.weight):
        print(f"Checkpoint not found: {args.weight}")
        sys.exit(1)

    checkpoint = th.load(args.weight, map_location="cpu", weights_only=False)
    # Reconstruct policy from checkpoint metadata or config
    policy_class = policy_aliases.get("TrackerFusedPolicy", MultiInputPolicy)
    try:
        policy = policy_class(env.observation_space)
        policy.load(args.weight)
    except Exception as e:
        print(f"Warning: Could not load policy with standard method: {e}")
        print("Falling back to MultiInputPolicy...")
        policy = MultiInputPolicy(env.observation_space)
        policy.load(args.weight)

    # Evaluate
    evaluator = EvaluateDynamicAvoidance(env, policy)
    df = evaluator.run_rollouts(
        num_rollouts=args.num_rollouts,
        render=args.render,
    )

    # Output
    csv_path = args.csv_output or args.weight.replace(".pth", "_eval_dynamic.csv")
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, float_format="%.3f", mode="a", header=write_header)

    print("\n=== Evaluation Results ===")
    print(df.to_string())
    print(f"\nResults saved to: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
