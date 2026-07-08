#!/usr/bin/env python3
"""
Confidence Proxy Data Collection Script.

Offline data collection for fitting the ConfidenceProxy model (Innovation 3).
Runs rollouts with DPTracker perception active, recording per-step tuples:
  (bearing, range, illumination, true_confidence)

These logs are consumed by fit_confidence_proxy.py to train the differentible
proxy that bypasses habitat-sim's non-differentiable rendering.

Per skill.md §3.14: Must run BEFORE using confidence_proxy in training.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch as th
import yaml
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from extreme_avoid.registry import env_aliases, policy_aliases
from extreme_avoid.risk.confidence_proxy import compute_bearing, compute_range
from extreme_avoid.vendor.depthnav.policies.multi_input_policy import MultiInputPolicy


def collect_logs(
    env,
    policy,
    num_rollouts: int = 50,
    max_steps_per_rollout: int = 256,
    render: bool = False,
) -> List[Dict]:
    """
    Collect (bearing, range, illumination, confidence) tuples.

    For each step:
      - Record drone state (position, yaw)
      - Record obstacle ground truth (position from DynamicObstacleManager)
      - Compute bearing & range analytically
      - Record scene illumination level
      - Run DPTracker perception to get true confidence
      - Store tuple for offline fitting
    """
    logs = []

    for rollout_idx in tqdm(range(num_rollouts), desc="Collecting logs"):
        obs = env.reset()

        for step in range(max_steps_per_rollout):
            # Get perception outputs
            if hasattr(policy, 'feature_extractor') and hasattr(policy.feature_extractor, 'fused_backbone'):
                color_key = None
                for k in obs:
                    if 'color' in k:
                        color_key = k
                        break
                if color_key:
                    color_img = obs[color_key]
                    if isinstance(color_img, np.ndarray):
                        color_img = th.from_numpy(color_img).float() / 255.0
                        if color_img.dim() == 3:
                            color_img = color_img.unsqueeze(0)
                        color_img = color_img.to(policy.device)

                    # Run backbone forward
                    with th.no_grad():
                        backbone_out = policy.feature_extractor.fused_backbone(color_img)
                        obstacle_out = backbone_out["obstacle"]
                        true_confidence = obstacle_out["confidence"].squeeze(0)  # (K,)

                # Get ground truth obstacle positions
                if hasattr(env, 'dynamic_obstacle_manager'):
                    dm = env.dynamic_obstacle_manager
                    gt_positions = dm.get_obstacle_positions_gt(0)  # (K, 3)
                else:
                    true_confidence = None
                    gt_positions = None
                    continue

                if gt_positions is not None and gt_positions.shape[0] > 0:
                    # Compute bearing and range
                    drone_pos = env.position[0:1]  # (1, 3)
                    yaw_vec = env.yaw_vector[0:1]  # (1, 3)
                    gt_pos_batch = gt_positions.unsqueeze(0)  # (1, K, 3)

                    bearing = compute_bearing(drone_pos, yaw_vec, gt_pos_batch).squeeze(0)  # (K,)
                    range_ = compute_range(drone_pos, gt_pos_batch).squeeze(0)  # (K,)

                    # Scene illumination (placeholder — replace with actual sensor reading)
                    illumination = 0.8  # default indoor level

                    for k in range(gt_positions.shape[0]):
                        logs.append({
                            "bearing": bearing[k].item(),
                            "range": range_[k].item(),
                            "illumination": illumination,
                            "true_confidence": true_confidence[k].item() if true_confidence is not None else 1.0,
                            "rollout": rollout_idx,
                            "step": step,
                        })

            # Step env
            if policy.is_recurrent:
                actions, _ = policy({
                    k: th.as_tensor(v, device=policy.device).unsqueeze(0) if isinstance(v, np.ndarray)
                    else v for k, v in obs.items()
                })
            else:
                actions = policy({
                    k: th.as_tensor(v, device=policy.device).unsqueeze(0) if isinstance(v, np.ndarray)
                    else v for k, v in obs.items()
                })

            obs, reward, done, info = env.step(actions, is_test=True)
            done = done[0] if isinstance(done, th.Tensor) else done
            if done:
                break

    return logs


def main():
    parser = argparse.ArgumentParser(description="Collect confidence proxy training data")
    parser.add_argument("--env_config", type=str,
                        default="extreme_avoid/configs/eval_cfg/eval_dynamic.yaml",
                        help="Environment config YAML")
    parser.add_argument("--policy_cfg_file", type=str,
                        default="extreme_avoid/configs/policy_cfg/tracker_fused_yaw.yaml",
                        help="Policy config YAML")
    parser.add_argument("--weight", type=str, default=None, help="Policy checkpoint (optional)")
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--output", type=str, default="./confidence_logs.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load env
    if os.path.exists(args.env_config):
        with open(args.env_config, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: {args.env_config} not found. Using defaults.")
        config = {
            "env_class": "dynamic_avoidance_env",
            "env": {"num_envs": 1, "visual": True, "single_env": True, "requires_grad": False}
        }

    env_class = env_aliases[config["env_class"]]
    env = env_class(**config["env"])

    # Load policy (or create placeholder)
    # Read policy_kwargs from policy_cfg YAML to construct policy correctly
    policy_kwargs = {}
    if os.path.exists(args.policy_cfg_file):
        with open(args.policy_cfg_file, "r") as f:
            policy_cfg = yaml.safe_load(f)
            policy_kwargs = policy_cfg.get("policy", {})
    else:
        print(f"Warning: {args.policy_cfg_file} not found. Policy may not be constructed correctly.")

    if args.weight and os.path.exists(args.weight):
        policy_class = policy_aliases.get("TrackerFusedPolicy", MultiInputPolicy)
        if issubclass(policy_class, MultiInputPolicy):
            policy = policy_class(env.observation_space, **policy_kwargs)
        else:
            policy = policy_class(**policy_kwargs)
        policy.load(args.weight)
        policy.eval()
    else:
        policy = MultiInputPolicy(env.observation_space, **policy_kwargs)
        policy.eval()

    # Collect logs
    print(f"Collecting {args.num_rollouts} rollouts...")
    logs = collect_logs(env, policy, num_rollouts=args.num_rollouts)

    # Save to CSV
    df = pd.DataFrame(logs)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(logs)} samples to {args.output}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSummary statistics:")
    print(df.describe())

    env.close()


if __name__ == "__main__":
    main()
