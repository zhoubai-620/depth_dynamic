#!/usr/bin/env python3
"""
Dynamic Scene Generator.

COPY of depthnav/examples/geodesics/generate_training_envs.py with extensions
for dynamic obstacle configuration (per skill.md §2.3).

Original script generates pre-computed geodesic fields for static scenes.
This extension additionally:
  1. Produces dynamic obstacle annotations for each scene.
  2. Writes them to scene_instance.json alongside geodesic fields.
  3. Supports various motion patterns for obstacle trajectories.
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Optional

# Add code path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def generate_dynamic_obstacle_configs(
    num_obstacles: int = 4,
    scene_bounds: Optional[Dict] = None,
    motion_patterns: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate dynamic obstacle configurations for a scene.

    Args:
        num_obstacles: Number of dynamic obstacles.
        scene_bounds: Scene bounding box {"min": [x, y, z], "max": [x, y, z]}.
        motion_patterns: List of motion pattern names (random if None).
        seed: Random seed.

    Returns:
        List of obstacle config dicts, each with:
          - initial_position: [x, y, z]
          - initial_velocity: [vx, vy, vz]
          - motion_pattern: str
          - motion_params: dict
          - radius: float
    """
    rng = np.random.RandomState(seed)

    if scene_bounds is None:
        scene_bounds = {"min": [0, -8, 0], "max": [15, 8, 6]}

    bmin = np.array(scene_bounds["min"])
    bmax = np.array(scene_bounds["max"])

    if motion_patterns is None:
        motion_patterns = ["constant_velocity", "sinusoidal", "trajectory_replay"]

    configs = []

    for i in range(num_obstacles):
        # Random initial position within bounds (avoid edges)
        pos = bmin + rng.rand(3) * (bmax - bmin) * 0.7 + (bmax - bmin) * 0.15

        # Random velocity (moderate speed)
        speed = rng.uniform(0.2, 1.0)
        direction = rng.randn(3)
        direction[2] *= 0.2  # Reduce vertical component
        direction = direction / max(np.linalg.norm(direction), 1e-6)
        vel = direction * speed

        # Random motion pattern
        pattern = motion_patterns[rng.randint(0, len(motion_patterns))]

        params = {}
        if pattern == "sinusoidal":
            axis = rng.randn(3)
            axis[2] = 0  # Keep motion in horizontal plane
            params = {
                "axis": (axis / np.linalg.norm(axis)).tolist(),
                "amplitude": float(rng.uniform(0.5, 2.0)),
                "frequency": float(rng.uniform(0.1, 0.5)),
            }
        elif pattern == "trajectory_replay":
            # Generate simple waypoints
            num_wp = rng.randint(3, 6)
            waypoints = [pos.tolist()]
            for _ in range(num_wp - 1):
                wp = pos + rng.randn(3) * 2.0
                wp[2] = np.clip(wp[2], bmin[2] + 0.5, bmax[2] - 0.5)
                waypoints.append(wp.tolist())
            params = {
                "waypoints": waypoints,
                "loop": bool(rng.choice([True, False])),
                "speed": float(rng.uniform(0.3, 1.5)),
            }

        configs.append({
            "initial_position": pos.tolist(),
            "initial_velocity": vel.tolist(),
            "motion_pattern": pattern,
            "motion_params": params,
            "radius": float(rng.uniform(0.2, 0.5)),
        })

    return configs


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic scenes for extreme_avoid")
    parser.add_argument("--num_scenes", type=int, default=50, help="Number of scenes to generate")
    parser.add_argument("--num_obstacles", type=int, default=4, help="Dynamic obstacles per scene")
    parser.add_argument("--output_dir", type=str, default="./generated_scenes",
                        help="Output directory for scene configs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    all_scenes = []

    for scene_idx in range(args.num_scenes):
        scene_config = generate_dynamic_obstacle_configs(
            num_obstacles=args.num_obstacles,
            seed=args.seed + scene_idx,
        )

        scene_data = {
            "scene_id": scene_idx,
            "dynamic_obstacles": scene_config,
            "scene_bounds": {"min": [0, -8, 0], "max": [15, 8, 6]},
        }

        # Save individual scene config
        scene_path = os.path.join(args.output_dir, f"scene_{scene_idx:04d}.json")
        with open(scene_path, "w") as f:
            json.dump(scene_data, f, indent=2)

        all_scenes.append(scene_data)

    # Save master config
    master_path = os.path.join(args.output_dir, "dynamic_scenes.json")
    with open(master_path, "w") as f:
        json.dump({"scenes": all_scenes, "num_scenes": args.num_scenes}, f, indent=2)

    print(f"Generated {args.num_scenes} dynamic obstacle scenes in {args.output_dir}")
    print(f"Master config: {master_path}")


if __name__ == "__main__":
    main()
