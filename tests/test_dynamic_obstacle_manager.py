"""
Unit tests for DynamicObstacleManager.

Tests:
  1. Constant velocity motion pattern
  2. Sinusoidal motion pattern
  3. Trajectory replay pattern (looping and non-looping)
  4. Multi-env obstacle management
  5. Ground truth accessor correctness
"""

import torch as th
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extreme_avoid.envs.dynamic_obstacle_manager import (
    DynamicObstacleManager, MotionPattern
)


def test_constant_velocity():
    """Obstacle moves at constant velocity."""
    manager = DynamicObstacleManager(num_envs=1, device=th.device("cpu"))
    manager.add_obstacle(
        env_id=0,
        initial_position=[0.0, 0.0, 0.0],
        initial_velocity=[1.0, 0.0, 0.0],
        motion_pattern="constant_velocity",
    )

    # Step for 10 timesteps at 50Hz = 0.2s
    for _ in range(10):
        manager.step(dt=0.02)

    pos = manager.get_obstacle_positions_gt(0)
    vel = manager.get_obstacle_velocities_gt(0)

    expected_pos = th.tensor([0.2, 0.0, 0.0])
    assert th.allclose(pos[0], expected_pos, atol=1e-4), f"Expected {expected_pos}, got {pos[0]}"
    print(f"PASS constant_velocity: pos={pos[0].tolist()}")


def test_sinusoidal():
    """Obstacle oscillates sinusoidally."""
    manager = DynamicObstacleManager(num_envs=1, device=th.device("cpu"))
    manager.add_obstacle(
        env_id=0,
        initial_position=[0.0, 0.0, 0.0],
        initial_velocity=[0.0, 0.0, 0.0],
        motion_pattern="sinusoidal",
        motion_params={"axis": [1.0, 0.0, 0.0], "amplitude": 1.0, "frequency": 1.0},
    )

    # Step to t=0.25 (quarter period)
    for _ in range(round(0.25 / 0.02)):
        manager.step(dt=0.02)

    pos = manager.get_obstacle_positions_gt(0)
    # At t=0.25 with freq=1.0, sin(2π*0.25) = sin(π/2) = 1.0
    expected = th.tensor([1.0, 0.0, 0.0])
    assert th.allclose(pos[0], expected, atol=0.05), f"Expected ≈{expected}, got {pos[0]}"
    print(f"PASS sinusoidal: pos={pos[0].tolist()}")


def test_trajectory_replay():
    """Obstacle follows waypoints."""
    manager = DynamicObstacleManager(num_envs=1, device=th.device("cpu"))
    waypoints = [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
    ]
    manager.add_obstacle(
        env_id=0,
        initial_position=[0.0, 0.0, 0.0],
        initial_velocity=[0.0, 0.0, 0.0],
        motion_pattern="trajectory_replay",
        motion_params={"waypoints": waypoints, "loop": False, "speed": 1.0},
    )

    # Step to t=10s (total path ≈ 10m, at speed 1.0→ should be at last waypoint)
    for _ in range(int(10 / 0.02)):
        manager.step(dt=0.02)

    pos = manager.get_obstacle_positions_gt(0)
    # Should be at or near the last waypoint
    expected = th.tensor([5.0, 5.0, 0.0])
    dist = (pos[0] - expected).norm()
    assert dist < 0.5, f"Expected near {expected.tolist()}, got {pos[0].tolist()}"
    print(f"PASS trajectory_replay: pos={pos[0].tolist()}")


def test_multiple_obstacles():
    """Multiple obstacles per environment."""
    manager = DynamicObstacleManager(num_envs=1, device=th.device("cpu"))

    for i in range(3):
        manager.add_obstacle(
            env_id=0,
            initial_position=[float(i), 0.0, 0.0],
            initial_velocity=[0.0, float(i + 1) * 0.5, 0.0],
            motion_pattern="constant_velocity",
        )

    assert manager.num_obstacles(0) == 3

    for _ in range(5):
        manager.step(dt=0.02)

    pos = manager.get_obstacle_positions_gt(0)
    vel = manager.get_obstacle_velocities_gt(0)
    assert pos.shape == (3, 3)
    assert vel.shape == (3, 3)
    print(f"PASS multiple: num_obstacles={manager.num_obstacles(0)}")


def test_ground_truth_accessors():
    """GT accessors return correct shapes."""
    manager = DynamicObstacleManager(num_envs=1, device=th.device("cpu"))
    manager.add_obstacle(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "constant_velocity", obstacle_radius=0.3)

    pos = manager.get_obstacle_positions_gt(0)
    vel = manager.get_obstacle_velocities_gt(0)
    radii = manager.get_obstacle_radii(0)

    assert pos.shape == (1, 3)
    assert vel.shape == (1, 3)
    assert radii.shape == (1,)
    assert radii[0] == 0.3
    print(f"PASS accessors: pos={pos.tolist()}, radii={radii.tolist()}")


if __name__ == "__main__":
    print("=== DynamicObstacleManager Tests ===\n")
    test_constant_velocity()
    test_sinusoidal()
    test_trajectory_replay()
    test_multiple_obstacles()
    test_ground_truth_accessors()
    print("\n=== All DynamicObstacleManager tests passed! ===")
