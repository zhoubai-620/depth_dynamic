"""
Unit tests for TTCRiskField and compute_ttc_risk.

Tests:
  1. Known collision scenarios (head-on, parallel, chase)
  2. Gradient flow verification
  3. Edge cases (no obstacles, zero velocity, static obstacles)
  4. Numerical stability at near-collision
"""

import torch as th
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extreme_avoid.risk.ttc_field import compute_ttc_risk, compute_min_ttc


def test_head_on_collision():
    """Drone and obstacle moving directly toward each other."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0]])
    drone_vel = th.tensor([[1.0, 0.0, 0.0]])    # Moving right
    obs_pos = th.tensor([[[5.0, 0.0, 0.0]]])    # 5m ahead
    obs_vel = th.tensor([[[-1.0, 0.0, 0.0]]])   # Moving left

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel,
        ttc_threshold=3.0, distance_safe=1.5
    )

    # Risk should be positive (collision in 2.5s, within 3s threshold)
    assert risk.item() > 0.0, f"Expected positive risk, got {risk.item()}"
    # Avoidance direction should point away (left, negative x)
    assert avoid_dir[0, 0] < 0.0, f"Expected avoid direction pointing left, got {avoid_dir}"
    print(f"PASS head-on: risk={risk.item():.4f}, avoid_dir={avoid_dir[0].tolist()}")


def test_parallel_no_collision():
    """Drone and obstacle moving parallel, never intersect."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0]])
    drone_vel = th.tensor([[1.0, 0.0, 0.0]])
    obs_pos = th.tensor([[[0.0, 10.0, 0.0]]])   # 10m to the side
    obs_vel = th.tensor([[[1.0, 0.0, 0.0]]])     # Same direction

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel,
        ttc_threshold=3.0, distance_safe=1.5
    )

    # Risk should be zero (moving away, parallel)
    assert risk.item() < 0.01, f"Expected near-zero risk, got {risk.item()}"
    print(f"PASS parallel: risk={risk.item():.4f}")


def test_chase_scenario():
    """Drone approaching obstacle from behind (positive TTC)."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0]])
    drone_vel = th.tensor([[2.0, 0.0, 0.0]])
    obs_pos = th.tensor([[[3.0, 0.0, 0.0]]])
    obs_vel = th.tensor([[[0.5, 0.0, 0.0]]])

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel,
        ttc_threshold=5.0, distance_safe=1.0
    )

    # Drone closing in on obstacle: TTC = 3/(2-0.5) = 2s → within 5s threshold
    assert risk.item() > 0.0, f"Expected positive risk for chase, got {risk.item()}"
    print(f"PASS chase: risk={risk.item():.4f}")


def test_no_obstacles():
    """Empty obstacle list."""
    drone_pos = th.tensor([[1.0, 1.0, 1.0]])
    drone_vel = th.tensor([[1.0, 0.0, 0.0]])
    obs_pos = th.zeros((1, 0, 3))
    obs_vel = th.zeros((1, 0, 3))

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel
    )

    assert risk.item() == 0.0, f"Expected zero risk, got {risk.item()}"
    print(f"PASS no_obstacles: risk={risk.item():.4f}")


def test_gradient_flow():
    """Verify autograd gradients exist and are finite."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    drone_vel = th.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    obs_pos = th.tensor([[[5.0, 0.0, 0.0]]], requires_grad=True)
    obs_vel = th.tensor([[[-1.0, 0.0, 0.0]]], requires_grad=True)

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel,
        ttc_threshold=3.0, distance_safe=1.5
    )

    # Backward through risk
    risk.backward()
    assert drone_pos.grad is not None, "No gradient for drone position"
    assert drone_pos.grad.abs().sum() > 0, "Zero gradient for drone position"
    assert not th.isnan(drone_pos.grad).any(), "NaN in drone position gradient"
    assert not th.isinf(drone_pos.grad).any(), "Inf in drone position gradient"
    print(f"PASS gradient: grad_norm={drone_pos.grad.norm().item():.6f}")


def test_min_ttc():
    """Verify minimum TTC computation."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0]])
    drone_vel = th.tensor([[1.0, 0.0, 0.0]])
    obs_pos = th.tensor([[[5.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    obs_vel = th.tensor([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])

    min_ttc = compute_min_ttc(drone_pos, drone_vel, obs_pos, obs_vel)

    # TTCs: 5/(1-(-1))=2.5s, 2/(1-(-1))=1.0s → min = 1.0s
    assert abs(min_ttc.item() - 1.0) < 0.1, f"Expected min_ttc≈1.0, got {min_ttc.item()}"
    print(f"PASS min_ttc: {min_ttc.item():.3f}")


def test_batch_operation():
    """Multiple batch elements."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    drone_vel = th.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    obs_pos = th.tensor([[[5.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    obs_vel = th.tensor([[[-1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]])

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel,
        ttc_threshold=3.0, distance_safe=1.5
    )

    assert risk.shape == (2,), f"Expected risk shape (2,), got {risk.shape}"
    assert risk[1] > risk[0], f"Closer obstacle should have higher risk"
    print(f"PASS batch: risk={risk.tolist()}")


def test_numerical_stability():
    """Test near-zero relative velocity (division by ~zero)."""
    drone_pos = th.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    drone_vel = th.tensor([[0.0, 0.0, 0.0]])
    obs_pos = th.tensor([[[1.0, 0.0, 0.0]]])
    obs_vel = th.tensor([[[0.0, 0.0, 0.0]]])

    risk, avoid_dir = compute_ttc_risk(
        drone_pos, drone_vel, obs_pos, obs_vel,
        ttc_threshold=3.0, distance_safe=1.5
    )

    # Static obstacle nearby: risk from distance-based barrier
    assert not th.isnan(risk).any(), "NaN in risk"
    assert not th.isinf(risk).any(), "Inf in risk"
    print(f"PASS stability: risk={risk.item():.4f}")


if __name__ == "__main__":
    print("=== TTCRiskField Tests ===\n")
    test_head_on_collision()
    test_parallel_no_collision()
    test_chase_scenario()
    test_no_obstacles()
    test_gradient_flow()
    test_min_ttc()
    test_batch_operation()
    test_numerical_stability()
    print("\n=== All TTC tests passed! ===")
