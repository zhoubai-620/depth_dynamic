"""
TTC (Time-to-Collision) Analytical Risk Field.

Pure PyTorch implementation — no scipy/skfmm dependency.
Replaces the static FMM-based geodesic field with an analytical,
differentiable collision risk computation.

Key design (per skill.md §3.5):
  - Fully differentiable (autograd native) — no `th.no_grad()` needed.
  - Softened barrier functions prevent gradient explosion at near-collision.
  - Supports two calling modes:
      1. Ground truth obstacle state (for reward supervision)
      2. Predicted obstacle state (for policy internal decision-making)
    The function logic is identical; only input sources differ.

Mathematics:
  Given drone (p_d, v_d) and obstacle (p_o, v_o):
    - Relative velocity: v_rel = v_d - v_o
    - Relative position: p_rel = p_o - p_d
    - Time to closest approach: t* = -(p_rel · v_rel) / ||v_rel||²
    - Distance at closest approach: d(t*) = ||p_rel + v_rel · t*||
    - Risk = barrier(log(1 + exp(β * (d_safe - d(t*)))))

  Avoidance direction: normalized gradient of risk w.r.t drone position.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class TTCRiskField(nn.Module):
    """
    Differentiable TTC collision risk field module.

    Can be used as an nn.Module (for trainable parameters if needed)
    or called statically via compute_ttc_risk.
    """

    def __init__(
        self,
        ttc_threshold: float = 2.0,
        distance_safe: float = 1.5,
        barrier_beta: float = 5.0,
        smoothing_eps: float = 1e-6,
        risk_scale: float = 1.0,
    ):
        """
        Args:
            ttc_threshold: TTC (seconds) below which risk penalty activates.
            distance_safe: Minimum safe separation distance (meters).
            barrier_beta: Softness parameter for the barrier function.
                Higher = sharper barrier transition, lower = smoother.
                Replicates the pattern from DepthNav:
                `log(1 + exp(beta_2 * (dis - radius)))`
            smoothing_eps: Numerical stability for division.
            risk_scale: Multiplier for the final risk value.
        """
        super().__init__()
        self.ttc_threshold = ttc_threshold
        self.distance_safe = distance_safe
        self.barrier_beta = barrier_beta
        self.eps = smoothing_eps
        self.risk_scale = risk_scale

    def forward(
        self,
        drone_position: th.Tensor,       # (B, 3) or (3,)
        drone_velocity: th.Tensor,       # (B, 3) or (3,)
        obstacle_positions: th.Tensor,   # (B, K, 3) or (K, 3)
        obstacle_velocities: th.Tensor,  # (B, K, 3) or (K, 3)
         obstacle_radii: Optional[th.Tensor] = None,  # (K,) or None
         obstacle_mask: Optional[th.Tensor] = None,   # (B, K) — 1 for real, 0 for phantom
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute TTC risk and avoidance direction.

        Args:
            drone_position: Drone world-frame position.
            drone_velocity: Drone world-frame velocity.
            obstacle_positions: Obstacle world-frame positions.
            obstacle_velocities: Obstacle world-frame velocities.
            obstacle_radii: Obstacle bounding radii (optional).
            obstacle_mask: (B,K) mask — 1 for real obstacles, 0 for padding phantoms.

        Returns:
            risk: Scalar risk value per batch element. Shape: (B,) or ().
            avoidance_direction: Normalized direction to move AWAY from danger.
                Shape matches drone_position.
        """
        return compute_ttc_risk(
            drone_position, drone_velocity,
            obstacle_positions, obstacle_velocities,
            obstacle_radii,
            ttc_threshold=self.ttc_threshold,
            distance_safe=self.distance_safe,
            barrier_beta=self.barrier_beta,
            eps=self.eps,
            risk_scale=self.risk_scale,
            obstacle_mask=obstacle_mask,
        )


def compute_ttc_risk(
    drone_position: th.Tensor,
    drone_velocity: th.Tensor,
    obstacle_positions: th.Tensor,
    obstacle_velocities: th.Tensor,
    obstacle_radii: Optional[th.Tensor] = None,
    ttc_threshold: float = 2.0,
    distance_safe: float = 1.5,
    barrier_beta: float = 5.0,
    eps: float = 1e-6,
    risk_scale: float = 1.0,
    obstacle_mask: Optional[th.Tensor] = None,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Pure-function version of TTC risk computation.

    Args:
        drone_position: (B, 3) or (3,) — drone world-frame position.
        drone_velocity: (B, 3) or (3,) — drone world-frame velocity.
        obstacle_positions: (B, K, 3) or (K, 3) — obstacle positions.
        obstacle_velocities: (B, K, 3) or (K, 3) — obstacle velocities.
        obstacle_radii: (K,) — per-obstacle radius for safety margin.
        ttc_threshold: Time-to-collision threshold (seconds).
        distance_safe: Safe separation distance (meters).
        barrier_beta: Barrier softness parameter.
        eps: Numerical epsilon.
        risk_scale: Output risk multiplier.

    Returns:
        risk: Aggregated scalar risk (B,) or ().
        avoidance_direction: Normalized avoid-direction vector.
    """
    # --- Shape normalization ---
    squeeze_single = (drone_position.dim() == 1)
    if squeeze_single:
        drone_position = drone_position.unsqueeze(0)
        drone_velocity = drone_velocity.unsqueeze(0)

    B = drone_position.shape[0]

    # Explicit 3D validation — NO ambiguous dim()==2 fallback.
    # 2D tensors like (B, 3) were misinterpreted as (K, 3) in prior versions,
    # causing cross-env data leakage when num_envs > 1.
    # Callers must explicitly unsqueeze to (B, K, 3) — even for K=1.
    assert obstacle_positions.dim() == 3, (
        f"obstacle_positions must be 3D (B, K, 3), got shape {obstacle_positions.shape}. "
        f"If K=1, unsqueeze to (B, 1, 3) explicitly."
    )
    assert obstacle_velocities.dim() == 3, (
        f"obstacle_velocities must be 3D (B, K, 3), got shape {obstacle_velocities.shape}. "
        f"If K=1, unsqueeze to (B, 1, 3) explicitly."
    )

    K = obstacle_positions.shape[1]

    if K == 0:
        risk = th.zeros(B, device=drone_position.device)
        avoid_dir = th.zeros(B, 3, device=drone_position.device)
        if squeeze_single:
            risk = risk.squeeze(0)
            avoid_dir = avoid_dir.squeeze(0)
        return risk, avoid_dir

    # Expand drone tensors to (B, K, 3) for pairwise computation
    p_d = drone_position.unsqueeze(1).expand(-1, K, -1)      # (B, K, 3)
    v_d = drone_velocity.unsqueeze(1).expand(-1, K, -1)      # (B, K, 3)

    # Relative kinematics
    # p_rel = obstacle - drone: vector from drone to obstacle
    # v_rel = v_obstacle - v_drone: rate of change of p_rel
    # If v_rel · p_rel < 0 → closing → TTC positive (distance shrinking)
    p_rel = obstacle_positions - p_d      # (B, K, 3) — relative position (obstacle minus drone)
    v_rel = obstacle_velocities - v_d     # (B, K, 3) — rate of change of p_rel

    # Distance to each obstacle
    dist = p_rel.norm(dim=2)             # (B, K)

    # --- TTC computation ---
    # v_rel · p_rel = projection of relative position onto relative velocity
    # t* = -(p_rel · v_rel) / ||v_rel||²
    # If objects are moving away (dot > 0), t* < 0 → no collision risk
    v_rel_dot_p_rel = (v_rel * p_rel).sum(dim=2)       # (B, K)
    v_rel_norm_sq = v_rel.norm(dim=2).pow(2) + eps     # (B, K)

    t_star = -v_rel_dot_p_rel / v_rel_norm_sq           # (B, K)

    # Distance at closest approach
    # d(t*) = ||p_rel + v_rel * t*||
    closest_approach_vec = p_rel + v_rel * t_star.unsqueeze(-1).clamp(min=0.0)  # (B, K, 3)
    d_closest = closest_approach_vec.norm(dim=2)         # (B, K)

    # --- Risk computation ---
    # Only count obstacles where t* is in the future (t* >= 0)
    valid_ttc = (t_star >= 0) & (t_star <= ttc_threshold)  # (B, K)

    # Effective safety margin: drone radius + obstacle radius + safe distance
    if obstacle_radii is not None:
        radii = obstacle_radii.unsqueeze(0).expand(B, -1)  # (B, K)
    else:
        radii = th.zeros(B, K, device=drone_position.device)
    effective_safety = distance_safe + radii           # (B, K)

    # Softened barrier function (replicates DepthNav's intersection_barrier pattern):
    #   risk_i = log(1 + exp(beta * (effective_safety - d_closest)))
    barrier_input = barrier_beta * (effective_safety - d_closest)
    # Clamp for numerical stability before exp
    barrier_input = barrier_input.clamp(max=50.0)
    per_obstacle_risk = th.log(1 + th.exp(barrier_input))  # (B, K)

    # Also penalize low TTC (closer = more dangerous)
    # risk_i *= (1 + (ttc_threshold - t*) / ttc_threshold) when valid
    ttc_penalty = th.ones_like(t_star)
    ttc_penalty = th.where(
        valid_ttc,
        1.0 + (ttc_threshold - t_star) / ttc_threshold,
        ttc_penalty,
    )

    per_obstacle_risk = per_obstacle_risk * ttc_penalty

    # Mask: only count valid TTC obstacles
    per_obstacle_risk = per_obstacle_risk * valid_ttc.float()

    # Mask out padded/phantom obstacles (from per-env padding in _get_obstacle_states_for_reward)
    if obstacle_mask is not None:
        per_obstacle_risk = per_obstacle_risk * obstacle_mask.float()

    # Aggregate: sum risk across obstacles
    risk = per_obstacle_risk.sum(dim=1) * risk_scale  # (B,)

    # --- Avoidance direction ---
    # Direction AWAY from each obstacle (from obstacle toward drone, normalized)
    away_direction = -p_rel / (dist.unsqueeze(-1) + eps)  # (B, K, 3): unit vector away

    # Weight each direction by that obstacle's risk contribution
    risk_weights = per_obstacle_risk / (per_obstacle_risk.sum(dim=1, keepdim=True) + eps)
    avoid_dir_weighted = (away_direction * risk_weights.unsqueeze(-1)).sum(dim=1)

    # Fallback: if no risk, return zero vector
    no_risk = risk < eps
    avoid_dir = th.where(
        no_risk.unsqueeze(-1),
        th.zeros_like(avoid_dir_weighted),
        F.normalize(avoid_dir_weighted, dim=1),
    )

    if squeeze_single:
        risk = risk.squeeze(0)
        avoid_dir = avoid_dir.squeeze(0)

    return risk, avoid_dir


def compute_min_ttc(
    drone_position: th.Tensor,
    drone_velocity: th.Tensor,
    obstacle_positions: th.Tensor,
    obstacle_velocities: th.Tensor,
    eps: float = 1e-6,
    obstacle_mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    Compute the minimum TTC across all obstacles (for evaluation metrics).

    Returns:
        min_ttc: (B,) or () — minimum time to collision (positive). Inf if no collision.
    """
    squeeze_single = (drone_position.dim() == 1)
    if squeeze_single:
        drone_position = drone_position.unsqueeze(0)
        drone_velocity = drone_velocity.unsqueeze(0)

    B = drone_position.shape[0]

    assert obstacle_positions.dim() == 3, (
        f"obstacle_positions must be 3D (B, K, 3), got shape {obstacle_positions.shape}"
    )
    assert obstacle_velocities.dim() == 3, (
        f"obstacle_velocities must be 3D (B, K, 3), got shape {obstacle_velocities.shape}"
    )

    K = obstacle_positions.shape[1]
    if K == 0:
        inf = th.full((B,), float('inf'), device=drone_position.device)
        return inf.squeeze(0) if squeeze_single else inf

    p_d = drone_position.unsqueeze(1).expand(-1, K, -1)
    v_d = drone_velocity.unsqueeze(1).expand(-1, K, -1)
    p_rel = obstacle_positions - p_d
    v_rel = obstacle_velocities - v_d
    v_rel_dot_p_rel = (v_rel * p_rel).sum(dim=2)
    v_rel_norm_sq = v_rel.norm(dim=2).pow(2) + eps
    t_star = -v_rel_dot_p_rel / v_rel_norm_sq

    t_star[t_star < 0] = float('inf')

    # Mask out padded phantom obstacles
    if obstacle_mask is not None:
        t_star = th.where(obstacle_mask.bool(), t_star, th.full_like(t_star, float('inf')))

    min_ttc = t_star.min(dim=1).values

    if squeeze_single:
        min_ttc = min_ttc.squeeze(0)
    return min_ttc
