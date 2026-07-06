"""
Confidence Proxy Model for Tracking Quality Estimation.

CRITICAL DESIGN CONSTRAINT (per skill.md §3.6):
  habitat-sim rendering is NON-differentiable — `th.from_numpy(self.sensor_obs["depth"])`
  produces tensors with no grad_fn. Therefore, real DPTracker confidence scores
  CANNOT backpropagate through the rendering pipeline.

SOLUTION (方案A, recommended by skill.md):
  Fit a small, fully differentiable regression model offline that maps geometric
  quantities (bearing angle, range, illumination intensity) to predicted tracking
  confidence. These inputs are all analytical functions of drone pose and obstacle
  state → autograd works natively without touching rendering.

Usage flow:
  1. collect_confidence_logs.py: Run rollouts, record (bearing, range, illum, conf_true)
  2. fit_confidence_proxy.py: Fit this model to the collected data
  3. DynamicAvoidanceEnv.get_reward(): Load checkpoint, call forward() for loss_track_conf

The proxy is designed as a small MLP that can be loaded as a frozen module
during training. Inputs are computed from drone state (position, yaw) and
obstacle (predicted position) — all within the autograd graph.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ConfidenceProxy(nn.Module):
    """
    Differentiable proxy model predicting DPTracker tracking confidence
    from geometric features: bearing angle, range, illumination intensity.

    This is a regression model — NOT a lookup table or heuristic.
    Parameters are learned offline via fit_confidence_proxy.py.
    """

    def __init__(
        self,
        hidden_dims: list = [64, 32],
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dims: Hidden layer dimensions (excluding input=3, output=1).
            activation: Activation function name.
            dropout: Dropout rate (applied during fitting, disabled during inference).
        """
        super().__init__()
        layers = []
        in_dim = 3  # bearing, range, illumination

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))   # Output: confidence ∈ [0, 1]
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        bearing: th.Tensor,        # (B, K) — relative bearing angle (radians)
        range_: th.Tensor,         # (B, K) — distance to obstacle (meters)
        illumination: th.Tensor,   # (B,) or scalar — scene illumination level
    ) -> th.Tensor:
        """
        Predict tracking confidence for each obstacle.

        Args:
            bearing: Relative bearing angle of each obstacle from drone's yaw.
            range_: Distance from drone to each obstacle.
            illumination: Illumination intensity (scene-level or per-frame).

        Returns:
            confidence: (B, K) — predicted confidence ∈ [0, 1] for each obstacle.
        """
        B = bearing.shape[0]
        K = bearing.shape[1]

        # Normalize inputs to reasonable ranges
        bearing_norm = bearing / (th.pi + 1e-8)           # [-1, 1]
        range_norm = th.tanh(range_ / 10.0)                 # compress [0, inf] → [0, ~1]
        illum = illumination.unsqueeze(-1).expand(-1, K) if illumination.dim() == 1 else illumination
        illum_norm = illum / (illum.max() + 1e-8)           # [0, 1]

        # Concatenate features: (B, K, 3)
        features = th.stack([bearing_norm, range_norm, illum_norm], dim=-1)

        # Reshape for MLP: (B*K, 3)
        features_flat = features.view(B * K, 3)
        conf_flat = self.net(features_flat)
        confidence = conf_flat.view(B, K)

        return confidence

    def get_weighted_confidence(
        self,
        bearing: th.Tensor,
        range_: th.Tensor,
        illumination: th.Tensor,
        risk_weights: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Compute a weighted average confidence, optionally weighted by risk.

        Args:
            bearing, range_, illumination: Same as forward().
            risk_weights: (B, K) per-obstacle risk weighting. If None, uniform.

        Returns:
            weighted_conf: (B,) average confidence per batch element.
        """
        confidence = self.forward(bearing, range_, illumination)  # (B, K)
        if risk_weights is None:
            K = confidence.shape[1]
            risk_weights = th.ones_like(confidence) / K
        weighted_conf = (confidence * risk_weights).sum(dim=1)  # (B,)
        return weighted_conf

    def save(self, filepath: str):
        """Save model checkpoint."""
        th.save({
            "state_dict": self.state_dict(),
            "config": {
                "hidden_dims": [m.out_features for m in self.net
                                if isinstance(m, nn.Linear)][:-1],
            },
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: th.device = th.device("cpu")) -> "ConfidenceProxy":
        """Load model from checkpoint."""
        checkpoint = th.load(filepath, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        model = cls(hidden_dims=config.get("hidden_dims", [64, 32]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model


def compute_bearing(
    drone_position: th.Tensor,    # (B, 3) or (3,)
    drone_yaw_vector: th.Tensor,  # (B, 3) or (3,) — forward-facing direction
    obstacle_positions: th.Tensor,  # (B, K, 3) or (K, 3)
) -> th.Tensor:
    """
    Compute relative bearing angle of each obstacle from drone's forward direction.

    Returns:
        bearing: (B, K) — angle in [-π, π]. 0 = directly ahead, π = directly behind.
    """
    squeeze_single = (drone_position.dim() == 1)
    if squeeze_single:
        drone_position = drone_position.unsqueeze(0)
        drone_yaw_vector = drone_yaw_vector.unsqueeze(0)

    B = drone_position.shape[0]
    if obstacle_positions.dim() == 2:
        obstacle_positions = obstacle_positions.unsqueeze(0).expand(B, -1, -1)

    # Direction from drone to each obstacle
    to_obstacle = obstacle_positions - drone_position.unsqueeze(1)  # (B, K, 3)

    # Project onto XY plane (yaw is horizontal angle)
    to_obstacle_xy = th.stack([
        to_obstacle[..., 0],
        to_obstacle[..., 1],
        th.zeros_like(to_obstacle[..., 2]),
    ], dim=-1)
    to_obstacle_xy = F.normalize(to_obstacle_xy, dim=-1)

    yaw_xy = th.stack([
        drone_yaw_vector[..., 0],
        drone_yaw_vector[..., 1],
        th.zeros_like(drone_yaw_vector[..., 2]),
    ], dim=-1)
    yaw_xy = F.normalize(yaw_xy, dim=-1)

    # atan2(cross product, dot product) gives signed angle
    dot = (yaw_xy.unsqueeze(1) * to_obstacle_xy).sum(dim=-1)  # (B, K)
    cross_z = (yaw_xy.unsqueeze(1)[..., 0] * to_obstacle_xy[..., 1]
               - yaw_xy.unsqueeze(1)[..., 1] * to_obstacle_xy[..., 0])  # (B, K)
    bearing = th.atan2(cross_z, dot.clamp(-1.0, 1.0))

    if squeeze_single:
        bearing = bearing.squeeze(0)

    return bearing


def compute_range(
    drone_position: th.Tensor,
    obstacle_positions: th.Tensor,
) -> th.Tensor:
    """
    Compute Euclidean distance from drone to each obstacle.

    Returns:
        range_: (B, K) — distances in meters.
    """
    squeeze_single = (drone_position.dim() == 1)
    if squeeze_single:
        drone_position = drone_position.unsqueeze(0)

    B = drone_position.shape[0]
    if obstacle_positions.dim() == 2:
        obstacle_positions = obstacle_positions.unsqueeze(0).expand(B, -1, -1)

    range_ = (obstacle_positions - drone_position.unsqueeze(1)).norm(dim=-1)

    if squeeze_single:
        range_ = range_.squeeze(0)
    return range_
