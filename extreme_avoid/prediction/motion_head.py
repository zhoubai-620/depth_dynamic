"""
Motion Prediction Head (Innovation 2).

A lightweight nn.Module that predicts future obstacle motion from the recurrent
latent state. This is the "consumer" side of the tracking pipeline.

Design (per skill.md §3.4):
  - Consumes the recurrent latent (from MultiInputPolicy's GRUCell/LayerNormGRUCell)
    — does NOT maintain its own history buffer.
  - Also consumes current obstacle embedding from TrackerFusedExtractor.
  - Outputs predicted obstacle position/velocity for the next K timesteps.
  - This prediction feeds into TTCRiskField and confidence_proxy.

The latent state already accumulates obstacle history via the recurrent mechanism
built into MultiInputPolicy. This module just decodes that history into a motion
forecast.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class MotionHead(nn.Module):
    """
    Predicts obstacle motion from recurrent latent + current obstacle embedding.

    Architecture:
      latent (from policy GRU) + obstacle_embedding (from extractor)
        → fusion MLP → position/velocity prediction for horizon steps.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 64],
        prediction_horizon: int = 5,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of policy's recurrent latent (hidden_size).
            embedding_dim: Dimension of obstacle embedding from extractor.
            hidden_dims: Hidden layer dimensions for fusion MLP.
            prediction_horizon: Number of future timesteps to predict.
            dropout: Dropout rate.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.prediction_horizon = prediction_horizon

        # Input projection
        in_dim = latent_dim + embedding_dim
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.fusion = nn.Sequential(*layers)
        self.fusion_dim = prev_dim

        # Prediction heads: position and velocity for each future step
        # Output: (B, prediction_horizon * 6) → ((px,py,pz), (vx,vy,vz)) per step
        self.position_head = nn.Linear(self.fusion_dim, prediction_horizon * 3)
        self.velocity_head = nn.Linear(self.fusion_dim, prediction_horizon * 3)

        # Uncertainty estimation (log variance)
        self.position_logvar_head = nn.Linear(self.fusion_dim, prediction_horizon * 3)
        self.velocity_logvar_head = nn.Linear(self.fusion_dim, prediction_horizon * 3)

        # Confidence: how reliable is this prediction?
        self.confidence_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: th.Tensor,
        obstacle_embedding: th.Tensor,
    ) -> Dict[str, th.Tensor]:
        """
        Predict obstacle motion for future timesteps.

        Args:
            latent: (B, latent_dim) — policy recurrent latent.
            obstacle_embedding: (B, embedding_dim) — obstacle identity embedding.

        Returns:
            Dict with:
              - 'predicted_positions': (B, horizon, 3) — predicted positions.
              - 'predicted_velocities': (B, horizon, 3) — predicted velocities.
              - 'position_logvar': (B, horizon, 3) — log variance for positions.
              - 'velocity_logvar': (B, horizon, 3) — log variance for velocities.
              - 'prediction_confidence': (B, 1) — overall prediction reliability.
              - 'step_1_position': (B, 3) — next-step position (convenience).
              - 'step_1_velocity': (B, 3) — next-step velocity (convenience).
        """
        B = latent.shape[0]
        combined = th.cat([latent, obstacle_embedding], dim=1)  # (B, latent_dim + emb_dim)
        fused = self.fusion(combined)  # (B, fusion_dim)

        # Predictions
        pos_raw = self.position_head(fused)      # (B, horizon * 3)
        vel_raw = self.velocity_head(fused)      # (B, horizon * 3)
        pos_logvar = self.position_logvar_head(fused)
        vel_logvar = self.velocity_logvar_head(fused)
        pred_conf = self.confidence_head(fused)  # (B, 1)

        # Reshape to (B, horizon, 3)
        H = self.prediction_horizon
        predicted_positions = pos_raw.view(B, H, 3)
        predicted_velocities = vel_raw.view(B, H, 3)
        position_logvar = pos_logvar.view(B, H, 3)
        velocity_logvar = vel_logvar.view(B, H, 3)

        return {
            "predicted_positions": predicted_positions,
            "predicted_velocities": predicted_velocities,
            "position_logvar": position_logvar,
            "velocity_logvar": velocity_logvar,
            "prediction_confidence": pred_conf,
            "step_1_position": predicted_positions[:, 0:1, :],        # (B, 1, 3) — explicit K=1 dim
            "step_1_velocity": predicted_velocities[:, 0:1, :],       # (B, 1, 3)
        }

    def get_next_step_prediction(
        self,
        latent: th.Tensor,
        obstacle_embedding: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Convenience: return only the next-step (t+1) prediction.

        Returns:
            next_pos: (B, 3) — predicted obstacle position at t+1.
            next_vel: (B, 3) — predicted obstacle velocity at t+1.
            confidence: (B, 1) — prediction confidence.
        """
        outputs = self.forward(latent, obstacle_embedding)
        return (
            outputs["step_1_position"],
            outputs["step_1_velocity"],
            outputs["prediction_confidence"],
        )


class ConstantVelocityMotionHead(nn.Module):
    """
    Naive motion predictor: assumes constant velocity.

    This is a baseline/fallback predictor that can be used when the
    learned MotionHead is not available (e.g., in early training stages).
    """
    def __init__(self, prediction_horizon: int = 5):
        super().__init__()
        self.prediction_horizon = prediction_horizon

    def forward(
        self,
        current_position: th.Tensor,  # (B, 3)
        current_velocity: th.Tensor,  # (B, 3)
        dt: float = 0.02,
    ) -> Dict[str, th.Tensor]:
        """
        Extrapolate constant-velocity motion.

        Args:
            current_position: Obstacle position at time t.
            current_velocity: Obstacle velocity at time t.
            dt: Timestep duration.

        Returns:
            Dict with same keys as MotionHead.forward().
        """
        B = current_position.shape[0]
        H = self.prediction_horizon
        t_steps = th.arange(1, H + 1, device=current_position.device).float() * dt
        t_steps = t_steps.view(1, H, 1).expand(B, -1, 3)  # (B, H, 3)

        pos_pred = current_position.unsqueeze(1) + current_velocity.unsqueeze(1) * t_steps
        vel_pred = current_velocity.unsqueeze(1).expand(-1, H, -1)

        return {
            "predicted_positions": pos_pred,
            "predicted_velocities": vel_pred,
            "position_logvar": th.zeros(B, H, 3, device=current_position.device),
            "velocity_logvar": th.zeros(B, H, 3, device=current_position.device),
            "prediction_confidence": th.ones(B, 1, device=current_position.device),
            "step_1_position": pos_pred[:, 0:1, :],
            "step_1_velocity": vel_pred[:, 0:1, :],
        }
