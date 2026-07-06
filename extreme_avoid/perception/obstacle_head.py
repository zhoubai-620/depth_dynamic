"""
Obstacle Detection Head.

A lightweight nn.Module that takes feature maps from the fused DPTracker backbone
and outputs obstacle detection results for the avoidance pipeline.

Outputs (per skill.md §3.1):
  1. Obstacle center + bbox in image/feature coordinates
  2. Tracking confidence scalar (identity continuity, replaces score map)
  3. Fixed-length embedding vector (feeds motion_head and tracker_fused_extractor)

Unlike DPTracker's original "is this my target?" score map, this head predicts:
  - "Is this obstacle the same one from the previous frame?" (identity continuity)
  - Obstacle 3D motion state estimation (velocity, acceleration)
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ObstacleHead(nn.Module):
    """
    Multi-task output head for obstacle detection and tracking.

    Input: Feature map from the fused backbone (prompters + fusion).
    Output: Dict with obstacle bbox, confidence, embedding.
    """

    def __init__(
        self,
        feature_dim: int = 192,      # Backbone output dimension
        max_obstacles: int = 8,      # Maximum obstacles to track
        bbox_output_dim: int = 4,    # (cx, cy, w, h) in normalized coords
        embedding_dim: int = 64,     # Obstacle identity embedding size
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_obstacles = max_obstacles
        self.embedding_dim = embedding_dim

        # Shared feature reduction
        self.reduce = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        reduced_dim = feature_dim // 2

        # Detection head: obstacle presence + bbox per slot
        self.detection_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim // 2, max_obstacles * (1 + bbox_output_dim)),
        )

        # Confidence head: per-obstacle identity continuity score
        self.confidence_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim // 2, max_obstacles),
        )

        # Embedding head: per-obstacle identity embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim // 2, max_obstacles * embedding_dim),
        )

        # Motion state head: per-obstacle velocity estimate (for 3D projection)
        self.motion_state_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim // 2, max_obstacles * 3),  # velocity (vx, vy, vz)
        )

    def forward(
        self,
        features: th.Tensor,
        return_all: bool = False,
    ) -> Dict[str, th.Tensor]:
        """
        Args:
            features: (B, C) — backbone output features (after pooling).
            return_all: If True, return intermediate features too.

        Returns:
            Dict with keys:
              - 'presence': (B, max_obstacles) — probability obstacle exists in slot
              - 'bbox': (B, max_obstacles, 4) — normalized (cx, cy, w, h)
              - 'confidence': (B, max_obstacles) — identity continuity score
              - 'embedding': (B, max_obstacles, embedding_dim) — identity embedding
              - 'velocity': (B, max_obstacles, 3) — estimated velocity
        """
        B = features.shape[0]
        x = self.reduce(features)  # (B, reduced_dim)

        # Detection: presence + bbox
        det_raw = self.detection_head(x)  # (B, max_obstacles * 5)
        presence_bbox = det_raw.view(B, self.max_obstacles, 5)
        presence = th.sigmoid(presence_bbox[..., 0])       # (B, max_obstacles)
        bbox = th.sigmoid(presence_bbox[..., 1:])           # (B, max_obstacles, 4)

        # Confidence
        confidence = th.sigmoid(self.confidence_head(x))    # (B, max_obstacles)

        # Embedding
        emb_raw = self.embedding_head(x)  # (B, max_obstacles * embedding_dim)
        embedding = emb_raw.view(B, self.max_obstacles, self.embedding_dim)
        embedding = F.normalize(embedding, dim=-1)          # Unit-norm embedding

        # Velocity estimate
        velocity_raw = self.motion_state_head(x)            # (B, max_obstacles * 3)
        velocity = velocity_raw.view(B, self.max_obstacles, 3)

        outputs = {
            "presence": presence,
            "bbox": bbox,
            "confidence": confidence,
            "embedding": embedding,
            "velocity": velocity,
        }

        if return_all:
            outputs["features"] = x

        return outputs

    def get_obstacle_predictions(
        self,
        features: th.Tensor,
        confidence_threshold: float = 0.5,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get filtered obstacle predictions above a confidence threshold.

        Returns:
            positions: (B, num_active, 4) — (cx, cy, w, h) for active obstacles
            velocities: (B, num_active, 3) — estimated velocity
            confidences: (B, num_active) — confidence scores
        """
        outputs = self.forward(features)
        presence = outputs["presence"]
        active_mask = presence > confidence_threshold  # (B, max_obstacles)

        # For simplicity, return all and let caller filter
        # This avoids dynamic tensor shapes which cause issues in BPTT
        return (
            th.cat([outputs["bbox"], outputs["presence"].unsqueeze(-1)], dim=-1),
            outputs["velocity"],
            outputs["confidence"],
        )
