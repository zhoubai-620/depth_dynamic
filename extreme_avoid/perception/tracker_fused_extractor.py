"""
Tracker-Fused Feature Extractor (Innovation 1).

Inherits from depthnav.policies.extractors.ImageExtractor and adds the
dual-prompt fused backbone (from fused_backbone.py) alongside the existing
depth CNN branch. This is the perception integration point for Innovation 1.

Design (per skill.md §3.3):
  - Inherit ImageExtractor → reuse depth preprocessing, CNN setup, feature dim calc
  - In _build(): additionally instantiate FusedBackbone
  - In extract(): run BOTH depth CNN AND fused backbone, then concatenate
  - Follow the existing "LayerNorm + cat/add" pattern from StateTargetImageExtractor
  - Assert color key exists in observation (backbone needs RGB input)

The fused backbone consumes the 'color' observation key and outputs:
  - features: pooled vector for policy
  - obstacle embeddings: for motion prediction and tracking

Dimension management:
  - Depth CNN output: determined by net_arch config (typically ~64-256 dims)
  - FusedBackbone output: embed_dim (default 192)
  - Total features_dim = depth_dim + embed_dim (when concatenate=True)
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Dict, Optional, Type

from extreme_avoid.vendor.depthnav.policies.extractors import (
    ImageExtractor,
    FeatureExtractor,
    set_mlp_feature_extractor,
    create_cnn,
    create_mlp,
)

from .fused_backbone import FusedBackbone


class TrackerFusedExtractor(ImageExtractor):
    """
    Feature extractor that fuses depth CNN features with DPTracker dual-prompt
    backbone features. Extends ImageExtractor to preserve existing depth
    processing pipeline.

    Requires both 'depth' and 'color' observation keys:
      - 'depth': Processed by parent's CNN (ImageExtractor)
      - 'color': Processed by FusedBackbone (illumination + viewpoint prompters)

    The concatenated features are passed to the policy network via
    StateTargetImageExtractor-style fusion (concatenate or elementwise add).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        # Assert required keys
        obs_keys = list(observation_space.spaces.keys()) if hasattr(observation_space, 'spaces') else []
        has_color = any("color" in key for key in obs_keys)
        assert has_color, (
            "TrackerFusedExtractor requires a 'color' key in observation_space. "
            "Add a color sensor in sensor_kwargs configuration."
        )

        super().__init__(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )

    def _build(self, observation_space, net_arch, activation_fn):
        """
        Build extractor: parent's depth CNN + fused backbone + state MLP.

        CRITICAL: net_arch may contain a 'color' key (for FusedBackbone input).
        The parent ImageExtractor._build() treats 'color' as an independent CNN
        branch, but color observations are uint8 and won't survive the parent's
        depth-only preprocessing. We MUST filter 'color' out before calling super.
        """
        # --- Parent: build depth CNN ---
        # Filter out 'color' — parent handles only depth (and state via MLP).
        # The 'color' branch is built separately via FusedBackbone below.
        depth_only_net_arch = {k: v for k, v in net_arch.items() if k != "color"}
        super()._build(observation_space, depth_only_net_arch, activation_fn)

        # Save depth features dimension before we add fusion
        self._depth_features_dim = self._features_dim

        # --- Fused backbone for color input ---
        backbone_kwargs = net_arch.get("fused_backbone", {})
        color_space = None
        for key in observation_space.spaces:
            if "color" in key:
                color_space = observation_space.spaces[key]
                break

        if color_space is not None and len(color_space.shape) == 3:
            in_chans = color_space.shape[0]
            img_h, img_w = color_space.shape[1], color_space.shape[2]
        else:
            # Default: 64x64 RGB
            in_chans = 3
            img_h, img_w = 64, 64

        self.fused_backbone = FusedBackbone(
            img_size=img_h,
            in_chans=in_chans,
            embed_dim=backbone_kwargs.get("embed_dim", 192),
            max_obstacles=backbone_kwargs.get("max_obstacles", 8),
            embedding_dim=backbone_kwargs.get("embedding_dim", 64),
            dropout=backbone_kwargs.get("dropout", 0.1),
        )
        backbone_features_dim = self.fused_backbone.embed_dim

        # --- State MLP (if state key exists) ---
        self._has_state = "state" in observation_space.spaces
        if self._has_state:
            _state_features_dim = set_mlp_feature_extractor(
                self,
                "state",
                observation_space["state"],
                net_arch.get("state", {}),
                activation_fn,
            )

        # --- Fusion mode ---
        self.concatenate = net_arch.get("concatenate", True)

        if self.concatenate:
            total_dim = self._depth_features_dim + backbone_features_dim
            if self._has_state:
                total_dim += _state_features_dim
            self._features_dim = total_dim
        else:
            # Elementwise add: all branches must have same dim
            assert self._depth_features_dim == backbone_features_dim
            if self._has_state:
                assert _state_features_dim == self._depth_features_dim
            self._features_dim = self._depth_features_dim

    def extract(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Extract fused features from observations.

        Args:
            observations: Dict with keys:
                - 'depth': (B, 1, H, W) — depth image
                - 'color': (B, 3, H, W) — RGB image
                - 'state': (B, S) — drone state vector (optional)

        Returns:
            features: (B, features_dim) — fused feature vector.
        """
        # Depth features (from parent ImageExtractor)
        depth_features = super().extract(observations)  # (B, depth_dim)

        # Color features (from fused backbone)
        color_key = None
        for key in observations:
            if "color" in key:
                color_key = key
                break

        if color_key is None:
            # No color input — use zero placeholder (robustness)
            color_image = th.zeros(
                (depth_features.shape[0], 3, 64, 64),
                device=depth_features.device,
            )
        else:
            color_image = observations[color_key]
            if color_image.dtype != th.float32:
                color_image = color_image.float() / 255.0

        backbone_outputs = self.fused_backbone(color_image)
        backbone_features = backbone_outputs["features"]  # (B, backbone_dim)

        # Store obstacle outputs for downstream use (motion_head, env.get_reward)
        self._last_obstacle_outputs = backbone_outputs["obstacle"]

        # Fusion
        if self._has_state:
            state_features = self.state_extractor(observations["state"])
            if self.concatenate:
                combined = th.cat([state_features, depth_features, backbone_features], dim=1)
            else:
                combined = state_features + depth_features + backbone_features
        else:
            if self.concatenate:
                combined = th.cat([depth_features, backbone_features], dim=1)
            else:
                combined = depth_features + backbone_features

        return combined

    @property
    def last_obstacle_outputs(self) -> Dict[str, th.Tensor]:
        """Access obstacle predictions from the last extract() call."""
        if hasattr(self, '_last_obstacle_outputs'):
            return self._last_obstacle_outputs
        return {}

    def get_obstacle_embedding(self, color_image: th.Tensor) -> th.Tensor:
        """
        Extract obstacle embedding for motion prediction.

        Returns:
            embedding: (B, embedding_dim) — obstacle identity embedding.
        """
        return self.fused_backbone.get_obstacle_embedding(color_image)
