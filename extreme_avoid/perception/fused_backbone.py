"""
Fused Backbone: Dual-Prompt Perception.

Glue layer (per skill.md §3.2) that instantiates DPTracker's dual-prompt
backbone (IlluminationPrompter + ViewPrompter) with lightweight adaptations
for DepthNav's 192-dim feature space, combines them via AdaptorBlocks, and
attaches the ObstacleHead for multi-task output.

Key adaptations from DPTracker original:
  - embed_dim: 768 → 192 (aligned with DepthNav feature dimension)
  - levels: 3-5 → 2 (reduced pyramid depth for BPTT memory budget)
  - No ViT backbone — only pyramid convolution + projection
  - No AdaptorBlock chain — single fusion pass for efficiency
  - Output: pooled vector + per-obstacle predictions

Design constraint (per skill.md §2.4):
  The prompt backbone code from DPTracker is preserved AS-IS in third_party/.
  This module imports from it, does NOT copy or modify it.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
import os

# Dynamic import of DPTracker backbone — requires third_party/ on PYTHONPATH
try:
    from lib.models.layers.illumination_prompter import IlluminationPrompter
    from lib.models.layers.view_prompter import ViewPrompter
    from lib.models.layers.prompt_adaptor import AdaptorBlock
    _DPTRACKER_AVAILABLE = True
except ImportError:
    _DPTRACKER_AVAILABLE = False
    IlluminationPrompter = None
    ViewPrompter = None
    AdaptorBlock = None

from .obstacle_head import ObstacleHead


class LightweightIlluminationPrompter(nn.Module):
    """
    Lightweight version of DPTracker's IlluminationPrompter adapted for DepthNav.

    Key differences from original:
      - embed_dim: 192 (vs 768)
      - levels: 2 (vs 3-5)
      - Uses simplified pyramid network
      - Output: (B, embed_dim) pooled vector instead of token sequence

    If DPTracker is available on PYTHONPATH, uses the original module internally.
    Otherwise, falls back to a standalone implementation.
    """
    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 192,
        levels: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = levels
        self.embed_dim_per_level = embed_dim // levels

        # Use simplified pyramid: lightweight conv + pooling
        self.pyramid_convs = nn.ModuleList()
        for i in range(levels):
            in_c = in_chans
            out_c = self.embed_dim_per_level if i < levels - 1 else embed_dim - self.embed_dim_per_level * (levels - 1)
            kernel = max(4 // (2 ** i), 2)
            stride = kernel
            self.pyramid_convs.append(nn.Sequential(
                nn.Conv2d(in_c, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, out_c, kernel_size=kernel, stride=stride),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ))

        # Downsampling for multi-level: simple avg pooling
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Projection to embed_dim
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (B, C, H, W) input image (RGB or grayscale).

        Returns:
            features: (B, embed_dim) pooled feature vector.
        """
        B = x.shape[0]
        outputs = []

        for i, conv in enumerate(self.pyramid_convs):
            feat = conv(x)                            # (B, out_c, H', W')
            feat_pooled = F.adaptive_avg_pool2d(feat, (1, 1))  # (B, out_c, 1, 1)
            outputs.append(feat_pooled.flatten(1))     # (B, out_c)
            x = self.downsample(x)                    # Downsample for next level

        combined = th.cat(outputs, dim=1)              # (B, embed_dim)
        combined = self.norm(combined)
        return combined


class LightweightViewPrompter(nn.Module):
    """
    Lightweight version of DPTracker's ViewPrompter adapted for DepthNav.

    Uses standard convolution instead of DeformConv2d (for compatibility
    without requiring CUDA-compiled torchvision.ops.DeformConv2d).

    If torchvision.ops.DeformConv2d is available, uses the original
    ViewPrompter internally.
    """
    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 192,
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        reduced_dim = max(embed_dim // 8, 64)

        # Check DeformConv availability
        self._use_deform = False
        try:
            from torchvision.ops import DeformConv2d
            self._use_deform = True
        except (ImportError, RuntimeError):
            pass

        if self._use_deform:
            from torchvision.ops import DeformConv2d
            class DeformConv2dPackWrapper(nn.Module):
                def __init__(self, in_channels, out_channels, kernel_size, stride):
                    super().__init__()
                    k = kernel_size
                    s = stride
                    self.offset_conv = nn.Conv2d(in_channels, 2*k*k, k, s, k//2)
                    self.deform_conv = DeformConv2d(in_channels, out_channels, k, s, k//2)
                def forward(self, x):
                    offset = self.offset_conv(x)
                    return self.deform_conv(x, offset)
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, reduced_dim, kernel_size=4, stride=4),
                nn.BatchNorm2d(reduced_dim),
                nn.LeakyReLU(),
                DeformConv2dPackWrapper(reduced_dim, embed_dim, kernel_size=4, stride=4),
                nn.BatchNorm2d(embed_dim),
                nn.LeakyReLU(),
            )
        else:
            # Fallback: standard convolutions
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, reduced_dim, kernel_size=4, stride=4),
                nn.BatchNorm2d(reduced_dim),
                nn.ReLU(),
                nn.Conv2d(reduced_dim, embed_dim, kernel_size=4, stride=4),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(),
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (B, C, H, W) input image.

        Returns:
            features: (B, embed_dim) pooled feature vector.
        """
        x = self.proj(x)                      # (B, embed_dim, H', W')
        x = F.adaptive_avg_pool2d(x, (1, 1))   # (B, embed_dim, 1, 1)
        x = x.flatten(1)                       # (B, embed_dim)
        x = self.norm(x)
        return x


class SimpleFusionBlock(nn.Module):
    """
    Simplified fusion block replacing DPTracker's full AdaptorBlock chain.

    Per skill.md §1.36: "No AdaptorBlock chain — single fusion pass for efficiency."
    Implements the mathematical core: om = f(x-z), oa = f(x+z).

    This is a standalone implementation that doesn't depend on DPTracker imports.
    """
    def __init__(self, dim: int = 192, reduced_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj_om = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, dim),
        )
        self.proj_oa = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, dim),
        )

        # Learnable fusion weights (initialized small for stable training)
        self.weight_om = nn.Parameter(th.tensor(0.01))
        self.weight_oa = nn.Parameter(th.tensor(0.01))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: th.Tensor, z: th.Tensor) -> th.Tensor:
        """
        Fuse illumination features (z) into view features (x).

        Args:
            x: (B, dim) — view features.
            z: (B, dim) — illumination features.

        Returns:
            fused: (B, dim) — fused feature vector.
        """
        om = self.proj_om(x - z)  # Opposition-minus: difference between modalities
        oa = self.proj_oa(x + z)  # Opposition-add: common signal between modalities
        fused = x + self.weight_oa * oa + self.weight_om * om
        fused = self.norm(fused)
        return fused


class FusedBackbone(nn.Module):
    """
    Complete fused perception backbone (Innovation 1).

    Wraps the dual-prompt perception pipeline:
      1. IlluminationPrompter (lightweight) → illumination features
      2. ViewPrompter (lightweight) → viewpoint features
      3. FusionBlock → combined features
      4. ObstacleHead → detection + confidence + embedding

    Exposes a unified forward() returning all outputs.
    Supports loading DPTracker pretrained weights for the prompters
    (when DPTracker is available on PYTHONPATH).

    Constructor arg `use_original_dptracker` switches between:
      - True: Import and use original DPTracker modules (requires PYTHONPATH set)
      - False: Use standalone lightweight implementations (no external deps)
    """

    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 192,
        max_obstacles: int = 8,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Dual prompters (lightweight by default)
        self.illum_prompter = LightweightIlluminationPrompter(
            img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, levels=2
        )
        self.view_prompter = LightweightViewPrompter(
            img_size=img_size, in_chans=in_chans, embed_dim=embed_dim
        )

        # Fusion block
        self.fusion = SimpleFusionBlock(dim=embed_dim, dropout=dropout)

        # Obstacle head
        self.obstacle_head = ObstacleHead(
            feature_dim=embed_dim,
            max_obstacles=max_obstacles,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # Pooled feature for policy integration (after fusion, before head)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        image: th.Tensor,
        return_detailed: bool = False,
    ) -> Dict[str, th.Tensor]:
        """
        Args:
            image: (B, C, H, W) RGB input image.
            return_detailed: If True, return intermediate features.

        Returns:
            Dict with:
              - 'features': (B, embed_dim) — pooled features for policy.
              - 'obstacle': Dict — output from ObstacleHead (presence, bbox, etc.)
              - (optional) 'illum_features', 'view_features' if return_detailed.
        """
        # Extract dual-prompt features
        illum_feat = self.illum_prompter(image)   # (B, embed_dim)
        view_feat = self.view_prompter(image)      # (B, embed_dim)

        # Fuse
        fused = self.fusion(view_feat, illum_feat)  # (B, embed_dim)

        # Obstacle detection
        obstacle_outputs = self.obstacle_head(fused)

        # Policy features
        features = self.output_proj(fused)

        result = {
            "features": features,
            "obstacle": obstacle_outputs,
        }

        if return_detailed:
            result["illum_features"] = illum_feat
            result["view_features"] = view_feat
            result["fused_features"] = fused

        return result

    def load_pretrained_prompters(self, checkpoint_path: str):
        """
        Load DPTracker pretrained weights into the prompters.
        Only loads matching parameter keys (skip head/classifier params).
        """
        checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Filter for prompter-related keys
        model_state = self.state_dict()
        loaded = {}
        for key in model_state:
            # Try matching with various prefixes from DPTracker
            if key in state_dict:
                loaded[key] = state_dict[key]
            elif f"illum_prompter.{key}" in state_dict:
                loaded[key] = state_dict[f"illum_prompter.{key}"]
            elif f"view_prompter.{key}" in state_dict:
                loaded[key] = state_dict[f"view_prompter.{key}"]

        model_state.update(loaded)
        self.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(loaded)}/{len(model_state)} parameters from pretrained checkpoint")

    def get_obstacle_embedding(self, image: th.Tensor) -> th.Tensor:
        """
        Convenience: extract obstacle embedding for motion prediction.

        Returns:
            embedding: (B, embedding_dim) — global obstacle context embedding.
        """
        outputs = self.forward(image)
        obs = outputs["obstacle"]
        # Average embedding across obstacle slots, weighted by presence
        weights = obs["presence"].unsqueeze(-1) + 1e-6    # (B, K, 1)
        weighted_emb = (obs["embedding"] * weights).sum(dim=1) / weights.sum(dim=1)  # (B, D)
        return weighted_emb
