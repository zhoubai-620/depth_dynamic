"""
Fused Backbone: Dual-Prompt Perception (Original Implementation).

Glue layer (per skill.md §3.2) that instantiates DPTracker's dual-prompt
backbone directly from the vendor directory, combines them via the original
AdaptorBlock, and attaches the ObstacleHead for multi-task output.

Design:
  - Strict reliance on extreme_avoid.vendor.dptracker.layers
  - Dimensions are aggregated (Global Average Pooling) post-fusion to map
    DPTracker's (B, N, C) token sequences to DepthNav's (B, C) policy input.
"""

import torch as th
import torch.nn as nn
from typing import Dict

from extreme_avoid.vendor.dptracker.layers.illumination_prompter import IlluminationPrompter
from extreme_avoid.vendor.dptracker.layers.view_prompter import ViewPrompter
from extreme_avoid.vendor.dptracker.layers.prompt_adaptor import AdaptorBlock

from .obstacle_head import ObstacleHead


class FusedBackbone(nn.Module):
    """
    Complete fused perception backbone using original DPTracker components.
    """

    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 192,
        max_obstacles: int = 8,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        num_adaptor_blocks: int = 1,
        pretrained_checkpoint: str = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.illum_prompter = IlluminationPrompter(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.view_prompter = ViewPrompter(
            img_size=img_size,
            patch_size=16,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=True
        )

        self.adaptor_blocks = nn.ModuleList([
            AdaptorBlock(dim=embed_dim) for _ in range(num_adaptor_blocks)
        ])

        self.obstacle_head = ObstacleHead(
            feature_dim=embed_dim,
            max_obstacles=max_obstacles,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Load DPTracker pretrained weights if checkpoint path provided.
        # Only illum_prompter and view_prompter weights are loaded;
        # obstacle_head and output_proj remain randomly initialized (per skill.md §2.4).
        if pretrained_checkpoint is not None:
            self.load_pretrained_prompters(pretrained_checkpoint)

    def forward(
        self,
        image: th.Tensor,
        return_detailed: bool = False,
    ) -> Dict[str, th.Tensor]:
        """
        Args:
            image: (B, C, H, W) RGB input image.
            return_detailed: If True, return intermediate features.
        """
        illum_feat = self.illum_prompter(image)
        view_feat = self.view_prompter(image)

        x, z = view_feat, illum_feat
        for block in self.adaptor_blocks:
            x, z = block(x, z)

        fused_pooled = x.mean(dim=1)

        obstacle_outputs = self.obstacle_head(fused_pooled)
        features = self.output_proj(fused_pooled)

        result = {
            "features": features,
            "obstacle": obstacle_outputs,
        }

        if return_detailed:
            result["illum_features"] = illum_feat
            result["view_features"] = view_feat
            result["fused_features"] = fused_pooled

        return result

    def load_pretrained_prompters(self, checkpoint_path: str):
        """
        Load DPTracker pretrained weights into the prompters.
        Only loads matching parameter keys (skip head/classifier params).
        """
        checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        model_state = self.state_dict()
        loaded = {}
        for key in model_state:
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
        """
        outputs = self.forward(image)
        obs = outputs["obstacle"]
        weights = obs["presence"].unsqueeze(-1) + 1e-6
        weighted_emb = (obs["embedding"] * weights).sum(dim=1) / weights.sum(dim=1)
        return weighted_emb