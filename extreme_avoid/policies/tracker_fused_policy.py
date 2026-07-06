"""
Tracker-Fused Policy (Innovation 1 + 2 integration).

Inherits from depthnav.policies.multi_input_policy.MultiInputPolicy and extends it with:
  1. TrackerFusedExtractor feature extractor alias registration.
  2. MotionHead for obstacle motion prediction.
  3. Extended forward() returning (actions, aux_dict, latent) instead of just (actions, latent).

Design (per skill.md §3.9):
  - Override class attribute `feature_extractor_alias` to add TrackerFusedExtractor.
  - Hold a MotionHead instance, invoked after recurrent latent is computed.
  - Extend forward() return to include aux_dict for env.get_reward() access.
  - Parent's MlpPolicy.forward() (policy net + output activation) is reused.
"""

import torch as th
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, Union, Type
from gymnasium import spaces

from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.policies.extractors import FeatureExtractor

from ..perception.tracker_fused_extractor import TrackerFusedExtractor
from ..prediction.motion_head import MotionHead


class TrackerFusedPolicy(MultiInputPolicy):
    """
    Policy with tracker-fused perception and motion prediction.

    Extends MultiInputPolicy's recurrent architecture:
      observations → feature_extractor → feature_norm → recurrent_extractor
        → motion_head (new) → policy_net → actions

    Returns (actions, aux_dict, latent) where aux_dict contains:
      - obstacle_prediction: motion_head outputs
      - obstacle_outputs: tracker detection results
    """

    # Extend parent's feature extractor alias registry
    feature_extractor_alias = {
        **MultiInputPolicy.feature_extractor_alias,
        "TrackerFusedExtractor": TrackerFusedExtractor,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        net_arch: Dict[str, list],
        activation_fn: Union[str, nn.Module],
        output_activation_fn: Union[str, nn.Module],
        feature_extractor_class: Union[Type[FeatureExtractor], str],
        output_activation_kwargs: Optional[Dict[str, Any]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        device: th.device = th.device("cuda"),
    ):
        super().__init__(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            feature_extractor_class=feature_extractor_class,
            output_activation_kwargs=output_activation_kwargs,
            feature_extractor_kwargs=feature_extractor_kwargs,
            device=device,
        )

        # Motion head — only if recurrent (latent exists)
        self._has_motion_head = False
        if self.is_recurrent:
            motion_cfg = net_arch.get("motion_head", {})
            self.motion_head = MotionHead(
                latent_dim=self._latent_dim,
                embedding_dim=motion_cfg.get("embedding_dim", 64),
                hidden_dims=motion_cfg.get("hidden_dims", [128, 64]),
                prediction_horizon=motion_cfg.get("prediction_horizon", 5),
                dropout=motion_cfg.get("dropout", 0.1),
            )
            self._has_motion_head = True

        # Cache for the last aux dict (read by env.get_reward)
        self._last_aux_dict: Dict[str, th.Tensor] = {}

    def forward(
        self,
        obs: Dict[str, th.Tensor],
        latent: Optional[th.Tensor] = None,
        return_aux: bool = True,
    ) -> Union[
        Tuple[th.Tensor, th.Tensor],
        Tuple[th.Tensor, Dict[str, th.Tensor], th.Tensor],
    ]:
        """
        Forward pass with extended return for env integration.

        Args:
            obs: Observation dict (state, depth, color, target, etc.).
            latent: Previous recurrent latent state (None on first step).
            return_aux: If True, return aux_dict for env.get_reward().

        Returns:
            If return_aux:
              (actions, aux_dict, latent)
            Else:
              (actions, latent)  — same as parent.
        """
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)

        aux_dict = {}

        if self.is_recurrent:
            latent = self.recurrent_extractor(features, latent)

            # Motion prediction (Innovation 2)
            if self._has_motion_head:
                # Get obstacle embedding from feature extractor
                if isinstance(self.feature_extractor, TrackerFusedExtractor):
                    obstacle_emb = self.feature_extractor.get_obstacle_embedding(
                        obs.get("color", th.zeros(features.shape[0], 3, 64, 64,
                                                   device=features.device))
                    )
                else:
                    obstacle_emb = th.zeros(
                        (features.shape[0], 64), device=features.device
                    )

                motion_pred = self.motion_head(latent, obstacle_emb)
                aux_dict["obstacle_prediction"] = motion_pred

                # Also include tracker outputs
                if isinstance(self.feature_extractor, TrackerFusedExtractor):
                    aux_dict["obstacle_outputs"] = (
                        self.feature_extractor.last_obstacle_outputs
                    )

            actions = super(MultiInputPolicy, self).forward(latent)
            self._last_aux_dict = aux_dict

            if return_aux:
                return actions, aux_dict, latent
            return actions, latent

        # Non-recurrent path
        actions = super(MultiInputPolicy, self).forward(features)
        self._last_aux_dict = aux_dict

        if return_aux:
            return actions, aux_dict, latent
        return actions

    @property
    def last_aux_dict(self) -> Dict[str, th.Tensor]:
        """The most recent aux_dict from forward(). Read by env.get_reward()."""
        return self._last_aux_dict

    @property
    def last_obstacle_prediction(self) -> Optional[Dict[str, th.Tensor]]:
        """Shortcut to get the last motion prediction outputs."""
        return self._last_aux_dict.get("obstacle_prediction", None)

    @property
    def last_obstacle_outputs(self) -> Optional[Dict[str, th.Tensor]]:
        """Shortcut to get the last tracker detection outputs."""
        return self._last_aux_dict.get("obstacle_outputs", None)
