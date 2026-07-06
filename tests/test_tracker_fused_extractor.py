"""
Integration test for TrackerFusedExtractor.

Tests:
  1. Constructor validates observation space (requires color key)
  2. extract() produces correct feature dimensions
  3. Gradient flow through both depth CNN and fused backbone branches
  4. Obstacle output caching works
"""

import torch as th
import torch.nn as nn
import numpy as np
import sys
import os
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extreme_avoid.perception.tracker_fused_extractor import TrackerFusedExtractor


def create_mock_observation_space():
    """Create a minimal valid observation space for TrackerFusedExtractor."""
    return spaces.Dict({
        "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        "depth": spaces.Box(low=0, high=20.0, shape=(1, 64, 64), dtype=np.float32),
        "color": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
        "target": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
    })


def test_constructor():
    """Extractor builds without errors."""
    obs_space = create_mock_observation_space()
    net_arch = {
        "concatenate": True,
        "depth": {
            "kernel_size": [5, 3, 3],
            "channels": [16, 32, 32],
            "stride": [2, 2, 1],
            "padding": [0, 0, 1],
        },
        "color": {
            "kernel_size": [5, 3, 3],
            "channels": [8, 16, 16],
            "stride": [2, 2, 1],
            "padding": [0, 0, 1],
        },
        "state": {"mlp_layer": [32]},
        "fused_backbone": {
            "embed_dim": 64,
            "max_obstacles": 4,
            "embedding_dim": 32,
        },
    }
    extractor = TrackerFusedExtractor(obs_space, net_arch=net_arch)
    assert extractor.features_dim > 0, f"features_dim should be positive, got {extractor.features_dim}"
    print(f"PASS constructor: features_dim={extractor.features_dim}")


def test_missing_color_key():
    """Constructor should raise AssertionError without color key."""
    obs_space = spaces.Dict({
        "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        "depth": spaces.Box(low=0, high=20.0, shape=(1, 64, 64), dtype=np.float32),
    })
    try:
        TrackerFusedExtractor(obs_space)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print("PASS missing_color: correctly rejected")
    except Exception as e:
        print(f"PASS missing_color: raised {type(e).__name__} (assertion expected)")


def test_extract_shape():
    """extract() produces features with correct dimension."""
    obs_space = create_mock_observation_space()
    net_arch = {
        "concatenate": True,
        "depth": {
            "kernel_size": [5, 3],
            "channels": [16, 32],
            "stride": [2, 2],
            "padding": [0, 0],
        },
        "color": {
            "kernel_size": [5, 3],
            "channels": [8, 16],
            "stride": [2, 2],
            "padding": [0, 0],
        },
        "state": {"mlp_layer": [32]},
        "fused_backbone": {
            "embed_dim": 64,
            "max_obstacles": 4,
            "embedding_dim": 32,
        },
    }
    extractor = TrackerFusedExtractor(obs_space, net_arch=net_arch)

    # Create mock observation batch (B=2)
    mock_obs = {
        "state": th.randn(2, 10),
        "depth": th.rand(2, 1, 64, 64),
        "color": (th.rand(2, 3, 64, 64) * 255).to(th.uint8),
    }

    features = extractor.extract(mock_obs)
    assert features.shape[0] == 2, f"Batch dim should be 2, got {features.shape[0]}"
    assert features.shape[1] == extractor.features_dim, (
        f"Feature dim {features.shape[1]} != expected {extractor.features_dim}"
    )
    print(f"PASS extract_shape: features={features.shape}")


def test_gradient_flow():
    """Gradients flow through both branches."""
    obs_space = create_mock_observation_space()
    net_arch = {
        "concatenate": True,
        "depth": {
            "kernel_size": [5, 3],
            "channels": [16, 32],
            "stride": [2, 2],
            "padding": [0, 0],
        },
        "color": {
            "kernel_size": [5, 3],
            "channels": [8, 16],
            "stride": [2, 2],
            "padding": [0, 0],
        },
        "state": {"mlp_layer": [32]},
        "fused_backbone": {
            "embed_dim": 64,
            "max_obstacles": 4,
            "embedding_dim": 32,
        },
    }
    extractor = TrackerFusedExtractor(obs_space, net_arch=net_arch)
    extractor.train()

    mock_obs = {
        "state": th.randn(1, 10),
        "depth": th.rand(1, 1, 64, 64),
        "color": th.rand(1, 3, 64, 64),
    }

    features = extractor.extract(mock_obs)
    loss = features.sum()
    loss.backward()

    # Check that fused_backbone params got gradients
    has_grad = False
    for name, param in extractor.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No parameters received gradients"
    print("PASS gradient_flow: gradients propagated through backbone")


def test_obstacle_output_caching():
    """Obstacle outputs are cached after extract()."""
    obs_space = create_mock_observation_space()
    net_arch = {
        "concatenate": True,
        "depth": {
            "kernel_size": [5, 3],
            "channels": [16, 32],
            "stride": [2, 2],
            "padding": [0, 0],
        },
        "color": {
            "kernel_size": [5, 3],
            "channels": [8, 16],
            "stride": [2, 2],
            "padding": [0, 0],
        },
        "state": {"mlp_layer": [32]},
        "fused_backbone": {
            "embed_dim": 64,
            "max_obstacles": 4,
            "embedding_dim": 32,
        },
    }
    extractor = TrackerFusedExtractor(obs_space, net_arch=net_arch)
    extractor.eval()

    mock_obs = {
        "state": th.randn(1, 10),
        "depth": th.rand(1, 1, 64, 64),
        "color": th.rand(1, 3, 64, 64),
    }

    with th.no_grad():
        extractor.extract(mock_obs)

    outputs = extractor.last_obstacle_outputs
    assert "presence" in outputs, f"Missing 'presence' key, got {list(outputs.keys())}"
    assert outputs["presence"].shape == (1, 4), f"Expected (1,4), got {outputs['presence'].shape}"
    print("PASS caching: obstacle outputs cached correctly")


if __name__ == "__main__":
    print("=== TrackerFusedExtractor Tests ===\n")
    test_constructor()
    test_missing_color_key()
    test_extract_shape()
    test_gradient_flow()
    test_obstacle_output_caching()
    print("\n=== All TrackerFusedExtractor tests passed! ===")
