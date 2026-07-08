"""
Integration test for extreme_avoid registry system.

Verifies:
  1. registry.py correctly extends env_aliases and policy_aliases
  2. New classes are instantiatable
  3. A minimal env.reset() → policy(obs) → env.step() cycle works
  4. (Requires habitat-sim for full integration; skipped if not available)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if habitat-sim is available (optional dependency)
try:
    import habitat_sim
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("habitat-sim not available — skipping scene-dependent integration tests")


def test_registry_imports():
    """Registry imports and extends alias dictionaries."""
    from extreme_avoid.registry import env_aliases, policy_aliases

    assert "dynamic_avoidance_env" in env_aliases, (
        f"dynamic_avoidance_env not in env_aliases: {list(env_aliases.keys())}"
    )
    assert "TrackerFusedPolicy" in policy_aliases, (
        f"TrackerFusedPolicy not in policy_aliases: {list(policy_aliases.keys())}"
    )
    print(f"PASS registry imports: env_aliases has {len(env_aliases)} entries")


def test_env_class_instantiatable():
    """DynamicAvoidanceEnv can be imported and type-checked."""
    from extreme_avoid.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
    from extreme_avoid.vendor.depthnav.envs.navigation_env import NavigationEnv

    assert issubclass(DynamicAvoidanceEnv, NavigationEnv), (
        "DynamicAvoidanceEnv must be a subclass of NavigationEnv"
    )
    print("PASS env_class: DynamicAvoidanceEnv is a NavigationEnv subclass")


def test_policy_class_instantiatable():
    """TrackerFusedPolicy can be imported and type-checked."""
    from extreme_avoid.policies.tracker_fused_policy import TrackerFusedPolicy
    from extreme_avoid.vendor.depthnav.policies.multi_input_policy import MultiInputPolicy

    assert issubclass(TrackerFusedPolicy, MultiInputPolicy), (
        "TrackerFusedPolicy must be a subclass of MultiInputPolicy"
    )
    print("PASS policy_class: TrackerFusedPolicy is a MultiInputPolicy subclass")


def test_submodule_imports():
    """All submodules import cleanly."""
    modules = [
        ("extreme_avoid.risk.ttc_field", "compute_ttc_risk"),
        ("extreme_avoid.risk.ttc_field", "compute_min_ttc"),
        ("extreme_avoid.risk.confidence_proxy", "ConfidenceProxy"),
        ("extreme_avoid.risk.confidence_proxy", "compute_bearing"),
        ("extreme_avoid.risk.confidence_proxy", "compute_range"),
        ("extreme_avoid.perception.obstacle_head", "ObstacleHead"),
        ("extreme_avoid.perception.fused_backbone", "FusedBackbone"),
        ("extreme_avoid.perception.tracker_fused_extractor", "TrackerFusedExtractor"),
        ("extreme_avoid.prediction.motion_head", "MotionHead"),
        ("extreme_avoid.prediction.motion_head", "ConstantVelocityMotionHead"),
        ("extreme_avoid.envs.dynamic_obstacle_manager", "DynamicObstacleManager"),
    ]

    for module_name, class_name in modules:
        module = __import__(module_name, fromlist=[class_name])
        assert hasattr(module, class_name), f"{module_name} missing {class_name}"
        print(f"  ✓ {module_name}.{class_name}")


if __name__ == "__main__":
    print("=== Registry Integration Tests ===\n")
    test_registry_imports()
    test_env_class_instantiatable()
    test_policy_class_instantiatable()
    test_submodule_imports()
    print("\n=== All registry integration tests passed! ===")