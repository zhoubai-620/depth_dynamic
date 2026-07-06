"""
Registry extension for extreme_avoid.

Extends DepthNav's existing env_aliases and policy_aliases dictionaries
with new dynamic avoidance classes. All scripts (train, eval) should import
from this file instead of depthnav's original alias files.

Key design decision (per skill.md §0.2):
  train_bptt.py has a hardcoded `if policy_class == MultiInputPolicy:` check.
  This MUST be replaced with `issubclass()` — done in train_bptt_ext.py.
"""

from depthnav.envs.env_aliases import env_aliases
from depthnav.policies.policy_aliases import policy_aliases

# Lazy imports to avoid circular dependencies
# New classes are registered when their modules are imported

def _register():
    """Register all new classes. Called on first import to avoid circular deps."""
    from extreme_avoid.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
    from extreme_avoid.policies.tracker_fused_policy import TrackerFusedPolicy

    env_aliases.update({
        "dynamic_avoidance_env": DynamicAvoidanceEnv,
    })
    policy_aliases.update({
        "TrackerFusedPolicy": TrackerFusedPolicy,
    })

# Auto-register on import
_register()
