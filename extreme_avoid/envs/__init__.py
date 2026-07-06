from .dynamic_obstacle_manager import DynamicObstacleManager

# Lazy import — requires depthnav installed
try:
    from .dynamic_avoidance_env import DynamicAvoidanceEnv
    __all__ = ["DynamicObstacleManager", "DynamicAvoidanceEnv"]
except ImportError:
    __all__ = ["DynamicObstacleManager"]
