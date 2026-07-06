from .obstacle_head import ObstacleHead
from .fused_backbone import FusedBackbone

# Lazy import — requires depthnav installed
try:
    from .tracker_fused_extractor import TrackerFusedExtractor
    __all__ = ["ObstacleHead", "FusedBackbone", "TrackerFusedExtractor"]
except ImportError:
    __all__ = ["ObstacleHead", "FusedBackbone"]
