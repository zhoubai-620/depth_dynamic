"""
extreme_avoid: Extreme Environment Dynamic Obstacle Tracking & Avoidance System.

Integrates DPTracker's dual-prompt perception backbone (illumination + viewpoint
prompters) with DepthNav's BPTT training framework to achieve robust dynamic
obstacle avoidance under extreme lighting conditions.

Three innovation pillars:
  1. Shared dual-prompt perception backbone (perception/)
  2. TTC-based dynamic collision risk field replacing static geodesic (risk/)
  3. Tracking-confidence-aware yaw policy (confidence_proxy.py + get_reward yaw)
"""

__version__ = "1.0.0"
