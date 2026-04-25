"""
Shared utilities for tln_variants inference nodes.
No ROS2 or TensorFlow imports — safe to import in any context (including unit tests).
"""
import numpy as np

# LiDAR preprocessing constants
NOISE_STD = 0.5       # Gaussian noise std dev for sim-to-real transfer (metres)
CLIP_MAX = 10.0       # Maximum valid LiDAR range (metres)

# Steering/speed constants
STEER_LIMIT = 0.52    # Physical steering angle limit (radians)
DEFAULT_MIN_SPEED = 1.0
DEFAULT_MAX_SPEED = 8.0
RLN_SPEED_OUT_MIN = -0.5
RLN_SPEED_OUT_MAX = 9.0

# Control loop
DEFAULT_HZ = 40.0
EXP_MAP_ALPHA = 3.0   # Curvature for exponential speed mapping in tln_standard


def linear_map(x, x_min, x_max, y_min, y_max):
    """
    Linear interpolation from [x_min, x_max] to [y_min, y_max].
    Returns midpoint if x_min == x_max to avoid division by zero.
    """
    if x_max == x_min:
        return (y_max + y_min) / 2.0
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


def preprocess_scan(ranges, num_output_ranges, add_noise=False,
                    noise_std=NOISE_STD, clip_max=CLIP_MAX):
    """
    Standard LiDAR preprocessing pipeline used by all TLN/RLN nodes.

    Args:
        ranges: Raw LaserScan.ranges (list or array-like).
        num_output_ranges: Number of range bins after downsampling.
        add_noise: Whether to inject Gaussian noise (True in simulation).
        noise_std: Standard deviation of noise (default 0.5 m).
        clip_max: Clip ranges beyond this distance (default 10 m).

    Returns:
        np.ndarray of shape (num_output_ranges,), dtype float32.
    """
    scans = np.array(ranges, dtype=np.float64)
    if add_noise:
        scans = scans + np.random.normal(0.0, noise_std, scans.shape)
    scans = np.clip(scans, 0.0, clip_max)
    scans = np.nan_to_num(scans, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.linspace(0, len(scans) - 1, num_output_ranges, dtype=int)
    return scans[idx].astype(np.float32)
