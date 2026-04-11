from typing import List, Tuple

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import LaserScan


def read_ros2_bag(
    bag_path: str,
    downsample: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one ROS 2 bag and return time-aligned arrays.

    Commands are interpolated onto LiDAR timestamps so scan[i] and command[i]
    correspond to the same point in time, regardless of publish-rate differences
    between the /scan and /drive topics.

    Returns
    -------
    lidar    : (N, L)  float32 – range values
    steering : (N,)   float32 – steering angle (rad)
    speed    : (N,)   float32 – speed (m/s)
    times    : (N,)   float64 – Unix timestamps of each scan
    """
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='',
                         output_serialization_format=''),
    )

    lidar_scans:  List[np.ndarray] = []
    lidar_times:  List[float]      = []
    cmd_steering: List[float]      = []
    cmd_speed:    List[float]      = []
    cmd_times:    List[float]      = []

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        t = t_ns * 1e-9

        if topic in ('/scan', 'scan', '/Lidar', 'Lidar'):
            msg  = deserialize_message(raw, LaserScan)
            scan = np.nan_to_num(np.array(msg.ranges, dtype=np.float32),
                                 posinf=0.0, neginf=0.0)
            if downsample > 1:
                scan = scan[::downsample]
            lidar_scans.append(scan)
            lidar_times.append(t)

        elif topic in ('/drive', 'drive', '/Ackermann', 'Ackermann'):
            msg = deserialize_message(raw, AckermannDriveStamped)
            cmd_steering.append(float(msg.drive.steering_angle))
            cmd_speed.append(float(msg.drive.speed))
            cmd_times.append(t)

    if not lidar_scans:
        raise RuntimeError(f'No LiDAR messages found in {bag_path!r}')
    if not cmd_times:
        raise RuntimeError(f'No drive commands found in {bag_path!r}')

    lidar_arr = np.array(lidar_scans,  dtype=np.float32)
    lidar_t   = np.array(lidar_times,  dtype=np.float64)
    cmd_t     = np.array(cmd_times,    dtype=np.float64)

    # Interpolate commands onto lidar timestamps.
    # np.interp clamps at boundary values (no extrapolation).
    steer_aligned = np.interp(lidar_t, cmd_t,
                              np.array(cmd_steering, dtype=np.float32))
    speed_aligned = np.interp(lidar_t, cmd_t,
                              np.array(cmd_speed,    dtype=np.float32))

    return (lidar_arr,
            steer_aligned.astype(np.float32),
            speed_aligned.astype(np.float32),
            lidar_t)


def linear_map(
    x: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Linearly map values from [x_min, x_max] to [y_min, y_max]."""
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
