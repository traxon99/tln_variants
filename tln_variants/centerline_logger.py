#!/usr/bin/env python3
"""
Records synchronized IPS position + LiDAR scans while driving Porto manually.
Drive one clean lap at low speed, then CTRL+C to save.

Output CSV: x,y,heading,r0,...,r539
  - heading: vehicle yaw in radians (world frame)
  - r0..r539: LiDAR ranges downsampled by 2 (540 rays from 1080)
    Perpendicular indices: left-90deg=450, right-90deg=90
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf_transformations import euler_from_quaternion
import numpy as np
import csv
import os


class CenterlineLogger(Node):
    LIDAR_RANGE_MAX = 10.0
    LIDAR_RANGE_MIN = 0.06

    def __init__(self):
        super().__init__('centerline_logger')

        self.declare_parameter('save_path', '/tmp/porto_centerline_raw.csv')
        self.declare_parameter('log_rate_hz', 5.0)

        self.save_path = self.get_parameter('save_path').value
        log_rate = self.get_parameter('log_rate_hz').value
        self._log_interval = 1.0 / log_rate
        self._last_log_time = 0.0
        self.samples = []

        odom_sub = Subscriber(self, Odometry, '/autodrive/roboracer_1/odom')
        lidar_sub = Subscriber(self, LaserScan, '/autodrive/roboracer_1/lidar')
        self._sync = ApproximateTimeSynchronizer(
            [odom_sub, lidar_sub], queue_size=10, slop=0.05
        )
        self._sync.registerCallback(self._callback)

        self.get_logger().info(
            f'CenterlineLogger ready — save path: {self.save_path}\n'
            'Drive Porto at low speed (~0.2 m/s). CTRL+C to stop and save.'
        )

    def _callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_log_time < self._log_interval:
            return
        self._last_log_time = now

        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        _, _, heading = euler_from_quaternion([q.x, q.y, q.z, q.w])

        scan = np.array(scan_msg.ranges, dtype=np.float32)
        scan = np.clip(scan[::2], self.LIDAR_RANGE_MIN, self.LIDAR_RANGE_MAX)

        self.samples.append([x, y, heading] + scan.tolist())

        n = len(self.samples)
        if n % 25 == 0:
            self.get_logger().info(f'Logged {n} samples')

    def save(self):
        if not self.samples:
            self.get_logger().warn('No samples collected — nothing saved.')
            return

        out = os.path.abspath(self.save_path)
        os.makedirs(os.path.dirname(out), exist_ok=True)

        with open(out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'heading'] + [f'r{i}' for i in range(540)])
            writer.writerows(self.samples)

        self.get_logger().info(
            f'Saved {len(self.samples)} samples → {out}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CenterlineLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
