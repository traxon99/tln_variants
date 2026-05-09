#!/usr/bin/env python3
"""
Pure pursuit + curvature-speed controller for the AutoDRIVE RoboRacer.

Loads a pre-optimized raceline CSV (x_m,y_m,speed_mps,heading_rad) and
runs a 40 Hz control loop.

Steering: published normalized [-1, 1]  (1 = full left = +0.52 rad physical)
Throttle: published normalized [0,  1]  (1 = full throttle)

RViz2 topics (all in frame 'world'):
  /pure_pursuit/raceline   — LINE_STRIP colored green→red by target speed
  /pure_pursuit/markers    — lookahead sphere + line from car to lookahead
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import numpy as np

_PHYS_MAX_STEER_RAD = 0.52   # physical steering limit, used for normalisation


def _speed_color(t: float) -> ColorRGBA:
    """Map t ∈ [0,1] to green → yellow → red."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        r, g = 2.0 * t, 1.0
    else:
        r, g = 1.0, 2.0 * (1.0 - t)
    c = ColorRGBA(); c.r = r; c.g = g; c.b = 0.0; c.a = 1.0
    return c


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        self.declare_parameter('raceline_path',  '')
        self.declare_parameter('lookahead_base', 0.5)    # m
        self.declare_parameter('lookahead_gain', 0.15)   # s (scales with speed)
        self.declare_parameter('wheelbase',      0.33)   # m
        self.declare_parameter('v_max',          1.0)    # throttle [0,1] ceiling
        self.declare_parameter('v_min',          0.0)    # throttle [0,1] floor
        self.declare_parameter('speed_scale',    0.5)    # overall multiplier for tuning
        self.declare_parameter('max_steer',      1.0)    # normalized steering limit [0,1]

        raceline_path = self.get_parameter('raceline_path').value
        if not raceline_path:
            self.get_logger().fatal('raceline_path parameter is required')
            raise ValueError('raceline_path not set')

        self.lookahead_base = self.get_parameter('lookahead_base').value
        self.lookahead_gain = self.get_parameter('lookahead_gain').value
        self.wheelbase      = self.get_parameter('wheelbase').value
        self.v_max          = self.get_parameter('v_max').value
        self.v_min          = self.get_parameter('v_min').value
        self.speed_scale    = self.get_parameter('speed_scale').value
        self.max_steer      = self.get_parameter('max_steer').value

        data = np.loadtxt(raceline_path, delimiter=',', skiprows=1)
        self._rl_xy    = data[:, :2]
        self._rl_speed = np.clip(data[:, 2] * self.speed_scale,
                                 self.v_min, self.v_max)
        self.get_logger().info(
            f'Loaded {len(self._rl_xy)} raceline points from {raceline_path}\n'
            f'speed_scale={self.speed_scale}  '
            f'throttle [{self._rl_speed.min():.3f}, {self._rl_speed.max():.3f}]'
        )

        # Control publishers
        self._steer_pub    = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self._throttle_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)

        # Viz — raceline published once with TRANSIENT_LOCAL so RViz2 gets it on connect
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._raceline_pub  = self.create_publisher(Marker,      '/pure_pursuit/raceline', latched)
        self._markers_pub   = self.create_publisher(MarkerArray, '/pure_pursuit/markers',  10)

        self._odom_sub = self.create_subscription(
            Odometry, '/autodrive/roboracer_1/odom', self._odom_cb, 10
        )

        self._publish_raceline_marker()
        self.get_logger().info('PurePursuit node started.')

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _publish_raceline_marker(self):
        """Publish the full raceline once as a speed-coloured LINE_STRIP."""
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'raceline'
        m.id     = 0
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03   # line width in metres
        m.pose.orientation.w = 1.0

        v_min = float(self._rl_speed.min())
        v_max = float(self._rl_speed.max())
        v_range = max(v_max - v_min, 1e-6)

        for (xi, yi), vi in zip(self._rl_xy, self._rl_speed):
            p = Point(); p.x = float(xi); p.y = float(yi); p.z = 0.0
            m.points.append(p)
            m.colors.append(_speed_color((vi - v_min) / v_range))

        # Close the loop
        p = Point(); p.x = float(self._rl_xy[0, 0]); p.y = float(self._rl_xy[0, 1]); p.z = 0.0
        m.points.append(p)
        m.colors.append(m.colors[0])

        self._raceline_pub.publish(m)

    def _publish_viz_markers(self, x: float, y: float,
                             lp: np.ndarray, closest_idx: int):
        """Publish lookahead sphere + line from vehicle to lookahead point."""
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # --- lookahead sphere ---
        sphere = Marker()
        sphere.header.frame_id = 'world'
        sphere.header.stamp    = now
        sphere.ns     = 'lookahead'
        sphere.id     = 1
        sphere.type   = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(lp[0])
        sphere.pose.position.y = float(lp[1])
        sphere.pose.position.z = 0.0
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
        sphere.color.r = 1.0; sphere.color.g = 1.0; sphere.color.b = 0.0
        sphere.color.a = 1.0
        markers.markers.append(sphere)

        # --- line from vehicle to lookahead ---
        line = Marker()
        line.header.frame_id = 'world'
        line.header.stamp    = now
        line.ns     = 'lookahead_line'
        line.id     = 2
        line.type   = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.02
        line.pose.orientation.w = 1.0
        line.color.r = 1.0; line.color.g = 1.0; line.color.b = 0.0
        line.color.a = 0.6
        car_pt = Point(); car_pt.x = x; car_pt.y = y; car_pt.z = 0.0
        lp_pt  = Point(); lp_pt.x  = float(lp[0]); lp_pt.y = float(lp[1]); lp_pt.z = 0.0
        line.points = [car_pt, lp_pt]
        markers.markers.append(line)

        # --- closest point on raceline ---
        closest = Marker()
        closest.header.frame_id = 'world'
        closest.header.stamp    = now
        closest.ns     = 'closest'
        closest.id     = 3
        closest.type   = Marker.SPHERE
        closest.action = Marker.ADD
        closest.pose.position.x = float(self._rl_xy[closest_idx, 0])
        closest.pose.position.y = float(self._rl_xy[closest_idx, 1])
        closest.pose.position.z = 0.0
        closest.pose.orientation.w = 1.0
        closest.scale.x = closest.scale.y = closest.scale.z = 0.10
        closest.color.r = 0.0; closest.color.g = 1.0; closest.color.b = 0.0
        closest.color.a = 1.0
        markers.markers.append(closest)

        self._markers_pub.publish(markers)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, heading = euler_from_quaternion([q.x, q.y, q.z, q.w])
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = float(np.hypot(vx, vy))

        self._publish_control(x, y, heading, speed)

    def _closest_idx(self, x: float, y: float) -> int:
        dists = np.linalg.norm(self._rl_xy - [x, y], axis=1)
        return int(np.argmin(dists))

    def _lookahead_point(self, start_idx: int, speed: float):
        L_d = self.lookahead_base + self.lookahead_gain * speed
        n = len(self._rl_xy)
        accum = 0.0
        i = start_idx
        for _ in range(n):
            j = (i + 1) % n
            seg = float(np.linalg.norm(self._rl_xy[j] - self._rl_xy[i]))
            accum += seg
            if accum >= L_d:
                t = max(0.0, 1.0 - (accum - L_d) / max(seg, 1e-9))
                pt = self._rl_xy[i] + t * (self._rl_xy[j] - self._rl_xy[i])
                return pt, j
            i = j
        return self._rl_xy[(start_idx + 1) % n], (start_idx + 1) % n

    def _publish_control(self, x: float, y: float, heading: float, speed: float):
        closest = self._closest_idx(x, y)
        lp, _   = self._lookahead_point(closest, speed)

        dx  = lp[0] - x
        dy  = lp[1] - y
        L_d = float(np.hypot(dx, dy))

        alpha = float(np.arctan2(dy, dx)) - heading
        alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi

        steer_rad = float(np.arctan2(2.0 * self.wheelbase * np.sin(alpha), max(L_d, 0.01)))
        steer = float(np.clip(steer_rad / _PHYS_MAX_STEER_RAD,
                              -self.max_steer, self.max_steer))

        # Throttle comes pre-normalised [0,1] from the raceline CSV.
        # speed_scale and v_max act as a global ceiling for safe testing.
        target_v = float(np.clip(self._rl_speed[closest], self.v_min, self.v_max))

        steer_msg    = Float32(); steer_msg.data    = steer
        throttle_msg = Float32(); throttle_msg.data = target_v
        self._steer_pub.publish(steer_msg)
        self._throttle_pub.publish(throttle_msg)

        self._publish_viz_markers(x, y, lp, closest)

    def _stop(self):
        z = Float32(); z.data = 0.0
        self._steer_pub.publish(z)
        self._throttle_pub.publish(z)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
