# Partially functional — joystick speed and/or steering override for TLN inference.
# Known limitation: simultaneous axis override requires both axes active at the same time.

import rclpy
import numpy as np
import time
import tensorflow as tf
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from collections import deque
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from tln_variants.node_utils import linear_map, preprocess_scan


class TLNStandardOverride(Node):
    def __init__(self):
        super().__init__('tln_override')
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.get_logger().info('TLNOverride node has been started.')

        self.model_path = "src/tln_variants/models/f1_tenth_model_small_noquantized.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        self.scan = []
        self.window_size = 10
        self.speed_queue = deque(maxlen=10)

        self.min_speed = 0
        self.max_speed = 10

        self.speed_vals = np.array([])

        self.steer_override = False
        self.speed_override = False
        self.speed = 0
        self.steering_angle = 0

        self._crashed = False
        self.create_timer(0.1, self._crash_shutdown_check)

    def _crash_shutdown_check(self):
        if self._crashed:
            self.get_logger().info('Crash detected — stopping node.')
            self.publish_ackermann_drive(0, 0)
            raise SystemExit

    def scan_callback(self, msg):
        if self._crashed:
            return

        raw = list(msg.ranges) + [20.0]  # model-specific size adjustment
        num_ranges = len(raw) // 2
        scans = preprocess_scan(raw, num_ranges, add_noise=True)
        scans = np.expand_dims(scans, axis=-1)  # (N, 1)
        scans = np.expand_dims(scans, axis=0)   # (1, N, 1)

        self.interpreter.set_tensor(self.input_index, scans)
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = (time.time() - start_time) * 1000
        self.get_logger().info(f'Inference time: {inf_time:.2f} ms')

        output = self.interpreter.get_tensor(self.output_index)
        steer = output[0, 0]
        speed = output[0, 1]

        speed = linear_map(speed, 0, 1, self.min_speed, self.max_speed)
        self.speed_vals = np.append(self.speed_vals, speed)
        self.speed_queue.append(speed)

        if self.detect_crash():
            self.get_logger().info('Crash detected.')
            self._crashed = True
            return

        # Joystick override: each axis is independent
        cmd_speed = self.speed if self.speed_override else speed
        cmd_steer = self.steering_angle if self.steer_override else steer
        self.publish_ackermann_drive(cmd_speed, cmd_steer)

    def joy_callback(self, msg):
        left_js_up_down = msg.axes[1]
        right_js_left_right = msg.axes[3]

        if left_js_up_down != 0:
            self.speed_override = True
            self.speed = left_js_up_down * self.max_speed
        else:
            self.speed_override = False

        if right_js_left_right != 0:
            self.steer_override = True
            self.steering_angle = right_js_left_right * 0.52
        else:
            self.steer_override = False

    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)
        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')

    def odom_callback(self, msg):
        pass  # Position/orientation not currently used

    def detect_crash(self):
        for i in range(len(self.scan) - self.window_size + 1):
            window = self.scan[i:i + self.window_size]
            if all(val <= 0.2 for val in window):
                return True
        if len(self.speed_queue) == self.speed_queue.maxlen and all(speed < 0.05 for speed in self.speed_queue):
            return True
        return False

    def print_avg_speed(self):
        avg = np.average(self.speed_vals)
        print(f"Average: {avg}")


def main(args=None):
    rclpy.init(args=args)
    node = TLNStandardOverride()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info('Node stopping.')
    finally:
        node.publish_ackermann_drive(0, 0)
        node.print_avg_speed()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
