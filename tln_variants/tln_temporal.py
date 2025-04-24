import rclpy
import numpy as np
import tensorflow as tf
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from collections import deque
from nav_msgs.msg import Odometry


class TLNStandard(Node):
    def __init__(self):
        super().__init__('tln_standard')
        self.get_logger().info('TLN Node has been started.')

        # ROS interfaces
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        # Load TFLite model
        self.model_path = "src/tln_variants/models/TLN_temporal_max_data_noquantized.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.input_shape = self.interpreter.get_input_details()[0]["shape"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        # Buffers
        self.scan_buffer = deque(maxlen=125)  # For temporal scan history
        self.speed_queue = deque(maxlen=10)   # For crash detection
        self.speed_vals = np.array([])        # For average speed
        self.scan = []                        # For crash detection

        # Odometry
        self.position = None
        self.orientation = None

        # Debug flag
        self.debug = True

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def scan_callback(self, msg):
        scans = np.array(msg.ranges)
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = np.clip(scans + noise, 0, 10)

        self.scan = scans  # for crash detection
        self.scan_buffer.append(scans)

        # Get previous scans
        t_5 = self.scan_buffer[-5] if len(self.scan_buffer) >= 5 else np.zeros_like(scans)
        t_10 = self.scan_buffer[-10] if len(self.scan_buffer) >= 10 else np.zeros_like(scans)

        # Combine scans (current, t-5, t-10)
        scans_combined = np.stack((scans, t_5, t_10), axis=0)
        scans_combined = np.expand_dims(scans_combined, axis=-1).astype(np.float32)  # (3, N, 1)
        scans_combined = np.expand_dims(scans_combined, axis=0)  # (1, 3, N, 1)

        if self.debug:
            self.get_logger().info(f'Input shape to model: {scans_combined.shape}')

        self.interpreter.set_tensor(self.input_index, scans_combined)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)

        steer = output[0, 0]
        raw_speed = output[0, 1]
        min_speed = 1
        max_speed = 13
        speed = self.linear_map(raw_speed, 0, 1, min_speed, max_speed)
        if not np.isfinite(speed) or not np.isfinite(steer):
            self.get_logger().warn("Invalid model output, skipping publish.")
            return

        self.speed_vals = np.append(self.speed_vals, speed)
        self.speed_queue.append(speed)

        self.publish_ackermann_drive(speed, steer)

    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published: speed={speed:.2f}, steering_angle={steering_angle:.2f}')

    def odom_callback(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation

    def detect_crash(self):
        # Option 1: Too close to object repeatedly
        if len(self.scan) >= self.window_size:
            for i in range(len(self.scan) - self.window_size + 1):
                if all(val <= 0.2 for val in self.scan[i:i + self.window_size]):
                    return True

        # Option 2: Stopped for too long
        if len(self.speed_queue) == self.speed_queue.maxlen and all(s < 0.05 for s in self.speed_queue):
            return True

        return False

    def print_avg_speed(self):
        avg = np.mean(self.speed_vals) if len(self.speed_vals) > 0 else 0.0
        self.get_logger().info(f'Average speed: {avg:.2f} m/s')


def main(args=None):
    rclpy.init(args=args)
    node = TLNStandard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.print_avg_speed()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
