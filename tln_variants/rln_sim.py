import rclpy
import numpy as np
import time
import tensorflow as tf
from collections import deque
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tln_variants.node_utils import linear_map, preprocess_scan

class RLNSim(Node):
    def __init__(self):
        super().__init__('rln_sim')
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('RLN started.')
        
        # self.model_path = "/home/jackson/sim_ws/src/tln_variants/models/TLN_sim_data_aus_mos_spl.tflite"
        self.model_path = "/home/jackson/sim_ws/src/tln_variants/models/RLN_TLN_M.tflite"
        
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()
        self.scans = [[] for i in range(5)]
        self.scan_ts = deque(maxlen=5)


    def scan_callback(self, msg):
        scans = np.array(msg.ranges)[::2]
        scans = np.append(scans, [10])

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        scans[scans > 10] = 10

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.scans.pop(0)
        self.scans.append(scans)
        self.scan_ts.append(t)
        if self.scans[0] == []:
            return

        # Build real ΔT channel from ROS header timestamps
        ts_arr = np.array(self.scan_ts, dtype=np.float32)
        if len(ts_arr) == 5 and np.all(np.diff(ts_arr) >= 0):
            diffs = np.diff(ts_arr)
        else:
            diffs = np.ones(4, dtype=np.float32) * 0.025  # fallback: 40 Hz
        dt_full = np.zeros(5, dtype=np.float32)
        dt_full[1:] = diffs
        dt_tiled = np.tile(dt_full[:, None], (1, len(scans)))

        scans = np.stack([self.scans, dt_tiled], axis=-1)
        scans = np.expand_dims(scans, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_index, scans)
        
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = time.time() - start_time
        inf_time = inf_time*1000
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        steer = output[0,0]
        speed = output[0,1]
        min_speed = 1
        max_speed = 8
        speed = linear_map(speed, 0, 1, min_speed, max_speed)
        self.publish_ackermann_drive(speed, steer)

    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')


def main(args=None):

    rclpy.init(args=args)
    node = RLNSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.publish_ackermann_drive(0,0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
