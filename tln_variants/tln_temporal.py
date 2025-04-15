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

#For average speed
min_max = (0,0)

class TLNStandard(Node):
    def __init__(self):
        super().__init__('tln_standard')
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.get_logger().info('TLNNode has been started.')

        self.model_path = "src/tln_variants/models/TLN_temporal_test.tflite"
        #self.model_path = "src/tln_variants/models/f1_tenth_model_small_noquantized.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.input_shape = self.interpreter.get_input_details()[0]["shape"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.scan_buffer = np.zeros((2, 20))

        # Evaluation additions
        self.scan = []
        self.window_size = 10
        # Initialize a deque with a maximum length of 10 to store speed values
        self.speed_queue = deque(maxlen=10)

        self.position = None
        self.orientation = None
        
        #for evaluation
        self.speed_vals = np.array([])

        #temporal scans (t-10, t-20)

        self.t_5_scans = deque(maxlen=5)  # Last 5 scans
        self.t_10_scans = deque(maxlen=10)  # Last 1 scans
   

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    


    def scan_callback(self, msg):
        scans = np.array(msg.ranges)
        # scans = np.append(scans, [20])
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        scans[scans > 10] = 10
        # scans = scans[::2]  # Downsample

        # # Append the current scan to both t_10_scans and t_20_scans
        # self.t_1_scans.append(scans)
        # self.t_5_scans.append(scans)
        # self.t_10_scans.append(scans)
        
        # # Ensure the shape of t_1, t_5, and t_10 is (541,)
        # if len(self.t_1_scans) >= 1:
        #     t_1 = self.t_1_scans[-1]  # Get the most recent scan from t_1_scans
        # else:
        #     t_1 = np.zeros_like(scans)  # Default if not enough data

        # if len(self.t_5_scans) >= 5:
        #     t_5 = self.t_5_scans[-5]  # Get the 5th most recent scan from t_5_scans
        # else:
        #     t_5 = np.zeros_like(scans)  # Default if not enough data

        # if len(self.t_10_scans) >= 10:
        #     t_10 = self.t_10_scans[-10]  # Get the 10th most recent scan from t_10_scans
        # else:
        #     t_10 = np.zeros_like(scans)  # Default if not enough data

        # # # Combine current scan, t-10, and t-20 scans
        # # print(scans.shape)  # Should be (541,)
        # # print(t_1.shape)    # Should be (541,)
        # # print(t_5.shape)    # Should be (541,)
        # # print(t_10.shape)
        #  # Append the current scan to both t_10_scans and t_20_scans
        
        # self.t_1_scans.append(scans)
        # self.t_5_scans.append(scans)
        # self.t_10_scans.append(scans)
        
        # # Ensure the shape of t_1, t_5, and t_10 is (541,)
        # if len(self.t_1_scans) >= 1:
        #     t_1 = self.t_1_scans[-1]  # Get the most recent scan from t_1_scans
        # else:
        #     t_1 = np.zeros_like(scans)  # Default if not enough data

        if len(self.t_5_scans) >= 62:
            t_5 = self.t_5_scans[-62]  # Get the 5th most recent scan from t_5_scans
        else:
            t_5 = np.zeros_like(scans)  # Default if not enough data

        if len(self.t_10_scans) >= 125:
            t_10 = self.t_10_scans[-125]  # Get the 10th most recent scan from t_10_scans
        else:
            t_10 = np.zeros_like(scans)  # Default if not enough data

        # # Combine current scan, t-10, and t-20 scans
        # print(scans.shape)  # Should be (541,)
        # print(t_1.shape)    # Should be (541,)
        # print(t_5.shape)    # Should be (541,)
        # print(t_10.shape)
        
        scans_combined = np.stack((scans, t_5, t_10), axis=0)
        # Stack the arrays along a new axis (axis=-1) to form a tensor with shape (541, 4)
        # scans_combined = np.stack((scans, t_1, t_5, t_10), axis=0)  # Shape: (4, 541)
        print(f"first: {scans_combined.shape}")
        # Add the batch dimension (Shape: (1, 4, 541)
        # scans_combined = np.expand_dims(scans_combined, axis=0)
        print(f"second: {scans_combined.shape}")
        # Add the channel dimension (Shape: (1, 4, 541, 1))
        scans_combined = np.expand_dims(scans_combined, axis=-1).astype(np.float32)
        print(f"final: {scans_combined.shape}")

        # print(self.interpreter.get_input_details())
        print("Input shape: ", self.input_shape)
        
        # Now the shape of scans_combined is (1, 4, 541, 1), which matches the model's input shape.
        self.interpreter.set_tensor(self.input_index, scans_combined)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)
        steer = output[0, 0]
        speed = output[0, 1]
        min_speed = 0
        max_speed = 10
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
        self.speed_vals = np.append(self.speed_vals, speed)
        #Publish the result
        self.publish_ackermann_drive(speed, steer)


    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')
    
    def odom_callback(self, msg):
        # Extract position and orientation from the Odometry message
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation

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
