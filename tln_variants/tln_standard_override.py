# TODO: clean up and comment better
# FYI this script does not work as intended.


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


#For average speed
min_max = (0,0)

class TLNStandardOverride(Node):
    def __init__(self):
        super().__init__('tln_standard')
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.get_logger().info('TLNNode has been started.')

        self.model_path = "src/tln_variants/models/f1_tenth_model_small_noquantized.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
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
        
        #for joystick override
        self.steer_override = False    
        self.speed_override = False
        self.speed = 0
        self.steering_angle = 0

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    

    def scan_callback(self, msg):
        scans = np.array(msg.ranges)
        scans = np.append(scans, [20])
        self.get_logger().info(f'num scans:{len(scans)}')
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        scans[scans > 10] = 10
        scans = scans[::2]  # Use every other value
        scans = np.expand_dims(scans, axis=-1).astype(np.float32)
        scans = np.expand_dims(scans, axis=0)


        self.interpreter.set_tensor(self.input_index, scans)
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = (time.time() - start_time) * 1000  # in milliseconds
        self.get_logger().info(f'Inference time: {inf_time:.2f} ms')

        output = self.interpreter.get_tensor(self.output_index)
        steer = output[0, 0]
        speed = output[0, 1]

        self.min_speed = 0
        self.max_speed = 10
        speed = self.linear_map(speed, 0, 1, self.min_speed, self.max_speed)
        self.speed_vals = np.append(self.speed_vals, speed)
        
        self.speed_queue.append(speed)

        if self.detect_crash():
            self.get_logger().info("Crash")
            self.publish_ackermann_drive(0,0)
            self.destroy_node()
            rclpy.shutdown()
            quit()
        
        #Joystick override
        if self.speed_override and self.steer_override: #All joystick
            self.publish_ackermann_drive(self.speed, self.steering_angle)
        elif self.speed_override and not(self.steer_override): #only speed
            self.publish_ackermann_drive(self.speed, steer)
        elif not(self.speed_override) and self.steer_override: #only steering
            self.publish_ackermann_drive(speed, self.steering_angle)
        else:
            self.publish_ackermann_drive(speed, steer)
    def joy_callback(self, msg):

        #get values [0,1] from joysticks
        left_js_up_down = msg.axes[1] #
        right_js_left_right = msg.axes[3] #
        
        if (left_js_up_down != 0):
            self.speed_override = True
            self.speed = left_js_up_down * self.max_speed

        elif (right_js_left_right != 0):
            self.steer_override = True
            self.steering_angle = right_js_left_right * 0.52
        else:
            self.steer_override = False
            self.speed_override = False
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
    node = TLNStandardOverride()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.publish_ackermann_drive(0,0)
        node.print_avg_speed()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
