# ROS2 TinyLidarNet Inference script
# Author: Jackson Yanek
# For University of Kansas CSL

import gc
import rclpy
import numpy as np
import time
import tensorflow as tf
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Joy

class TLNStandard(Node):
    def __init__(self):
        super().__init__('tln_standard')

        self.get_logger().info('TLNNode has been started.')

        # Declare ROS2 parameters (overridable via config file or command line)
        self.declare_parameter('sim', True)
        self.declare_parameter('min_speed', 0.05)
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('downscale_factor', 2)
        self.declare_parameter('model_path', None)
        self.declare_parameter('TLN_M', False)
        self.declare_parameter('print_debug', False)
        self.declare_parameter('autodrive_model', True)
        
        # Load parameters
        self.sim = self.get_parameter('sim').value
        self.init_min_speed = self.get_parameter('min_speed').value
        self.init_max_speed = self.get_parameter('max_speed').value
        self.downscale_factor = self.get_parameter('downscale_factor').value
        self.model_path = self.get_parameter('model_path').value
        self.TLN_M = self.get_parameter('TLN_M').value
        self.debug = self.get_parameter('print_debug').value
        self.autodrive_model = self.get_parameter('autodrive_model').value
        
        #global boolean for Autonomous control
        self.go = False
        self.min_speed = self.init_min_speed
        self.max_speed = self.init_max_speed

        self.launch = False
        self.launching = False

        self.speed_mappings = [self.linear_map, self.exp_map_abs]
        self.speed_map = self.speed_mappings[0]


        self.steering_publisher = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_publisher = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        # self.stats_publisher = self.create_publisher('/stats', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.scan_callback, 10)
        # self.joy_subscription = self.create_subscription(Joy,'joy',self.joy_callback, 10)

        # 0.2       5hz
        # 0.1       10hz
        # 0.05      20hz
        # 0.025     40hz


        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        # Pre-allocate scan buffer — avoids per-callback numpy allocation and GC pressure
        n_rays = self.interpreter.get_input_details()[0]['shape'][1]
        self._scan_buf = np.zeros((1, n_rays, 1), dtype=np.float32)

        # Disable cyclic GC in the hot path — prevents 20-80ms pauses that drop scan callbacks
        gc.collect()
        gc.disable()

        self.get_logger().warn(f'TLN Node Ready. sim={self.sim}, model={self.model_path}')
        if self.debug:
            self.get_logger().info(f"TLN_M:\t{self.TLN_M}\nDebug\t{self.debug}")
            
        if not self.sim:
            self.get_logger().warn('Press right bumper to activate.')

    # Utility functions
    
    def ns_2_s(self, ns):
        #nanoseconds to seconds
        return ns / 1_000_000_000

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        #linear map from x to y
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    
    
    
    def exp_map_abs(self, z, x_min, x_max, vmin, vmax):
        alpha=3.0
        # Map NN output z (expected in [0,1]) to [vmin,vmax] using exponential curve.
        # alpha > 0 controls curvature: higher alpha = more bias toward low speeds.

        z = np.clip(z, 0.0, 1.0)
        return vmin + (vmax - vmin) * (np.exp(alpha * z) - 1) / (np.exp(alpha) - 1)


    def scan_callback(self, msg):
        if not (self.go or self.sim):
            self.publish_drive(0, 0)
            return



        ranges = msg.ranges if not self.TLN_M else list(msg.ranges) + [20.0]
        raw = np.asarray(ranges, dtype=np.float32)[::self.downscale_factor]
        np.clip(raw, 0.0, 10.0, out=raw)
        self._scan_buf[0, :len(raw), 0] = raw

        self.interpreter.set_tensor(self.input_index, self._scan_buf)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)

        steer = output[0, 0]
        speed = output[0, 1]


        if self.debug:
            self.get_logger().info(f"Before Mapping: speed: {speed},steer: {steer}")
        speed = self.linear_map(speed, 0, 1, self.min_speed, self.max_speed)

        if self.autodrive_model:
            steer = float(np.clip(steer, -1.0, 1.0))
        else:
            steer = self.linear_map(steer, -0.52, 0.52, -1, 1)

        # if self.debug:
        #     self.get_logger().info(f"speed: {speed},steer: {steer}")
        self.publish_drive(speed, steer)

    def publish_drive(self, speed, steering_angle):
        
        speed_msg = Float32()
        steering_msg = Float32()
        
        speed_msg.data = float(speed)
        steering_msg.data = float(steering_angle)
        # Pretty much a boilerplate publishing function
        self.steering_publisher.publish(steering_msg)
        self.throttle_publisher.publish(speed_msg)
        

def main(args=None):
    # Init ROS2
    rclpy.init(args=args)
    # Create TLN Node
    node = TLNStandard()
    
    # Spin, look for interrupts
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.publish_drive(0, 0)
        node.destroy_node()
        gc.enable()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
