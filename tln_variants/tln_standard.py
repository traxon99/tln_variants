import rclpy
import numpy as np
import time
import tensorflow as tf
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from sensor_msgs.msg import Joy

TLN_M = False

class TLNStandard(Node):
    def __init__(self):
        super().__init__('tln_standard')
        
        self.get_logger().info('TLNNode has been started.')
        
        
        #global boolean for Autonomous control
        self.go = False
        self.sim = True
        self.init_min_speed = 1#1
        self.init_max_speed = 8#8
        self.min_speed = self.init_min_speed
        self.max_speed = self.init_max_speed
        self.working_max_speed = self.init_max_speed
        
        
        self.launch = False
        self.launching = False
        
        self.speed_mappings = [self.linear_map, self.exp_map_abs]
        self.speed_map = self.speed_mappings[0]
        
        
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        # self.stats_publisher = self.create_publisher('/stats', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # self.joy_subscription = self.create_subscription(Joy,'joy',self.joy_callback, 10)

        # 0.2       5hz
        # 0.1       10hz
        # 0.05      20hz
        # 0.025     40hz
        
        
        
        self.timer = self.create_timer(0.0001, self.inference_dnn)
        self.downscale_factor = 2
        self.scan = None

        # self.model_path = "/home/jackson/sim_ws/src/tln_variants/train/Models/TLN_noquantized.tflite"
        # self.model_path = "/home/jackson/sim_ws/src/tln_variants/models/TLN_sim_data_aus_mos_spl.tflite" # good
        # self.model_path = "/home/jackson/sim_ws/src/tln_variants/models/TLN_Forza.tflite"
        # self.model_path = "/home/jackson/sim_ws/src/tln_variants/models/Forza_GLC_smile_ot_ez.tflite" # almost good
        # self.model_path = "/home/jackson/sim_ws/src/tln_variants/models/TLNETH_with_edgecases.tflite"
        # self.model_path= "/home/jackson/sim_ws/src/tln_variants/train/Models/very_close.tflite"
        
        #TFlite
        # self.model_path= "/home/jackson/sim_ws/src/tln_variants/train/Models/lidar_imitation_model_noquantized_d.tflite"
        self.model_path= "/home/jackson/sim_ws/src/tln_variants/train/Models/TLN_Forza.tflite" # Last used
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        
        self.get_logger().warn('TLN Node Ready.')
        if not(self.sim):
            self.get_logger().warn('Press right bumper to activate.')

    # Utility functions
    def ns_2_s(self, ns):
        #nanoseconds to seconds
        return ns / 1_000_000_000


    # Speed Mapping Functions 

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    
    
    
    def exp_map_abs(self, z, x_min, x_max, vmin, vmax):
        alpha=3.0
    # Map NN output z (expected in [0,1]) to [vmin,vmax] using exponential curve.
    # alpha > 0 controls curvature: higher alpha = more bias toward low speeds.

        z = np.clip(z, 0.0, 1.0)
        return vmin + (vmax - vmin) * (np.exp(alpha * z) - 1) / (np.exp(alpha) - 1)

    #Callbacks
    
    def joy_callback(self, msg):
        right_shoulder = msg.buttons[10]
        triangle = msg.buttons[3]
        square = msg.buttons[2]
        dpad_up = msg.buttons[11]
        dpad_down = msg.buttons[12]

        
        if right_shoulder:# and self.sim == False:
            self.go = True
        else:
            self.go = False
        
        #Caution mode
        if triangle == 1:
            self.min_speed, self.max_speed = 0.0, 4.0
        
        #Launch
        if (square == True) and (self.go == False):
            self.get_logger().warn('Launch Mode Activated. Min & Max Speed will be boosted for first second of racing.')
            self.min_speed, self.max_speed = 15.0, 15.0
            self.launch = True
        
        if self.launch == True and self.go == True:
            self.get_logger().warn("Launch started")
            self.launching = True
            self.launch_time = self.get_clock().now().nanoseconds
            self.launch = False
        
        if self.launching:
            launch_dur = self.ns_2_s(self.get_clock().now().nanoseconds - self.launch_time)
            
            if launch_dur >= 1.0:
                self.launching = False
                self.get_logger().warn("Launch ended")
                self.min_speed, self.max_speed = self.init_min_speed, self.init_max_speed
                
        #Speed adjustment
        if dpad_up == 1:
            
            if self.max_speed < 20.0:
                self.max_speed += 0.05
                self.get_logger().warn(f"Max Speed increased: {self.max_speed}")
            else:
                self.max_speed = 20.0
                
        elif dpad_down == 1:
            
            if self.max_speed > self.min_speed:
                self.max_speed -= 0.05
                self.get_logger().warn(f"Max Speed decreased: {self.max_speed}")
            else:
                self.max_speed = self.min_speed


            
    def scan_callback(self, msg):
        if self.go or self.sim:
            scan_proc_time_start = time.time()
            
            scans = np.array(msg.ranges)
            if TLN_M:
                scans = np.append(scans, [20]) #Only for Original TLN (541 scans)
            
            # self.get_logger().info(f'num scans:{len(scans)}')
            
            #Needs to be if sim == true -> Add noise
            noise = np.random.normal(0, 0.5, scans.shape)
            scans = scans + noise
            
            #Clip values beyond 10m
            scans[scans > 10] = 10
            scans = scans[::self.downscale_factor]  # Use every other value
            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0)

            self.scan = scans
            # self.interpreter.set_tensor(self.input_index, scans)
            # start_time = time.time()
            # self.interpreter.invoke()
            # inf_time = (time.time() - start_time) * 1000  # in milliseconds


            # output = self.interpreter.get_tensor(self.output_index)
            # steer = output[0, 0]
            # speed = output[0, 1]

            # speed = self.speed_map(speed, 0, 1, self.min_speed, self.max_speed)
            # self.get_logger().info(f"speed: {speed},steer: {steer}")
            # self.publish_ackermann_drive(speed, steer)
            
            # scan_proc_time = (time.time() - scan_proc_time_start) * 1000  # in milliseconds
            # self.get_logger().info(f'Inference time: {inf_time:.2f} ms')
            # self.get_logger().info(f'Processing time: {scan_proc_time:.2f} ms')
            
            # self.publish_stats(inf_time, scan_proc_time)
        else:
            self.publish_ackermann_drive(0, 0)
    def inference_dnn(self):
        if self.scan is not None:
            self.interpreter.set_tensor(self.input_index, self.scan)
            start_time = time.time()
            self.interpreter.invoke()
            inf_time = (time.time() - start_time) * 1000  # in milliseconds


            output = self.interpreter.get_tensor(self.output_index)
            steer = output[0, 0]
            speed = output[0, 1]

            speed = self.speed_map(speed, 0, 1, self.min_speed, self.max_speed)
            self.get_logger().info(f"speed: {speed},steer: {steer}")
            self.publish_ackermann_drive(speed, steer)
        
        
    def publish_stats(self, inf_time, scan_proc_time):
        pass
    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        # self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')

def main(args=None):

    rclpy.init(args=args)
    node = TLNStandard()
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
