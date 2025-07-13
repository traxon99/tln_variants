import rclpy
import numpy as np
import time
import tensorflow as tf
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header


#For average speed
min_max = (0,0)

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
        self.scan_buffer = np.zeros((2, 20))

        self.scans = [[] for i in range(5)]


    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    

    def scan_callback(self, msg):
        scans = np.array(msg.ranges)[::2]
        scans = np.append(scans, [10])

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        
        scans = np.array(scans)
        scans[scans>10] = 10

        self.scans.pop(0)
        self.scans.append(scans)
        if self.scans[0] == []:
            return 0.0, 0.0
        scans = np.stack([self.scans, [[i*0.025]*len(scans) for i in range(5)]], axis=-1)
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
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed) 
        self.publish_ackermann_drive(speed, steer)

        # scans = np.array(msg.ranges)
        # scans = np.append(scans, [20])
        # self.get_logger().info(f'num scans:{len(scans)}')
        # noise = np.random.normal(0, 0.5, scans.shape)
        # scans = scans + noise
        # scans[scans > 10] = 10
        # scans = scans[::2]  # Use every other value
        # scans = np.expand_dims(scans, axis=-1).astype(np.float32)
        # scans = np.expand_dims(scans, axis=0)


        # self.interpreter.set_tensor(self.input_index, scans)
        # start_time = time.time()
        # self.interpreter.invoke()
        # inf_time = (time.time() - start_time) * 1000  # in milliseconds
        # self.get_logger().info(f'Inference time: {inf_time:.2f} ms')

        # output = self.interpreter.get_tensor(self.output_index)
        # steer = output[0, 0]
        # speed = output[0, 1]

        # min_speed = -0.5
        # max_speed = 9
        # speed = self.linear_map(speed, 0, 1, min_speed, max_speed)

        # self.publish_ackermann_drive(speed, steer)

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
